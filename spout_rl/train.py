"""PPO training for Spout.

    uv run python -m spout_rl.train --run-name first
    uv run tensorboard --logdir runs
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import time

# Idle OpenMP threads (ours and PyTorch's) sleep instead of spinning on every core.
os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from .env import DEFAULT_CONFIG, MODE_NAMES, NUM_SCALARS, PLAY_H, W, SpoutVec
from .evaluate import run_episodes, save_replay
from .model import Agent, FrameStack

ROOT = Path(__file__).resolve().parent.parent


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-name", default=time.strftime("%Y%m%d-%H%M%S"))
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--total-steps", type=int, default=300_000_000, help="agent decisions (x frameskip = game frames)")
    p.add_argument("--num-envs", type=int, default=256)
    p.add_argument("--num-steps", type=int, default=128, help="rollout length per update")
    p.add_argument("--lr", type=float, default=2.5e-4)
    p.add_argument("--gamma", type=float, default=0.995)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--minibatches", type=int, default=8)
    p.add_argument("--clip", type=float, default=0.1)
    p.add_argument("--ent", type=float, default=0.01)
    p.add_argument("--vf", type=float, default=0.5)
    p.add_argument("--max-grad-norm", type=float, default=0.5)
    p.add_argument("--stack", type=int, default=4)
    p.add_argument("--no-compile", action="store_true", help="skip torch.compile for the update step")
    p.add_argument("--env-threads", type=int, default=8, help="CPU threads per half of the games")
    p.add_argument("--hit-anneal-frac", type=float, default=0.5,
                   help="fade the rock-hitting shaping reward to 0 over this fraction of training")
    p.add_argument("--eval-every", type=int, default=100, help="updates between evaluations")
    p.add_argument("--eval-episodes", type=int, default=32)
    p.add_argument("--save-every", type=int, default=50)
    p.add_argument("--resume", default=None, help="checkpoint to continue from (same run, same settings)")
    p.add_argument("--init-from", default=None,
                   help="start a new run from a checkpoint's network weights only (fresh optimizer, "
                        "step count and learning-rate schedule), e.g. to fine-tune with new rewards")
    for k, v in DEFAULT_CONFIG.items():
        p.add_argument(f"--env-{k.replace('_', '-')}", type=type(v), default=v, dest=f"env_{k}")
    return p.parse_args()


def main():
    args = parse_args()
    env_cfg = {k: getattr(args, f"env_{k}") for k in DEFAULT_CONFIG}
    run_dir = ROOT / "runs" / args.run_name
    (run_dir / "replays").mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps({"args": vars(args), "env_cfg": env_cfg}, indent=2))

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    torch.set_num_threads(4)
    torch.distributions.Distribution.set_default_validate_args(False)  # avoids GPU syncs
    amp = dict(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda")

    agent = Agent(stack=args.stack).to(device).to(memory_format=torch.channels_last)
    opt = torch.optim.Adam(agent.parameters(), lr=args.lr, eps=1e-5)
    start_update, global_step, best_eval = 1, 0, -1.0
    if args.resume:
        ck = torch.load(args.resume, map_location=device, weights_only=False)
        agent.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        start_update, global_step = ck["update"] + 1, ck["global_step"]
        best_eval = ck.get("best_eval", -1.0)
        print(f"resumed from {args.resume} at update {ck['update']}")
    elif args.init_from:
        ck = torch.load(args.init_from, map_location=device, weights_only=False)
        if ck["args"]["stack"] != args.stack:
            raise SystemExit(f"--init-from checkpoint uses --stack {ck['args']['stack']}, not {args.stack}")
        agent.load_state_dict(ck["model"])
        print(f"initialized weights from {args.init_from} ({ck['global_step']/1e6:.0f}M steps)")

    # Compiled copy (shares weights) for the big minibatch passes in the update.
    train_agent = agent if args.no_compile or device.type != "cuda" else torch.compile(agent)

    N, T = args.num_envs, args.num_steps
    batch = N * T
    mb_size = batch // args.minibatches
    num_updates = args.total_steps // batch
    base_hit = env_cfg["r_hit_above"]

    # The games are split into two halves: while the GPU picks moves for one
    # half, the CPU steps the other. Results land in pinned memory so copies
    # to the GPU don't block.
    assert N % 2 == 0
    H = N // 2
    cuda = device.type == "cuda"

    def pinned(shape, dtype):
        return torch.empty(shape, dtype=dtype, pin_memory=cuda)

    class Half:
        pass

    halves = []
    for h in range(2):
        x = Half()
        x.sl = slice(h * H, (h + 1) * H)
        x.obs_pin = pinned((H, PLAY_H, W), torch.uint8)
        x.sc_pin = pinned((H, NUM_SCALARS), torch.float32)
        x.rew_pin = pinned((H,), torch.float32)
        x.done_pin = pinned((H,), torch.uint8)
        x.act_pin = pinned((H,), torch.int32)
        x.env = SpoutVec(H, seed=args.seed * 2 + h + start_update * 7919, threads=args.env_threads,
                         buffers=dict(obs=x.obs_pin.numpy(), scalars=x.sc_pin.numpy(),
                                      rewards=x.rew_pin.numpy(), dones=x.done_pin.numpy()),
                         **env_cfg)
        x.env.reset()
        x.stack = FrameStack(H, args.stack, device)
        x.stack.reset(x.obs_pin.to(device))
        x.sc = x.sc_pin.to(device)
        x.event = None
        x.t = 0
        halves.append(x)

    b_obs = torch.zeros((T, N, args.stack, PLAY_H, W), dtype=torch.uint8, device=device)
    b_sc = torch.zeros((T, N, NUM_SCALARS), device=device)
    b_act = torch.zeros((T, N), dtype=torch.long, device=device)
    b_logp = torch.zeros((T, N), device=device)
    b_rew = torch.zeros((T, N), device=device)
    b_done = torch.zeros((T, N), device=device)
    b_val = torch.zeros((T, N), device=device)

    writer = SummaryWriter(str(run_dir))
    recent = {m: deque(maxlen=200) for m in MODE_NAMES}
    print(f"run {run_dir}  updates {num_updates}  batch {batch}  device {device}")

    # On SIGTERM/SIGINT (e.g. a cluster job hitting its time limit), finish the
    # current update, save latest.pt and exit cleanly.
    stop_flag = [False]

    def request_stop(signum, frame):
        stop_flag[0] = True

    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGUSR1):
        signal.signal(sig, request_stop)

    t_start = time.time()
    steps_at_start = global_step
    for update in range(start_update, num_updates + 1):
        frac = (update - 1) / num_updates
        for g in opt.param_groups:
            g["lr"] = args.lr * (1.0 - frac)
        hit = base_hit * max(0.0, 1.0 - frac / args.hit_anneal_frac) if args.hit_anneal_frac > 0 else 0.0
        for x in halves:
            x.env.set_config(r_hit_above=hit)

        # ---- rollout ----
        t_roll = time.time()

        def infer(x, t):
            b_obs[t, x.sl] = x.stack.buf
            b_sc[t, x.sl] = x.sc
            with torch.no_grad(), torch.autocast(**amp):
                logits, value = agent(x.stack.buf, x.sc)
            dist = torch.distributions.Categorical(logits=logits)
            a = dist.sample()
            b_act[t, x.sl], b_logp[t, x.sl], b_val[t, x.sl] = a, dist.log_prob(a), value
            x.act_pin.copy_(a, non_blocking=True)
            x.event = torch.cuda.Event() if cuda else None
            if cuda:
                x.event.record()
            x.t = t

        def advance(x):
            if x.event is not None:
                x.event.synchronize()
            x.env.step(x.act_pin.numpy())
            done = x.done_pin.to(device, non_blocking=True).bool()
            b_rew[x.t, x.sl] = x.rew_pin.to(device, non_blocking=True)
            b_done[x.t, x.sl] = done.float()
            x.stack.push(x.obs_pin.to(device, non_blocking=True), done)
            x.sc = x.sc_pin.to(device, non_blocking=True)
            d = x.env.dones
            if d.any():
                for row in x.env.info[d == 1]:
                    recent[int(row[4])].append(row.copy())

        A, B = halves
        for t in range(T):
            infer(A, t)
            if t > 0:
                advance(B)  # B's moves from step t-1, while the GPU works on A
            infer(B, t)
            advance(A)  # A's moves from step t, while the GPU works on B
        advance(B)
        global_step += batch
        t_roll = time.time() - t_roll

        # ---- GAE ----
        with torch.no_grad(), torch.autocast(**amp):
            next_val = torch.cat([agent(x.stack.buf, x.sc)[1] for x in halves])
        adv = torch.zeros_like(b_rew)
        last = torch.zeros(N, device=device)
        for t in reversed(range(T)):
            nv = next_val if t == T - 1 else b_val[t + 1]
            nonterm = 1.0 - b_done[t]
            delta = b_rew[t] + args.gamma * nv * nonterm - b_val[t]
            last = delta + args.gamma * args.gae_lambda * nonterm * last
            adv[t] = last
        ret = adv + b_val

        # ---- PPO update ----
        t_upd = time.time()
        f_obs = b_obs.reshape(batch, args.stack, PLAY_H, W)
        f_sc, f_act = b_sc.reshape(batch, -1), b_act.reshape(batch)
        f_logp, f_adv, f_ret = b_logp.reshape(batch), adv.reshape(batch), ret.reshape(batch)
        clipfracs, approx_kls = [], []
        for _ in range(args.epochs):
            perm = torch.randperm(batch, device=device)
            for k in range(0, batch, mb_size):
                idx = perm[k:k + mb_size]
                with torch.autocast(**amp):
                    logits, v = train_agent(f_obs[idx], f_sc[idx])
                dist = torch.distributions.Categorical(logits=logits)
                logp = dist.log_prob(f_act[idx])
                ratio = (logp - f_logp[idx]).exp()
                mb_adv = f_adv[idx]
                mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)
                pg_loss = torch.max(-mb_adv * ratio,
                                    -mb_adv * ratio.clamp(1 - args.clip, 1 + args.clip)).mean()
                v_loss = 0.5 * ((v - f_ret[idx]) ** 2).mean()
                ent = dist.entropy().mean()
                loss = pg_loss + args.vf * v_loss - args.ent * ent
                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                opt.step()
                with torch.no_grad():
                    approx_kls.append(((ratio - 1) - (logp - f_logp[idx])).mean().item())
                    clipfracs.append(((ratio - 1).abs() > args.clip).float().mean().item())
        t_upd = time.time() - t_upd

        # ---- logging ----
        sps = (global_step - steps_at_start) / (time.time() - t_start)
        writer.add_scalar("perf/sps", sps, global_step)
        writer.add_scalar("perf/rollout_s", t_roll, global_step)
        writer.add_scalar("perf/update_s", t_upd, global_step)
        writer.add_scalar("loss/policy", pg_loss.item(), global_step)
        writer.add_scalar("loss/value", v_loss.item(), global_step)
        writer.add_scalar("loss/entropy", ent.item(), global_step)
        writer.add_scalar("loss/approx_kl", np.mean(approx_kls), global_step)
        writer.add_scalar("loss/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("train/lr", opt.param_groups[0]["lr"], global_step)
        writer.add_scalar("train/r_hit_above", hit, global_step)
        parts = []
        for m, name in MODE_NAMES.items():
            if recent[m]:
                e = np.array(recent[m])
                gained = e[:, 1] - e[:, 5]
                writer.add_scalar(f"{name}/return", e[:, 0].mean(), global_step)
                writer.add_scalar(f"{name}/height_gained", gained.mean(), global_step)
                writer.add_scalar(f"{name}/barriers", e[:, 6].mean(), global_step)
                writer.add_scalar(f"{name}/timeout_rate", (e[:, 7] == 1).mean(), global_step)
                if m == 0:
                    writer.add_scalar("normal/score", e[:, 2].mean(), global_step)
                    parts.append(f"normal score {e[:, 2].mean():6.0f} h {e[:, 1].mean():5.0f}")
                else:
                    parts.append(f"{name} +h {gained.mean():5.0f} bar {e[:, 6].mean():.2f}")
        if update % 5 == 0 or update == start_update:
            print(f"upd {update:5d}/{num_updates} step {global_step/1e6:7.2f}M sps {sps:6.0f} | "
                  + " | ".join(parts) + f" | ent {ent.item():.2f} kl {np.mean(approx_kls):.4f}",
                  flush=True)

        # ---- eval + checkpoints ----
        stop_requested = stop_flag[0]
        def save(path, **extra):
            # Write to a temp file and rename, so a kill mid-save never leaves
            # a half-written checkpoint behind.
            tmp = Path(str(path) + ".tmp")
            torch.save(dict(model=agent.state_dict(), optimizer=opt.state_dict(), update=update,
                            global_step=global_step, args=vars(args), env_cfg=env_cfg,
                            best_eval=best_eval, **extra), tmp)
            os.replace(tmp, path)

        if update % args.eval_every == 0 or update == num_updates:
            agent.eval()
            res = run_episodes(agent, list(range(10_000, 10_000 + args.eval_episodes)), device,
                               frameskip=env_cfg["frameskip"])
            agent.train()
            scores = np.array([r["score"] for r in res])
            heights = np.array([r["height"] for r in res])
            bars = np.array([r["barriers"] for r in res])
            writer.add_scalar("eval/score_mean", scores.mean(), global_step)
            writer.add_scalar("eval/score_max", scores.max(), global_step)
            writer.add_scalar("eval/height_mean", heights.mean(), global_step)
            writer.add_scalar("eval/barriers_mean", bars.mean(), global_step)
            best = max(res, key=lambda r: r["score"])
            save_replay(best, run_dir / "replays" / f"upd{update:05d}_score{best['score']}.json")
            print(f"  EVAL upd {update}: score mean {scores.mean():.0f} median {np.median(scores):.0f} "
                  f"max {scores.max()} | height mean {heights.mean():.0f} | barriers {bars.mean():.2f}",
                  flush=True)
            if scores.mean() > best_eval:
                best_eval = float(scores.mean())
                save(run_dir / "best.pt")
                save_replay(best, run_dir / "replays" / "best.json")

        if update % args.save_every == 0 or update == num_updates or stop_requested:
            save(run_dir / "latest.pt")
        if stop_requested:
            print(f"stop requested: saved latest.pt at update {update}, exiting", flush=True)
            break

    writer.close()
    for x in halves:
        x.env.close()


if __name__ == "__main__":
    main()
