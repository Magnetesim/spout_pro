"""Play full games from the normal start with a trained policy and save replays."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .env import SpoutVec, START_NORMAL
from .model import Agent, FrameStack


@torch.no_grad()
def run_episodes(agent: Agent, seeds: list[int], device, frameskip: int = 4,
                 max_frames: int = 50 * 60 * 10, greedy: bool = False, sample_seed: int = 0):
    """Play one normal-start game per seed. Returns a list of dicts with the
    score, height and the action list (enough to replay the game exactly)."""
    n = len(seeds)
    env = SpoutVec(n, frameskip=frameskip, max_frames=max_frames, p_drill=0.0, p_random=0.0)
    for i, s in enumerate(seeds):
        env.start(i, s, START_NORMAL)
    stack = FrameStack(n, agent.stack, device)
    obs = stack.reset(torch.from_numpy(env.obs).to(device))
    sc = torch.from_numpy(env.scalars).to(device)
    gen = torch.Generator(device=device).manual_seed(sample_seed)

    alive = np.ones(n, bool)
    actions = [[] for _ in range(n)]
    results = [None] * n
    while alive.any():
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits, _ = agent(obs, sc)
        if greedy:
            a = logits.argmax(-1)
        else:
            a = torch.multinomial(torch.softmax(logits, -1), 1, generator=gen).squeeze(-1)
        a_np = a.cpu().numpy().astype(np.int32)
        for i in np.nonzero(alive)[0]:
            actions[i].append(int(a_np[i]))
        _, _, _, dones, info = env.step(a_np)
        for i in np.nonzero(alive & (dones == 1))[0]:
            alive[i] = False
            results[i] = dict(seed=int(seeds[i]), mode=START_NORMAL, frameskip=frameskip,
                              score=int(info[i, 2]), height=int(info[i, 1]),
                              barriers=int(info[i, 6]), frames=int(info[i, 3]),
                              timed_out=bool(info[i, 7] == 1),
                              actions="".join(map(str, actions[i])))
        d = torch.from_numpy(dones.astype(bool)).to(device)
        obs = stack.push(torch.from_numpy(env.obs).to(device), d)
        sc = torch.from_numpy(env.scalars).to(device)
    env.close()
    return results


def load_agent(path: str | Path, device):
    ck = torch.load(path, map_location=device, weights_only=False)
    agent = Agent(stack=ck["args"]["stack"]).to(device)
    agent.load_state_dict(ck["model"])
    agent.eval()
    return agent, ck


def save_replay(result: dict, path: str | Path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(result))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("checkpoint")
    p.add_argument("--episodes", type=int, default=32)
    p.add_argument("--seed", type=int, default=100000)
    p.add_argument("--greedy", action="store_true")
    p.add_argument("--save-best", default=None, help="write the best game's replay here")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent, ck = load_agent(args.checkpoint, device)
    res = run_episodes(agent, list(range(args.seed, args.seed + args.episodes)), device,
                       frameskip=ck["env_cfg"]["frameskip"], greedy=args.greedy)
    scores = np.array([r["score"] for r in res])
    heights = np.array([r["height"] for r in res])
    print(f"{len(res)} games: score mean {scores.mean():.0f} median {np.median(scores):.0f} "
          f"max {scores.max()} | height mean {heights.mean():.0f} max {heights.max()} | "
          f"barriers mean {np.mean([r['barriers'] for r in res]):.2f}")
    if args.save_best:
        best = max(res, key=lambda r: r["score"])
        save_replay(best, args.save_best)
        print(f"best game (score {best['score']}) saved to {args.save_best}")


if __name__ == "__main__":
    main()
