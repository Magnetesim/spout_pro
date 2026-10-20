"""Watch or play Spout using the training engine.

    uv run python -m spout_rl.play human                     # play it yourself (arrows + space/x)
    uv run python -m spout_rl.play watch runs/v1/best.pt     # watch the agent play live
    uv run python -m spout_rl.play replay runs/v1/replays/best.json
    uv run python -m spout_rl.play replay runs/v1/replays/best.json --video best.mp4
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np

from .env import SpoutVec, START_NORMAL, action_of, to_rgb

FPS = 50


def replay_frames(rep: dict):
    """Yield every screen (88x128 colors) of a recorded game."""
    env = SpoutVec(1, record_frames=True, frameskip=rep["frameskip"], max_frames=0,
                   p_drill=0.0, p_random=0.0)
    env.start(0, rep["seed"], rep.get("mode", START_NORMAL), rep.get("height", 0))
    for ch in rep["actions"]:
        _, _, _, d, _ = env.step(np.array([int(ch)], np.int32))
        for f in env.frames[0]:
            yield f
        if d[0]:
            break
    for _ in range(FPS):  # hold the game-over screen for a second
        yield env.frames[0, -1]


def write_video(frames, path: str, zoom: int = 4):
    import imageio.v2 as imageio

    with imageio.get_writer(path, fps=FPS, codec="libx264", quality=8, macro_block_size=1) as w:
        for f in frames:
            w.append_data(to_rgb(f, zoom))


class Window:
    def __init__(self, zoom: int, title: str):
        import pygame

        self.pg = pygame
        pygame.init()
        self.zoom = zoom
        self.screen = pygame.display.set_mode((128 * zoom, 88 * zoom))
        pygame.display.set_caption(title)
        self.clock = pygame.time.Clock()

    def show(self, frame) -> bool:
        pg = self.pg
        for e in pg.event.get():
            if e.type == pg.QUIT or (e.type == pg.KEYDOWN and e.key == pg.K_q):
                return False
        surf = pg.surfarray.make_surface(to_rgb(frame, self.zoom).swapaxes(0, 1))
        self.screen.blit(surf, (0, 0))
        pg.display.flip()
        self.clock.tick(FPS)
        return True


def cmd_replay(args):
    rep = json.loads(Path(args.file).read_text())
    print(f"seed {rep['seed']}  score {rep.get('score')}  height {rep.get('height')}  "
          f"barriers {rep.get('barriers')}  ({len(rep['actions'])} decisions)")
    if args.video:
        write_video(replay_frames(rep), args.video, args.zoom)
        print(f"wrote {args.video}")
        return
    win = Window(args.zoom, "spout replay")
    for f in replay_frames(rep):
        if not win.show(f):
            break


def cmd_human(args):
    import pygame as pg

    win = Window(args.zoom, "spout (training engine)")
    env = SpoutVec(1, record_frames=True, frameskip=1, max_frames=0, p_drill=0.0, p_random=0.0)
    env.start(0, args.seed if args.seed is not None else random.randrange(1 << 30))
    while True:
        k = pg.key.get_pressed()
        a = action_of(k[pg.K_LEFT], k[pg.K_RIGHT], k[pg.K_SPACE] or k[pg.K_x] or k[pg.K_z])
        _, _, _, d, info = env.step(np.array([a], np.int32))
        if not win.show(env.frames[0, 0]):
            return
        if d[0]:
            print(f"score {int(info[0, 2])}  height {int(info[0, 1])}")
            for _ in range(FPS):
                win.show(env.frames[0, 0])
            env.start(0, random.randrange(1 << 30))


def cmd_watch(args):
    import torch

    from .evaluate import load_agent
    from .model import FrameStack

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent, ck = load_agent(args.checkpoint, device)
    fs = ck["env_cfg"]["frameskip"]
    win = Window(args.zoom, "spout agent")
    env = SpoutVec(1, record_frames=True, frameskip=fs, max_frames=0, p_drill=0.0, p_random=0.0)
    while True:
        env.start(0, random.randrange(1 << 30))
        stack = FrameStack(1, agent.stack, device)
        obs = stack.reset(torch.from_numpy(env.obs).to(device))
        while True:
            with torch.no_grad():
                logits, _ = agent(obs, torch.from_numpy(env.scalars).to(device))
            a = torch.multinomial(torch.softmax(logits, -1), 1).item()
            _, _, _, d, info = env.step(np.array([a], np.int32))
            for f in env.frames[0]:
                if not win.show(f):
                    return
            if d[0]:
                print(f"score {int(info[0, 2])}  height {int(info[0, 1])}  barriers {int(info[0, 6])}")
                break
            obs = stack.push(torch.from_numpy(env.obs).to(device), torch.zeros(1, dtype=torch.bool, device=device))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--zoom", type=int, default=5)
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("replay")
    r.add_argument("file")
    r.add_argument("--video", default=None, help="write an .mp4 instead of opening a window")
    h = sub.add_parser("human")
    h.add_argument("--seed", type=int, default=None)
    w = sub.add_parser("watch")
    w.add_argument("checkpoint")
    args = p.parse_args()
    {"replay": cmd_replay, "human": cmd_human, "watch": cmd_watch}[args.cmd](args)


if __name__ == "__main__":
    main()
