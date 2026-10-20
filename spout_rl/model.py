"""Policy/value network: a small CNN over the last few screens plus a few
numbers the player can read off the screen (time, height, ship motion)."""

import numpy as np
import torch
import torch.nn as nn

from .env import NUM_ACTIONS, NUM_SCALARS, PLAY_H, W


def layer_init(layer, std=np.sqrt(2), bias=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias)
    return layer


class Agent(nn.Module):
    def __init__(self, stack: int = 4, hidden: int = 512):
        super().__init__()
        self.stack = stack
        # Standard Atari layout. The 8x8 filters overlap (stride 4), so a
        # 1-pixel barrier line or the 3x3 ship still shows up.
        self.cnn = nn.Sequential(
            layer_init(nn.Conv2d(stack, 32, 8, 4)), nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 4, 2)), nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, 1)), nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            n = self.cnn(torch.zeros(1, stack, PLAY_H, W)).shape[1]
        self.fc = nn.Sequential(layer_init(nn.Linear(n + NUM_SCALARS, hidden)), nn.ReLU())
        self.actor = layer_init(nn.Linear(hidden, NUM_ACTIONS), std=0.01)
        self.critic = layer_init(nn.Linear(hidden, 1), std=1.0)

    def forward(self, obs: torch.Tensor, scalars: torch.Tensor):
        """obs: (B, stack, 78, 128) uint8 colors 0..3; scalars: (B, 8) float."""
        x = (obs.float() * (1.0 / 3.0)).contiguous(memory_format=torch.channels_last)
        h = self.fc(torch.cat([self.cnn(x), scalars.float()], dim=1))
        return self.actor(h).float(), self.critic(h).squeeze(-1).float()


class FrameStack:
    """Keeps the last `stack` screens per env on the GPU."""

    def __init__(self, n: int, stack: int, device):
        self.buf = torch.zeros((n, stack, PLAY_H, W), dtype=torch.uint8, device=device)

    def reset(self, obs: torch.Tensor):
        self.buf[:] = obs.unsqueeze(1)
        return self.buf

    def push(self, obs: torch.Tensor, dones: torch.Tensor):
        """Add a screen; where an episode just restarted, fill the stack with it.
        Written without data-dependent branches so it never waits on the GPU."""
        buf = torch.roll(self.buf, shifts=-1, dims=1)
        buf[:, -1] = obs
        fresh = obs.unsqueeze(1).expand_as(buf)
        self.buf = torch.where(dones.view(-1, 1, 1, 1), fresh, buf)
        return self.buf
