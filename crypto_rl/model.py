"""PyTorch actor-critic network used by PPO."""
from __future__ import annotations

import torch
from torch import nn
from torch.distributions import Categorical

from .data import MARKET_FEATURES
from .env import STATE_DIM


class ActorCritic(nn.Module):
    def __init__(self, lookback: int, action_dim: int, hidden_size: int = 128) -> None:
        super().__init__()
        self.lookback = int(lookback)
        self.market_dim = len(MARKET_FEATURES)
        self.state_dim = STATE_DIM
        self.conv = nn.Sequential(
            nn.Conv1d(self.market_dim, 64, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.trunk = nn.Sequential(
            nn.Linear(64 + self.state_dim, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
        )
        self.policy_head = nn.Linear(hidden_size, action_dim)
        self.value_head = nn.Linear(hidden_size, 1)
        self.apply(self._init)
        nn.init.orthogonal_(self.policy_head.weight, gain=0.01)
        nn.init.zeros_(self.policy_head.bias)
        nn.init.orthogonal_(self.value_head.weight, gain=1.0)
        nn.init.zeros_(self.value_head.bias)

    @staticmethod
    def _init(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            nn.init.orthogonal_(module.weight, gain=2**0.5)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, observation: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if observation.ndim == 1:
            observation = observation.unsqueeze(0)
        market_size = self.lookback * self.market_dim
        market = observation[:, :market_size].reshape(-1, self.lookback, self.market_dim).transpose(1, 2)
        state = observation[:, market_size:]
        temporal = self.conv(market).squeeze(-1)
        hidden = self.trunk(torch.cat((temporal, state), dim=-1))
        return self.policy_head(hidden), self.value_head(hidden).squeeze(-1)

    @torch.no_grad()
    def act(self, observation: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, value = self(observation)
        distribution = Categorical(logits=logits)
        action = distribution.sample()
        return action, distribution.log_prob(action), value

    def evaluate_actions(
        self, observation: torch.Tensor, action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, value = self(observation)
        distribution = Categorical(logits=logits)
        return distribution.log_prob(action), distribution.entropy(), value
