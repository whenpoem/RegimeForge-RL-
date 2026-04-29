"""Dreamer-style World Model for RegimeForge.

Implements a Recurrent State-Space Model (RSSM) that learns latent
dynamics from observations, then trains a policy entirely inside the
learned environment (imagination).

Architecture:
- Encoder: observation -> embedding
- RSSM: deterministic recurrent state + stochastic latent state
- Decoder: latent state -> reconstructed observation
- Reward predictor: latent state -> scalar reward
- Actor: latent state -> action distribution
- Critic: latent state -> value estimate
"""

from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Any, Deque

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical, Independent, Normal, OneHotCategorical

from .config import TrainingConfig, config_to_snapshot


# ---------------------------------------------------------------------------
# RSSM Components
# ---------------------------------------------------------------------------

class ObsEncoder(nn.Module):
    """Encode observation to a fixed-size embedding."""

    def __init__(self, obs_dim: int, embed_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.SiLU(),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


class ObsDecoder(nn.Module):
    """Decode latent state back to observation."""

    def __init__(self, latent_dim: int, obs_dim: int, embed_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, obs_dim),
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        return self.net(latent)


class RewardPredictor(nn.Module):
    """Predict scalar reward from latent state."""

    def __init__(self, latent_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        return self.net(latent).squeeze(-1)


class RSSM(nn.Module):
    """Recurrent State-Space Model.

    State = (deterministic h, stochastic z)
    - h_t = gru(h_{t-1}, [z_{t-1}, a_{t-1}])
    - z_t ~ p(z_t | h_t)  prior
    - z_t ~ q(z_t | h_t, o_t)  posterior (used during training)
    """

    def __init__(
        self,
        obs_embed_dim: int,
        action_dim: int,
        latent_dim: int = 32,
        recurrent_dim: int = 64,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.recurrent_dim = recurrent_dim

        # Prior network: h_t -> z_t
        self.prior_net = nn.Sequential(
            nn.Linear(recurrent_dim, latent_dim * 2),
            nn.SiLU(),
            nn.Linear(latent_dim * 2, latent_dim * 2),
        )

        # Posterior network: [h_t, embed_t] -> z_t
        self.posterior_net = nn.Sequential(
            nn.Linear(recurrent_dim + obs_embed_dim, latent_dim * 2),
            nn.SiLU(),
            nn.Linear(latent_dim * 2, latent_dim * 2),
        )

        # Recurrent transition: GRU input = [z_{t-1}, a_{t-1}]
        self.transition_input = nn.Linear(latent_dim + action_dim, recurrent_dim)
        self.gru = nn.GRUCell(recurrent_dim, recurrent_dim)

    def initial_state(self, batch_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        h = torch.zeros(batch_size, self.recurrent_dim, device=device)
        z = torch.zeros(batch_size, self.latent_dim, device=device)
        return h, z

    def observe_step(
        self,
        h: torch.Tensor,
        z: torch.Tensor,
        action: torch.Tensor,
        obs_embed: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """One step with posterior (training mode).

        Returns: (h_new, z_prior, z_posterior, kl_loss)
        """
        # Transition
        transition_in = self.transition_input(torch.cat([z, action], dim=-1))
        h_new = self.gru(transition_in, h)

        # Prior
        prior_params = self.prior_net(h_new)
        prior_mean, prior_log_std = prior_params.chunk(2, dim=-1)
        prior_std = prior_log_std.clamp(-5, 2).exp()
        prior_dist = Independent(Normal(prior_mean, prior_std), 1)

        # Posterior
        post_params = self.posterior_net(torch.cat([h_new, obs_embed], dim=-1))
        post_mean, post_log_std = post_params.chunk(2, dim=-1)
        post_std = post_log_std.clamp(-5, 2).exp()
        post_dist = Independent(Normal(post_mean, post_std), 1)

        z_post = post_dist.rsample()

        # KL divergence
        kl = torch.distributions.kl_divergence(post_dist, prior_dist)
        kl = kl.clamp(min=1.0)  # free nats

        return h_new, prior_dist.rsample(), z_post, kl

    def imagine_step(
        self,
        h: torch.Tensor,
        z: torch.Tensor,
        action: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """One step with prior only (imagination mode).

        Returns: (h_new, z_new)
        """
        transition_in = self.transition_input(torch.cat([z, action], dim=-1))
        h_new = self.gru(transition_in, h)
        prior_params = self.prior_net(h_new)
        prior_mean, prior_log_std = prior_params.chunk(2, dim=-1)
        prior_std = prior_log_std.clamp(-5, 2).exp()
        dist = Independent(Normal(prior_mean, prior_std), 1)
        z_new = dist.rsample()
        return h_new, z_new


class ImagActor(nn.Module):
    """Policy network operating in latent space."""

    def __init__(self, latent_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, action_dim),
        )
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, latent: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean = self.net(latent)
        std = self.log_std.exp().expand_as(mean)
        return mean, std

    def sample(self, latent: torch.Tensor, deterministic: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        mean, std = self.forward(latent)
        if deterministic:
            action = torch.tanh(mean)
            log_prob = torch.zeros(mean.size(0), device=mean.device)
        else:
            dist = Normal(mean, std)
            pre_tanh = dist.rsample()
            action = torch.tanh(pre_tanh)
            correction = torch.log1p(-action.pow(2) + 1e-6)
            log_prob = dist.log_prob(pre_tanh).sum(dim=-1) - correction.sum(dim=-1)
        return action, log_prob


class ImagCritic(nn.Module):
    """Value network operating in latent space."""

    def __init__(self, latent_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        return self.net(latent).squeeze(-1)


# ---------------------------------------------------------------------------
# World Model Agent
# ---------------------------------------------------------------------------

class WorldModelAgent:
    """Dreamer-style agent: learn world model, then imagine and plan."""

    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        config: TrainingConfig,
        device: str = "cpu",
    ):
        self.device = torch.device(device)
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.config = config
        self.latent_dim = config.latent_dim
        self.recurrent_dim = config.recurrent_dim
        self.imag_horizon = config.imag_horizon
        self.gamma = config.gamma

        embed_dim = 64
        self.encoder = ObsEncoder(observation_dim, embed_dim).to(self.device)
        self.decoder = ObsDecoder(config.latent_dim, observation_dim, embed_dim).to(self.device)
        self.reward_pred = RewardPredictor(config.latent_dim).to(self.device)
        self.rssm = RSSM(embed_dim, action_dim, config.latent_dim, config.recurrent_dim).to(self.device)
        self.actor = ImagActor(config.latent_dim, action_dim).to(self.device)
        self.critic = ImagCritic(config.latent_dim).to(self.device)

        self.world_optimizer = torch.optim.Adam(
            list(self.encoder.parameters())
            + list(self.decoder.parameters())
            + list(self.reward_pred.parameters())
            + list(self.rssm.parameters()),
            lr=config.world_model_lr,
        )
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=config.learning_rate)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=config.learning_rate)

        self.rng = np.random.default_rng(config.seed)
        torch.manual_seed(config.seed)

        # Episode buffer for world model training
        self._episode_obs: list[np.ndarray] = []
        self._episode_actions: list[np.ndarray] = []
        self._episode_rewards: list[float] = []
        self._episode_dones: list[bool] = []

        # Current RSSM state
        self._h: torch.Tensor | None = None
        self._z: torch.Tensor | None = None

    def reset(self) -> None:
        """Reset RSSM state at episode boundary."""
        self._h, self._z = self.rssm.initial_state(1, self.device)
        self._episode_obs = []
        self._episode_actions = []
        self._episode_rewards = []
        self._episode_dones = []

    def act(self, observation: np.ndarray, deterministic: bool = False) -> dict[str, np.ndarray]:
        """Select action using the imagination policy."""
        if self._h is None:
            self.reset()
        obs_t = torch.as_tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            embed = self.encoder(obs_t)
            self._h, _, self._z, _ = self.rssm.observe_step(
                self._h, self._z,
                torch.zeros(1, self.action_dim, device=self.device),
                embed,
            )
            action, _ = self.actor.sample(self._z, deterministic=deterministic)
        action_np = action.squeeze(0).cpu().numpy()
        return {"action": action_np}

    def store_transition(
        self,
        observation: np.ndarray,
        action: np.ndarray,
        reward: float,
        done: bool,
    ) -> None:
        self._episode_obs.append(observation.copy())
        self._episode_actions.append(action.copy())
        self._episode_rewards.append(float(reward))
        self._episode_dones.append(bool(done))

    def update_world_model(self) -> dict[str, float] | None:
        """Train the world model on collected episode data."""
        if len(self._episode_obs) < 2:
            return None

        obs = torch.as_tensor(np.stack(self._episode_obs), dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(np.stack(self._episode_actions), dtype=torch.float32, device=self.device)
        rewards = torch.as_tensor(self._episode_rewards, dtype=torch.float32, device=self.device)

        h, z = self.rssm.initial_state(1, self.device)
        total_recon_loss = torch.zeros((), device=self.device)
        total_reward_loss = torch.zeros((), device=self.device)
        total_kl = torch.zeros((), device=self.device)

        for t in range(len(obs) - 1):
            embed = self.encoder(obs[t:t+1])
            h, _, z, kl = self.rssm.observe_step(h, z, actions[t:t+1], embed)
            recon = self.decoder(z)
            pred_reward = self.reward_pred(z)
            total_recon_loss = total_recon_loss + nn.functional.mse_loss(recon, obs[t+1:t+2])
            total_reward_loss = total_reward_loss + nn.functional.mse_loss(pred_reward, rewards[t:t+1])
            total_kl = total_kl + kl.mean()

        n = len(obs) - 1
        world_loss = (total_recon_loss + total_reward_loss + 0.5 * total_kl) / n

        self.world_optimizer.zero_grad(set_to_none=True)
        world_loss.backward()
        nn.utils.clip_grad_norm_(
            list(self.encoder.parameters())
            + list(self.decoder.parameters())
            + list(self.reward_pred.parameters())
            + list(self.rssm.parameters()),
            max_norm=1.0,
        )
        self.world_optimizer.step()

        return {
            "recon_loss": float(total_recon_loss.item() / n),
            "reward_loss": float(total_reward_loss.item() / n),
            "kl_loss": float(total_kl.item() / n),
            "world_loss": float(world_loss.item()),
        }

    def update_actor_critic(self) -> dict[str, float] | None:
        """Train actor and critic using imagined trajectories."""
        if len(self._episode_obs) < 2:
            return None

        obs = torch.as_tensor(np.stack(self._episode_obs), dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(np.stack(self._episode_actions), dtype=torch.float32, device=self.device)

        # Get a starting latent state from real data
        with torch.no_grad():
            embed = self.encoder(obs[:1])
            h, z = self.rssm.initial_state(1, self.device)
            for t in range(min(len(obs) - 1, 10)):
                a = torch.zeros(1, self.action_dim, device=self.device) if t == 0 else actions[t:t+1]
                h, _, z, _ = self.rssm.observe_step(h, z, a, embed)
                if t < len(obs) - 2:
                    embed = self.encoder(obs[t+1:t+2])

        # Imagine forward
        imagined_latents = [z]
        imagined_actions = []
        for _ in range(self.imag_horizon):
            action, _ = self.actor.sample(z)
            h, z = self.rssm.imagine_step(h, z, action)
            imagined_latents.append(z)
            imagined_actions.append(action)

        latents = torch.cat(imagined_latents, dim=0)
        rewards = self.reward_pred(latents[:-1])
        values = self.critic(latents)

        # Compute lambda-returns
        returns = torch.zeros_like(rewards)
        last = values[-1]
        for t in range(len(rewards) - 1, -1, -1):
            last = rewards[t] + self.gamma * last
            returns[t] = last

        # Actor loss: maximize expected returns
        actor_loss = -returns.mean()

        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
        self.actor_optimizer.step()

        self.world_optimizer.zero_grad(set_to_none=True)
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_pred = self.critic(latents[:-1].detach())
        critic_loss = nn.functional.mse_loss(critic_pred, returns.detach())
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
        self.critic_optimizer.step()

        return {
            "actor_loss": float(actor_loss.item()),
            "critic_loss": float(critic_loss.item()),
            "imagined_reward": float(rewards.mean().item()),
        }

    def update(self) -> dict[str, float] | None:
        """Full update: world model + actor/critic."""
        wm_metrics = self.update_world_model()
        ac_metrics = self.update_actor_critic()
        if wm_metrics is None or ac_metrics is None:
            return wm_metrics or ac_metrics
        return {**wm_metrics, **ac_metrics}

    def save_checkpoint(self, directory: str | Path) -> None:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        torch.save(self.encoder.state_dict(), d / "encoder.pt")
        torch.save(self.decoder.state_dict(), d / "decoder.pt")
        torch.save(self.reward_pred.state_dict(), d / "reward.pt")
        torch.save(self.rssm.state_dict(), d / "rssm.pt")
        torch.save(self.actor.state_dict(), d / "actor.pt")
        torch.save(self.critic.state_dict(), d / "critic.pt")
        torch.save(self.world_optimizer.state_dict(), d / "world_opt.pt")
        torch.save(self.actor_optimizer.state_dict(), d / "actor_opt.pt")
        torch.save(self.critic_optimizer.state_dict(), d / "critic_opt.pt")
        torch.save({"rng_state": self.rng.bit_generator.state}, d / "meta.pt")

    def load_checkpoint(self, directory: str | Path, weights_only: bool = True) -> None:
        d = Path(directory)
        self.encoder.load_state_dict(torch.load(d / "encoder.pt", map_location=self.device, weights_only=True))
        self.decoder.load_state_dict(torch.load(d / "decoder.pt", map_location=self.device, weights_only=True))
        self.reward_pred.load_state_dict(torch.load(d / "reward.pt", map_location=self.device, weights_only=True))
        self.rssm.load_state_dict(torch.load(d / "rssm.pt", map_location=self.device, weights_only=True))
        self.actor.load_state_dict(torch.load(d / "actor.pt", map_location=self.device, weights_only=True))
        self.critic.load_state_dict(torch.load(d / "critic.pt", map_location=self.device, weights_only=True))
        if not weights_only:
            for name, opt in [("world_opt", self.world_optimizer), ("actor_opt", self.actor_optimizer), ("critic_opt", self.critic_optimizer)]:
                p = d / f"{name}.pt"
                if p.exists():
                    opt.load_state_dict(torch.load(p, map_location=self.device, weights_only=True))
            meta_path = d / "meta.pt"
            if meta_path.exists():
                meta = torch.load(meta_path, map_location="cpu", weights_only=False)
                if isinstance(meta, dict) and "rng_state" in meta:
                    self.rng.bit_generator.state = meta["rng_state"]


__all__ = ["WorldModelAgent"]
