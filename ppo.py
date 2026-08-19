"""
ppo.py
------
Proximal Policy Optimisation (PPO) controller for the evader, as described in
the project plan (Sections 3 & 4).

ACTION DESIGN
-------------
The network does NOT output a raw velocity [vx, vy]. It outputs a single
scalar "deroute multiplier" m in [-1, 1]. That multiplier is turned into an
actual velocity by rotating the straight-line-to-target heading by an angle
proportional to m:

    heading = angle(target - evader_pos) + m * MAX_DETOUR_ANGLE

m = 0  -> evader heads exactly at the target (the "original path").
m = +1 -> evader turns up to MAX_DETOUR_ANGLE one way off that line.
m = -1 -> turns the same amount the other way.

This is exactly what the project asked for: the network's single output IS
"how much (and which way) to deroute", not an arbitrary 2D vector, and the
value it learns to output is entirely a function of what the reward system
(Section 5: progress to target, safety margin from pursuers, terminal
reward/penalty) rewards - i.e. it is the end result of all those reward
factors combined into one interpretable number.

Because it's a rotation (not an additive vector blend), a large multiplier
can produce a genuinely sharp curve/loop around a pursuer - not capped at
~90 degrees the way blending two fixed-length vectors would be.

This is used by evader.py: whenever the barrier value B < 0 (i.e. the
classical differential-game solution has already determined that the evader
loses the pursuit), evader.return_velocity() hands control over to the PPO
policy trained here instead of following the fixed losing trajectory.

Contents
--------
- ActorCritic            : shared-torso Gaussian policy + value network,
                            1-dimensional bounded action (the multiplier)
- RolloutBuffer          : stores on-policy trajectories for one PPO update
- PPOAgent               : action selection + clipped-objective update
- deviation_to_velocity  : the ONE place the multiplier -> velocity
                            conversion happens, used identically by training
                            (PursuitEvasionEnv.step) and inference
                            (get_ppo_velocity), so the two can never drift
                            out of sync with each other.
- PursuitEvasionEnv      : 2-pursuer vs 1-evader training environment that
                            reuses the *existing* classical pursuer controller
                            (pursuer.py), with a curriculum that ramps
                            pursuer speed and danger_radius up over training.
- train()                : runs the PPO training loop across many randomised
                            simulations and saves a checkpoint
- PPOEvaderPolicy        : lightweight inference wrapper (loads a checkpoint
                            once and is cached at module level)
- get_ppo_velocity()     : convenience function called from evader.py
"""

import os
import numpy as np

try:
    import torch
    import torch.nn as nn
    from torch.distributions import Normal
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

from pursuer import pursuer as ClassicalPursuer

# --------------------------------------------------------------------------- #
# Problem constants (Section 4 of the project plan)
# --------------------------------------------------------------------------- #
OBS_DIM = 8            # [xe, ye, xp1, yp1, xp2, yp2, xt, yt]
ACT_DIM = 1             # [deroute_multiplier], bounded to [-1, 1]
DEFAULT_CHECKPOINT = os.path.join(os.path.dirname(__file__), "ppo_evader.pth")

# How far (in degrees) a multiplier of +/-1 is allowed to rotate the evader
# off the straight line to the target. 160 deg allows near-full loops around
# a pursuer while still leaving a small margin so it never moves in exactly
# the wrong direction even at full deroute.
MAX_DETOUR_ANGLE_DEG = 160.0
MAX_DETOUR_ANGLE_RAD = np.deg2rad(MAX_DETOUR_ANGLE_DEG)

# --------------------------------------------------------------------------- #
# PPO / curriculum hyperparameters (tune here)
# --------------------------------------------------------------------------- #
ENTROPY_COEF_DEFAULT = 0.05
# NOTE: the classical Apollonius-circle formulas in pursuer.py/evader.py
# (alpha = evader_speed / pursuer_speed, terms like 1/(1-alpha**2)) are only
# mathematically defined for alpha < 1, i.e. pursuer_speed > evader_speed.
# CURRICULUM_START_SPEED must stay above the evader's speed (18) or alpha
# passes through exactly 1 partway through training and the classical
# controller divides by zero.
CURRICULUM_START_SPEED = 20.0        # pursuer speed at episode 0 (must be > evader_speed=18)
CURRICULUM_END_SPEED = 30.0          # pursuer speed at final episode
CURRICULUM_START_RADIUS = 150.0      # danger radius at episode 0
CURRICULUM_END_RADIUS = 80.0         # danger radius at final episode


# --------------------------------------------------------------------------- #
# Network
# --------------------------------------------------------------------------- #
if _TORCH_AVAILABLE:

    class ActorCritic(nn.Module):
        """Gaussian policy (actor) + state-value estimator (critic).

        The actor outputs ONE number: the deroute multiplier, bounded to
        [-1, 1] via tanh directly inside forward(). This is the only place
        the bound is applied - select_action() and inference both use this
        same bounded mean, so there is no train/inference mismatch.
        """

        def __init__(self, obs_dim=OBS_DIM, act_dim=ACT_DIM, hidden=64):
            super().__init__()
            self.shared = nn.Sequential(
                nn.Linear(obs_dim, hidden), nn.Tanh(),
                nn.Linear(hidden, hidden), nn.Tanh(),
            )
            self.mean_head = nn.Linear(hidden, act_dim)
            # log_std initialised for an action space of scale ~1 (the
            # multiplier lives in [-1,1]), giving healthy exploration from
            # the first rollout without needing a huge std.
            self.log_std = nn.Parameter(torch.ones(act_dim) * np.log(0.8))
            self.value_head = nn.Linear(hidden, 1)

        def forward(self, obs):
            z = self.shared(obs)
            mean = torch.tanh(self.mean_head(z))  # bounded to [-1, 1]
            value = self.value_head(z).squeeze(-1)
            std = torch.exp(self.log_std)
            return mean, std, value

        def act(self, obs, deterministic=False):
            mean, std, value = self.forward(obs)
            if deterministic:
                return mean, None, value
            dist = Normal(mean, std)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(-1)
            return action, log_prob, value

        def evaluate(self, obs, action):
            mean, std, value = self.forward(obs)
            dist = Normal(mean, std)
            log_prob = dist.log_prob(action).sum(-1)
            entropy = dist.entropy().sum(-1)
            return log_prob, entropy, value


# --------------------------------------------------------------------------- #
# Rollout storage
# --------------------------------------------------------------------------- #
class RolloutBuffer:
    def __init__(self):
        self.obs, self.actions, self.log_probs = [], [], []
        self.rewards, self.dones, self.values = [], [], []

    def add(self, obs, action, log_prob, reward, done, value):
        self.obs.append(obs)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value)

    def clear(self):
        self.__init__()

    def __len__(self):
        return len(self.obs)


# --------------------------------------------------------------------------- #
# PPO agent
# --------------------------------------------------------------------------- #
class PPOAgent:
    def __init__(self, obs_dim=OBS_DIM, act_dim=ACT_DIM, lr=3e-4,
                 gamma=0.99, lam=0.95, clip_eps=0.2,
                 epochs=10, minibatch_size=64, entropy_coef=ENTROPY_COEF_DEFAULT,
                 value_coef=0.5, max_speed=18.0, device="cpu"):
        if not _TORCH_AVAILABLE:
            raise ImportError(
                "PyTorch is required to train/run the PPO evader controller. "
                "Install it with: pip install torch"
            )
        self.device = torch.device(device)
        self.net = ActorCritic(obs_dim, act_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.gamma = gamma
        self.lam = lam
        self.clip_eps = clip_eps
        self.epochs = epochs
        self.minibatch_size = minibatch_size
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_speed = max_speed

    def select_action(self, obs, deterministic=False):
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            action, log_prob, value = self.net.act(obs_t, deterministic=deterministic)
        action_np = action.squeeze(0).cpu().numpy()
        # The mean is already tanh-bounded to [-1,1], but a *sampled* action
        # (mean + Gaussian noise) can still land slightly outside that range.
        # Clip it here, and store this SAME clipped value in the training
        # buffer (see train()) so what gets executed, what gets logged, and
        # what the log-prob update is computed against are always identical.
        action_np = np.clip(action_np, -1.0, 1.0)
        log_prob_val = None if log_prob is None else log_prob.item()
        return action_np, log_prob_val, value.item()

    def _compute_gae(self, rewards, values, dones, last_value):
        advantages = np.zeros(len(rewards), dtype=np.float32)
        gae = 0.0
        values = values + [last_value]
        for t in reversed(range(len(rewards))):
            mask = 1.0 - dones[t]
            delta = rewards[t] + self.gamma * values[t + 1] * mask - values[t]
            gae = delta + self.gamma * self.lam * mask * gae
            advantages[t] = gae
        returns = advantages + np.array(values[:-1], dtype=np.float32)
        return advantages, returns

    def update(self, buffer: RolloutBuffer, last_value: float):
        obs = torch.as_tensor(np.array(buffer.obs), dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(np.array(buffer.actions), dtype=torch.float32, device=self.device)
        old_log_probs = torch.as_tensor(np.array(buffer.log_probs), dtype=torch.float32, device=self.device)

        advantages, returns = self._compute_gae(buffer.rewards, buffer.values, buffer.dones, last_value)
        advantages = torch.as_tensor(advantages, dtype=torch.float32, device=self.device)
        returns = torch.as_tensor(returns, dtype=torch.float32, device=self.device)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        n = len(buffer)
        idx = np.arange(n)
        last_stats = {}
        for _ in range(self.epochs):
            np.random.shuffle(idx)
            for start in range(0, n, self.minibatch_size):
                mb_idx = idx[start:start + self.minibatch_size]
                mb_idx_t = torch.as_tensor(mb_idx, dtype=torch.long, device=self.device)

                log_probs, entropy, values = self.net.evaluate(obs[mb_idx_t], actions[mb_idx_t])
                ratio = torch.exp(log_probs - old_log_probs[mb_idx_t])

                surr1 = ratio * advantages[mb_idx_t]
                surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advantages[mb_idx_t]
                # L_CLIP = E[min(r_t(theta) * A_t, clip(r_t(theta), 1-eps, 1+eps) * A_t)]
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = nn.functional.mse_loss(values, returns[mb_idx_t])
                entropy_loss = -entropy.mean()

                loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), 0.5)
                self.optimizer.step()

                last_stats = {
                    "policy_loss": policy_loss.item(),
                    "value_loss": value_loss.item(),
                    "entropy": -entropy_loss.item(),
                }
        return last_stats

    def save(self, path=DEFAULT_CHECKPOINT):
        torch.save({"model_state": self.net.state_dict(), "max_speed": self.max_speed}, path)

    def load(self, path=DEFAULT_CHECKPOINT):
        checkpoint = torch.load(path, map_location=self.device)
        self.net.load_state_dict(checkpoint["model_state"])
        self.max_speed = checkpoint.get("max_speed", self.max_speed)


# --------------------------------------------------------------------------- #
# THE single place the multiplier becomes a velocity - used identically by
# training (env.step) and inference (get_ppo_velocity).
# --------------------------------------------------------------------------- #
def deviation_to_velocity(deviation_multiplier, evader_pos, target_pos, speed,
                          pursuer_positions=None):
    """
    m = 0   -> straight at the target
    m > 0   -> rotate away from the nearest pursuer (prioritize safety)
    m < 0   -> rotate toward the target (prioritize progress)
    
    This makes the learned multiplier encode a safety-vs-progress tradeoff
    rather than an arbitrary heading offset.
    """
    evader_pos = np.asarray(evader_pos, dtype=np.float64).flatten()[:2]
    target_pos = np.asarray(target_pos, dtype=np.float64).flatten()[:2]
    m = float(np.clip(deviation_multiplier, -1.0, 1.0))

    to_target = target_pos - evader_pos
    dist_to_target = np.linalg.norm(to_target)
    forward_angle = np.arctan2(to_target[1], to_target[0]) if dist_to_target > 1e-6 else 0.0

    # The report uses two pursuers, but only the pursuer selected by the
    # linear-program assignment is an active threat.  The other pursuer is a
    # dummy/idle agent and must not steer the evader.
    if pursuer_positions is not None and len(pursuer_positions) > 0:
        active_pursuer = np.asarray(pursuer_positions[0])
        to_pursuer = active_pursuer - evader_pos
        away_angle = np.arctan2(to_pursuer[1], to_pursuer[0]) + np.pi  # opposite direction

        # `away_angle` is the heading that points directly away from the pursuer.
        # A positive multiplier should move the evader away from the pursuer, not
        # toward it. Using the same rotation sign convention as the project docs:
        #    m > 0 -> rotate away from pursuer, m < 0 -> rotate back toward target.
        heading = forward_angle + m * (away_angle - forward_angle)
    else:
        # Fallback if no pursuer info: just rotate off the forward line
        heading = forward_angle + m * MAX_DETOUR_ANGLE_RAD

    return np.array([np.cos(heading), np.sin(heading)]) * speed


# --------------------------------------------------------------------------- #
# LP role assignment for the 2-pursuer / 1-evader report scenario
# --------------------------------------------------------------------------- #
def select_active_pursuer_lp(evader_pos, pursuer_positions, target,
                              evader_speed, pursuer_speed):
    """Return the pursuer index selected by the same LP objective used in
    assign.py. Exactly one pursuer is active; the other remains dummy/idle."""
    xe = np.asarray(evader_pos, dtype=float).flatten()[:2]
    ps = np.asarray(pursuer_positions, dtype=float).reshape(-1, 2)
    n = len(ps)
    alpha = evader_speed / np.asarray(pursuer_speed, dtype=float).reshape(-1)
    nes = np.sum(xe ** 2)
    nps = np.sum(ps ** 2, axis=1)
    B = nes - (alpha ** 2) * nps

    # Same value expressions as assignment.val_mat() for m=1.
    dist = np.linalg.norm(ps - xe[None, :], axis=1)
    difference = xe[None, :] - ps
    dist2 = np.maximum(np.linalg.norm(difference, axis=1), 1e-12)
    val = np.zeros(n)
    idx1 = (B >= 0) & (alpha == 1)
    val[idx1] = 0.5 * (nes - nps[idx1] / np.maximum(dist[idx1], 1e-9))
    idx2 = (B >= 0) & (alpha < 1)
    alphasq = alpha ** 2
    val[idx2] = (np.linalg.norm(xe[None, :] - alphasq[:, None] * ps, axis=1)[idx2]
                  - (alpha[idx2] / (1 - alphasq[idx2])) * dist2[idx2])
    idx3 = (B < 0) & (alpha <= 1)
    val[idx3] = -np.sqrt(nps[idx3]) + np.sqrt(nes) / alpha[idx3]

    # assign.py uses a = val only for B >= 0 and alpha <= 1, then the LP
    # maximises that objective subject to exactly one evader assignment.
    feasible = (B >= 0) & (alpha <= 1)
    objective = np.where(feasible, val, 0.0)
    return int(np.argmax(objective))


# --------------------------------------------------------------------------- #
# Training environment (2 pursuers vs 1 PPO evader, target at origin)
# --------------------------------------------------------------------------- #
class PursuitEvasionEnv:
    """
    Reproduces the scenario in Section 4 of the project plan:
    - 2 pursuers using the existing differential-game controller (pursuer.py)
    - 1 evader controlled by the PPO policy being trained (via the deroute
      multiplier -> velocity conversion above)
    - target fixed at the origin
    - pursuers are faster than the evader, ramped up over a curriculum

    Reward (Section 5, with adaptive safety weighting):
        R = w_target * R_target + w_safety_eff * R_safety - step_penalty
        R_target  = d_target[t-1]   - d_target[t]
        R_safety  = d_pursuer[t]    - d_pursuer[t-1]      (nearest pursuer)
        terminal  : +a on reaching target, -a on capture

    w_safety_eff is NOT a fixed constant - it scales up smoothly as the
    nearest pursuer gets closer than `danger_radius`, so the network learns
    to output a larger deroute multiplier well before a pursuer is
    dangerously close, and to relax back toward multiplier ~= 0 (straight to
    target) once nothing is nearby. This reward shaping is the entire
    mechanism that teaches the network what multiplier value to output -
    there is no separate hand-coded switch for "when to deroute".
    """

    def __init__(self, bounds=300.0, pursuer_speed=30.0, evader_speed=29.0,
                 capture_radius=5.0, target_radius=5.0, dt=0.1, max_steps=500,
                 w_target=4.0, w_safety=5, danger_radius=80.0,
                 max_safety_boost=4.0, step_penalty=0.01, terminal_reward=400.0,
                 w_turn_bonus=8.0):
        self.bounds = bounds
        self.pursuer_speed = pursuer_speed
        self.evader_speed = evader_speed
        self.capture_radius = capture_radius
        self.target_radius = target_radius
        self.dt = dt
        self.max_steps = max_steps
        self.w_target = w_target
        self.w_safety = w_safety
        self.danger_radius = danger_radius
        self.max_safety_boost = max_safety_boost
        self.step_penalty = step_penalty
        self.terminal_reward = terminal_reward
        self.w_turn_bonus = w_turn_bonus
        self.target = np.zeros(2)

        self.pursuers = [ClassicalPursuer(np.zeros(2), pursuer_speed, i) for i in range(2)]

    def reset(self, difficulty=1.0):
        """
        difficulty in [0, 1]. At difficulty=1 this reproduces the real
        scenario (fully random start positions everywhere). At lower
        difficulty, pursuers are kept at least `min_sep` away from the
        evader's start position, giving a fresh policy a chance to learn
        basic navigate-to-target / avoid-nearby-threat behaviour before
        being thrown into worst-case starting configurations.
        """
        self.t = 0
        self.evader_pos = np.random.uniform(-self.bounds, self.bounds, size=2)
        min_sep = self.bounds * 0.6 * (1.0 - np.clip(difficulty, 0.0, 1.0))
        for p in self.pursuers:
            candidate = np.random.uniform(-self.bounds, self.bounds, size=2)
            for _ in range(20):
                if np.linalg.norm(candidate - self.evader_pos) >= min_sep:
                    break
                candidate = np.random.uniform(-self.bounds, self.bounds, size=2)
            p.position = candidate
            p.status = 0
            p.reset_lock()

        # Match assign.py: the LP assigns exactly ONE of the two pursuers to
        # the single evader. That pursuer is active; the other is a dummy and
        # remains stationary for the entire episode.
        self.active_pursuer_idx = select_active_pursuer_lp(
            self.evader_pos, [p.position for p in self.pursuers],
            self.target, self.evader_speed,
            [p.speed for p in self.pursuers])

        self._prev_target_dist = np.linalg.norm(self.evader_pos - self.target)
        self._prev_pursuer_dist = self._active_pursuer_dist()
        return self._get_obs()

    def _active_pursuer_dist(self):
        # assign.py's updateStatus() only ever checks the SPECIFIC assigned
        # pursuer for capture, never "whichever pursuer is nearest" - the
        # unassigned pursuer is not a threat at all, frozen or not.
        return np.linalg.norm(self.evader_pos - self.pursuers[self.active_pursuer_idx].position)

    def _nearest_pursuer_dist(self):
        return min(np.linalg.norm(self.evader_pos - p.position) for p in self.pursuers)

    def _barrier_value(self, p):
        alpha = self.evader_speed / p.speed
        return np.sum(self.evader_pos ** 2) - (alpha ** 2) * np.sum(p.position ** 2)

# In ppo.py, PursuitEvasionEnv._get_obs():

    def _get_obs(self):
        # Current (raw positions):
        # p0, p1 = self.pursuers[0].position, self.pursuers[1].position
        # return np.array([...evader_pos..., p0[0], p0[1], p1[0], p1[1], ...])

        # BETTER (relative vectors):
        p0 = self.pursuers[0].position - self.evader_pos  # vector FROM evader TO pursuer
        p1 = self.pursuers[1].position - self.evader_pos  # same
        target = self.target - self.evader_pos  # vector FROM evader TO target
        
        return np.array([
            self.evader_pos[0], self.evader_pos[1],
            p0[0], p0[1],  # relative pursuer 1
            p1[0], p1[1],  # relative pursuer 2
            target[0], target[1],  # relative target
        ], dtype=np.float32)

    class _EvaderProxy:
        """Minimal stand-in so pursuer.return_velocity() can read position/speed."""
        def __init__(self, position, speed):
            self.position = position
            self.speed = speed

    def step(self, action):
        self.t += 1
        deviation_multiplier = float(np.asarray(action).flatten()[0])
        active_position = self.pursuers[self.active_pursuer_idx].position
        velocity = deviation_to_velocity(
            deviation_multiplier, self.evader_pos, self.target, self.evader_speed,
            pursuer_positions=[active_position])

        # move evader
        self.evader_pos = self.evader_pos + self.dt * velocity

        # Only the assigned/active pursuer moves - the other stays frozen
        # for the whole episode, exactly matching assign.py's step(), where
        # an unassigned pursuer's updatePos() is simply never called.
        evader_proxy = self._EvaderProxy(self.evader_pos, self.evader_speed)
        active = self.pursuers[self.active_pursuer_idx]
        B = self._barrier_value(active)
        vel = active.return_velocity(evader_proxy, B, self.capture_radius)
        active.position = active.position + self.dt * np.asarray(vel).flatten()

        target_dist = np.linalg.norm(self.evader_pos - self.target)
        pursuer_dist = self._active_pursuer_dist()

        r_target = self._prev_target_dist - target_dist
        r_safety = pursuer_dist - self._prev_pursuer_dist

        danger = np.clip(1.0 - pursuer_dist / self.danger_radius, 0.0, 1.0)
        w_safety_eff = self.w_safety * (1.0 + self.max_safety_boost * danger)

        # Direct shaping term: reward producing a large |multiplier| whenever
        # danger is high, regardless of whether THIS step's distance delta
        # happened to improve. Against a faster pursuer, r_safety can stay
        # ambiguous/negative even while turning is the right call, since the
        # pursuer keeps closing regardless - this term removes that
        # ambiguity and gives an immediate, unmistakable "turn now" signal.
        turn_bonus = self.w_turn_bonus * danger * abs(deviation_multiplier)

        reward = self.w_target * r_target + w_safety_eff * r_safety - self.step_penalty + turn_bonus

        self._prev_target_dist = target_dist
        self._prev_pursuer_dist = pursuer_dist

        done = False
        info = {"outcome": "running"}
        if pursuer_dist < self.capture_radius:
            reward -= self.terminal_reward
            done = True
            info["outcome"] = "captured"
        elif target_dist < self.target_radius:
            reward += self.terminal_reward
            done = True
            info["outcome"] = "reached_target"
        elif self.t >= self.max_steps:
            done = True
            info["outcome"] = "timeout"
        elif np.any(np.abs(self.evader_pos) > self.bounds * 1.5):
            done = True
            info["outcome"] = "out_of_bounds"

        return self._get_obs(), reward, done, info


# --------------------------------------------------------------------------- #
# Training loop
# --------------------------------------------------------------------------- #
def train(num_episodes=2000, steps_per_update=2048, save_path=DEFAULT_CHECKPOINT,
          log_every=20, seed=0, warmup_episodes=400,

          pursuer_speed_start=CURRICULUM_START_SPEED,
          pursuer_speed_end=CURRICULUM_END_SPEED,
          danger_radius_start=CURRICULUM_START_RADIUS,
          danger_radius_end=CURRICULUM_END_RADIUS,
          entropy_coef=ENTROPY_COEF_DEFAULT):
    if not _TORCH_AVAILABLE:
        raise ImportError("PyTorch is required to train the PPO evader controller.")

    np.random.seed(seed)
    torch.manual_seed(seed)

    env = PursuitEvasionEnv(pursuer_speed=pursuer_speed_start, danger_radius=danger_radius_start)
    agent = PPOAgent(max_speed=env.evader_speed, entropy_coef=entropy_coef)
    buffer = RolloutBuffer()

    def current_difficulty(ep):
        if warmup_episodes <= 0:
            return 1.0
        return float(np.clip(ep / warmup_episodes, 0.0, 1.0))

    obs = env.reset(difficulty=current_difficulty(0))
    episode_reward = 0.0
    episode_count = 0
    outcomes = []
    ep_lengths = []
    ep_len = 0

    # ── History tracking for the training-time analysis graphs ──────────────
    episode_reward_history = []   # reward per completed episode
    episode_outcome_history = []  # outcome string per completed episode
    loss_history = []             # (episode_count_at_update, combined_loss)

    collected = 0
    while episode_count < num_episodes:
        collected = 0
        buffer.clear()
        while collected < steps_per_update:
            action, log_prob, value = agent.select_action(obs)
            next_obs, reward, done, info = env.step(action)
            buffer.add(obs, action, log_prob, reward, float(done), value)
            episode_reward += reward
            ep_len += 1
            obs = next_obs
            collected += 1

            if done:
                outcomes.append(info["outcome"])
                ep_lengths.append(ep_len)
                ep_len = 0
                episode_count += 1
                episode_reward_history.append(episode_reward)
                episode_outcome_history.append(info["outcome"])
                if episode_count % log_every == 0:
                    recent = outcomes[-log_every:]
                    win_rate = np.mean([o == "reached_target" for o in recent])
                    capture_rate = np.mean([o == "captured" for o in recent])
                    timeout_rate = np.mean([o == "timeout" for o in recent])
                    print(f"Episode {episode_count}/{num_episodes} "
                          f"difficulty={current_difficulty(episode_count):.2f} "
                          f"pursuer_speed={env.pursuer_speed:.1f} "
                          f"danger_radius={env.danger_radius:.0f} "
                          f"reward={episode_reward:.2f} "
                          f"avg_len={np.mean(ep_lengths[-log_every:]):.0f} "
                          f"win={win_rate:.2f} captured={capture_rate:.2f} timeout={timeout_rate:.2f}")

                # curriculum: ramp pursuer speed / danger radius across training
                progress = min(1.0, episode_count / max(1, num_episodes))
                env.pursuer_speed = pursuer_speed_start + progress * (pursuer_speed_end - pursuer_speed_start)
                env.danger_radius = danger_radius_start + progress * (danger_radius_end - danger_radius_start)
                for p in env.pursuers:
                    p.speed = env.pursuer_speed

                obs = env.reset(difficulty=current_difficulty(episode_count))
                episode_reward = 0.0

            if collected >= steps_per_update:
                break

        _, _, last_value = agent.select_action(obs)
        stats = agent.update(buffer, last_value)
        if stats:
            combined_loss = stats["policy_loss"] + stats["value_loss"]
            loss_history.append((episode_count, combined_loss))

    agent.save(save_path)
    print(f"Training complete. Model saved to {save_path}")
    _plot_training_curves(episode_reward_history, episode_outcome_history, loss_history)
    return agent


def _plot_training_curves(episode_rewards, episode_outcomes, loss_history, window=50):

    import matplotlib.pyplot as plt

    episodes = np.arange(1, len(episode_rewards) + 1)
    is_win = np.array([o == "reached_target" for o in episode_outcomes], dtype=float)
    is_captured = np.array([o == "captured" for o in episode_outcomes], dtype=float)

    def rolling_mean(x, w):
        if len(x) == 0:
            return x
        w = min(w, len(x))
        kernel = np.ones(w) / w
        return np.convolve(x, kernel, mode="valid")

    win_rate_roll = rolling_mean(is_win, window)
    captured_rate_roll = rolling_mean(is_captured, window)
    win_rate_x = episodes[window - 1:] if len(episodes) >= window else episodes[len(episodes) - len(win_rate_roll):]
    captured_rate_x = win_rate_x

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), num="Training Curves")

    axes[0, 0].plot(episodes, episode_rewards, color='tab:blue', label="Episode reward")
    axes[0, 0].set_title("Episode Reward Over Training")
    axes[0, 0].set_xlabel("Episode")
    axes[0, 0].set_ylabel("Reward")
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    axes[0, 1].plot(win_rate_x, win_rate_roll, color='tab:green', label=f"Win rate ({window}-ep rolling)")
    axes[0, 1].set_title("Win Rate Over Training")
    axes[0, 1].set_xlabel("Episode")
    axes[0, 1].set_ylabel("Win rate")
    axes[0, 1].set_ylim(-0.05, 1.05)
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    axes[1, 0].plot(captured_rate_x, captured_rate_roll, color='tab:red', label=f"Captured rate ({window}-ep rolling)")
    axes[1, 0].set_title("Captured Rate Over Training")
    axes[1, 0].set_xlabel("Episode")
    axes[1, 0].set_ylabel("Captured rate")
    axes[1, 0].set_ylim(-0.05, 1.05)
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    if loss_history:
        loss_x, loss_y = zip(*loss_history)
        axes[1, 1].plot(loss_x, loss_y, color='tab:orange', label="PPO loss (policy + value)")
    axes[1, 1].set_title("Training Loss Over Training")
    axes[1, 1].set_xlabel("Episode")
    axes[1, 1].set_ylabel("Loss")
    axes[1, 1].legend()
    axes[1, 1].grid(True)

    fig.tight_layout()
    fig.savefig("training_curves.png", dpi=150)
    print("Training curves saved to training_curves.png")
    try:
        plt.show()
    except Exception:
        pass



class PPOEvaderPolicy:
    """Loads a trained checkpoint once and reuses it for every call."""

    _instance = None

    def __init__(self, checkpoint_path=DEFAULT_CHECKPOINT, max_speed=18.0):
        self.available = False
        self.agent = None
        if not _TORCH_AVAILABLE:
            print("[ppo.py] PyTorch not installed - PPO evader controller unavailable.")
            return
        if not os.path.exists(checkpoint_path):
            print(f"[ppo.py] No trained checkpoint found at '{checkpoint_path}'. "
                  f"Run ppo.train() first. Falling back to a heuristic evasion policy.")
            return
        try:
            self.agent = PPOAgent(max_speed=max_speed)
            self.agent.load(checkpoint_path)
            self.available = True
        except Exception as exc:
            print(f"[ppo.py] Failed to load PPO checkpoint: {exc}. "
                  f"Falling back to a heuristic evasion policy.")

    @classmethod
    def get_instance(cls, checkpoint_path=DEFAULT_CHECKPOINT, max_speed=18.0):
        if cls._instance is None:
            cls._instance = cls(checkpoint_path=checkpoint_path, max_speed=max_speed)
        return cls._instance

    def get_deviation_multiplier(self, obs):

        if self.available:
            action, _, _ = self.agent.select_action(obs, deterministic=True)
            return float(np.asarray(action).flatten()[0])
        return _heuristic_deviation(obs)


def _heuristic_deviation(obs):

    xe, ye, xp1, yp1, xp2, yp2, xt, yt = obs
    evader_pos = np.array([xe, ye])
    target = np.array([xt, yt])
    pursuers = [np.array([xp1, yp1]), np.array([xp2, yp2])]

    to_target = target - evader_pos
    dist_to_target = np.linalg.norm(to_target)
    forward = to_target / dist_to_target if dist_to_target > 1e-6 else np.array([1.0, 0.0])

    nearest = min(pursuers, key=lambda p: np.linalg.norm(evader_pos - p))
    nearest_dist = np.linalg.norm(evader_pos - nearest)


    to_pursuer = nearest - evader_pos
    cross = forward[0] * to_pursuer[1] - forward[1] * to_pursuer[0]
    side = -1.0 if cross > 0 else 1.0

    danger_radius = 100.0
    strength = np.clip(1.0 - nearest_dist / danger_radius, 0.0, 1.0)
    return float(np.clip(side * strength, -1.0, 1.0))


def get_ppo_velocity(evader_pos, pursuers, target, max_speed,
                      checkpoint_path=DEFAULT_CHECKPOINT, return_multiplier=False,
                      active_pursuer_index=0):

    evader_pos = np.asarray(evader_pos, dtype=np.float32).flatten()[:2]
    target = np.asarray(target, dtype=np.float32).flatten()[:2]

    has_objects = len(pursuers) > 0 and hasattr(pursuers[0], "position")
    positions = [np.asarray(p.position if has_objects else p, dtype=np.float32).flatten()[:2] for p in pursuers]
    # Preserve pursuer identity/order for the 8-D report observation.
    if len(positions) == 0:
        positions = [evader_pos.copy(), evader_pos.copy()]
    elif len(positions) == 1:
        positions = [positions[0], positions[0]]
    else:
        positions = positions[:2]
    active_pursuer_index = int(np.clip(active_pursuer_index, 0, len(positions) - 1))

    obs = np.array([
        evader_pos[0], evader_pos[1],
        positions[0][0], positions[0][1],
        positions[1][0], positions[1][1],
        target[0], target[1],
    ], dtype=np.float32)

    policy = PPOEvaderPolicy.get_instance(checkpoint_path=checkpoint_path, max_speed=max_speed)
    deviation_multiplier = policy.get_deviation_multiplier(obs)
    velocity = deviation_to_velocity(
        deviation_multiplier, evader_pos, target, max_speed,
        pursuer_positions=[positions[active_pursuer_index]])
    if return_multiplier:
        return velocity, deviation_multiplier
    return velocity


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train the PPO evader controller.")
    parser.add_argument("--episodes", type=int, default=2000)
    parser.add_argument("--steps-per-update", type=int, default=2048)
    parser.add_argument("--save-path", type=str, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--warmup-episodes", type=int, default=400,
                         help="Episodes over which start-position difficulty ramps to full random")
    parser.add_argument("--entropy-coef", type=float, default=ENTROPY_COEF_DEFAULT)
    parser.add_argument("--pursuer-speed-start", type=float, default=CURRICULUM_START_SPEED)
    parser.add_argument("--pursuer-speed-end", type=float, default=CURRICULUM_END_SPEED)
    parser.add_argument("--danger-radius-start", type=float, default=CURRICULUM_START_RADIUS)
    parser.add_argument("--danger-radius-end", type=float, default=CURRICULUM_END_RADIUS)
    args = parser.parse_args()

    train(num_episodes=args.episodes,
          steps_per_update=args.steps_per_update,
          save_path=args.save_path,
          warmup_episodes=args.warmup_episodes,
          pursuer_speed_start=args.pursuer_speed_start,
          pursuer_speed_end=args.pursuer_speed_end,
          danger_radius_start=args.danger_radius_start,
          danger_radius_end=args.danger_radius_end,
          entropy_coef=args.entropy_coef)
