"""RL training for Cascadia using Gymnasium + Stable-Baselines3.

The Rust game engine runs as a subprocess in "gym server" mode.
The agent selects from ~4-16 candidate moves per turn.
Reward is the final game score (sparse, end of episode).

Usage:
    python3 train_rl.py --timesteps 500000
    python3 train_rl.py --timesteps 1000000 --n-envs 8
"""

import argparse
import json
import subprocess
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback
import time


RUST_CLI = "./target/release/cascadia-cli"
NUM_FEATURES = 7670
MAX_CANDIDATES = 20  # pad/truncate to fixed action space


class CascadiaEnv(gym.Env):
    """Gymnasium environment wrapping the Rust Cascadia game engine."""

    metadata = {"render_modes": []}

    def __init__(self, weights_path="nnue_weights_hybrid_iter4.bin"):
        super().__init__()
        self.weights_path = weights_path
        self.proc = None

        # Observation: board state features (single NNUE feature vector for current board)
        # Plus per-candidate NNUE value scores (MAX_CANDIDATES floats)
        # Total: NUM_FEATURES + MAX_CANDIDATES
        self.observation_space = spaces.Box(
            low=-100.0, high=200.0,
            shape=(NUM_FEATURES + MAX_CANDIDATES,),
            dtype=np.float32,
        )

        # Action: pick one of MAX_CANDIDATES candidates
        self.action_space = spaces.Discrete(MAX_CANDIDATES)

        self.n_candidates = 0
        self.current_score = 0
        self.candidate_features = None
        self._start_process()

    def _start_process(self):
        if self.proc:
            self.proc.kill()
        self.proc = subprocess.Popen(
            [RUST_CLI, "--gym", "--weights", self.weights_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )

    def _send(self, cmd):
        self.proc.stdin.write(cmd + "\n")
        self.proc.stdin.flush()
        line = self.proc.stdout.readline().strip()
        return json.loads(line)

    def _get_obs(self):
        """Get board features + candidate value scores as observation."""
        resp = self._send("obs")
        obs = np.zeros(NUM_FEATURES + MAX_CANDIDATES, dtype=np.float32)
        # Board features (sparse binary)
        for fi in resp.get("board_features", []):
            if fi < NUM_FEATURES:
                obs[fi] = 1.0
        # Candidate NNUE value scores (dense)
        for i, score in enumerate(resp.get("candidate_scores", [])[:MAX_CANDIDATES]):
            obs[NUM_FEATURES + i] = score
        return obs

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        try:
            resp = self._send("reset")
        except (BrokenPipeError, ValueError):
            self._start_process()
            resp = self._send("reset")

        self.n_candidates = resp["n_candidates"]
        self.current_score = resp["current_score"]

        if resp.get("done", False) or self.n_candidates == 0:
            # Game already over (rare edge case)
            obs = np.zeros((MAX_CANDIDATES, NUM_FEATURES), dtype=np.float32)
            return obs, {"n_candidates": 0, "score": self.current_score}

        obs = self._get_obs()
        return obs, {"n_candidates": self.n_candidates, "score": self.current_score}

    def step(self, action):
        # Clamp action to valid range
        action = min(action, self.n_candidates - 1) if self.n_candidates > 0 else 0

        resp = self._send(f"step {action}")

        done = resp["done"]
        reward = resp["reward"]  # per-step delta score (return-to-go)
        self.n_candidates = resp["n_candidates"]
        self.current_score = resp["current_score"]

        if done:
            obs_size = NUM_FEATURES + MAX_CANDIDATES
            obs = np.zeros(obs_size, dtype=np.float32)
            return obs, reward, True, False, {"score": self.current_score}
        else:
            obs = self._get_obs()
            return obs, reward, False, False, {
                "n_candidates": self.n_candidates,
                "score": self.current_score,
            }

    def close(self):
        if self.proc:
            try:
                self.proc.stdin.write("quit\n")
                self.proc.stdin.flush()
            except:
                pass
            self.proc.kill()
            self.proc = None


class ScoreCallback(BaseCallback):
    """Log episode scores."""
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_scores = []
        self.last_report = 0

    def _on_step(self):
        dones = self.locals.get("dones", [])
        infos = self.locals.get("infos", [])
        for done, info in zip(dones, infos):
            if done and "score" in info:
                if info["score"] > 0:
                    self.episode_scores.append(info["score"])

        # Report every 100 episodes
        if len(self.episode_scores) >= self.last_report + 100:
            recent = self.episode_scores[-100:]
            mean = np.mean(recent)
            p10 = np.percentile(recent, 10)
            p90 = np.percentile(recent, 90)
            print(f"  Episodes {len(self.episode_scores)}: mean={mean:.1f} P10={p10:.0f} P90={p90:.0f}")
            self.last_report = len(self.episode_scores)

        return True


def make_env(weights_path):
    def _init():
        return CascadiaEnv(weights_path=weights_path)
    return _init


def collect_warmstart_data(weights_path, n_episodes=500):
    """Run NNUE-greedy games to collect (obs, best_action) pairs.
    The 'best action' is argmax of the candidate score features in the obs."""
    print(f"Collecting warmstart data ({n_episodes} NNUE-greedy episodes)...")
    env = CascadiaEnv(weights_path=weights_path)
    obs_list = []
    action_list = []
    scores = []

    for ep in range(n_episodes):
        obs, info = env.reset()
        n_cands = info.get("n_candidates", 0)
        done = False
        while not done and n_cands > 0:
            # NNUE's choice = argmax over candidate score features
            cand_scores = obs[NUM_FEATURES:NUM_FEATURES + MAX_CANDIDATES]
            best_action = int(np.argmax(cand_scores[:n_cands]))
            obs_list.append(obs.copy())
            action_list.append(best_action)
            obs, reward, done, _, info = env.step(best_action)
            n_cands = info.get("n_candidates", 0)
        scores.append(info.get("score", 0))
        if (ep + 1) % 100 == 0:
            print(f"  {ep+1}/{n_episodes} episodes, mean score: {np.mean(scores):.1f}")

    env.close()
    print(f"  Collected {len(obs_list)} (obs, action) pairs from {n_episodes} games (avg {np.mean(scores):.1f})")
    return np.array(obs_list, dtype=np.float32), np.array(action_list, dtype=np.int64)


def pretrain_policy(model, obs_data, action_data, epochs=30, batch_size=256, lr=1e-3):
    """Supervised pretraining of the PPO policy network."""
    import torch
    print(f"Pretraining policy ({len(obs_data)} samples, {epochs} epochs, lr={lr})...")

    device = next(model.policy.parameters()).device
    obs_t = torch.from_numpy(obs_data).to(device)
    action_t = torch.from_numpy(action_data).to(device)

    # Train only policy params (not value head)
    policy_params = list(model.policy.mlp_extractor.policy_net.parameters()) + \
                    list(model.policy.action_net.parameters())
    optimizer = torch.optim.Adam(policy_params, lr=lr)

    n = len(obs_data)
    for epoch in range(epochs):
        perm = np.random.permutation(n)
        total_loss = 0.0
        total_correct = 0

        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            batch_obs = obs_t[idx]
            batch_act = action_t[idx]

            # Forward through PPO policy
            features = model.policy.extract_features(batch_obs, model.policy.pi_features_extractor)
            latent_pi = model.policy.mlp_extractor.forward_actor(features)
            logits = model.policy.action_net(latent_pi)

            loss = torch.nn.functional.cross_entropy(logits, batch_act)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(idx)
            total_correct += (logits.argmax(dim=1) == batch_act).sum().item()

        avg_loss = total_loss / n
        accuracy = total_correct / n * 100
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1}/{epochs}: loss={avg_loss:.4f} acc={accuracy:.1f}%")

    print(f"  Pretraining complete: final accuracy {accuracy:.1f}%")


def main():
    parser = argparse.ArgumentParser(description='RL training for Cascadia')
    parser.add_argument('--timesteps', type=int, default=500000)
    parser.add_argument('--n-envs', type=int, default=4, help='Parallel environments')
    parser.add_argument('--weights', default='nnue_weights_hybrid_iter4.bin')
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--save', default='rl_policy.zip')
    parser.add_argument('--warmstart', action='store_true', help='Pretrain policy from NNUE play')
    parser.add_argument('--warmstart-episodes', type=int, default=500)
    parser.add_argument('--warmstart-epochs', type=int, default=30)
    args = parser.parse_args()

    print(f"Cascadia RL Training (PPO)")
    print(f"  Timesteps: {args.timesteps:,}")
    print(f"  Envs: {args.n_envs}")
    print(f"  LR: {args.lr}")
    print(f"  Weights: {args.weights}")

    # Create vectorized environments
    if args.n_envs > 1:
        env = SubprocVecEnv([make_env(args.weights) for _ in range(args.n_envs)])
    else:
        env = CascadiaEnv(weights_path=args.weights)

    # PPO with MLP policy
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=args.lr,
        n_steps=512,
        batch_size=args.batch_size,
        n_epochs=4,
        gamma=1.0,  # no discounting — we want total game score
        verbose=1,
        policy_kwargs=dict(
            net_arch=dict(pi=[256, 128], vf=[256, 128]),
        ),
    )

    print(f"\nPolicy architecture: {model.policy}")
    print(f"Total parameters: {sum(p.numel() for p in model.policy.parameters()):,}")

    # Warmstart: pretrain policy on NNUE play before RL
    if args.warmstart:
        obs_data, action_data = collect_warmstart_data(args.weights, n_episodes=args.warmstart_episodes)
        pretrain_policy(model, obs_data, action_data, epochs=args.warmstart_epochs)

    callback = ScoreCallback()
    t0 = time.time()

    model.learn(
        total_timesteps=args.timesteps,
        callback=callback,
        progress_bar=False,
    )

    elapsed = time.time() - t0
    print(f"\nTraining complete in {elapsed:.0f}s")
    print(f"Total episodes: {len(callback.episode_scores)}")
    if callback.episode_scores:
        final_100 = callback.episode_scores[-100:]
        print(f"Final 100 episodes: mean={np.mean(final_100):.1f} P10={np.percentile(final_100, 10):.0f} P90={np.percentile(final_100, 90):.0f}")

    model.save(args.save)
    print(f"Saved policy to {args.save}")

    env.close()


if __name__ == "__main__":
    main()
