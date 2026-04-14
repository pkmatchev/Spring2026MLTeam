import random
import math
from collections import deque, namedtuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
 
SEED            = 42
NUM_EPISODES    = 600           # max num training episodes
MAX_STEPS       = 500           # CartPole-v1 time limit
BATCH_SIZE      = 128
GAMMA           = 0.99          # discount factor
LR              = 1e-3          # Adam learning rate
MEMORY_CAPACITY = 10_000        # replay buffer size
TARGET_UPDATE   = 10            # sync target net every N episodes
EPS_START       = 1.0           # epsilon initial value
EPS_END         = 0.01          # epsilon floor
EPS_DECAY       = 500           # epsilon decay speed (steps)
SOLVE_SCORE     = 475           # average score over 100 episodes to declare solved
 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 

Transition = namedtuple("Transition",
                        ("state", "action", "next_state", "reward", "done"))
 
 
class ReplayMemory:
    def __init__(self, capacity: int):
        self.memory = deque(maxlen=capacity)
 
    def push(self, *args):
        self.memory.append(Transition(*args))
 
    def sample(self, batch_size: int):
        return random.sample(self.memory, batch_size)
 
    def __len__(self):
        return len(self.memory)
 
 
class DQN(nn.Module):
    def __init__(self, obs_dim: int = 4, act_dim: int = 2, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, act_dim),
        )
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
 
 
class DQNAgent:
    def __init__(self, obs_dim: int, act_dim: int):
        self.act_dim  = act_dim
        self.steps    = 0                          
 
        self.policy_net = DQN(obs_dim, act_dim).to(DEVICE)
        self.target_net = DQN(obs_dim, act_dim).to(DEVICE)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
 
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LR)
        self.memory    = ReplayMemory(MEMORY_CAPACITY)
        self.loss_fn   = nn.SmoothL1Loss()

    def select_action(self, state: np.ndarray) -> int:
        eps = EPS_END + (EPS_START - EPS_END) * math.exp(-self.steps / EPS_DECAY)
        self.steps += 1
 
        if random.random() < eps:
            return random.randrange(self.act_dim) 
 
        with torch.no_grad():
            t = torch.tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            return self.policy_net(t).argmax(dim=1).item() 

    def remember(self, state, action, next_state, reward, done):
        self.memory.push(
            torch.tensor(state,      dtype=torch.float32),
            torch.tensor([action],   dtype=torch.long),
            torch.tensor(next_state, dtype=torch.float32),
            torch.tensor([reward],   dtype=torch.float32),
            torch.tensor([done],     dtype=torch.bool),
        )

    def learn(self) -> float | None:
        if len(self.memory) < BATCH_SIZE:
            return None
 
        batch      = self.memory.sample(BATCH_SIZE)
        batch      = Transition(*zip(*batch))
 
        states      = torch.stack(batch.state).to(DEVICE)
        actions     = torch.stack(batch.action).to(DEVICE)
        rewards     = torch.stack(batch.reward).to(DEVICE)
        next_states = torch.stack(batch.next_state).to(DEVICE)
        dones       = torch.stack(batch.done).to(DEVICE)
 
        q_values = self.policy_net(states).gather(1, actions)
 
        with torch.no_grad():
            next_q = self.target_net(next_states).max(1, keepdim=True).values
            next_q[dones] = 0.0
 
        targets = rewards + GAMMA * next_q
 
        loss = self.loss_fn(q_values, targets)
        self.optimizer.zero_grad()
        loss.backward()

        nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=10)
        self.optimizer.step()
 
        return loss.item()

    def update_target(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())
 
    def save(self, path: str = "cartpole_dqn.pth"):
        torch.save(self.policy_net.state_dict(), path)
        print(f"  Model saved → {path}")
 
    def load(self, path: str = "cartpole_dqn.pth"):
        self.policy_net.load_state_dict(torch.load(path, map_location=DEVICE))
        self.update_target()
        print(f"  Model loaded ← {path}")
 
 
def train():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
 
    env   = gym.make("CartPole-v1")
    agent = DQNAgent(obs_dim=4, act_dim=2)
 
    scores        = []
    recent_scores = deque(maxlen=100)   
 
    print(f"Training on {DEVICE}\n{'─'*50}")
 
    for episode in range(1, NUM_EPISODES + 1):
        state, _ = env.reset(seed=SEED + episode)
        total_reward = 0.0
 
        for _ in range(MAX_STEPS):
            action              = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done                = terminated or truncated
 
            agent.remember(state, action, next_state, reward, done)
            agent.learn()
 
            state        = next_state
            total_reward += reward
 
            if done:
                break
 
        scores.append(total_reward)
        recent_scores.append(total_reward)
        avg = np.mean(recent_scores)
 
        if episode % TARGET_UPDATE == 0:
            agent.update_target()
 
        if episode % 20 == 0 or avg >= SOLVE_SCORE:
            eps = EPS_END + (EPS_START - EPS_END) * math.exp(-agent.steps / EPS_DECAY)
            print(
                f"Ep {episode:>4}  "
                f"Score {total_reward:>6.1f}  "
                f"Avg-100 {avg:>6.1f}  "
                f"ε {eps:.3f}"
            )
 
        if avg >= SOLVE_SCORE:
            print(f"\n✓ Solved in {episode} episodes (avg {avg:.1f} ≥ {SOLVE_SCORE})")
            agent.save()
            break
 
    env.close()
    return agent, scores
 
 
def evaluate(agent: DQNAgent, n_episodes: int = 10, render: bool = False):
    mode = "human" if render else "rgb_array"
    env  = gym.make("CartPole-v1", render_mode=mode)
 
    agent.policy_net.eval()
    scores = []
 
    print(f"\n{'─'*50}\nEvaluating over {n_episodes} episodes …")
    for ep in range(1, n_episodes + 1):
        state, _ = env.reset()
        total    = 0.0
 
        for _ in range(MAX_STEPS):
            with torch.no_grad():
                t      = torch.tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)
                action = agent.policy_net(t).argmax(dim=1).item()
 
            state, reward, terminated, truncated, _ = env.step(action)
            total += reward
            if terminated or truncated:
                break
 
        scores.append(total)
        print(f"  Episode {ep:>2}: {total:.0f}")
 
    env.close()
    print(f"\nMean score: {np.mean(scores):.1f}  |  Min: {np.min(scores):.0f}  |  Max: {np.max(scores):.0f}")
 
 
def plot_scores(scores: list[float]):
    try:
        import matplotlib.pyplot as plt
 
        window = 100
        rolling = [
            np.mean(scores[max(0, i - window): i + 1])
            for i in range(len(scores))
        ]
 
        plt.figure(figsize=(10, 4))
        plt.plot(scores,  alpha=0.4, label="Episode score")
        plt.plot(rolling, linewidth=2, label=f"{window}-ep rolling avg")
        plt.axhline(SOLVE_SCORE, color="red", linestyle="--", label=f"Solve threshold ({SOLVE_SCORE})")
        plt.xlabel("Episode")
        plt.ylabel("Score")
        plt.title("CartPole-v1 — DQN Training")
        plt.legend()
        plt.tight_layout()
        plt.savefig("cartpole_training.png", dpi=150)
        plt.show()
        print("Plot saved → cartpole_training.png")
    except ImportError:
        print("matplotlib not installed — skipping plot.")
 
 
if __name__ == "__main__":
    agent, scores = train()
    evaluate(agent, n_episodes=10, render=False)
    plot_scores(scores)
 