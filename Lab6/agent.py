# MODIFY THIS FILE.
# Complete every section marked with TODO.
# Do not change any function signatures.

import numpy as np
from gridworld import GridWorld


# You will tune these in Parts C and D via run.py arguments.
# Do not hardcode these values inside your functions.
ALPHA         = 0.1    # learning rate
GAMMA         = 0.9    # discount factor
EPSILON_START = 1.0    # initial exploration rate
EPSILON_END   = 0.01   # minimum exploration rate
EPSILON_DECAY = 0.995  # multiplicative decay per episode
EPISODES      = 600


def choose_action(state, Q, epsilon):
    """
    ε-greedy action selection.

    With probability epsilon, return a random action (explore).
    Otherwise return the action with the highest Q-value (exploit).

    Args:
        state   (int):        current state (0-15)
        Q       (np.ndarray): Q-table of shape (16, 4)
        epsilon (float):      exploration probability in [0, 1]

    Returns:
        action (int): chosen action — 0=UP  1=DOWN  2=LEFT  3=RIGHT

    Hints:
        np.random.random()                    → uniform float in [0, 1)
        np.argmax(Q[state])                   → index of max value
        np.random.randint(0, GridWorld.NACTIONS) → random action
    """
    # TODO
    # With probability epsilon: return a random action
    # Otherwise:                return argmax Q[state]
    if np.random.random() < epsilon:
        return np.random.randint(0, GridWorld.NACTIONS)
    else:
        return np.argmax(Q[state])


def update_Q_learning(Q, state, action, reward, next_state, alpha, gamma):
    """
    One Q-learning (off-policy TD) update. Modifies Q in place.

    Update rule:
        td_target        = reward + gamma * max_a' Q[next_state, a']
        td_error (delta) = td_target - Q[state, action]
        Q[state, action] += alpha * td_error

    Args:
        Q          (np.ndarray): Q-table, shape (16, 4) — update IN PLACE
        state      (int)
        action     (int)
        reward     (float)
        next_state (int)
        alpha      (float): learning rate
        gamma      (float): discount factor

    Hints:
        np.max(Q[next_state])  → best Q-value available from next_state
    """
    # TODO
    td_target = reward + gamma * np.max(Q[next_state])
    td_error  = td_target - Q[state, action]
    Q[state, action] += alpha * td_error


def update_SARSA(Q, state, action, reward, next_state, next_action, alpha, gamma):
    """
    One SARSA (on-policy TD) update. Modifies Q in place.

    Identical structure to Q-learning with ONE change:
    instead of max_a' Q[next_state], use Q[next_state, next_action]
    where next_action is the action the policy ACTUALLY chose next.

    Update rule:
        td_target        = reward + gamma * Q[next_state, next_action]
        td_error (delta) = td_target - Q[state, action]
        Q[state, action] += alpha * td_error

    Args:
        Q           (np.ndarray): Q-table, shape (16, 4) — update IN PLACE
        state       (int)
        action      (int)
        reward      (float)
        next_state  (int)
        next_action (int): the action the policy will ACTUALLY take next
        alpha       (float)
        gamma       (float)
    """
    # TODO
    td_target = reward + gamma * Q[next_state, next_action]
    td_error  = td_target - Q[state, action]
    Q[state, action] += alpha * td_error
    pass


def train(
    algorithm     = "qlearning",  # "qlearning" or "sarsa"
    episodes      = EPISODES,
    alpha         = ALPHA,
    gamma         = GAMMA,
    epsilon_start = EPSILON_START,
    epsilon_end   = EPSILON_END,
    epsilon_decay = EPSILON_DECAY,
):
    """
    Unified training loop for Q-learning and SARSA.

    The loop structure differs between the two algorithms:

        Q-learning (off-policy):
            loop:
                choose action  (ε-greedy)
                take action    (env.step)
                update Q       (uses max over next state)
                advance state

        SARSA (on-policy):
            choose FIRST action before the loop
            loop:
                take action    (env.step)
                choose NEXT action  ← must happen BEFORE the update
                update Q       (uses next_action Q-value)
                action = next_action
                advance state

    The SARSA structure is subtle: next_action must be chosen
    BEFORE calling update_SARSA, because the update needs it.
    Then it becomes the current action on the next iteration.

    When done=True (terminal state), set next_action = 0 as a
    placeholder — it won't be used because Q[terminal] = 0
    and the loop exits immediately after.

    Args:
        algorithm     (str):   "qlearning" or "sarsa"
        episodes      (int):   number of training episodes
        alpha         (float): learning rate
        gamma         (float): discount factor
        epsilon_start (float): initial epsilon
        epsilon_end   (float): minimum epsilon (floor for decay)
        epsilon_decay (float): multiply epsilon by this after each episode

    Returns:
        Q            (np.ndarray):  final Q-table, shape (16, 4)
        rewards_log  (list[float]): total reward collected per episode
        epsilon_log  (list[float]): epsilon value at the START of each episode
    """
    env         = GridWorld()
    Q           = np.zeros((GridWorld.NSTATES, GridWorld.NACTIONS))
    rewards_log = []
    epsilon_log = []
    epsilon     = epsilon_start

    for episode in range(episodes):
        state        = env.reset()
        total_reward = 0.0
        done         = False

        epsilon_log.append(epsilon)  # record epsilon at episode start

        # ── SARSA: choose the very first action before entering the loop ──
        # TODO: if algorithm == "sarsa", call choose_action here and store
        #       the result in `action`. For q-learning, set action = None.
        if algorithm == "sarsa":
            action = choose_action(state, Q, epsilon)
        else:
            action = None

        max_steps = 2
        steps = 0

        while not done and steps < max_steps:
            steps += 1

            if algorithm == "qlearning":
                # 1. Choose action using choose_action(state, Q, epsilon)
                action = choose_action(state, Q, epsilon)
                # 2. Take a step: next_state, reward, done = env.step(action)
                next_state, reward, done = env.step(action)
                # 3. Call update_Q_learning(...)
                update_Q_learning(Q, state, action, reward, next_state, alpha, gamma)
                # 4. state = next_state
                state = next_state
                # 5. total_reward += reward
                total_reward += reward

            elif algorithm == "sarsa":
                # 1. Take the current action: next_state, reward, done = env.step(action)
                next_state, reward, done = env.step(action)
                # 2. Choose next_action = choose_action(next_state, Q, epsilon)
                #    BUT if done is True, set next_action = 0 (placeholder)
                if done:
                    next_action = 0
                else:
                    next_action = choose_action(next_state, Q, epsilon)
                # 3. Call update_SARSA(..., next_action=next_action, ...)
                update_SARSA(Q, state, action, reward, next_state, next_action, alpha, gamma)
                # 4. action = next_action   (carry forward for next iteration)
                action = next_action
                # 5. state = next_state
                state = next_state
                # 6. total_reward += reward
                total_reward += reward

        rewards_log.append(total_reward)

        # TODO: apply epsilon decay after each episode
        epsilon = max(epsilon_end, epsilon * epsilon_decay)

    return Q, rewards_log, epsilon_log
