import gymnasium as gym
import numpy as np
import time
import matplotlib.pyplot as plt

env = gym.make("CartPole-v1")
state, _ = env.reset()

# Discretization
NUM_BINS = [24, 24, 24, 24]  # More bins for better accuracy
obs_space_high = env.observation_space.high#modifying permissible limits
obs_space_low = env.observation_space.low
obs_space_low[1] = -10#cart velocity
obs_space_high[1] = 10
obs_space_low[3] = -np.radians(70)#pole angular velocity
obs_space_high[3] = np.radians(70)
bin_width = (obs_space_high - obs_space_low) / NUM_BINS

def get_discrete_state(state):
    ratios = (state - obs_space_low) / bin_width
    discrete_state = np.clip(ratios.astype(int), 0, np.array(NUM_BINS) - 1)
    return tuple(discrete_state)

q_table = np.random.uniform(low=-1, high=1, size=(NUM_BINS + [env.action_space.n]))

# Hyperparameters
EPISODES = 15000
LEARNING_RATE = 0.1
DISCOUNT = 0.95
epsilon = 1
START_EPSILON_DECAY = 1
END_EPSILON_DECAY = EPISODES // 2
epsilon_decay_value = epsilon / (END_EPSILON_DECAY - START_EPSILON_DECAY)
MAX_STEPS = 200

rewards = []

for episode in range(EPISODES):
    state, _ = env.reset()
    discrete_state = get_discrete_state(state)
    done = False
    episode_reward = 0

    for step in range(MAX_STEPS):
        if np.random.random() > epsilon:
            action = np.argmax(q_table[discrete_state])
        else:
            action = np.random.randint(0, env.action_space.n)

        new_state, reward, terminated, truncated, _ = env.step(action)
        new_discrete_state = get_discrete_state(new_state)
        done = terminated or truncated
        episode_reward += reward


        if not done:
            max_future_q = np.max(q_table[new_discrete_state])
            current_q = q_table[discrete_state + (action,)]
            new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)
            q_table[discrete_state + (action,)] = new_q
        else:
            q_table[discrete_state + (action,)] = reward

        discrete_state = new_discrete_state

        if done:
            break

    rewards.append(episode_reward)
    if END_EPSILON_DECAY >= episode >= START_EPSILON_DECAY:
        epsilon -= epsilon_decay_value

    print(f"Episode:{episode} Reward:{episode_reward}")

plt.plot(range(len(rewards)), rewards)
plt.show()

env.close()

# Demo Run
print("Running trained model...")
env = gym.make("CartPole-v1", render_mode="human")
state, _ = env.reset()
discrete_state = get_discrete_state(state)
done = False
velocity=[]
while not done:
    action = np.argmax(q_table[discrete_state])
    new_state, reward, terminated, truncated, _ = env.step(action)
    discrete_state = get_discrete_state(new_state)
    done = terminated or truncated
    velocity.append(discrete_state[1])
    time.sleep(0.03)

velocity = np.array(velocity)
print(f"{velocity.max()} {velocity.min()}")#getting cart velocity limits in actual run
env.close()
