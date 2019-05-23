# dd7e3410-38c0-11e8-9b58-00505601122b
# 6e14ef6b-3281-11e8-9de3-00505601122b

#!/usr/bin/env python3
import numpy as np

import cart_pole_evaluator

if __name__ == "__main__":
    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", default=500, type=int, help="Training episodes.")
    parser.add_argument("--epsilon", default=0.05, type=float, help="Exploration factor.")
    parser.add_argument("--render_each", default=0, type=int, help="Render some episodes.")
    args = parser.parse_args()

    # Fix random seed
    np.random.seed(42)

    # Create the environment
    env = cart_pole_evaluator.environment(discrete=True)

    # Create Q, C and other variables
    # TODO:
    # - Create Q, a zero-filled NumPy array with shape [env.states, env.actions],
    #   representing estimated Q value of a given (state, action) pair.
    # - Create C, a zero-filled NumPy array with shape [env.states, env.actions],
    #   representing number of observed returns of a given (state, action) pair.
    Q, C = np.zeros([env.states, env.actions]), np.zeros([env.states, env.actions])

    for _ in range(args.episodes):
        # Perform episode
        state = env.reset()
        states, actions, rewards = [], [], []
        while True:
            if args.render_each and env.episode > 0 and env.episode % args.render_each == 0:
                env.render()

            # TODO: Compute `action` using epsilon-greedy policy. Therefore,
            # with probability of args.epsilon, use a random actions (there are env.actions of them),
            # otherwise, choose and action with maximum Q[state, action].
            action = np.argmax(Q[state])
            if args.epsilon > np.random.uniform():
                action = np.random.randint(low = 0, high = env.actions, size = 1)[0]


            next_state, reward, done, _ = env.step(action)

            states.append(state)
            actions.append(action)
            rewards.append(reward)

            state = next_state
            if done:
                break

        # TODO: Compute returns from the observed rewards.
        # TODO: Update Q and C
        states.reverse()
        actions.reverse()
        rewards.reverse()

        episode_len = len(rewards)
        g = 0
        lambd = 1
        for i in range(episode_len - 1):
            reward = rewards[i + 1]
            state = states[i]
            action = actions[i]
            g = lambd * g + reward
            C[state, action] += 1
            Q[state, action] += (1 / C[state, action]) * (g - Q[state, action])  


    # Final evaluation
    while True:
        state, done = env.reset(True), False
        while not done:
            action = np.argmax(Q[state])
            state, reward, done, _ = env.step(action)
