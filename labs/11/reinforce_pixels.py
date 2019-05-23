# dd7e3410-38c0-11e8-9b58-00505601122b
# 6e14ef6b-3281-11e8-9de3-00505601122b

#!/usr/bin/env python3
import numpy as np
import tensorflow as tf

import cart_pole_pixels_evaluator

class Network:
    def __init__(self, env, args):
        # TODO: Define suitable model, similarly to `reinforce` or `reinforce_with_baseline`.
        #
        # Use Adam optimizer with given `args.learning_rate`.
        self.model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(16, 5, strides = 2, input_shape = [80, 80, 3], activation = tf.nn.relu),
            tf.keras.layers.MaxPool2D((2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(50, activation = tf.nn.relu),
            tf.keras.layers.Dense(env.actions, activation = tf.nn.softmax)
        ])

        self.model.compile(
            optimizer = tf.optimizers.Adam(args.learning_rate), 
            loss      = 'sparse_categorical_crossentropy'
            )

        self.base_line = tf.keras.Sequential([
            tf.keras.layers.Conv2D(16, 5, strides = 2, input_shape = [80, 80, 3], activation = tf.nn.relu),
            tf.keras.layers.MaxPool2D((2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(50, activation = tf.nn.relu),
            tf.keras.layers.Dense(1)
        ])

        self.base_line.compile(
            optimizer = tf.optimizers.Adam(args.learning_rate), 
            loss      = 'mean_squared_error'
            )





    def train(self, states, actions, returns):
        states, actions, returns = np.array(states), np.array(actions), np.array(returns)
        # TODO: Train the model using the states, actions and observed returns.
        predicted_baseline = self.base_line.predict_on_batch(states)
        predicted_baseline = np.reshape(predicted_baseline, returns.shape)
        self.model.train_on_batch(states, actions, sample_weight = returns - predicted_baseline)
        self.base_line.train_on_batch(states, returns)


    def predict(self, states):
        states = np.array(states)
        # TODO: Predict distribution over actions for the given input states
        return self.model.predict_on_batch(states)


if __name__ == "__main__":
    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=2, type=int, help="Number of episodes to train on.")
    parser.add_argument("--episodes", default=250, type=int, help="Training episodes.")
    parser.add_argument("--learning_rate", default=0.001, type=float, help="Learning rate.")
    parser.add_argument("--render_each", default=0, type=int, help="Render some episodes.")
    parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
    args = parser.parse_args()

    # Fix random seed
    np.random.seed(42)
    tf.random.set_seed(42)
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)

    # Create the environment
    env = cart_pole_pixels_evaluator.environment()

    # Construct the network
    network = Network(env, args)

    # Training
    for _ in range(args.episodes // args.batch_size):
        batch_states, batch_actions, batch_returns = [], [], []
        for _ in range(args.batch_size):
            # Perform episode
            states, actions, rewards = [], [], []
            state, done = env.reset(), False
            while not done:
                if args.render_each and env.episode > 0 and env.episode % args.render_each == 0:
                    env.render()

                probabilities = network.predict([state])[0]
                # TODO(reinforce): Compute `action` according to the distribution returned by the network.
                # The `np.random.choice` method comes handy.
                action = np.random.choice(len(probabilities), 1, p = probabilities)[0]

                next_state, reward, done, _ = env.step(action)

                states.append(state)
                actions.append(action)
                rewards.append(reward)

                state = next_state

            # TODO(reinforce): Compute `returns` from the observed `rewards`.
            returns = []
            g = 0
            for i in range(len(rewards)):
                returns.append(sum(rewards[i:]))

            batch_states += states
            batch_actions += actions
            batch_returns += returns

        network.train(batch_states, batch_actions, batch_returns)

    # Final evaluation
    while True:
        state, done = env.reset(True), False
        while not done:
            probabilities = network.predict([state])[0]
            action = np.argmax(probabilities)
            state, reward, done, _ = env.step(action)