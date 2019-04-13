#!/usr/bin/env python3
# dd7e3410-38c0-11e8-9b58-00505601122b
# 6e14ef6b-3281-11e8-9de3-00505601122b
import numpy as np
import tensorflow as tf

from mnist import MNIST


# The neural network model
class Network:
    def __init__(self, args):
        # TODO: Add a `self.model` which has two inputs, both images of size [MNIST.H, MNIST.W, MNIST.C].
        # It then passes both through the same network (with shared weights), performing
        # - convolution with 10 filters, 3x3 kernel size, stride 2, "valid" padding, ReLU activation
        # - convolution with 20 filters, 3x3 kernel size, stride 2, "valid" padding, ReLU activation
        # - flattening layer
        # - fully connected layer with 200 neurons and ReLU activation
        #
        # Then, it produces three outputs:
        # - classify the computed representation of the first image using a densely connected layer
        #   into 10 classes
        # - classify the computed representation of the second image using the
        #   same connected layer (with shared weights) into 10 classes
        # - concatenate the two image representations, process them using another fully connected
        #   layer with 200 neurons and ReLU, and finally compute one output with tf.nn.sigmoid
        #   activation; the goal is to predict if the first digit is larger than the second
        #
        # Train the outputs using SparseCategoricalCrossentropy for the first two inputs
        # and BinaryCrossentropy for the third one, utilizing Adam with default arguments.
        inputs1 = tf.keras.layers.Input((MNIST.H, MNIST.W, MNIST.C))
        inputs2 = tf.keras.layers.Input((MNIST.H, MNIST.W, MNIST.C))

        hidden = tf.keras.layers.Conv2D(filters=10, kernel_size=(3, 3), strides=2, activation=tf.nn.relu, padding='valid')
        hidden1 = hidden(inputs1)
        hidden2 = hidden(inputs2)

        hidden = tf.keras.layers.Conv2D(filters=20, kernel_size=(3, 3), strides=2, activation=tf.nn.relu, padding='valid')
        hidden1 = hidden(hidden1)
        hidden2 = hidden(hidden2)
        
        hidden = tf.keras.layers.Flatten()
        hidden1 = hidden(hidden1)
        hidden2 = hidden(hidden2)

        hidden = tf.keras.layers.Dense(200, activation=tf.nn.relu)
        hidden1 = hidden(hidden1)
        hidden2 = hidden(hidden2)
        
        classification = tf.keras.layers.Dense(MNIST.LABELS, activation=tf.nn.softmax)
        classification1 = classification(hidden1)
        classification2 = classification(hidden2)

        concat = tf.keras.layers.Concatenate()([hidden1, hidden2])
        hidden = tf.keras.layers.Dense(200, activation=tf.nn.relu)(concat)
        out = tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)(hidden)

        self.model = tf.keras.Model(inputs=[inputs1, inputs2], outputs=[classification1, classification2, out])
        self.model.compile(
            loss = [tf.keras.losses.SparseCategoricalCrossentropy(),
                    tf.keras.losses.SparseCategoricalCrossentropy(),
                    tf.keras.losses.BinaryCrossentropy()],
            optimizer = tf.optimizers.Adam(),
        )

    @staticmethod
    def _prepare_batches(batches_generator):
        batches = []
        for batch in batches_generator:
            batches.append(batch)
            if len(batches) >= 2:
                # TODO: yield the suitable modified inputs and targets using batches[0:2]
                model_inputs = (batches[0]["images"], batches[1]["images"])
                model_targets = (batches[0]["labels"],
                    batches[1]["labels"],
                    batches[0]["labels"] > batches[1]["labels"])
                yield (model_inputs, model_targets)
                batches.clear()

    def train(self, mnist, args):
        for epoch in range(args.epochs):
            # TODO: Train for one epoch using `model.train_on_batches`.
            for inputs, targets in self._prepare_batches(mnist.train.batches(args.batch_size)):
                self.model.train_on_batch(x=inputs, y=targets)

            # Print development evaluation
            print("Dev {}: directly predicting: {:.4f}, comparing digits: {:.4f}".format(epoch + 1, *self.evaluate(mnist.dev, args)))

    def eval_indirect(self, out, targets):
        num1, num2 = out[0].argmax(axis=1), out[1].argmax(axis=1)
        return np.sum((num1 > num2) == targets[2])

    def eval_direct(self, out, targets):
        return np.sum((out[2].T > 0.5) == targets[2])

    def evaluate(self, dataset, args):
        # TODO: Evaluate the given dataset, returning two accuracies, the first being
        # the direct prediction of the model, and the second computed by comparing predicted
        # labels of the images.
        direct_right, indirect_right, n = 0, 0, 0
        for inputs, targets in self._prepare_batches(dataset.batches(args.batch_size)):
            out = self.model.predict_on_batch(inputs)

            direct_right += self.eval_direct(out, targets)
            indirect_right += self.eval_indirect(out, targets)
            n += targets[2].shape[0]

        direct_accuracy = direct_right / n
        indirect_accuracy = indirect_right / n  
        return direct_accuracy, indirect_accuracy

if __name__ == "__main__":
    import argparse
    import datetime
    import os
    import re

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=50, type=int, help="Batch size.")
    parser.add_argument("--epochs", default=10, type=int, help="Number of epochs.")
    parser.add_argument("--recodex", default=False, action="store_true", help="Evaluation in ReCodEx.")
    parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
    args = parser.parse_args()

    # Fix random seeds and number of threads
    np.random.seed(42)
    tf.random.set_seed(42)
    if args.recodex:
        tf.keras.utils.get_custom_objects()["glorot_uniform"] = lambda: tf.keras.initializers.glorot_uniform(seed=42)
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)

    # Create logdir name
    args.logdir = os.path.join("logs", "{}-{}-{}".format(
        os.path.basename(__file__),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", key), value) for key, value in sorted(vars(args).items())))
    ))

    # Load the data
    mnist = MNIST()

    # Create the network and train
    network = Network(args)
    network.train(mnist, args)
    with open("mnist_multiple.out", "w") as out_file:
        direct, indirect = network.evaluate(mnist.test, args)
        print("{:.2f} {:.2f}".format(100 * direct, 100 * indirect), file=out_file)
