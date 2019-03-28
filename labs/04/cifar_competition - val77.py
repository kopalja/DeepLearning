# dd7e3410-38c0-11e8-9b58-00505601122b
# 6e14ef6b-3281-11e8-9de3-00505601122b

import numpy as np
import tensorflow as tf

from cifar10 import CIFAR10


# The neural network model
class Network(tf.keras.Sequential):
    def call(self, inputs, training=None, mask=None):
        pass

    def __init__(self, args):
        # TODO: Define a suitable model, by calling `super().__init__`
        # with appropriate inputs and outputs.
        #
        # Alternatively, if you prefer to use a `tf.keras.Sequential`,
        # replace the `Network` parent, call `super().__init__` at the beginning
        # of this constructor and add layers using `self.add`.
        super().__init__()
        self.add(tf.keras.layers.Input((CIFAR10.H, CIFAR10.W, CIFAR10.C)))

        self.add(tf.keras.layers.Conv2D(filters=64, kernel_size=(5, 5), activation=tf.nn.relu))
        self.add(tf.keras.layers.MaxPool2D())
        self.add(tf.keras.layers.BatchNormalization())

        self.add(tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), activation=tf.nn.relu))
        self.add(tf.keras.layers.BatchNormalization())

        self.add(tf.keras.layers.Conv2D(filters=192, kernel_size=(3, 3), activation=tf.nn.relu))
        self.add(tf.keras.layers.BatchNormalization())
        self.add(tf.keras.layers.MaxPool2D())

        self.add(tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), activation=tf.nn.relu))
        self.add(tf.keras.layers.BatchNormalization())

        self.add(tf.keras.layers.Conv2D(filters=384, kernel_size=(3, 3), activation=tf.nn.relu))
        self.add(tf.keras.layers.BatchNormalization())

        self.add(tf.keras.layers.Flatten())

        self.add(tf.keras.layers.Dense(400, activation=tf.nn.relu,
                                       activity_regularizer=tf.keras.regularizers.l2(0.001)))
        self.add(tf.keras.layers.Dropout(0.5))
        self.add(tf.keras.layers.BatchNormalization())

        self.add(tf.keras.layers.Dense(400, activation=tf.nn.relu,
                                       activity_regularizer=tf.keras.regularizers.l2(0.001)))
        self.add(tf.keras.layers.Dropout(0.5))

        self.add(tf.keras.layers.Dense(CIFAR10.LABELS, activation=tf.nn.softmax))

        # TODO: After creating the model, call `self.compile` with appropriate arguments.
        self.compile(
            optimizer=tf.keras.optimizers.Adam(),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")]
        )
        self.tb_callback = tf.keras.callbacks.TensorBoard(args.logdir, update_freq=1000, profile_batch=1)
        self.tb_callback.on_train_end = lambda *_: None

    def train(self, cifar, args):
        self.fit(
            x=cifar.train.data["images"], y=cifar.train.data["labels"],
            batch_size=args.batch_size, epochs=args.epochs,
            validation_data=(cifar.dev.data["images"], cifar.dev.data["labels"]),
            callbacks=[self.tb_callback],
        )


if __name__ == "__main__":
    import argparse
    import datetime
    import os
    import re

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=40, type=int, help="Batch size.")
    parser.add_argument("--epochs", default=75, type=int, help="Number of epochs.")
    parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
    args = parser.parse_args()

    # Fix random seeds
    np.random.seed(42)
    tf.random.set_seed(42)
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)

    # Create logdir name
    args.logdir = os.path.join("logs", "{}-{}-{}".format(
        os.path.basename(__file__),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", key), value) for key, value in sorted(vars(args).items())))
    ))

    # Load data
    cifar = CIFAR10()

    # Create the network and train
    network = Network(args)
    network.train(cifar, args)

    # Generate test set annotations, but in args.logdir to allow parallel execution.
    with open(os.path.join(args.logdir, "cifar_competition_test.txt"), "w", encoding="utf-8") as out_file:
        for probs in network.predict(cifar.test.data["images"], batch_size=args.batch_size):
            print(np.argmax(probs), file=out_file)
