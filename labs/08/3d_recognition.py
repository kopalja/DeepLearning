#!/usr/bin/env python3
import numpy as np
import tensorflow as tf

from modelnet import ModelNet
from tensorflow.keras.layers import Input, Conv3D, Activation, MaxPooling3D, BatchNormalization, Dense, GlobalAveragePooling3D, SpatialDropout3D


# The neural network model
class Network:
    def __init__(self, modelnet, args):
        # TODO: Define a suitable model, and either `.compile` it, or prepare
        # optimizer and loss manually.
        inp = Input(shape=(modelnet.H, modelnet.W, modelnet.D, modelnet.C))

        hidden = Conv3D(24, (3,3,3), activation=None, padding='same')(inp)
        hidden = SpatialDropout3D(0.4)(hidden)
        hidden = BatchNormalization()(hidden)
        hidden = Activation(tf.nn.relu)(hidden)

        hidden = Conv3D(24, (3,3,3), activation=None, padding='same')(hidden)
        hidden = SpatialDropout3D(0.4)(hidden)
        hidden = BatchNormalization()(hidden)
        hidden = Activation(tf.nn.relu)(hidden)

        hidden = MaxPooling3D((2,2,2))(hidden)

        hidden = Conv3D(48, (3,3,3), activation=None)(hidden)
        hidden = BatchNormalization()(hidden)
        hidden = Activation(tf.nn.relu)(hidden)

        hidden = Conv3D(48, (3,3,3), activation=None)(hidden)
        hidden = BatchNormalization()(hidden)
        hidden = Activation(tf.nn.relu)(hidden)

        hidden = MaxPooling3D((2,2,2))(hidden)

        hidden = Conv3D(96, (3,3,3), activation=None)(hidden)
        hidden = BatchNormalization()(hidden)
        hidden = Activation(tf.nn.relu)(hidden)

        hidden = Conv3D(96, (3,3,3), activation=None)(hidden)
        hidden = BatchNormalization()(hidden)
        hidden = Activation(tf.nn.relu)(hidden)

        hidden = MaxPooling3D((2,2,2))(hidden)

        hidden = GlobalAveragePooling3D()(hidden)

        output = Dense(len(modelnet.LABELS), activation=tf.nn.softmax)(hidden)

        self.model = tf.keras.Model(inputs=inp, outputs=output)
        self.model.compile(
            optimizer=tf.optimizers.Adam(),
            loss=tf.losses.SparseCategoricalCrossentropy(),
            metrics=[tf.metrics.SparseCategoricalAccuracy(name="accuracy")]
        )

        self.tb_callback=tf.keras.callbacks.TensorBoard(args.logdir, update_freq=1000, profile_batch=1)
        self.tb_callback.on_train_end = lambda *_: None

    def train(self, modelnet, args):
        # TODO: Train the network on a given dataset.
        self.model.fit(x=modelnet.train.data["voxels"], y=modelnet.train.data["labels"],
            batch_size=args.batch_size, epochs=args.epochs,
            validation_data=(modelnet.dev.data["voxels"], modelnet.dev.data["labels"]))

    def predict(self, dataset, args):
        # TODO: Predict method should return a list/np.ndarray of
        # label probabilities from the test set
        return self.model.predict(dataset.data["voxels"])


if __name__ == "__main__":
    import argparse
    import datetime
    import os
    import re

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=32, type=int, help="Batch size.")
    parser.add_argument("--modelnet", default=32, type=int, help="ModelNet dimension.")
    parser.add_argument("--epochs", default=50, type=int, help="Number of epochs.")
    parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
    args = parser.parse_args()

    # Fix random seeds and number of threads
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

    # Load the data
    modelnet = ModelNet(args.modelnet)

    # Create the network and train
    network = Network(modelnet, args)
    network.train(modelnet, args)

    # Generate test set annotations, but in args.logdir to allow parallel execution.
    out_path = "3d_recognition_test.txt"
    if os.path.isdir(args.logdir): out_path = os.path.join(args.logdir, out_path)
    with open(out_path, "w", encoding="utf-8") as out_file:
        for probs in network.predict(modelnet.test, args):
            print(np.argmax(probs), file=out_file)
