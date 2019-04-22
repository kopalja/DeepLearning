#!/usr/bin/env python3
import numpy as np
import tensorflow as tf
import tensorflow_hub as tfhub # Note: you need to install tensorflow_hub

from caltech42 import Caltech42

def image_processing(img: np.ndarray) -> np.ndarray:
        img = tf.image.decode_image(img, channels=3, dtype=tf.float32)
        img = tf.image.resize(img, size=(224,224)).numpy()
        return img


# The neural network model
class Network:
    def __init__(self, args):
        # TODO: You should define `self.model`. You should use the following layer:
        #   mobilenet = tfhub.KerasLayer("https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/2", output_shape=[1280])
        # The layer:
        # - if given `trainable=True/False` to KerasLayer constructor, the layer weights
        #   either are marked or not marked as updatable by an optimizer;
        # - however, batch normalization regime is set independently, by `training=True/False`
        #   passed during layer execution.
        #
        # Therefore, to not train the layer at all, you should use
        #   mobilenet = tfhub.KerasLayer("https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/2", output_shape=[1280], trainable=False)
        #   features = mobilenet(inputs, training=False)
        # On the other hand, to fully train it, you should use
        #   mobilenet = tfhub.KerasLayer("https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/2", output_shape=[1280], trainable=True)
        #   features = mobilenet(inputs)
        # where the `training` argument to `mobilenet` is passed automatically in that case.
        #
        # Note that a model with KerasLayer can currently be saved only using
        #   tf.keras.experimental.export_saved_model(model, path, serving_only=True/False)
        # where `serving_only` controls whether only prediction, or also training/evaluation
        # graphs are saved. To again load the model, use
        #   model = tf.keras.experimental.load_from_saved_model(path, {"KerasLayer": tfhub.KerasLayer})

        mobilenet_url = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/2"

        inp = tf.keras.layers.Input((224, 224, 3))
        mobilenet = tfhub.KerasLayer(mobilenet_url, output_shape=[1280], trainable=False)

        hidden = mobilenet(inp, training=False)
        hidden = tf.keras.layers.Dropout(0.5)(hidden)
        hidden = tf.keras.layers.BatchNormalization()(hidden)

        hidden = tf.keras.layers.Dense(960, activation=tf.nn.relu)(hidden)
        hidden = tf.keras.layers.Dropout(0.5)(hidden)

        out = tf.keras.layers.Dense(Caltech42.LABELS, activation=tf.nn.softmax)
        out = out(hidden)

        self.model = tf.keras.Model(inputs=inp, outputs=out)
        self.model.compile(
            loss=tf.losses.SparseCategoricalCrossentropy(),
            optimizer=tf.keras.optimizers.Adam(),
            metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy')]
        )

        self.tb_callback=tf.keras.callbacks.TensorBoard(args.logdir, update_freq=1000, profile_batch=1)
        self.tb_callback.on_train_end = lambda *_: None

    def train(self, caltech42, args):
        self.model.fit(
            x=caltech42.train.data["images"], y=caltech42.train.data["labels"],
            batch_size=args.batch_size, callbacks=[self.tb_callback],
            validation_data=(caltech42.dev.data["images"], caltech42.dev.data["labels"]),
            epochs=args.epochs)

    def predict(self, caltech42, args):
        return self.model(caltech42.data["images"]).numpy()



if __name__ == "__main__":
    import argparse
    import datetime
    import os
    import re

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=32, type=int, help="Batch size.")
    parser.add_argument("--epochs", default=100, type=int, help="Number of epochs.")
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
    caltech42 = Caltech42(image_processing=image_processing)

    # Create the network and train
    network = Network(args)
    network.train(caltech42, args)

    # Generate test set annotations, but in args.logdir to allow parallel execution.
    with open(os.path.join(args.logdir, "caltech42_competition_test.txt"), "w", encoding="utf-8") as out_file:
        for probs in network.predict(caltech42.test, args):
            print(np.argmax(probs), file=out_file)

    

