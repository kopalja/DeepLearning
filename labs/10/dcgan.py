# dd7e3410-38c0-11e8-9b58-00505601122b
# 6e14ef6b-3281-11e8-9de3-00505601122b

import numpy as np
import tensorflow as tf

from mnist import MNIST

# The neural network model
class Network:
    def __init__(self, args):
        self._z_dim = args.z_dim

        # TODO: Define `self.generator` as a Model, which
        # - takes vectors of [args.z_dim] shape on input
        # - applies batch normalized dense layer with 1024 units and ReLU (do not forget about `use_bias=False` anywhere in the model)
        # - applies batch normalized dense layer with MNIST.H // 4 * MNIST.W // 4 * 64 units and ReLU
        # - reshapes the current hidder output to [MNIST.H // 4, MNIST.W // 4, 64]
        # - applies batch normalized transposed convolution with 32 filters, kernel size 5,
        #   stride 2, same padding, and ReLU activation
        # - applies transposed convolution with 1 filters, kernel size 5,
        #   stride 2, same padding, and sigmoid activation
        gen_in = tf.keras.layers.Input([args.z_dim])

        hidden = tf.keras.layers.Dense(1024, use_bias=False)(gen_in)
        hidden = tf.keras.layers.BatchNormalization()(hidden)
        hidden = tf.keras.layers.Activation(tf.nn.relu)(hidden)

        hidden = tf.keras.layers.Dense(MNIST.H // 4 * MNIST.W // 4 * 64, use_bias=False)(hidden)
        hidden = tf.keras.layers.BatchNormalization()(hidden)
        hidden = tf.keras.layers.Activation(tf.nn.relu)(hidden)

        hidden = tf.keras.layers.Reshape([MNIST.H // 4, MNIST.W // 4, 64])(hidden)

        hidden = tf.keras.layers.Conv2DTranspose(kernel_size=5, filters=32, strides=2, padding='same', use_bias=False)(hidden)
        hidden = tf.keras.layers.BatchNormalization()(hidden)
        hidden = tf.keras.layers.Activation(tf.nn.relu)(hidden)

        gen_out = tf.keras.layers.Conv2DTranspose(kernel_size=5, filters=1, strides=2, padding='same', activation=tf.nn.sigmoid)(hidden)

        self.generator = tf.keras.Model(inputs=gen_in, outputs=gen_out)

        # TODO: Define `self.discriminator` as a Model, which
        # - takes input images with shape [MNIST.H, MNIST.W, MNIST.C]
        # - computes batch normalized convolution with 32 filters, kernel size 5,
        #   same padding, and ReLU activation (do not forget `use_bias` anywhere in the model)
        # - max-pools with kernel size 2 and stride 2
        # - computes batch normalized convolution with 64 filters, kernel size 5,
        #   same padding, and ReLU activation (do not forget `use_bias` anywhere in the model)
        # - max-pools with kernel size 2 and stride 2
        # - flattens the current representation
        # - applies batch normalized dense layer with 1024 uints and ReLU activation
        # - applies output dense layer with one output and a suitable activation function

        disc_in = tf.keras.layers.Input([MNIST.H, MNIST.W, MNIST.C])

        hidden = tf.keras.layers.Conv2D(filters=32, kernel_size=5, padding='same', use_bias=False)(disc_in)
        hidden = tf.keras.layers.BatchNormalization()(hidden)
        hidden = tf.keras.layers.Activation(tf.nn.relu)(hidden)

        hidden = tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=(2,2))(hidden)

        hidden = tf.keras.layers.Conv2D(filters=64, kernel_size=5, padding='same', use_bias=False)(hidden)
        hidden = tf.keras.layers.BatchNormalization()(hidden)
        hidden = tf.keras.layers.Activation(tf.nn.relu)(hidden)

        hidden = tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=(2,2))(hidden)

        hidden = tf.keras.layers.Flatten()(hidden)

        hidden = tf.keras.layers.Dense(1024, use_bias=False)(hidden)
        hidden = tf.keras.layers.BatchNormalization()(hidden)
        hidden = tf.keras.layers.Activation(tf.nn.relu)(hidden)

        disc_out = tf.keras.layers.Dense(1, tf.nn.sigmoid)(hidden)

        self.discriminator = tf.keras.Model(inputs=disc_in, outputs=disc_out)

        self._generator_optimizer, self._discriminator_optimizer = tf.optimizers.Adam(), tf.optimizers.Adam()
        self._loss_fn = tf.losses.BinaryCrossentropy()
        self._discriminator_accuracy = tf.metrics.Mean()
        self._writer = tf.summary.create_file_writer(args.logdir, flush_millis=10 * 1000)

    def _sample_z(self, batch_size):
        """Sample random latent variable."""
        return tf.random.uniform([batch_size, self._z_dim], -1, 1, seed=42)

    @tf.function
    def train_batch(self, images):
        # TODO(gan): Generator training. Using a Gradient tape:
        # - generate random images using a `generator`; do not forget about `training=True`
        # - run discriminator on the generated images, also using `training=True` (even if
        #   not updating discriminator parameters, we want to perform possible BatchNorm in it)
        # - compute loss using `_loss_fn`, with target labels `tf.ones_like(discriminator_output)`
        # Then, compute the gradients with respect to generator trainable variables and update
        # generator trainable weights using self._generator_optimizer.
        with tf.GradientTape() as tape:
            random_images = self.generator(self._sample_z(tf.shape(images)[0]), training=True)
            discriminator_output = self.discriminator(random_images, training=True)
            generator_loss = self._loss_fn(tf.ones_like(discriminator_output), discriminator_output)
        gradients = tape.gradient(generator_loss, self.generator.trainable_variables)
        self._generator_optimizer.apply_gradients(zip(gradients, self.generator.trainable_variables))

        # TODO(gan): Discriminator training. Using a Gradient tape:
        # - discriminate `images`, storing results in `discriminated_real`
        # - discriminate images generated in generator training, storing results in `discriminated_fake`
        # - compute loss by summing
        #   - `_loss_fn` on discriminated_real with suitable target labes
        #   - `_loss_fn` on discriminated_fake with suitable targets (`tf.{ones,zeros}_like` come handy).
        # Then, compute the gradients with respect to discriminator trainable variables and update
        # discriminator trainable weights using self._discriminator_optimizer.
        with tf.GradientTape() as tape:
            discriminated_real = self.discriminator(images, training=True)
            discriminated_fake = self.discriminator(random_images, training=True)
            discriminator_loss = self._loss_fn(tf.ones_like(discriminated_real), discriminated_real) + self._loss_fn(tf.zeros_like(discriminated_fake), discriminated_fake)
        gradients = tape.gradient(discriminator_loss, self.discriminator.trainable_variables)
        self._discriminator_optimizer.apply_gradients(zip(gradients, self.discriminator.trainable_variables))

        self._discriminator_accuracy(tf.greater(discriminated_real, 0.5))
        self._discriminator_accuracy(tf.less(discriminated_fake, 0.5))
        tf.summary.experimental.set_step(self._discriminator_optimizer.iterations)
        with self._writer.as_default():
            tf.summary.scalar("gan/generator_loss", generator_loss)
            tf.summary.scalar("gan/discriminator_loss", discriminator_loss)
            tf.summary.scalar("gan/discriminator_accuracy", self._discriminator_accuracy.result())

        return generator_loss + discriminator_loss

    def generate(self):
        GRID = 20

        # Generate GRIDxGRID images
        random_images = self.generator(self._sample_z(GRID * GRID))

        starts, ends = self._sample_z(GRID), self._sample_z(GRID)
        interpolated_z = tf.concat(
            [starts[i] + (ends[i] - starts[i]) * tf.expand_dims(tf.linspace(0., 1., GRID), -1) for i in range(GRID)], axis=0)
        interpolated_images = self.generator(interpolated_z)

        # Stack the random images, then an empty row, and finally interpolated imates
        image = tf.concat(
            [tf.concat(list(images), axis=1) for images in tf.split(random_images, GRID)] +
            [tf.zeros([MNIST.H, MNIST.W * GRID, MNIST.C])] +
            [tf.concat(list(images), axis=1) for images in tf.split(interpolated_images, GRID)], axis=0)
        with self._writer.as_default():
            tf.summary.image("gan/images", tf.expand_dims(image, 0))

    def train_epoch(self, dataset, args):
        self._discriminator_accuracy.reset_states()
        loss = 0
        for batch in dataset.batches(args.batch_size):
            loss += self.train_batch(batch["images"])
        self.generate()
        return loss


if __name__ == "__main__":
    import argparse
    import datetime
    import os
    import re

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=50, type=int, help="Batch size.")
    parser.add_argument("--dataset", default="mnist", type=str, help="MNIST-like dataset to use.")
    parser.add_argument("--discriminator_layers", default="128", type=str, help="Discriminator layers.")
    parser.add_argument("--epochs", default=100, type=int, help="Number of epochs.")
    parser.add_argument("--generator_layers", default="128", type=str, help="Generator layers.")
    parser.add_argument("--recodex", default=False, action="store_true", help="Evaluation in ReCodEx.")
    parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
    parser.add_argument("--z_dim", default=100, type=int, help="Dimension of Z.")
    args = parser.parse_args()
    args.discriminator_layers = [int(discriminator_layer) for discriminator_layer in args.discriminator_layers.split(",")]
    args.generator_layers = [int(generator_layer) for generator_layer in args.generator_layers.split(",")]

    # Fix random seeds
    np.random.seed(42)
    tf.random.set_seed(42)
    if args.recodex:
        tf.keras.utils.get_custom_objects()["glorot_uniform"] = lambda: tf.initializers.glorot_uniform(seed=42)
        tf.keras.utils.get_custom_objects()["orthogonal"] = lambda: tf.initializers.orthogonal(seed=42)
        tf.keras.utils.get_custom_objects()["uniform"] = lambda: tf.initializers.RandomUniform(seed=42)
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)

    # Create logdir name
    args.logdir = os.path.join("logs", "{}-{}-{}".format(
        os.path.basename(__file__),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", key), value) for key, value in sorted(vars(args).items())))
    ))

    # Load data
    mnist = MNIST(args.dataset)

    # Create the network and train
    network = Network(args)
    for epoch in range(args.epochs):
        loss = network.train_epoch(mnist.train, args)

    with open("gan.out", "w") as out_file:
        print("{:.2f}".format(loss), file=out_file)
