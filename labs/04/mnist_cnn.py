# dd7e3410-38c0-11e8-9b58-00505601122b
# 6e14ef6b-3281-11e8-9de3-00505601122b


#!/usr/bin/env python3
import numpy as np
import tensorflow as tf

from mnist import MNIST

# The neural network model
class Network(tf.keras.Model):
    def __init__(self, args):
        inputs = tf.keras.layers.Input(shape=[MNIST.H, MNIST.W, MNIST.C])

        # TODO: Add CNN layers specified by `args.cnn`, which contains
        # comma-separated list of the following layers:
        # - `C-filters-kernel_size-stride-padding`: Add a convolutional layer with ReLU
        #   activation and specified number of filters, kernel size, stride and padding.
        # - `CB-filters-kernel_size-stride-padding`: Same as `C`, but use batch normalization.
        #   In detail, start with a convolutional layer without bias and activation,
        #   then add batch normalization layer, and finally ReLU activation.
        # - `M-kernel_size-stride`: Add max pooling with specified size and stride.
        # - `R-[layers]`: Add a residual connection. The `layers` contain a specification
        #   of at least one convolutional layer (but not a recursive residual connection `R`).
        #   The input to the specified layers is then added to their output.
        # - `F`: Flatten inputs. Must appear exactly once in the architecture.
        # - `D-hidden_layer_size`: Add a dense layer with ReLU activation and specified size.
        # Produce the results in variable `hidden`.

        def feed_forward(layer, hidden):
            if layer[0] == 'CB':
                hidden = tf.keras.layers.Conv2D(layer[1], layer[2], strides = layer[3], padding = layer[4], use_bias = False)(hidden)
                hidden = tf.keras.layers.BatchNormalization()(hidden)
                return tf.keras.layers.ReLU()(hidden)
            elif layer[0] == 'C':
                return tf.keras.layers.Conv2D(layer[1], layer[2], strides = layer[3], padding = layer[4], activation = tf.nn.relu)(hidden)
            elif layer[0] == 'M':
                return tf.keras.layers.MaxPool2D(*layer[1:])(hidden)
            elif layer[0] == 'R':
                inpt = hidden
                for l in layer[1:]:
                    hidden = feed_forward(l, hidden)
                return hidden + inpt
            elif layer[0] == 'F':
                return tf.keras.layers.Flatten()(hidden)
            elif layer[0] == 'D':
                return tf.keras.layers.Dense(layer[1], activation = tf.nn.relu)(hidden)    
                    

        hidden = inputs
        if args.cnn != None:
            # preprocess args.cnn argument
            in_residual = False
            dsc = list(args.cnn)
            for i in range(len(dsc)):
                if dsc[i] == '[':
                    in_residual = True
                elif dsc[i] == ']':
                    in_residual = False
                elif dsc[i] == ',' and in_residual:
                    dsc[i] = ';'
            changed_argument = "".join(dsc)

            print(changed_argument)

            # parse args.cnn armument into parsed_layers 
            parsed_layers = []
            for layer in changed_argument.split(','):
                if layer[0] == 'R':
                    layers = layer[3:][:-1]
                    l = ['R']
                    for layer in layers.split(';'):
                        l.append([int(p) if p.isdigit() else p for p in layer.split('-')])
                    parsed_layers.append(l)
                else:      
                    parsed_layers.append([int(p) if p.isdigit() else p for p in layer.split('-')])

            print(parsed_layers)
            for layer in parsed_layers:
                hidden = feed_forward(layer, hidden)

        # Add the final output layer
        outputs = tf.keras.layers.Dense(MNIST.LABELS, activation=tf.nn.softmax)(hidden)

        super().__init__(inputs=inputs, outputs=outputs)

        self.compile(
            optimizer=tf.keras.optimizers.Adam(),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
        )

        self.tb_callback=tf.keras.callbacks.TensorBoard(args.logdir, update_freq=1000, profile_batch=1)
        self.tb_callback.on_train_end = lambda *_: None

    def train(self, mnist, args):
        self.fit(
            mnist.train.data["images"], mnist.train.data["labels"],
            batch_size=args.batch_size, epochs=args.epochs,
            validation_data=(mnist.dev.data["images"], mnist.dev.data["labels"]),
            callbacks=[self.tb_callback],
        )

    def test(self, mnist, args):
        test_logs = self.evaluate(mnist.test.data["images"], mnist.test.data["labels"], batch_size=args.batch_size)
        self.tb_callback.on_epoch_end(1, dict(("val_test_" + metric, value) for metric, value in zip(self.metrics_names, test_logs)))
        return test_logs[self.metrics_names.index("accuracy")]


if __name__ == "__main__":
    import argparse
    import datetime
    import os
    import re

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=50, type=int, help="Batch size.")
    parser.add_argument("--cnn", default="C-8-3-5-valid,R-[C-8-3-1-same,CB-8-3-1-same],F,D-50", type=str, help="CNN architecture.")
    parser.add_argument("--epochs", default=30, type=int, help="Number of epochs.")
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

    # Compute test set accuracy and print it
    accuracy = network.test(mnist, args)
    with open("mnist_cnn.out", "w") as out_file:
        print("{:.2f}".format(100 * accuracy), file=out_file)