# dd7e3410-38c0-11e8-9b58-00505601122b
# 6e14ef6b-3281-11e8-9de3-00505601122b

import argparse
import datetime
import os
import re
import math

import numpy as np
import tensorflow as tf

from mnist import MNIST

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=50, type=int, help="Batch size.")
parser.add_argument("--decay", default=None, type=str, help="Learning decay rate type")
parser.add_argument("--epochs", default=2, type=int, help="Number of epochs.")
parser.add_argument("--hidden_layer", default=200, type=int, help="Size of the hidden layer.")
parser.add_argument("--learning_rate", default=0.01, type=float, help="Initial learning rate.")
parser.add_argument("--learning_rate_final", default="0.000001", type=float, help="Final learning rate.")
parser.add_argument("--momentum", default=None, type=float, help="Momentum.")
parser.add_argument("--optimizer", default="SGD", type=str, help="Optimizer to use.")
parser.add_argument("--recodex", default=False, action="store_true", help="Evaluation in ReCodEx.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
args = parser.parse_args()
# Fix random seeds
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

# Load data
mnist = MNIST()



# Create the model
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=[MNIST.H, MNIST.W, MNIST.C]),
    tf.keras.layers.Dense(args.hidden_layer, activation=tf.nn.relu),
    tf.keras.layers.Dense(MNIST.LABELS, activation=tf.nn.softmax),
])

# TODO: Use the required `args.optimizer` (either `SGD` or `Adam`).
# For `SGD`, `args.momentum` can be specified. If `args.decay` is
# not specified, pass the given `args.learning_rate` directly to the
# optimizer as a `learning_rate` argument. If `args.decay` is set, then
# - for `polynomial`, use `tf.keras.optimizers.schedules.PolynomialDecay`
#   using the given `args.learning_rate_final`;
# - for `exponential`, use `tf.keras.optimizers.schedules.ExponentialDecay`
#   and set `decay_rate` appropriately to reach `args.learning_rate_final`
#   just after the training (and keep the default `staircase=False`).
# In both cases, `decay_steps` should be total number of training batches.
# If a learning rate schedule is used, you can find out current learning rate
# by using `model.optimizer.learning_rate(model.optimizer.iterations)`,
# so after training this value should be `args.learning_rate_final`.

def set_optimizer(args, decay_steps):
    if args.decay == None:
        scheduler = args.learning_rate
    elif args.decay == "polynomial":
        scheduler = tf.keras.optimizers.schedules.PolynomialDecay(
            args.learning_rate,
            end_learning_rate = args.learning_rate_final,
            power = 1,
            decay_steps = decay_steps)
    elif args.decay == "exponential":
        scheduler = tf.optimizers.schedules.ExponentialDecay(
            args.learning_rate,
            decay_rate = math.pow((args.learning_rate_final / args.learning_rate), (1 / decay_steps)),
            decay_steps = 1)
    if args.optimizer == "SGD":
        if args.momentum == None:
            return tf.keras.optimizers.SGD(learning_rate = scheduler)
        else:
            return tf.keras.optimizers.SGD(learning_rate = scheduler, momentum = args.momentum)
    elif args.optimizer == "Adam":
        return tf.keras.optimizers.Adam(learning_rate = scheduler)
        

optimizer = set_optimizer(args, len(mnist.train.data["images"]) * args.epochs // args.batch_size)

model.compile(
    optimizer = optimizer,
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
)

tb_callback=tf.keras.callbacks.TensorBoard(args.logdir, histogram_freq=1, update_freq=1000, profile_batch=1)
tb_callback.on_train_end = lambda *_: None
model.fit(
    mnist.train.data["images"], mnist.train.data["labels"],
    batch_size=args.batch_size, epochs=args.epochs,
    validation_data=(mnist.dev.data["images"], mnist.dev.data["labels"]),
    callbacks=[tb_callback],
)


#args.learning_rate_final = model.optimizer.learning_rate(model.optimizer.iterations)

test_logs = model.evaluate(
    mnist.test.data["images"], mnist.test.data["labels"], batch_size=args.batch_size,
)

accuracy = test_logs[1]

tb_callback.on_epoch_end(1, dict(("val_test_" + metric, value) for metric, value in zip(model.metrics_names, test_logs)))

# TODO: Write test accuracy as percentages rounded to two decimal places.
with open("mnist_training.out", "w") as out_file:
    print("{:.2f}".format(100 * accuracy), file=out_file)
