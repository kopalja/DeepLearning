# dd7e3410-38c0-11e8-9b58-00505601122b
# 6e14ef6b-3281-11e8-9de3-00505601122b
import numpy as np

import tensorflow as tf

from fashion_masks_data import FashionMasks

# TODO: Define a suitable model in the Network class.
# A suitable starting model contains some number of shared
# convolutional layers, followed by two heads, one predicting
# the label and the other one the masks.
class Network:
    def __init__(self, fashion_masks, args):
        inpt = tf.keras.layers.Input((fashion_masks.H, fashion_masks.W, fashion_masks.C))
        hidden = inpt
        for i in range(1, 6):
            hidden = tf.keras.layers.Conv2D(filters= i * 5, kernel_size=(3, 3), strides=1, padding='same')(hidden)
            hidden = tf.keras.layers.BatchNormalization()(hidden)
            hidden = tf.keras.layers.ReLU()(hidden)
            
           
        hidden1 = hidden  
        for i in range(6, 12):
            hidden1 = tf.keras.layers.Conv2D(filters= i * 5, kernel_size=(3, 3), strides=1, padding='same')(hidden1)
            hidden1 = tf.keras.layers.BatchNormalization()(hidden1)
            hidden1 = tf.keras.layers.ReLU()(hidden1)   
            hidden1 = tf.keras.layers.Dropout(0.3)(hidden1)
        hidden1 = tf.keras.layers.Conv2D(filters= 10, kernel_size=(3, 3), strides=1, activation=tf.nn.relu, padding='same')(hidden1)
        mask = tf.keras.layers.Conv2D(filters= 1, kernel_size=(3, 3), strides=1, activation=tf.nn.relu, padding='same')(hidden1)

        hidden2 = hidden
        for f in [100, 120]:
            hidden2 =  tf.keras.layers.Conv2D(filters = f, kernel_size=(3, 3), strides=1, padding='valid')(hidden2)
            hidden2 = tf.keras.layers.BatchNormalization()(hidden2)
            hidden2 = tf.keras.layers.ReLU()(hidden2)  
            hidden2 = tf.keras.layers.MaxPool2D(2)(hidden2) 
            hidden2 = tf.keras.layers.Dropout(0.3)(hidden2)
        hidden2 = tf.keras.layers.Flatten()(hidden2)
        hidden2 = tf.keras.layers.Dense(200, activation=tf.nn.relu)(hidden2)
        hidden2 = tf.keras.layers.Dropout(0.5)(hidden2)
        label = tf.keras.layers.Dense(fashion_masks.LABELS, activation = tf.nn.softmax)(hidden2)

        self.model = tf.keras.Model(inputs = [inpt], outputs = [label, mask])
        self.model.compile(
            loss = [tf.keras.losses.SparseCategoricalCrossentropy(),
                    tf.keras.losses.MeanSquaredError()],
            optimizer = tf.optimizers.Adam(),
        )


    def train(self, fashion_masks, args):
        self.model.fit(
            x = [fashion_masks.train.data["images"]], 
            y = [fashion_masks.train.data["labels"], fashion_masks.train.data["masks"]],
            batch_size = args.batch_size,
            epochs = args.epochs
        )  




if __name__ == "__main__":
    import argparse
    import datetime
    import os
    import re

    #Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=30, type=int, help="Batch size.")
    parser.add_argument("--epochs", default=50, type=int, help="Number of epochs.")
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
    fashion_masks = FashionMasks()


    # Create the network and train
    network = Network(fashion_masks, args)


    network.train(fashion_masks, args)

    # Predict test data in args.logdir
    #print(args.logdir)
    with open(os.path.join("fashion_masks_test.txt"), "w", encoding="utf-8") as out_file:
        # TODO: Predict labels and masks on fashion_masks.test.data["images"],
        # into test_labels and test_masks (test_masks is assumed to be
        # a Numpy array with values 0/1).
        test_labels, test_masks = network.model.predict(fashion_masks.test.data["images"])
        for label, mask in zip(test_labels, test_masks):
            for i in range(len(mask)):
                for j in range(len(mask[i])):
                    if mask[i][j] < 0.5:
                        mask[i][j] = 0
                    else:
                        mask[i][j] = 1
            print(np.argmax(label), *mask.astype(np.uint8).flatten(), file=out_file)
         
