# dd7e3410-38c0-11e8-9b58-00505601122b
# 6e14ef6b-3281-11e8-9de3-00505601122b


#!/usr/bin/env python3
import numpy as np
import tensorflow as tf

from nli_dataset import NLIDataset

class Network:
    def __init__(self, args):
        # TODO: Define a suitable model.
        self._languagesNum = 11
        word_ids = tf.keras.layers.Input((None,), dtype=tf.int32)
        embed_words = tf.keras.layers.Embedding(input_dim=60268, output_dim=64, mask_zero=False)(word_ids)
        # charseq_ids = tf.keras.layers.Input((None,), dtype=tf.int32)
        # charseqs = tf.keras.layers.Input((None,))
        hidden = embed_words
        cnn_cles = []
        for i in range(3, 8):
            cnn_cle = tf.keras.layers.Conv1D(filters = (i + 1) * 10, kernel_size= i , strides=1, padding='valid', activation=tf.nn.relu)(embed_words)
            cnn_cle = tf.keras.layers.GlobalMaxPooling1D()(cnn_cle)
            cnn_cles.append(cnn_cle)
        cnn_cle = tf.keras.layers.Concatenate()(cnn_cles)  
        hidden = tf.keras.layers.Dense(360, activation=tf.nn.relu)(cnn_cle)
        output = tf.keras.layers.Dense(self._languagesNum, activation=tf.nn.softmax)(hidden)

        self._model = tf.keras.Model(inputs = word_ids, outputs = output)


        self._loss = tf.keras.losses.CategoricalCrossentropy()
        self._optimizer = tf.optimizers.Adam()


        self._writer = tf.summary.create_file_writer(args.logdir, flush_millis=10 * 1000)


    def train_batch(self, words, charseq_ids, charseqs, languages):
        with tf.GradientTape() as tape:
            output = self._model(words, training = True)
            loss = self._loss(tf.one_hot([languages], self._languagesNum, dtype = tf.float32), output) 
            gradients = tape.gradient(loss, self._model.trainable_variables)
        self._optimizer.apply_gradients(zip(gradients, self._model.trainable_variables))
        return loss

    def train_epoch(self, nli, args):
        i = 0
        for batch in nli.train.batches(args.batch_size):
            i += 1
            loss = self.train_batch(batch.word_ids, batch.charseq_ids, batch.charseqs, batch.languages)
            if i % 10 == 0:
                print(loss)

    def predict_batch(self, words, charseq_ids, charseqs, languages):
        output = self._model(words, training = False)
        result = tf.argmax(output, axis =  1)
        return list(result.numpy())

    def predict(self, dataset, args):
        # TODO: Predict method should return a list/np.ndaddar, each element
        # being the predicted language for a sencence.
        predictions = []
        for batch in dataset.batches(args.batch_size):
            prediction = self.predict_batch(batch.word_ids, batch.charseq_ids, batch.charseqs, batch.languages)
            predictions.extend(prediction)
        return predictions

if __name__ == "__main__":
    import argparse
    import datetime
    import os
    import re

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=30, type=int, help="Batch size.")
    parser.add_argument("--epochs", default=8, type=int, help="Number of epochs.")
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
    nli = NLIDataset()


    # Create the network and train
    network = Network(args)
    for epoch in range(args.epochs):
        network.train_epoch(nli, args)
        predictions = network.predict(nli.dev, args)
        desired = nli.dev._languages
        correct = 0
        for i in range(len(predictions)):
            if desired[i] == predictions[i]:
                correct += 1
        print("accuracy: {0}".format(correct / len(predictions)))

    # Generate test set annotations, but in args.logdir to allow parallel execution.
    out_path = "nli_competition_test.txt"
    #if os.path.isdir(args.logdir): out_path = os.path.join(args.logdir, out_path)
    with open(out_path, "w", encoding="utf-8") as out_file:
        languages = network.predict(nli.test, args)
        for language in languages:
            print(nli.test.vocabulary("languages")[language], file=out_file)
