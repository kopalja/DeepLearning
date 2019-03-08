#!/usr/bin/env python3
import numpy as np


def dict_to_array(d):
    last_key = ord(max(d.keys())) - ord('A')
    array, number = np.zeros(last_key + 1), sum(d.values())
    for key, value in d.items():
        array[ord(key) - ord('A')] = value / number
    return array

if __name__ == "__main__":
    # Load data distribution, each data point on a line
    data_dict = dict()
    with open("numpy_entropy_data.txt", "r") as data:
        for line in data:
            line = line.rstrip("\n")
            if line in data_dict:
                data_dict[line] += 1
            else:
                data_dict[line] = 1


    # TODO: Create a NumPy array containing the data distribution. The
    # NumPy array should contain only data, not any mapping. If required,
    # the NumPy array might be created after loading the model distribution.
    data_array = dict_to_array(data_dict)


    # Load model distribution, each line `word \t probability`.
    model_dict = dict() 
    with open("numpy_entropy_model.txt", "r") as model:
        for line in model:
            line = line.rstrip("\n")
            # TODO: process the line, aggregating using Python data structures
            key, value = line.split('\t')
            model_dict[key] = float(value)

    # TODO: Create a NumPy array containing the model distribution.
    model_array = dict_to_array(model_dict)

    # TODO: Compute and print the entropy H(data distribution). You should not use
    # manual for/while cycles, but instead use the fact that most NumPy methods
    # operate on all elements (for example `*` is vector element-wise multiplication).
    entropy = - np.sum(data_array * np.log(data_array))
    print("{:.2f}".format(entropy))

    # TODO: Compute and print cross-entropy H(data distribution, model distribution)
    # and KL-divergence D_KL(data distribution, model_distribution)
    data_size = data_array.shape[0]
    model_size = model_array.shape[0]
    if data_size < model_size:
        data_array = np.append(data_array, np.zeros(model_size - data_size))
    elif data_size > model_size:
        model_array = np.append(model_array, np.zeros(data_size - model_size))



    cross_entropy = - np.sum(data_array * np.log(model_array))
    print("{:.2f}".format(cross_entropy))

    kl_divergence = cross_entropy - entropy
    print("{:.2f}".format(kl_divergence))