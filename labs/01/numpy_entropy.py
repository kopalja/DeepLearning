#!/usr/bin/env python3
# dd7e3410-38c0-11e8-9b58-00505601122b
# 6e14ef6b-3281-11e8-9de3-00505601122b

import numpy as np


def dict_to_array(d):
    array = np.zeros(len(d))
    number = sum(d.values())
    i = 0
    for key, value in sorted(d.items()):
        array[i] = value / number
        i += 1
    return array

def add_keys(source, dest):
    for key in source:
        if not key in dest:
            dest[key] = 0      

if __name__ == "__main__":
    # Load data distribution, each data point on a line
    data_dict = dict()
    with open("numpy_entropy_data.txt", "r") as data:
        for line in data:
            line = line.rstrip("\n")
            # TODO: process the line, aggregating using Python data structures
            if line in data_dict:
                data_dict[line] += 1
            else:
                data_dict[line] = 1


    # TODO: Create a NumPy array containing the data distribution. The
    # NumPy array should contain only data, not any mapping. If required,
    # the NumPy array might be created after loading the model distribution.
    #data_array = dict_to_array(data_dict)


    # Load model distribution, each line `word \t probability`.
    model_dict = dict() 
    with open("numpy_entropy_model.txt", "r") as model:
        for line in model:
            line = line.rstrip("\n")
            # TODO: process the line, aggregating using Python data structures
            key, value = line.split('\t')
            model_dict[key] = float(value)
    


    # TODO: Create a NumPy array containing the model distribution.
    data_array = dict_to_array(data_dict)

    # TODO: Compute and print the entropy H(data distribution). You should not use
    # manual for/while cycles, but instead use the fact that most NumPy methods
    # operate on all elements (for example `*` is vector element-wise multiplication).
    entropy = - np.sum(data_array * np.log(data_array))
    print("{:.2f}".format(entropy))

    add_keys(model_dict, data_dict)
    add_keys(data_dict, model_dict)
    data_array = dict_to_array(data_dict)
    model_array = dict_to_array(model_dict)

    # TODO: Compute and print cross-entropy H(data distribution, model distribution)
    # and KL-divergence D_KL(data distribution, model_distribution)
    cross_entropy = - np.sum(data_array * np.log(model_array))
    print("{:.2f}".format(cross_entropy))
    kl_divergence = cross_entropy - entropy
    print("{:.2f}".format(kl_divergence))