#import numpy as np
import itertools
import pickle


def create_minibatch(generator, mbatch_size, nested_tuples=1):    
    if nested_tuples == 1:
        while b_data := list(itertools.islice(generator, mbatch_size)):
            yield [[b_data[i][j] for i in range(len(b_data))] for j in range(len(b_data[0]))]
    elif nested_tuples == 2:
        while b_data := list(itertools.islice(generator, mbatch_size)):
            yield [ [ [b_data[i][j][k] for i in range(len(b_data))] for k in range(len(b_data[0][0])) ] for j in range(len(b_data[0])) ]
    elif nested_tuples == 0:
        while b_data := list(itertools.islice(generator, mbatch_size)):
            yield b_data
