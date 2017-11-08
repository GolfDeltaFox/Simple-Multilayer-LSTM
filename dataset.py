import random
import numpy as np

class data_generator(object):
    def __init__(self, filename, batch_size=32, num_steps=200):
        self.filename = filename
        self.gen_data(batch_size=batch_size, num_steps=batch_size)
        self.cursors = [random.randint(0, self.n_chars - num_steps - 1 ) for _ in range(batch_size)]

    def gen_data(self, batch_size=32, num_steps=200):
        self.raw_text = open(self.filename).read()
        self.raw_text = self.raw_text.lower()
        self.chars = sorted(list(set(self.raw_text)))
        self.char_to_int = dict((c, i) for i, c in enumerate(self.chars))
        self.n_chars = len(self.raw_text)
        self.n_classes = len(self.chars)
        X = [ self.char_to_int[elt] for elt in  self.raw_text]
        Y = [ self.char_to_int[elt] for elt in  self.raw_text[1:]]
        self.X = X
        self.Y = Y
        print("Initializing done.")

    def next(self, batch_size, num_steps):
        X_batch = []
        Y_batch = []
        for i in range(batch_size):
            cursor = self.cursors[i]
            if cursor > self.n_chars - num_steps - 1:
                cursor = (self.n_chars - cursor)
            X_batch.append(self.X[cursor:cursor+num_steps])
            Y_batch.append(self.Y[cursor:cursor+num_steps])
            self.cursors[i] = cursor+num_steps
        return np.array(X_batch), np.array(Y_batch)
