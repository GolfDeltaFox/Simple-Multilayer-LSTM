import os

def initializer():
    paths = ['./logs', './checkpoints']
    for path in paths:
        if not os.path.exists(path):
            os.makedirs(path)
