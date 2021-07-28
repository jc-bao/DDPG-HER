import numpy as np

class Normalizer:
    def __init__(self, size, eps=1e-2, default_clip_range=np.inf):
        self.size = size # input size
        self.eps = eps # minimum of std
        self.default_clip_range = default_clip_range
        self.count = 0
        self.sum = np.zeros(size, np.float32)
        self.sumsq = np.zeros(size, np.float32)
        self.mean = np.zeros(size, np.float32)
        self.std = np.zeros(size, np.float32) + np.square(self.eps)

    def normalize(self, x, clip_range = None):
        if clip_range is None:
            clip_range = self.default_clip_range
        return np.clip((x-self.mean)/self.std, -clip_range, clip_range)

    def update(self, x):
        x = x.reshape(-1,self.size)
        self.sum += x.sum(axis = 0)
        self.sumsq += np.square(x).sum(axis = 0)
        self.count += x.shape[0]
        self.mean = self.sum/self.count
        std = (self.sumsq / self.count) - np.square(self.sum / self.count)
        self.std = np.sqrt(np.maximum(np.square(self.eps), std))