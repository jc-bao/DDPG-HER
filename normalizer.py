import numpy as np
from mpi4py import MPI
import threading
class Normalizer:
    def __init__(self, size, eps=1e-2, default_clip_range=np.inf):
        self.size = size # input size
        self.eps = eps # minimum of std
        self.default_clip_range = default_clip_range
        # local param
        self.count = np.zeros(1, np.float32)
        self.sum = np.zeros(size, np.float32)
        self.sumsq = np.zeros(size, np.float32)
        # global param
        self.total_sum = np.zeros(self.size, np.float32)
        self.total_sumsq = np.zeros(self.size, np.float32)
        self.total_count = np.ones(1, np.float32)
        self.mean = np.zeros(size, np.float32)
        self.std = np.zeros(size, np.float32) + np.square(self.eps)
        # thread locker
        self.lock = threading.Lock()

    def normalize(self, x, clip_range = None):
        if clip_range is None:
            clip_range = self.default_clip_range
        return np.clip((x-self.mean)/self.std, -clip_range, clip_range)

    def update(self, x):
        x = x.reshape(-1,self.size)
        with self.lock:
            self.sum += x.sum(axis = 0)
            self.sumsq += np.square(x).sum(axis = 0)
            self.count += x.shape[0]
            local_count = self.count.copy()
            local_sum = self.sum.copy()
            local_sumsq = self.sumsq.copy()
            # reset
            self.count[...] = 0
            self.sum[...] = 0
            self.sumsq[...] = 0
        sync_sum, sync_sumsq, sync_count = self._sync(local_sum, local_sumsq, local_count)
        self.total_sum += sync_sum
        self.total_sumsq += sync_sumsq
        self.total_count += sync_count
        # mean and std update
        self.mean = self.sum/self.count
        std = (self.sumsq / self.count) - np.square(self.sum / self.count)
        self.std = np.sqrt(np.maximum(np.square(self.eps), std))

    def _sync(self, summ, sumsq, count):
        summ[...] = self._mpi_average(summ)
        sumsq[...] = self._mpi_average(sumsq)
        count[...] = self._mpi_average(count)
        return summ, sumsq, count

    def _mpi_average(self, x):
        buf = np.zeros_like(x)
        MPI.COMM_WORLD.Allreduce(x, buf, op=MPI.SUM)
        buf /= MPI.COMM_WORLD.Get_size()
        return buf