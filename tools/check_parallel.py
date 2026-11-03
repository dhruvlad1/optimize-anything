"""Does a CPU-bound task slow down when several run at once? (Track D check)"""
import os
for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[k] = "1"          # must be set before numpy is imported

import time
import numpy as np
from multiprocessing import Pool


def work(_):
    a = np.random.default_rng(0).random((200, 200))
    t = time.perf_counter()
    for _ in range(300):
        a = a @ a
        a /= np.abs(a).max()
    return time.perf_counter() - t


if __name__ == "__main__":
    print("logical CPUs:", os.cpu_count())
    print("solo task:", round(work(0), 3), "s")
    for p in (4, 8):
        with Pool(p) as pool:
            pool.map(work, range(p))                 # warm-up
            times = pool.map(work, range(p))
        print(f"{p} in parallel, each task:", [round(x, 3) for x in times])