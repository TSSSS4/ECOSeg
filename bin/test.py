import multiprocessing
import time
import torch
import sys
import numpy as np


def worker(dataset, index_queue, data_queue, done_event, collate_fn, seed, init_fn, worker_id):
    print('Worker start')
    print('Worker end')
    return


def test():

    data = torch.tensor(np.random.randint(0, 100, (3, 9)))
    max_num, idxs = data.max(1)

    base_seed = torch.LongTensor(1).random_().item()
    worker_result_queue = multiprocessing.Queue()
    done_event = multiprocessing.Event()

    jobs = []
    for i in range(1):
        index_queue = multiprocessing.Queue()
        index_queue.cancel_join_thread()
        p = multiprocessing.Process(
            target=worker,
            args=(None, index_queue,
                  worker_result_queue, done_event,
                  None, base_seed + 0,
                  None, 0))
        jobs.append(p)
        p.start()
    print('end1')
    print('end2')
    print('end3')
    print('end4')
    return


if __name__=='__main__':
    test()


