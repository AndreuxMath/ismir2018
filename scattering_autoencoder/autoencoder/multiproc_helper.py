import random
import torch
import torch.multiprocessing as multiprocessing
from torch._C import _set_worker_signal_handlers, _error_if_any_worker_fails
from torch._C import _remove_worker_pids, _update_worker_pids
import time
import sys
import threading
import signal


class ExceptionWrapper(object):
    r"""Wraps an exception plus traceback to communicate across threads"""

    def __init__(self, exc_info):
        self.exc_type = exc_info[0]
        self.exc_msg = "".join(traceback.format_exception(*exc_info))


_use_shared_memory = False


def my_default_collate(batch):
    """Puts each data field into a tensor with outer dimension batch size"""
    if torch.is_tensor(batch[0]):
        out = None
        if _use_shared_memory:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum([x.numel() for x in batch])
            storage = batch[0].storage()._new_shared(numel)
            out = batch[0].new(storage)
        return torch.stack(batch, 0, out=out)
    else:
        err_msg = "batch must contain tensors; found {}"
        raise TypeError(err_msg.format(type(batch[0])))


def _worker_loop(gen, data_queue, collate_fn, batch_size, seed, worker_id):
    global _use_shared_memory
    _use_shared_memory = True

    # Initialize C side signal handlers for SIGBUS and SIGSEGV
    _set_worker_signal_handlers()

    torch.set_num_threads(1)
    random.seed(seed)
    torch.manual_seed(seed)

    while True:
        try:
            samples = collate_fn([gen.get() for _ in range(batch_size)])
        except Exception:
            data_queue.put(ExceptionWrapper(sys.exc_info()))
        else:
            data_queue.put(samples, True)


_SIGCHLD_handler_set = False


def _set_SIGCHLD_handler():
    # Windows doesn't support SIGCHLD handler
    if sys.platform == 'win32':
        return
    # can't set signal in child threads
    if not isinstance(threading.current_thread(), threading._MainThread):
        return
    global _SIGCHLD_handler_set
    if _SIGCHLD_handler_set:
        return
    previous_handler = signal.getsignal(signal.SIGCHLD)
    if not callable(previous_handler):
        previous_handler = None

    def handler(signum, frame):
        # This following call uses `waitid` with WNOHANG from C side.
        # Therefore, Python can still get and update the process
        # status successfully.
        _error_if_any_worker_fails()
        if previous_handler is not None:
            previous_handler(signum, frame)

    signal.signal(signal.SIGCHLD, handler)
    _SIGCHLD_handler_set = True


class MPGenerator(object):
    def __init__(self, generator, collate_fn, num_workers=1, queue_size=1,
                 batch_size=16):
        self.generator = generator
        self.batch_size = batch_size
        self.queue_size = queue_size
        self.num_workers = num_workers
        self.done_event = threading.Event()
        self.collate_fn = collate_fn
        self.batch_size = batch_size

        if self.num_workers > 0:
            self.data_queue = multiprocessing.Queue(self.queue_size)
            # base_seed = 107
            base_seed = torch.LongTensor(1).random_()[0]
            self.shutdown = False
            self.worker_pids_set = False
            self.workers = [
                multiprocessing.Process(
                    target=_worker_loop,
                    args=(self.generator, self.data_queue, self.collate_fn,
                          self.batch_size, base_seed + i, i)
                )
                for i in range(self.num_workers)
            ]
            for w in self.workers:
                w.daemon = True  # ensure that the worker exits on process exit
                w.start()

            # set workers' PID
            _update_worker_pids(id(self), tuple(w.pid for w in self.workers))
            _set_SIGCHLD_handler()
            self.worker_pids_set = True

    def _get_batch(self):
        # Allows to encapsulate the timeout function here
        return self.data_queue.get()

    def get(self):
        if self.num_workers == 0:
            batch = self.collate_fn([self.generator.get()
                                     for _ in range(self.batch_size)])
        else:
            batch = self._get_batch()
        return batch

    def _shutdown_workers(self):
        try:
            if not self.shutdown:
                self.shutdown = True
                self.done_event.set()
                try:
                    while not self.data_queue.empty():
                        self.data_queue.get()
                except FileNotFoundError:
                    pass
        finally:
            # remove pids no matter what
            if self.worker_pids_set:
                _remove_worker_pids(id(self))
                self.worker_pids_set = False

    def __del__(self):
        if self.num_workers > 0:
            self._shutdown_workers()


class DummyGenerator():
    def __init__(self):
        pass

    def get(self):
        return torch.rand(1)


if __name__ == '__main__':
    queue_size = 10
    num_workers = 1
    batch_size = 128

    gen = DummyGenerator()

    mini = MPGenerator(gen, my_default_collate, num_workers=num_workers,
                       queue_size=queue_size, batch_size=batch_size)
    for i in range(10):
        sample = mini.get()
        if hasattr(mini, 'data_queue'):
            print(i, 'sample =', sample, 'queue_size =',
                  mini.data_queue.qsize())
        else:
            print(i, 'sample =', sample)
        time.sleep(1)
