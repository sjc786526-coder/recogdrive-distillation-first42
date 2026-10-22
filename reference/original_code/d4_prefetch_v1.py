"""Bounded ordered thread prefetch; Tensor IPC is unnecessary on CPFS."""
from collections import deque
from concurrent.futures import ThreadPoolExecutor


class PrefetchLoader:
    def __init__(self,dataset,workers=2,prefetch_factor=2,timeout=120):
        self.dataset=dataset;self.workers=workers;self.depth=max(1,workers*prefetch_factor);self.timeout=timeout

    def __iter__(self):
        if self.workers<=0:
            for i in range(len(self.dataset)):yield self.dataset[i]
            return
        with ThreadPoolExecutor(max_workers=self.workers,thread_name_prefix='d4_read') as pool:
            pending=deque();position=0
            for _ in range(min(self.depth,len(self.dataset))):
                pending.append(pool.submit(self.dataset.__getitem__,position));position+=1
            while pending:
                item=pending.popleft().result(timeout=self.timeout)
                if position<len(self.dataset):pending.append(pool.submit(self.dataset.__getitem__,position));position+=1
                yield item
