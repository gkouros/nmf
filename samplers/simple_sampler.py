import torch

class SimpleSampler:
    def __init__(self, total, batch):
        self.total = total
        self.batch = batch
        self.curr = total
        self.ids = None

    def nextids(self, batch=None):
        batch = self.batch if batch is None else batch
        self.curr += batch
        if self.curr + batch > self.total:
            # self.ids = torch.LongTensor(np.random.permutation(self.total))
            self.ids = torch.randperm(self.total, dtype=torch.long, device=device)
            self.curr = 0
        ids = self.ids[self.curr : self.curr + batch]
        return ids, ids
