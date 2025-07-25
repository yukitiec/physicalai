import numpy as np

# cosine scheduler
class CosineScheduler:
    def __init__(self, epochs, lr, warmup_length):
        """
        Arguments
        ---------
        epochs : int
            学習のエポック数．
        lr : float
            学習率．
        warmup_length : int
            warmupを適用するエポック数．
        """
        self.epochs = epochs
        self.lr = lr
        self.lr_min =1.0e-4
        self.warmup = warmup_length

    def __call__(self, epoch):
        """
        Arguments
        ---------
        epoch : int
            現在のエポック数．
        """
        progress = (epoch - self.warmup) / (self.epochs - self.warmup)
        lr = self.lr_min + 0.5*(self.lr-self.lr_min)*(1 + np.cos(np.pi * progress))

        #print(f"{progress=},{lr=}")
        if epoch<self.warmup:
            lr = lr * min(1, (epoch+1) / self.warmup)

        return lr