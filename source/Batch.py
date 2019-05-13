import numpy as np


class Batch:

    def __init__(self, labels, imgs):
        self.imgs = np.stack(imgs, axis=0)
        self.labels = labels
