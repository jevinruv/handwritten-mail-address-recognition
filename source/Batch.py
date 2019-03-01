import numpy as np


class Batch:
    "batch containing images and ground truth texts"

    def __init__(self, gtTexts, imgs):
        self.imgs = np.stack(imgs, axis=0)
        self.gtTexts = gtTexts
