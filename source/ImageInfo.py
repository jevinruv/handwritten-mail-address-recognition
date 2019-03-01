import numpy as np


class ImageInfo:

    def __init__(self, label, image, file_path):
        self.file_path = file_path
        self.image - np.stack(image, axis=0)
        self.label = label
