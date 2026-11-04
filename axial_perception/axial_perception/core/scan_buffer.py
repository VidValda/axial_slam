from collections import deque

import numpy as np


class ScanBuffer:
    """Last n scans, already expressed in one fixed frame."""

    def __init__(self, n):
        self.scans = deque(maxlen=n)

    def add(self, pts):
        self.scans.append(pts)

    def points(self):
        return np.concatenate(self.scans) if self.scans else np.empty((0, 3))

    def __len__(self):
        return len(self.scans)
