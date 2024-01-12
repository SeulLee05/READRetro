# -*- coding: utf-8 -*-
"""
A place for data split algorithms.
"""
import os

from megan.split import DatasetSplit


class DefaultSplit(DatasetSplit):
    """
    A default split that is downloaded from some external source.
    """

    def __init__(self):
        super(DefaultSplit, self).__init__()

    @property
    def key(self) -> str:
        return 'default'

    def split_dataset(self, dataset):
        path = self.path(dataset.splits_dir)
        if not os.path.exists(path):
            raise FileNotFoundError(f"DefaultSplit should already be in f{path}. "
                                    f"You should probably download it manually")
