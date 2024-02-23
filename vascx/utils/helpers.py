import os
import random
from pathlib import Path
from typing import List

import pandas as pd

from vascx import Retina


class RetinaLoader:
    def __init__(
        self,
        mask_paths: List[str],
        disc_paths: List[str] = None,
        fundus_paths: List[str] = None,
        meta_path: str = None,
    ):
        self.mask_paths = mask_paths
        self.ids = [int(f.stem) for f in self.mask_paths]
        if disc_paths is not None:
            assert len(disc_paths) == len(
                mask_paths
            ), f"{len(disc_paths)} != {len(mask_paths)}"
        self.disc_paths = disc_paths
        if fundus_paths is not None:
            assert len(fundus_paths) == len(mask_paths)
        self.fundus_paths = fundus_paths

        self.meta_path = meta_path
        if meta_path is not None:
            self.meta = pd.read_csv(meta_path, index_col="id")
        else:
            print("Meta not provided. Retinas will be created with scaling_factor=None")
            self.meta = None

        self.id_to_index = {i: f.stem for i, f in enumerate(self.mask_paths)}
        self.count = -1

    def _get_one_retina(self, index):
        retina = Retina.from_file(self.mask_paths[index])
        if self.disc_paths is not None:
            retina.load_disc(self.disc_paths[index])
        if self.fundus_paths is not None:
            retina.load_fundus_image(self.fundus_paths[index])
        if self.meta is not None:
            retina.scaling_factor = (
                self.meta.loc[self.ids[index], "scaling_w"]
                * self.meta.loc[self.ids[index], "scale"]
            )

        # retina.load_annotation(label_path)
        return retina

    def __getitem__(self, key):
        if isinstance(key, slice):
            start = key.start if key.start is not None else 0
            stop = key.stop if key.stop is not None else len(self)
            step = key.step if key.step is not None else 1
            return [self._get_one_retina(i) for i in range(start, stop, step)]
        else:
            return self._get_one_retina(key)

    def by_id(self, id):
        return self[self.id_to_index[id]]

    def __iter__(self):
        return self

    def __next__(self):
        if self.count < len(self.mask_paths) - 1:
            self.count += 1
            return self[self.count]
        else:
            raise StopIteration

    def __len__(self):
        return len(self.mask_paths)

    def sample(self, n):
        indices = random.sample(range(len(self)), n)
        mask_paths = [self.mask_paths[i] for i in indices]
        disc_paths = (
            [self.disc_paths[i] for i in indices]
            if self.disc_paths is not None
            else None
        )
        fundus_paths = (
            [self.fundus_paths[i] for i in indices]
            if self.fundus_paths is not None
            else None
        )
        return RetinaLoader(mask_paths, disc_paths, fundus_paths, self.meta_path)

    @classmethod
    def from_folders(
        cls,
        masks_folder: str,
        discs_folder: str = None,
        fundus_folder: str = None,
        meta_path: str = None,
    ):
        mask_paths = sorted(list(Path(masks_folder).glob("*.png")))
        filenames = [f.name for f in mask_paths]

        if discs_folder is not None:
            # disc_paths = sorted(list(Path(discs_folder).glob("*.png")))
            disc_paths = [Path(os.path.join(discs_folder), f) for f in filenames]
        else:
            disc_paths = None

        if fundus_folder is not None:
            # fundus_paths = sorted(list(Path(fundus_folder).glob("*.png")))
            disc_paths = [Path(os.path.join(fundus_folder), f) for f in filenames]
        else:
            fundus_paths = None

        return cls(mask_paths, disc_paths, fundus_paths, meta_path)
