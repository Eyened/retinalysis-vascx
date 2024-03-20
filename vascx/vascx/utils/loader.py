import os
import warnings
from pathlib import Path
from typing import List

from rtnls_enface.loader import FundusLoader

from vascx.retina import Retina


class RetinaLoader(FundusLoader):
    def __init__(self, av_paths: List[str], *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.num_items != 0:
            assert len(av_paths) == self.num_items
        else:
            self.num_items = len(av_paths)
        self.av_paths = av_paths

    def _get_one(self, index):
        args = super()._get_one_dict(index)
        args["av_path"] = self.av_paths[index]
        return Retina.from_file(**args)

    @classmethod
    def from_folders(
        cls,
        av_folder: str = None,
        discs_folder: str = None,
        fundus_folder: str = None,
        fovea_locations_csv: str = None,
        meta_csv: str = None,
    ):
        disc_paths, fundus_paths, fovea_locations, metadata = cls._read_folders(
            discs_folder, fundus_folder, fovea_locations_csv, meta_csv
        )

        if av_folder is not None:
            av_paths = sorted(list(Path(av_folder).glob("*.png")))
        else:
            av_paths = None

        return cls(av_paths, disc_paths, fundus_paths, fovea_locations, metadata)

    @classmethod
    def from_folder(
        cls,
        base_folder: str | Path,
        av_subfolder: str = "av",
        discs_subfolder: str = "discs",
        fundus_subfolder: str = "rgb",
        fovea_locations_csv: str = "fovea.csv",
        meta_csv: str = "meta.csv",
    ):
        base = Path(base_folder)
        if not os.path.exists(meta_csv):
            warnings.warn(f"file {base/meta_csv} not found")
            meta_csv = None
        if not os.path.exists(fovea_locations_csv):
            warnings.warn(f"file {base/fovea_locations_csv} not found")
            fovea_locations_csv = None
        return cls.from_folders(
            base / av_subfolder if av_subfolder is not None else None,
            base / discs_subfolder if discs_subfolder is not None else None,
            base / fundus_subfolder if fundus_subfolder is not None else None,
            base / fovea_locations_csv if fovea_locations_csv is not None else None,
            base / meta_csv if meta_csv is not None else None,
        )

    def to_dict(self):
        res = super().to_dict()
        for el, av_path in zip(res, self.av_paths):
            el["av_path"] = av_path
        return res
