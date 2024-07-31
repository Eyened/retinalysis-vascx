import os
import warnings
from pathlib import Path
from typing import List, Union

from rtnls_enface.loader import FundusLoader
from vascx.fundus.retina import Retina


class RetinaLoader(FundusLoader):
    def __init__(self, av_paths: List[str], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.av_paths = av_paths
        self.num_items = self._check_lengths(
            av_paths,
            self.disc_paths,
            self.fundus_paths,
            self.fovea_locations,
            self.metadata,
            self.ids,
        )

    def _get_one(self, index):
        args = super()._get_one_dict(index)
        args["av_path"] = self.av_paths[index]
        return Retina.from_file(**args)

    @classmethod
    def from_paths(
        cls, av_paths, disc_paths, fundus_paths, fovea_locations_path, meta_path
    ):
        if av_paths is None and disc_paths is None and fundus_paths is None:
            raise ValueError(
                "One of av_paths, disc_paths or fundus_paths must be provided"
            )
        (av_paths, disc_paths, fundus_paths), stems = cls._get_filenames(
            av_paths, disc_paths, fundus_paths
        )
        fovea_locations = cls._read_fovea_locations(fovea_locations_path, stems)
        metadata = cls._read_meta(meta_path, stems)
        return cls(av_paths, disc_paths, fundus_paths, fovea_locations, metadata, stems)

    @classmethod
    def from_folders(
        cls,
        av_folder: str = None,
        discs_folder: str = None,
        fundus_folder: str = None,
        fovea_locations_csv: str = None,
        meta_csv: str = None,
    ):
        av_paths = (
            sorted(list(Path(av_folder).glob("*.png")))
            if av_folder is not None
            else None
        )
        disc_paths = (
            sorted(list(Path(discs_folder).glob("*.png")))
            if discs_folder is not None
            else None
        )
        fundus_paths = (
            sorted(list(Path(fundus_folder).glob("*.png")))
            if fundus_folder is not None
            else None
        )
        (av_paths, disc_paths, fundus_paths), stems = cls._get_filenames(
            av_paths, disc_paths, fundus_paths
        )
        fovea_locations = cls._read_fovea_locations(fovea_locations_csv, stems)
        metadata = cls._read_meta(meta_csv, stems)

        return cls(
            av_paths, disc_paths, fundus_paths, fovea_locations, metadata, ids=stems
        )

    @classmethod
    def from_folder(
        cls,
        base_folder: Union[str, Path],
        av_subfolder: str = "av",
        discs_subfolder: str = "discs",
        fundus_subfolder: str = "rgb",
        fovea_locations_csv: str = "fovea.csv",
        meta_csv: str = "meta.csv",
    ):
        base = Path(base_folder)
        if not os.path.exists(base / meta_csv):
            warnings.warn(f"file {base/meta_csv} not found")
            meta_csv = None
        if not os.path.exists(base / fovea_locations_csv):
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
