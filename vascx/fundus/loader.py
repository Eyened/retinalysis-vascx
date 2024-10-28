import os
import warnings
from pathlib import Path
from typing import List, Union

from rtnls_enface.loader import FundusLoader
from vascx.fundus.retina import Retina


class RetinaLoader(FundusLoader):
    def make_object(self, id):
        return Retina.from_file(**self.get_item(id))

    @classmethod
    def from_paths(
        cls, 
        av_paths: List[str] = None, 
        vessels_paths: List[str] = None, 
        disc_paths: List[str] = None, 
        fundus_paths: List[str] = None, 
        fovea_locations_path: str = None, 
        meta_path: str = None
    ):
        if av_paths is None and vessels_paths is None and disc_paths is None and fundus_paths is None:
            raise ValueError(
                "One of av_paths, disc_paths or fundus_paths must be provided"
            )
        items = cls._get_items_from_files(av_path=av_paths, vessels_path=vessels_paths, disc_path=disc_paths, fundus_path=fundus_paths)
        cls._add_fovea_locations(items, fovea_locations_path)
        cls._add_meta(items, meta_path)
        return cls(items)

    @classmethod
    def from_folders(
        cls,
        av_folder: str = None,
        vessels_folder: str = None,
        discs_folder: str = None,
        fundus_folder: str = None,
        fovea_locations_csv: str = None,
        meta_csv: str = None,
    ):
        items = cls._get_items_from_folders(
            av_path=av_folder,
            disc_path=discs_folder,
            fundus_path=fundus_folder,
            vessels_path=vessels_folder,
        )
        cls._add_fovea_locations(items, fovea_locations_csv)
        cls._add_meta(items, meta_csv)

        return cls(items)

    @classmethod
    def from_folder(
        cls,
        base: Union[str, Path],
        av_subfolder: str = "av",
        vessels_subfolder: str = "vessels",
        discs_subfolder: str = "discs",
        fundus_subfolder: str = "rgb",
        fovea_locations_csv: str = "fovea.csv",
        meta_csv: str = "meta.csv",
    ):
        base = Path(base)
        if av_subfolder is not None and not os.path.exists(base / av_subfolder):
            warnings.warn(f"folder {base/av_subfolder} not found")
            av_subfolder = None
        if vessels_subfolder is not None and not os.path.exists(base / vessels_subfolder):
            warnings.warn(f"folder {base/vessels_subfolder} not found")
            vessels_subfolder = None
        if discs_subfolder is not None and not os.path.exists(base / discs_subfolder):
            warnings.warn(f"folder {base/discs_subfolder} not found")
            discs_subfolder = None
        if fundus_subfolder is not None and not os.path.exists(base / fundus_subfolder):
            warnings.warn(f"folder {base/fundus_subfolder} not found")
            fundus_subfolder = None
        if meta_csv is not None and not os.path.exists(base / meta_csv):
            warnings.warn(f"file {base/meta_csv} not found")
            meta_csv = None
        if fovea_locations_csv is not None and not os.path.exists(base / fovea_locations_csv):
            warnings.warn(f"file {base/fovea_locations_csv} not found")
            fovea_locations_csv = None
        return cls.from_folders(
            base / av_subfolder if av_subfolder is not None else None,
            base / vessels_subfolder if vessels_subfolder is not None else None,
            base / discs_subfolder if discs_subfolder is not None else None,
            base / fundus_subfolder if fundus_subfolder is not None else None,
            base / fovea_locations_csv if fovea_locations_csv is not None else None,
            base / meta_csv if meta_csv is not None else None,
        )
