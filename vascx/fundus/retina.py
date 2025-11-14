from pathlib import Path
from typing import Any, Callable, Optional, Tuple, Union

import cv2
import numpy as np
from typing_extensions import TypeAlias
from matplotlib import pyplot as plt
from rtnls_enface.fundus import Fundus
from rtnls_enface.utils.data_loading import open_binary_mask
from rtnls_enface.utils.image import match_resolution
from vascx.fundus.features.base import LayerFeature, RetinaFeature, VesselsLayerFeature
from vascx.fundus.layer import VesselTreeLayer
from vascx.fundus.vessels_layer import FundusVesselsLayer
from vascx.shared.features import FeatureSet
from vascx.shared.segment import Segment
from vascx.utils import load_av_segmentation

Aggregator: TypeAlias = Callable[[np.ndarray], np.float32]
FeatureType: TypeAlias = Union[VesselTreeLayer, Segment]


class Retina(Fundus):
    @property
    def arteries(self) -> VesselTreeLayer:
        return self.layers["arteries"]

    @property
    def veins(self) -> VesselTreeLayer:
        return self.layers["veins"]

    @property
    def vessels(self) -> FundusVesselsLayer:
        return self.layers["vessels"]

    def set_retina(self, retina):
        self.retina = retina

    def load_annotation(self, fpath: str):
        layers = load_av_segmentation(fpath)
        layers = {
            key: match_resolution(
                mask.astype(np.uint8), self.resolution, cv2.INTER_NEAREST
            ).astype(bool)
            for key, mask in layers.items()
        }

        self.gt = Retina(layers)
        if self.disc is not None:
            self.gt.disc = self.disc

    def dice(self, ignore_disk=True):
        assert self.gt is not None

        res = {}
        for layer in self.layers.keys():
            res[layer] = self.layers[layer].auc(
                self.gt.layers[layer], remove_disk=ignore_disk
            )

        return res

    def calc_features(self, feature_set: FeatureSet, plots_folder: Optional[str] = None):
        all_features = {}
        for feature_name, feature in feature_set.items():
            if isinstance(feature, RetinaFeature):
                targets = [("retina", self)]
            elif isinstance(feature, LayerFeature):
                targets = [(layer.name, layer) for layer in self.get_layers(VesselTreeLayer)]
            elif isinstance(feature, VesselsLayerFeature):
                targets = [(layer.name, layer) for layer in self.get_layers(FundusVesselsLayer)]
            else:
                continue

            if len(targets) == 0:
                print(f"No targets found on which to compute feature {feature_name}.")

            for target_name, target in targets:
                res = feature.compute(target)
                if isinstance(res, dict):
                    for k, v in res.items():
                        all_features[f"{feature_name}_{target.name}_{k}"] = v
                else:
                    all_features[f"{feature_name}_{target.name}"] = res
                
                if plots_folder is not None and res is not None: # only save plots if the feature was computed successfully
                    fig, ax = plt.subplots(1, 1, figsize=(8, 8), dpi=300)
                    feature.plot(ax, target)
                    fname_prefix = (str(self.id) if self.id is not None else "sample")
                    fname = f"{fname_prefix}_{feature_name}_{target_name}.png"
                    fig.savefig(Path(plots_folder) / fname, dpi=200, bbox_inches="tight")
                    plt.close(fig)

        return all_features


    @classmethod
    def from_file(
        cls,
        av_path: str | Path = None,
        vessels_path: str | Path = None,
        disc_path: str | Path = None,
        fundus_path: str | Path = None,
        fovea_location: Tuple[float, float] = None,
        bounds=None,
        threshold=0.5,
        scaling_factor=1,
        id: Any = None,
        **kwargs,
    ):
        # assert av_path is not None or vessels_path is not None, (
        #     "Either av_path or vessels_path must be provided"
        # )

        layers = {}
        if av_path is not None:
            av_layers = load_av_segmentation(av_path, threshold)

            def get_layer_color(layer_name):
                if layer_name == "arteries":
                    return (1, 0, 0)
                elif layer_name == "veins":
                    return (0, 0, 1)
                else:
                    return (1, 1, 1)

            for key, val in av_layers.items():
                layers[key] = VesselTreeLayer(key, val, color=get_layer_color(key))

        if vessels_path is not None:
            layers["vessels"] = FundusVesselsLayer(
                name="vessels", mask=open_binary_mask(vessels_path)
            )

        retina = cls(
            disc_path_or_mask=disc_path,
            fundus_path_or_mask=fundus_path,
            layers=layers,
            fovea_location=fovea_location,
            scaling_factor=scaling_factor,
            bounds=bounds,
            id=id,
        )

        return retina
