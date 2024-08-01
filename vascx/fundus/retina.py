from pathlib import Path
from typing import Any, Callable, Tuple, Union

import cv2
import numpy as np
from typing_extensions import TypeAlias

from rtnls_enface import Fundus
from rtnls_enface.utils.image import match_resolution
from vascx.fundus.features.base import LayerFeature
from vascx.fundus.layer import VesselTreeLayer
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

    def calc_features(self, feature_set: FeatureSet):
        layer_features = {
            p: fn for p, fn in feature_set.items() if isinstance(fn, LayerFeature)
        }

        all_features = {}
        for feature_name, feature in layer_features.items():
            for layer_name, layer in self.layers.items():
                res = feature.compute(layer)
                if isinstance(res, dict):
                    for k, v in res.items():
                        all_features[f"{feature_name}_{layer_name}_{k}"] = v
                else:
                    all_features[f"{feature_name}_{layer_name}"] = res

        return all_features

    @classmethod
    def from_file(
        cls,
        av_path: str,
        disc_path: Union[str, Path] = None,
        fundus_path: Union[str, Path] = None,
        fovea_location: Tuple[float, float] = None,
        bounds=None,
        threshold=0.5,
        scaling_factor=1,
        id: Any = None,
        **kwargs,
    ):
        layers = load_av_segmentation(av_path, threshold)

        def get_layer_color(layer_name):
            if layer_name == "arteries":
                return (1, 0, 0)
            elif layer_name == "veins":
                return (0, 0, 1)
            else:
                return (1, 1, 1)

        layers = {
            key: VesselTreeLayer(val, name=key, color=get_layer_color(key))
            for key, val in layers.items()
        }

        retina = cls(
            disc_path_or_mask=disc_path,
            fundus_path_or_mask=fundus_path,
            layers=layers,
            fovea_location=fovea_location,
            scaling_factor=scaling_factor,
            bounds=bounds,
            id=id,
        )
        retina.id = Path(av_path).stem

        return retina
