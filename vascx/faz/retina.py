from pathlib import Path
from typing import Any

import numpy as np
from rtnls_utils.data_loading import load_image
from vascx.faz.layer import FazLayer
from vascx.fundus.features.base import FeatureSet, LayerFeature
from vascx.utils import load_av_segmentation

from rtnls_enface.faz_enface import FAZEnface
from rtnls_enface.utils.data_loading import open_mask


class Retina(FAZEnface):
    @property
    def arteries(self) -> FazLayer:
        return self.layers["arteries"]

    @property
    def veins(self) -> FazLayer:
        return self.layers["veins"]

    def set_retina(self, retina):
        self.retina = retina

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
        av_path: str | Path,
        faz_path: str | Path = None,
        image_path: str | Path = None,
        threshold=0.5,
        id: Any = None,
        av_loader=load_av_segmentation,
    ):
        if faz_path is not None:
            im = open_mask(faz_path)
            if len(im.shape) == 2:
                faz_mask = (im > 0).astype(np.uint8).squeeze()
            else:
                faz_mask = im[:, :, 0].squeeze()
        else:
            faz_mask = None

        layers = av_loader(av_path, threshold)

        def get_layer_color(layer_name):
            if layer_name == "arteries":
                return (1, 0, 0)
            elif layer_name == "veins":
                return (0, 0, 1)
            else:
                return (1, 1, 1)

        layers = {
            key: FazLayer(val, name=key, color=get_layer_color(key))
            for key, val in layers.items()
        }

        image = load_image(image_path) if image_path is not None else None

        return cls(faz_mask, layers, image=image, id=id)
