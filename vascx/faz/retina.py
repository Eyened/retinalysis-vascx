from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
from vascx.faz.features.base import FAZLayerFeature
from vascx.faz.layer import FAZLayer
from vascx.shared.features import FeatureSet
from vascx.utils import load_av_segmentation, load_image

from rtnls_enface.faz_enface import FAZEnface
from rtnls_enface.utils.data_loading import open_binary_mask, open_mask


class FAZRetina(FAZEnface):
    @property
    def arteries(self) -> FAZLayer:
        return self.layers["arteries"]

    @property
    def veins(self) -> FAZLayer:
        return self.layers["veins"]
    
    @property
    def vessels(self) -> FAZLayer:
        return self.layers["vessels"]

    def set_retina(self, retina):
        self.retina = retina

    def calc_features(self, feature_set: FeatureSet):
        all_features = {}
        for feature in feature_set:
            if not isinstance(feature, FAZLayerFeature):
                continue
            for layer_name, layer in self.layers.items():
                res = feature.compute(layer)
                feature_name = feature.canonical_name(layer_name=layer_name)
                all_features[feature_name] = res

        return all_features

    @classmethod
    def from_file(
        cls,
        av_path: Optional[Union[str, Path]] = None,
        vessels_path: Optional[Union[str, Path]] = None,
        faz_path: Optional[Union[str, Path]] = None,
        image_path: Optional[Union[str, Path]] = None,
        threshold=0.5,
        id: Any = None,
        av_loader=load_av_segmentation,
    ):
        assert (
            av_path is not None or vessels_path is not None
        ), "Either av_path or vessels_path must be provided"


        if faz_path is not None:
            im = open_mask(faz_path)
            if len(im.shape) == 2:
                faz_mask = (im > 0).astype(np.uint8).squeeze()
            else:
                faz_mask = im[:, :, 0].squeeze()
        else:
            faz_mask = None

        layers = {}
        if av_path is not None:
            av_layers = av_loader(av_path, threshold)

            def get_layer_color(layer_name):
                if layer_name == "arteries":
                    return (1, 0, 0)
                elif layer_name == "veins":
                    return (0, 0, 1)
                else:
                    return (1, 1, 1)

            for key, val in av_layers.items():
                layers[key] = FAZLayer(val, name=key, color=get_layer_color(key))

        if vessels_path is not None:
            layers["vessels"] = FAZLayer(open_binary_mask(vessels_path))

        image = load_image(image_path) if image_path is not None else None

        return cls(faz_mask, layers, image=image, id=id)
