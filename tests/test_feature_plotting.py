from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

from typing import Any, Iterator

import matplotlib.pyplot as plt
import pytest

from tests.regression_helpers import SAMPLES_DIR
from vascx.fundus.feature_sets.full_v2 import fs_full_v2
from vascx.fundus.features.base import LayerFeature, RetinaFeature, VesselsLayerFeature
from vascx.fundus.loader import RetinaLoader
from vascx.fundus.retina import Retina


def iter_feature_targets(retina: Retina) -> Iterator[tuple[str, Any, Any]]:
    """Yield the same feature-target pairs used in the plotting notebook."""
    for feature in fs_full_v2:
        if isinstance(feature, LayerFeature):
            for layer in (retina.arteries, retina.veins):
                yield feature.display_name(layer_name=layer.name), feature, layer
        elif isinstance(feature, VesselsLayerFeature):
            yield feature.display_name(layer_name=retina.vessels.name), feature, retina.vessels
        elif isinstance(feature, RetinaFeature):
            yield feature.display_name(), feature, retina


def axis_has_drawn_content(ax: plt.Axes) -> bool:
    """Return whether plotting added visible artists beyond annotation text."""
    return any(
        (
            len(ax.images) > 0,
            len(ax.lines) > 0,
            len(ax.collections) > 0,
            len(ax.patches) > 0,
            len(ax.artists) > 0,
        )
    )


@pytest.mark.plotting
def test_feature_plotting_smoke() -> None:
    """Render each feature plot and ensure the target axis is not left blank."""
    loader = RetinaLoader.from_folder(SAMPLES_DIR)
    assert len(loader) > 0, f"No retinas found in {SAMPLES_DIR}"
    retina = loader[0]

    for title, feature, target in iter_feature_targets(retina):
        fig, ax = plt.subplots(figsize=(7, 7), dpi=150)
        try:
            if isinstance(feature, RetinaFeature):
                feature.plot(ax=ax, retina=target)
            else:
                feature.plot(ax=ax, layer=target)
            ax.set_title(title)
            fig.tight_layout()

            assert axis_has_drawn_content(ax), f"Plot for '{title}' left the axis blank."
        finally:
            plt.close(fig)
