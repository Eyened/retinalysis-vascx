from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

from typing import Any, Iterator

import matplotlib.pyplot as plt
import pytest

from tests.regression_helpers import SAMPLES_DIR
import vascx.fundus.feature_sets  # Register feature sets.
from vascx.shared.features import FeatureSet
from vascx.fundus.features.base import LayerFeature, RetinaFeature, VesselsLayerFeature
from vascx.fundus.loader import RetinaLoader
from vascx.fundus.retina import Retina


def iter_feature_targets(retina: Retina, feature_set: FeatureSet) -> Iterator[tuple[str, Any, Any]]:
    """Yield the same feature-target pairs used in the plotting notebook."""
    for feature in feature_set:
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
@pytest.mark.parametrize("feature_set_name", ["macula_centered", "od_centered"])
def test_feature_plotting_smoke(feature_set_name, tmp_path) -> None:
    """Render each feature plot and ensure the target axis is not left blank."""
    loader = RetinaLoader.from_folder(SAMPLES_DIR)
    assert len(loader) > 0, f"No retinas found in {SAMPLES_DIR}"
    retina = loader[0]

    feature_set = FeatureSet.get_by_name(feature_set_name)
    for index, (title, feature, target) in enumerate(iter_feature_targets(retina, feature_set)):
        fig, ax = plt.subplots(figsize=(7, 7), dpi=150)
        try:
            if isinstance(feature, RetinaFeature):
                feature.plot(ax=ax, retina=target)
            else:
                feature.plot(ax=ax, layer=target)
            ax.set_title(title)
            fig.tight_layout()
            fig.savefig(tmp_path / f"{feature_set_name}_{index:03d}.png")

            assert axis_has_drawn_content(ax), f"Plot for '{title}' left the axis blank."
        finally:
            plt.close(fig)


def test_compound_bifurcation_plot_single_image_transparent_markers_and_labels(tmp_path):
    """One image retains every compact measurement label and uncapped values."""
    from dataclasses import replace
    import numpy as np
    from matplotlib.patches import Arc
    from vascx.fundus.features.bifurcation import Bifurcation, _BifurcationLabels
    retina = RetinaLoader.from_folder(SAMPLES_DIR)[0]
    feature = Bifurcation(min_bifurcations=1)
    records = feature._measurements(retina.arteries)
    assert len(records) > 1
    records = tuple(replace(r, branching_coefficient=3., jed=1.5) for r in records)
    records = (replace(records[0], branching_coefficient=np.nan, jed=np.nan), *records[1:])
    feature._measurements = lambda layer: records
    values = feature.compute(retina.arteries)
    assert values[1] == 3 and values[2] == 1.5
    fig = feature.plot_figure(retina.arteries, computed_value=values, plot_caption=False)
    try:
        assert tuple(fig.get_size_inches()) == feature.preferred_plot_size
        ax = fig.axes[0]
        assert not ax.child_axes
        assert not ax.axison
        assert any(t.get_gid() == "bifurcation-summary" for t in ax.texts)
        panel = ax
        assert len(panel.images) == 1
        assert len(panel.lines) == 2*len(records)
        arcs = [p for p in panel.patches if isinstance(p, Arc)]
        assert len(arcs) == len(records) and all(not arc.get_fill() for arc in arcs)
        for record, arc in zip(records, arcs):
            assert (arc.theta2 - arc.theta1) % 360 == pytest.approx(record.angle)
        for i, record in enumerate(records):
            for ray in panel.lines[2*i:2*i+2]:
                np.testing.assert_allclose([ray.get_xdata()[0], ray.get_ydata()[0]],
                                           record.node.position.tuple_xy)
        labels = next(a for a in panel.artists if isinstance(a, _BifurcationLabels))
        assert [label.get_text() for label in labels.labels] == [
            f'{r.angle:.0f}° | N/A | N/A' if i == 0 else f'{r.angle:.0f}° | 3.00 | 1.50'
            for i, r in enumerate(records)]
        for size, scale in (((8, 8), 1), ((4, 4), 1), ((8, 8), 3)):
            fig.set_size_inches(*size)
            for label in labels.labels:
                label.set_fontsize(7.2 * scale)
            for text in ax.texts:
                if text.get_gid() == "bifurcation-summary":
                    text.set_fontsize(8 * scale)
            fig.canvas.draw()
            assert len(labels.drawn_extents) == len(records)
            for i, bbox in enumerate(labels.drawn_extents):
                assert not any(bbox.overlaps(other) for other in labels.drawn_extents[i+1:])
        fig.savefig(tmp_path/'bifurcation_labels.png')
    finally:
        plt.close(fig)
