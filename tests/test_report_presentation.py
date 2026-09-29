"""Reporting should preserve configurations, missing samples, and captions."""
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from rtnls_enface.grids.circle import CircleField
from rtnls_enface.grids.specifications import CircleGridSpecification, GridFieldSpecification
from vascx.fundus.features.caliber import Caliber
from vascx.shared.aggregators import median, LengthWeightedAggregator
from vascx.shared.features import Feature, FeatureSet
from vascx.utils.feature_docs import write_feature_set_readme, get_biomarker_definitions
from vascx.utils.biomarker_report import _render_feature_panel, render_biomarker_plots


class ExampleFeature(Feature):
    def compute(self, layer):
        raise AssertionError("Stored missing values must not be recomputed")

    def _plot(self, ax, layer, **kwargs):
        """Colored pixels show the test image, including missing measurements."""
        ax.imshow(layer.image)
        return ax

    def name_tokens(self, **kwargs):
        return ["example"]

    def display_name(self, **kwargs):
        return "Example"


def test_readme_groups_regions_and_layers_but_separates_parameters(tmp_path):
    grid = CircleGridSpecification(band_crop_fraction=0.12)
    full = GridFieldSpecification(grid, CircleField.FullGrid)
    superior = GridFieldSpecification(grid, CircleField.Superior)
    fs = FeatureSet("report_grouping_test", [
        Caliber(grid_field=full, aggregator=median),
        Caliber(grid_field=superior, aggregator=median),
        Caliber(grid_field=full, aggregator=LengthWeightedAggregator()),
        Caliber(grid_field=full, aggregator=median, min_numpoints=20),
    ])
    for naming in ("canonical", "resolved"):
        path = write_feature_set_readme(fs, tmp_path / f"{naming}.md", naming=naming)
        text = path.read_text()
        assert text.count("\n### ") == 3
        assert text.count(Caliber.general_description) == 3
        assert "superior" in text.lower()
        assert "band crop fraction=0.12" in text
        for item in get_biomarker_definitions(fs, naming=naming):
            assert f"`{item.variable}`" in text


@pytest.mark.parametrize("value", [None, float("nan"), np.float64("nan")])
def test_missing_measurement_still_draws_image_and_docstring(value):
    feature = ExampleFeature()
    target = SimpleNamespace(image=np.zeros((10, 10)))
    fig, ax = plt.subplots(layout="constrained")
    try:
        result = _render_feature_panel(ax, feature=feature, target=target,
            image_id="sample", variable="example",
            dataframe=pd.DataFrame({"example": [value]}, index=["sample"]))
        assert result
        assert len(ax.images) == 1
        assert any(t.get_text() == "N/A" for t in ax.texts)
        assert any(t.get_gid() == "biomarker-caption" and "Colored pixels" in t.get_text()
                   for t in ax.texts)
        fig.canvas.draw()
    finally:
        plt.close(fig)


def test_standalone_caption_can_be_disabled():
    feature = ExampleFeature()
    target = SimpleNamespace(image=np.zeros((10, 10)))
    fig = feature.plot_figure(target, computed_value=None, plot_caption=False)
    try:
        assert not any(t.get_gid() == "biomarker-caption" for t in fig.axes[0].texts)
    finally:
        plt.close(fig)


def test_failed_overlay_keeps_image_and_distinct_failure_label():
    class Broken(ExampleFeature):
        def _plot(self, ax, layer, **kwargs):
            """An overlay that cannot be generated."""
            raise ValueError("missing geometry")
    fig, ax = plt.subplots()
    try:
        with pytest.warns(UserWarning, match="missing geometry"):
            assert _render_feature_panel(ax, feature=Broken(),
                target=SimpleNamespace(image=np.zeros((10, 10))), image_id="s",
                variable="example", dataframe=pd.DataFrame({"example": [np.nan]}, index=["s"]))
        assert len(ax.images) == 1
        assert any("N/A — visualization unavailable" in t.get_text() for t in ax.texts)
        assert not any(t.get_gid() == "biomarker-caption" for t in ax.texts)
    finally:
        plt.close(fig)


@pytest.mark.parametrize("sample_count", [1, 2, 5])
@pytest.mark.parametrize("paired", [True, False])
@pytest.mark.parametrize("sample_size", [None, 2])
def test_all_nan_composite_is_saved_and_linked(tmp_path, monkeypatch, sample_count, paired, sample_size):
    from types import MethodType
    from vascx.fundus.features.sparsity import Sparsity
    feature = Caliber(plot=True) if paired else Sparsity(plot=True)
    monkeypatch.setattr(feature, "_plot", MethodType(ExampleFeature._plot, feature))
    fs = FeatureSet(f"all_nan_report_test_{sample_count}_{paired}_{sample_size}", [feature])
    target = SimpleNamespace(image=np.zeros((10, 10)))
    targets = [("arteries", target), ("veins", target)] if paired else [("vessels", target)]
    retina = SimpleNamespace(feature_targets=lambda _feature: targets)
    variables = [item.variable for item in get_biomarker_definitions(fs)]
    sample_ids = [f"sample_{i}" for i in range(sample_count)]
    frame = pd.DataFrame({variable: [np.nan] * sample_count for variable in variables},
                         index=sample_ids)
    from matplotlib.figure import Figure
    original_savefig = Figure.savefig
    saved_captions = []

    def check_caption(fig, *args, **kwargs):
        captions = [t for t in fig.texts if t.get_gid() == "biomarker-caption"]
        assert len(captions) == 1
        caption = " ".join(captions[0].get_text().split())
        assert "Colored pixels" in caption
        explanation = ("The left column displays the explanatory plots for arteries, "
                       "and the right column for veins.")
        assert (explanation in caption) == paired
        limit = sample_size if sample_size is not None else (3 if paired else 4)
        expected_panels = min(sample_count, limit) * (2 if paired else 1)
        assert sum(len(ax.images) for ax in fig.axes) == expected_panels
        assert not any(t.get_gid() == "biomarker-caption"
                       for ax in fig.axes for t in ax.texts)
        saved_captions.append(captions[0].get_text())
        return original_savefig(fig, *args, **kwargs)

    monkeypatch.setattr(Figure, "savefig", check_caption)
    paths = render_biomarker_plots([(sample_id, retina) for sample_id in sample_ids],
                                   fs, tmp_path, dataframe=frame, sample_size=sample_size)
    assert len(saved_captions) == 1
    assert set(paths) == set(variables)
    assert len(list((tmp_path / "plots").glob("*.png"))) == 1
    for links in paths.values():
        assert (tmp_path / links[0][1]).is_file()


def test_retina_feature_standalone_figure_uses_docstring_caption():
    from vascx.fundus.features.base import RetinaFeature

    class RetinaExample(RetinaFeature):
        def display_name(self, **kwargs):
            return "Retina example"

        def compute(self, retina):
            return None

        def _plot(self, ax, retina, **kwargs):
            """The grayscale image shows the entire retina."""
            ax.imshow(retina.image, cmap="gray")
            return ax

    fig = RetinaExample().plot_figure(SimpleNamespace(image=np.zeros((10, 10))))
    try:
        assert len(fig.axes[0].images) == 1
        assert any(t.get_gid() == "biomarker-caption" for t in fig.axes[0].texts)
        assert any(t.get_text() == "N/A" for t in fig.axes[0].texts)
        fig.canvas.draw()
    finally:
        plt.close(fig)
