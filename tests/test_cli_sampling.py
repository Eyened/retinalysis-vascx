"""CLI samples determine both extraction rows and report figure contents."""
from pathlib import Path
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")
from matplotlib.figure import Figure
import numpy as np
import pandas as pd
import pytest
from click.testing import CliRunner

from vascx.cli import cli
from vascx.fundus.features.caliber import Caliber
from vascx.fundus.features.luminance import Luminance
from vascx.shared.features import FeatureSet
from vascx.utils.feature_docs import get_biomarker_definitions


@pytest.mark.filterwarnings("error:Could not render report plot")
@pytest.mark.parametrize(
    "available,sample,extracted,paired_images,other_images,extra_args",
    [
        (24, None, 24, 3, 4, []),
        (24, 1, 1, 1, 1, []),
        (24, 10, 10, 10, 10, []),
        (24, 16, 16, 16, 16, []),
        (24, 20, 20, 16, 16, []),
        (2, 10, 2, 2, 2, []),
        (4, 100, 4, 4, 4, []),
        (24, 10, 10, 2, 2, ["--report-image", "image_02", "--report-image", "image_04"]),
        (24, 10, 10, 0, 0, ["--no-report"]),
    ],
)
def test_sample_controls_extracted_rows_and_plot_panels(
    tmp_path, monkeypatch, available, sample, extracted,
    paired_images, other_images, extra_args,
):
    monkeypatch.setattr(FeatureSet, "_registry", dict(FeatureSet._registry))
    caliber, luminance = Caliber(plot=True), Luminance(plot=True)
    feature_set = FeatureSet("cli_sampling_test", [caliber, luminance])
    image = np.zeros((2, 2))
    target = SimpleNamespace(image=image)
    retina = SimpleNamespace(feature_targets=lambda feature: (
        [("arteries", target), ("veins", target)] if feature is caliber
        else [("retina", target)]
    ))
    def plot(ax, target, **kwargs):
        ax.imshow(target.image)
        return ax

    for feature in feature_set:
        monkeypatch.setattr(feature, "_plot", plot)

    ids = [f"image_{i:02d}" for i in range(available)]
    monkeypatch.setattr("vascx.cli.make_examples", lambda _path, **kwargs: [{"id": i} for i in ids])
    extracted_ids = []

    def extract(examples, **kwargs):
        extracted_ids.extend(example["id"] for example in examples)
        return pd.DataFrame(
            {item.variable: 1.0 for item in get_biomarker_definitions(feature_set)},
            index=extracted_ids,
        )

    # Exercise the actual CLI, report selection, and figure assembly while
    # avoiding unrelated image processing and large PNG encodes.
    monkeypatch.setattr("vascx.utils.analysis.extract_in_parallel", extract)
    monkeypatch.setattr("vascx.utils.analysis._load_retina_input", lambda *args, **kwargs: retina)
    saved_panels = {}

    def savefig(fig, filename, **kwargs):
        saved_panels[Path(filename).stem] = sum(len(ax.images) for ax in fig.axes)
        Path(filename).touch()

    monkeypatch.setattr(Figure, "savefig", savefig)
    output = tmp_path / "report"
    args = [
        "calc-biomarkers", str(tmp_path), str(output),
        "--feature-set", feature_set.name, "--no-use-fovea",
    ]
    if sample is not None:
        args += ["--sample", str(sample)]
    result = CliRunner().invoke(cli, args + extra_args)
    assert result.exit_code == 0, result.output
    assert extracted_ids == ids[:extracted]
    assert pd.read_csv(output / "biomarkers.csv")["id"].tolist() == ids[:extracted]
    assert (output / "data_dictionary.csv").is_file()
    if "--no-report" in extra_args:
        assert saved_panels == {}
        assert not (output / "README.md").exists()
        return
    assert saved_panels == {
        "diam_arteries": 2 * paired_images,
        "luminance": other_images,
    }
    assert (output / "README.md").is_file()
    assert (output / "report.json").is_file()


def test_sampling_help_and_removed_report_samples_option(tmp_path):
    runner = CliRunner()
    help_result = runner.invoke(cli, ["calc-biomarkers", "--help"])
    assert help_result.exit_code == 0
    help_text = " ".join(help_result.output.split())
    assert "first N available images" in help_text
    assert "maximum 16 images per plot" in help_text
    assert "--report-samples" not in help_text

    result = runner.invoke(cli, [
        "calc-biomarkers", str(tmp_path), str(tmp_path / "report"),
        "--feature-set", "od_centered", "--report-samples", "3",
    ])
    assert result.exit_code != 0
    assert "No such option: --report-samples" in result.output
