"""CLI samples determine both extraction rows and report figure contents."""
import json
import random
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
        (1, 3, 1, 1, 1, []),
        (2, 10, 2, 2, 2, []),
        (4, 4, 4, 4, 4, []),
        (24, None, 24, 3, 4, ["--seed", "1"]),
        (4, 100, 4, 4, 4, []),
        (24, 10, 10, 2, 2, ["--seed", "1", "--report-image", "image_02", "--report-image", "image_04"]),
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
    assert len(extracted_ids) == len(set(extracted_ids)) == extracted
    assert set(extracted_ids).issubset(ids)
    if sample is None or sample >= available:
        assert extracted_ids == ids
    else:
        assert extracted_ids != ids[:extracted]
    assert pd.read_csv(output / "biomarkers.csv")["id"].tolist() == extracted_ids
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
    manifest = json.loads((output / "report.json").read_text())
    requested_ids = [
        extra_args[i + 1] for i, arg in enumerate(extra_args) if arg == "--report-image"
    ]
    assert manifest["dataset"]["report_sample_ids"] == (
        requested_ids or extracted_ids[:max(paired_images, other_images)]
    )


def test_sampling_help_and_removed_report_samples_option(tmp_path):
    runner = CliRunner()
    help_result = runner.invoke(cli, ["calc-biomarkers", "--help"])
    assert help_result.exit_code == 0
    help_text = " ".join(help_result.output.split())
    assert "random sample of up to N available images" in help_text
    assert "--seed INTEGER" in help_text
    assert "ignored without --sample. [default: 0]" in help_text
    assert "maximum 16 images per plot" in help_text
    assert "--report-samples" not in help_text

    result = runner.invoke(cli, [
        "calc-biomarkers", str(tmp_path), str(tmp_path / "report"),
        "--feature-set", "od_centered", "--report-samples", "3",
    ])
    assert result.exit_code != 0
    assert "No such option: --report-samples" in result.output


@pytest.fixture
def sampling_cli(tmp_path, monkeypatch):
    """Capture the selected inputs without running image processing."""
    ids = [f"image_{i:02d}" for i in range(40)]
    monkeypatch.setattr(
        "vascx.cli.make_examples", lambda _path, **kwargs: [{"id": i} for i in ids],
    )
    calls = []

    def extract(*, retinas, **kwargs):
        selected = [image["id"] for image in retinas]
        calls.append(selected)
        return pd.DataFrame({"value": 1.0}, index=selected)

    monkeypatch.setattr("vascx.cli.extract_biomarkers_to_folder", extract)
    runner = CliRunner()

    def run(*options):
        return runner.invoke(cli, [
            "calc-biomarkers", str(tmp_path), str(tmp_path / "report"),
            "--feature-set", "od_centered", "--sample", "8", *options,
        ])

    return run, ids, calls


def test_random_sample_is_reproducible_and_seed_does_not_change_global_rng(sampling_cli):
    run, ids, calls = sampling_cli
    global_state = random.getstate()
    for options in [(), (), ("--seed", "0"), ("--seed", "1"), ("--seed", "1")]:
        result = run(*options)
        assert result.exit_code == 0, result.output
        assert random.getstate() == global_state

    assert calls[0] == calls[1] == calls[2]
    assert calls[3] == calls[4]
    assert set(calls[0]) != set(calls[3])
    for selected in calls:
        assert len(selected) == len(set(selected)) == 8
        assert set(selected).issubset(ids)
        assert set(selected) != set(ids[:8])


def test_report_image_must_belong_to_random_sample(sampling_cli):
    run, ids, calls = sampling_cli
    result = run()
    assert result.exit_code == 0, result.output
    unselected = next(image_id for image_id in ids if image_id not in calls[0])

    result = run("--report-image", unselected)
    assert result.exit_code != 0
    assert f"Unknown --report-image ID(s): {unselected}" in result.output
    assert len(calls) == 1
