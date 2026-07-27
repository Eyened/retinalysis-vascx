from __future__ import annotations

import csv
import json

import pandas as pd

from click.testing import CliRunner

from vascx.cli import cli
from vascx.fundus.feature_sets import *  # noqa: F401,F403
from vascx.fundus.feature_sets.macula_centered_rs import CIRCLE_CROPPED_INF, fs_macula_centered_rs
from vascx.fundus.features.base import get_grid_field_tokens
from vascx.fundus.features.caliber import Caliber
from vascx.fundus.features.tortuosity import (
    LengthMeasure,
    Tortuosity,
    TortuosityMeasure,
    TortuosityMode,
)
from vascx.fundus.retina import Retina
from vascx.shared.aggregators import LengthWeightedAggregator
from vascx.shared.features import FeatureSet
from vascx.shared.naming import make_feature_names
from rtnls_enface.grids.circle import CircleField
from rtnls_enface.grids.specifications import (
    CircleGridSpecification,
    DiscCenteredGridSpecification,
    EllipseGridSpecification,
    ETDRSGridSpecification,
    GridFieldSpecification,
    HemifieldGridSpecification,
)
from vascx.utils.feature_docs import write_variable_display_mapping


def _targets(feature):
    return Retina._target_names_for_feature(feature)


def test_structured_canonical_names_match_existing_convention():
    for feature_set in list(FeatureSet._registry.values()):
        names = make_feature_names(feature_set, _targets, "canonical")
        for (feature_index, target_name), item in names.items():
            feature = list(feature_set)[feature_index]
            assert item.name == feature.canonical_name(layer_name=target_name)


def test_resolved_names_are_unique_and_drop_family_constants():
    feature_set = FeatureSet(
        "test_naming_resolved_constants",
        [
            Tortuosity(
                mode=TortuosityMode.Segments,
                measure=TortuosityMeasure.Distance,
                length_measure=LengthMeasure.Splines,
                aggregator=LengthWeightedAggregator(),
            ),
            Tortuosity(
                mode=TortuosityMode.Segments,
                measure=TortuosityMeasure.Distance,
                length_measure=LengthMeasure.Splines,
                max_segment_len=0.2,
                aggregator=LengthWeightedAggregator(),
            ),
        ],
    )
    names = make_feature_names(feature_set, _targets, "resolved")
    assert len({item.name for item in names.values()}) == len(names)
    assert all(not item.name.startswith("lw_") for item in names.values())
    assert any("max_segment_len_0p2" in item.name for item in names.values())


def test_display_annotations_are_trailing_and_human_readable():
    feature_set = FeatureSet(
        "test_naming_display_annotations",
        [
            Tortuosity(
                mode=TortuosityMode.Segments,
                measure=TortuosityMeasure.Distance,
                max_segment_len=0.2,
                spline_error_fraction=0.1,
                aggregator=LengthWeightedAggregator(),
            ),
            Tortuosity(
                mode=TortuosityMode.Segments,
                measure=TortuosityMeasure.Curvature,
                aggregator=LengthWeightedAggregator(),
            ),
        ],
    )
    names = make_feature_names(feature_set, _targets, "canonical")
    annotated = next(item for item in names.values() if "max_segment_len" in item.name)
    assert annotated.display_name.endswith(")")
    assert "max segment len=0.2" in annotated.display_name
    assert "spline error fraction=0.1" in annotated.display_name
    assert annotated.display_name.index("(") > annotated.display_name.index("Tortuosity")


def test_mapping_uses_selected_machine_and_display_names(tmp_path):
    feature_set_name = "full_v3"
    resolved_path = tmp_path / "resolved.csv"
    canonical_path = tmp_path / "canonical.json"

    write_variable_display_mapping(feature_set_name, resolved_path, naming="resolved")
    write_variable_display_mapping(feature_set_name, canonical_path, as_json=True, naming="canonical")

    with resolved_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    resolved_names = make_feature_names(
        FeatureSet.get_by_name(feature_set_name), _targets, "resolved"
    )
    assert {row["variable"] for row in rows} == {item.name for item in resolved_names.values()}
    assert len({row["display_name"] for row in rows}) == len(rows)

    canonical_mapping = json.loads(canonical_path.read_text(encoding="utf-8"))
    canonical_names = make_feature_names(
        FeatureSet.get_by_name(feature_set_name), _targets, "canonical"
    )
    assert set(canonical_mapping) == {item.name for item in canonical_names.values()}


def test_cli_naming_options_default_to_resolved():
    runner = CliRunner()
    for command in ("calc-biomarkers", "write-mapping"):
        result = runner.invoke(cli, [command, "--help"])
        assert result.exit_code == 0, result.output
        assert "--naming [resolved|canonical]" in result.output
        assert "[default: resolved]" in result.output


def test_resolved_grid_name_keeps_identity_and_drops_shared_parameters():
    names = make_feature_names(fs_macula_centered_rs, _targets, "resolved")
    target = next(
        item
        for (feature_index, target_name), item in names.items()
        if target_name == "veins"
        and isinstance(list(fs_macula_centered_rs)[feature_index], Caliber)
        and list(fs_macula_centered_rs)[feature_index].grid_field_spec
        == CIRCLE_CROPPED_INF
    )
    assert target.name == "diam_crcl_inferior_veins"
    assert "multiplier" not in target.name
    assert "center" not in target.name
    assert "radius_multiplier" not in target.name
    assert "band_crop_fraction" not in target.name


def test_grid_name_override_is_generic_and_not_a_parameter():
    specifications = [
        (CircleGridSpecification(center=0.9, band_crop_fraction=0.12, name="crcl"), CircleField.FullGrid),
        (CircleGridSpecification(multiplier=0.9, band_crop_fraction=0.12, name="crcl"), CircleField.FullGrid),
        (DiscCenteredGridSpecification(multiplier=7 / 6, band_crop_fraction=0.06, name="crcl"), None),
        (EllipseGridSpecification(name="ellipse"), None),
        (ETDRSGridSpecification(name="etdrs_custom"), None),
        (HemifieldGridSpecification(name="hemifield"), None),
    ]
    for spec, field in specifications:
        if field is None:
            assert spec.name
            assert "name" not in spec.init_kwargs()
            continue
        field_spec = GridFieldSpecification(spec, field)
        tokens = get_grid_field_tokens(field_spec)
        assert tokens[:1] == [spec.name]
        assert "name" not in tokens


def test_calc_biomarkers_writes_matching_json_mapping(tmp_path, monkeypatch):
    input_path = tmp_path / "segmentations"
    input_path.mkdir()
    output_csv = tmp_path / "biomarkers.csv"

    monkeypatch.setattr("vascx.cli.make_examples", lambda _path: [{"id": "sample"}])
    monkeypatch.setattr(
        "vascx.cli.extract_in_parallel",
        lambda **_kwargs: pd.DataFrame({"dummy": [1.0]}, index=["sample"]),
    )

    result = CliRunner().invoke(
        cli,
        [
            "calc-biomarkers",
            str(input_path),
            str(output_csv),
            "--feature_set",
            "full_v3",
            "--naming",
            "resolved",
        ],
    )
    assert result.exit_code == 0, result.output

    names_json = output_csv.with_suffix(".names.json")
    assert output_csv.exists()
    assert names_json.exists()
    expected = make_feature_names(
        FeatureSet.get_by_name("full_v3"), _targets, "resolved"
    )
    mapping = json.loads(names_json.read_text(encoding="utf-8"))
    assert set(mapping) == {item.name for item in expected.values()}
