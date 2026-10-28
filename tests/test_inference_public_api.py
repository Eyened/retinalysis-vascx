"""VascX uses public inference methods while preserving its output contracts."""

import numpy as np
import pytest
import torch
from PIL import Image
from rtnls_inference.ensembles import (
    SegmentationEnsemble,
)
from vascx.inference import inference


class Members(torch.nn.Module):
    def __init__(self, kind):
        super().__init__()
        self.kind = kind
        self.anchor = torch.nn.Parameter(torch.zeros(()), requires_grad=False)

    def forward(self, image):
        count, _, height, width = image.shape
        if self.kind == "quality":
            values = torch.tensor([1.0, 3.0, -2.0], device=image.device) + self.anchor
            return values[None, None].expand(2, count, -1)
        channels = 1 if self.kind == "fovea" else 3
        values = torch.zeros(2, count, channels, height, width, device=image.device)
        if self.kind == "fovea":
            values[:, :, 0, 1, 2] = 10
        else:
            values[:, :, 0] = 10
        return values + self.anchor


class RefinedAV(SegmentationEnsemble):
    def postprocess_item(self, item):
        assert self._inference_setting("return_postprocess_intermediates", False)
        result = super().postprocess_item(item)
        # Deliberately differ from argmax, to catch accidental re-decoding in VascX.
        result["output"][:] = 2
        return result


@pytest.fixture
def setup(tmp_path, monkeypatch):
    paths = []
    for index in range(2):
        path = tmp_path / f"sample_{index}.png"
        Image.fromarray(np.full((8, 8, 3), 100 + index, dtype=np.uint8)).save(path)
        paths.append(path)
    calls = []

    def load(cls, model, **kwargs):
        config = {
            "datamodule": {
                "test_transform": {
                    "square_size": 8,
                    "resize": 4,
                    "contrast_enhance": False,
                    "normalize": "zero-one",
                }
            },
            "lightningmodule": {"n_class": 1 if model == "fovea" else 3},
            "inference": {"tracing_input_size": [4, 4]},
        }
        if model == "quality":
            config["inference"]["classification"] = {
                "task": "multiclass",
                "label_map": {0: "good", 1: "usable", 2: "bad"},
            }
        ensemble = (RefinedAV if model == "av" else cls)(
            Members(model), config, **kwargs
        )
        public_loader = ensemble.make_inference_dataloader

        def loader(data, **options):
            calls.append((model, options.copy()))
            options.update(num_workers=0, batch_size=1)
            return public_loader(data, **options)

        monkeypatch.setattr(ensemble, "make_inference_dataloader", loader)
        predict = ensemble.predict_batch

        def production_prediction(batch, *, fields=None):
            assert fields is None
            return predict(batch)

        monkeypatch.setattr(ensemble, "predict_batch", production_prediction)
        return ensemble

    monkeypatch.setattr(inference, "_load_ensemble", load)
    return paths, calls


@pytest.mark.parametrize("kind", ["quality", "fovea"])
@pytest.mark.parametrize("empty", [False, True])
def test_numeric_outputs_preserve_vascx_columns_and_values(setup, kind, empty):
    paths, calls = setup
    paths = [] if empty else paths
    if kind == "quality":
        result = inference.run_quality_estimation(paths, model=kind, device="cpu")
        columns = ["q1", "q2", "q3"]
        expected = [[1, 3, -2]] * len(paths)
    else:
        result = inference.run_fovea_detection(paths, model=kind, device="cpu")
        columns = ["x_fovea", "y_fovea"]
        expected = [[5, 3]] * len(paths)
    assert list(result) == columns
    assert result.index.tolist() == [path.stem for path in paths]
    assert result.index.name is None
    if paths:
        np.testing.assert_array_equal(result.to_numpy(), expected)
    assert calls and all(
        options["input_space"] == "preprocessed" and options["ignore_exceptions"]
        for _, options in calls
    )


def test_numeric_iterators_keep_their_existing_result_fields(setup):
    paths, _ = setup
    data = inference._create_inputs(paths)
    quality = list(
        inference.iterate_quality_estimation(data, device="cpu", model="quality")
    )
    fovea = list(inference.iterate_fovea_detection(data, device="cpu", model="fovea"))
    assert set(quality[0]) == {"id", "logits"}
    np.testing.assert_array_equal(quality[0]["logits"], [1, 3, -2])
    np.testing.assert_array_equal(fovea[0]["keypoints"], [[5, 3]])


@pytest.mark.parametrize(
    "predict_av,predict_vessels", [(True, True), (True, False), (False, True)]
)
def test_segmentation_callbacks_preserve_probabilities_and_use_final_masks(
    setup, tmp_path, predict_av, predict_vessels
):
    paths, _ = setup
    items = []
    inference.run_segmentation_vessels_and_av(
        paths,
        device="cpu",
        callback=items.append,
        predict_av=predict_av,
        predict_vessels=predict_vessels,
        av_model="av",
        vessels_model="vessels",
    )
    assert [item["id"] for item in items] == [path.stem for path in paths]
    for item in items:
        if predict_av:
            assert np.all(item["av"]["refined_mask"] == 2)
            assert np.all(item["av"]["image"].argmax(-1) == 0)
            assert item["av"]["image"].shape == (8, 8, 3)
        else:
            assert item["av"] is None
        if predict_vessels:
            assert item["vessels"]["image"].shape == (8, 8, 3)
        else:
            assert item["vessels"] is None
    av_path = tmp_path / "av" if predict_av else None
    vessels_path = tmp_path / "vessels" if predict_vessels else None
    inference.run_segmentation_vessels_and_av(
        paths,
        device="cpu",
        av_path=av_path,
        vessels_path=vessels_path,
        av_model="av",
        vessels_model="vessels",
    )
    for path in paths:
        if av_path:
            assert np.all(np.asarray(Image.open(av_path / path.name)) == 2)
        if vessels_path:
            assert np.all(np.asarray(Image.open(vessels_path / path.name)) == 0)


def test_disc_callbacks_and_exports_use_public_predictions(setup, tmp_path):
    paths, _ = setup
    items = []
    inference.run_segmentation_disc(
        paths, device="cpu", callback=items.append, model="disc"
    )
    assert len(items) == len(paths)
    assert items[0]["image"].shape == (8, 8, 3)
    output = tmp_path / "disc"
    inference.run_segmentation_disc(
        paths, device="cpu", output_path=output, model="disc"
    )
    np.testing.assert_array_equal(
        np.asarray(Image.open(output / paths[0].name)), items[0]["output"]
    )


def test_specializing_a_release_preserves_requested_intermediates(monkeypatch):
    loaded = SegmentationEnsemble(
        Members("av"), {"datamodule": {}}, return_postprocess_intermediates=True
    )
    monkeypatch.setattr(inference, "get_ensemble_class", lambda config: RefinedAV)
    specialized = inference._specialize_ensemble(loaded, SegmentationEnsemble)
    assert isinstance(specialized, RefinedAV)
    assert specialized._inference_setting("return_postprocess_intermediates", False)
