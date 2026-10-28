import os
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, TypeVar

import numpy as np
import pandas as pd
import torch
from PIL import Image
from rtnls_inference.inputs import ImageInput
from rtnls_inference.ensembles import get_ensemble_class
from rtnls_inference.ensembles.ensemble_classification import ClassificationEnsemble
from rtnls_inference.ensembles.ensemble_heatmap_regression import (
    HeatmapRegressionEnsemble,
)
from rtnls_inference.ensembles.ensemble_segmentation import SegmentationEnsemble
from tqdm import tqdm

from vascx.inference.device import resolve_device
from vascx.inference.model_config import (
    DEFAULT_AV_MODEL,
    DEFAULT_DISC_MODEL,
    DEFAULT_FOVEA_MODEL,
    DEFAULT_QUALITY_MODEL,
    DEFAULT_VESSELS_MODEL,
)

EnsembleT = TypeVar("EnsembleT")


def _load_ensemble(
    ensemble_cls: type[EnsembleT],
    model: str | Path,
    **kwargs,
) -> EnsembleT:
    """Load an ensemble from a HuggingFace string, local release, or release file."""
    model_path = Path(model).expanduser()
    if model_path.exists():
        if model_path.is_dir():
            from rtnls_inference.ensembles import make_ensemble

            return make_ensemble(model_path, **kwargs)
        if model_path.suffix.lower() == ".onnx":
            loaded = ensemble_cls.from_onnx(model_path, **kwargs)
        else:
            loaded = ensemble_cls.from_torchscript(model_path, **kwargs)
        return _specialize_ensemble(loaded, ensemble_cls)

    model_str = str(model)
    if model_str.startswith("hf@"):
        loaded = ensemble_cls.from_modelstring(model_str, **kwargs)
        return _specialize_ensemble(loaded, ensemble_cls)
    if ":" in model_str:
        loaded = ensemble_cls.from_huggingface(model_str, **kwargs)
        return _specialize_ensemble(loaded, ensemble_cls)
    loaded = ensemble_cls.from_modelstring(model_str, **kwargs)
    return _specialize_ensemble(loaded, ensemble_cls)


def _specialize_ensemble(loaded: EnsembleT, fallback: type[EnsembleT]) -> EnsembleT:
    """Honor embedded inference classes, retaining legacy explicit-class fallback."""
    try:
        resolved = get_ensemble_class(loaded.config)
    except (ImportError, ValueError, KeyError):
        return loaded
    if resolved is fallback or isinstance(loaded, resolved):
        return loaded
    return resolved(
        loaded.ensemble, loaded.config, loaded.fpath, **loaded._inference_overrides
    )


def _segmentation_item(result: dict[str, Any]) -> dict[str, Any]:
    """Keep probability-map callbacks and expose the final mask for saving."""
    return {
        "id": result.get("id"),
        "image": result["probabilities"],
        "output": result["output"],
    }


def _create_inputs(
    rgb_paths: List[Path],
    ids: Optional[List[str]] = None,
) -> List[ImageInput]:
    """Create input records from paths; the inference dataset validates them."""
    if ids is None:
        ids = [p.stem for p in rgb_paths]

    if len(rgb_paths) != len(ids):
        raise ValueError("rgb_paths and ids must have the same length")

    return [
        {"id": str(id_val), "image": str(rgb_path)}
        for id_val, rgb_path in zip(ids, rgb_paths)
    ]


def iterate_quality_estimation(
    data: List[ImageInput],
    device: torch.device | None = None,
    model: str | Path = DEFAULT_QUALITY_MODEL,
) -> Iterator[Dict[str, Any]]:
    """Yield quality logits through the public inference workflow."""
    ensemble = (
        _load_ensemble(ClassificationEnsemble, model).to(resolve_device(device)).eval()
    )
    loader = ensemble.make_inference_dataloader(
        data,
        input_space="preprocessed",
        batch_size=16,
        num_workers=8,
        ignore_exceptions=True,
    )
    for item in ensemble.iter_predictions(loader):
        yield {"id": item.get("id"), "logits": item["output"]}


def run_quality_estimation(
    fpaths,
    ids: Optional[List[str]] = None,
    device: torch.device | None = None,
    model: str | Path = DEFAULT_QUALITY_MODEL,
):
    ensemble = (
        _load_ensemble(ClassificationEnsemble, model).to(resolve_device(device)).eval()
    )
    loader = ensemble.make_inference_dataloader(
        _create_inputs(fpaths, ids=ids),
        input_space="preprocessed",
        batch_size=16,
        num_workers=8,
        ignore_exceptions=True,
    )
    table = ensemble.predict_dataloader(loader).set_index("id")
    result = table.filter(regex="^logit_").astype(float)
    if result.empty:
        result = pd.DataFrame(index=table.index, columns=["q1", "q2", "q3"])
    else:
        result.columns = ["q1", "q2", "q3"]
    result.index.name = None
    return result


def iterate_segmentation_vessels_and_av(
    data: List[ImageInput],
    device: torch.device | None = None,
    predict_av: bool = True,
    predict_vessels: bool = True,
    av_model: str | Path = DEFAULT_AV_MODEL,
    vessels_model: str | Path = DEFAULT_VESSELS_MODEL,
) -> Iterator[Dict[str, Any]]:
    """Yield final masks and probability maps, keeping both models on the same batch."""
    device = resolve_device(device)
    if not predict_av and not predict_vessels:
        return
    ensemble_av = (
        _load_ensemble(
            SegmentationEnsemble, av_model, return_postprocess_intermediates=True
        )
        .to(device)
        .eval()
        if predict_av
        else None
    )
    ensemble_vessels = (
        _load_ensemble(SegmentationEnsemble, vessels_model).to(device).eval()
        if predict_vessels
        else None
    )
    reference = ensemble_av if ensemble_av is not None else ensemble_vessels
    loader = reference.make_inference_dataloader(
        data,
        input_space="preprocessed",
        batch_size=8,
        num_workers=8,
        ignore_exceptions=True,
    )
    for batch in tqdm(loader):
        if not batch:
            continue
        items_av = []
        if ensemble_av is not None:
            predictions = ensemble_av.predict_batch(batch)
            for result in ensemble_av.postprocess_batch(predictions, batch):
                items_av.append(
                    {
                        **_segmentation_item(result),
                        "refined_mask": result["output"],
                        "output_space": result["output_space"],
                    }
                )
            del predictions
        items_vessels = []
        if ensemble_vessels is not None:
            predictions = ensemble_vessels.predict_batch(batch)
            items_vessels = [
                _segmentation_item(result)
                for result in ensemble_vessels.postprocess_batch(predictions, batch)
            ]
            del predictions
        for index in range(len(batch["image"])):
            av = items_av[index] if items_av else None
            vessels = items_vessels[index] if items_vessels else None
            yield {
                "id": (av if av is not None else vessels)["id"],
                "av": av,
                "vessels": vessels,
            }
        del items_av, items_vessels, batch


def run_segmentation_vessels_and_av(
    rgb_paths: List[Path],
    ids: Optional[List[str]] = None,
    av_path: Optional[Path] = None,
    vessels_path: Optional[Path] = None,
    device: torch.device | None = None,
    callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    predict_av: bool = False,
    predict_vessels: bool = False,
    av_model: str | Path = DEFAULT_AV_MODEL,
    vessels_model: str | Path = DEFAULT_VESSELS_MODEL,
) -> None:
    """
    Run AV and vessel segmentation on the provided images.

    Args:
        rgb_paths: List of paths to RGB fundus images
        ids: Optional list of ids to pass to make_inference_dataloader
        av_path: Folder where to store output AV segmentations
        vessels_path: Folder where to store output vessel segmentations
        device: Device to run inference on
        callback: Optional callback to process results instead of saving them
        predict_av: Whether to predict AV segmentation (default False, overriden by av_path)
        predict_vessels: Whether to predict vessel segmentation (default False, overriden by vessels_path)
        av_model: Model string/path for the artery-vein segmentation ensemble
        vessels_model: Model string/path for the vessel segmentation ensemble
    """
    if av_path is not None:
        av_path.mkdir(exist_ok=True, parents=True)
    if vessels_path is not None:
        vessels_path.mkdir(exist_ok=True, parents=True)

    should_predict_av = (av_path is not None) or predict_av
    should_predict_vessels = (vessels_path is not None) or predict_vessels

    device = resolve_device(device)
    data = _create_inputs(rgb_paths, ids=ids)

    for result in iterate_segmentation_vessels_and_av(
        data,
        device=device,
        predict_av=should_predict_av,
        predict_vessels=should_predict_vessels,
        av_model=av_model,
        vessels_model=vessels_model,
    ):
        if callback is not None:
            callback(result)
        else:
            if av_path is not None and result["av"] is not None:
                fpath = os.path.join(av_path, f"{result['id']}.png")
                mask = result["av"]["output"]
                Image.fromarray(mask.squeeze().astype(np.uint8)).save(fpath)

            if vessels_path is not None and result["vessels"] is not None:
                fpath = os.path.join(vessels_path, f"{result['id']}.png")
                mask = result["vessels"]["output"]
                Image.fromarray(mask.squeeze().astype(np.uint8)).save(fpath)


def iterate_segmentation_disc(
    data: List[ImageInput],
    device: torch.device | None = None,
    model: str | Path = DEFAULT_DISC_MODEL,
) -> Iterator[Dict[str, Any]]:
    """Yield disc probability maps and final masks through the public API."""
    ensemble = (
        _load_ensemble(SegmentationEnsemble, model).to(resolve_device(device)).eval()
    )
    loader = ensemble.make_inference_dataloader(
        data,
        input_space="preprocessed",
        batch_size=32,
        num_workers=16,
        ignore_exceptions=True,
    )
    for result in ensemble.iter_predictions(loader):
        yield _segmentation_item(result)


def run_segmentation_disc(
    rgb_paths: List[Path],
    ids: Optional[List[str]] = None,
    output_path: Optional[Path] = None,
    device: torch.device | None = None,
    callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    model: str | Path = DEFAULT_DISC_MODEL,
) -> None:
    device = resolve_device(device)
    if output_path is None and callback is None:
        raise ValueError(
            "Either output_path or callback must be provided for disc segmentation"
        )

    if output_path is not None:
        output_path.mkdir(exist_ok=True, parents=True)

    data = _create_inputs(rgb_paths, ids=ids)

    for item in iterate_segmentation_disc(data, device=device, model=model):
        if callback is not None:
            callback(item)
        elif output_path is not None:
            fpath = os.path.join(output_path, f"{item['id']}.png")
            mask = item["output"]
            Image.fromarray(mask.squeeze().astype(np.uint8)).save(fpath)


def iterate_fovea_detection(
    data: List[ImageInput],
    device: torch.device | None = None,
    model: str | Path = DEFAULT_FOVEA_MODEL,
) -> Iterator[Dict[str, Any]]:
    """Yield fovea coordinates in preprocessed-image space."""
    ensemble = (
        _load_ensemble(HeatmapRegressionEnsemble, model)
        .to(resolve_device(device))
        .eval()
    )
    loader = ensemble.make_inference_dataloader(
        data,
        input_space="preprocessed",
        batch_size=8,
        num_workers=8,
        ignore_exceptions=True,
    )
    yield from ensemble.iter_predictions(loader)


def run_fovea_detection(
    rgb_paths: List[Path],
    ids: Optional[List[str]] = None,
    device: torch.device | None = None,
    model: str | Path = DEFAULT_FOVEA_MODEL,
) -> pd.DataFrame:
    ensemble = (
        _load_ensemble(HeatmapRegressionEnsemble, model)
        .to(resolve_device(device))
        .eval()
    )
    loader = ensemble.make_inference_dataloader(
        _create_inputs(rgb_paths, ids=ids),
        input_space="preprocessed",
        batch_size=8,
        num_workers=8,
        ignore_exceptions=True,
    )
    table = ensemble.predict_dataloader(loader).set_index("id")
    result = table.reindex(columns=["x0", "y0"]).astype(float)
    result.columns = ["x_fovea", "y_fovea"]
    result.index.name = None
    return result
