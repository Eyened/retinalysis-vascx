import os
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, TypeVar

import numpy as np
import pandas as pd
import torch
from PIL import Image
from rtnls_inference.dtos_inference import ModelInputDTO
from rtnls_inference.ensembles import get_ensemble_class
from rtnls_inference.ensembles.ensemble_classification import ClassificationEnsemble
from rtnls_inference.ensembles.ensemble_heatmap_regression import (
    HeatmapRegressionEnsemble,
)
from rtnls_inference.ensembles.ensemble_segmentation import SegmentationEnsemble
from rtnls_inference.ensembles.predict_output import (
    decollate_predict_full,
    restore_array_to_preprocessed,
)
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
    return resolved(loaded.ensemble, loaded.config, loaded.fpath)


def _full_items(ensemble, batch) -> list[dict[str, Any]]:
    """Run one supported full-contract batch and split its normalized batch axis."""
    return decollate_predict_full(ensemble.predict_step_full(batch))


def _canonical_probability_item(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": item.get("id"),
        "image": restore_array_to_preprocessed(
            item["aggregate"], item["geometry"], "bilinear"
        ),
    }


def _create_dtos(
    rgb_paths: List[Path],
    ids: Optional[List[str]] = None,
) -> List[ModelInputDTO]:
    """Helper to create ModelInputDTOs from paths."""
    if ids is None:
        ids = [p.stem for p in rgb_paths]

    if len(rgb_paths) != len(ids):
        raise ValueError("rgb_paths and ids must have the same length")

    return [
        ModelInputDTO(
            id=str(id_val),
            image=str(rgb_path),
        )
        for id_val, rgb_path in zip(ids, rgb_paths)
    ]


def iterate_quality_estimation(
    data: List[ModelInputDTO],
    device: torch.device | None = None,
    model: str | Path = DEFAULT_QUALITY_MODEL,
) -> Iterator[Dict[str, Any]]:
    """Yield quality ensemble inference items."""
    device = resolve_device(device)
    ensemble_quality = _load_ensemble(ClassificationEnsemble, model).to(device)

    inputs = {"images": [item.to_serialized_dict() for item in data]}
    dataloader = ensemble_quality._make_inference_dataloader(
        inputs,
        num_workers=8,
        preprocess=False,
        batch_size=16,
    )

    with torch.no_grad():
        for batch in tqdm(dataloader):
            if len(batch) == 0:
                continue

            items = [
                {"id": item.get("id"), "logits": item["aggregate"]}
                for item in _full_items(ensemble_quality, batch)
            ]

            for item in items:
                yield item


def run_quality_estimation(
    fpaths,
    ids: Optional[List[str]] = None,
    device: torch.device | None = None,
    model: str | Path = DEFAULT_QUALITY_MODEL,
):
    device = resolve_device(device)
    data = _create_dtos(fpaths, ids=ids)
    output_ids, outputs = [], []

    for item in iterate_quality_estimation(data, device=device, model=model):
        output_ids.append(item["id"])
        outputs.append(item["logits"].tolist())

    return pd.DataFrame(
        outputs,
        index=output_ids,
        columns=["q1", "q2", "q3"],
    )


def iterate_segmentation_vessels_and_av(
    data: List[ModelInputDTO],
    device: torch.device | None = None,
    predict_av: bool = True,
    predict_vessels: bool = True,
    av_model: str | Path = DEFAULT_AV_MODEL,
    vessels_model: str | Path = DEFAULT_VESSELS_MODEL,
) -> Iterator[Dict[str, Any]]:
    """Yield raw segmentation items for AV and vessels."""
    device = resolve_device(device)
    if not predict_av and not predict_vessels:
        return

    ensemble_av = (
        _load_ensemble(SegmentationEnsemble, av_model).to(device).eval()
        if predict_av
        else None
    )
    ensemble_vessels = (
        _load_ensemble(SegmentationEnsemble, vessels_model).to(device).eval()
        if predict_vessels
        else None
    )
    reference_ensemble = ensemble_av or ensemble_vessels
    if reference_ensemble is None:
        return

    inputs = {"images": [item.to_serialized_dict() for item in data]}
    dataloader = reference_ensemble._make_inference_dataloader(
        inputs,
        num_workers=8,
        preprocess=False,
        batch_size=8,
    )

    with torch.no_grad():
        for batch in tqdm(dataloader):
            if len(batch) == 0:
                continue

            if ensemble_av:
                av_full_items = _full_items(ensemble_av, batch)
                items_av = []
                for full_item in av_full_items:
                    item = _canonical_probability_item(full_item)
                    processed = ensemble_av.postprocess_item(full_item)
                    item["refined_mask"] = processed["output"]
                    item["output_space"] = processed["output_space"]
                    items_av.append(item)
            else:
                items_av = None
            items_vessels = (
                [
                    _canonical_probability_item(item)
                    for item in _full_items(ensemble_vessels, batch)
                ]
                if ensemble_vessels
                else None
            )

            num_items = (
                len(items_av)
                if items_av is not None
                else len(items_vessels)
                if items_vessels is not None
                else 0
            )

            for idx in range(num_items):
                av_item = items_av[idx] if items_av is not None else None
                vessel_item = items_vessels[idx] if items_vessels is not None else None

                yield {
                    "id": (
                        av_item["id"]
                        if av_item is not None
                        else vessel_item["id"]
                        if vessel_item is not None
                        else None
                    ),
                    "av": av_item,
                    "vessels": vessel_item,
                }


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
        ids: Optional list of ids to pass to _make_inference_dataloader
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
    data = _create_dtos(rgb_paths, ids=ids)

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
                mask = result["av"].get("refined_mask")
                if mask is None:
                    mask = np.argmax(result["av"]["image"], -1)
                Image.fromarray(mask.squeeze().astype(np.uint8)).save(fpath)

            if vessels_path is not None and result["vessels"] is not None:
                fpath = os.path.join(vessels_path, f"{result['id']}.png")
                mask = np.argmax(result["vessels"]["image"], -1)
                Image.fromarray(mask.squeeze().astype(np.uint8)).save(fpath)


def iterate_segmentation_disc(
    data: List[ModelInputDTO],
    device: torch.device | None = None,
    model: str | Path = DEFAULT_DISC_MODEL,
) -> Iterator[Dict[str, Any]]:
    """Yield disc segmentation inference items."""
    device = resolve_device(device)
    ensemble_disc = _load_ensemble(SegmentationEnsemble, model).to(device).eval()

    inputs = {"images": [item.to_serialized_dict() for item in data]}
    dataloader = ensemble_disc._make_inference_dataloader(
        inputs,
        num_workers=16,
        preprocess=False,
        batch_size=32,
    )

    with torch.no_grad():
        for batch in tqdm(dataloader):
            if len(batch) == 0:
                continue

            items = [
                _canonical_probability_item(item)
                for item in _full_items(ensemble_disc, batch)
            ]

            for item in items:
                yield item


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

    data = _create_dtos(rgb_paths, ids=ids)

    for item in iterate_segmentation_disc(data, device=device, model=model):
        if callback is not None:
            callback(item)
        elif output_path is not None:
            fpath = os.path.join(output_path, f"{item['id']}.png")
            mask = np.argmax(item["image"], -1)
            Image.fromarray(mask.squeeze().astype(np.uint8)).save(fpath)


def iterate_fovea_detection(
    data: List[ModelInputDTO],
    device: torch.device | None = None,
    model: str | Path = DEFAULT_FOVEA_MODEL,
) -> Iterator[Dict[str, Any]]:
    """Yield fovea detection inference items."""
    device = resolve_device(device)
    ensemble_fovea = _load_ensemble(HeatmapRegressionEnsemble, model).to(device)

    inputs = {"images": [item.to_serialized_dict() for item in data]}
    dataloader = ensemble_fovea._make_inference_dataloader(
        inputs,
        num_workers=8,
        preprocess=False,
        batch_size=8,
    )

    with torch.no_grad():
        for batch in tqdm(dataloader):
            if len(batch) == 0:
                continue

            items = [
                ensemble_fovea.postprocess_item(item)
                for item in _full_items(ensemble_fovea, batch)
            ]

            for item in items:
                yield item


def run_fovea_detection(
    rgb_paths: List[Path],
    ids: Optional[List[str]] = None,
    device: torch.device | None = None,
    model: str | Path = DEFAULT_FOVEA_MODEL,
) -> pd.DataFrame:
    device = resolve_device(device)
    data = _create_dtos(rgb_paths, ids=ids)
    output_ids, outputs = [], []

    for item in iterate_fovea_detection(data, device=device, model=model):
        output_ids.append(item["id"])
        outputs.append(
            [
                *item["keypoints"][0].tolist(),
            ]
        )

    return pd.DataFrame(
        outputs,
        index=output_ids,
        columns=["x_fovea", "y_fovea"],
    )
