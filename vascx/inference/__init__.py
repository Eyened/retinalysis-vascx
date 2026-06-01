from .utils import batch_create_overlays, create_fundus_overlay

__all__ = [
    "batch_create_overlays",
    "create_fundus_overlay",
    "iterate_fovea_detection",
    "iterate_quality_estimation",
    "iterate_segmentation_disc",
    "iterate_segmentation_vessels_and_av",
    "run_fovea_detection",
    "run_quality_estimation",
    "run_segmentation_disc",
    "run_segmentation_vessels_and_av",
]

_INFERENCE_NAMES = {
    "iterate_fovea_detection",
    "iterate_quality_estimation",
    "iterate_segmentation_disc",
    "iterate_segmentation_vessels_and_av",
    "run_fovea_detection",
    "run_quality_estimation",
    "run_segmentation_disc",
    "run_segmentation_vessels_and_av",
}


def __getattr__(name: str):
    if name in _INFERENCE_NAMES:
        from . import inference as inference_mod

        return getattr(inference_mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
