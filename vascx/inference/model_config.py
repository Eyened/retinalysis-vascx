from pathlib import Path

HUGGINGFACE_MODEL_REPO_ID = "Eyened/vascx"
HUGGINGFACE_MODEL_REPO_URL = f"https://huggingface.co/{HUGGINGFACE_MODEL_REPO_ID}"

MODEL_DIR_FILES = {
    "quality": Path("quality/quality.pt"),
    "av": Path("artery_vein/av_july24.pt"),
    "vessels": Path("vessels/vessels_july24.pt"),
    "disc": Path("disc/disc_july24.pt"),
    "fovea": Path("fovea/fovea_july24.pt"),
}

DEFAULT_MODEL_PATHS = {
    name: f"{HUGGINGFACE_MODEL_REPO_ID}:{path.as_posix()}"
    for name, path in MODEL_DIR_FILES.items()
}

DEFAULT_QUALITY_MODEL = DEFAULT_MODEL_PATHS["quality"]
DEFAULT_AV_MODEL = DEFAULT_MODEL_PATHS["av"]
DEFAULT_VESSELS_MODEL = DEFAULT_MODEL_PATHS["vessels"]
DEFAULT_DISC_MODEL = DEFAULT_MODEL_PATHS["disc"]
DEFAULT_FOVEA_MODEL = DEFAULT_MODEL_PATHS["fovea"]
