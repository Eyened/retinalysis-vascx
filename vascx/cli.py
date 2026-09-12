import os
from pathlib import Path

import click
import pandas as pd
import logging
import numpy as np

from vascx.inference.model_config import (
    DEFAULT_AV_MODEL,
    DEFAULT_DISC_MODEL,
    DEFAULT_FOVEA_MODEL,
    DEFAULT_QUALITY_MODEL,
    DEFAULT_VESSELS_MODEL,
    MODEL_DIR_FILES,
)

from .utils.analysis import extract_biomarkers_to_folder
from .utils.feature_docs import (
    write_feature_set_readme,
    write_variable_display_mapping,
)


def _resolve_model_paths(
    model_dir,
    quality_model,
    av_model,
    vessels_model,
    disc_model,
    fovea_model,
    run_quality,
    run_vessels,
    run_disc,
    run_fovea,
):
    model_dir = model_dir or os.environ.get("VASCX_MODEL_DIR")
    model_dir = Path(model_dir).expanduser() if model_dir else None
    if model_dir is not None and not model_dir.is_dir():
        raise click.ClickException(f"Model directory does not exist: {model_dir}")

    def resolve(name, explicit, default):
        if explicit is not None:
            return Path(explicit).expanduser()
        if model_dir is not None:
            return model_dir / MODEL_DIR_FILES[name]
        return default

    resolved = {
        "quality": resolve("quality", quality_model, DEFAULT_QUALITY_MODEL),
        "av": resolve("av", av_model, DEFAULT_AV_MODEL),
        "vessels": resolve("vessels", vessels_model, DEFAULT_VESSELS_MODEL),
        "disc": resolve("disc", disc_model, DEFAULT_DISC_MODEL),
        "fovea": resolve("fovea", fovea_model, DEFAULT_FOVEA_MODEL),
    }

    required = {
        "quality": run_quality,
        "av": run_vessels,
        "vessels": run_vessels,
        "disc": run_disc,
        "fovea": run_fovea,
    }
    for name, is_required in required.items():
        model = resolved[name]
        if is_required and isinstance(model, Path) and not model.exists():
            raise click.ClickException(
                f"Missing {name} model file: {model}. "
                "Pass an explicit --*-model path, use --model-dir with the documented "
                "layout, or omit local model options to use the Hugging Face defaults."
            )

    return resolved, model_dir


def _read_run_models_csv(data_path: Path):
    df = pd.read_csv(data_path, dtype={"id": str, "path": str}, keep_default_na=False)
    required_columns = {"id", "path"}
    missing_columns = sorted(required_columns - set(df.columns))
    if missing_columns:
        missing = ", ".join(f"'{column}'" for column in missing_columns)
        raise click.ClickException(f"CSV must contain {missing} column(s)")

    path_values = df["path"].astype(str).str.strip()
    missing_path_mask = path_values == ""
    if missing_path_mask.any():
        rows = ", ".join(str(i + 2) for i in path_values.index[missing_path_mask].tolist())
        raise click.ClickException(f"CSV contains empty path values on row(s): {rows}")

    files = [Path(value) for value in path_values]
    relative_paths = [str(path) for path in files if not path.is_absolute()]
    if relative_paths:
        examples = ", ".join(relative_paths[:3])
        suffix = "" if len(relative_paths) <= 3 else f", ... ({len(relative_paths)} total)"
        raise click.ClickException(
            "CSV 'path' values must be absolute paths; relative path(s) found: "
            f"{examples}{suffix}"
        )

    id_values = df["id"].astype(str).str.strip()
    missing_id_mask = id_values == ""
    if missing_id_mask.any():
        rows = ", ".join(str(i + 2) for i in id_values.index[missing_id_mask].tolist())
        raise click.ClickException(f"CSV contains empty id values on row(s): {rows}")

    ids = id_values.tolist()
    duplicated_ids = sorted(id_values[id_values.duplicated()].unique().tolist())
    if duplicated_ids:
        examples = ", ".join(duplicated_ids[:5])
        suffix = "" if len(duplicated_ids) <= 5 else f", ... ({len(duplicated_ids)} total)"
        raise click.ClickException(f"CSV 'id' values must be unique; duplicate id(s): {examples}{suffix}")

    return files, ids


@click.group(name="vascx")
def cli():
    pass


@cli.command()
@click.argument("data_path", type=click.Path(exists=True))
@click.argument("output_path", type=click.Path())
@click.option(
    "--preprocess/--no-preprocess",
    default=True,
    help=(
        "Run preprocessing first. With --no-preprocess, reuse existing RGB PNGs "
        "named <id>.png in OUTPUT_PATH/preprocessed_rgb/."
    ),
)
@click.option(
    "--vessels/--no-vessels",
    default=True,
    help=(
        "Run vessel and artery-vein segmentation. With --no-vessels, skip this "
        "step and leave any existing OUTPUT_PATH/vessels/ and "
        "OUTPUT_PATH/artery_vein/ outputs in place."
    ),
)
@click.option(
    "--disc/--no-disc",
    default=True,
    help=(
        "Run optic disc segmentation. With --no-disc, skip this step and leave "
        "any existing OUTPUT_PATH/disc/ outputs in place."
    ),
)
@click.option(
    "--quality/--no-quality",
    default=True,
    help=(
        "Run image quality estimation. With --no-quality, skip this step and "
        "leave any existing OUTPUT_PATH/quality.csv in place."
    ),
)
@click.option(
    "--fovea/--no-fovea",
    default=True,
    help=(
        "Run fovea detection. With --no-fovea, skip this step and leave any "
        "existing OUTPUT_PATH/fovea.csv in place."
    ),
)
@click.option(
    "--overlay/--no-overlay",
    default=True,
    help=(
        "Create visualization overlays from model outputs. Use --no-overlay "
        "when overlays are already present or not needed."
    ),
)
@click.option("--n_jobs", "--n-jobs", type=int, default=4, help="Number of preprocessing workers")
@click.option(
    "--device",
    default=None,
    help=(
        "PyTorch device for model inference (e.g. cuda:0, mps, cpu). "
        "When omitted, uses the first available CUDA GPU, Apple MPS, or CPU."
    ),
)
@click.option(
    "--model-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help=(
        "Directory containing manually downloaded model files. Defaults to "
        "VASCX_MODEL_DIR when set. Expected layout: quality/quality.pt, "
        "artery_vein/av_wsoft_patches_02_finetune.pt, vessels/vessels_may26.pt, "
        "disc/disc_may26.pt, fovea/fovea_may26.pt."
    ),
)
@click.option(
    "--quality-model",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Path to a local quality model file. Overrides --model-dir for quality.",
)
@click.option(
    "--av-model",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Path to a local artery-vein segmentation model file. Overrides --model-dir for AV.",
)
@click.option(
    "--vessels-model",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Path to a local vessel segmentation model file. Overrides --model-dir for vessels.",
)
@click.option(
    "--disc-model",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Path to a local optic disc segmentation model file. Overrides --model-dir for disc.",
)
@click.option(
    "--fovea-model",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Path to a local fovea detection model file. Overrides --model-dir for fovea.",
)
def run_models(
    data_path,
    output_path,
    preprocess,
    vessels,
    disc,
    quality,
    fovea,
    overlay,
    n_jobs,
    device,
    model_dir,
    quality_model,
    av_model,
    vessels_model,
    disc_model,
    fovea_model,
):
    """Run the complete inference pipeline on fundus images.

    DATA_PATH is either a directory containing images or a CSV file with 'id' and
    'path' columns. CSV path values must be absolute paths, and IDs must be unique.
    OUTPUT_PATH is the directory where results will be stored.

    With --no-preprocess, DATA_PATH must still be an existing path for CLI
    compatibility, but model inputs are read from OUTPUT_PATH/preprocessed_rgb/*.png.
    Preprocessed images must be RGB PNG files named <id>.png; the file stem is used
    as the ID for all model outputs. If you plan to run calc-biomarkers later,
    OUTPUT_PATH/bounds.csv must also be present.

    The --no-* flags are intended for re-executing part of a previous pipeline
    run. Skipped steps are not regenerated, so keep the corresponding existing
    files in OUTPUT_PATH if later steps need them.

    By default, model weights are loaded from Hugging Face. For offline use,
    provide local model files with --model-dir, VASCX_MODEL_DIR, or per-model
    options such as --disc-model. Per-model options take precedence over
    --model-dir and VASCX_MODEL_DIR.
    """
    try:
        import importlib.util

        if importlib.util.find_spec("torch") is None:
            raise ImportError
    except ImportError as e:
        raise click.ClickException(
            "run-models requires PyTorch. Install torch for your platform."
        ) from e

    from vascx.inference.device import resolve_device
    from vascx.inference.inference import (
        run_fovea_detection,
        run_quality_estimation,
        run_segmentation_disc,
        run_segmentation_vessels_and_av,
    )
    from vascx.inference.utils import batch_create_overlays

    output_path = Path(output_path)
    output_path.mkdir(exist_ok=True, parents=True)

    # Setup output directories
    preprocess_rgb_path = output_path / "preprocessed_rgb"
    vessels_path = output_path / "vessels"
    av_path = output_path / "artery_vein"
    disc_path = output_path / "disc"
    overlay_path = output_path / "overlays"

    # Create required directories
    if preprocess:
        preprocess_rgb_path.mkdir(exist_ok=True, parents=True)
    if vessels:
        av_path.mkdir(exist_ok=True, parents=True)
        vessels_path.mkdir(exist_ok=True, parents=True)
    if disc:
        disc_path.mkdir(exist_ok=True, parents=True)
    if overlay:
        overlay_path.mkdir(exist_ok=True, parents=True)

    bounds_path = output_path / "bounds.csv"
    quality_path = output_path / "quality.csv"
    fovea_path = output_path / "fovea.csv"

    models, active_model_dir = _resolve_model_paths(
        model_dir=model_dir,
        quality_model=quality_model,
        av_model=av_model,
        vessels_model=vessels_model,
        disc_model=disc_model,
        fovea_model=fovea_model,
        run_quality=quality,
        run_vessels=vessels,
        run_disc=disc,
        run_fovea=fovea,
    )
    if active_model_dir is not None:
        click.echo(f"Using model directory: {active_model_dir}")

    data_path = Path(data_path)
    files = []
    ids = None

    if preprocess:
        # Determine if input is a folder or CSV file
        is_csv = data_path.suffix.lower() == ".csv"

        # Get files to process
        if is_csv:
            click.echo(f"Reading file paths from CSV: {data_path}")
            try:
                files, ids = _read_run_models_csv(data_path)
                click.echo("Using IDs from CSV 'id' column")
            except click.ClickException:
                raise
            except Exception as e:
                click.echo(f"Error reading CSV file: {e}")
                return
        else:
            click.echo(f"Finding files in directory: {data_path}")
            files = list(data_path.glob("*"))
            ids = [f.stem for f in files]

        if not files:
            click.echo("No files found to process")
            return

        click.echo(f"Found {len(files)} files to process")
    else:
        click.echo(
            f"Skipping preprocessing; using preprocessed images from {preprocess_rgb_path}"
        )

    # Step 1: Preprocess images if requested
    if preprocess:
        try:
            from rtnls_fundusprep.cli import _run_preprocessing
        except ImportError as exc:
            raise click.ClickException(
                "run-models --preprocess requires retinalysis-fundusprep. "
                "Install it with pip install 'retinalysis-vascx[fundusprep]' "
                "or run with --no-preprocess."
            ) from exc

        click.echo("Running preprocessing...")
        _run_preprocessing(
            files=files,
            ids=ids,
            rgb_path=preprocess_rgb_path,
            bounds_path=bounds_path,
            n_jobs=n_jobs,
        )
        
    # Use the preprocessed images for subsequent steps
    preprocessed_files = sorted(preprocess_rgb_path.glob("*.png"))
    if not preprocessed_files:
        raise click.ClickException(
            f"No preprocessed PNG files found in {preprocess_rgb_path}. "
            "Expected RGB PNG files named <id>.png. When using --no-preprocess, "
            "create this directory before running run-models."
        )
    ids = [f.stem for f in preprocessed_files]

    # Set up inference device
    inference_device = resolve_device(device)
    click.echo(f"Using device: {inference_device}")

    # Step 2: Run quality estimation if requested
    if quality:
        click.echo("Running quality estimation...")
        df_quality = run_quality_estimation(
            fpaths=preprocessed_files,
            ids=ids,
            device=inference_device,
            model=models["quality"],
        )
        df_quality.to_csv(quality_path)
        click.echo(f"Quality results saved to {quality_path}")

    # Step 3: Run vessels and AV segmentation if requested
    if vessels:
        click.echo("Running vessels and AV segmentation...")
        run_segmentation_vessels_and_av(
            rgb_paths=preprocessed_files,
            ids=ids,
            av_path=av_path,
            vessels_path=vessels_path,
            device=inference_device,
            av_model=models["av"],
            vessels_model=models["vessels"],
        )
        click.echo(f"Vessel segmentation saved to {vessels_path}")
        click.echo(f"AV segmentation saved to {av_path}")

    # Step 4: Run optic disc segmentation if requested
    if disc:
        click.echo("Running optic disc segmentation...")
        run_segmentation_disc(
            rgb_paths=preprocessed_files,
            ids=ids,
            output_path=disc_path,
            device=inference_device,
            model=models["disc"],
        )
        click.echo(f"Disc segmentation saved to {disc_path}")

    # Step 5: Run fovea detection if requested
    df_fovea = None
    if fovea:
        click.echo("Running fovea detection...")
        df_fovea = run_fovea_detection(
            rgb_paths=preprocessed_files,
            ids=ids,
            device=inference_device,
            model=models["fovea"],
        )
        df_fovea.to_csv(fovea_path)
        click.echo(f"Fovea detection results saved to {fovea_path}")

    # Step 6: Create overlays if requested
    if overlay:
        click.echo("Creating visualization overlays...")

        # read fovea data if necessary
        if df_fovea is None:
            df_fovea = pd.read_csv(fovea_path)
        fovea_data = {
            idx: (row["x_fovea"], row["y_fovea"])
            for idx, row in df_fovea.iterrows() # type: ignore[arg-type]
        }

        # Create visualization overlays
        batch_create_overlays(
            rgb_dir=preprocess_rgb_path,
            output_dir=overlay_path,
            av_dir=av_path,
            disc_dir=disc_path,
            fovea_data=fovea_data,
        )

        click.echo(f"Visualization overlays saved to {overlay_path}")

    click.echo(f"All requested processing complete. Results saved to {output_path}")


def _get_fovea_columns(fovea_df):
    for x_col, y_col in (("x_fovea", "y_fovea"), ("mean_x", "mean_y")):
        if x_col in fovea_df.columns and y_col in fovea_df.columns:
            return x_col, y_col
    raise click.ClickException(
        "fovea.csv must contain x_fovea/y_fovea or mean_x/mean_y columns"
    )


def make_examples(input_path):
    # Required subpaths
    preprocess_rgb_path = input_path / "preprocessed_rgb"
    av_dir = input_path / "artery_vein"
    vessels_dir = input_path / "vessels"
    disc_dir = input_path / "disc"
    fovea_csv = input_path / "fovea.csv"
    bounds_csv = input_path / "bounds.csv"

    # Load metadata CSVs
    if not fovea_csv.exists() or not bounds_csv.exists():
        click.echo("Error: fovea.csv and bounds.csv must exist in INPUT_PATH")
        return
    fovea_df = pd.read_csv(fovea_csv, index_col=0)
    bounds_df = pd.read_csv(bounds_csv, index_col=0)
    fovea_df.index = fovea_df.index.astype(str)
    bounds_df.index = bounds_df.index.astype(str)
    x_fovea_col, y_fovea_col = _get_fovea_columns(fovea_df)

    # Discover candidate IDs from artery_vein folder
    candidate_files = sorted(av_dir.glob("*.png"))
    candidate_ids = [p.stem for p in candidate_files]
    if not candidate_ids:
        click.echo("No artery_vein PNG files found; nothing to extract.")
        return

    # Filter by presence of required counterparts
    def has_required(id_: str) -> bool:
        return (
            (disc_dir / f"{id_}.png").exists()
            and (id_ in fovea_df.index)
            and (id_ in bounds_df.index)
        )

    ids = [id_ for id_ in candidate_ids if has_required(id_)]
    if not ids:
        click.echo("No matching IDs with required disc/fovea/bounds found.")
        return
    click.echo(f"Found {len(ids)} valid IDs for biomarker extraction")

    # Build examples list
    examples = []
    for id_ in ids:
        try:
            fx = float(fovea_df.loc[id_, x_fovea_col])  # type: ignore[arg-type]
            fy = float(fovea_df.loc[id_, y_fovea_col])  # type: ignore[arg-type]
            bounds_str = bounds_df.loc[id_, "bounds"]
            bounds = eval(bounds_str, {"np": np}) if isinstance(bounds_str, str) else bounds_str

            example = {
                "id": id_,
                "fundus_path": preprocess_rgb_path / f"{id_}.png",
                "av_path": av_dir / f"{id_}.png",
                "vessels_path": vessels_dir / f"{id_}.png",
                "disc_path": disc_dir / f"{id_}.png",
                "fovea_location": (fx, fy),
                "bounds": bounds,
            }
            examples.append(example)
        except Exception as e:
            click.echo(f"Skipping {id_}: error assembling inputs: {e}")
    
    return examples

@cli.command()
@click.argument("input_path", type=click.Path(exists=True))
@click.argument(
    "output_folder",
    type=click.Path(file_okay=False, path_type=Path),
)
@click.option(
    "--feature-set",
    "--feature_set",
    "feature_set",
    required=True,
    help="Name of the feature set to run",
)
@click.option(
    "--n-jobs",
    "--n_jobs",
    "n_jobs",
    type=click.IntRange(min=1),
    default=8,
    help="Number of extraction workers",
)
@click.option("--logfile", type=click.Path(), default=None, help="Optional log file for warnings")
@click.option(
    "--sample",
    type=click.IntRange(min=1),
    default=None,
    help="Extract only the first N examples for testing.",
)
@click.option(
    "--report/--no-report",
    default=True,
    show_default=True,
    help="Generate README, manifest, and sample biomarker plots.",
)
@click.option(
    "--report-samples",
    type=click.IntRange(min=0),
    default=3,
    show_default=True,
    help="Number of input images to visualize in the report.",
)
@click.option(
    "--report-image",
    "report_images",
    multiple=True,
    help="Specific image ID to visualize; may be supplied more than once.",
)
@click.option(
    "--overwrite",
    is_flag=True,
    help="Replace report-owned files already present in OUTPUT_FOLDER.",
)
@click.option(
    "--naming",
    type=click.Choice(["resolved", "canonical"]),
    default="resolved",
    show_default=True,
    help="Biomarker naming convention for output columns.",
)
def calc_biomarkers(
    input_path,
    output_folder,
    feature_set,
    n_jobs,
    logfile,
    sample,
    report,
    report_samples,
    report_images,
    overwrite,
    naming,
):
    """Extract vascular biomarkers into a self-contained output folder.

    INPUT_PATH is the output directory from `vascx run-models` containing
    `preprocessed_rgb/`, `artery_vein/`, `vessels/`, `disc/`, `bounds.csv`,
    and `fovea.csv`. OUTPUT_FOLDER receives the biomarker CSV and documentation.
    """
    input_path = Path(input_path)
    output_folder = Path(output_folder)
    if output_folder.suffix.lower() == ".csv":
        raise click.ClickException(
            "OUTPUT_FOLDER now expects a directory, not a CSV path. "
            "Pass a folder; biomarkers will be written as biomarkers.csv inside it."
        )

    examples = make_examples(input_path)
    if not examples:
        raise click.ClickException("No valid examples assembled; aborting.")

    original_count = len(examples)
    if sample is not None and sample < original_count:
        examples = examples[:sample]
        click.echo(f"Using the first {sample} of {original_count} examples")
    elif sample is not None:
        click.echo(f"--sample={sample} >= {original_count}; using all examples")

    report_retinas = None
    if report_images:
        examples_by_id = {str(example["id"]): example for example in examples}
        missing = [image_id for image_id in report_images if image_id not in examples_by_id]
        if missing:
            raise click.ClickException(
                "Unknown --report-image ID(s): " + ", ".join(missing)
            )
        report_retinas = [examples_by_id[image_id] for image_id in report_images]

    logger = None
    if logfile is not None:
        logfile_path = Path(logfile)
        logfile_path.parent.mkdir(parents=True, exist_ok=True)
        logger = logging.getLogger("vascx.extract")
        logger.setLevel(logging.INFO)
        logger.propagate = False
        resolved_logfile = str(logfile_path.resolve())
        if not any(
            isinstance(handler, logging.FileHandler)
            and getattr(handler, "baseFilename", None) == resolved_logfile
            for handler in logger.handlers
        ):
            handler = logging.FileHandler(logfile_path)
            handler.setLevel(logging.WARNING)
            handler.setFormatter(
                logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
            )
            logger.addHandler(handler)

    click.echo(
        f"Extracting features using feature set '{feature_set}' with n_jobs={n_jobs}..."
    )
    try:
        dataframe = extract_biomarkers_to_folder(
            retinas=examples,
            feature_set=feature_set,
            output_folder=output_folder,
            n_jobs=n_jobs,
            logger=logger,
            print_stack_trace=True,
            naming=naming,
            generate_report=report,
            report_retinas=report_retinas,
            report_sample_size=report_samples,
            overwrite=overwrite,
        )
    except (FileExistsError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc

    click.echo(
        f"Saved {len(dataframe)} rows of biomarkers to "
        f"{output_folder / 'biomarkers.csv'}"
    )
    click.echo(f"Data dictionary written to {output_folder}")
    if report:
        click.echo(f"Biomarker report written to {output_folder / 'README.md'}")


@cli.command()
@click.argument("output_file", type=click.Path())
@click.option(
    "--feature-set",
    "--feature_set",
    "feature_set",
    required=True,
    help="Name of the feature set",
)
@click.option(
    "--naming",
    type=click.Choice(["resolved", "canonical"]),
    default="resolved",
    show_default=True,
    help="Naming convention used in the biomarker table.",
)
def write_readme(output_file, feature_set, naming):
    """Write Markdown documentation for FEATURE_SET to OUTPUT_FILE."""
    write_feature_set_readme(
        feature_set,
        Path(output_file),
        naming=naming,
    )
    click.echo(f"Feature-set README written to {output_file}")


@cli.command("write-mapping")
@click.argument("output_file", type=click.Path())
@click.option(
    "--feature-set",
    "--feature_set",
    "feature_set",
    required=True,
    help="Name of the feature set",
)
@click.option("--json", "as_json", is_flag=True, help="Write mapping as JSON instead of CSV.")
@click.option(
    "--naming",
    type=click.Choice(["resolved", "canonical"]),
    default="resolved",
    show_default=True,
    help="Naming convention for mapping keys and display names.",
)
def write_mapping(output_file, feature_set, as_json, naming):
    """Write biomarker names and descriptions for FEATURE_SET.

    JSON preserves the legacy variable-to-display-name mapping. CSV also includes
    a description column.
    """
    write_variable_display_mapping(
        feature_set, Path(output_file), as_json=as_json, naming=naming
    )
    output_format = "JSON" if as_json else "CSV"
    click.echo(f"Biomarker metadata written to {output_file} as {output_format}")
