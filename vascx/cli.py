import inspect
from pathlib import Path

import click
import pandas as pd
import torch
import logging
import numpy as np
import random

from rtnls_fundusprep.cli import _run_preprocessing
from vascx.shared.features import FeatureSet

from .inference import (
    run_fovea_detection,
    run_quality_estimation,
    run_segmentation_disc,
    run_segmentation_vessels_and_av,
    batch_create_overlays
)
from .utils.analysis import extract_in_parallel
from .utils.feature_docs import write_feature_descriptions


@click.group(name="vascx")
def cli():
    pass


@cli.command()
@click.argument("data_path", type=click.Path(exists=True))
@click.argument("output_path", type=click.Path())
@click.option(
    "--preprocess/--no-preprocess",
    default=True,
    help="Run preprocessing or use preprocessed images",
)
@click.option(
    "--vessels/--no-vessels", default=True, help="Run vessels and AV segmentation"
)
@click.option("--disc/--no-disc", default=True, help="Run optic disc segmentation")
@click.option(
    "--quality/--no-quality", default=True, help="Run image quality estimation"
)
@click.option("--fovea/--no-fovea", default=True, help="Run fovea detection")
@click.option(
    "--overlay/--no-overlay", default=True, help="Create visualization overlays"
)
@click.option("--n_jobs", type=int, default=4, help="Number of preprocessing workers")
def run_models(
    data_path, output_path, preprocess, vessels, disc, quality, fovea, overlay, n_jobs
):
    """Run the complete inference pipeline on fundus images.

    DATA_PATH is either a directory containing images or a CSV file with 'path' column.
    OUTPUT_PATH is the directory where results will be stored.
    """

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

    # Determine if input is a folder or CSV file
    data_path = Path(data_path)
    is_csv = data_path.suffix.lower() == ".csv"

    # Get files to process
    files = []
    ids = None

    if is_csv:
        click.echo(f"Reading file paths from CSV: {data_path}")
        try:
            df = pd.read_csv(data_path)
            if "path" not in df.columns:
                click.echo("Error: CSV must contain a 'path' column")
                return

            # Get file paths and convert to Path objects
            files = [Path(p) for p in df["path"]]

            if "id" in df.columns:
                ids = df["id"].tolist()
                click.echo("Using IDs from CSV 'id' column")

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

    # Step 1: Preprocess images if requested
    if preprocess:
        click.echo("Running preprocessing...")
        _run_preprocessing(
            files=files,
            ids=ids,
            rgb_path=preprocess_rgb_path,
            bounds_path=bounds_path,
            n_jobs=n_jobs,
        )
        
    # Use the preprocessed images for subsequent steps
    preprocessed_files = list(preprocess_rgb_path.glob("*.png"))
    ids = [f.stem for f in preprocessed_files]

    # Set up GPU device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    click.echo(f"Using device: {device}")

    # Step 2: Run quality estimation if requested
    if quality:
        click.echo("Running quality estimation...")
        df_quality = run_quality_estimation(
            fpaths=preprocessed_files, ids=ids, device=device
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
            device=device,
        )
        click.echo(f"Vessel segmentation saved to {vessels_path}")
        click.echo(f"AV segmentation saved to {av_path}")

    # Step 4: Run optic disc segmentation if requested
    if disc:
        click.echo("Running optic disc segmentation...")
        run_segmentation_disc(
            rgb_paths=preprocessed_files, ids=ids, output_path=disc_path, device=device
        )
        click.echo(f"Disc segmentation saved to {disc_path}")

    # Step 5: Run fovea detection if requested
    df_fovea = None
    if fovea:
        click.echo("Running fovea detection...")
        df_fovea = run_fovea_detection(
            rgb_paths=preprocessed_files, ids=ids, device=device
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

    # Discover candidate IDs from artery_vein folder
    candidate_files = list(av_dir.glob("*.png"))
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
            fx = float(fovea_df.loc[id_, "x_fovea"])  # type: ignore[arg-type]
            fy = float(fovea_df.loc[id_, "y_fovea"])  # type: ignore[arg-type]
            bounds_str = bounds_df.loc[id_, "bounds"]
            bounds = eval(bounds_str, {"np": np}) if isinstance(bounds_str, str) else bounds_str

            examples.append(
                {
                    "id": id_,
                    "fundus_path": preprocess_rgb_path / f"{id_}.png",
                    "av_path": av_dir / f"{id_}.png",
                    "vessels_path": vessels_dir / f"{id_}.png",
                    "disc_path": disc_dir / f"{id_}.png",
                    "fovea_location": (fx, fy),
                    "bounds": bounds,
                }
            )
        except Exception as e:
            click.echo(f"Skipping {id_}: error assembling inputs: {e}")
    
    return examples

@cli.command()
@click.argument("input_path", type=click.Path(exists=True))
@click.argument("output_csv", type=click.Path())
@click.option("--feature_set", required=True, help="Name of the feature set to run")
@click.option("--n_jobs", type=int, default=8, help="Number of extraction workers")
@click.option("--logfile", type=click.Path(), default=None, help="Optional log file for warnings")
@click.option("--plots_folder", type=click.Path(), default=None, help="Optional folder to save per-feature plots")
@click.option("--sample", type=int, default=None, help="Sample N examples for testing")
def calc_biomarkers(input_path, output_csv, feature_set, n_jobs, logfile, plots_folder, sample):
    """Extract vascular biomarkers from a run_models output folder and save to CSV.

    INPUT_PATH is the output directory from 'vascx run-models' containing folders
    like 'preprocessed_rgb/', 'artery_vein/', 'vessels/', 'disc/' plus 'bounds.csv'
    and 'fovea.csv'. OUTPUT_CSV is the destination CSV file path for features.
    """

    input_path = Path(input_path)
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    
    examples = make_examples(input_path)
    if not examples:
        click.echo("No valid examples assembled; aborting.")
        return

    # Optionally sample a subset for quick tests
    orig_len = len(examples)
    if sample is not None:
        if sample < orig_len:
            examples = random.sample(examples, sample)
            click.echo(f"Sampling {sample} of {orig_len} examples")
        else:
            click.echo(f"--sample={sample} >= {orig_len}; using all examples")

    # Optional logger
    logger = None
    if logfile is not None:
        try:
            logger = logging.getLogger("vascx.extract")
            logger.setLevel(logging.INFO)
            logger.propagate = False
            # avoid duplicate handlers
            if not any(isinstance(h, logging.FileHandler) and getattr(h, 'baseFilename', None) == str(Path(logfile)) for h in logger.handlers):
                fh = logging.FileHandler(logfile)
                fh.setLevel(logging.WARNING)
                fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
                logger.addHandler(fh)
        except Exception as e:
            raise RuntimeError(f"Warning: could not initialize logfile '{logfile}': {e}") from e

    # Run extraction
    click.echo(f"Extracting features using feature set '{feature_set}' with n_jobs={n_jobs}...")
    df = extract_in_parallel(
        examples=examples,
        feature_set_name=feature_set,
        n_jobs=n_jobs,
        logger=logger,
        plots_folder=plots_folder,
        print_stack_trace=True,
    )

    # Write feature descriptions to file
    write_feature_descriptions(feature_set, output_csv.parent / "feature_descriptions.txt")
    
    # Save results
    df.to_csv(output_csv)
    click.echo(f"Features saved to {output_csv}")


@cli.command()
@click.argument("output_file", type=click.Path())
@click.option("--feature_set", required=True, help="Name of the feature set")
def write_readme(output_file, feature_set):
    """Write only the feature descriptions to OUTPUT_FILE."""
    write_feature_descriptions(feature_set, Path(output_file))
    click.echo(f"Feature descriptions written to {output_file}")
