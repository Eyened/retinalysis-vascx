"""Shared test tolerances for local and public runs. Edit values here."""

# Maximum relative biomarker change on each image, in percent (5 means 5%).
# Zero references require zero; NaN transitions still fail. The pytest option
# --vascx-max-percent-change overrides this value for a single run.
BIOMARKER_MAX_PERCENT_CHANGE = 5.0

# Minimum Dice overlap for each foreground class on each image, from 0 to 1.
SEGMENTATION_MIN_DICE = 0.99

# Maximum absolute error in each fovea coordinate, in preprocessed-image pixels.
FOVEA_ABS_TOL_PIXELS = 2.0

# Maximum average extraction time per image, in seconds, after JIT warm-up.
PIPELINE_MAX_SECONDS_PER_CALL = 3.0

# Maximum wall-clock time in seconds for each CLI subprocess before termination.
CLI_TIMEOUT_SECONDS = 3600

# Floating-point epsilon multiplier used only to include the exact percentage
# boundary despite rounding; this does not add an absolute biomarker tolerance.
BIOMARKER_BOUNDARY_EPS_MULTIPLIER = 8
