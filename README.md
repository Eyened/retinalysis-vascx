## VascX retinal vascular analysis

VascX was created for the extraction of vascular features from fundus image segmentations.

### Installation

To install the entire fundus analysis pipeline including fundus preprocessing, model inference code and vascular biomarker extraction:

1. Create a conda or virtualenv virtual environment, or otherwise ensure a clean environment.

2. Install the [rtnls_inference package](https://github.com/Eyened/retinalysis-inference):

```
pip install retinalysis-inference
```

5. Install VascX:

```
pip install retinalysis-vascx
```

### Usage

To speed up re-execution of vascx we recommend to run the segmentation and feature extraction steps separately:

To run on the provided samples folder (in the git):

```
git clone git@github.com:Eyened/retinalysis-vascx.git rtnls_vascx
cd rtnls_vascx
vascx run-models ./samples/fundus/original/ /path/to/segmentations
vascx run-biomarker-extraction /path/to/segmentations /path/to/features.csv --feature_set full --n-jobs 8 --logfile /path/to/logfile.txt
```


We also provide notebooks with the three stages:

1. Preprocessing. See [this notebook](./notebooks/0_preprocess.ipynb). This step is CPU-heavy and benefits from parallelization (see notebook).

2. Inference. See [this notebook](./notebooks/1_segment_preprocessed.ipynb). All models can be ran in a single GPU with >10GB VRAM.

3. Feature extraction. See [this notebook](./notebooks/2_feature_extraction.ipynb). This step is CPU-heavy again and benefits from parallelization (see notebook).



### Implementation

VascX processes vessel segmentations through four main stages, each producing different data representations:

- **Input masks**: `np.ndarray[bool]` per layer; optic disc and fovea metadata from segmentation models.

- **Stage 1 - Binary/skeleton**: 
  - `binary`: filled vessel mask after disc masking
  - `binary_nodisc`: vessel mask without disc region
  - `skeleton`: skeletonized vessel centerlines using skimage skeletonization

- **Stage 2 - Undirected graph**: 
  - NetworkX `Graph` with skeleton pixels as nodes
  - `Segment` objects stored on edges containing skeleton points and geometric properties
  - Each segment represents a vessel segment between junction points

- **Stage 3 - Directed digraph**: 
  - NetworkX `DiGraph` with flow direction from optic disc outward
  - `trees`: root nodes representing vessel trees emanating from disc
  - `nodes`: `Endpoint` and `Bifurcation` objects with spatial positions
  - `segments`: directed vessel segments with computed properties (diameter, length, etc.)

- **Stage 4 - Resolved vessels**: 
  - Merged vessel graph after running vessel resolution algorithm
  - `resolved_segments`: final vessel segments after merging short segments
  - Segment-to-pixel mapping for spatial feature computation

Biomarker families use different representations: mask-based features use `binary`; topology features use `digraph` and `nodes`; morphological features use `segments` with computed diameters; spatial features use segment-to-pixel mappings.

### Biomarkers

VascX computes retinal vascular biomarkers using different data representations from the processing pipeline:

**VascularDensity**
- *Representation*: Uses `VesselTreeLayer.binary` vessel mask
- *Computation*: Fraction of vessel pixels within an OD–fovea-oriented ellipse or ETDRS `GridField` over total area
- *Options*: `grid_field` (region selection), `cut_mask` (handle out-of-bounds regions)

**BifurcationCount**
- *Representation*: Uses `Bifurcation` nodes from `layer.nodes` in the directed graph
- *Computation*: Count of bifurcation points, optionally within a specified `GridField`
- *Options*: `grid_field` (spatial filtering)

**BifurcationAngles**
- *Representation*: Uses `Bifurcation` geometry from `digraph` with outgoing branch directions
- *Computation*: Aggregation of angles at bifurcations measured at distance `delta` along branches
- *Options*: `delta` (measurement distance), `max_angle` (angle filter), `grid_field`, `aggregator`

**Caliber**
- *Representation*: Uses `segments` with computed `median_diameter` from skeleton-derived measurements
- *Computation*: Aggregate of segment diameters filtered by minimum length and optional spatial region
- *Options*: `min_numpoints` (length filter), `grid_field`, `aggregator`

**Length**
- *Representation*: Uses `segments` with skeleton or spline-based length measurements
- *Computation*: Mean segment length across all qualifying segments
- *Options*: `min_numpoints` (minimum segment length filter)

**Tortuosity**
- *Representation*: Uses `segments` or `resolved_segments`; curvature mode uses segment splines
- *Computation*: Vessel tortuosity by distance ratio, curvature, or inflection counts; optionally length-normalized
- *Options*: `mode` (segments vs vessels), `measure` (distance/curvature/inflections), `length_measure`, `norm_measure`, `min_numpoints`, `grid_field`, `aggregator`

**CRE (Central Retinal Equivalents)**
- *Representation*: Uses `segments` with `median_diameter` and circle intersection geometry around optic disc
- *Computation*: Recursive diameter combining of segments intersecting concentric circles; returns median across radii
- *Options*: None exposed (uses implicit artery/vein classification constants)

**TemporalAngle**
- *Representation*: Uses `resolved_segments` with circle intersections and OD–fovea spatial geometry
- *Computation*: Median angle between two dominant temporal arcades at circles from 2/3 OD–fovea distance outward
- *Options*: `od_to_fovea_fraction`, `increment` (circle spacing)

**Coverage** (VesselsLayerFeature)
- *Representation*: Uses `FundusVesselsLayer.distance_transform` normalized by OD–fovea distance
- *Computation*: Mean distance to nearest vessel pixel across all retinal pixels
- *Options*: `ignore_fovea` (reserved for future use)

**VarianceOfLaplacian** (RetinaFeature)
- *Representation*: Uses `Retina.laplacian` - global image Laplacian operator
- *Computation*: Variance of Laplacian map as image sharpness proxy
- *Options*: None

**DiscFoveaDistance** (RetinaFeature)
- *Representation*: Uses `Retina` optic disc and fovea spatial coordinates
- *Computation*: Euclidean distance between optic disc center and fovea location
- *Options*: None

Ready-to-run feature sets are available under `vascx/fundus/feature_sets` (e.g., `full`, `bergmann`, `quality`) and can be selected by name when using `extract_in_parallel`. To generate feature descriptions alongside extraction:

```python
df = extract_in_parallel(examples, "full", n_jobs=8, descriptions_output_path="feature_descriptions_full.txt")
```
