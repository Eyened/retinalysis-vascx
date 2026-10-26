---
title: "Tortuosity"
---

<!-- Authored page: edit directly. Neither documentation generator overwrites this text. -->

**Tortuosity.** Three complementary measures are provided per segment (or per resolved vessel). Let $L_{\text{arc},i}$ be arclength and $L_{\text{chord},i}$ the end‑to‑end Euclidean distance.

- Distance factor:

$$
T_i^{\text{DF}} = \frac{L_{\text{arc},i}}{L_{\text{chord},i}}.
$$

- Curvature-based measure: mean absolute planar curvature sampled uniformly in the spline parameter $t$ (inverse pixels):

$$
T_i^{\kappa} = \frac{1}{N_i} \sum_{j=1}^{N_i} |\kappa_i(t_j)|.
$$

- Inflection count (number of curvature sign changes along the centerline):

$$
T_i^{\text{INF}} = N^{(i)}_{\text{inflections}}.
$$

When reporting a single score over multiple segments, VascX use a length-weighted mean by default:

$$
T_{\text{tot}} = \sum_{i \in S} \left( \frac{\ell_i}{\sum_{j \in S} \ell_j} \right) t_i,
$$

with $t_i$ any of the measures above.

Curvature is reported in inverse pixels. Distance ratios are dimensionless and inflections are counts. Fractional segment splitting requires an available disc–fovea distance.

## Report examples

### Macula-centered

Distance tortuosity with maximum segment length 0.2 × the disc–fovea distance.

![Macula-centered tortuosity explanation from the Rotterdam Study report](/assets/biomarkers/macula-centered/tort_dist_max_segment_len_0p2_crcl_arteries.jpg)

### OD-centered

Distance tortuosity with maximum segment length 0.2 × the disc–fovea distance.

![OD-centered tortuosity](/assets/biomarkers/od-centered/tort_dist_max_segment_len_0p2_crcl_arteries.jpg)
