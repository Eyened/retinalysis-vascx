import os
from pathlib import Path
import pandas as pd
from tqdm.notebook import tqdm
from joblib import Parallel, delayed

from vascx import Retina
from vascx.features import Tortuosity
from vascx.features.tortuosity import TortuosityMeasure, Zone


def process_one(av_path, disc_path, scaling_factor, id=None):
    r = Retina.from_file(av_path, scaling_factor=scaling_factor)
    r.load_disc(disc_path)

    features = [
        ("tortuosity_curvature", Tortuosity(TortuosityMeasure.Curvature, Zone.All)),
        ("tortuosity_distance", Tortuosity(TortuosityMeasure.Distance, Zone.All)),
        ("tortuosity_inflections", Tortuosity(TortuosityMeasure.Inflections, Zone.All)),
    ]

    res = {}
    for layer_name, layer in [("veins", r.veins), ("arteries", r.arteries)]:
        for feature_name, feature in features:
            res[layer_name + "_" + feature_name] = feature.compute(layer)

    res["sharpness"] = r.variance_of_laplacian()

    if id is not None:
        res["id"] = id
    return res


def process_many(examples):
    args = [
        {"id": ex[0], "av_path": ex[2], "disc_path": ex[3], "scaling_factor": ex[1]}
        for ex in examples
    ]
    return Parallel(n_jobs=20)(delayed(process_one)(**ex) for ex in tqdm(args))


# def process_one(av_path, disc_path, scaling_factor, csv_path):
#     try:
#         r = Retina.from_file(av_path, scaling_factor=scaling_factor)
#         r.load_disc(disc_path)

#         tortuosity_curvature = Tortuosity(TortuosityMeasure.Curvature, Zone.All)
#         tortuosity_distance = Tortuosity(TortuosityMeasure.Distance, Zone.All)
#         tortuosity_inflections = Tortuosity(TortuosityMeasure.Inflections, Zone.All)

#         vessels = []
#         for vessel in r.arteries.vessels.filter_segments_by_numpoints(4):
#             vessels.append(
#                 {
#                     "artery": True,
#                     "median_diameter": vessel.median_diameter,
#                     "length": vessel.length,
#                     "chord_length": vessel.chord_length,
#                     "tortuosity_curvature": tortuosity_curvature._compute_for_segment(
#                         vessel
#                     ),
#                     "tortuosity_distance": tortuosity_distance._compute_for_segment(
#                         vessel
#                     ),
#                     "tortuosity_inflections": tortuosity_inflections._compute_for_segment(
#                         vessel
#                     ),
#                 }
#             )

#         for vessel in r.veins.vessels.filter_segments_by_numpoints(4):
#             vessels.append(
#                 {
#                     "artery": False,
#                     "median_diameter": vessel.median_diameter,
#                     "length": vessel.length,
#                     "chord_length": vessel.chord_length,
#                     "tortuosity_curvature": tortuosity_curvature._compute_for_segment(
#                         vessel
#                     ),
#                     "tortuosity_distance": tortuosity_distance._compute_for_segment(
#                         vessel
#                     ),
#                     "tortuosity_inflections": tortuosity_inflections._compute_for_segment(
#                         vessel
#                     ),
#                 }
#             )

#         pd.DataFrame(vessels).to_csv(csv_path)

#     except Exception:
#         print(f"An exception occurred with file: {av_path}")
