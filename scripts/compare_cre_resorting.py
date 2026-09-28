"""Compare the legacy and corrected reducers on identical sample vessel calibers.

Run from the repository root: python scripts/compare_cre_resorting.py
Outputs are in samples/cre_resorting/; measurements are in pixels.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from rtnls_enface.grids.hemifields import HemifieldField
from vascx.fundus.features.cre import CRE, CREMode
from vascx.fundus.loader import RetinaLoader
from vascx.utils.analysis import _load_retina_input


def legacy_cre(calibers, cte):
    values = sorted(calibers)
    while len(values) > 1:
        n = len(values)
        values = [
            values[i] if i == n // 2 else cte * np.sqrt(values[i] ** 2 + values[-i - 1] ** 2)
            for i in range((n + 1) // 2)
        ]
    return values[0] if values else None


def main():
    root = Path(__file__).resolve().parents[1]
    output = root / 'samples' / 'cre_resorting'
    output.mkdir(exist_ok=True)
    variants = {
        'temporal_4': CRE(),
        'nasal_4': CRE(CREMode.Nasal),
        'full_6': CRE(CREMode.Full),
        'temporal_superior_2': CRE(hemifield=HemifieldField.Superior),
        'temporal_inferior_2': CRE(hemifield=HemifieldField.Inferior),
        'temporal_3': CRE(max_vessels=3),
        'temporal_6': CRE(max_vessels=6),
        'full_narrow_6': CRE(CREMode.Full, inner_circle=0.8, outer_circle=1.275),
    }
    rows, circles = [], []
    for item in RetinaLoader.from_folder(root / 'samples' / 'fundus').to_dict():
        retina = _load_retina_input(item)
        print(item['id'], flush=True)
        for variant, feature in variants.items():
            for layer_name in ('arteries', 'veins'):
                layer = retina.layers[layer_name]
                cte = 0.88 if layer_name == 'arteries' else 0.95
                old_values, new_values = [], []
                for radius in feature.get_circle_multiples():
                    new, intersections = feature.compute_cre_for_circle(layer, feature.get_circle(layer, radius))
                    calibers = [seg.get_median_diameter(feature.spline_error_fraction) for seg, _ in intersections]
                    old = legacy_cre(calibers, cte)
                    assert (old is None) == (new is None)
                    if new is not None:
                        old_values.append(old)
                        new_values.append(new)
                    circles.append(dict(sample=item['id'], variant=variant, layer=layer_name,
                                        radius=radius, without_resorting=old, with_resorting=new,
                                        calibers=repr(sorted(calibers))))
                old = float(np.median(old_values)) if len(old_values) >= feature.min_circles else np.nan
                new = float(np.median(new_values)) if len(new_values) >= feature.min_circles else np.nan
                rows.append(dict(sample=item['id'], variant=variant, layer=layer_name,
                                 valid_circles=len(new_values), without_resorting=old, with_resorting=new,
                                 difference=new-old, percent_change=100*(new-old)/old))
    frame = pd.DataFrame(rows)
    frame.to_csv(output / 'comparison.csv', index=False)
    pd.DataFrame(circles).to_csv(output / 'circles.csv', index=False)
    print(frame.to_string(index=False))


if __name__ == '__main__':
    main()
