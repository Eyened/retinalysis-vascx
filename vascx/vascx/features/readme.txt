ta_arteries -  opening of the arterial main arches (degrees). Note that it is often hard to identify a single main artery/vein
ta_veins - opening of the vein main arches (degrees). Note that it is often hard to identify a single main artery/vein
cre_arteries - CRE of the temporal arteries. CRE is a more robust caliber measurement than the diam values
cre_veins - CRE of the temporal veins. CRE is a more robust caliber measurement than the diam values
vd_arteries - Vascular density calculated from the arterial mask as mask_area / total_area on a specific region
vd_veins - Vascular density calculated from the vein mask
diam_arteries_median - Median diameter calculated over vessel diameter measurements. Each vessel measurement is itself a median over multiple measurements along the length of the vessel. This measure is sensitive to image and segmentation quality
diam_arteries_std - Standard deviation of vessel diameters
diam_veins_median
diam_veins_std
tort_arteries - Average arterial tortuosity over all arteries, calculated as arc_length / endpoint_distance. This is the standard measure of tortuosity.
tort_veins- Average venous tortuosity over all arteries, calculated as arc_length / endpoint_distance. This is the standard measure of tortuosity.
bif_arteries - Number of bifurcations in the artery tree. This measure is likely sensitive to image quality.
bif_veins - Number of bifurcations in the vein tree. This measure is likely sensitive to image quality.