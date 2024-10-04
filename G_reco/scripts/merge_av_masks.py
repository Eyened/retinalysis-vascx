# This script merges the A and V masks into a single mask for each image
# This needs to be done after applying G_reco, to get the proper mask format for vascx

import imageio.v3
import glob
import os
import re
import numpy as np
from pathlib import Path
from collections import defaultdict

input_directory = "/SSD/home/sacha/Programmation/retinalysis_versions/v2/retinalysis/rtnls_vascx/G_reco/fundus_sample/reco_tmp/reco_output/01-26-2024_10-34"  # replace with G_reco's output directory
input_image_list = glob.glob(f"{input_directory}/*.png")
output_directory = "/SSD/home/sacha/Programmation/retinalysis_versions/v2/retinalysis/rtnls_vascx/G_reco/fundus_sample/av_reco"  # replace with new av directory
threshold = 0.5 # threshold for binarization of the masks

# Create the output directory if it doesn't exist
os.makedirs(output_directory, exist_ok=True)

# Sort the list in ascending order by their suffixes
input_image_list.sort(key=lambda f: int(re.findall(r'\d+', os.path.splitext(f)[0])[-1]))
input_image_list = glob.glob(f"{input_directory}/*.png")


# Initialize the dictionary
image_dict = defaultdict(dict)

# Populate the dictionary
for image_path in input_image_list:
    filename = Path(image_path).stem  # Get the filename without extension
    id1, id2, id3, id4, vessel_type, iteration = filename.split('_')  # Split the filename into id, vessel type, and iteration
    if iteration == "10":
        id = '_'.join([id1, id2, id3, id4])  # Concatenate id1, id2, id3, and id4 with "_" in between
        image_dict[id][vessel_type] = image_path


total_images = len(image_dict)


# Merge each pair of images A and V into a new mask
for idx, (id, vessel_dict) in enumerate(image_dict.items(), start=1):
    img_array_A = imageio.v3.imread(vessel_dict['A'])
    img_array_A = ((img_array_A >= threshold * 255) * 255).astype(np.uint8)
    img_array_A = np.where(img_array_A != 0, 1, 0)
    
    img_array_V = imageio.v3.imread(vessel_dict['V'])
    img_array_V = ((img_array_V >= threshold * 255) * 255).astype(np.uint8)
    img_array_V = np.where(img_array_V != 0, 2, 0)
    
    merged_array = img_array_A + img_array_V
    
    # Convert the merged array to uint8
    merged_array = merged_array.astype(np.uint8)
    
    # Save the merged array back to a file
    output_path = os.path.join(output_directory, f"{id}.png")
    imageio.imsave(output_path, merged_array)

    print(f'\rProcessed {idx} of {total_images} images', end='', flush=True)