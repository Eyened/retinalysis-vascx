# Creates separate masks for arteries and veins from av folder
# This needs to be run before apply G_reco

import imageio.v2
import glob
import os
import re
import numpy as np
from pathlib import Path

input_directory = "/SSD/home/sacha/Programmation/retinalysis_versions/v2/retinalysis/rtnls_vascx/G_reco/fundus_sample/av" # replace with your av directory
output_directory = "/SSD/home/sacha/Programmation/retinalysis_versions/v2/retinalysis/rtnls_vascx/G_reco/fundus_sample/reco_tmp/av_separated" #replace with your output directory

# Create the output directory if it doesn't exist
os.makedirs(output_directory, exist_ok=True)

input_image_list = glob.glob(f"{input_directory}/*.png")

# Sort the list in ascending order by their suffixes
input_image_list.sort(key=lambda f: int(re.findall(r'\d+', os.path.splitext(f)[0])[-1]))

# Read each image into an array and plot
for i, image_path in enumerate(input_image_list, 1):
    img_array = imageio.v2.imread(image_path)
    
    # Replace 2 with 0 and 3 with 1
    img_array_A = np.copy(img_array)
    img_array_A = np.where(img_array == 2, 0, img_array_A)
    img_array_A = np.where(img_array == 3, 1, img_array_A)

    # Replace 1 with 0 and 3 with 2
    img_array_V = np.copy(img_array)
    img_array_V = np.where(img_array == 1, 0, img_array_V)
    img_array_V = np.where(img_array == 3, 2, img_array_V)
    
    # Save the modified array back to a file
    imageio.imsave(os.path.join(output_directory, Path(image_path).stem + "_A" + ".png"), img_array_A)
    imageio.imsave(os.path.join(output_directory, Path(image_path).stem + "_V" + ".png"), img_array_V)

    # Print the progress
    print(f'\rProcessed {i} of {len(input_image_list)} images', end='', flush=True)