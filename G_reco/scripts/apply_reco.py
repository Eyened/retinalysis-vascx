import numpy as np
import os
import torch
from glob import glob
import json
import monai
from monai.inferers import sliding_window_inference

os.chdir("/SSD/home/sacha/Programmation/retinalysis_versions/v2/retinalysis/rtnls_vascx/G_reco/model_inference")

import sys
sys.path.append('/SSD/home/sacha/Programmation/retinalysis_versions/v2/retinalysis/rtnls_vascx/G_reco/model_inference')
import imageUtils




########################################################
def monai_predict_image(image, model, roi_size, sw_batch_size = 5, mode = "gaussian", overlap = 0.5, device="cpu"):
    image = torch.from_numpy(image)
    image = image.float().unsqueeze(0).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        output = sliding_window_inference(image, roi_size, sw_batch_size, model, mode = mode, overlap = overlap)
    output = output.squeeze()
    output = torch.sigmoid(output).cpu().numpy()
    return output

########################################################





# parameters to initialize
directory = "/SSD/home/sacha/Programmation/retinalysis_versions/v2/retinalysis/rtnls_vascx/G_reco/fundus_sample/reco_tmp/av_separated" # source directory contains separated masks (arteries and veins)
result_path = "/SSD/home/sacha/Programmation/retinalysis_versions/v2/retinalysis/rtnls_vascx/G_reco/fundus_sample/reco_tmp"
output_directory = os.path.join(result_path, "reco_output")
path_model = "01-26-2024_10-34"
iterations = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# load model
model_file =f"modele_2D/{path_model}/best_metric_model.pth"
parameters_training = open(f"modele_2D/{path_model}/config_training.json")
parameters_training = json.load(parameters_training)
norm = parameters_training["norm"]
model = monai.networks.nets.UNet(
    spatial_dims=2,
    in_channels=1,
    out_channels=1,
    channels=(16, 32, 64, 128),
    strides=(2, 2, 2),
    num_res_units=2,
    norm=(norm)
).to(device)

if device == "cuda":
    model.load_state_dict(torch.load(model_file)).to(device)
else:
    model.load_state_dict(torch.load(model_file, map_location="cpu"))

roi_size = (96, 96)



# Create the output directory if it doesn't exist
os.makedirs(output_directory, exist_ok=True)

if f"{path_model}" not in os.listdir(f"{output_directory}"):
    os.mkdir(f"{output_directory}/{path_model}")



image_list = glob(f"{directory}/*.png")
total_images = len(image_list)

for idx, image_path in enumerate(image_list, 1):
    image = imageUtils.readImage(image_path)
    image = ((image >= 0.5) * 255).astype(np.uint8)
    name_image = image_path.split("/")[-1].split(".")[0]

    for i in range(1, iterations + 1):
        image = imageUtils.normalizeImage(image, 1)
        image = monai_predict_image(image, model, roi_size, sw_batch_size=5, mode="gaussian", overlap=0.5, device=device)
        if i == iterations:
            image_to_save = (image.copy() * 255).astype(np.uint8)
            output_path_reco = f"{output_directory}/{path_model}/{name_image}_{i:02d}.png"
            imageUtils.saveImage(image_to_save, output_path_reco)

    # Print the progress
    print(f'\rProcessed {idx} of {total_images} images', end='', flush=True)