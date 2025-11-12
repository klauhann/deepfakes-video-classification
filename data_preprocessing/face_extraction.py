from facenet_pytorch import MTCNN
import cv2
from PIL import Image
import torch

import os
from os import listdir, makedirs
import glob
from os.path import join, exists
from skimage.io import imsave
import imageio.core.util
import numpy as np


def ignore_warnings(*args, **kwargs):
    pass


imageio.core.util._precision_warn = ignore_warnings

# Create face detector
# If you want to change the default size of image saved from 160, you can
# uncomment the second line and set the parameter accordingly.

# choose device: use CUDA if available, otherwise CPU
device = "cuda:0" if torch.cuda.is_available() else "cpu"

mtcnn = MTCNN(
    margin=40,
    select_largest=False,
    post_process=False,
    device=device,
    # image_size=256,
)

# Directory containing images respective to each video
source_frames_folders = ["../train_frames/0/", "../train_frames/1/"]

# Destination location where faces cropped out from images will be saved
dest_faces_folders = ["../train_face/0/", "../train_face/1/"]

for src_folder, dest_faces_folder in zip(source_frames_folders, dest_faces_folders):
    counter = 0
    for j in listdir(src_folder):
        imgs = glob.glob(join(src_folder, j, "*.jpg"))
        if counter % 100 == 0:
            print("Number of videos done:", counter)
        if not exists(join(dest_faces_folder, j)):
            makedirs(join(dest_faces_folder, j))
        for k in imgs:
            frame = cv2.imread(k)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(frame)
            face = mtcnn(frame)

            if face is None:
                print(f"No face detected for {k}")
                continue

            try: 
                # Tensor (C, H, W) -> NumPy (H, W, C)
                face_img = face.permute(1, 2, 0).cpu().numpy()
                # Wertebereich 0–255 und uint8
                face_img = np.clip(face_img, 0, 255).astype(np.uint8)
                imsave(join(dest_faces_folder, j, os.path.basename(k)), face_img)

            except Exception as e:
                print("Image skipping (save error):", e)
            
        counter += 1
