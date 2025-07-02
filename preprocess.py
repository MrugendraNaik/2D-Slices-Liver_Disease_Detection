import os
import nibabel as nib
import numpy as np
from PIL import Image

def save_slice_as_image(image_slice, output_path):
    norm_slice = (image_slice - np.min(image_slice)) / (np.max(image_slice) - np.min(image_slice) + 1e-6)
    img = Image.fromarray((norm_slice * 255).astype(np.uint8)).convert("L")
    img.save(output_path)

def preprocess_liver_dataset(images_dir, labels_dir, output_base_dir):
    os.makedirs(os.path.join(output_base_dir, "train", "healthy"), exist_ok=True)
    os.makedirs(os.path.join(output_base_dir, "train", "disease"), exist_ok=True)

    for file in os.listdir(images_dir):
        if not file.endswith(".nii.gz") or file.startswith("._"):
            continue
        img_path = os.path.join(images_dir, file)
        label_path = os.path.join(labels_dir, file)

        img_nii = nib.load(img_path)
        label_nii = nib.load(label_path)

        img_data = img_nii.get_fdata()
        label_data = label_nii.get_fdata()

        for i in range(img_data.shape[2]): 
            img_slice = img_data[:, :, i]
            label_slice = label_data[:, :, i]

            has_tumor = 2 in label_slice
            cls = "disease" if has_tumor else "healthy"

            filename = file.replace(".nii.gz", f"_slice{i}.jpg")
            output_path = os.path.join(output_base_dir, "train", cls, filename)
            save_slice_as_image(img_slice, output_path)
            print(f"Saved: {output_path} [{cls}]")

preprocess_liver_dataset(
    r"D:\FINAL LIVER\Task03_Liver\imagesTr",
    r"D:\FINAL LIVER\Task03_Liver\labelsTr",
    "data"
)
