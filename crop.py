from PIL import Image
import nibabel as nib
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score, recall_score, precision_score
import os
import numpy as np
import nibabel as nib

def crop_nifti(image_path, mask_path, output_path):
    # 加载影像和掩码
    image = nib.load(image_path)
    mask = nib.load(mask_path)

    image_data = image.get_fdata()
    mask_data = mask.get_fdata()
    affine = image.affine

    image_shape = image_data.shape
    tumor_voxels = np.array(np.nonzero(mask_data))

    min_coords = tumor_voxels.min(axis=1)
    max_coords = tumor_voxels.max(axis=1) + 1


    min_coords = np.maximum(min_coords - 5, 0)
    max_coords = np.minimum(max_coords + 5, image_shape)

    cropped_image_data = image_data[
                         min_coords[0]:max_coords[0],
                         min_coords[1]:max_coords[1],
                         min_coords[2]:max_coords[2]
                         ]
    cropped_mask_data = mask_data[
                         min_coords[0]:max_coords[0],
                         min_coords[1]:max_coords[1],
                         min_coords[2]:max_coords[2]
                         ]

    cropped_image = nib.Nifti1Image(cropped_image_data, affine)

    nib.save(cropped_image, output_path)
    print(f"Saved cropped image to {output_path}")


image_dir = "./test_18_imagesnii"
mask_dir = "./test_18_masknii"
output_dir = "./cropped_test18_nifti"
os.makedirs(output_dir, exist_ok=True)
resampled_dir = "./resampled_test18"
shape_list = []

for image_filename in os.listdir(image_dir):
    if image_filename.endswith("_image.nii") or image_filename.endswith("_image.nii.gz"):
        base_name = image_filename.replace("_image.nii", "")
        mask_filename = base_name + "_mask.nii" if image_filename.endswith(".nii") else base_name + "_mask.nii.gz"
        print(mask_filename)
        image_path = os.path.join(image_dir, image_filename)
        mask_path = os.path.join(mask_dir, mask_filename)
        output_path = os.path.join(output_dir, image_filename)

        if os.path.exists(mask_path):
            crop_nifti(image_path, mask_path, output_path)
        else:
            print(f"Mask for {image_filename} not found!")



import os
import numpy as np
import nibabel as nib
from scipy.ndimage import zoom

cropped_dir = output_dir
target_size = (64, 64, 64)

def resample_nifti(input_path, output_path, target_size):
    nii = nib.load(input_path)
    data = nii.get_fdata()
    affine = nii.affine

    original_size = data.shape
    zoom_factors = [t / o for t, o in zip(target_size, original_size)]
    resampled_data = zoom(data, zoom_factors, order=3)

    resampled_nii = nib.Nifti1Image(resampled_data, affine)

    nib.save(resampled_nii, output_path)
    print(f"Resampled {input_path} to {output_path} with size {target_size}")


for filename in os.listdir(cropped_dir):
    if filename.endswith(".nii") or filename.endswith(".nii.gz"):
        input_path = os.path.join(cropped_dir, filename)
        output_path = os.path.join(resampled_dir, filename)
        resample_nifti(input_path, output_path, target_size)



