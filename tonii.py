import os
import shutil

source_root = r'C:\Users\28207\Desktop\ruxian\all'
images_test = r'C:\Users\28207\Desktop\ruxian\all_image_nii'
mask_test = r'C:\Users\28207\Desktop\ruxian\all_mask_nii'

def split_images_mask(source_root, images_test, mask_test):
    # 创建目标目录（如果不存在）
    os.makedirs(images_test, exist_ok=True)
    os.makedirs(mask_test, exist_ok=True)

    patient_folders = [f for f in os.listdir(source_root) if os.path.isdir(os.path.join(source_root, f))]

    for patient in patient_folders:
        patient_path = os.path.join(source_root, patient)

        middle_folders = [f for f in os.listdir(patient_path) if os.path.isdir(os.path.join(patient_path, f))]

        if not middle_folders:
            print(f"警告：患者文件夹 '{patient}' 下没有找到中间文件夹。跳过。")
            continue

        middle_folder = os.path.join(patient_path, middle_folders[0])

        subfolders = [f for f in os.listdir(middle_folder) if os.path.isdir(os.path.join(middle_folder, f))]

        image_folder = None
        roi_folder = None

        name = patient.split(' ')[0] + '.'

        for sub in subfolders:
            sub_lower = sub.lower()
            if 'left' in sub_lower or 'right' in sub_lower:
                image_folder = os.path.join(middle_folder, sub)
            elif ('roi' in sub_lower or 'all' in sub_lower or 'cluster' or 'rooi'
                  in sub_lower or '_ca' in sub_lower or 'roio' in sub_lower or name in sub_lower):
                roi_folder = os.path.join(middle_folder, sub)

        # 复制影像文件夹
        if image_folder and os.path.exists(image_folder):
            # 构建目标文件夹路径，添加患者名称前缀以避免冲突
            image_folder_name = f"{patient}_{os.path.basename(image_folder)}"
            dest_image_path = os.path.join(images_test, image_folder_name)

            if not os.path.exists(dest_image_path):
                shutil.copytree(image_folder, dest_image_path)
                print(f": {image_folder} -> {dest_image_path}")
            else:
                print(f": {dest_image_path}。")
        else:
            print(f"'{patient}' no 'left' or 'right'）")

        if roi_folder and os.path.exists(roi_folder):
            roi_folder_name = f"{patient}_{os.path.basename(roi_folder)}"
            dest_roi_path = os.path.join(mask_test, roi_folder_name)

            if not os.path.exists(dest_roi_path):
                shutil.copytree(roi_folder, dest_roi_path)
                print(f"复制 ROI 文件夹: {roi_folder} -> {dest_roi_path}")
            else:
                print(f"目标 ROI 文件夹已存在: {dest_roi_path}，跳过复制。")
        else:
            print(f"警告：患者文件夹 '{patient}' 下没有找到 ROI 子文件夹（含 'all'）。")

    print("所有文件夹已成功提取和复制。")


import shutil
import logging
import SimpleITK as sitk


def convert_dicom_to_nifti(dicom_dir, nifti_path):
    try:
        reader = sitk.ImageSeriesReader()
        dicom_names = reader.GetGDCMSeriesFileNames(dicom_dir)
        if not dicom_names:
            raise ValueError(f"No DICOM files found in {dicom_dir}")
        reader.SetFileNames(dicom_names)
        image = reader.Execute()
        sitk.WriteImage(image, nifti_path)
    except Exception as e:
        logging.error(f"fail: {dicom_dir} -> {nifti_path}，error: {e}")
def sort_key(path):
    return int(path.name.split('-')[0])


def process_all_images(images_dir, nifti_output_dir):
    # 排序文件路径
    images_path = sorted(Path(images_dir).iterdir())
    nifti_output_path = Path(nifti_output_dir)
    nifti_output_path.mkdir(parents=True, exist_ok=True)

    for i, img_folder in enumerate(images_path, start=1):
        if img_folder.is_dir():
            img_path = str(img_folder)

            nifti_filename = f"{i}_image.nii"
            nifti_path = nifti_output_path / nifti_filename

            convert_dicom_to_nifti(img_path, str(nifti_path))

    logging.info("all DICOM to NIfTI")

def verify_nifti(image_path):
    image_nifti = sitk.ReadImage(image_path)
    image_array = sitk.GetArrayFromImage(image_nifti)
    print(f"Image shape: {image_array.shape}")
    image_affine = np.array(image_nifti.GetDirection()).reshape(3, 3)
    print("Affine matrix:\n", image_affine)



import os
import pydicom
import numpy as np
import cv2
import nibabel as nib
from pathlib import Path

def DCM(img_path, mask_path):

    slices = [pydicom.dcmread(os.path.join(img_path, s), force=True)
              for s in os.listdir(img_path) if s.endswith('.dcm')]
    slices.sort(key=lambda x: int(x.InstanceNumber))

    image = np.stack([s.pixel_array for s in slices])
    image = image.astype(np.int16)

    position_offset = np.array(slices[0].ImagePositionPatient)  # 包含 x, y, z

    if hasattr(slices[0], 'SpacingBetweenSlices'):
        slice_spacing = float(slices[0].SpacingBetweenSlices)
    else:
        slice_spacing = float(slices[0].SliceThickness)

    size = image.shape  # (num_slices, height, width)
    mask = np.zeros(size, dtype=np.uint8)

    dcm = pydicom.dcmread(mask_path)
    contours = dcm.ROIContourSequence[0].ContourSequence

    for contour in contours:
        coor_xyz = np.array(contour.ContourData).reshape(-1, 3)
        coor_xyz = coor_xyz - position_offset  # 调整位置偏移
        coor_xy = coor_xyz[:, 0:2].astype('int')

        z_coord = coor_xyz[0, 2]
        z_index = int(round(z_coord / slice_spacing))

        # 检查 z_index 是否在 image 范围内
        if 0 <= z_index < size[0]:
            temp_slice = mask[z_index, :, :]
            cv2.drawContours(temp_slice, [coor_xy], 0, 1, -1)
            mask[z_index, :, :] = temp_slice
        else:
            print(f"Warning: z_index {z_index} out of bounds for image shape {size}")

    # 检查 mask 和 image 的最终形状是否一致
    if mask.shape != image.shape:
        print(f"Warning: Shape mismatch between image {image.shape} and mask {mask.shape}")

    return mask, position_offset, slice_spacing, slices[0].PixelSpacing, slices[0].SliceThickness


def sort_key(path):
    return int(path.name.split('-')[0])


def process_all_patients(images_dir, labels_dir, output_dir):
    from pathlib import Path

    images_path = sorted(Path(images_dir).iterdir())
    labels_path = sorted(Path(labels_dir).iterdir())

    print("Images path sorted:")
    print([p.name for p in images_path])

    print("Labels path sorted:")
    print([p.name for p in labels_path])
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if len(images_path) != len(labels_path):
        print("Warning: The number of image and label folders does not match!")
        return

    for i, (img_folder, label_folder) in enumerate(zip(images_path, labels_path), start=1):
        if img_folder.is_dir() and label_folder.is_dir():
            # 获取三维掩模
            img_path = str(img_folder)
            mask_path = str(next(label_folder.glob('*.dcm'), None))

            if mask_path is None:
                print(f"Warning: No ROI file found in {label_folder}")
                continue

            try:

                mask, position_offset, slice_spacing, pixel_spacing, slice_thickness = DCM(img_path, mask_path)
                print(f"Patient {i} - Mask shape: {mask.shape}")

                mask_transposed = np.transpose(mask, (2, 1, 0))
                print(f"Patient {i} - Transposed mask shape: {mask_transposed.shape}")

                # 构建 affine 矩阵
                affine = np.array([
                    [pixel_spacing[0], 0, 0, position_offset[0]],
                    [0, pixel_spacing[1], 0, position_offset[1]],
                    [0, 0, slice_thickness, position_offset[2]],
                    [0, 0, 0, 1]
                ])

                nii_img = nib.Nifti1Image(mask_transposed, affine=affine)
                output_nii_path = output_path / f"{i}_mask.nii"
                nib.save(nii_img, str(output_nii_path))
                print(f"Processed patient {i} successfully and saved as NIfTI.")

                loaded_nii = nib.load(str(output_nii_path))
                print(f"Loaded NIfTI shape: {loaded_nii.shape}")

            except Exception as e:
                print(f"Error processing patient {i}: {e}")
    output_files = sorted(Path(output_dir).iterdir(), key=lambda p: int(p.stem.split('_')[0]))
    print("Generated NIfTI files:")
    print([f.name for f in output_files])

def main():
    # 定义源目录和目标目录
    source_root = r'./ruxian_pet'
    images_test = r'./image_dicom'
    mask_test = r'./mask_dicom'
    nifti_output_dir = r'./image_nii'
    output_dir = r'./mask_nii'

    split_images_mask(source_root, images_test, mask_test)

    images_dir = images_test

    process_all_images(images_dir, nifti_output_dir)

    first_image = sorted(Path(nifti_output_dir).iterdir())[0]
    verify_nifti(str(first_image))

    images_dir = images_test
    labels_dir = mask_test

    logging.info(" mask to NIfTI。")
    process_all_patients(images_dir, labels_dir, output_dir)
    logging.info("fine")
if __name__ == "__main__":
    main()











