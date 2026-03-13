# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn import preprocessing
from scipy import signal
import SimpleITK as sitk
import pydicom
import os
from sklearn.cluster import KMeans
import cv2
from fea_extract import featureextractor
import warnings
from itertools import islice
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")


def cluster(image, mask, clu_num):
    kernal = np.ones((3, 3, 3))
    xmin = np.min(image)
    xmax = np.max(image)
    image = (image - xmin) / (xmax - xmin)
    image = 255 * image
    img_masked = np.zeros_like(image)
    po = np.argwhere(mask != 0)

    for i in range(po.shape[0]):
        img_masked[po[i, 0], po[i, 1], po[i, 2]] = image[po[i, 0], po[i, 1], po[i, 2]]

    fea_image = signal.fftconvolve(image, kernal, mode='same')
    fea_masked = fea_image[mask != 0]
    julei_col = img_masked[mask != 0]

    julei_2 = julei_col.reshape(-1, 1)
    julei_1 = fea_masked.reshape(-1, 1)

    julei = np.stack([julei_2, julei_1], axis=1)
    julei = julei.reshape(-1, 2)
    julei = preprocessing.scale(julei)

    pre = KMeans(n_clusters=clu_num, random_state=20).fit_predict(julei)
    aa = []
    bb = []
    for i in range(pre.shape[0]):
        if pre[i] == 0:
            aa.append(julei_col[i])
        if pre[i] == 1:
            bb.append(julei_col[i])

    aa = np.array(aa)
    bb = np.array(bb)

    am = np.average(aa)
    bm = np.average(bb)

    if am > bm:
        pre[pre == 0] = 3
        pre[pre == 1] = 0
        pre[pre == 3] = 1

    pre[pre == 0] = clu_num

    for i in range(po.shape[0]):
        img_masked[po[i, 0], po[i, 1], po[i, 2]] = pre[i]
    return img_masked


import SimpleITK as sitk
def DCM(img_path, mask_path):
    print(img_path)

    print(os.path.exists(img_path))
    print(os.path.exists(img_path + '/' + os.listdir(img_path)[0]))
    print(img_path + '/' + os.listdir(img_path)[0])
    print(os.path.join(img_path, os.listdir(img_path)[0]))
    print(os.path.exists(os.path.join(img_path, os.listdir(img_path)[0])))
    slices = [pydicom.dcmread(os.path.join(img_path, s)) for s in os.listdir(img_path)]
    slices.sort(key=lambda x: int(x.InstanceNumber))
    image = np.stack([s.pixel_array for s in slices]).astype(np.int16)

    position_offset = slices[0].ImagePositionPatient
    position_offset = np.array([position_offset[0], position_offset[1], 0])
    size = image.shape

    img = np.zeros(size)

    dcm = pydicom.dcmread(mask_path)
    contours = dcm.ROIContourSequence[0].ContourSequence
    for contour in contours:
        coor_xyz = np.array(contour.ContourData).reshape(-1, 3)
        coor_xyz -= position_offset
        coor_xy = coor_xyz[:, 0:2].astype('int')
        temp = img[int(coor_xyz[0, 2]), :, :]
        cv2.drawContours(temp, [coor_xy], 0, 1, -1)
    # plt.imshow(image[71, :, :], cmap='gray')
    # plt.show()


    return image, img


def fenge(dcm_path, storge):

    # for patient_folder in sorted(os.listdir(dcm_path), key=lambda x: int(x.split('-')[0])):
    for patient_folder in sorted(os.listdir(dcm_path)):

        patient_path = os.path.join(dcm_path, patient_folder)
        studies = [folder for folder in os.listdir(patient_path) if os.path.isdir(os.path.join(patient_path, folder))]
        studies_path = os.path.join(patient_path, studies[0])

        keywords_image = ['left', 'right']
        image_folders = [folder for folder in os.listdir(studies_path)
                        if any(keyword in folder.lower() for keyword in keywords_image)]

        name = patient_folder.split(' ')[0] + '.'
        keywords_mask = ['roi', 'all', 'cluster', 'rooi', '_ca', 'roio', name]
        mask_folders = [folder for folder in os.listdir(studies_path)
                        if any(keyword in folder.lower() for keyword in keywords_mask)]

        for image_folder in image_folders:
            img_path = os.path.join(studies_path, image_folder)
            mask_path = os.path.join(studies_path, mask_folders[0])  # Assume there is at least one mask folder

            mask_file_path = os.path.join(mask_path, os.listdir(mask_path)[0])
            image, mask = DCM(img_path, mask_file_path)

            # 图像高斯平滑
            gaussian_filter = sitk.SmoothingRecursiveGaussianImageFilter()
            image_gauss = sitk.GetArrayFromImage(gaussian_filter.Execute(sitk.GetImageFromArray(image)))

            # 调用聚类
            image_cluster = cluster(image_gauss, mask, 2)
            img_cluster_1 = image_cluster.copy()
            img_cluster_2 = image_cluster.copy()
            img_cluster_1[image_cluster != 1] = 0
            img_cluster_2[image_cluster != 2] = 0
            img_cluster_2[img_cluster_2 == 2] = 1

            # 保存结果
            patient_feature_folder = os.path.join(storge, patient_folder)
            if not os.path.exists(patient_feature_folder):
                os.makedirs(patient_feature_folder)
            sitk.WriteImage(sitk.GetImageFromArray(image_gauss), os.path.join(patient_feature_folder, 'image.nrrd'))
            sitk.WriteImage(sitk.GetImageFromArray(img_cluster_1), os.path.join(patient_feature_folder, '1.nrrd'))
            sitk.WriteImage(sitk.GetImageFromArray(img_cluster_2), os.path.join(patient_feature_folder, '2.nrrd'))
            sitk.WriteImage(sitk.GetImageFromArray(mask), os.path.join(patient_feature_folder, 'mask.nrrd'))
        continue

def filter_features(featureVector):
    filtered_features = dict(islice(featureVector.items(), 37, None))
    return filtered_features


def radiomics_extraction(storge_path, patient_folder, combined_features_all_patients, feature_names_all_patients):
    settings = "settings.yaml"
    extractor = featureextractor.RadiomicsFeatureExtractor(settings)

    combined_features = []
    feature_names = []

    for mask_index in ['mask', '1', '2']:
        mask_file = f'{mask_index}.nrrd' if mask_index != 'mask' else 'mask.nrrd'
        img_file = os.path.join(storge_path, patient_folder, 'image.nrrd')

        mask_file_path = os.path.join(storge_path, patient_folder, mask_file)
        if not os.path.isfile(mask_file_path):
            print(f"Warning: Mask file {mask_file_path} does not exist. Skipping...")
            continue

        feature_vector = extractor.execute(img_file, mask_file_path)

        feature_vector = filter_features(feature_vector)

        for key, value in feature_vector.items():
            combined_features.append(value)
            feature_names.append(key)

        print(f"{patient_folder}: {mask_file} finished")


    combined_features_all_patients.append(combined_features)

    all_features_df = pd.DataFrame(combined_features_all_patients)
    all_features_df.to_excel(os.path.join('single_patients_features.xlsx'))

    feature_names_all_patients.append(feature_names)


def liucheng(dcm_path, storge):
    # 存储所有病例的特征
    combined_features_all_patients = []
    feature_names_all_patients = []

    fenge(dcm_path, storge)

    tmp = os.listdir(storge)
    tmp.sort()
    for patient_folder in tmp:
        radiomics_extraction(storge, patient_folder, combined_features_all_patients, feature_names_all_patients)

    all_features_df = pd.DataFrame(combined_features_all_patients)

    all_features_df.columns = feature_names_all_patients[0]

    all_features_df.to_excel(os.path.join(storge, dcm_path + 'features.xlsx'), index=False)
    print(f"Features for all patients saved to 'features.xlsx'.")

    # process_excel_and_save(storge)


if __name__ == "__main__":
    dcm_path = r"./ruxian_pet"
    storge = './feature_out'
    liucheng(dcm_path, storge)
