# -*- coding: UTF-8 -*-
# The code is adapted from PyPI ITHscore 0.3.3
import os
import numpy as np
import pydicom
import cv2
from scipy.spatial.distance import cdist

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import silhouette_score
import os
import pydicom as dicom
import functools
import concurrent.futures

from scipy import ndimage
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from utils import *
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score

silhouette_score = davies_bouldin_score
def pca_dimensionality_reduction(features, explained_variance_threshold=0.90):
    # Fit PCA without specifying n_components to get the explained variance ratio
    pca = PCA()
    pca.fit(features)

    # Calculate the cumulative explained variance
    cumulative_explained_variance = np.cumsum(pca.explained_variance_ratio_)

    # Find the number of components that explain at least the threshold of the variance
    n_components = np.argmax(
        cumulative_explained_variance >= explained_variance_threshold) + 1  # +1 because index starts at 0

    print(
        f"Automatically selected {n_components} components to explain {explained_variance_threshold * 100}% of the variance.")

    # Apply PCA with the selected number of components
    pca = PCA(n_components=n_components)
    reduced_features = pca.fit_transform(features)

    return reduced_features, n_components

def pca_dimensionality_reduction(features, n_components=7):
    pca = PCA(n_components=n_components)
    return pca.fit_transform(features)

def silhouette_analysis(features, max_k=10):

    silhouette_scores = []
    for k in range(2, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(features)
        score = silhouette_score(features, kmeans.labels_)
        silhouette_scores.append(score)

    optimal_k = np.argmax(silhouette_scores) + 2  # Adding 2 to account for index offset (starting from 2)
    return optimal_k, silhouette_scores

def plot_clusters(X, labels, title):
    plt.figure(figsize=(8, 6))
    plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=50)
    plt.title(title)
    plt.colorbar(label='Cluster')
    plt.show()

def evaluate_clustering(X, labels):
    return silhouette_score(X, labels)

def compare_clustering_algorithms(features, sub_mask):
    if isinstance(features, dict):
        features = np.hstack((features['first'], features['shape'], features['glcm'], features['gldm'],
                              features['glrlm'], features['glszm'], features['ngtdm']))

    features = MinMaxScaler().fit_transform(features)

    # n_components = 7
    # Step 1: Reduce dimensionality using PCA
    reduced_features = pca_dimensionality_reduction(features)

    # Step 2: Find the optimal number of clusters for KMeans using silhouette score
    optimal_k_km, silhouette_scores = silhouette_analysis(reduced_features, max_k=9)
    print(f"Optimal K for KMeans: {optimal_k_km}")

    # Step 3: Apply KMeans with optimal k
    kmeans_labels = kmeans_clustering(reduced_features, optimal_k_km)
    kmeans_score = round(evaluate_clustering(reduced_features, kmeans_labels),2)
    # Step 4: Estimate eps for DBSCAN using KNN
    # Step 4: Estimate optimal parameters for DBSCAN
    best_dbscan_score = -1
    best_eps = 0.5
    best_min_samples = 14  # Set min_samples to be twice the feature dimension
    best_dbscan_labels = None

    for eps in np.arange(0.1, 1.2, 0.05):  # Allow small range around estimated eps
        # We now fix min_samples as 14
        dbscan_labels = dbscan_clustering(reduced_features, eps=eps, min_samples=best_min_samples)
        unique_dbscan_labels = set(dbscan_labels)

        # Remove noise points (label -1)
        if -1 in unique_dbscan_labels:
            unique_dbscan_labels.remove(-1)
        # If only 1 cluster is found (i.e., no noise), assign all points to cluster 1

        # # Only evaluate if more than 1 cluster is found
        if len(unique_dbscan_labels) > 1:
            # Remove noise points (label -1) from the silhouette score calculation
            filtered_dbscan_labels = [label if label != -1 else -1 for label in dbscan_labels]
            score = silhouette_score(reduced_features, filtered_dbscan_labels)

            if score > best_dbscan_score:
                best_dbscan_score = score
                best_eps = eps
                best_dbscan_labels = dbscan_labels

    print(f"Optimal DBSCAN parameters: eps={best_eps}, min_samples={best_min_samples}")

    if best_dbscan_score == -1:
        # print(f"Only one cluster found for eps={eps} and min_samples={best_min_samples}. Assigning all points to a single cluster.")
        dbscan_labels = [1] * len(dbscan_labels)  # Assign all points to the same cluster (label 1)
        best_dbscan_score = 1.0  # Set the score to 1 since the clustering is trivial but valid
        best_eps = eps
        best_dbscan_labels = dbscan_labels

    dbscan_score = round(best_dbscan_score,2)
    dbscan_labels = best_dbscan_labels


    # 计算DBSCAN的簇数（去除噪声标签）
    if -1 in unique_dbscan_labels:
        unique_dbscan_labels.remove(-1)  # 去除噪声点

    unique_dbscan_labels = set(dbscan_labels)

    optimal_k_db = len(unique_dbscan_labels)

    # Plotting the clusters
    plot_clusters(reduced_features, kmeans_labels, "KMeans Clustering")
    plot_clusters(reduced_features, dbscan_labels, "DBSCAN Clustering")

    print(f"KMeans Silhouette Score: {kmeans_score}")
    print(f"DBSCAN Silhouette Score: {dbscan_score}")

    return kmeans_labels, dbscan_labels, optimal_k_km, optimal_k_db, kmeans_score, dbscan_score


# def estimate_eps(features, k_neighbors=5):
#     """
#     Estimate an optimal eps value for DBSCAN using KNN distance.
#     Args:
#         features: Numpy array. The input features.
#         k_neighbors: Integer. The number of neighbors to consider for estimating eps.
#     Returns:
#         eps: Float. Estimated eps value based on the k-th nearest neighbor distance.
#     """
#     neighbors = NearestNeighbors(n_neighbors=k_neighbors)
#     neighbors.fit(features)
#     distances, _ = neighbors.kneighbors(features)
#
#     # Get the average distance to the k-th nearest neighbor as the candidate eps
#     eps = np.mean(distances[:, -1])
#     return eps


def create_label_map(sub_mask, labels):

    label_map = sub_mask.copy()
    cnt = 0
    for i in range(len(sub_mask)):
        for j in range(len(sub_mask[0])):
            if sub_mask[i][j] == 1:
                label_map[i][j] = labels[cnt] + 1  # Assign label + 1 to avoid 0 background
                cnt += 1
            else:
                label_map[i][j] = 0  # Background pixels remain 0
    return label_map


def load_seg(path):
    """
    Load segmentation mask
    Args:
        path: Str. Path to the .nii or .dcm mask.
    Returns:
        seg: Numpy array. The mask of ROI with the same shape of image.
    """
    if path.endswith(".dcm"):
        # RTStruct dcm file sometimes cannot be loaded by SimpleITK
        ds = dicom.read_file(path)
        seg = ds.pixel_array
    else:
        sitk_seg = sitk.ReadImage(path)
        seg = sitk.GetArrayFromImage(sitk_seg)

    return seg

# def get_largest_slice(img3d, mask3d):
#     """
#     Get the slice with largest tumor area
#     Args:
#         img3d: Numpy array. The whole CT volume (3D)
#         mask3d: Numpy array. Same size as img3d, binary mask with tumor area set as 1, background as 0
#     Returns:
#         img: Numpy array. The 2D image slice with largest tumor area
#         mask: Numpy array. The subset of mask in the same position of sub_img
#     """
#     area = np.sum(mask3d == 1, axis=(1, 2))
#     area_index = np.argsort(area)[-1]
#     img = img3d[area_index, :, :]
#     mask = mask3d[area_index, :, :]
#
#     return img, mask



def get_largest_slice(img3d, mask3d, dimension):
    """
    Get the slice with largest tumor area from a specified dimension.
    Args:
        img3d: Numpy array. The whole CT volume (3D)
        mask3d: Numpy array. Same size as img3d, binary mask with tumor area set as 1, background as 0
        dimension: Integer. The dimension to traverse (0 for X, 1 for Y, 2 for Z)
    Returns:
        img: Numpy array. The 2D image slice with largest tumor area
        mask: Numpy array. The subset of mask in the same position of sub_img
    """
    if dimension == 0:
        # Traverse along X axis (slice along depth dimension)
        area = np.sum(mask3d == 1, axis=(1, 2))
        area_index = np.argsort(area)[-1]
        img = img3d[area_index, :, :]
        mask = mask3d[area_index, :, :]

    elif dimension == 1:
        # Traverse along Y axis
        area = np.sum(mask3d == 1, axis=(0, 2))
        area_index = np.argsort(area)[-1]
        img = img3d[:, area_index, :]
        mask = mask3d[:, area_index, :]

    elif dimension == 2:
        # Traverse along Z axis
        area = np.sum(mask3d == 1, axis=(0, 1))
        area_index = np.argsort(area)[-1]
        img = img3d[:, :, area_index]
        mask = mask3d[:, :, area_index]

    return img, mask


def locate_tumor(img, mask, padding=2):
    """
    Locate and extract tumor from CT image using mask
    Args:
        img: Numpy array. The whole image
        mask: Numpy array. Same size as img, binary mask with tumor area set as 1, background as 0
        padding: Int. Number of pixels padded on each side after extracting tumor
    Returns:
        sub_img: Numpy array. The tumor area defined by mask
        sub_mask: Numpy array. The subset of mask in the same position of sub_img
    """
    top_margin = min(np.where(mask == 1)[0])
    bottom_margin = max(np.where(mask == 1)[0])
    left_margin = min(np.where(mask == 1)[1])
    right_margin = max(np.where(mask == 1)[1])
    # padding two pixels at each edges for further computation
    sub_img = img[top_margin - padding:bottom_margin + padding + 1, left_margin - padding:right_margin + padding + 1]
    sub_mask = mask[top_margin - padding:bottom_margin + padding + 1,
                    left_margin - padding:right_margin + padding + 1]

    return sub_img, sub_mask

from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import MinMaxScaler


def kmeans_clustering(features, n_clusters):

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(features)
    return labels


def dbscan_clustering(features, eps=0.5, min_samples=5):

    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(features)
    return labels



def extract_radiomic_features(sub_img, sub_mask, parallel=True, workers=10):

    features = {}
    first_features = []
    shape_features = []
    glcm_features = []
    gldm_features = []
    glrlm_features = []
    glszm_features = []
    ngtdm_features = []

    if parallel:
        ps, qs = [], []
        partial_extract_feature_unit = functools.partial(extract_feature_unit, sub_img)
        for p in range(len(sub_img)):
            for q in range(len(sub_img[0])):
                if (sub_mask[p][q] == 1):
                    ps.append(p)
                    qs.append(q)
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
            results = executor.map(partial_extract_feature_unit, ps, qs)
            for result in results:
                first_features.append(result["first"])
                shape_features.append(result["shape"])
                glcm_features.append(result["glcm"])
                gldm_features.append(result["gldm"])
                glrlm_features.append(result["glrlm"])
                glszm_features.append(result["glszm"])
                ngtdm_features.append(result["ngtdm"])
    else:
        for p in range(len(sub_img)):
            for q in range(len(sub_img[0])):
                if (sub_mask[p][q] == 1):
                    features_temp = extract_feature_unit(sub_img, p, q, padding=2)
                    first_features.append(features_temp["first"])
                    shape_features.append(features_temp["shape"])
                    glcm_features.append(features_temp["glcm"])
                    gldm_features.append(features_temp["gldm"])
                    glrlm_features.append(features_temp["glrlm"])
                    glszm_features.append(features_temp["glszm"])
                    ngtdm_features.append(features_temp["ngtdm"])
    features['first'] = MinMaxScaler().fit_transform(first_features)
    features['shape'] = MinMaxScaler().fit_transform(shape_features)
    features['glcm'] = MinMaxScaler().fit_transform(glcm_features)
    features['gldm'] = MinMaxScaler().fit_transform(gldm_features)
    features['glrlm'] = MinMaxScaler().fit_transform(glrlm_features)
    features['glszm'] = MinMaxScaler().fit_transform(glszm_features)
    features['ngtdm'] = MinMaxScaler().fit_transform(ngtdm_features)

    return features


def pixel_clustering(sub_mask, features, cluster=6):

    if isinstance(features, dict):
        features = np.hstack((features['first'], features['shape'], features['glcm'], features['gldm'],
                              features['glrlm'], features['glszm'], features['ngtdm']))

    features = MinMaxScaler().fit_transform(features)
    label_map = sub_mask.copy()

    clusters = KMeans(n_clusters=cluster).fit_predict(features)
    cnt = 0
    for i in range(len(sub_mask)):
        for j in range(len(sub_mask[0])):
            if sub_mask[i][j] == 1:
                label_map[i][j] = clusters[cnt] + 1
                cnt += 1
            else:
                label_map[i][j] = 0

    return label_map


def pixel_clustering2(sub_mask, features, max_k=10):

    if isinstance(features, dict):
        features = np.hstack((features['first'], features['shape'], features['glcm'], features['gldm'],
                              features['glrlm'], features['glszm'], features['ngtdm']))
    features = MinMaxScaler().fit_transform(features)
    # Get the optimal k
    optimal_k, elbow_point = determine_optimal_k(features, max_k=max_k)

    # Assuming features is now a numpy array
    label_map = sub_mask.copy()

    clusters = KMeans(n_clusters=optimal_k).fit_predict(features)
    cnt = 0
    for i in range(len(sub_mask)):
        for j in range(len(sub_mask[0])):
            if sub_mask[i][j] == 1:
                label_map[i][j] = clusters[cnt] + 1
                cnt += 1
            else:
                label_map[i][j] = 0

    return label_map, optimal_k, elbow_point



def visualize(img, sub_img, mask, sub_mask, features, cluster=6):

    if cluster != "all":
        if not isinstance(cluster, int):
            raise Exception("Please input an integer or string 'all'")
        fig = plt.figure()
        label_map = pixel_clustering(sub_mask, features, cluster)
        plt.matshow(label_map, fignum=0)
        plt.xlabel(f"Cluster pattern (K={cluster})", fontsize=15)

        return fig


    else:   # generate cluster pattern with multiple resolutions, together with whole lung CT
        max_cluster = 9
        # Subplot 1: CT image of the whole lung
        fig = plt.figure(figsize=(12, 12))
        plt.subplot(3, (max_cluster + 2) // 3, 1)
        plt.title('Raw Image')
        plt.imshow(img, cmap='gray')
        plt.scatter(np.where(mask == 1)[1], np.where(mask == 1)[0], marker='o', color='red', s=0.2)

        # Subplot 2: CT iamge of the nodule
        plt.subplot(3, (max_cluster + 2) // 3, 2)
        plt.title('Tumor')
        plt.imshow(sub_img, cmap='gray')

        # Subplot 3~n: cluster label map with different K
        area = np.sum(sub_mask==1)
        for clu in range(3, max_cluster + 1):
            label_map = pixel_clustering(sub_mask, features, clu)
            plt.subplot(3, (max_cluster + 2) // 3, clu)
            plt.matshow(label_map, fignum=0)
            plt.xlabel(str(clu) + ' clusters', fontsize=15)
        plt.subplots_adjust(hspace=0.3)
        #     plt.subplots_adjust(wspace=0.01)
        plt.suptitle(f'Cluster pattern with multiple resolutions (area = {area})', fontsize=15)

        return fig


def visualize2(img, sub_img, mask, sub_mask, label_map_km=None
            ,label_map_db=None, optimal_k_km=None, optimal_k_db=None):

    # Create a 2x2 grid using plt.subplot
    # Subplot 1: CT image of the whole lung
    fig = plt.figure(figsize=(12, 12))
    plt.subplot(2, 2, 1)
    plt.title('Raw Image')
    plt.imshow(img, cmap='gray')
    plt.scatter(np.where(mask == 1)[1], np.where(mask == 1)[0], marker='o', color='red', s=0.2)

    # Subplot 2: CT iamge of the nodule
    plt.subplot(2, 2, 2)
    plt.title('Tumor')
    plt.imshow(sub_img, cmap='gray')

    # Subplot 3~n: cluster label map with different K
    area = np.sum(sub_mask == 1)

    label_map = label_map_km
    plt.subplot(2, 2, 3)
    plt.matshow(label_map, fignum=0)
    plt.xlabel(str(optimal_k_km) + ' cluster_km')
    #     plt.subplots_adjust(wspace=0.01)

    label_map = label_map_db
    plt.subplot(2, 2, 4)
    plt.matshow(label_map, fignum=0)
    plt.xlabel(str(optimal_k_db) + ' cluster_db')
    plt.subplots_adjust(hspace=0.3)
    plt.suptitle(f'Cluster pattern with multiple resolutions (area = {area})', fontsize=15)
    plt.tight_layout()
    return fig

def calITHscore(label_map, min_area=200, thresh=1):

    size = np.sum(label_map > 0)  # Record the number of total pixels
    num_regions_list = []
    max_area_list = []
    for i in np.unique(label_map)[1:]:  # For each gray level except 0 (background)
        flag = 1  # Flag to count this gray level, in case this gray level has only one pixel
        # Find (8-) connected-components. "num_regions" is the number of connected components
        labeled, num_regions = ndimage.label(label_map==i, structure=ndimage.generate_binary_structure(2,2))
        max_area = 0
        for j in np.unique(labeled)[1:]:  # 0 is background (here is all the other regions)
            # Ignore the region with only 1 or "thresh" px
            if size <= min_area:
                if np.sum(labeled == j) <= thresh:
                    num_regions -= 1
                    if num_regions == 0:  # In case there is only one region
                        flag = 0
                else:
                    temp_area = np.sum(labeled == j)
                    if temp_area > max_area:
                        max_area = temp_area
            else:
                if np.sum(labeled == j) <= 1:
                    num_regions -= 1
                    if num_regions == 0:  # In case there is only one region
                        flag = 0
                else:
                    temp_area = np.sum(labeled == j)
                    if temp_area > max_area:
                        max_area = temp_area
        if flag == 1:
            num_regions_list.append(num_regions)
            max_area_list.append(max_area)
    # Calculate the ITH score
    ith_score = 0
    # print(num_regions_list)
    for k in range(len(num_regions_list)):
        ith_score += float(max_area_list[k]) / num_regions_list[k]
    # Normalize each area with total size
    ith_score = ith_score / size
    ith_score = 1 - ith_score

    return ith_score


def determine_optimal_k(features, max_k=10):

    sse = []  # List to store SSE for each k value
    silhouette_scores = []  # List to store silhouette scores for each k

    for k in range(2, max_k + 1):  # Silhouette score is only defined for k >= 2
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(features)
        sse.append(kmeans.inertia_)  # SSE for the given k

        # Calculate silhouette score for this k
        score = silhouette_score(features, kmeans.labels_)
        silhouette_scores.append(score)

    # Plot the SSE and Silhouette score for each k
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    ax[0].plot(range(2, max_k + 1), sse, marker='o')
    ax[0].set_xlabel('Number of clusters (k)')
    ax[0].set_ylabel('SSE (Inertia)')
    ax[0].set_title('Elbow Method For Optimal k')

    ax[1].plot(range(2, max_k + 1), silhouette_scores, marker='o')
    ax[1].set_xlabel('Number of clusters (k)')
    ax[1].set_ylabel('Silhouette Score')
    ax[1].set_title('Silhouette Score For Optimal k')
    plt.show()

    # Find the optimal k using the elbow method (SSE)
    elbow_point = np.diff(sse).argmin() + 2  # Adding 2 because diff reduces the length by 1

    # Find the optimal k using silhouette score
    optimal_k_silhouette = np.argmax(silhouette_scores) + 2  # Adding 2 because range starts at 2

    print(f"Optimal number of clusters (k) using elbow method: {elbow_point}")
    print(f"Optimal number of clusters (k) using silhouette score: {optimal_k_silhouette}")

    return optimal_k_silhouette,elbow_point  # Or optimal_k_silhouette depending on your preference


def visualize_pixel_features(sub_img, sub_mask, features, feature_name):
    feature_values = np.zeros_like(sub_mask, dtype=float)

    # For each pixel in the mask, assign the feature value to the corresponding pixel
    cnt = 0
    for i in range(len(sub_mask)):
        for j in range(len(sub_mask[0])):
            if sub_mask[i][j] == 1:  # Only for tumor area (mask == 1)
                # Assuming features[feature_name][cnt] returns a list or array
                # Select a specific feature, for example, the first feature (index 0)
                feature_values[i][j] = features[feature_name][cnt][0]  # Use the first feature if it's a list/array
                cnt += 1

    # Plot the feature map (e.g., shape, GLCM feature map)
    plt.figure(figsize=(8, 6))
    plt.imshow(feature_values, cmap='hot', interpolation='nearest')
    plt.title(f'{feature_name} Feature Map')
    plt.colorbar(label=f'{feature_name} value')
    plt.show()


def DCM(img_path, mask_path):
    slices = [pydicom.dcmread(os.path.join(img_path, s), force=True) for s in os.listdir(img_path)]
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

    return image, img


ithscore_data = []

dcm_path = r'./ruxian_pet'
import time
for patient_folder in sorted(os.listdir(dcm_path), key=lambda x: int(x.split('-')[0])):
    start_time = time.time()
    if os.path.isdir(os.path.join(patient_folder, dcm_path)):
        continue

    patient_path = os.path.join(dcm_path, patient_folder)
    studies = [folder for folder in os.listdir(patient_path) if os.path.isdir(os.path.join(patient_path, folder))]
    studies_path = os.path.join(patient_path, studies[0])

    keywords_image = ['left', 'right']
    image_folders = [folder for folder in os.listdir(studies_path)
                    if any(keyword in folder.lower() for keyword in keywords_image)]

    keywords_mask = ['roi', 'all', 'cluster', 'rooi']
    mask_folders = [folder for folder in os.listdir(studies_path)
                    if any(keyword in folder.lower() for keyword in keywords_mask)]


    dimension = 1
    for image_folder in image_folders:
        img_path = os.path.join(studies_path, image_folder)
        mask_path = os.path.join(studies_path, mask_folders[0])

        mask_file_path = os.path.join(mask_path, os.listdir(mask_path)[0])
        image, seg = DCM(img_path, mask_file_path)

        img, mask = get_largest_slice(image, seg, dimension)
        sub_img, sub_mask = locate_tumor(img, mask)


        features = extract_radiomic_features(sub_img, sub_mask, parallel=False)

        # Compare KMeans and DBSCAN clustering and map results to label_map
        kmeans_labels, dbscan_labels, optimal_k_km, optimal_k_db, kmeans_score, dbscan_score = compare_clustering_algorithms(features, sub_mask)

        label_map_km = create_label_map(sub_mask, kmeans_labels)
        label_map_db = create_label_map(sub_mask, dbscan_labels)

        ithscore_km = calITHscore(label_map_km, min_area=190 ,thresh=3)
        ithscore_db = calITHscore(label_map_db, min_area=190 ,thresh=3)

        fig = visualize2(img, sub_img, mask, sub_mask, label_map_km=label_map_km,
                        label_map_db=label_map_db, optimal_k_km=optimal_k_km, optimal_k_db=optimal_k_db)
        plt.show()

        output_dir = os.path.join("./cluster_compare", patient_folder)
        os.makedirs(output_dir, exist_ok=True)

        # fig = visualize(img, sub_img, mask, sub_mask, features, cluster="all")
        fig_filename = os.path.join(output_dir, f"dimension{dimension}.png")
        fig.savefig(fig_filename)
        plt.close(fig)

        ithscore_data.append([patient_folder, ithscore_km, optimal_k_km, kmeans_score, ithscore_db, optimal_k_db, dbscan_score])

    last_time = time.time()
    print('itscore km and db:', ithscore_km, ithscore_db)
    print(patient_folder + ' time cost:' + str(last_time - start_time))

    csv_filename = f"ithscore_km_db_dimension_{dimension}.csv"
    ithscore_df = pd.DataFrame(ithscore_data,
                               columns=["Sample", "ITH-kmeans", "Opt_kmeans", "clu_km_Score" , "ITH-db", "Opt_db", "clu_db_Score"])
    ithscore_df.to_csv(csv_filename, index=False)

    # print(f"ITHscore results saved to '{csv_filename}'")

