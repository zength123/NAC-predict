# NAC-predict

Data Folder Structure

radiomics_features/
Extracted radiomics feature tables (derived from imaging data), provided as de-identified tabular files.

dl_deep_features/
Deep learning feature embeddings , extracted from the penultimate layer of the deep learning model.

fusion_deep_features/
Feature-fusion model embeddings, per-case 128-dimensional deep fused features ,

Scripts
build_resnet3D.py
Training and inference pipeline for the 3D ResNet deep learning model.

build_radiomics.py
Radiomics model construction.

feature_fusion.py
Feature-level fusion model integrating deep learning and radiomics features.

fea_extract.py
Radiomics feature extraction.

crop.py
ROI cropping and preprocessing of tumor regions.

tonii.py
Image format conversion utilities.

clinic.py
Clinical feature processing and integration.

ITH.py
Intratumoral heterogeneity analysis based on imaging-derived features.

README.md
This documentation file.
