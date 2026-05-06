# NAC-predict
This work was conducted by the Jiangsu Key Laboratory of Intelligent Medical Image Computing, Nanjing University of Information Science and Technology.

Please visit our laboratory homepage for more information about our research:

- Lab homepage: https://imic.nuist.edu.cn/index.htm
- GitHub organization: https://github.com/imicjs

We welcome academic exchange, feedback, and potential collaborations related to medical image computing, radiomics, deep learning, and AI-assisted oncology.

Data Folder Structure: 

radiomics_features/
Extracted radiomics feature tables (derived from imaging data), provided as de-identified tabular files.
dl_deep_features/
Deep learning feature embeddings , extracted from the penultimate layer of the deep learning model.
fusion_deep_features/n
Feature-fusion model embeddings, per-case 128-dimensional deep fused features ,

Scripts: 

build_resnet3D.py
Training and inference pipeline for the 3D ResNet deep learning model.
build_radiomics.py/
Radiomics model construction.
feature_fusion.py/
Feature-level fusion model integrating deep learning and radiomics features.
fea_extract.py/
Radiomics feature extraction.
crop.py/
ROI cropping and preprocessing of tumor regions.
tonii.py/
Image format conversion utilities.
ITH.py/
Intratumoral heterogeneity analysis based on imaging-derived features.
README.md/
This documentation file.
checkpoints are avaiable from : https://huggingface.co/zength123/nac_prediction/tree/main
