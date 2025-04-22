# PEHTrans
The code for the paper " PEHTrans: A Hybrid Network with Parallel Encoders and Hierarchical Transformers for Multi-Phase Breast Cancer Segmentation " submitted to Neurocomputing

# Usage
## 1.Installation
### Install PEHTrans as below
```
cd PEHTrans
pip install -e .
```
## 2.Pre-processing
All compared methods use the same pre-processing steps as nnUNet.

You can perform data preprocessing through the following command
```
nnUNetv2_plan_and_preprocess -d DATASET_ID --verify_dataset_integrity
```
## 3.Training
Training models is done with the command.
```
nnUNetv2_train DATASET_ID  UNET_CONFIGURATION   FOLD  -tr PEHTransTrainerDeepSupervision
```
UNET_CONFIGURATION is a string that identifies the requested U-Net configuration (defaults: 2d, 3d_fullres, 3d_lowres, 3d_cascade_lowres).It should be noted that our network only supports 3d input. DATASET_NAME_OR_ID specifies what dataset should be trained on and FOLD specifies which fold of the 5-fold-cross-validation is trained.
## 4.Validation
Validation is also done with the command.
```
nnUNetv2_predict -d  DATASET_ID  -i   "your_nnUNet_raw_path/imagesTs/"  -o   "your_output_path"   -f FOLD  -c UNET_CONFIGURATION -tr PEHTransTrainerDeepSupervision

```
# Dataset
A total of two datasets were used in our paper,among which a private datasets was breast cancer, and the other dataset was a publicly available breast tumor dataset.If you wish to download this publicly available dataset, please refer to the relevant [paper](https://arxiv.org/abs/2406.13844v1)[GitHub](https://github.com/LidiaGarrucho/MAMA-MIA)



# Baseline Models
[nnUNet](https://github.com/MIC-DKFZ/nnUNet)
[nnFormer](https://github.com/282857341/nnFormer)
[PA-Net](https://github.com/Houjunfeng203934/PA-Net)
[PLHN](https://github.com/ZhouL-lab/PLHN)
[PHTrans](https://github.com/lseventeen/PHTrans)


# Acknowledgements
Part of codes are reused from the nnU-Net. Thanks to Fabian Isensee for the codes of nnU-Net.
