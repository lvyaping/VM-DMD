# VM-DMD

This repo contains the supported code and configuration files to reproduce semantic segmentaion results of [VM-DMD](VM-DMD: Direction-Aware Multi-Directional Distillation of Visual Mamba for Medical Image Segmentation).

## Abstract

Medical image segmentation is confronted with challenges, including data scarcity, expensive manual annotation, and low image quality. Approaches based on state space models offer linear computational complexity that is favorable for clinical deployment, yet they suffer from substantial training overhead and inherent sensitivity to feature directionality. Despite their outstanding performance, Transformer-based models are limited in real-world deployment due to their quadratic computational complexity and substantial resource requirements. To address this challenge, we propose VM-DMD (VM-UNet Direction-aware Multi-directional Distillation), a framework designed to transfer the rich knowledge of a pre-trained Swin Transformer teacher model to a VM-UNet student model. VM-DMD works in two stages: first, a selective weight initialization strategy transfers the teacher model’s deep semantic knowledge layer by layer while preserving the Mamba model’s main state parameters; second, Adaptive Multi-directional Distillation (AMD) aligns the teacher’s direction-invariant global features with the student’s direction-sensitive local features across four geometric transformations (original, transposed, horizontal flip, and vertical flip). This stage also incorporates multi-scale semantic weighting and dynamic loss scheduling to guide training effectively. Comprehensive evaluations on the ISIC 2017/2018 skin lesion and BUSI/BUS breast ultrasound segmentation datasets demonstrate that VM-DMD substantially enhances the student model’s segmentation performance, outperforming both the baseline VM-UNet and the teacher model, while maintaining lower parameter counts and computational complexity. These results show that the proposed framework is effective in achieving efficient and accurate knowledge transfer between different model architectures for medical image segmentation.

## 0. Main Environments

```bash
conda create -n vmdmd python=3.8 -y
conda activate vmdmd
pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html
python -m pip install openmim
python -m mim install mmcv-full==1.3.0
git clone https://github.com/SwinTransformer/Swin-Transformer-Semantic-Segmentation
cd Swin-Transformer-Semantic-Segmentation
pip install -e .
pip install packaging
pip install timm==0.4.12
pip install pytest chardet yacs termcolor
pip install submitit tensorboardX
pip install triton==2.0.0
pip install causal_conv1d==1.0.0  # causal_conv1d-1.0.0+cu118torch1.13cxx11abiFALSE-cp38-cp38-linux_x86_64.whl
pip install mamba_ssm==1.0.1  # mmamba_ssm-1.0.1+cu118torch1.13cxx11abiFALSE-cp38-cp38-linux_x86_64.whl
pip install scikit-learn matplotlib thop h5py SimpleITK scikit-image medpy yacs
conda install -c simpleitk simpleitk
```
The .whl files of causal_conv1d and mamba_ssm could be found here. {[Baidu](https://pan.baidu.com/s/1Tibn8Xh4FMwj0ths8Ufazw?pwd=uu5k) or [GoogleDrive](https://drive.google.com/drive/folders/1ZJjc7sdyd-6KfI7c8R6rDN8bcTz3QkCx?usp=sharing)}


## 1. Prepare the dataset
- The datasets can be found here {[Baidu](https://pan.baidu.com/s/1Ff3qRl8GXHg3kk3O0Ymu5A?pwd=rjhn)}. 

- After downloading the datasets, you are supposed to put them into './data/isic17/', './data/isic18/', './data/bus/', and './data/busi/', and the file format reference is as follows. (take the ISIC17 dataset as an example.)

- './data/isic17/'
  - train
    - images
      - .png
    - masks
      - .png
  - val
    - images
      - .png
    - masks
      - .png
  - test
    - images
      - .png
    - masks
      - .png

## 2. Prepare the pre_trained weights
- The weights of the pre-trained VMamba could be downloaded from [Baidu](https://pan.baidu.com/s/1CXFCdn8hxxCNWWs2Q1GJ5Q?pwd=bkaf).After that, the pre-trained weights should be stored in 'Swin-Transformer-Segmentation/pretrained_weights/'.

## 3. Train the VM-DMD

```bash
cd VM-UNet
bash scripts/train_XXX.sh 4
```
or 

```bash
cd VM-UNet
CUDA_VISIBLE_DEVICES=0,1,3 bash scripts/train_XXX 3
```

**NOTE**: If you want to use the trained checkpoint for inference testing only and save the corresponding test images, you can follow these steps:  

- **In `config_setting`**:  
   - Set the parameter `only_test_and_save_figs` to `True`.  
   - Fill in the path of the trained checkpoint in `best_ckpt_path`.  
   - Specify the save path for test images in `img_save_path`.  

- **Execute the script**:  
   After setting the above parameters, you can run `train_XXX.sh`.

## 4. Obtain the outputs
- After training, you could obtain the results in './results/'

## 5. Acknowledgments
- We thank the authors of [VM-UNet](https://github.com/JCruan519/VM-UNet) and [Swin-Transformer-segmentation](https://github.com/SwinTransformer/Swin-Transformer-Semantic-Segmentation) for their open-source codes.