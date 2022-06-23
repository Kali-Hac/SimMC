![Python >=3.5](https://img.shields.io/badge/Python->=3.5-blue.svg)
![Tensorflow >=1.14.0](https://img.shields.io/badge/Tensorflow->=1.14.0-yellow.svg)
![Pytorch >=1.1.0](https://img.shields.io/badge/Pytorch->=1.1.0-green.svg)
![Faiss-gpu >= 1.6.3](https://img.shields.io/badge/Faiss->=1.6.3-orange.svg)

# SimMC: Simple Masked Contrastive Learning of Skeleton Representations for Unsupervised Person Re-Identification
By Haocong Rao and Chunyan Miao. In IJCAI 2022 (In press, [**Preprint**](https://arxiv.org/abs/2204.09826)).

## Introduction
This is the official implementation of SimMC presented by "SimMC: Simple Masked Contrastive Learning of Skeleton Representations for Unsupervised Person Re-Identification". The codes are used to reproduce experimental results of the proposed SimMC framework in the [**paper**](https://arxiv.org/pdf/2204.09826).

![image](https://github.com/Kali-Hac/SimMC/blob/main/img/overview.png)
Abstract: Recent advances in skeleton-based person re-identification (re-ID) obtain impressive performance via either hand-crafted skeleton descriptors or skeleton representation learning with deep learning paradigms. However, they typically require skeletal pre-modeling and label information for training, which leads to limited applicability of these methods. In this paper, we focus on unsupervised skeleton-based person re-ID, and present a generic Simple Masked Contrastive learning (SimMC) framework to learn effective representations from unlabeled 3D skeletons for person re-ID. Specifically, to fully exploit skeleton features within each skeleton sequence, we first devise a masked prototype contrastive learning (MPC) scheme to cluster the most typical skeleton features (skeleton prototypes) from different subsequences randomly masked from raw sequences, and contrast the inherent similarity between skeleton features and different prototypes to learn discriminative skeleton representations without using any label. Then, considering that different subsequences within the same sequence usually enjoy strong correlations due to the nature of motion continuity, we propose the masked intra-sequence contrastive learning (MIC) to capture intra-sequence pattern consistency between subsequences, so as to encourage learning more effective skeleton representations for person re-ID. Extensive experiments validate that the proposed SimMC outperforms most state-of-the-art skeleton-based methods. We further show its scalability and efficiency in enhancing the performance of existing models.


## Environment
- Python >= 3.5
- Tensorflow-gpu >= 1.14.0
- Pytorch >= 1.1.0
- Faiss-gpu >= 1.6.3

Here we provide a configuration file to install the extra requirements (if needed):
```bash
conda install --file requirements.txt
```

**Note**: This file will not install tensorflow/tensorflow-gpu, faiss-gpu, pytroch/torch, please install them according to the cuda version of your graphic cards: [**Tensorflow**](https://www.tensorflow.org/install/pip), [**Pytorch**](https://pytorch.org/get-started/locally/). Take cuda 9.0 for example:
```bash
conda install faiss-gpu cuda90 -c pytorch
conda install pytorch==1.1.0 torchvision==0.3.0 cudatoolkit=9.0 -c pytorch
conda install tensorflow==1.14
conda install sklearn
```

## Datasets and Models
We provide three already pre-processed datasets (IAS-Lab, BIWI, KGBD) with various sequence lengths (**f=4/6/8/10/12**) [**here (pwd: 7je2)**](https://pan.baidu.com/s/1R7CEsyMJsEnZGFLqwvchBg) and the pre-trained models [**here (pwd: 12o7)**](https://pan.baidu.com/s/1TxRHEMbojEavhxc2g45nwA). Since we report the average performance of our approach on all datasets, here the provided models may produce better results than the paper. <br/>

Please download the pre-processed datasets and model files while unzipping them to ``Datasets/`` and ``ReID_Models/`` folders in the current directory. <br/>

**Note**: The access to the Vislab Multi-view KS20 dataset and large-scale RGB-based gait dataset CASIA-B are available upon request. If you have signed the license agreement and been granted the right to use them, please email us with the signed agreement and we will share the complete pre-processed KS20 and CASIA-B data. The original datasets can be downloaded here: [IAS-Lab](http://robotics.dei.unipd.it/reid/index.php/downloads), [BIWI](http://robotics.dei.unipd.it/reid/index.php/downloads), [KGBD](https://www.researchgate.net/publication/275023745_Kinect_Gait_Biometry_Dataset_-_data_from_164_individuals_walking_in_front_of_a_X-Box_360_Kinect_Sensor), [KS20](http://vislab.isr.ist.utl.pt/datasets/#ks20), [CASIA-B](http://www.cbsr.ia.ac.cn/english/Gait%20Databases.asp). We also provide the ``Preprocess.py`` for directly transforming original datasets to the formated training and testing data. <br/> 

## Dataset Pre-Processing
To (1) extract 3D skeleton sequences of length **f=6** from original datasets and (2) process them in a unified format (``.npy``) for the model inputs, please simply run the following command: 
```bash
python Pre-process.py 6
```
**Note**: If you hope to preprocess manually (or *you can get the [already preprocessed data (pwd: 7je2)](https://pan.baidu.com/s/1R7CEsyMJsEnZGFLqwvchBg)*), please frist download and unzip the original datasets to the current directory with following folder structure:
```bash
[Current Directory]
├─ BIWI
│    ├─ Testing
│    │    ├─ Still
│    │    └─ Walking
│    └─ Training
├─ IAS
│    ├─ TestingA
│    ├─ TestingB
│    └─ Training
├─ KGBD
│    └─ kinect gait raw dataset
└─ KS20
     ├─ frontal
     ├─ left_diagonal
     ├─ left_lateral
     ├─ right_diagonal
     └─ right_lateral
```
After dataset preprocessing, the auto-generated folder structure of datasets is as follows:
```bash
Datasets
├─ BIWI
│    └─ 6
│      ├─ test_npy_data
│      │    ├─ Still
│      │    └─ Walking
│      └─ train_npy_data
├─ IAS
│    └─ 6
│      ├─ test_npy_data
│      │    ├─ A
│      │    └─ B
│      └─ train_npy_data
├─ KGBD
│    └─ 6
│      ├─ test_npy_data
│      │    ├─ gallery
│      │    └─ probe
│      └─ train_npy_data
└─ KS20
    └─ 6
      ├─ test_npy_data
      │    ├─ gallery
      │    └─ probe
      └─ train_npy_data
```
**Note**: KS20 data need first transforming ".mat" to ".txt". If you are interested in the complete preprocessing of KS20 and CASIA-B, please contact us and we will share. We recommend to directly download the preprocessed data [**here (pwd: 7je2)**](https://pan.baidu.com/s/1R7CEsyMJsEnZGFLqwvchBg).

## Model Usage

To (1) train the unsupervised SimMC to obtain skeleton representations and (2) validate their effectiveness on the person re-ID task on a specific dataset (probe), please simply run the following command:  

```bash
python SimMC.py --dataset KS20 --probe probe

# Default options: --dataset KS20 --probe probe --length 6  --gpu 0
# --dataset [IAS, KS20, BIWI, KGBD]
# --probe ['probe' (the only probe for KS20 or KGBD), 'A' (for IAS-A probe), 'B' (for IAS-B probe), 'Walking' (for BIWI-Walking probe), 'Still' (for BIWI-Still probe)] 
# --length [4, 6, 8, 10, 12] 
# --(mask_x, lambda, t, H, lr, eps, min_samples, etc.) with default settings for each dataset
# --mode [Train (for training), Eval (for testing)]
# --gpu [0, 1, ...]

```
Please see ```SimMC.py``` for more details.

To print evaluation results (Top-1, Top-5, Top-10 Accuracy, mAP) of the best model saved in default directory (```ReID_Models/(Dataset)/(Probe)```), run:

```bash
python SimMC.py --dataset KS20 --probe probe --mode Eval
```

## SimMC for Unsupervised Feature Fine-Tuning

### Download Pre-trained Skeleton Representations
We provide skeleton representations (f=6) extrated from pre-trained state-of-the-art models: [**SGELA (pwd: 8hx8)**](https://pan.baidu.com/s/1_LrPdO1nqegXoAPDAOpcPA), [**MG-SCR (pwd: ox49)**](https://pan.baidu.com/s/1ssHEi1P1N2NdCkJTcq3BlQ), and [**SM-SGE (pwd: hrwx)**](https://pan.baidu.com/s/1P7Jk-SEb3Ix2T9o_NrKTAg). Please download and unzip the files to the current directory.

### Usage
To exploit the SimMC framework to fine-tune above skeleton representations pre-trained by the **source** models (SGELA, MG-SCR, SM-SGE) for the person re-ID task, please simply run the following command:  
```bash
python SimMC-UF.py --dataset KS20 --probe probe --source SM-SGE

# Default options: --dataset KS20 --probe probe --gpu 0
# --source [SGELA, MG-SCR, SM-SGE]
# --dataset [IAS, KS20, BIWI, KGBD]
# --probe ['probe' (the only probe for KS20 or KGBD), 'A' (for IAS-A probe), 'B' (for IAS-B probe), 'Walking' (for BIWI-Walking probe), 'Still' (for BIWI-Still probe)]  
# --(mask_x, lambda, t, H, lr, eps, min_samples, etc.) with default settings for each dataset
# --mode [Train (for training), Eval (for testing)]
# --gpu [0, 1, ...]

```

Please see ```SimMC-UF.py``` for more details.


## Application to Model-Estimated Skeleton Data 

### Estimate 3D Skeletons from RGB-Based Scenes
To apply our SimMC to person re-ID under the large-scale RGB scenes (CASIA B), we exploit pose estimation methods to extract 3D skeletons from RGB videos of CASIA B as follows:
- Step 1: Download [CASIA-B Dataset](http://www.cbsr.ia.ac.cn/english/Gait%20Databases.asp)
- Step 2: Extract the 2D human body joints by using [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose)
- Step 3: Estimate the 3D human body joints by using [3DHumanPose](https://github.com/flyawaychase/3DHumanPose)


We provide already pre-processed skeleton data of CASIA B for **single-condition** (Nm-Nm, Cl-Cl, Bg-Bg) and **cross-condition evaluation** (Cl-Nm, Bg-Nm) (**f=40/50/60**) [**here (pwd: 07id)**](https://pan.baidu.com/s/1_Licrunki68r7F3EWQwYng). 
Please download the pre-processed datasets into the directory ``Datasets/``. <br/>

### Usage
To (1) train the SimMC to obtain skeleton representations and (2) validate their effectiveness on the person re-ID task on CASIA B under **single-condition** and **cross-condition** settings, please simply run the following command:

```bash
python SimMC.py --dataset CAISA_B --probe_type nm.nm --length 40

# --length [40, 50, 60] 
# --probe_type ['nm.nm' (for 'Nm' probe and 'Nm' gallery), 'cl.cl', 'bg.bg', 'cl.nm' (for 'Cl' probe and 'Nm' gallery), 'bg.nm']  
# --(mask_x, lambda, t, H, lr, eps, min_samples, etc.) with default settings
# --gpu [0, 1, ...]

```

Please see ```SimMC.py``` for more details.

## Results
|                    |     KS20 |          |          |          |     KGBD |          |          |          |    IAS-A |          |          |          |
|--------------------|---------:|---------:|---------:|---------:|---------:|---------:|---------:|---------:|---------:|---------:|---------:|---------:|
| Methods            |    top-1 |    top-5 |   top-10 |      mAP |    top-1 |    top-5 |   top-10 |      mAP |    top-1 |    top-5 |   top-10 |      mAP |
| D-13 Descriptors   |     39.4 |     71.7 |     81.7 |     18.9 |     17.0 |     34.4 |     44.2 |      1.9 |     40.0 |     58.7 |     67.6 |     24.5 |
| D-16 Descriptors   |     51.7 |     77.1 |     86.9 |     24.0 |     31.2 |     50.9 |     59.8 |      4.0 |     42.7 |     62.9 |     70.7 |     25.2 |
| PoseGait           |     49.4 |     80.9 |     90.2 |     23.5 |     50.6 |     67.0 |     72.6 |     13.9 |     28.4 |     55.7 |     69.2 |     17.5 |
| SGELA + DF         |     49.7 |     67.0 |     77.1 |     22.2 |     43.7 |     58.7 |     65.0 |      7.1 |     18.0 |     32.1 |     46.2 |     13.5 |
| MG-SCR             |     46.3 |     75.4 |     84.0 |     10.4 |     44.0 |     58.7 |     64.6 |      6.9 |     36.4 |     59.6 |     69.5 |     14.1 |
| SM-SGE + DF        |     49.8 |     78.1 |     85.2 |     11.7 |     43.2 |     58.6 |     64.6 |      7.5 |     38.5 |     63.2 |     73.9 |     15.0 |
| AGE                |     43.2 |     70.1 |     80.0 |      8.9 |      2.9 |      5.6 |      7.5 |      0.9 |     31.1 |     54.8 |     67.4 |     13.4 |
| SGELA              |     45.0 |     65.0 |     75.1 |     21.2 |     38.1 |     53.5 |     60.0 |      4.5 |     16.7 |     30.2 |     44.0 |     13.2 |
| SM-SGE             |     45.9 |     71.9 |     81.2 |      9.5 |     38.2 |     54.2 |     60.7 |      4.4 |     34.0 |     60.5 |     71.6 |     13.6 |
| **SimMC (Ours)**   | **66.4** | **80.7** | **87.0** | **22.3** | **54.9** | **66.2** | **70.6** | **11.7** | **44.8** | **65.3** | **72.9** | **18.7** |
| SGELA + **SimMC**  |     47.3 |     69.7 |     79.3 |     20.1 |     51.7 |     62.7 |     67.9 |     15.1 |     16.8 |     33.3 |     48.7 |     12.0 |
| MG-SCR + **SimMC** |     71.1 |     83.6 |     89.1 |     22.7 |     47.4 |     59.3 |     64.9 |     11.0 |     47.2 |     69.0 |     77.3 |     22.4 |
| SM-SGE + **SimMC** |     67.2 |     82.2 |     88.5 |     23.0 |     47.1 |     59.2 |     64.9 |     10.8 |     51.3 |     69.9 |     75.6 |     27.3 |

|                    |    IAS-B |          |          |          |   BIWI-W |          |          |          |   BIWI-S |          |          |          |
|-------------------:|---------:|---------:|---------:|---------:|---------:|---------:|---------:|---------:|---------:|---------:|---------:|---------:|
| Methods            |    top-1 |    top-5 |   top-10 |      mAP |    top-1 |    top-5 |   top-10 |      mAP |    top-1 |    top-5 |   top-10 |      mAP |
| D-13 Descriptors   |     43.7 |     68.6 |     76.7 |     23.7 |     14.2 |     20.6 |     23.7 |     17.2 |     28.3 |     53.1 |     65.9 |     13.1 |
| D-16 Descriptors   |     44.5 |     69.1 |     80.2 |     24.5 |     17.0 |     25.3 |     29.6 |     18.8 |     32.6 |     55.7 |     68.3 |     16.7 |
| PoseGait           |     28.9 |     51.6 |     62.9 |     20.8 |      8.8 |     23.0 |     31.2 |     11.1 |     14.0 |     40.7 |     56.7 |      9.9 |
| SGELA + DF         |     23.6 |     42.9 |     51.9 |     14.8 |     13.9 |     15.3 |     16.7 |     22.9 |     29.2 |     65.2 |     73.8 |     23.5 |
| MG-SCR             |     32.4 |     56.5 |     69.4 |     12.9 |     10.8 |     20.3 |     29.4 |     11.9 |     20.1 |     46.9 |     64.1 |      7.6 |
| SM-SGE + DF        |     44.3 |     68.2 |     77.5 |     14.9 |     16.7 |     31.0 |     40.2 |     18.7 |     34.8 |     60.6 |     71.5 |     12.8 |
| AGE                |     31.1 |     52.3 |     64.2 |     12.8 |     11.7 |     21.4 |     27.3 |     12.6 |     25.1 |     43.1 |     61.6 |      8.9 |
| SGELA              |     22.2 |     40.8 |     50.2 |     14.0 |     11.7 |     14.0 |     14.7 |     19.0 |     25.8 |     51.8 |     64.4 | **15.1** |
| SM-SGE             |     38.9 |     64.1 |     75.8 |     13.3 |     13.2 |     25.8 |     33.5 |     15.2 |     31.3 |     56.3 |     69.1 |     10.1 |
| **SimMC (Ours)**   | **46.3** | **68.1** | **77.0** | **22.9** | **24.5** | **36.7** | **44.5** | **19.9** | **41.7** | **66.6** | **76.8** |     12.3 |
| SGELA + **SimMC**  |     21.2 |     39.1 |     48.8 |     14.0 |     18.4 |     23.1 |     25.0 |     28.7 |     51.8 |     71.3 |     74.4 |     43.3 |
| MG-SCR + **SimMC** |     52.4 |     72.0 |     78.8 |     29.1 |     25.1 |     37.5 |     46.4 |     20.3 |     28.3 |     51.6 |     64.8 |     10.9 |
| SM-SGE + **SimMC** |     55.3 |     72.6 |     80.3 |     34.1 |     25.9 |     39.2 |     45.2 |     22.4 |     42.6 |     64.8 |     76.2 |     15.4 |

## Model Size & Computational Complexity
| Methods          |  # Params |   GFLOPs |
|------------------|----------:|---------:|
| D-13 Descriptors |         — |        — |
| D-16 Descriptors |         — |        — |
| PoseGait         |     8.93M |   121.60 |
| SGELA + DF       |     9.09M |     7.48 |
| MG-SCR           |     0.35M |     6.60 |
| SM-SGE + DF      |     6.25M |    23.92 |
| AGE              |     7.15M |    37.37 |
| SGELA            |     8.47M |     7.47 |
| SM-SGE           |     5.58M |    22.61 |
| **SimMC (Ours)** | **0.15M** | **0.99** |

## Citation
If you find our work useful for your research, please cite our paper
```bash
@inproceedings{rao2022simmc,
  title={SimMC: Simple Masked Contrastive Learning of Skeleton Representations for Unsupervised Person Re-Identification},
  author={Rao, Haocong and Miao, Chunyan},
  booktitle = {IJCAI},
  publisher = {ijcai.org},
  year      = {2022}
}
```

## License

SimMC is released under the MIT License. Our models and codes must only be used for the purpose of research.
