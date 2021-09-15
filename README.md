
# Motion Capture and Two-photon dataset

<p align="center">
  <img src="https://user-images.githubusercontent.com/20509861/132999255-34327a13-7ea2-4391-8af9-f00e71f5a14d.png" width="700">
</p>

A tethered Drosophila melanogaster behaving freely while neural and behavior activity is recorded using multi-view infrared cameras and two-photon microscopy imaging. Please check the [publication](todo) for more information.

## Introduction

- Download the files using the [link](https://drive.google.com/drive/folders/16BuWqIbahrRSrnyOMsp5bpYwv6UkCIOP?usp=sharing). The dataset includes 8 different animals and total of 133 folders. The whole dataset is close to 200GB. 
The file format will look like this:

```sh
+-- 201008_G23xU1_Fly1_001
|   +-- 2p_dff.mm
|   +-- 2p_dff_resized.mm
|   +-- behData_images_camera_1.mp4
|   +-- sync_indices.pkl
|   +-- pose_result.pkl
|   +-- pose_result_inverse_kinematics.pkl
|   +-- rest.npy
+-- 201008_G23xU1_Fly1_002
+-- 201008_G23xU1_Fly1_003
+-- 201008_G23xU1_Fly1_004
+-- 201008_G23xU1_Fly1_005 
+-- 201008_G23xU1_Fly1_006
...
+-- Readme.md
```

## How to Use

- You can directly read raw files and visualize them:

```python
import functional import NF
import matplotlib.pyplot as plt
import mediapy as media

dff = NF.read_dff('./201008_G23xU1_Fly1_001')
dff = NF.normalize_video(dff)
plt.imshow(dff[0])
```
or to load the 2D pose as a time sequence, use

```python
import functional import NF
import augmentation import PRUnnorm

# raw 2d data
pr = NF.read_pr('./201008_G23xU1_Fly1_001')
# preprocess
pr = PRUnnorm()(pr)

media.show_video(NP.plot_pts2d_video(pr))
```
- You can use pytorch dataloaders to train a model. To get a single modality:
```python
import dataset as ND
dat = ND.DatasetUM(
    path_list=['./201008_G23xU1_Fly1_001'],
    n_frames=32,
    modal='dff',
    stride=1
)

dff, _ = dat[0]
```

You can choose from modalities, dff, resized_dff, pr and angles. To get a multiple synchronized modalities: 

```python
import dataset as ND
ND.DatasetMM(
    path_list=['./201008_G23xU1_Fly1_001'],
    modal1='dff',
    modal2='pr',
    n_frames1=32,
    n_frames2=8,
    stride=1,
),
(dff, _), (pr, _) = dat[0]
```

## BibTeX
```bash
@InProceedings{gunel21,
  title = "Contrastive Learning of Neural Representations using Animal Behavior",
  booktitle = "arXiv",
  year = "2021"
}
```
