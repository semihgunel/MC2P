
# Motion Capture and Two-photon dataset

<p align="center">
  <img src="https://user-images.githubusercontent.com/20509861/132999255-34327a13-7ea2-4391-8af9-f00e71f5a14d.png" width="700">
</p>

A tethered Drosophila melanogaster behaving freely while neural and behavior activity is recorded using multi-view infrared cameras and two-photon microscopy imaging. Please check the [publication](todo) for more information.

## Introduction

- Download the files using the following anonymized [link](https://drive.google.com/drive/folders/1i0xUcxp5ptXbpw28p-WFsE8pyp8HnueY?usp=sharing). __During the review process, we give a dataset of 8 of the animals.__ The full dataset will be released with the camera-ready version, on a non-anonymized link. The whole dataset includes 40 different animals and total of 364 folders. The whole dataset is close to 200GB. 
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

## How to Use the Dataset

- Install the dependencies using 
```bash
conda env create -f environment.yml
conda activate mc2p
```

- You can directly read raw files and visualize:

```python
import functional as NF
dff = NF.read_dff('/data/MM/data/201014_G23xU1_Fly1_005')
dff = NF.normalize_video(dff)
plt.imshow(dff[0], cmap='jet')
```
or to load the 2D pose as a time sequence, use

```python
import functional as NF
import augmentation as NA

pr = NF.read_pose_result_mm('/data/MM/data/201014_G23xU1_Fly1_005')
pr = pr + 0 # make it writable
pr = NA.PRunNorm()(pr.reshape(-1, 38, 2)) # preprocess
pr = pr.cpu().data.numpy().reshape(-1, 38, 2) # convert to numpy to visualize
media.show_video(NP.plot_pts2d_video(pr[:100]), height=200)
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

dff, meta = dat[0]
```

You can choose from modalities, dff, resized_dff, pr and angles. To get a multiple synchronized modalities: 

```python
import dataset as ND
dat = ND.DatasetMM(
    path_list=['./201008_G23xU1_Fly1_001'],
    modal1='dff',
    modal2='pr',
    n_frames1=32,
    n_frames2=8,
    aug1=nn.Identity(),
    aug2=nn.Identity(),
    stride=1,
)
((dff, _),  (pr, _)) = dat[0]
```
