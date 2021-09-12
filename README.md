
# Motion Capture and Two-photon dataset.

<p align="center">
  <img src="https://user-images.githubusercontent.com/20509861/132999255-34327a13-7ea2-4391-8af9-f00e71f5a14d.png" width="700">
</p>
A tethered Drosophila melanogaster behaving freely while neural and behavior activity is recorded using multi-view infrared cameras and two-photon microscopy imaging. Please read the [publication](todo) for more information.

## Introduction

- Download the files using the [link](todo). The whole dataset is close to 200GB. 
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
import nelydataloader.functional import NF
import matplotlib.pyplot as plt
dff = NF.read_dff('./201008_G23xU1_Fly1_001')

# normalize and colormap
plt.imshow(dff[0])
```
or to load the 2D pose as a time sequence, use

```python
import nelydataloader.functional import NF
pr = NF.read_pr('./201008_G23xU1_Fly1_001')
```


```python
import nelydataloader.dataset as ND
ND.DatasetUM(
    path_list=['./201008_G23xU1_Fly1_001'],
    n_frames=32,
    modal='dff',
    stride=1
)
```



## BibTeX
```bash
@InProceedings{gunel21,
  title = "Contrastive Learning of Neural Representations using Animal Behavior",
  booktitle = "arXiv",
  year = "2021"
}
```
