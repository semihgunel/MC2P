import glob
import os
import pickle
from pathlib import Path
import torch
from typing import Callable, Dict, List, Optional, Set, Tuple

import numpy as np
import re


def read_dff(p: str) -> np.ndarray:
    n = np.memmap(get_dff_path(p), dtype="float32", mode="r").size // 128 // 128
    return np.memmap(get_dff_path(p), dtype="float32", mode="r", shape=(n, 128, 128))


def read_resized_dff(p: str) -> np.ndarray:
    n = np.memmap(get_resized_dff_path(p), dtype="float32", mode="r").size // 64 // 64
    return np.memmap(
        get_resized_dff_path(p), dtype="float32", mode="r", shape=(n, 64, 64)
    )


def read_resized_dff_ae(p: str) -> np.ndarray:
    n = np.memmap(get_resized_dff_ae_path(p), dtype="float32", mode="r").size // 64
    return np.memmap(
        get_resized_dff_ae_path(p), dtype="float32", mode="r", shape=(n, 64)
    )
  
 
def read_warped_green(p: str) -> np.ndarray:
    n = (
        np.memmap(get_warped_green_path(p), dtype="float32", mode="r").size
        // 128
        // 128
    )
    return np.memmap(
        get_warped_green_path(p), dtype="float32", mode="r", shape=(n, 128, 128)
    )


def read_raw_green(p: str) -> np.ndarray:
    n = np.memmap(get_raw_green_path(p), dtype="float32", mode="r").size // 128 // 128
    return np.memmap(
        get_raw_green_path(p), dtype="float32", mode="r", shape=(n, 128, 128)
    )


def read_raw_affine_green(p: str) -> np.ndarray:
    n = (
        np.memmap(get_raw_affine_green_path(p), dtype="float32", mode="r").size
        // 128
        // 128
    )
    return np.memmap(
        get_raw_affine_green_path(p), dtype="float32", mode="r", shape=(n, 128, 128)
    )


def read_raw_red(p: str) -> np.ndarray:
    n = np.memmap(get_raw_red_path(p), dtype="float32", mode="r").size // 128 // 128
    return np.memmap(
        get_raw_red_path(p), dtype="float32", mode="r", shape=(n, 128, 128)
    )


def read_raw_affine_red(p: str) -> np.ndarray:
    n = (
        np.memmap(get_raw_affine_red_path(p), dtype="float32", mode="r").size
        // 128
        // 128
    )
    return np.memmap(
        get_raw_affine_red_path(p), dtype="float32", mode="r", shape=(n, 128, 128)
    )


def read_angles(p: str):
    n = np.memmap(get_angles_path(p), dtype="float32", mode="r").size // 42
    return np.memmap(get_angles_path(p), dtype="float32", mode="r", shape=(n, 42))


def read_pose_result_mm(p: str):
    n = np.memmap(get_pose_result_mm_path(p), dtype="float32", mode="r").size // 76
    return np.memmap(
        get_pose_result_mm_path(p), dtype="float32", mode="r", shape=(n, 76)
    )


def read_indices(p: str):
    return np.load(get_alignment_path(p))


def read_rgb_mask(p: str):
    return np.load(get_rgb_mask_path(p))


def read_rest(p: str):
    return np.load(get_rest_path(p))


def get_rgb_path(p):
    return glob.glob(os.path.join(p, "*camera_1.mp4.mm"))[0]


def get_pose_result_path(p: str):
    return glob.glob(os.path.join(p, "*pose_result*.pkl"))[0]


def get_angles_path(p: str):
    return glob.glob(os.path.join(p, "angles*.pkl"))[0]


def get_pose_result_mm_path(p: str) -> str:
    return glob.glob(os.path.join(p, "*pose_result*.mm"))[0]

def get_dff_path(p: str) -> str:
    return glob.glob(os.path.join(p, "*2p_dff_tif_padded.mm"))[0]


def get_resized_dff_path(p: str) -> str:
    return glob.glob(os.path.join(p, "*2p_dff_tif_padded_resized.mm"))[0]


def get_resized_dff_ae_path(p: str) -> str:
    return glob.glob(os.path.join(p, "*2p_dff_tif_padded_resized_ae.mm"))[0]


def get_dff_compressed_path(p: str) -> str:
    return glob.glob(os.path.join(p, "*2p_dff_tif_padded.mm_compressed.mm*"))[0]


def get_warped_green_path(p: str) -> str:
    return glob.glob(os.path.join(p, "*2p_warped_green_tif.mm"))[0]


def get_raw_red_path(p: str) -> str:
    """ mistakenly saved red channel as red :/ """
    return glob.glob(os.path.join(p, "*raw_green.mm"))[0]


def get_raw_affine_red_path(p: str) -> str:
    """ raw red with affine registration to the first frame """
    return glob.glob(os.path.join(p, "*raw_affine_red.mm"))[0]


def get_raw_green_path(p: str) -> str:
    return glob.glob(os.path.join(p, "*raw_red.mm"))[0]


def get_raw_affine_green_path(p: str) -> str:
    """ raw red with affine registration to the first frame """
    return glob.glob(os.path.join(p, "*raw_affine_green.mm"))[0]


def get_roi_traces_path(p: str) -> str:
    return glob.glob(os.path.join(p, "*roi_traces.mm"))[0]
  

def get_alignment_path(p: str) -> str:
    return glob.glob(os.path.join(p, "indices.npy"))[0]


def get_neighbors_path(p: str) -> str:
    return glob.glob(os.path.join(p, "behav_nn_nf7.pkl"))[0]


def get_rgb_mask_path(p: str) -> str:
    return glob.glob(os.path.join(p, "rgb_mask.npy"))[0]


def get_rest_path(p: str) -> str:
    return glob.glob(os.path.join(p, "rest.npy"))[0]


def get_2p_mp4_path(p: str) -> str:
    return glob.glob(os.path.join(p, "*tif.mp4"))[0]
