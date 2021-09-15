from typing import Callable, Dict, List, Optional, Set, Tuple

import cv2
import numpy as np


def __plot_pts2d_frame(
    pts2d: np.ndarray,
    bones: List[List[int]],
    colors: List[Tuple],
    img: np.ndarray = None,
) -> np.ndarray:
    assert pts2d.ndim == 2
    assert pts2d.shape[0] == 19

    pts2d = pts2d.astype(int)

    img = np.zeros((480, 960, 3)) if img is None else img
    for idx, bone in enumerate(bones):
        img = cv2.line(
            img,
            (pts2d[bone[0]][0], pts2d[bone[0]][1]),
            (pts2d[bone[1]][0], pts2d[bone[1]][1]),
            colors[idx],
            thickness=10,
        )
    return img


def plot_pts2d_frame(pts2d: np.ndarray):
    assert pts2d.ndim == 2
    assert pts2d.shape[0] == 38

    # bones = np.stack([np.arange(0, 14), np.arange(1, 15)]).T
    bones = np.concatenate([np.arange(0, 5), np.arange(10, 14), np.arange(5, 10)])
    bones = np.stack([bones, bones + 1]).T
    bones = [b for b in bones if (b[1] % 5) != 0]

    # taken from https://github.com/NeLy-EPFL/DeepFly3D/blob/master/deepfly/skeleton_fly.py
    colors = [
        (255, 0, 0),
        (0, 0, 255),
        (0, 255, 0),
        (255, 255, 0),
        (255, 0, 255),
        (0, 255, 255),
    ]

    flatten_list = lambda x: sum(x, [])

    # fmt: off
    colors_right = flatten_list([[colors[i] for _ in range(4)] for i in range(3)]) # each leg has 5 joints
    img = __plot_pts2d_frame(pts2d=pts2d[:19] + [300, 240], bones=bones, colors=colors_right)
    colors_left = flatten_list([[colors[i] for _ in range(4)] for i in range(3,6)]) # each leg has 5 joints
    img = __plot_pts2d_frame(pts2d=pts2d[19:] + [660, 240], img=img, bones=bones, colors=colors_left)
    # fmt: on

    return img


def plot_pts2d_video(pts2d: np.ndarray) -> np.ndarray:
    """ accepts (t, j, 2), returns a video in (t, h, w) """
    assert pts2d.ndim == 3
    assert pts2d.shape[1] == 38

    return [plot_pts2d_frame(p) for p in pts2d]
