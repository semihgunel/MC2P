import bisect
from os import remove
import random
from typing import Callable, Dict, List, Optional, Set, Tuple
from augmentation import DFFTransformEval, TimeSeriesTransformEval
from torch import Tensor

import numpy as np
import pytorch_lightning as pl
import torch
import torchvision.transforms.functional as TF
from einops import repeat
from skimage.transform import resize
from torch import nn
from torch.utils.data import DataLoader, Dataset
import functional as NF


def clipmd_to_clip(clip_middle: int, n_frames: int, stride: int, upper: int = None):
    """
        >>> clipmd_to_clip(100, 3, 3):
    """
    clip = np.arange(
        clip_middle - ((n_frames // 2) * stride),
        clip_middle + ((n_frames // 2) * stride) + n_frames % 2,
        stride,
    )
    if upper is not None:
        clip = np.clip(clip, 0, upper - 1)
    return clip


def get_clipsmid(
    movie_len: List[int], n_frames: int, stride: int, rest: List[List[bool]],
) -> List[List[int]]:
    """ calculate possible cmid locations, given path list, 
        expected number of frames in each clip and stride between the clips
    """
    return [
        get_clipsmid_single(m, n_frames, stride, rest[idx])
        for idx, m in enumerate(movie_len)
    ]


# fmt: off
def get_clipsmid_single( movie_len: int, n_frames: int, stride: int, rest: bool = None) -> List[int]:
    ''' clipmid is removed if rest[clipmid]'''
    is_rest = np.zeros((movie_len,), dtype=bool) if rest is None else rest
    #st, end = n_frames // 2, movie_len - (n_frames // 2)
    st, end = 0, movie_len
    c = [mid for mid in np.arange(st, end, stride) if not is_rest[mid]]

    return c
# fmt: on


class DatasetBase(torch.utils.data.Dataset):
    def __init__(
        self, path_list: List[str], n_frames: int, clips_mid: List[int], transform
    ):
        self.path_list = path_list
        """ replaces unique identity with integers
          >>> ['a', 'b', 'c', 'a'] -> [0, 1, 2, 0]
        """
        _, self.identity_list = np.unique(
            [NF.path2ident(p) for p in path_list], return_inverse=True
        )
        _, self.trial_list = np.unique(
            [NF.path2trial(p) for p in path_list], return_inverse=True
        )
        self.n_frames = n_frames
        self.clips_mid = clips_mid
        self.transform = transform

        assert len(path_list) == len(clips_mid)

        clip_lengths = torch.as_tensor([len(v) for v in self.clips_mid])
        self.cumulative_sizes = clip_lengths.cumsum(0).tolist()

    def get_clip_location(self, idx):
        """ converts a flattened representation of the indices into a video_idx, clip_idx representation. used by get_clip_mid """
        video_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        clip_idx = idx - (self.cumulative_sizes[video_idx - 1] if video_idx != 0 else 0)
        return video_idx, clip_idx

    def get_clip_mid(self, idx):
        """ given idx returns video_idx and clip_mid. used by get_clip
            **extended in the subclasses**
        """
        video_idx, clip_idx = self.get_clip_location(idx)
        clip_mid = self.clips_mid[video_idx][clip_idx]
        return video_idx, clip_mid

    def get_clip(self, idx):
        """ converts clip mid into clip """
        video_idx, clip_mid = self.get_clip_mid(idx)
        clip = clipmd_to_clip(
            clip_mid,
            self.n_frames,
            self.get_stride(video_idx),
            self.data[video_idx].shape[0],
        )
        return video_idx, clip

    def __getitem__(self, idx):
        video_idx, clip = self.get_clip(idx)
        return self.getitem(video_idx, clip)

    def getitem(self, video_idx, clip):
        return self.data(video_idx, clip)

    def get_stride(self, video_idx):
        return 1

    def get_meta(self, video_idx: int, clip: int):
        return {
            "video_idx": video_idx,
            "clip": clip,
            "identity": self.identity_list[video_idx],
            "trial": self.trial_list[video_idx],
        }

    def __len__(self):
        return self.cumulative_sizes[-1]

    def path2video_idx(self, s):
        return self.path_list.index(s)


class DatasetNeuralImage(DatasetBase):
    def __init__(
        self,
        path_list: List[str],
        n_frames: int,
        clips_mid: List[np.ndarray],
        f: Callable[[str], np.ndarray],
        transform: Callable[[Tensor], Tensor] = nn.Identity(),
    ):
        super().__init__(path_list, n_frames, clips_mid, transform)
        self.data = [f(p) for p in path_list]
        self.is_beh = False

    def resize_video(self, video, res=64):
        video_res = np.zeros((video.shape[0], res, res))
        for i in range(video_res.shape[0]):
            video_res[i] = resize(video[i], (res, res))
        return video_res

    def getitem(self, video_idx, clip):
        """ clip is asssumed to be in neural indices """

        assert not np.all(clip > self.data[video_idx].shape[0])
        assert not np.all(clip < 0)
        clip = np.clip(clip, 0, self.data[video_idx].shape[0] - 1)

        video = self.data[video_idx][clip]
        # video = np.clip(video, -30, 30)
        # video = self.resize_video(video, 64)
        video = torch.from_numpy(video + 0)  # convert to writeable
        video = repeat(video, "t h w -> c t h w", c=1)

        for i in range(video.size(0)):
            video[i] = self.transform(video[i])

        return video.float(), self.get_meta(video_idx, clip)


class DatasetNeuralLatent(DatasetBase):
    def __init__(
        self,
        path_list: List[str],
        n_frames: int,
        clips_mid: List[np.ndarray],
        f: Callable[[str], np.ndarray],
        transform: Callable[[Tensor], Tensor] = nn.Identity(),
    ):
        super().__init__(path_list, n_frames, clips_mid, transform)
        self.data = [f(p) for p in path_list]
        self.is_beh = False

    def getitem(self, video_idx, clip):
        """ clip is asssumed to be in neural indices """

        assert not np.all(clip > self.data[video_idx].shape[0])
        assert not np.all(clip < 0)
        clip = np.clip(clip, 0, self.data[video_idx].shape[0] - 1)

        video = self.data[video_idx][clip]
        video = torch.from_numpy(video + 0)  # convert to writeable

        video = self.transform(video)

        return video.float(), self.get_meta(video_idx, clip)


class DatasetBehav(DatasetBase):
    """ Base class"""

    def __init__(
        self,
        path_list: List[str],
        n_frames: int,
        f: Callable,
        clips_mid: List[np.ndarray],
        transform=nn.Identity(),
    ):
        super().__init__(path_list, n_frames, clips_mid, transform)
        self.data = [f(p) for p in path_list]
        self.is_beh = True

    def getitem(self, video_idx, clip):
        """ returns data in (t, jd) format"""

        assert not np.all(clip > self.data[video_idx].shape[0])
        assert not np.all(clip < 0)
        clip = np.clip(clip, 0, self.data[video_idx].shape[0] - 1)

        pts = self.data[video_idx][clip] + 0  # to read into memory
        pts = self.transform(pts)

        pts = torch.from_numpy(pts) if not torch.is_tensor(pts) else pts
        return pts.float(), self.get_meta(video_idx, clip)

    # fmt: off
    def get_stride(self, video_idx: str) -> int:
        path = self.path_list[video_idx]
        return (3 if ("G23xU1" in path and "CLC" not in path and "ABO" not in path) else 1)
    # fmt: on


from nelydataloader.augmentation import TimeSeriesTransformEval
import pandas as pd
from sklearn import preprocessing


class DatasetMM(torch.utils.data.Dataset):
    """multi modal dataset. loads aligned modalities at the same time in a tuple. 
        first: pose result, second: dff
    """

    def __init__(
        self, path_list, modal1, modal2, aug1, aug2, n_frames1, n_frames2, stride,
    ):
        self.path_list = path_list
        self.dataset1 = DatasetUM(
            path_list, n_frames1, modal1, stride=stride, transform=aug1
        )
        self.dataset2 = DatasetUM(
            path_list, n_frames2, modal2, stride=stride, transform=aug2
        )

        # make sure the clips mid are the same across datasets
        assert all(
            np.array_equal(c1, c2)
            for (c1, c2) in zip(
                self.dataset1.clips_mid_behav, self.dataset2.clips_mid_behav
            )
        )

    def __getitem__(self, i):
        return self.dataset1[i], self.dataset2[i]

    def __len__(self):
        assert len(self.dataset1) == len(self.dataset2)
        return len(self.dataset2)


class DatasetUM(torch.utils.data.Dataset):
    """uni modal dataset. loads aligned modalities at the same time in a tuple. 
        modal can be pr or dff
    """

    def __init__(
        self,
        path_list: List[str],
        n_frames: int,
        modal: str,
        stride: int = 1,
        transform: Callable = nn.Identity(),
        remove_rest: bool = True,
    ):
        self.path_list = path_list
        movie_len = [NF.read_pose_result_mm(p).shape[0] for p in path_list]
        rest = [
            NF.read_rest(p) if remove_rest else torch.zeros_like(NF.read_rest(p))
            for idx, p in enumerate(path_list)
        ]
        self.clips_mid_behav = get_clipsmid(movie_len, n_frames, stride, rest)
        self.dataset = None
        if modal == "pr":
            self.dataset = DatasetBehav(
                self.path_list,
                f=NF.read_pose_result_mm,
                n_frames=n_frames,
                clips_mid=self.clips_mid_behav,
                transform=transform,
            )
        elif modal == "angles":
            self.dataset = DatasetBehav(
                self.path_list,
                f=NF.read_angles,
                n_frames=n_frames,
                clips_mid=self.clips_mid_behav,
                transform=transform,
            )

        indices = {p: NF.read_indices(p) for p in self.path_list}
        self.clips_mid_neural = [
            [indices[path_list[v]][c] for c in self.clips_mid_behav[v]]
            for v in range(len(self.clips_mid_behav))
        ]
        if modal == "dff":
            self.dataset = DatasetNeuralImage(
                path_list=path_list,
                n_frames=n_frames,
                clips_mid=self.clips_mid_neural,
                f=NF.read_dff,
                transform=transform,
            )
        elif modal == "resized_dff":
            self.dataset = DatasetNeuralImage(
                path_list=path_list,
                n_frames=n_frames,
                clips_mid=self.clips_mid_neural,
                f=NF.read_resized_dff,
                transform=transform,
            )
        elif modal == "resized_dff_ae":
            self.dataset = DatasetNeuralLatent(
                path_list=path_list,
                n_frames=n_frames,
                clips_mid=self.clips_mid_neural,
                f=NF.read_resized_dff_ae,
                transform=transform,
            )
        assert self.dataset is not None, f"{modal} is not in the list of DatasetUM"

    def __getitem__(self, i):
        return self.dataset[i]

    def getitem(self, v, c):
        return self.dataset.getitem(v, c)

    @property
    def is_beh(self):
        return self.dataset.is_beh

    def __len__(self):
        return len(self.dataset)

    def get_stride(self, video_idx):
        return self.dataset.get_stride(video_idx)
