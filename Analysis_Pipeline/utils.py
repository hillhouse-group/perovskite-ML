from __future__ import print_function, division, absolute_import
import json
from glob import *
from natsort import natsorted
import os
import numpy as np
from skimage.io import imread
from skimage.transform import warp, AffineTransform
from skimage.feature import register_translation
from tqdm import tqdm
import warnings
import fnmatch


def load_videos_and_exposure(working_dir, indices_keep=None, grad=0, loc=0,
                             videos_kwargs=dict(), expsosure_kwargs=dict()):
    primary_videos, valid_indices_primary = load_primary_videos(
        working_dir, indices_keep=indices_keep, grad=grad, loc=loc,
        **videos_kwargs)
    exp_times, valid_indices_exp = load_exposure(
        working_dir, indices_keep=indices_keep, grad=grad, loc=loc,
        **expsosure_kwargs)

    valid_indices = np.logical_and(valid_indices_primary, valid_indices_exp)
    exp_times = exp_times[valid_indices_primary[valid_indices_exp]]
    primary_videos = primary_videos[valid_indices_exp[valid_indices_primary]]

    return primary_videos, exp_times, valid_indices


def load_exposure(working_dir, metadata_name='MMStack_Pos0_metadata.txt',
                  exposure_keyname='HamamatsuHam_DCAM-Exposure',
                  one_each_primary_video=True, indices_keep=None, grad=0, loc=0):
    """
    Return
    ------
    exposure_times in sec.
    """
    meta_datas, corrupted_index = _load_json_metadatas(
        working_dir, metadata_name, indices_keep, grad=grad, loc=loc)
    
    
    
    exposure_times = []
    for metadata in meta_datas:
        
        """
        ######### Use this if different cameras are used in the same run ######
        exposure_times_i = []
        for frame in natsorted(metadata.keys()):
            exposure_list = fnmatch.filter(metadata[frame].keys(), "*-Exposure*")
            if len(exposure_list)<3:
                continue
            for k in exposure_list:
                try:
                    _ = float(metadata[frame][k])
                    exposure_keyname = k
                    break
                except:
                    continue
            exposure_times_i += [float(metadata[frame][exposure_keyname])]
        exposure_times += [exposure_times_i]
        """
        
        exposure_times.append([float(metadata[key][exposure_keyname])
                               for key in natsorted(metadata.keys())
                               if exposure_keyname in metadata[key]])
            
    
    exposure_times = np.asarray(exposure_times)
    
    if one_each_primary_video:
        exposure_times_flatten = exposure_times[:, 0]
        if not np.all(exposure_times == exposure_times_flatten[:, None]):
            print('Something wrong with this!')
        exposure_times = exposure_times_flatten

    return np.clip(exposure_times, 1, None) / 1000, (~corrupted_index)


def load_pixel_per_um(working_dir, metadata_name='MMStack_Pos0_metadata.txt',
                      indices_keep=None, target_key='PixelSize_um'):
    meta_datas, corrupted_index = _load_json_metadatas(working_dir, metadata_name, indices_keep)
    return np.asarray([metadata['Summary'][target_key] for metadata in meta_datas])


def _load_json_metadatas(working_dir, metadata_name='MMStack_Pos0_metadata.txt',
                         indices_keep=None, grad=0, loc=0):
    all_pathes = np.array(
        natsorted(glob(os.path.join(working_dir, '*', metadata_name))))
    grad_str = 'grad' + str(grad) + '_'
    loc_str = 'loc' + str(loc) + '_'
    all_pathes = np.array([s for s in all_pathes if grad_str in s and loc_str in s])
    all_pathes = _indexing(all_pathes, indices_keep)

    meta_datas = []
    corrupted_index = np.zeros(len(all_pathes), dtype=bool)
    for ix, path in enumerate(all_pathes):
        try:
            with open(path, 'r') as fn:
                meta_datas.append(json.load(fn))
        except json.decoder.JSONDecodeError:
            warnings.warn('corrupted camera metadata file: %s' % path)
            corrupted_index[ix] = True

    return meta_datas, corrupted_index


def load_primary_videos(working_dir, tiff_name='MMStack_Pos0.ome.tif',
                        indices_keep=None, grad=0, loc=0):
    def _load_one_primary_vid(all_pathes):
        for ix, path in enumerate(all_pathes):
            try:
                yield imread(path), ix
            except:  # What is the exception for this?
                yield None
    all_pathes = np.array(
        natsorted(glob(os.path.join(working_dir, '*', tiff_name))))
    grad_str = 'grad' + str(grad) + '_'
    loc_str = 'loc' + str(loc) + '_'
    all_pathes = np.array([s for s in all_pathes if grad_str in s and loc_str in s])
    if all_pathes.shape[0] == 0:
        raise FileNotFoundError('The directory do not have files, please check')
    all_pathes = _indexing(all_pathes, indices_keep)
    primary_vids, valid_indices_ = map(
        np.asarray, zip(*[res for res in _load_one_primary_vid(all_pathes)
                          if res is not None]))

    valid_indices = np.zeros(all_pathes.shape[0], dtype=bool)
    valid_indices[valid_indices_] = True
    return primary_vids, valid_indices


def _indexing(array, indices_keep=None):
    if indices_keep is None:
        return array
    elif isinstance(indices_keep, int):
        return array[:indices_keep]
    elif isinstance(indices_keep, tuple) or \
         (isinstance(indices_keep, list) and len(indices_keep) == 2):
        six, eix = indices_keep
        return array[six:eix]
    elif isinstance(indices_keep, np.ndarray) or isinstance(indices_keep, list):
        indices_keep = np.asarray(indices_keep)
        return array[indices_keep]
    else:
        raise NotImplementedError(
            'type: %s not yet implemented' % type(indices_keep))



def nsun_generator(Ts=120, sep=15):
    return np.repeat(2 ** np.arange(np.ceil(Ts / sep)), sep)[:Ts]


def load_sample_metadata(working_dir, filename=None, is_sample=False):
    if filename is None and is_sample:
        filename = 'sample_info.json'
    elif filename is None:
        filename = 'experiment_info.json'
    filepath = os.path.join(working_dir, filename)
    
    with open(filepath, 'r') as json_file:
        sample_metadata = json.load(json_file)
    return sample_metadata


def compute_photobrightening(video, axis=0):
    video = np.moveaxis(video, axis, 0)
    x = np.vstack([np.arange(video.shape[0]), np.ones(video.shape[0])]).T
    xTx_inv_xT = np.linalg.inv(x.T.dot(x)).dot(x.T) # 2xn matrix
    beta = np.einsum('dn,ntsz->dtsz', xTx_inv_xT, video)

    if np.ma.is_masked(video):
        mask = np.repeat(np.all(video.mask, axis=0)[None, ...], 2, axis=0)
        return np.ma.array(beta, mask=mask)
    else:
        return beta


def movement_canceling(primary_videos, upsample_factor=50, clip_to_min=True):
    est_rw_x, est_rw_y = estimate_movement(primary_videos, upsample_factor)

    primary_videos_nomove = []
    for mx, my, primary_vid in tqdm(zip(est_rw_x, est_rw_y, primary_videos),
                                    total=primary_videos.shape[0],
                                    desc='Movement canceling'):
        frames = cancel_movement_per_primary_vid(primary_vid, mx, my)
        primary_videos_nomove.append(frames)
    primary_videos_nomove = np.asarray(primary_videos_nomove)
    if clip_to_min:
        return np.clip(primary_videos_nomove, primary_videos.min(), None)
    else:
        return primary_videos_nomove



def estimate_movement(primary_videos, upsample_factor=50):
    secondary_mean_vids = primary_videos.mean(1)
    translation_x, translation_y = [], []
    for mv, mvn in zip(secondary_mean_vids[:-1], secondary_mean_vids[1:]):
        movex, movey = register_translation(mv, mvn, upsample_factor)[0]
        translation_x.append(movex)
        translation_y.append(movey)

    translation_x, translation_y = map(np.array, [translation_x, translation_y])
    est_rw_x = np.hstack([[0], -np.cumsum(translation_x)])
    est_rw_y = np.hstack([[0], -np.cumsum(translation_y)])

    return est_rw_x, est_rw_y


def cancel_movement_per_primary_vid(primary_vid, movement_x, movement_y):
    tform = AffineTransform(translation=(-movement_y, -movement_x))
    frames = []
    for frame in primary_vid:
        frames.append(warp(frame, tform.inverse, preserve_range=True))

    return np.array(frames)
