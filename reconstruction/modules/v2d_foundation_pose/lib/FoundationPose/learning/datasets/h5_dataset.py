# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.



import os,sys,h5py,bisect,io,json
code_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(f'{code_dir}/../../../../')
from Utils import *
from learning.datasets.pose_dataset import *


#: Measured peak of the full-res depth -> xyz map step: ~46-48 bytes per pixel per pose (kornia
#: warp grid + xyz map). Rounded up so the estimated chunk rarely needs the OOM fallback.
_XYZMAP_BYTES_PER_PIXEL = 56
_XYZMAP_MEMORY_FRACTION = 0.95


def _xyzmap_chunk_size(bs, H_ori, W_ori):
  free, _ = torch.cuda.mem_get_info()
  free += torch.cuda.memory_reserved() - torch.cuda.memory_allocated()  # cached blocks are reusable
  per_pose = _XYZMAP_BYTES_PER_PIXEL * H_ori * W_ori
  return int(max(1, min(bs, free * _XYZMAP_MEMORY_FRACTION // per_pose)))


def crop_depth_to_crop_xyzmap(depths, crop_to_oris, tf_to_crops, Ks, H_ori, W_ori, H, W):
  '''Crop depth -> full-res depth -> xyz map -> crop xyz map (B,3,H,W).

  The full-res intermediates do not fit on a 24GB GPU for a whole hypothesis batch at
  2028x1520, so poses (independent here) are processed in chunks sized to the free GPU memory.
  The chunk is halved on OOM. The result is identical to the one-shot version.
  '''
  bs = len(crop_to_oris)
  depths = depths.cuda().expand(bs,-1,-1,-1)
  chunk = _xyzmap_chunk_size(bs, H_ori, W_ori)
  while True:
    try:
      out = []
      for s in range(0, bs, chunk):
        depth_ori = kornia.geometry.transform.warp_perspective(depths[s:s+chunk], crop_to_oris[s:s+chunk], dsize=(H_ori, W_ori), mode='nearest', align_corners=False)
        xyz_map = depth2xyzmap_batch(depth_ori[:,0], Ks[s:s+chunk], zfar=np.inf).permute(0,3,1,2)  #(b,3,H_ori,W_ori)
        del depth_ori
        out.append(kornia.geometry.transform.warp_perspective(xyz_map, tf_to_crops[s:s+chunk], dsize=(H,W), mode='nearest', align_corners=False))
        del xyz_map
      logging.info(f'xyz map: {bs} poses at {W_ori}x{H_ori} in chunks of {chunk}')
      xyz_maps = torch.cat(out)
      del out
      # Hand the cached full-res blocks back to the driver: nvdiffrast and TensorRT allocate
      # outside PyTorch's caching allocator and would otherwise find the GPU full.
      torch.cuda.empty_cache()
      return xyz_maps
    except torch.OutOfMemoryError:
      if chunk == 1:
        raise
      out = None
      torch.cuda.empty_cache()
      chunk = max(1, chunk//2)





class PairH5Dataset(torch.utils.data.Dataset):
  def __init__(self, cfg, h5_file, mode='train', max_num_key=None, cache_data=None):
    self.cfg = cfg
    self.h5_file = h5_file
    self.mode = mode

    logging.info(f"self.h5_file:{self.h5_file}")
    self.n_perturb = None
    self.H_ori = None
    self.W_ori = None
    self.cache_data = cache_data

    if self.mode=='test':
      pass
    else:
      self.object_keys = []
      key_file = h5_file.replace('.h5','_keys.pkl')
      if os.path.exists(key_file):
        with open(key_file, 'rb') as ff:
          self.object_keys = pickle.load(ff)
        logging.info(f'object_keys loaded#:{len(self.object_keys)} from {key_file}')
        if max_num_key is not None:
          self.object_keys = self.object_keys[:max_num_key]
      else:
        with h5py.File(h5_file, 'r', libver='latest') as hf:
          for k in hf:
            self.object_keys.append(k)
            if max_num_key is not None and len(self.object_keys)>=max_num_key:
              logging.info("break due to max_num_key")
              break

      logging.info(f'self.object_keys#:{len(self.object_keys)}, max_num_key:{max_num_key}')

      with h5py.File(h5_file, 'r', libver='latest') as hf:
        group = hf[self.object_keys[0]]
        cnt = 0
        for k_perturb in group:
          if 'i_perturb' in k_perturb:
            cnt += 1
          if 'crop_ratio' in group[k_perturb]:
            self.cfg['crop_ratio'] = float(group[k_perturb]['crop_ratio'][()])
          if self.H_ori is None:
            if 'H_ori' in group[k_perturb]:
              self.H_ori = int(group[k_perturb]['H_ori'][()])
              self.W_ori = int(group[k_perturb]['W_ori'][()])
            else:
              self.H_ori = 540
              self.W_ori = 720
        self.n_perturb = cnt
        logging.info(f'self.n_perturb:{self.n_perturb}')


  def __len__(self):
    if self.mode=='test':
      return 1
    return len(self.object_keys)



  def transform_depth_to_xyzmap(self, batch:BatchPoseData, H_ori, W_ori, bound=1):
    bs = len(batch.rgbAs)
    H,W = batch.rgbAs.shape[-2:]
    mesh_radius = batch.mesh_diameters.cuda()/2
    tf_to_crops = batch.tf_to_crops.cuda()
    crop_to_oris = batch.tf_to_crops.inverse().cuda()  #(B,3,3)
    batch.poseA = batch.poseA.cuda()
    batch.Ks = batch.Ks.cuda()

    if batch.xyz_mapAs is None:
      batch.xyz_mapAs = crop_depth_to_crop_xyzmap(batch.depthAs, crop_to_oris, tf_to_crops, batch.Ks, H_ori, W_ori, H, W)
    batch.xyz_mapAs = batch.xyz_mapAs.cuda()
    if self.cfg['normalize_xyz']:
      invalid = batch.xyz_mapAs[:,2:3]<0.001
    batch.xyz_mapAs = batch.xyz_mapAs-batch.poseA[:,:3,3].reshape(bs,3,1,1)
    if self.cfg['normalize_xyz']:
      batch.xyz_mapAs *= 1/mesh_radius.reshape(bs,1,1,1)
      invalid = invalid.expand(bs,3,-1,-1) | (torch.abs(batch.xyz_mapAs)>=2)
      batch.xyz_mapAs[invalid.expand(bs,3,-1,-1)] = 0

    if batch.xyz_mapBs is None:
      batch.xyz_mapBs = crop_depth_to_crop_xyzmap(batch.depthBs, crop_to_oris, tf_to_crops, batch.Ks, H_ori, W_ori, H, W)
    batch.xyz_mapBs = batch.xyz_mapBs.cuda()
    if self.cfg['normalize_xyz']:
      invalid = batch.xyz_mapBs[:,2:3]<0.001
    batch.xyz_mapBs = batch.xyz_mapBs-batch.poseA[:,:3,3].reshape(bs,3,1,1)
    if self.cfg['normalize_xyz']:
      batch.xyz_mapBs *= 1/mesh_radius.reshape(bs,1,1,1)
      invalid = invalid.expand(bs,3,-1,-1) | (torch.abs(batch.xyz_mapBs)>=2)
      batch.xyz_mapBs[invalid.expand(bs,3,-1,-1)] = 0

    return batch



  def transform_batch(self, batch:BatchPoseData, H_ori, W_ori, bound=1):
    '''Transform the batch before feeding to the network
    !NOTE the H_ori, W_ori could be different at test time from the training data, and needs to be set
    '''
    bs = len(batch.rgbAs)
    batch.rgbAs = batch.rgbAs.cuda().float()/255.0
    batch.rgbBs = batch.rgbBs.cuda().float()/255.0

    batch = self.transform_depth_to_xyzmap(batch, H_ori, W_ori, bound=bound)
    return batch




class TripletH5Dataset(PairH5Dataset):
  def __init__(self, cfg, h5_file, mode, max_num_key=None, cache_data=None):
    super().__init__(cfg, h5_file, mode, max_num_key, cache_data=cache_data)


  def transform_depth_to_xyzmap(self, batch:BatchPoseData, H_ori, W_ori, bound=1):
    bs = len(batch.rgbAs)
    H,W = batch.rgbAs.shape[-2:]
    mesh_radius = batch.mesh_diameters.cuda()/2
    tf_to_crops = batch.tf_to_crops.cuda()
    crop_to_oris = batch.tf_to_crops.inverse().cuda()  #(B,3,3)
    batch.poseA = batch.poseA.cuda()
    batch.Ks = batch.Ks.cuda()

    if batch.xyz_mapAs is None:
      batch.xyz_mapAs = crop_depth_to_crop_xyzmap(batch.depthAs, crop_to_oris, tf_to_crops, batch.Ks, H_ori, W_ori, H, W)
    batch.xyz_mapAs = batch.xyz_mapAs.cuda()
    invalid = batch.xyz_mapAs[:,2:3]<0.1
    batch.xyz_mapAs = (batch.xyz_mapAs-batch.poseA[:,:3,3].reshape(bs,3,1,1))
    if self.cfg['normalize_xyz']:
      batch.xyz_mapAs *= 1/mesh_radius.reshape(bs,1,1,1)
      invalid = invalid.expand(bs,3,-1,-1) | (torch.abs(batch.xyz_mapAs)>=2)
      batch.xyz_mapAs[invalid.expand(bs,3,-1,-1)] = 0

    if batch.xyz_mapBs is None:
      batch.xyz_mapBs = crop_depth_to_crop_xyzmap(batch.depthBs, crop_to_oris, tf_to_crops, batch.Ks, H_ori, W_ori, H, W)
    batch.xyz_mapBs = batch.xyz_mapBs.cuda()
    invalid = batch.xyz_mapBs[:,2:3]<0.1
    batch.xyz_mapBs = (batch.xyz_mapBs-batch.poseA[:,:3,3].reshape(bs,3,1,1))
    if self.cfg['normalize_xyz']:
      batch.xyz_mapBs *= 1/mesh_radius.reshape(bs,1,1,1)
      invalid = invalid.expand(bs,3,-1,-1) | (torch.abs(batch.xyz_mapBs)>=2)
      batch.xyz_mapBs[invalid.expand(bs,3,-1,-1)] = 0

    return batch


  def transform_batch(self, batch:BatchPoseData, H_ori, W_ori, bound=1):
    bs = len(batch.rgbAs)
    batch.rgbAs = batch.rgbAs.cuda().float()/255.0
    batch.rgbBs = batch.rgbBs.cuda().float()/255.0

    batch = self.transform_depth_to_xyzmap(batch, H_ori, W_ori, bound=bound)
    return batch



class ScoreMultiPairH5Dataset(TripletH5Dataset):
  def __init__(self, cfg, h5_file, mode, max_num_key=None, cache_data=None):
    super().__init__(cfg, h5_file, mode, max_num_key, cache_data=cache_data)
    if mode in ['train', 'val']:
      self.cfg['train_num_pair'] = self.n_perturb


class PoseRefinePairH5Dataset(PairH5Dataset):
  def __init__(self, cfg, h5_file, mode='train', max_num_key=None, cache_data=None):
    super().__init__(cfg=cfg, h5_file=h5_file, mode=mode, max_num_key=max_num_key, cache_data=cache_data)

    if mode!='test':
      with h5py.File(h5_file, 'r', libver='latest') as hf:
        group = hf[self.object_keys[0]]
        for key_perturb in group:
          depthA = imageio.imread(group[key_perturb]['depthA'][()])
          depthB = imageio.imread(group[key_perturb]['depthB'][()])
          self.cfg['n_view'] = min(self.cfg['n_view'], depthA.shape[1]//depthB.shape[1])
          logging.info(f'n_view:{self.cfg["n_view"]}')
          self.trans_normalizer = group[key_perturb]['trans_normalizer'][()]
          if isinstance(self.trans_normalizer, np.ndarray):
            self.trans_normalizer = self.trans_normalizer.tolist()
          self.rot_normalizer = group[key_perturb]['rot_normalizer'][()]/180.0*np.pi
          logging.info(f'self.trans_normalizer:{self.trans_normalizer}, self.rot_normalizer:{self.rot_normalizer}')
          break


  def transform_batch(self, batch:BatchPoseData, H_ori, W_ori, bound=1):
    '''Transform the batch before feeding to the network
    !NOTE the H_ori, W_ori could be different at test time from the training data, and needs to be set
    '''
    bs = len(batch.rgbAs)
    batch.rgbAs = batch.rgbAs.cuda().float()/255.0
    batch.rgbBs = batch.rgbBs.cuda().float()/255.0

    batch = self.transform_depth_to_xyzmap(batch, H_ori, W_ori, bound=bound)
    return batch

