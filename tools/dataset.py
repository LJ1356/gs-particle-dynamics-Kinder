from tracemalloc import is_tracing
import torch
import numpy as np
import os
import math
import json
import copy
from PIL import Image
from pytorch3d.structures import Pointclouds
from scipy.spatial.transform import Rotation as R
from sklearn.neighbors import KDTree
from sklearn.preprocessing import normalize
from externals.pointconvformer.util.voxelize import voxelize
from tools.utils import compute_velocity_input, get_one_hot_by_majority_vote_numpy_ver


def find_knns(pc, nsamples):

    tree = KDTree(pc)
    if pc.shape[0] >= nsamples:
        nn_idx = tree.query(pc, k=nsamples, return_distance=False)
    else:
        knn_reduced_size = pc.shape[0]
        nn_idx = tree.query(pc, k=knn_reduced_size, return_distance=False)
        dummy_num = (nsamples - knn_reduced_size)*knn_reduced_size
        dummy_nn_idx = np.random.choice(knn_reduced_size, dummy_num)
        dummy_nn_idx = dummy_nn_idx.reshape(pc.shape[0], -1)
        nn_idx= np.concatenate([nn_idx, dummy_nn_idx], axis=1)
    
    return nn_idx


def find_knns_across_levels(pc, feat, nsamples_min, knn, knn_k_decay_factor, grid_size, scaling):

    # grid downsample
    num_points = pc.shape[0]
    idx = np.arange(num_points)
    idx_sel = torch.tensor(idx, dtype=torch.long)
    pc_sel = torch.tensor(pc, dtype=torch.float)
    feat_sel = torch.tensor(feat, dtype=torch.float)
    pc_all = [pc_sel]
    idx_all = [idx_sel]
    feat_all = [feat_sel]

    for i in range(len(grid_size)):
        assert nsamples_min >= 1, (
            "Too many downsampling layers for the current knn number or decay factor. "
            "Either increase the initial knn number, decrease the decay factor, "
            "or reduce the number of downsampling layers."
        )
        idx = voxelize(pc, scaling * grid_size[i])
        if idx.shape[0] < nsamples_min:
            raise NotImplementedError("Too few points after downsampling.")
        nsamples_min = int(nsamples_min // knn_k_decay_factor)
        pc_sel = pc[idx]
        feat_sel = feat[idx]
        idx_sel = torch.tensor(idx, dtype=torch.long)
        pc_sel = torch.tensor(pc_sel, dtype=torch.float)
        feat_sel = torch.tensor(feat_sel, dtype=torch.float)
        idx_all.append(idx_sel)
        pc_all.append(pc_sel)
        feat_all.append(feat_sel)

    # compute knns between downsampled points clouds
    assert len(grid_size) == (len(pc_all) - 1)
    k_curr = knn
    k_all = []
    nn_idx_self = []
    for i in range(len(pc_all)):
        k_curr = k_curr if i == 0 else int(k_curr // knn_k_decay_factor)
        assert k_curr >= 1
        tree = KDTree(pc_all[i])
        nn_idx = tree.query(pc_all[i], k=k_curr, return_distance=False)
        nn_idx_self.append(torch.tensor(nn_idx, dtype=torch.int32))
        k_all.append(k_curr)
    
    # edge forward
    assert len(k_all) == len(pc_all)
    nn_idx_foward = []
    for i in range(1, len(pc_all)):
        tree = KDTree(pc_all[i - 1])
        nn_idx = tree.query(pc_all[i], k=k_all[i], return_distance=False)
        nn_idx_foward.append(torch.tensor(nn_idx, dtype=torch.int32))
    
    # edge propagate
    nn_idx_propagate = []
    for i in range(len(pc_all) - 1):
        tree = KDTree(pc_all[i + 1])
        nn_idx = tree.query(pc_all[i], k=k_all[i], return_distance=False)
        nn_idx_propagate.append(torch.tensor(nn_idx, dtype=torch.int32))

    nn_idx_all = {}
    nn_idx_all['self'] = nn_idx_self
    nn_idx_all['forward'] = nn_idx_foward
    nn_idx_all['propagate'] = nn_idx_propagate

    return pc_all, feat_all, idx_all, nn_idx_all


class Dataset(torch.utils.data.Dataset):

    def __init__(self, phase, cfg, hard_examples=None, hard_examples_frames=None):

        self.cfg = cfg
        self.phase = phase
        self.start_timestep = 0
        self.all_trials = []
        self.n_rollout = 0
        self.lookahead_frames = 0 if phase == "test" else cfg[phase]['seq_len'] - (cfg['model']['input_frame_num'] + 1)
        self.hard_examples = hard_examples
        self.hard_examples_frames = hard_examples_frames
        assert self.lookahead_frames >= 0, "Lookahead frames should be non-negative"
        
        if phase == "train_hem":
            self.data_dir = os.path.join(cfg['dataset']['data_dir'], cfg['dataset']['scenario'], "train")
        else:
            self.data_dir = os.path.join(cfg['dataset']['data_dir'], cfg['dataset']['scenario'], phase)

        # skip scenes that are not included in the dataset release
        if cfg['dataset']['scenario'] == 'bowling':
            skip_list = [13, 48, 50, 127, 128, 130, 141]
        elif cfg['dataset']['scenario'] == 'cube_stacks':
            skip_list = [36, 72, 74, 75, 78, 81, 82, 83, 105, 106, 124, 130, 131, 133, 161, 162, 163, 174, 191, 193, 208, 234]
        else:
            raise NotImplementedError
   
        trial_names_all = os.listdir(self.data_dir)
        scene_folders = [os.path.join(self.data_dir, trial_name, "gs") for trial_name in trial_names_all if int(trial_name.split('_')[1]) not in skip_list]
        trial_names = []
        for scene_folder in scene_folders:
            frame_path = os.listdir(scene_folder)
            frame_path = [int(d) for d in frame_path if 'done' not in d]
            frame_path.sort()
            start_frame = int(frame_path[0])
            end_frame = int(frame_path[-1])
            # filter out scenes that don't have enough frames to look ahead 
            if end_frame - self.lookahead_frames < start_frame:
                print(scene_folder.split("/")[-2] + " doesn't have enough frames to look ahead, skipping...")
                continue
            trial_names.append(scene_folder.split("/")[-2])

        if phase == "train" or phase == "test":
            n_trials = len(trial_names)
            self.all_trials += [os.path.join(self.data_dir, trial_name, "gs") for trial_name in trial_names]
            self.n_rollout += n_trials
            self.mean_time_step = self.cfg[phase]['frames_per_scene'] if phase == "train" else 1
        elif phase == "train_hem":
            # For HEM, I specify the full path including a frame number rather than randomly sampling a frame
            scene_folders = [os.path.join(self.data_dir, trial_name, "gs") for trial_name in trial_names]
            for scene_folder in scene_folders:
                frame_path = os.listdir(scene_folder)
                frame_path = [int(d) for d in frame_path if 'done' not in d]
                frame_path.sort()
                start_frame = int(frame_path[0])
                end_frame = int(frame_path[-1])
                frame_path = frame_path[:-(self.lookahead_frames)] if self.lookahead_frames > 0 else frame_path
                self.all_trials += [os.path.join(scene_folder, str(fr)) for fr in frame_path]
        else:
            raise NotImplementedError("phase should be one of train, train_hem, test")

    def __len__(self):

        length = len(self.all_trials) if self.phase == "train_hem" else self.n_rollout * self.mean_time_step

        return length

    def __getitem__(self, idx):
        
        if self.phase == "train" or self.phase == "test":
            idx = idx % self.n_rollout
            trial_dir = self.all_trials[idx]
            trial_fullname = trial_dir.split("/")[-2]

            frame_list = os.listdir(trial_dir)
            frame_list = [int(d) for d in frame_list if 'done' not in d]
            frame_list.sort()
            start_timestep = int(frame_list[0])

            if self.phase == "train":
                # hard example sampling
                if len(self.hard_examples) > 0:
                    loss = self.hard_examples[trial_fullname]
                    sum_loss = sum(loss)
                    all_frames = self.hard_examples_frames[trial_fullname]
                    sampling_weights = [p / sum_loss for p in loss]
                    timestep_sel = np.random.choice(all_frames, p=sampling_weights)
                # random sampling
                else:
                    end_timestep_plus_one = int(frame_list[-1]) + 1 - self.lookahead_frames
                    assert end_timestep_plus_one > start_timestep, trial_fullname + " doesn't have enough frames to look ahead"
                    timestep_sel = np.random.randint(start_timestep, end_timestep_plus_one)
            else:
                timestep_sel = start_timestep
                self.lookahead_frames = int(frame_list[-1]) - timestep_sel # just for logging GT poses for visualization later
            
            data_path = os.path.join(trial_dir, str(timestep_sel))
        else:
            data_path = self.all_trials[idx]
            trial_fullname = data_path.split("/")[-3]
            timestep_sel = int(data_path.split("/")[-1])
            trial_dir = data_path.rsplit('/', 1)[0]

        # the "train" folder below is from GS optimiztaion, which produces input guassians 
        params = dict(np.load(os.path.join(data_path, "params_coarse.npz")))
        id_data = dict(np.load(os.path.join(data_path, "gs_soft_ids_coarse.npz"), allow_pickle=True))
        pose_data = self._get_pose_data(trial_fullname, timestep_sel)
        gs_soft_ids = id_data['gaussian_ids_to_object_ids']
        obj_id_to_color = id_data['obj_id_to_color']        
        gs_ids = np.expand_dims(np.sum(np.sum(gs_soft_ids, axis=-1), axis=-1), axis=-1)    

        with open(os.path.join(self.data_dir, trial_fullname, 'camera_meta.json'), 'r') as f:    
            md = json.load(f)  # metadata
        
        idx_sel = self._get_foreground_idx(gs_ids)
        params, gs_soft_ids = self._resample_arrays(params, gs_soft_ids, idx_sel)
        # update positions and rotations of params using pose_data
        params = self._update_params_with_pose_data(params, gs_soft_ids, pose_data, obj_id_to_color)
        # params = self._update_params_with_pose_data_with_explicit_file_save(params, gs_soft_ids, pose_data, obj_id_to_color, trial_fullname)
        
        # modify point cloud scale and later get it back to the original scale (not sure if this is helping training)
        params['means3D'] = self.cfg['pointcloud']['scaling'] * params['means3D']
        gs_ids = np.expand_dims(np.sum(np.sum(gs_soft_ids, axis=-1), axis=-1), axis=-1)    
        assert np.sum(gs_ids == 0) == 0, "There should be no background gaussians when loading foreground only data"

        seq_len = 4
        # rotation
        if self.phase == 'train':
            azimuth = np.random.uniform(0, 360)
            # seq length is 4 (default input:3 and default output:1) + lookahead_frames
            params, rot, inv_rot = self._rotate_scene(params, azimuth, seq_len + self.lookahead_frames)               
        else:            
            rot = np.eye(3)
            inv_rot = np.eye(3)
            # debugging
            # azimuth = np.random.uniform(0,360)
            # params, rot, inv_rot = self.rotate_scene(params, azimuth, seq_len + self.lookahead_frames)  

        last_frame_ind = seq_len - 1
        gt_frame = timestep_sel + last_frame_ind
        gt = self._get_dataset(trial_fullname, gt_frame, md, inv_rot)
           
        # load more future frames
        gt_lookahead = []
        for i in range(self.lookahead_frames): # metadata
            gt_frame = timestep_sel + (i + 1) + last_frame_ind
            gt_lookahead.append(self._get_dataset(trial_fullname, gt_frame, md, inv_rot))

        example = {}
        ct = 0
        reference_frame_idx = 0 # scale, color and opicity from the first frame
        for t in range(1, last_frame_ind): # excluding the last frame here as it's GT
            pos = params['means3D'][t]
            pos_vel = params['means3D'][t] - params['means3D'][t - 1]            
            rot_vel = normalize(params['unnorm_rotations'][t]) - normalize(params['unnorm_rotations'][t - 1])
            log_scale = params['log_scales'][reference_frame_idx]
            rgb = params['shs'][reference_frame_idx] # N x 16 x 3
            rgb = rgb.reshape(-1, 16*3)
            gs_soft_ids_vec = gs_soft_ids.reshape(-1, self.cfg['dataset']['camera_num'] * self.cfg['dataset']['max_object_num'])
            assert pos.shape[0] == gs_soft_ids_vec.shape[0]
            assert rgb.shape[0] == pos.shape[0]
            unnorm_rot = params['unnorm_rotations'][t]
            seg = np.zeros((pos.shape[0], 3)) # dummy
            logit = params['logit_opacities'][reference_frame_idx]
            vel = pos_vel
            data = [pos, vel]            
            data_aux = [np.concatenate((unnorm_rot, logit, log_scale, seg, rgb, gs_soft_ids_vec), axis=1)]
            obj_seq_info, nn_idx = self._prepare_model_input(data, data_aux)

            assert ct == t - 1 
            if ct == 0:
                example['seq_name'] = trial_fullname
                example['start_frame'] = timestep_sel
                example['inv_rot'] = torch.tensor(inv_rot)
                example['seq_info'] = {}
                example['gt'] = gt
                example['gt_lookahead'] = gt_lookahead
            elif ct == 1:
                example['nn_idx'] = nn_idx # neighbor indices from the most recent input frame (t == 2)
            else:
                raise NotImplementedError
            example['seq_info'][ct] = obj_seq_info
            ct += 1
        
        # add position gt labels
        gt_position = []
        gt_rotation = []
        for t in range(last_frame_ind, last_frame_ind + 1 + self.lookahead_frames):
            gt_position.append(torch.tensor(params['means3D'][t]))
            gt_rotation.append(torch.tensor(params['unnorm_rotations'][t]))
        assert len(params['means3D']) == last_frame_ind + 1 + self.lookahead_frames
        example['gt']['position'] = torch.stack(gt_position)
        example['gt']['rotation'] = torch.stack(gt_rotation)

        # For including the input frames for visualization purpose
        if self.phase == 'test':
            output_dir = os.path.join(self.cfg['output_dir'], self.cfg['exp_name_epoch'])
            output_scene_dir = os.path.join(output_dir, trial_fullname)
            if not os.path.isdir(output_dir):
                os.mkdir(output_dir)
            os.mkdir(output_scene_dir)
            params['means3D'] = (1.0 / self.cfg['pointcloud']['scaling']) * params['means3D']
            np.savez(os.path.join(output_scene_dir, 'params_gt.npz'), **params)

        assert self.cfg['model']['input_frame_num'] == 2, "Only support 2-frame velocity input for now"
        assert self.cfg['model']['input_frame_num'] < self.cfg['train']['seq_len']
        assert self.phase == 'test' or ct == self.cfg['train']['seq_len'] - 1 - self.lookahead_frames

        return example

    def _get_foreground_idx(self, gs_ids):

        idx_sel = np.squeeze(gs_ids != 0.0, axis=-1)

        return idx_sel
    
    def _update_params_with_pose_data(self, params, gs_soft_ids, pose_data, obj_id_to_color):

        obj_id_to_color = obj_id_to_color.item()
        color_map = {}
        for key in obj_id_to_color.keys():
            for id_color in obj_id_to_color[key]:
                obj_id, color = id_color
                # zero pad the color values
                color_key = f"{str(color[0]).zfill(3)}_{str(color[1]).zfill(3)}_{str(color[2]).zfill(3)}"
                if color_key in color_map:
                    assert obj_id == color_map[color_key], "Color collision between different object IDs"
                else:
                    color_map[color_key] = obj_id
                
                if len(color_map) == self.cfg['dataset']['max_object_num']:
                    break

        # initialize new params. The first frame is used as the reference frame so just copy it over
        # +3 because original the clip size is 4 frames, and the means3D length is only 1 because I only run GS on the first frame during preprocessing now
        num_frames = len(params['means3D']) + 3 
        new_means3D = np.zeros((num_frames + self.lookahead_frames, params['means3D'].shape[1], 3), dtype=np.float32)
        new_rotation = np.zeros((num_frames + self.lookahead_frames, params['means3D'].shape[1], 4), dtype=np.float32)
        new_means3D[0, :, :] = params['means3D'][0, :, :]
        new_rotation[0, :, :] = params['unnorm_rotations'][0, :, :]

        obj_ids = get_one_hot_by_majority_vote_numpy_ver(gs_soft_ids)
        sz_check = 0        
        for obj_id in pose_data.keys():
            red, green, blue = obj_id.split('_')
            red = int(float(red) * 255)
            green = int(float(green) * 255)
            blue = int(float(blue) * 255)
            color_key = f"{str(red).zfill(3)}_{str(green).zfill(3)}_{str(blue).zfill(3)}"
            curr_obj_id = color_map[color_key]
            curr_obj_idx = curr_obj_id - 1
            idx_sel = np.argmax(obj_ids, axis=1) == curr_obj_idx

            if np.sum(idx_sel) == 0: continue

            sz_check += np.sum(idx_sel)
            # use the first frame as the reference
            reference_frame_idx = 0
            means3D_src = params['means3D'][reference_frame_idx][idx_sel, :] # N x 3
            rotations_src = normalize(params['unnorm_rotations'][reference_frame_idx][idx_sel, :])
            # convert quaternion to rotation matrix
            rotations_src = R.from_quat(np.concatenate((rotations_src[:, 1:4], rotations_src[:, [0]]), axis=1)).as_matrix() # N x 3 x 3
            assert len(pose_data[obj_id]) == num_frames + self.lookahead_frames - 1
            for t in range(num_frames + self.lookahead_frames - 1):
                # 4x4 transformation matrix
                rigid_transformation = pose_data[obj_id][t]
                means3D_tgt = np.matmul(rigid_transformation[:3, :3], means3D_src.T).T + rigid_transformation[None, :3, 3]
                rotation_tgt = np.matmul(rigid_transformation[None, :3, :3], rotations_src)
                # convert rotation matrix back to quaternion
                rotation_tgt = R.from_matrix(rotation_tgt).as_quat()
                new_means3D[t + 1, idx_sel, :] = means3D_tgt
                new_rotation[t + 1, idx_sel, :] = np.concatenate((rotation_tgt[:, [3]], rotation_tgt[:, :3]), axis=1)
        
        assert sz_check == params['means3D'][0].shape[0], "Some points are not assigned to any object"

        params['means3D'] = new_means3D
        params['unnorm_rotations'] = new_rotation

        return params

    def _update_params_with_pose_data_with_explicit_file_save(self, params, gs_soft_ids, pose_data, obj_id_to_color, trial_name):

        obj_id_to_color = obj_id_to_color.item()
        color_map = {}
        for key in obj_id_to_color.keys():
            for id_color in obj_id_to_color[key]:
                obj_id, color = id_color
                # zero pad the color values
                color_key = f"{str(color[0]).zfill(3)}_{str(color[1]).zfill(3)}_{str(color[2]).zfill(3)}"
                if color_key in color_map:
                    assert obj_id == color_map[color_key], "Color collision between different object IDs"
                else:
                    color_map[color_key] = obj_id
                
                if len(color_map) == self.cfg['dataset']['max_object_num']:
                    break

        # initialize new params. The first frame is used as the reference frame so just copy it over
        # +3 because original the clip size is 4 frames, and the means3D length is only 1 because I only run GS on the first frame during preprocessing now
        num_frames = len(params['means3D']) + 3 
        new_means3D = np.zeros((num_frames + self.lookahead_frames, params['means3D'].shape[1], 3), dtype=np.float32)
        new_rotation = np.zeros((num_frames + self.lookahead_frames, params['means3D'].shape[1], 4), dtype=np.float32)
        new_means3D[0, :, :] = params['means3D'][0, :, :]
        new_rotation[0, :, :] = params['unnorm_rotations'][0, :, :]

        obj_ids = get_one_hot_by_majority_vote_numpy_ver(gs_soft_ids)
        sz_check = 0        
        for obj_id in pose_data.keys():
            red, green, blue = obj_id.split('_')
            red = int(float(red) * 255)
            green = int(float(green) * 255)
            blue = int(float(blue) * 255)
            color_key = f"{str(red).zfill(3)}_{str(green).zfill(3)}_{str(blue).zfill(3)}"
            curr_obj_id = color_map[color_key]
            curr_obj_idx = curr_obj_id - 1
            idx_sel = np.argmax(obj_ids, axis=1) == curr_obj_idx

            if np.sum(idx_sel) == 0: continue

            sz_check += np.sum(idx_sel)
            # use the first frame as the reference
            reference_frame_idx = 0
            means3D_src = params['means3D'][reference_frame_idx][idx_sel, :] # N x 3
            rotations_src = normalize(params['unnorm_rotations'][reference_frame_idx][idx_sel, :])
            # convert quaternion to rotation matrix
            rotations_src = R.from_quat(np.concatenate((rotations_src[:, 1:4], rotations_src[:, [0]]), axis=1)).as_matrix() # N x 3 x 3
            assert len(pose_data[obj_id]) == num_frames + self.lookahead_frames - 1
            for t in range(num_frames + self.lookahead_frames - 1):
                # 4x4 transformation matrix
                rigid_transformation = pose_data[obj_id][t]
                means3D_tgt = np.matmul(rigid_transformation[:3, :3], means3D_src.T).T + rigid_transformation[None, :3, 3]
                rotation_tgt = np.matmul(rigid_transformation[None, :3, :3], rotations_src)
                # convert rotation matrix back to quaternion
                rotation_tgt = R.from_matrix(rotation_tgt).as_quat()
                new_means3D[t + 1, idx_sel, :] = means3D_tgt
                new_rotation[t + 1, idx_sel, :] = np.concatenate((rotation_tgt[:, [3]], rotation_tgt[:, :3]), axis=1)
        
        assert sz_check == params['means3D'][0].shape[0], "Some points are not assigned to any object"

        params['means3D'] = new_means3D
        params['unnorm_rotations'] = new_rotation

        # store new_means3D
        output_dir = os.path.join(self.cfg['output_dir'], self.cfg['exp_name_epoch'], "means3D_" + trial_name)
        os.makedirs(output_dir)
        np.savez_compressed(os.path.join(output_dir, 'means3D.npz'), **params, obj_ids=obj_ids)

        return params

    def _get_pose_data(self, scene_num, frame_num):

        scene_path = os.path.join(self.data_dir, scene_num, 'obj_poses')
        cameras = os.listdir(scene_path)
        pose_data = {}
        pose_data_error = {}
        for cam in cameras:
            tmp = dict(np.load(os.path.join(scene_path, cam, 'obj_poses.npz'), allow_pickle=True))
            obj_poses = tmp['obj_poses'].item()
            obj_poses_error = tmp['obj_poses_error'].item()

            # initialize pose_data and pose_data_error
            if len(pose_data) == 0:
                for obj_id in obj_poses.keys():
                    pose_data[obj_id] = {}
                    pose_data_error[obj_id] = {}
                    for i in range(frame_num, frame_num + 3 + self.lookahead_frames + 1):
                        pose_data[obj_id][i] = []
                        pose_data_error[obj_id][i] = []
                    
            # extract 4 frames (original clip size) + lookahead frames starting from frame_num
            for obj_id in obj_poses.keys():
                for i in range(frame_num, frame_num + 3 + self.lookahead_frames + 1):
                    try:
                        pose_data[obj_id][i].append(obj_poses[obj_id][i])
                        pose_data_error[obj_id][i].append(obj_poses_error[obj_id][i])
                    # some object may not be seen from certain camera views
                    except KeyError:         
                        # print(f"Object {obj_id} not found in camera {cam} for frame {i} in scene {scene_num}")
                        pass            

        # NOTE: investigate to see some potential impact created from this (some pose data for the first frame was missing in obj_poses.npz)
        #       This was happening for some cube scenes, probably becase we used backward video object segmentation for some scenes.
        #       In the case of backward tracking, some objects may not be visible in the first frame in all views. This will lead to some training noise.
        # assert pose_data.keys() == obj_poses.keys()
        # if pose_data.keys() != obj_poses.keys():
        #     missing_objs = set(obj_poses.keys()) - set(pose_data.keys())
        #     raise ValueError(f"Some objects are missing pose data across all cameras in scene {scene_num}: {missing_objs}")

        # for each frame, pick the pose with the minimum error across different cameras
        for obj_id in pose_data.keys():
            for i in range(frame_num, frame_num + 3 + self.lookahead_frames + 1):
                min_error_idx = np.argmin(pose_data_error[obj_id][i])
                pose_data[obj_id][i] = pose_data[obj_id][i][min_error_idx]
        
        # align to the first frame
        pose_final = {}
        for obj_id in pose_data.keys():
            pose_first = np.array(pose_data[obj_id][frame_num])
            pose_first_inv = np.linalg.inv(pose_first)
            pose_final[obj_id] = []
            for i in range(frame_num + 1, frame_num + 3 + self.lookahead_frames + 1):
                pose_curr = np.array(pose_data[obj_id][i])
                pose_aligned = np.matmul(pose_curr, pose_first_inv)
                pose_final[obj_id].append(pose_aligned)
            assert len(pose_final[obj_id]) == 4 + self.lookahead_frames - 1

        return pose_final

    def _resample_arrays(self, params, gs_soft_id, idx_sel):

        for key in params.keys():
            if key in ['means3D', 'shs', 'unnorm_rotations', 'logit_opacities', 'log_scales']:                
                params[key] = params[key][:, idx_sel]
            elif key in ['rgb_colors']: # shs is used instead
                continue
            else:
                raise AssertionError("Unknown dictionary key")
        gs_soft_id = gs_soft_id[idx_sel]

        return params, gs_soft_id

    def _rotate_scene(self, params, azimuth, seq_len):
        
        neg_azimuth = - azimuth
        if self.cfg['dataset']['gravity_axis'] == 1:
            rot = np.array([[np.cos(math.radians(azimuth)), 0 , -np.sin(math.radians(azimuth))],
                    [0 , 1 ,0 ],
                    [np.sin(math.radians(azimuth)), 0, np.cos(math.radians(azimuth))]])
            inv_rot = np.array([[np.cos(math.radians(neg_azimuth)), 0 , -np.sin(math.radians(neg_azimuth))],
                    [0 , 1 ,0 ],
                    [np.sin(math.radians(neg_azimuth)), 0, np.cos(math.radians(neg_azimuth))]])
        elif self.cfg['dataset']['gravity_axis'] == 2:
            rot = np.array([[np.cos(math.radians(azimuth)), np.sin(math.radians(azimuth)), 0],
                    [-np.sin(math.radians(azimuth)), np.cos(math.radians(azimuth)), 0],
                    [0, 0, 1]])
            inv_rot = np.array([[np.cos(math.radians(neg_azimuth)), np.sin(math.radians(neg_azimuth)), 0],
                    [-np.sin(math.radians(neg_azimuth)), np.cos(math.radians(neg_azimuth)), 0],
                    [0, 0, 1]])
        else:
            raise NotImplementedError

        assert seq_len == len(params['means3D'])
        for frame in range(seq_len):
            params['means3D'][frame] = np.dot(rot, params['means3D'][frame].T).T
            curr_rot = normalize(params['unnorm_rotations'][frame])
            # R.from_quat expects the scalar-last format
            curr_rot_tmp = R.from_quat(np.concatenate((curr_rot[:, 1:4], curr_rot[:, [0]]), axis=1)).as_matrix()
            aug_rot_tmp = np.matmul(rot[None, :, :], curr_rot_tmp)
            aug_rot_tmp = R.from_matrix(aug_rot_tmp).as_quat()
            aug_rot_tmp = np.concatenate((aug_rot_tmp[:, [3]], aug_rot_tmp[:, :3]), axis=1)
            params['unnorm_rotations'][frame] = aug_rot_tmp

        return params, rot, inv_rot

    def _get_dataset(self, scene_name, t_sel, md, rot):
        view_sel = np.random.choice(
            np.arange(self.cfg['dataset']['camera_num']),
            self.cfg[self.phase]['view_num'],
            replace=False
        )
        w, h = md['w'], md['h']

        k_all = []
        w2c_all = []
        im_all = []
        seg_col_all = []
        depth_all = []
        id_all = []
        for sel_idx in range(view_sel.shape[0]):
            rot_aug = np.eye(4)
            rot_aug[:3, :3] = rot
            w2c_tmp = np.stack(md['w2c'][sel_idx])
            w2c_tmp = np.matmul(w2c_tmp, rot_aug)
            w2c = w2c_tmp.tolist()
            k = md['k'][sel_idx]
            cam_id = md['cam_id'][sel_idx]
            im_path = os.path.join(self.data_dir, scene_name, 'rgb', cam_id, f"{t_sel:05d}.png")
            seg_path = os.path.join(self.data_dir, scene_name, 'seg', cam_id, f"{t_sel:05d}.png")
            im = np.array(copy.deepcopy(Image.open(im_path)))
            seg = np.array(copy.deepcopy(Image.open(seg_path))).astype(np.float32)
            seg = seg * 255
            assert np.max(seg) != 1
            im = torch.tensor(im).float().permute(2, 0, 1) / 255
            seg = torch.tensor(seg).float()
            seg = seg / 255
            z_depth = im # currently not using depth, so putting dummy here for now
            # I don't use background gaussians so I don't need the last column
            # seg_col = torch.stack((seg, torch.zeros_like(seg), 1 - seg))   
            seg_col = torch.stack((seg, torch.zeros_like(seg), torch.zeros_like(seg)))
            k_all.append(torch.tensor(k))
            w2c_all.append(torch.tensor(w2c))
            im_all.append(im)
            seg_col_all.append(seg_col)
            depth_all.append(z_depth)
            id_all.append(sel_idx)        
        k_all = torch.stack(k_all)
        w2c_all = torch.stack(w2c_all)
        im_all = torch.stack(im_all)
        seg_col_all = torch.stack(seg_col_all)
        depth_all = torch.stack(depth_all)

        return {
            'cam': [w, h, k_all, w2c_all],
            'im': im_all,
            'seg': seg_col_all,
            'depth': depth_all,
            'id': id_all
        }
    
    def _prepare_model_input(self, data_curr, data_curr_aux):

        positions_curr, velocities_curr = data_curr
        assert len(data_curr_aux) == 1
        feats_curr = data_curr_aux[0]
        assert positions_curr.shape[0] == feats_curr.shape[0]
        assert positions_curr.shape[0] == velocities_curr.shape[0]

        levels = self.cfg['pointcloud']['downsampling_layer_num'] + 1
        idx_curr_list = {}
        pos_curr_list = {}
        feats_curr_list = {}
        nn_idx_self_list = {}
        nn_idx_forward_list = {}
        nn_idx_propagate_list = {}
        for i in range(levels - 1):
            idx_curr_list[i] = []
            pos_curr_list[i] = []
            feats_curr_list[i] = []
            nn_idx_self_list[i] = []
            nn_idx_forward_list[i] = []
            nn_idx_propagate_list[i] = []
        idx_curr_list[levels - 1] = []
        pos_curr_list[levels - 1] = []
        feats_curr_list[levels - 1] = []
        nn_idx_self_list[levels - 1] = []
        instance_idx = [0, positions_curr.shape[0]]

        for i in range(len(instance_idx) - 1):
            st, ed = instance_idx[i], instance_idx[i + 1]
            pc_all, feat_all, grid_idx_all, nn_idx_all = find_knns_across_levels(
                positions_curr[st:ed],
                feats_curr[st:ed],
                self.cfg['pointcloud']['minimum_point_num'],
                self.cfg['pointcloud']['knn'],
                self.cfg['pointcloud']['knn_k_decay_factor'],
                self.cfg['pointcloud']['grid_size'],
                self.cfg['pointcloud']['scaling']
            )
            grid_idx_offset = st
            assert len(grid_idx_all) == levels
            assert len(pc_all) == levels
            assert len(feat_all) == levels
            assert len(nn_idx_all['self']) == levels
            assert len(nn_idx_all['forward']) == levels - 1
            assert len(nn_idx_all['propagate']) == levels - 1
            idx_curr_list[levels - 1].append(grid_idx_all[levels - 1] + grid_idx_offset)
            pos_curr_list[levels - 1].append(pc_all[levels - 1])
            feats_curr_list[levels - 1].append(feat_all[levels - 1])
            nn_idx_self_list[levels - 1].append(nn_idx_all['self'][levels - 1])
            for j in range(levels - 1):
                idx_curr_list[j].append(grid_idx_all[j] + grid_idx_offset)
                pos_curr_list[j].append(pc_all[j])
                feats_curr_list[j].append(feat_all[j])
                nn_idx_self_list[j].append(nn_idx_all['self'][j])
                nn_idx_forward_list[j].append(nn_idx_all['forward'][j])
                nn_idx_propagate_list[j].append(nn_idx_all['propagate'][j])
        assert grid_idx_offset == 0, "I don't loop over per-object pointclouds anymore so Grid index offset should be 0"
        assert ed == positions_curr.shape[0]

        grid_idx_curr = []
        pos_curr = []
        nn_idx_self = []
        nn_idx_forward = []
        nn_idx_propagate = []
        for i in range(levels - 1):
            pos_curr.append(Pointclouds(points=pos_curr_list[i], features=feats_curr_list[i]))
            nn_idx_self.append(Pointclouds(points=pos_curr_list[i], features=nn_idx_self_list[i]))
            nn_idx_forward.append(Pointclouds(points=pos_curr_list[i + 1], features=nn_idx_forward_list[i]))
            nn_idx_propagate.append(Pointclouds(points=pos_curr_list[i], features=nn_idx_propagate_list[i]))
            grid_idx_curr.append(torch.cat(idx_curr_list[i], dim=0))
        pos_curr.append(Pointclouds(points=pos_curr_list[levels - 1], features=feats_curr_list[levels - 1]))
        nn_idx_self.append(Pointclouds(points=pos_curr_list[levels - 1], features=nn_idx_self_list[levels - 1]))
        grid_idx_curr.append(torch.cat(idx_curr_list[levels - 1], dim=0))

        obj_seq_info = {}
        pc = pos_curr[0] 
        pos_packed = pc.points_packed()
        vel_packed = torch.tensor(velocities_curr)
        assert vel_packed.shape[1] == 3, "velocities should only contain position velocities"
        obj_seq_info['pc'] = pos_curr 
        obj_seq_info['grid_idx'] = grid_idx_curr
        # currently some arguments are not used because I only use position velocity
        obj_seq_info['velocity'] = compute_velocity_input(pos_packed, vel_packed, None, None, self.cfg)

        nn_idx = {}
        nn_idx['self'] = nn_idx_self
        nn_idx['forward'] = nn_idx_forward
        nn_idx['propagate'] = nn_idx_propagate

        return obj_seq_info, nn_idx

    def update_hard_examples(self, hard_examples, hard_examples_frames):

        self.hard_examples = hard_examples
        self.hard_examples_frames = hard_examples_frames


def collate_fn(data):

    batch = {}    
    # get scene info here
    all_seq_name = []
    all_start_frame = []    
    all_inv_rot = []
    all_gt = []
    all_gt_lookahead = []
    for i in range(len(data)):
        all_seq_name.append(data[i]['seq_name'])
        all_start_frame.append(data[i]['start_frame'])
        all_inv_rot.append(data[i]['inv_rot'])
        all_gt.append(data[i]['gt'])
        all_gt_lookahead.append(data[i]['gt_lookahead'])
    batch['seq_name'] = all_seq_name
    batch['start_frame'] = all_start_frame
    batch['inv_rot'] = all_inv_rot
    batch['gt'] = all_gt
    batch['gt_lookahead'] = all_gt_lookahead

    # per-step action (and TCP for RPE), stacked in the same cloud order as the point clouds
    if 'action' in data[0]:
        batch['action'] = torch.stack([d['action'] for d in data], dim=0)  # (B, num_gt, 10)
    if 'tcp_xyz' in data[0]:
        batch['tcp_xyz'] = torch.stack([d['tcp_xyz'] for d in data], dim=0)  # (B, num_gt, 3)

    # get mullti-frame info here
    batch['pc'] = {}
    batch['grid_idx'] = {}
    batch['velocity'] = {}
    for frame_num in range(len(data[0]['seq_info'].keys())):
        offset_grid_idx = 0
        all_grid_idx_tmp = {}
        all_points_tmp = {}
        all_feats_tmp = {}
        all_velocities = []        
        for i in range(len(data)):
            pointclouds = data[i]['seq_info'][frame_num]['pc']            
            grid_idx = data[i]['seq_info'][frame_num]['grid_idx']
            velocities = data[i]['seq_info'][frame_num]['velocity']
            assert len(pointclouds) == len(grid_idx)
            if i == 0:
                for j in range(len(pointclouds)):
                    all_grid_idx_tmp[j] = []
                    all_points_tmp[j] = []
                    all_feats_tmp[j] = []
            for j in range(len(pointclouds)):
                all_grid_idx_tmp[j].append(grid_idx[j] + offset_grid_idx)
                all_points_tmp[j] += pointclouds[j].points_list()
                all_feats_tmp[j] += pointclouds[j].features_list()
            offset_grid_idx += pointclouds[0].points_packed().shape[0]
            all_velocities.append(velocities)        
        
        pc = []
        grid_idx = []
        for j in sorted(all_points_tmp.keys()):
            pc.append(Pointclouds(points=all_points_tmp[j], features=all_feats_tmp[j]))
            grid_idx.append(torch.cat(all_grid_idx_tmp[j], dim=0))
            _ = pc[-1].points_packed()

        batch['pc'][frame_num] = pc
        batch['grid_idx'][frame_num] = grid_idx
        batch['velocity'][frame_num] = torch.cat(all_velocities, dim=0)

    # set up variables
    batch_idx = {}
    obj_num = {}
    offset_self = {}
    all_nn_self = {}
    all_points_self = {}
    pointclouds_self_tmp = data[0]['nn_idx']['self']
    for j in range(len(pointclouds_self_tmp)):
        batch_idx[j] = torch.zeros(len(data), dtype=torch.int)  
        obj_num[j] = torch.zeros(len(data))
        offset_self[j] = 0
        all_nn_self[j] = []
        all_points_self[j] = []

    # merge knn indices here (at the same level)
    nn_self = {}
    for i in range(len(data)):
        pointclouds_self = data[i]['nn_idx']['self']  
        for j in range(len(pointclouds_self)):
            nn_self[j] = []
            for ii, features in enumerate(pointclouds_self[j].features_list()):
                nn_self[j].append(features + offset_self[j])
                offset_self[j] += features.shape[0]
            all_nn_self[j] += nn_self[j]
            all_points_self[j] += pointclouds_self[j].points_list() # dummy points
            batch_idx[j][i] = torch.sum(pointclouds_self[j].num_points_per_cloud())       
            obj_num[j][i] = len(all_points_self[j]) 

    # set up variables
    offset_forward = {}
    offset_propagate = {}
    all_nn_forward = {}
    all_nn_propagate = {}
    all_points_forward = {}
    all_points_propagate = {}
    levels = len(pointclouds_self_tmp)
    for j in range(len(pointclouds_self_tmp)):
        offset_forward[j] = 0
        offset_propagate[j] = 0
        all_nn_forward[j] = []
        all_nn_propagate[j] = []
        all_points_forward[j] = []
        all_points_propagate[j] = []

    # merge knn indices here (across levels)
    nn_forward = {}
    nn_propagate = {}
    for i in range(len(data)):
        pointclouds_self = data[i]['nn_idx']['self']
        pointclouds_forward = data[i]['nn_idx']['forward']
        pointclouds_propagate = data[i]['nn_idx']['propagate']
        assert len(pointclouds_forward) == len(pointclouds_propagate)
        for j in range(len(pointclouds_forward)):
            pc_self_dense = pointclouds_self[j].features_list()
            pc_self_sparse = pointclouds_self[j+1].features_list()
            pc_forward_list = pointclouds_forward[j].features_list()
            pc_propagate_list = pointclouds_propagate[j].features_list()
            nn_forward[j] = []
            nn_propagate[j] = []
            assert len(pc_forward_list) == len(pc_propagate_list)
            for k in range(len(pc_forward_list)):
                dense_pc = pc_self_dense[k]
                sparse_pc = pc_self_sparse[k]
                features_forward = pc_forward_list[k]
                features_propagate = pc_propagate_list[k]
                nn_forward[j].append(features_forward + offset_forward[j])
                nn_propagate[j].append(features_propagate + offset_propagate[j])
                offset_forward[j] += dense_pc.shape[0]
                offset_propagate[j] += sparse_pc.shape[0]
            all_nn_forward[j] += nn_forward[j]
            all_nn_propagate[j] += nn_propagate[j]
            all_points_forward[j] += pointclouds_forward[j].points_list() # dummy points            
            all_points_propagate[j] += pointclouds_propagate[j].points_list() # dummy points
            assert len(all_points_forward[j]) ==  len(all_points_propagate[j])

    pc_self =  {}
    pc_forward = {}
    pc_propagagte = {}    
    for i in range(levels):
        pc_self[i] = Pointclouds(points=all_points_self[i], features=all_nn_self[i])
        if i != levels - 1:
            pc_forward[i] = Pointclouds(points=all_points_forward[i], features=all_nn_forward[i])
            pc_propagagte[i] = Pointclouds(points=all_points_propagate[i], features=all_nn_propagate[i])

    batch['obj_num'] = obj_num
    batch['batch_idx'] = batch_idx
    batch['nn_idx_self'] = pc_self
    batch['nn_idx_forward'] = pc_forward
    batch['nn_idx_propagate'] = pc_propagagte

    return batch


def parse_image_info(img_dict, device):

    tmp = {}
    for key2 in img_dict.keys():
        if key2 in ['im', 'seg', 'depth', 'position', 'rotation']:
            tmp[key2] = img_dict[key2].to(device)
        elif key2 == 'id':
            tmp[key2] = img_dict[key2]
        elif key2 == 'cam':
            tmp['w'] = img_dict[key2][0]
            tmp['h'] = img_dict[key2][1]
            tmp['k'] = img_dict[key2][2].to(device)
            tmp['w2c'] = img_dict[key2][3].to(device)
        else:
            raise NotImplementedError
    
    return tmp


def move_to_gpu(tensor_dict, device):

    tensors_out = {}
    for key1 in tensor_dict.keys():
        if key1 in ['seq_name', 'start_frame']:
            tensors_out[key1] = tensor_dict[key1]
        elif key1 in ('action', 'tcp_xyz'):
            tensors_out[key1] = tensor_dict[key1].to(device)
        elif key1 == 'inv_rot':
            tensors_out[key1] = []
            for tensor in tensor_dict[key1]:
                tensors_out[key1].append(tensor.to(device))
        elif key1 == 'gt':
            tensors_out[key1] = []
            for dict_tmp in tensor_dict[key1]:
                tensors_out[key1].append(parse_image_info(dict_tmp, device))
        elif key1 == 'gt_lookahead':
            tensors_out[key1] = []
            for dict_list in tensor_dict[key1]:
                dict_out_list = []
                for dict_tmp in dict_list:
                    dict_out_list.append(parse_image_info(dict_tmp, device))
                tensors_out[key1].append(dict_out_list)
        else:
            tensors_out[key1] = {}
            for key2 in sorted(tensor_dict[key1].keys()):
                if key1 in ['pc', 'grid_idx']:
                    tensors_out[key1][key2] = []
                    for tensor in tensor_dict[key1][key2]:
                        tensors_out[key1][key2].append(tensor.to(device))
                else:                
                    tensors_out[key1][key2] = tensor_dict[key1][key2].to(device)
    
    return tensors_out
