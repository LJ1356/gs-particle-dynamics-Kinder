import os
import numpy as np
import torch
import torch.utils.checkpoint
import torch.nn.functional as F
from pytorch3d.transforms import rotation_6d_to_matrix as r6d2m
from pytorch3d.transforms import quaternion_to_matrix as q2m
from pytorch3d.transforms import matrix_to_quaternion as m2q
from pytorch3d.structures import Pointclouds
from scipy.spatial.transform import Rotation as R
from gsplat.rendering import rasterization
from tools.utils import compute_velocity_input, calc_rigid_transform_torch_ver, get_one_hot_by_majority_vote
from externals.physion_particles.utils import calc_rigid_transform


class Model_Runner(object):

    def __init__(self, models, criterion, optimizer, cfg):

        self.pointconv_internaction_network = models[0]
        self.predictor = models[1]
        assert(len(models) == 2)
        self.criterion = criterion
        self.optimizer = optimizer
        self.cfg = cfg
        self.all_params = []
        for model in models:
            self.all_params += list(model.parameters())

    def put_models_in_eval_mode(self):
        self.pointconv_internaction_network.eval()
        self.predictor.eval()

    def put_models_in_train_mode(self):
        self.pointconv_internaction_network.train()
        self.predictor.train()

    def store_checkpoint(self, cur_epoch):

        exp_folder = os.path.join(self.cfg['exp_dir'], self.cfg['exp_name'])
        out_folder = os.path.join(exp_folder, 'epoch_%02d' % cur_epoch)
        out_file_path = os.path.join(out_folder, 'model.pt')
        os.makedirs(out_folder, exist_ok=True)

        torch.save(
            {'epoch': cur_epoch,
             'pointconv_internaction_network': self.pointconv_internaction_network.state_dict(),                
             'predictor': self.predictor.state_dict(),
             'optim_state_dict': self.optimizer.state_dict()},
            out_file_path
        )

    def optimize(self, loss):
                        
        loss.backward()   
        self.optimizer.step()
        self.optimizer.zero_grad()        

    def compute_loss(self, batch_all, phase):

        pred_log = {}
        pred = None
        loss = 0.0
        point_loss = 0.0 # for hard example mining
        frame_loss = 0.0 # for hard example mining
        seq_len = self._get_seq_len(phase)
        single_frame_input = {}
        idx_info = self._get_idx_info(batch_all) # doesn't change over time
        gt_info = batch_all['gt']
        gt_lookahead_info = batch_all['gt_lookahead']
        lookahead_frame_num = len(gt_lookahead_info[0])
        lookahead_frame_ct = 0

        for curr_frame in range(0, seq_len - 1):
            assert(curr_frame not in single_frame_input.keys())
            curr_batch = self._get_batch(pred, batch_all, curr_frame)
            single_frame_input[curr_frame] = curr_batch
            # skip if the number of input frames is fewer than a required number
            if curr_frame < self.cfg['model']['input_frame_num'] - 1: continue

            # get multi-frame frame prediction
            multi_frame_input = []
            for j in range(self.cfg['model']['input_frame_num']):
                multi_frame_input.append(single_frame_input[curr_frame - j])
            # per-step action (+ TCP for RPE) for this rollout step (Alignment A): step index
            # matches gt_pos_idx
            action_step = None
            tcp_step = None
            if 'action' in batch_all:
                rollout_step = curr_frame - (self.cfg['model']['input_frame_num'] - 1)
                action_step = batch_all['action'][:, rollout_step, :]  # (B, action_dim)
                if 'tcp_xyz' in batch_all:
                    tcp_step = batch_all['tcp_xyz'][:, rollout_step, :]  # (B, 3)
            # get prediction
            multi_frame_feats = self._run_pointconv_internaction_networks(
                multi_frame_input, idx_info, action_step, tcp_step)
            out = self._run_prediction_heads(multi_frame_feats)
            pred = self._collect_pointwise_pred_and_gt(out, curr_batch, curr_frame, phase)

            pos_loss_weight = 1.0            
            rend_loss_weight = 3.0
            gt_pos_idx = curr_frame - (self.cfg['model']['input_frame_num'] - 1)
            if phase == "train" or phase == "train_hem":
                if self._is_gt_frame(curr_frame):
                    assert gt_pos_idx == 0
                    if self.cfg['train']['loss_type'] == 'all' or self.cfg['train']['loss_type'] == 'position':
                        point_loss_avg_curr, point_loss_curr, _ = self._compute_position_loss(pred, gt_info ,gt_pos_idx)
                        loss += pos_loss_weight * point_loss_avg_curr                    
                        point_loss += pos_loss_weight * point_loss_curr
                    if self.cfg['train']['loss_type'] == 'all' or self.cfg['train']['loss_type'] == 'rendering':
                        frame_loss_avg_curr, frame_loss_curr = self._compute_rendering_loss(pred, idx_info, gt_info, -1)
                        loss += rend_loss_weight * frame_loss_avg_curr
                        frame_loss += rend_loss_weight * frame_loss_curr
                else:
                    if self.cfg['train']['loss_type'] == 'all' or self.cfg['train']['loss_type'] == 'position':
                        point_loss_avg_curr, point_loss_curr, _ = self._compute_position_loss(pred, gt_info ,gt_pos_idx)
                        loss += pos_loss_weight * point_loss_avg_curr
                        point_loss += pos_loss_weight * point_loss_curr
                    if self.cfg['train']['loss_type'] == 'all' or self.cfg['train']['loss_type'] == 'rendering':
                        frame_loss_avg_curr, frame_loss_curr = self._compute_rendering_loss(pred, idx_info, gt_lookahead_info, lookahead_frame_ct)
                        loss += rend_loss_weight * frame_loss_avg_curr
                        frame_loss += rend_loss_weight * frame_loss_curr
                    lookahead_frame_ct += 1
            elif phase == "test":          
                # loss_curr, loss_all, topk_inds = self._compute_position_loss_test(pred, gt_info)   
                # _ = self._compute_rendering_loss_test(pred, vars, idx_info, cam_info, gt_info, topk_inds, loss_all)   
                self._update_pred_log_using_gs_format(pred, pred_log, gt_info, gt_pos_idx)                
                if gt_pos_idx == len(gt_info[0]['position']) - 1: break # stop inference when the rollout frame reaches the end of GT frames.
            else:
                raise NotImplementedError("phase should be one of train, train_hem, test")
        
        # average over predicted frames
        actual_pred_frame_num = seq_len - 1 - (self.cfg['model']['input_frame_num'] - 1)
        loss = loss / actual_pred_frame_num
        point_loss = point_loss / actual_pred_frame_num
        frame_loss = frame_loss / actual_pred_frame_num

        assert phase == 'test' or gt_pos_idx == len(gt_info[0]['position']) - 1
        assert phase == 'test' or lookahead_frame_num == lookahead_frame_ct
        assert phase == 'test' or actual_pred_frame_num == (1 + lookahead_frame_ct)

        if phase == "test":
            self._store_pred_log(pred_log, batch_all['seq_name'])

        return loss, point_loss, frame_loss

    def _compute_position_loss(self, pred, gt_info, gt_pos_idx):

        all_gt = []
        for i in range(len(gt_info)):
            all_gt.append(gt_info[i]['position'][gt_pos_idx])
        gt_position = torch.cat(all_gt, dim=0)        
        pred_position = pred['pc'][0].points_packed()

        loss_points = self.criterion(pred_position, gt_position)
        # top_k = int(loss_points.shape[0]*1.0)
        # top_k_out = torch.topk(torch.sum(loss_points, dim=-1), k=top_k)
        # loss = torch.mean(top_k_out.values)
        point_loss = torch.sum(loss_points, dim=-1)
        loss = torch.mean(point_loss)

        # return loss, torch.sum(loss_points, dim=-1), top_k_out.indices
        return loss, point_loss, None

    def _compute_position_loss_test(self, pred, gt_info):

        all_gt = []
        for i in range(len(gt_info)):
            all_gt.append(gt_info[i]['position'])
        gt_position = torch.cat(all_gt, dim=0)        
        pred_position = pred['pc'][0].points_packed()

        error = (pred_position - gt_position).abs().detach()
        weight = (error + 1e-8).pow(1.0)
        weight = weight/ (weight.mean() + 1e-8)

        loss_points = self.criterion(pred_position, gt_position)
        top_k = int(pred_position.shape[0] * 0.025)        
        top_k_out = torch.topk(torch.sum(loss_points, dim=-1), k=top_k)
        loss = torch.mean(top_k_out.values)

        return loss, torch.sum(loss_points, dim=-1), top_k_out.indices

    def _compute_rendering_loss(self, pred, idx_info, gt_info, lookahead_frame_ct):
        
        num_pts_per_batch = idx_info['batch_idx'][0] # densest level
        batch_num = len(num_pts_per_batch)
        assert sum(num_pts_per_batch) == pred['pc'][0].points_packed().shape[0]        
        assert len(gt_info) == batch_num
        # TODO: remove the hard-coded value
        assert pred['pc'][0].features_packed().shape[1] == 59 + self.cfg['dataset']['camera_num'] * self.cfg['dataset']['max_object_num']
            
        st = 0
        loss = 0.0
        loss_list = []
        for i in range(batch_num):
            ed = st + num_pts_per_batch[i]

            params = {
                'means3D': pred['pc'][0].points_packed()[st:ed],
                'shs': pred['pc'][0].features_packed()[st:ed, 11:59].reshape(-1, 16, 3),
                'seg_colors': pred['pc'][0].features_packed()[st:ed, 8:11],
                'unnorm_rotations': pred['pc'][0].features_packed()[st:ed, :4],
                'logit_opacities': torch.unsqueeze(pred['pc'][0].features_packed()[st:ed, 4], dim=1),
                'log_scales': pred['pc'][0].features_packed()[st:ed, 5:8],
            }

            # TODO: batch rendering -- now, just using a for loop
            if lookahead_frame_ct != -1:
                loss_curr = self._get_loss(params, gt_info[i][lookahead_frame_ct])
            else:
                loss_curr = self._get_loss(params, gt_info[i])
            
            loss_list.append(loss_curr)
            loss += loss_curr
            st = ed        

        return loss / batch_num, torch.stack(loss_list, dim=0)

    def _update_pred_log_using_gs_format(self, pred, pred_log, gt_info, gt_pos_idx):

        # # for pseudo-GT visualization
        # all_gt_pos = []
        # all_gt_rot = []
        # for i in range(len(gt_info)):
        #     all_gt_pos.append(gt_info[i]['position'][gt_pos_idx])
        #     all_gt_rot.append(gt_info[i]['rotation'][gt_pos_idx])
        # gt_position = torch.cat(all_gt_pos, dim=0) 
        # gt_rotation = torch.cat(all_gt_rot, dim=0)

        pc = pred['pc'][0]
        # go back to the original scale of the pointcloud
        pos = pc.points_packed().clone().detach().cpu().numpy() * (1 / self.cfg['pointcloud']['scaling'])
        # pos = gt_position.clone().detach().cpu().numpy() * (1 / self.cfg['pointcloud']['scaling'])
        feats = pc.features_packed().clone().detach().cpu().numpy()

        if len(pred_log.keys()) == 0:
            pred_log['means3D'] = []
            pred_log['shs'] = []
            pred_log['unnorm_rotations'] = []
            pred_log['seg_colors'] = feats[:, 8:11]
            pred_log['logit_opacities'] = feats[:, [4]]
            pred_log['log_scales'] = feats[:, 5:8]

        pred_log['means3D'].append(pos)
        pred_log['shs'].append(feats[:, 11:59].reshape(-1, 16, 3))
        pred_log['unnorm_rotations'].append(feats[:, :4])
        # pred_log['unnorm_rotations'].append(gt_rotation.clone().detach().cpu().numpy())


    def _store_pred_log(self, pred_log, seq_name):

        pred_log['means3D'] = np.stack(pred_log['means3D'])
        pred_log['shs'] = np.stack(pred_log['shs'])
        pred_log['unnorm_rotations'] = np.stack(pred_log['unnorm_rotations'])
        
        output_dir = os.path.join(self.cfg['output_dir'], self.cfg['exp_name_epoch'])
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)

        assert len(seq_name) == 1
        out_scene_dir = os.path.join(output_dir, seq_name[0])
        if not os.path.isdir(out_scene_dir):
            os.mkdir(out_scene_dir)
        np.savez(f"{out_scene_dir}/params", **pred_log)


    def _get_loss(self, params, curr_data):

        scales = torch.exp(params['log_scales'])
        opacities = torch.squeeze(torch.sigmoid(params['logit_opacities']), axis=-1)

        render_colors, render_alphas, info = rasterization(
            means=params['means3D'] * (1.0 /self.cfg['pointcloud']['scaling']), # back to the original scale for rendering
            quats=params['unnorm_rotations'],
            scales=scales,
            opacities=opacities,
            colors=params['shs'],
            viewmats=curr_data['w2c'],  # [C, 4, 4]
            Ks=curr_data['k'],  # [C, 3, 3]
            width=curr_data['w'],
            height=curr_data['h'],
            sh_degree=3,
            render_mode='RGB+ED'
        )

        # render_depth = torch.squeeze(render_colors[..., 3:4], dim=-1)
        render_colors = render_colors[..., 0:3]

        # first channel encodes the foreground
        valid_mask = curr_data['seg'][:, [0]].float()

        render_colors = render_colors.permute(0, 3, 1, 2)
        predicted_colors_masked = render_colors
        gt_colors_masked = torch.mul(curr_data['im'], valid_mask)
        l1loss = F.l1_loss(predicted_colors_masked, gt_colors_masked, reduction='mean')

        return l1loss

    def _get_seq_len(self, phase):

        return self.cfg[phase]['seq_len']

    def _get_batch(self, pred, batch, curr_frame):

        if self._is_gt_frame(curr_frame):
            curr_batch = self._get_input_data_from_gt(batch, curr_frame)        
        else:
            curr_batch = self._get_input_data_from_prediction(pred, batch, curr_frame)

        return curr_batch

    def _is_gt_frame(self, curr_frame):

        input_frame_num = self.cfg['model']['input_frame_num']

        return curr_frame < input_frame_num 

    def _get_input_data_from_gt(self, batch, curr_frame):

        curr_pc = batch['pc'][curr_frame]
        curr_pc_grid_idx = batch['grid_idx'][curr_frame]
        curr_velocity = batch['velocity'][curr_frame] # curr_pc - prev_pc
        
        data = {       
            'curr_pc': curr_pc, # input
            'curr_pc_grid_idx': curr_pc_grid_idx,
            'curr_velocity': curr_velocity, # input
        }

        return data

    def _get_input_data_from_prediction(self, prediction, batch, curr_frame):

        curr_pc = prediction['pc']
        grid_frame = self.cfg['model']['input_frame_num'] - 1
        curr_pc_grid_idx = batch['grid_idx'][grid_frame] # grid_idx doesn't have to come from prediction
        curr_velocity = prediction['velocity'] # curr_pc - prev_pc

        data = {       
            'curr_pc': curr_pc, # input
            'curr_pc_grid_idx': curr_pc_grid_idx,
            'curr_velocity': curr_velocity, # input
        }

        return data

    def _run_pointconv_internaction_networks(self, multi_frame_input, idx_info, action=None, tcp_xyz=None):

        recent_frame = 0
        velocity = []
        for frame in range(self.cfg['model']['input_frame_num']):
            velocity.append(multi_frame_input[frame]['curr_velocity'])
        velocity = torch.cat(velocity, dim=1) # [B, 8]
        pc = []
        pc_feat = []
        num_points_per_cloud = []

        for l in range(len(multi_frame_input[recent_frame]['curr_pc'])):
            pc.append(multi_frame_input[recent_frame]['curr_pc'][l].points_packed())
            pc_feat.append(multi_frame_input[recent_frame]['curr_pc'][l].features_packed())
            num_points_per_cloud.append(multi_frame_input[recent_frame]['curr_pc'][l].num_points_per_cloud().tolist())

        pc_feat = self.pointconv_internaction_network(velocity, pc_feat, pc, num_points_per_cloud, idx_info, action=action, tcp_xyz=tcp_xyz)

        return pc_feat

    def _run_prediction_heads(self, feats):

        return self.predictor(feats)

    def _collect_pointwise_pred_and_gt(self, network_output, batch, frame, phase):

        curr_pc_pred = batch['curr_pc']
        curr_velocity = batch['curr_velocity']
        curr_pc_grid_idx = batch['curr_pc_grid_idx']
     
        if phase == "train" or phase == "train_hem":
            assert self._get_seq_len(phase) >= self.cfg['model']['input_frame_num'] + 1
            next_pc_all_layers = []     
            next_feat_all_layers = []
            for i in range(len(curr_pc_pred)):
                idx_sel = curr_pc_grid_idx[i]

                if i == 0:
                    network_pos_vel_output_sel = network_output[idx_sel, :3]
                    network_rot_vel_output_sel = network_output[idx_sel, 3:]
                    vel_sel = curr_velocity[idx_sel, :3]
                    assert curr_velocity.shape[1] == 4
                    assert not torch.isnan(curr_velocity[idx_sel, :3]).any()
                    assert torch.equal(vel_sel[:, :3], curr_velocity[:, :3])                    
                    vel_dont_ignore = 1.0 if self.cfg['model']['acceleration_prediction'] else 0.0
                    curr_pc = curr_pc_pred[i].points_packed()                    
                    curr_rot = torch.nn.functional.normalize(curr_pc_pred[i].features_packed()[:, :4])                    

                    next_pc = network_pos_vel_output_sel + vel_dont_ignore*vel_sel + curr_pc

                    if self.cfg['model']['rigid_pose_fitting']:
                        num_points_per_cloud = batch['curr_pc'][0].num_points_per_cloud().tolist()
                        soft_id = curr_pc_pred[i].features_packed()[:, 59:].reshape(
                            -1, self.cfg['dataset']['camera_num'], self.cfg['dataset']['max_object_num']
                        )
                        soft_id = get_one_hot_by_majority_vote(soft_id)
                        soft_id_list = torch.split(soft_id, num_points_per_cloud, dim=0)
                        curr_pc_list = torch.split(curr_pc, num_points_per_cloud, dim=0)
                        curr_rot_list = torch.split(curr_rot, num_points_per_cloud, dim=0)
                        next_pos_list = torch.split(next_pc, num_points_per_cloud, dim=0)
                        
                        next_pos_all = []
                        next_rot_all = []
                        for obj_soft_id, obj_curr_pos, obj_curr_rot, obj_next_pos in zip(
                            soft_id_list, curr_pc_list, curr_rot_list, next_pos_list
                        ):                            
                            next_pos_out, next_rot_out = self._run_pose_fitting(
                                obj_curr_pos, obj_curr_rot, obj_next_pos, obj_soft_id
                            )
                            next_pos_all.append(next_pos_out)
                            next_rot_all.append(next_rot_out)
                        next_pc = torch.cat(next_pos_all, dim=0)
                        next_rot = torch.cat(next_rot_all, dim=0)
                        assert next_pc.shape[0] == curr_pc.shape[0]
                        assert next_rot.shape[0] == curr_pc.shape[0]
                    else:
                        # rotation is already in the scalar first format
                        curr_rot_mat = q2m(curr_rot)
                        rot_delta_mat = r6d2m(network_rot_vel_output_sel)
                        next_rot_mat = torch.matmul(rot_delta_mat, curr_rot_mat)
                        next_rot = m2q(next_rot_mat)
  
                    next_pos_vel = next_pc - curr_pc
                    next_rot_vel = torch.nn.functional.normalize(next_rot) - curr_rot
                    # NOTE: next_scale is currently occupying 5-7 colums.
                    #       I am inserting assert here because I need to check if this is still true if I get more features
                    assert curr_pc_pred[i].features_packed().shape[1] == 59 + self.cfg['dataset']['camera_num'] * self.cfg['dataset']['max_object_num']
                    # currently scale doesn't change over time
                    next_scale = curr_pc_pred[i].features_packed()[:, 5:8]
                    next_feat = torch.concat((next_rot, curr_pc_pred[i].features_packed()[:, 4:]), dim=1)                    
                    assert next_feat.shape[1] == curr_pc_pred[i].features_packed().shape[1]
                else:                    
                    next_pc = next_pc_all_layers[0].points_packed()[idx_sel]
                    next_feat = next_pc_all_layers[0].features_packed()[idx_sel]

                next_pc_list = torch.split(next_pc, curr_pc_pred[i].num_points_per_cloud().tolist(), dim=0)
                next_feat_list = torch.split(next_feat, curr_pc_pred[i].num_points_per_cloud().tolist(), dim=0)
                next_pc_list = [next_pl for next_pl in next_pc_list]
                next_feat_list = [next_fl for next_fl in next_feat_list]
                next_pc = Pointclouds(points=next_pc_list, features=next_feat_list)
                next_feat = torch.cat(next_feat_list, dim=0)
                next_pc_all_layers.append(next_pc)
                next_feat_all_layers.append(next_feat)

                if i == 0:
                    velocity_pred = compute_velocity_input(
                        next_pc.points_packed(), next_pos_vel, next_rot_vel, next_scale, self.cfg
                    )
        else:
            next_pc_all_layers = []
            next_feat_all_layers = []
            rel_R_for_eval = []
            rel_T_for_eval = []
            for i in range(len(curr_pc_pred)):
                idx_sel = curr_pc_grid_idx[i]
                assert len(curr_pc_pred[i].points_list()) == 1, "It should be a single tensor in the list when scene point clouds cannot be divided into object point clouds and batch size is 1"

                if i == 0:
                    network_pos_vel_output_sel = network_output[idx_sel, :3]
                    network_rot_vel_output_sel = network_output[idx_sel, 3:]
                    vel_sel = curr_velocity[idx_sel, :3]
                    assert curr_velocity.shape[1] == 4
                    assert torch.equal(vel_sel[:, :3], curr_velocity[:, :3])                    
                    vel_dont_ignore = 1.0 if self.cfg['model']['acceleration_prediction'] else 0.0                
                    pos_acc = network_pos_vel_output_sel

                    curr_pc = curr_pc_pred[i].points_packed()
                    curr_rot = torch.nn.functional.normalize(curr_pc_pred[i].features_packed()[:, :4])
                    
                    next_pc = pos_acc + vel_dont_ignore*vel_sel + curr_pc

                    if self.cfg['model']['rigid_pose_fitting']:
                        num_points_per_cloud = batch['curr_pc'][0].num_points_per_cloud().tolist()
                        assert len(num_points_per_cloud) == 1
                        soft_id = curr_pc_pred[i].features_packed()[:, 59:].reshape(
                            -1, self.cfg['dataset']['camera_num'], self.cfg['dataset']['max_object_num']
                        )
                        soft_id = get_one_hot_by_majority_vote(soft_id)
                        next_pc, next_rot = self._run_pose_fitting(
                            curr_pc, curr_rot, next_pc, soft_id, rel_R_for_eval, rel_T_for_eval
                        )
                    else:
                        # rotation is already in the scalar first format
                        curr_rot_mat = q2m(curr_rot)
                        rot_delta_mat = r6d2m(network_rot_vel_output_sel)
                        next_rot_mat = torch.matmul(rot_delta_mat, curr_rot_mat)
                        next_rot = m2q(next_rot_mat)

                    next_pos_vel = (next_pc - curr_pc)
                    next_rot_vel = torch.nn.functional.normalize(next_rot) - curr_rot
                    next_feat = torch.concat((next_rot, curr_pc_pred[i].features_packed()[:, 4:]), dim=1)
                    # NOTE: next_scale is currently occupying 5-7 colums.
                    #       I am inserting assert here because I need to check if this is still true if I get more features
                    assert curr_pc_pred[i].features_packed().shape[1] == 59 + self.cfg['dataset']['camera_num'] * self.cfg['dataset']['max_object_num']
                    next_scale = curr_pc_pred[i].features_packed()[:, 5:8]
                else:                               
                    next_pc = next_pc_all_layers[0].points_packed()[idx_sel]
                    next_feat = next_pc_all_layers[0].features_packed()[idx_sel]

                next_pc_list = torch.split(next_pc, curr_pc_pred[i].num_points_per_cloud().tolist(), dim=0)
                next_feat_list = torch.split(next_feat, curr_pc_pred[i].num_points_per_cloud().tolist(), dim=0)
                next_pc_list = [next_pl for next_pl in next_pc_list]
                next_feat_list = [next_fl for next_fl in next_feat_list]
                next_pc = Pointclouds(points=next_pc_list, features=next_feat_list)
                next_feat = torch.cat(next_feat_list, dim=0)
                next_pc_all_layers.append(next_pc)
                next_feat_all_layers.append(next_feat)

                if i == 0:
                    velocity_pred = compute_velocity_input(
                        next_pc.points_packed(), next_pos_vel, next_rot_vel, next_scale, self.cfg
                    )

        pred = {}
        pred['velocity'] = velocity_pred        
        pred['pc'] = next_pc_all_layers

        return pred

    def _run_pose_fitting(self, curr_pc, curr_rot, next_pc, soft_id, rel_R_for_eval=None, rel_T_for_eval=None):

        next_pos_out = torch.zeros_like(next_pc)
        next_rot_out = torch.zeros_like(curr_rot)
        size_check = 0
        for curr_id in range(soft_id.shape[1]):
            soft_idx = torch.argmax(soft_id, dim=1) == curr_id
            size_check += torch.sum(soft_idx)
            if torch.sum(soft_idx) == 0: continue
            Rot_mat, Trl = calc_rigid_transform_torch_ver(curr_pc[soft_idx], next_pc[soft_idx])
            # NOTE: In some corner cases, the torch and numpy version give different results.
            # Rot_mat_ref, Trl_ref = calc_rigid_transform(curr_pc[soft_idx].detach().cpu().numpy(), next_pc[soft_idx].detach().cpu().numpy())
            # Rot_mat = torch.from_numpy(Rot_mat_ref).to(curr_rot.device).float()
            # Trl = torch.from_numpy(Trl_ref).to(curr_rot.device).float()                        
            # assert np.allclose(Rot_mat_ref, Rot_mat.detach().cpu().numpy(), atol=1e-4)
            # assert np.allclose(Trl_ref, Trl.detach().cpu().numpy(), atol=1e-4)
            Rot_curr = q2m(curr_rot[soft_idx])
            Rot_next_fitted = torch.matmul(Rot_mat[None, :, :], Rot_curr)
            quat_next_fitted = m2q(Rot_next_fitted)
            next_pos_fitted = (torch.matmul(Rot_mat, curr_pc[soft_idx].T) + Trl).T
            next_pos_out[soft_idx] = next_pos_fitted
            next_rot_out[soft_idx] = quat_next_fitted
        assert size_check == next_pos_out.shape[0]

        if rel_R_for_eval is not None and rel_T_for_eval is not None:
            rel_R_for_eval.append(Rot_mat.detach().cpu().numpy())
            rel_T_for_eval.append(Trl.detach().cpu().numpy())

        return next_pos_out, next_rot_out

    def _get_idx_info(self, batch):

        idx = {}
        idx['nn_idx_self'] = []
        idx['nn_idx_forward'] = []
        idx['nn_idx_propagate'] = []
        idx['batch_idx'] = []
        for l in sorted(batch['nn_idx_self'].keys()):
            idx['nn_idx_self'].append(
                batch['nn_idx_self'][l].features_packed().to(dtype=torch.long)
            )
            idx['batch_idx'].append(
                    batch['batch_idx'][l].tolist()
            )
            if l != max(batch['nn_idx_self'].keys()):
                idx['nn_idx_forward'].append(
                    batch['nn_idx_forward'][l].features_packed().to(dtype=torch.long)
                )
                idx['nn_idx_propagate'].append(
                    batch['nn_idx_propagate'][l].features_packed().to(dtype=torch.long)
                )
        
        return idx

    def _get_cam_info(self, batch):

        return {'cam_m': batch['cam_m'], 'cam_c': batch['cam_c']}

