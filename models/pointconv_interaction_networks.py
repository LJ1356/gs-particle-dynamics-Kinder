import torch.nn as nn
import torch
import time
import os
import torch.nn.functional as F
import torch.utils.checkpoint as gradient_checkpoint
from easydict import EasyDict
from tools.utils import get_one_hot_by_majority_vote, get_soft_distribution
from externals.pointconvformer.model_architecture import get_default_configs
from externals.pointconvformer.layers import (
    PointConvStridePE_obj_custom,
    PointConvTransposePE_obj_custom,
    PointConvStridePE_rel_custom,
)


class ActionCrossAttention(nn.Module):
    """Segmented per-cloud action cross-attention for the [1, total_points, C] layout.

    Each cloud's points (queries) attend only to that cloud's [register, action] tokens,
    so action information never leaks across clouds concatenated in the same batch. A
    zero-init residual gate (gamma) makes the block an identity at initialization, so
    enabling conditioning does not perturb a trained/baseline checkpoint at step 0.
    """

    def __init__(self, dim, action_dim=10, num_heads=4, dropout=0.0):
        super().__init__()
        self.action_enc = nn.Sequential(
            nn.Linear(action_dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
        self.register_token = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.normal_(self.register_token, std=0.02)
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, features, counts, action):
        # features: [1, total_points, dim] (one bottleneck level, all clouds concatenated)
        # counts:   list[int] per-cloud point counts at this level (sums to total_points)
        # action:   [B, action_dim], one row per cloud, in the same order as `counts`
        assert features.dim() == 3 and features.shape[0] == 1
        assert len(counts) == action.shape[0], (
            f"num clouds from counts ({len(counts)}) must match action batch "
            f"({action.shape[0]})"
        )
        act_tokens = self.action_enc(action).unsqueeze(1)  # [B, 1, dim]
        outs = []
        for b, f_b in enumerate(torch.split(features, counts, dim=1)):  # [1, n_b, dim]
            if f_b.shape[1] == 0:
                outs.append(f_b)
                continue
            ctx = torch.cat([self.register_token, act_tokens[b:b + 1]], dim=1)  # [1, 2, dim]
            kv = self.norm_kv(ctx)
            upd, _ = self.attn(self.norm_q(f_b), kv, kv, need_weights=False)
            outs.append(f_b + self.gamma * upd)
        return torch.cat(outs, dim=1)


def build_pointconv_interaction_nets(cfg, phase):

    if cfg['model']['type'] == 'Interaction_PointConv':
        model = Interaction_PointConv(
            cfg['pointcloud']['input_dim'],
            cfg['pointcloud']['knn'],
            cfg['pointcloud']['knn_k_decay_factor'],
            cfg['pointcloud']['scaling'],
            cfg['pointcloud']['downsampling_layer_num'],
            cfg['model']['input_frame_num'],
            cfg['model']['pointcloud_feat'],
            cfg['model']['dist_threshold'],
            cfg['model']['soft_id'],
            cfg['model']['pointwise_prediction'],
            cfg['dataset']['camera_num'],
            cfg['dataset']['max_object_num'],
            phase,
            use_action_conditioning=cfg['model'].get('use_action_conditioning', False),
            action_dim=cfg['model'].get('action_dim', 10),
            action_inject_blocks=cfg['model'].get('action_inject_blocks', [0]),
            action_num_heads=cfg['model'].get('action_attention_num_heads', 4),
        )
    else:
        raise NotImplementedError
    
    if phase == 'test':        
        model_path = os.path.join(cfg['exp_dir'], cfg['exp_name'], 'epoch_%02d/model.pt' % cfg['epoch_sel'])
        chkpt = torch.load(model_path)
        state_dict = chkpt['pointconv_internaction_network']
        state_dict = {k.replace('object_modeling_blocks', 'object_modeling')
                       .replace('interaction_modeling_blocks', 'interaction_modeling'): v
                      for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
        model.eval()

    return model.to(cfg['device'])


def build_multi_frame_mlp(cfg, phase):

    pred_dim = cfg['model']['output_dim']
    model = Multi_Frame_MLP(
        cfg['model']['pred_mlp_input_feat_dim'],
        1, # single frame mlp (right now, multiple frames are taken care by the backbone)
        cfg['model']['pred_mlp_hidden_dim'],
        pred_dim
    )
    if phase == 'test':
        model_path = os.path.join(cfg['exp_dir'], cfg['exp_name'], 'epoch_%02d/model.pt' % cfg['epoch_sel'])
        chkpt = torch.load(model_path)
        model.load_state_dict(chkpt['predictor'])
        model.eval()

    return model.to(cfg['device'])


class Interaction_PointConv(nn.Module):

    def __init__(
        self,
        point_dim,
        knn,
        knn_k_decay_factor,
        scaling,
        block_num,
        input_history_len,
        feature_dim,
        dist_threshold,
        soft_id,
        pointwise_prediction,
        camera_num,
        max_obj_num,
        phase,
        relu_slope=0.2,
        use_action_conditioning=False,
        action_dim=10,
        action_inject_blocks=(0,),
        action_num_heads=4,
    ):
        super(Interaction_PointConv, self).__init__()
        self.knn = knn
        self.knn_k_decay_factor = knn_k_decay_factor
        self.block_num = block_num
        self.soft_id = soft_id
        self.camera_num = camera_num
        self.max_obj_num = max_obj_num
        self.phase = phase
        # group_num = 1
        self.conv1 = nn.Linear(input_history_len*(point_dim + 1), feature_dim)
        # self.gn1 = nn.GroupNorm(group_num, feature_dim)
        self.bn1 = nn.BatchNorm1d(feature_dim, momentum=0.1)
        self.conv2 = nn.Linear(feature_dim, feature_dim)
        # self.gn2 = nn.GroupNorm(group_num, feature_dim)
        self.bn2 = nn.BatchNorm1d(feature_dim, momentum=0.1)
        self.conv3 = nn.Linear(feature_dim, feature_dim)
        # self.gn3 = nn.GroupNorm(group_num, feature_dim)
        self.bn3 = nn.BatchNorm1d(feature_dim, momentum=0.1)
        self.relu_slope = relu_slope
        self.dist_threshold = dist_threshold * scaling
        self.pointwise_prediction = pointwise_prediction
        self.object_modeling = nn.ModuleList()
        self.interaction_modeling = nn.ModuleList()
        unet_feature_dim = feature_dim

        # first interaction block without downsampling
        self.object_modeling.append(Object_PointConv(unet_feature_dim, unet_feature_dim))
        self.interaction_modeling.append(Relation_PointConv(unet_feature_dim, unet_feature_dim))
        unet_feature_dim = 2*unet_feature_dim

        # downsampling interaction blocks
        for _ in range(self.block_num):
            self.object_modeling.append(Object_PointConv(int(unet_feature_dim // 2), unet_feature_dim))
            self.interaction_modeling.append(Relation_PointConv(unet_feature_dim, unet_feature_dim))
            unet_feature_dim = 2*unet_feature_dim
        
        # bottleneck interaction blocks
        unet_feature_dim = int(unet_feature_dim // 2)
        bottleneck_dim = unet_feature_dim
        for _ in range(self.block_num):
            self.object_modeling.append(Object_PointConv(unet_feature_dim, unet_feature_dim))
            self.interaction_modeling.append(Relation_PointConv(unet_feature_dim, unet_feature_dim))

        # action conditioning: segmented per-cloud cross-attention injected after a
        # bottleneck block's relational conv. object_modeling indices for the bottleneck
        # blocks are [block_num+1 .. 2*block_num]; action_inject_blocks selects which ones
        # (0 = paper Block 4 = first bottleneck block, 1 = Block 5 = second).
        self.use_action_conditioning = use_action_conditioning
        self.action_inject_layers = set()
        self.action_cross_attn = nn.ModuleDict()
        if use_action_conditioning:
            first_bottleneck_layer = self.block_num + 1
            for o in action_inject_blocks:
                layer_idx = first_bottleneck_layer + o
                assert first_bottleneck_layer <= layer_idx <= 2 * self.block_num, (
                    f"action_inject_blocks entry {o} maps to layer {layer_idx}, "
                    f"outside the bottleneck range [{first_bottleneck_layer}, {2 * self.block_num}]"
                )
                self.action_inject_layers.add(layer_idx)
                self.action_cross_attn[str(layer_idx)] = ActionCrossAttention(
                    bottleneck_dim, action_dim=action_dim, num_heads=action_num_heads
                )

        # upsampling interaction blocks
        for _ in range(self.block_num):
            self.object_modeling.append(Object_PointConv_Transpose(unet_feature_dim, int(unet_feature_dim // 2)))
            self.interaction_modeling.append(Relation_PointConv(int(unet_feature_dim // 2), int(unet_feature_dim // 2)))
            unet_feature_dim = int(unet_feature_dim // 2)
        
        # last interaction block without upsampling (no relational conv in the last block)
        self.object_modeling.append(Object_PointConv(unet_feature_dim, unet_feature_dim))

    def forward(self, xyz_delta, xyz_feat, xyz_sets, num_points_per_cloud, idx_info, action=None):

        assert len(xyz_feat) == len(xyz_sets)
        assert len(num_points_per_cloud) == len(xyz_sets)
        if self.use_action_conditioning:
            assert action is not None, "use_action_conditioning is set but action is None"

        xyz_delta = torch.unsqueeze(xyz_delta, dim=0).contiguous() # [1, B, C]
        xyz_sets_obj_conv = []
        obj_id_conv = []
        nn_idx_obj_conv_self = []
        nn_idx_obj_conv_forward = []
        nn_idx_obj_conv_propagate = []
        cum_len = []
        max_len = []
        for l in range(len(xyz_sets)):
            xyz_sets_obj_conv.append(torch.unsqueeze(xyz_sets[l], dim=0).contiguous()) 
            assert xyz_feat[l][:, 59:].shape[1] == self.camera_num * self.max_obj_num
            obj_id = self._get_id_representation(xyz_feat[l][:, 59:])
            obj_id_conv.append(torch.unsqueeze(obj_id, dim=0).contiguous())          
            nn_idx_obj_conv_self.append(torch.unsqueeze(idx_info['nn_idx_self'][l], dim=0))
            pad = torch.tensor(0, device=xyz_delta.device).reshape(1)
            tmp = torch.cumsum(torch.tensor(num_points_per_cloud[l], device=xyz_delta.device), dim=0)
            cum_len.append(torch.cat((pad, tmp)).to(torch.int32))
            max_len.append(max(num_points_per_cloud[l]))
            if l != len(xyz_sets) - 1:
                nn_idx_obj_conv_forward.append(torch.unsqueeze(idx_info['nn_idx_forward'][l], dim=0))
                nn_idx_obj_conv_propagate.append(torch.unsqueeze(idx_info['nn_idx_propagate'][l], dim=0))

        # knn search for relational pointconv
        nn_idx_rel_conv_self = []
        curr_k = self.knn
        k_all = []
        if self.phase == 'test':
            assert len(idx_info['batch_idx'][0]) == 1, "Testing with batch size > 1 is not supported."
            # compute idx_info again by doing knn on pytorch tensors
            for l in range(len(xyz_sets)):
                curr_k = curr_k if l == 0 else int(curr_k // self.knn_k_decay_factor)
                assert curr_k >= 1
                nn_idx_rel_conv_self.append(self._knn_point(curr_k, xyz_sets_obj_conv[l], xyz_sets_obj_conv[l]))
                k_all.append(curr_k)
        else:
            for i in range(len(xyz_sets)):
                curr_k = curr_k if i == 0 else int(curr_k // self.knn_k_decay_factor)
                assert curr_k >= 1
                nn_idx_rel_conv_self_list = []
                level_l = torch.split(xyz_sets_obj_conv[i], idx_info['batch_idx'][i], dim=1)
                offset = 0
                for level_l_pcd in level_l:
                    nn_idx_rel_conv_self_list.append(self._knn_point(curr_k, level_l_pcd, level_l_pcd) + offset)
                    offset += level_l_pcd.shape[1]
                nn_idx_rel_conv_self.append(torch.cat(nn_idx_rel_conv_self_list, dim=1))
                k_all.append(curr_k)

        # point wise encoding
        features = self.conv1(xyz_delta)         
        features = F.leaky_relu(
            # self.gn1(features.permute(1, 2, 0)).permute(2, 0, 1),            
            self.bn1(features.permute(0, 2, 1)).permute(0, 2, 1),
            negative_slope=self.relu_slope
        )
        features = self.conv2(features)
        features = F.leaky_relu(
            # self.gn2(features.permute(1, 2, 0)).permute(2, 0, 1),
            self.bn2(features.permute(0, 2, 1)).permute(0, 2, 1),
            negative_slope=self.relu_slope
        )

        # first interaction block without downsampling
        features_all = []
        ct = 0
        layer = 0
        features = gradient_checkpoint.checkpoint(
            self.object_modeling[layer],
            self.dist_threshold,
            xyz_sets_obj_conv[ct],
            features,
            nn_idx_obj_conv_self[ct],
            obj_id_conv[ct],
            use_reentrant = False
        )
        features = gradient_checkpoint.checkpoint(
            self.interaction_modeling[layer],
            self.dist_threshold,
            xyz_sets_obj_conv[ct],
            features,
            nn_idx_rel_conv_self[ct],
            obj_id_conv[ct],
            use_reentrant = False
        )
        features_all.append(features)

        # downsampling interaction blocks
        layer += 1        
        for _ in range(self.block_num):
            features = gradient_checkpoint.checkpoint(
                self.object_modeling[layer],
                self.dist_threshold,
                xyz_sets_obj_conv[ct],
                features,
                nn_idx_obj_conv_forward[ct],
                obj_id_conv[ct],
                xyz_sets_obj_conv[ct + 1],
                obj_id_conv[ct + 1],
                use_reentrant = False
            )
            features = gradient_checkpoint.checkpoint(
                self.interaction_modeling[layer],
                self.dist_threshold,
                xyz_sets_obj_conv[ct + 1],
                features,
                nn_idx_rel_conv_self[ct + 1],
                obj_id_conv[ct + 1],
                use_reentrant = False
            )
            features_all.append(features)
            layer += 1
            ct += 1

        # bottleneck interaction blocks
        ct = len(xyz_sets_obj_conv) - 1
        for _ in range(self.block_num):
            features = gradient_checkpoint.checkpoint(
                self.object_modeling[layer],
                self.dist_threshold,
                xyz_sets_obj_conv[ct],
                features,
                nn_idx_obj_conv_self[ct],
                obj_id_conv[ct],
                use_reentrant = False
            )
            features = gradient_checkpoint.checkpoint(
                self.interaction_modeling[layer],
                self.dist_threshold,
                xyz_sets_obj_conv[ct],
                features,
                nn_idx_rel_conv_self[ct],
                obj_id_conv[ct],
                use_reentrant = False
            )
            # inject action after this bottleneck block's relational conv; the next
            # bottleneck/decoder relational conv then propagates it across objects.
            if self.use_action_conditioning and layer in self.action_inject_layers:
                features = self.action_cross_attn[str(layer)](
                    features, num_points_per_cloud[ct], action
                )
            layer += 1

        # upsampling interaction blocks
        ct = len(xyz_sets_obj_conv) - 1
        for _ in range(self.block_num):
            features = gradient_checkpoint.checkpoint(
                self.object_modeling[layer],
                self.dist_threshold,
                xyz_sets_obj_conv[ct],
                features,
                nn_idx_obj_conv_propagate[ct - 1],
                obj_id_conv[ct],
                xyz_sets_obj_conv[ct - 1],
                obj_id_conv[ct - 1],
                features_all[ct - 1],
                use_reentrant = False
            )
            features = gradient_checkpoint.checkpoint(
                self.interaction_modeling[layer],
                self.dist_threshold,
                xyz_sets_obj_conv[ct - 1],
                features,
                nn_idx_rel_conv_self[ct - 1],
                obj_id_conv[ct - 1],
                use_reentrant = False
            )
            layer += 1
            ct -= 1
        
        # last interaction block without upsampling (no relational conv in the last block)
        ct = 0
        features = gradient_checkpoint.checkpoint(
            self.object_modeling[layer],
            self.dist_threshold,
            xyz_sets_obj_conv[ct],
            features,
            nn_idx_obj_conv_self[ct],
            obj_id_conv[ct],
            use_reentrant = False
        )
        features = torch.squeeze(features, dim=0)

        return features

    def _get_id_representation(self, obj_id_raw):
        
        if self.soft_id:
            obj_id = get_soft_distribution(obj_id_raw.reshape(-1, self.camera_num, self.max_obj_num))
        else:
            obj_id = get_one_hot_by_majority_vote(obj_id_raw.reshape(-1, self.camera_num, self.max_obj_num))
        
        return obj_id

    def _compute_knn(self, xyz_sets, knn):

        offset = 0
        nn_ids = []
        for i in range(len(xyz_sets)):
            nn_ids_out = self._knn_point(knn, xyz_sets[i], xyz_sets[i])
            nn_ids.append(nn_ids_out + offset)
            offset += nn_ids_out.shape[1]
        
        return  torch.concat(nn_ids, dim=1)

    def _knn_point(self, nsample, xyz, new_xyz):

        sqrdists = torch.cdist(new_xyz, xyz)
        _, group_idx = torch.topk(
            sqrdists, nsample, dim = -1, largest=False, sorted=False
        )

        return group_idx


class Object_PointConv(nn.Module):

    def __init__(self, channel_in, channel_out, is_res=True):
        
        super(Object_PointConv, self).__init__()
        cfg = get_default_configs(EasyDict())
        cfg.USE_VI = False
        cfg.USE_PE = True
        cfg.num_heads = 1
        cfg.mid_dim = 4

        weightnet_input_dim = cfg.point_dim
        weightnet = [weightnet_input_dim, cfg.mid_dim]
        self.pointconv_res_block = PointConvStridePE_obj_custom(
            channel_in, channel_out, cfg, weightnet, is_res
        )

    def forward(self, dist, dense_xyz, dense_feats, nn_idx, dense_xyz_id, sparse_xyz=None, sparse_xyz_id=None):
        
        features, _ = self.pointconv_res_block(
            dist, dense_xyz, dense_feats, nn_idx, dense_xyz_id, sparse_xyz, sparse_xyz_id
        )

        return features


class Object_PointConv_Transpose(nn.Module):

    def __init__(self, channel_in, channel_out):
        
        super(Object_PointConv_Transpose, self).__init__()
        cfg = get_default_configs(EasyDict())
        cfg.USE_VI = False
        cfg.num_heads = 1
        cfg.mid_dim = 4
        weightnet_input_dim = cfg.point_dim
        weightnet = [weightnet_input_dim, cfg.mid_dim]
        mlp2 = [channel_out, channel_out]

        self.pointconv_res_block = PointConvTransposePE_obj_custom(
            channel_in, channel_out, cfg, weightnet, mlp2
        )

    def forward(self, dist, sparse_xyz, sparse_feats, nn_idx, sparse_xyz_id, dense_xyz, dense_xyz_id, dense_feats):
        
        features, _ = self.pointconv_res_block(
            dist, sparse_xyz, sparse_feats, nn_idx, sparse_xyz_id, dense_xyz, dense_xyz_id, dense_feats
        )

        return features


class Relation_PointConv(nn.Module):
    
    def __init__(self, channel_in, channel_out):
        
        super(Relation_PointConv, self).__init__()
        cfg = get_default_configs(EasyDict())
        cfg.USE_VI = False
        cfg.num_heads = 1
        cfg.mid_dim = 4
        weightnet_input_dim = cfg.point_dim
        weightnet = [weightnet_input_dim, cfg.mid_dim]

        self.pointconv_res_block = PointConvStridePE_rel_custom(
            channel_in, channel_out, cfg, weightnet
        )

    def forward(self, dist, dense_xyz, dense_feats, nn_idx, dense_xyz_id, sparse_xyz=None):
                        
        features = self.pointconv_res_block(
            dist, dense_xyz, dense_feats, nn_idx, dense_xyz_id, sparse_xyz
        )
        
        return features


class MLP(nn.Module):

    def __init__(self, in_feat_dim, hidden_size, out_feat_dim):

        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_feat_dim, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, out_feat_dim),
        )

    def forward(self, x):
    
        return self.layers(x)


class Multi_Frame_MLP(nn.Module):

    def __init__(self, features_dim, frame_num, hidden_size, states_dim):

        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(features_dim*frame_num, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, states_dim),
        )

    def forward(self, x):
    
        return self.layers(x)
