import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from model.module.Linear_super import LinearSuper
from model.module.layernorm_super import LayerNormSuper
from model.module.multihead_super import AttentionSuper
from model.module.embedding_super import PatchembedSuper
from model.utils import trunc_normal_
from model.utils import DropPath
import numpy as np
from einops import rearrange, repeat

from model.utils import mode_hylite

def gelu(x: torch.Tensor) -> torch.Tensor:
    if hasattr(torch.nn.functional, 'gelu'):
        return torch.nn.functional.gelu(x.float()).type_as(x)
    else:
        return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class Vision_TransformerSuper(nn.Module): #change

    def __init__(self, img_size=7, patch_size=1, in_chans=49, num_patches=200, num_classes=16, embed_dim=256, depth=12, local_attn=0,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., pre_norm=True, scale=False, gp=False, relative_position=False, change_qkv=False, abs_pos = True, max_relative_position=14, return_attn=False):
        super(Vision_TransformerSuper, self).__init__()
        # the configs of super arch
        self.super_embed_dim = embed_dim
        # self.super_embed_dim = args.embed_dim
        if mode_hylite:
            self.super_local_attn = local_attn
        self.super_mlp_ratio = mlp_ratio
        self.super_layer_num = depth
        self.super_num_heads = num_heads
        self.super_dropout = drop_rate
        self.super_attn_dropout = attn_drop_rate
        self.num_classes = num_classes
        self.pre_norm=pre_norm
        self.scale=scale
        self.patch_embed_super = PatchembedSuper(img_size=img_size, patch_size=patch_size,
                                                 in_chans=in_chans, embed_dim=embed_dim, num_patches=num_patches)
        self.gp = gp
        self.return_attn = return_attn

        # configs for the sampled subTransformer
        self.sample_embed_dim = None
        self.sample_mlp_ratio = None
        self.sample_layer_num = None
        self.sample_num_heads = None
        if mode_hylite:
            self.sample_local_attn = None
        self.sample_dropout = None
        self.sample_output_dim = None

        self.blocks = nn.ModuleList()
        
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

        for i in range(depth):
            self.blocks.append(TransformerEncoderLayer(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, local_attn=local_attn,
                                                       qkv_bias=qkv_bias, qk_scale=qk_scale, dropout=drop_rate,
                                                       attn_drop=attn_drop_rate, drop_path=dpr[i],
                                                       pre_norm=pre_norm, scale=self.scale,
                                                       change_qkv=change_qkv, relative_position=relative_position,
                                                       max_relative_position=max_relative_position, return_attn=return_attn))

        # parameters for vision transformer
        num_patches = self.patch_embed_super.num_patches

        self.skipcat = nn.ModuleList([])
        for _ in range(depth-2):
            self.skipcat.append(nn.Conv2d(num_patches+1, num_patches+1, [1, 2], 1, 0))

        self.abs_pos = abs_pos
        # print(self.abs_pos) #True while training
        if self.abs_pos:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
            trunc_normal_(self.pos_embed, std=.02)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        trunc_normal_(self.cls_token, std=.02)

        # self.pos_drop = nn.Dropout(p=drop_rate)
        if self.pre_norm:
            self.norm = LayerNormSuper(super_embed_dim=embed_dim)

        # classifier head
        self.head = LinearSuper(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):

        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'rel_pos_embed'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def set_sample_config(self, config: dict):
        self.sample_embed_dim = config['embed_dim']
        self.sample_mlp_ratio = config['mlp_ratio']
        self.sample_layer_num = config['layer_num']
        self.sample_num_heads = config['num_heads']
        if mode_hylite:
            self.sample_local_attn = config['local_attn']
        self.sample_dropout = calc_dropout(self.super_dropout, self.sample_embed_dim[0], self.super_embed_dim)
        self.patch_embed_super.set_sample_config(self.sample_embed_dim[0])
        self.sample_output_dim = [out_dim for out_dim in self.sample_embed_dim[1:]] + [self.sample_embed_dim[-1]]
        for i, blocks in enumerate(self.blocks):
            # not exceed sample layer number
            if i < self.sample_layer_num:
                sample_dropout = calc_dropout(self.super_dropout, self.sample_embed_dim[i], self.super_embed_dim)
                sample_attn_dropout = calc_dropout(self.super_attn_dropout, self.sample_embed_dim[i], self.super_embed_dim)
                if mode_hylite and self.sample_local_attn[i] == 1:
                    blocks.set_sample_config(is_identity_layer=False,
                                            sample_embed_dim=self.sample_embed_dim[i],
                                            sample_mlp_ratio=self.sample_mlp_ratio[i],
                                            sample_num_heads=self.sample_num_heads[i],
                                            sample_local_attn=self.sample_local_attn[i],
                                            sample_dropout=sample_dropout,
                                            sample_out_dim=self.sample_output_dim[i],
                                            sample_attn_dropout=sample_attn_dropout)
                else:
                    blocks.set_sample_config(is_identity_layer=False,
                                             sample_embed_dim=self.sample_embed_dim[i],
                                             sample_mlp_ratio=self.sample_mlp_ratio[i],
                                             sample_num_heads=self.sample_num_heads[i],
                                             sample_dropout=sample_dropout,
                                             sample_out_dim=self.sample_output_dim[i],
                                             sample_attn_dropout=sample_attn_dropout)
            # exceeds sample layer number
            else:
                blocks.set_sample_config(is_identity_layer=True)
        if self.pre_norm:
            self.norm.set_sample_config(self.sample_embed_dim[-1])
        self.head.set_sample_config(self.sample_embed_dim[-1], self.num_classes)

    def get_sampled_params_numel(self, config):
        self.set_sample_config(config)
        numels = []
        # for name, module in self.named_modules():
        #     if hasattr(module, 'calc_sampled_param_num'):
        #         print(name)
        for name, module in self.named_modules():
            if hasattr(module, 'calc_sampled_param_num'):
                if name.split('.')[0] == 'blocks' and int(name.split('.')[1]) >= config['layer_num']:
                    continue
                if mode_hylite:
                    if not (name.split('.')[0] == 'blocks' and self.sample_local_attn[int(name.split('.')[1])] == 0 and
                        'local_' in name.split('.')[2]):
                        # print(name)
                        numels.append(module.calc_sampled_param_num())
                else:
                    numels.append(module.calc_sampled_param_num())

        return sum(numels) + self.sample_embed_dim[0]* (2 +self.patch_embed_super.num_patches)
    def get_complexity(self, sequence_length):
        total_flops = 0
        total_flops += self.patch_embed_super.get_complexity(sequence_length)
        total_flops += np.prod(self.pos_embed[..., :self.sample_embed_dim[0]].size()) / 2.0
        for blk in self.blocks:
            total_flops +=  blk.get_complexity(sequence_length+1)
        total_flops += self.head.get_complexity(sequence_length+1)
        return total_flops
    def forward_features(self, x):
        # b, n, c = x.shape
        B = x.shape[0]
        x = self.patch_embed_super(x)

        # print("x size is:")
        # print(x.shape)
        cls_tokens = self.cls_token[..., :self.sample_embed_dim[0]].expand(B, -1, -1)
        # cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        # print("x size after cls is:")
        # print(x.shape)
        if self.abs_pos:
            x = x + self.pos_embed[..., :self.sample_embed_dim[0]]
            # x += self.pos_embed[:, :(n + 1)]
        # print(self.sample_dropout) #0.0167
        x = F.dropout(x, p=self.sample_dropout, training=self.training)
        # x = F.dropout(x, p=0.1)

        # start_time = time.time()
        last_output = []
        nl = 0
        all_attns = []
        for blk in self.blocks:
            last_output.append(x)
            if nl > 1:             
               x = self.skipcat[nl-2](torch.cat([x.unsqueeze(3), last_output[nl-2].unsqueeze(3)], dim=3)).squeeze(3)
            if self.return_attn:
                x, attn = blk(x)
                if not isinstance(attn, int):
                    all_attns.append(attn)
            else:
                x = blk(x)
            nl += 1

        loss_align = torch.cdist(x[:, 0], x[:, 1:].mean(1)).mean()
        # loss_align = torch.sum((x[:, 0, ].unsqueeze(1) - x[:, 1:, ]).pow(2),
        #                        dim=-1, keepdim=True).mean(1).mean()

        # print(time.time()-start_time)
        if self.pre_norm:  # True
            x = self.norm(x)

        # print(loss_align, self.gp)

        if self.gp: #True for search, False for train
            if self.return_attn:
                return torch.mean(x[:, 1:], dim=1), loss_align, all_attns
            else:
                return torch.mean(x[:, 1:] , dim=1), loss_align

        if self.return_attn:
            return x[:, 0], loss_align, all_attns
        else:
            return x[:, 0], loss_align

    def forward(self, x):
        if self.return_attn:
            x, loss_align, all_attns = self.forward_features(x)

            # ## move norm from top to here to be the same as hylite
            # # no difference in terms of results
            # if self.pre_norm:  # True during training
            #     x = self.norm(x)

            x = self.head(x)
            # print(x.shape)
            return x, loss_align, all_attns
        else:
            x, loss_align = self.forward_features(x)
            x = self.head(x)
            return x, loss_align

    def forward_features_pre_GAP(self, x):
        B = x.shape[0]
        x = self.patch_embed_super(x)
        cls_tokens = self.cls_token[..., :self.sample_embed_dim[0]].expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        if self.abs_pos:
            x = x + self.pos_embed[..., :self.sample_embed_dim[0]]

        x = F.dropout(x, p=self.sample_dropout, training=self.training)

        # start_time = time.time()
        # for blk in self.blocks:
        #     x = blk(x)
        last_output_1 = []
        nll = 0
        for blk in self.blocks:
            last_output_1.append(x)
            if nll > 1:             
               x = self.skipcat[nll-2](torch.cat([x.unsqueeze(3), last_output_1[nll-2].unsqueeze(3)], dim=3)).squeeze(3)
            x = blk(x)
            nll += 1
        # # print(time.time()-start_time)
        if self.pre_norm:
            x = self.norm(x)

        if self.gp:
           return torch.mean(x[:, 1:] , dim=1)

        return x
    def forward_pre_GAP(self, x):
        x = self.forward_features_pre_GAP(x)
        return x

class TransformerEncoderLayer(nn.Module):
    """Encoder layer block.

    Args:
        args (argparse.Namespace): parsed command-line arguments which
    """

    def __init__(self, dim, num_heads, mlp_ratio=4., local_attn=0, qkv_bias=False, qk_scale=None, dropout=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, pre_norm=True, scale=False, return_attn=False,
                 relative_position=False, change_qkv=False, max_relative_position=14):
        super().__init__()

        # the configs of super arch of the encoder, three dimension [embed_dim, mlp_ratio, and num_heads]
        self.super_embed_dim = dim
        self.super_mlp_ratio = mlp_ratio
        # self.super_ffn_embed_dim_this_layer = int(mlp_ratio)
        self.super_ffn_embed_dim_this_layer = int(mlp_ratio * dim)
        self.super_num_heads = num_heads
        if mode_hylite:
            self.super_local_attn = local_attn
        self.return_attn = return_attn
        self.normalize_before = pre_norm
        self.super_dropout = attn_drop
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.scale = scale
        self.relative_position = relative_position
        # self.super_activation_dropout = getattr(args, 'activation_dropout', 0)

        # the configs of current sampled arch
        self.sample_embed_dim = None
        self.sample_mlp_ratio = None
        if mode_hylite:
            self.sample_local_attn_this_layer = None
        self.sample_ffn_embed_dim_this_layer = None
        self.sample_num_heads_this_layer = None
        self.sample_scale = None
        self.sample_dropout = None
        self.sample_attn_dropout = None
        self.sample_out_dim = None

        self.is_identity_layer = None
        self.attn = AttentionSuper(
            dim, num_heads=num_heads, local_attn=0, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop,
            proj_drop=dropout, scale=self.scale, relative_position=self.relative_position, change_qkv=change_qkv,
            max_relative_position=max_relative_position, return_attn=return_attn
        )

        if mode_hylite:
            # if self.super_local_attn == 1:
            self.local_attn_layer = AttentionSuper(
                dim, num_heads=num_heads, local_attn=local_attn, qkv_bias=qkv_bias, qk_scale=qk_scale,
                attn_drop=attn_drop,
                proj_drop=dropout, scale=self.scale, relative_position=self.relative_position, change_qkv=change_qkv,
                max_relative_position=max_relative_position,
            )

        self.attn_layer_norm = LayerNormSuper(self.super_embed_dim)
        self.ffn_layer_norm = LayerNormSuper(self.super_embed_dim)
        # self.dropout = dropout
        self.activation_fn = torch.nn.GELU()
        # self.normalize_before = args.encoder_normalize_before

        self.fc1 = LinearSuper(super_in_dim=self.super_embed_dim, super_out_dim=self.super_ffn_embed_dim_this_layer)
        self.fc2 = LinearSuper(super_in_dim=self.super_ffn_embed_dim_this_layer, super_out_dim=self.super_embed_dim)


    def set_sample_config(self, is_identity_layer, sample_embed_dim=None, sample_mlp_ratio=None, sample_num_heads=None, sample_local_attn=None,
                          sample_dropout=None, sample_attn_dropout=None, sample_out_dim=None):

        if is_identity_layer:
            self.is_identity_layer = True
            return

        self.is_identity_layer = False

        self.sample_embed_dim = sample_embed_dim
        self.sample_out_dim = sample_out_dim
        self.sample_mlp_ratio = sample_mlp_ratio
        # self.sample_ffn_embed_dim_this_layer = int(sample_mlp_ratio)
        self.sample_ffn_embed_dim_this_layer = int(sample_embed_dim * sample_mlp_ratio)
        self.sample_num_heads_this_layer = sample_num_heads
        if mode_hylite:
            self.sample_local_attn_this_layer = sample_local_attn
        self.sample_dropout = sample_dropout
        # self.sample_dropout = 0.1
        self.sample_attn_dropout = sample_attn_dropout
        self.attn_layer_norm.set_sample_config(sample_embed_dim=self.sample_embed_dim)

        # self.attn.set_sample_config(sample_q_embed_dim=self.sample_num_heads_this_layer*64, sample_num_heads=self.sample_num_heads_this_layer,
        #                             sample_local_attn=0, sample_in_embed_dim=self.sample_embed_dim)

        ##new_60
        # self.attn.set_sample_config(sample_q_embed_dim=self.sample_num_heads_this_layer*60, sample_num_heads=self.sample_num_heads_this_layer,
        #                             sample_local_attn=0, sample_in_embed_dim=self.sample_embed_dim)

        # self.attn.set_sample_config(sample_q_embed_dim=self.sample_num_heads_this_layer * self.sample_embed_dim,
        #                            sample_num_heads=self.sample_num_heads_this_layer,
        #                            sample_local_attn=0, sample_in_embed_dim=self.sample_embed_dim)

        # ## new_setp
        self.attn.set_sample_config(sample_q_embed_dim=self.sample_num_heads_this_layer * (self.sample_embed_dim // self.sample_num_heads_this_layer),
                                    sample_num_heads=self.sample_num_heads_this_layer,
                                    sample_local_attn=0, sample_in_embed_dim=self.sample_embed_dim)

        if mode_hylite:
            if self.sample_local_attn_this_layer == 1:
                self.local_attn_layer.set_sample_config(sample_q_embed_dim=self.sample_num_heads_this_layer*64, sample_num_heads=self.sample_num_heads_this_layer,
                                            sample_local_attn=self.sample_local_attn_this_layer, sample_in_embed_dim=201)

        self.fc1.set_sample_config(sample_in_dim=self.sample_embed_dim, sample_out_dim=self.sample_ffn_embed_dim_this_layer)
        self.fc2.set_sample_config(sample_in_dim=self.sample_ffn_embed_dim_this_layer, sample_out_dim=self.sample_out_dim)

        self.ffn_layer_norm.set_sample_config(sample_embed_dim=self.sample_embed_dim)


    def forward(self, x):
        """
        Args:
            x (Tensor): input to the layer of shape `(batch, patch_num , sample_embed_dim)`

        Returns:
            encoded output of shape `(batch, patch_num, sample_embed_dim)`
        """
        if self.is_identity_layer:
            if self.return_attn:
                return x, 0
            else:
                return x

        residual = x
        x = self.maybe_layer_norm(self.attn_layer_norm, x, before=True)
        if self.return_attn:
            x, attn = self.attn(x)
        else:
            x = self.attn(x)
        x = F.dropout(x, p=self.sample_attn_dropout, training=self.training)
        x = self.drop_path(x)
        x = residual + x
        x = self.maybe_layer_norm(self.attn_layer_norm, x, after=True)

        if mode_hylite:
            if self.sample_local_attn_this_layer == 1:
                # print('local attention layer forward')
                residual = x
                # x = x[:, 1:].permute(0,2,1) #torch.Size([1, 128, 201])
                x = self.maybe_layer_norm(self.attn_layer_norm, x, before=True)
                x = x.permute(0, 2, 1)  # torch.Size([1, 128, 201])
                # residual = x
                # x = self.maybe_layer_norm(self.attn_layer_norm, x, before=True)
                if self.return_attn:
                    x, attn_l = self.local_attn_layer(x)
                else:
                    x = self.local_attn_layer(x)
                # x = self.attn_layer_norm(x.permute(0, 2, 1)).permute(0,2,1)
                x = F.dropout(x, p=self.sample_attn_dropout, training=self.training)
                x = self.drop_path(x)
                # x = F.dropout(x, p=0.1)
                # x = self.maybe_layer_norm(self.attn_layer_norm, x, after=True)
                # x = residual + torch.cat([residual[:,0].unsqueeze(1), x.permute(0,2,1)], dim=1)
                x = residual + x.permute(0,2,1)
                # x = x + residual
                # x = x.permute(0,2,1)
                x = self.maybe_layer_norm(self.attn_layer_norm, x, after=True)

        residual = x
        x = self.maybe_layer_norm(self.ffn_layer_norm, x, before=True)
        x = self.activation_fn(self.fc1(x))
        x = F.dropout(x, p=self.sample_dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.sample_dropout, training=self.training)
        if self.scale:
            x = x * (self.super_mlp_ratio / self.sample_mlp_ratio)
        x = self.drop_path(x)
        x = residual + x
        x = self.maybe_layer_norm(self.ffn_layer_norm, x, after=True)

        if self.return_attn:
            return x, attn
        else:
            return x

    def maybe_layer_norm(self, layer_norm, x, before=False, after=False):
        assert before ^ after
        if after ^ self.normalize_before:
            return layer_norm(x)
        else:
            return x
    def get_complexity(self, sequence_length):
        total_flops = 0
        if self.is_identity_layer:
            return total_flops
        total_flops += self.attn_layer_norm.get_complexity(sequence_length+1)
        total_flops += self.attn.get_complexity(sequence_length+1)
        total_flops += self.ffn_layer_norm.get_complexity(sequence_length+1)
        total_flops += self.fc1.get_complexity(sequence_length+1)
        total_flops += self.fc2.get_complexity(sequence_length+1)
        return total_flops

def calc_dropout(dropout, sample_embed_dim, super_embed_dim):
    return dropout * 1.0 * sample_embed_dim / super_embed_dim





