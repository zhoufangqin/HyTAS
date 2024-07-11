import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from collections import OrderedDict

def load_state_dict(model, state_dict, prefix='', ignore_missing="relative_position_index"):
    missing_keys = []
    unexpected_keys = []
    error_msgs = []
    metadata = getattr(state_dict, '_metadata', None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata

    def load(module, prefix=''):
        local_metadata = {} if metadata is None else metadata.get(
            prefix[:-1], {})
        module._load_from_state_dict(
            state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + '.')

    load(model, prefix=prefix)

    warn_missing_keys = []
    ignore_missing_keys = []
    for key in missing_keys:
        keep_flag = True
        for ignore_key in ignore_missing.split('|'):
            if ignore_key in key:
                keep_flag = False
                break
        if keep_flag:
            warn_missing_keys.append(key)
        else:
            ignore_missing_keys.append(key)

    missing_keys = warn_missing_keys

    if len(missing_keys) > 0:
        print("Weights of {} not initialized from pretrained model: {}".format(
            model.__class__.__name__, missing_keys))
    if len(unexpected_keys) > 0:
        print("Weights from pretrained model not used in {}: {}".format(
            model.__class__.__name__, unexpected_keys))
    if len(ignore_missing_keys) > 0:
        print("Ignored weights of {} not initialized from pretrained model: {}".format(
            model.__class__.__name__, ignore_missing_keys))
    if len(error_msgs) > 0:
        print('\n'.join(error_msgs))


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        out = self.fn(x, **kwargs)
        # if (not torch.is_tensor(out)) and (isinstance(out, tuple) and len(
        if (not torch.is_tensor(out)) and isinstance(out, tuple) and len(
                out) == 2:  ## might have problem if the last batch size is 2, needs to be changed
            out_, attn = out
            return out_ + x, attn
        else:
            return out + x
        # return self.fn(x, **kwargs) + x


# -------------------------------------------------------------------------------
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        if len(x) == 2:
            x, attn = x
            return self.fn(self.norm(x), **kwargs), attn
        else:
            return self.fn(self.norm(x), **kwargs)


# -------------------------------------------------------------------------------
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


# -------------------------------------------------------------------------------
class Attention(nn.Module):
    def __init__(self, dim, heads, dim_head, dropout):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, mask=None):
        if len(x.size()) == 2:
            x = x.unsqueeze(0)
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        mask_value = -torch.finfo(dots.dtype).max

        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value=True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, mask_value)
            del mask
        attn = dots.softmax(dim=-1)
        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out, {'attn': attn, 'dot_qk': dots}


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_head, dropout, num_channel, mode, use_class_attn=False,
                 spatial_attn=True, cls_token=1):
        super().__init__()

        self.layers = nn.ModuleList([])
        self.spatial_attn = spatial_attn
        self.cls_token = cls_token
        if self.spatial_attn:
            for _ in range(depth):
                self.layers.append(nn.ModuleList([
                    Residual(PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout))),
                    Residual(PreNorm(num_channel + self.cls_token,
                                     Attention(num_channel + self.cls_token, heads=heads, dim_head=dim_head,
                                               dropout=dropout))),
                    Residual(PreNorm(dim, FeedForward(dim, mlp_head, dropout=dropout)))
                ]))
        else:
            for _ in range(depth):
                self.layers.append(nn.ModuleList([
                    # Residual(PreNorm(dim, NeighborhoodAttention(dim, kernel_size=3, dilation=1, num_heads=heads,
                    #                                             qkv_bias=True, attn_drop=dropout, proj_drop=dropout))),
                    Residual(PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout))),
                    Residual(PreNorm(dim, FeedForward(dim, mlp_head, dropout=dropout)))
                ]))

        self.use_class_attn = use_class_attn
        self.mode = mode
        self.skipcat = nn.ModuleList([])
        if mode == 'CAF':
            for _ in range(depth - 2):
                self.skipcat.append(nn.Conv2d(num_channel + self.cls_token, num_channel + self.cls_token, [1, 2], 1, 0))

    def forward(self, x, mask=None):
        if self.mode == 'ViT':
            if self.spatial_attn:
                for attn, attn2, ff in self.layers:
                    x, attn_map = attn(x, mask=mask)
                    x, attn_map_s = attn2(x.permute(0, 2, 1), mask=mask)
                    x = x.permute(0, 2, 1)
                    x = ff(x)

                    if self.use_class_attn:
                        for _, blk in enumerate(self.blocks_token):
                            cls_token_ca, _ = blk(x[:, 1:, ], x[:, 0, :].unsqueeze(1))
                        x = torch.cat((cls_token_ca, x[:, 1:, ]), dim=1)  # torch.Size([64, 201, 64])
            else:
                for attn, ff in self.layers:
                    x, attn_map = attn(x, mask=mask)
                    x = ff(x)

                    if self.use_class_attn:
                        for _, blk in enumerate(self.blocks_token):
                            cls_token_ca, _ = blk(x[:, 1:, ], x[:, 0, :].unsqueeze(1))
                        x = torch.cat((cls_token_ca, x[:, 1:, ]), dim=1)  # torch.Size([64, 201, 64])

        elif self.mode == 'CAF':
            last_output = []
            nl = 0
            if self.spatial_attn:
                for attn, attn2, ff in self.layers:
                    last_output.append(x)
                    if nl > 1:
                        x = self.skipcat[nl - 2](
                            torch.cat([x.unsqueeze(3), last_output[nl - 2].unsqueeze(3)], dim=3)).squeeze(3)

                    x, attn_map_spe = attn(x, mask=mask)
                    # x_cls = x[:,0,:].unsqueeze(1)
                    x, attn_map = attn2(x.permute(0, 2, 1), mask=mask)
                    # x, attn_map_s = attn2(x[:,1:,].permute(0,2,1), mask=mask) ## performed a bit worse: 86.39/74.59/87.79
                    x = x.permute(0, 2, 1)
                    # x = torch.cat((x_cls, x), dim=1)
                    x = ff(x)

                    if self.use_class_attn:
                        for _, blk in enumerate(self.blocks_token):
                            cls_token_ca, _ = blk(x[:, self.cls_token:, ], x[:, 0:self.cls_token, :])
                        #     cls_token_ca = blk(x[:, 1:, ], x_cls.unsqueeze(1))
                        # cls_token_ca += x[:, 0, ].unsqueeze(1) #performed a bit worse
                        x = torch.cat((cls_token_ca, x[:, self.cls_token:, ]), dim=1)  # torch.Size([64, 201, 64])

                    nl += 1
            else:
                for attn, ff in self.layers:
                    last_output.append(x)
                    if nl > 1:
                        x = self.skipcat[nl - 2](
                            torch.cat([x.unsqueeze(3), last_output[nl - 2].unsqueeze(3)], dim=3)).squeeze(3)
                    # x = attn(x) #for NAT
                    x, attn_map = attn(x, mask=mask)
                    x = ff(x)

                    if self.use_class_attn:
                        for _, blk in enumerate(self.blocks_token):
                            cls_token_ca, _ = blk(x[:, self.cls_token:, ], x[:, 0:self.cls_token, :])
                        x = torch.cat((cls_token_ca, x[:, self.cls_token:, ]), dim=1)  # torch.Size([64, 201, 64])

                    nl += 1
        return x, attn_map


class VisionTransformerEncoder(nn.Module):
    def __init__(self,
                 image_size,
                 near_band,
                 num_patches,
                 num_classes,
                 dim,
                 depth,
                 heads,
                 mlp_dim,
                 pool='cls',
                 channels=1,
                 dim_head=16,
                 dropout=0.,
                 emb_dropout=0.,
                 mode='ViT',
                 mask_ratio=0.75,
                 init_scaler=0.,
                 mask_clf=None,
                 mask_method=None,
                 use_class_attn=False,
                 align_loss=None,
                 spatial_attn=False,
                 use_sar=False,
                 use_se=False,
                 ):
        super().__init__()
        self.image_size = image_size
        self.near_band = near_band
        patch_dim = image_size ** 2 * near_band
        band_dim = num_patches
        self.use_cls = True
        self.num_classes = num_classes
        self.num_patches = num_patches
        self.mask_clf = mask_clf
        self.mask_method = mask_method
        self.use_class_attn = use_class_attn
        self.use_sar = use_sar
        self.align_loss = align_loss
        if self.use_cls:
            self.patch_cls = 1  # num_classes
            self.cls_token = nn.Parameter(torch.randn(1, self.patch_cls, dim))
            self.cls_token_ca = nn.Parameter(torch.zeros(1, self.patch_cls, dim))

        else:
            self.patch_cls = 0
        self.pos_embedding = nn.Parameter(
            torch.randn(1, band_dim + self.patch_cls, dim))  # randomly initialised learnable embedding
        self.patch_to_embedding = nn.Linear(patch_dim, dim)
        self.dropout = nn.Dropout(emb_dropout)
        if mode == 'ViT':  ## for pretraining using partial bands
            self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout,
                                           num_patches - int(num_patches * mask_ratio), mode,
                                           use_class_attn=use_class_attn, spatial_attn=spatial_attn)
        else:  ## for finetuning using full bands
            self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout, band_dim, mode,
                                           use_class_attn=use_class_attn, spatial_attn=spatial_attn,
                                           cls_token=self.patch_cls)
        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(nn.LayerNorm(dim),
                                      nn.Linear(dim, num_classes)) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        x = self.patch_to_embedding(x)
        b, n, c = x.shape

        if self.use_cls:
            cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
            # cls_tokens = self.cls_proj(x_cls).unsqueeze(1)
            x = torch.cat((cls_tokens, x), dim=1)
            x += self.pos_embedding[:, :(n + self.patch_cls)]
            x = self.dropout(x)
        else:
            x = self.dropout(x + self.pos_embedding)
        return x

    def forward(self, x, masking_pos=None, mask=None, return_attn=False):
        x = self.forward_features(x)
        b, _, c = x.shape

        x, attn_map = self.transformer(x, mask)

        if self.use_sar:
            attn_mat = attn_map['dot_qk'][:, :, 0:self.patch_cls, self.patch_cls:].max(2).values
            loss_sar = self.sar(attn_mat)

        if self.num_classes > 0:
            if self.align_loss is None:
                x = x.mean(axis=1) if self.pool == 'mean' else self.to_latent(x[:, 0])
                if return_attn:
                    return self.mlp_head(x), attn_map['attn']
                else:
                    return self.mlp_head(x)
            else:
                loss_align = torch.cdist(x[:, 0:self.patch_cls].max(1).values, x[:, self.patch_cls:].mean(1)).mean()
                x = x.mean(axis=1) if self.pool == 'mean' else self.to_latent(x[:, 0:self.patch_cls].max(1).values)
                if self.use_sar:
                    return self.mlp_head(x), loss_align, loss_sar
                else:
                    return self.mlp_head(x), loss_align
        elif (self.use_cls and self.num_classes == 0):
            if self.align_loss is None:
                return self.mlp_head(x)
            else:
                loss_align = torch.cdist(x[:, 0], x[:, 1:].mean(1))
                loss_align = torch.nn.functional.normalize(loss_align).mean()
                return self.mlp_head(x), loss_align
        else:
            x = self.to_latent(x)
        return self.mlp_head(x)


def load_pretrained_model(args, model_file='../pretrained_model/checkpoint-Indian_clf_scratch_full_SF_7_1.pth'):
    pretrained_model = VisionTransformerEncoder(
        image_size=args.input_size,
        near_band=args.in_chans,
        num_patches=args.num_patches,
        num_classes=args.nb_classes,
        dim=64,
        depth=5,  # 5
        heads=4,  # 4
        mlp_dim=8,
        dropout=0.1,  # 0.1
        emb_dropout=0.1,  # 0.1
        mode='CAF',
    )

    checkpoint = torch.load(model_file, map_location='cpu')
    print("Load ckpt from %s" % model_file)
    checkpoint_model = None

    for model_key in 'model|module'.split('|'):
        if model_key in checkpoint:
            checkpoint_model = checkpoint[model_key]
            print("Load state_dict by model_key = %s" % model_key)
            break

    if checkpoint_model is None:
        checkpoint_model = checkpoint
    state_dict = pretrained_model.state_dict()

    for k in ['mlp_head.weight', 'mlp_head.bias']:
        if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
            print(f"Removing key {k} from pretrained checkpoint")
            del checkpoint_model[k]

    all_keys = list(checkpoint_model.keys())
    new_dict = OrderedDict()
    for key in all_keys:
        if key.startswith('backbone.'):
            new_dict[key[9:]] = checkpoint_model[key]
        elif key.startswith('encoder.'):
            # elif key == 'encoder.patch_to_embedding.weight':
            #     new_dict['patch_to_embedding.weight'] = checkpoint_model[key]
            # elif key == 'encoder.patch_to_embedding.bias':
            #     new_dict['patch_to_embedding.bias'] = checkpoint_model[key]
            # elif key == 'encoder.pos_embedding':
            #     new_dict['pos_embedding'] = checkpoint_model[key]
            # elif key == 'encoder.cls_token':
            #     new_dict['cls_token'] = checkpoint_model[key]
            # elif key.startswith('decoder.'):
            new_dict[key[8:]] = checkpoint_model[key]
        else:
            new_dict[key] = checkpoint_model[key]
    checkpoint_model = new_dict

    if 'pos_embedding' in checkpoint_model:
        pos_embed_checkpoint = checkpoint_model['pos_embedding']
        embedding_size = pos_embed_checkpoint.shape[-1]
        num_patches = pretrained_model.num_patches
        num_extra_tokens = pretrained_model.pos_embedding.shape[-2] - num_patches
        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
        new_size = int(num_patches ** 0.5)

        if orig_size != new_size:
            print("Position interpolate from %dx%d to %dx%d" % (orig_size, orig_size, new_size, new_size))
            extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
            pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
            pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
            pos_tokens = torch.nn.functional.interpolate(
                pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
            pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
            new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
            checkpoint_model['pos_embedding'] = new_pos_embed
    load_state_dict(pretrained_model, checkpoint_model, prefix='')
    return pretrained_model