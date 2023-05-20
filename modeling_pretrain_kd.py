#import math
from functools import partial

import os
import torch
import torch.nn as nn
#import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.models.layers import trunc_normal_ as __call_trunc_normal_
from timm.models.registry import register_model

from modeling_finetune_kd import (Block, PatchEmbed, _cfg,
                               get_sinusoid_encoding_table)
from mlp import MLP


def trunc_normal_(tensor, mean=0., std=1.):
    __call_trunc_normal_(tensor, mean=mean, std=std, a=-std, b=std)


def get_kl(mu, var):
    num_preds = var.shape[0] * var.shape[1]  # batchsize * num_patches
    loss = -0.5 * torch.sum(1 + var - mu.pow(2) - var.exp()) / num_preds
    return loss


__all__ = [
    'pretrain_videomae_small_patch16_224',
    'pretrain_videomae_base_patch16_224',
    'pretrain_videomae_large_patch16_224',
    'pretrain_videomae_huge_patch16_224',
    'pretrain_videomaevae_base_patch16_224',
    'pretrain_videomaevaev1_base_patch16_224',
    'pretrain_videomaevaev1_id1_base_patch16_224'
]


class PretrainVisionTransformerEncoder(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self,
                 img_size=224,
                 patch_size=16,
                 in_chans=3,
                 num_classes=0,
                 embed_dim=768,
                 depth=12,
                 num_heads=12,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 norm_layer=nn.LayerNorm,
                 init_values=None,
                 tubelet_size=2,
                 use_checkpoint=False,
                 use_learnable_pos_emb=False):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.patch_embed = PatchEmbed(img_size=img_size,
                                      patch_size=patch_size,
                                      in_chans=in_chans,
                                      embed_dim=embed_dim,
                                      tubelet_size=tubelet_size)
        num_patches = self.patch_embed.num_patches
        self.use_checkpoint = use_checkpoint

        # TODO: Add the cls token
        if use_learnable_pos_emb:
            self.pos_embed = nn.Parameter(
                torch.zeros(1, num_patches + 1, embed_dim))
        else:
            # sine-cosine positional embeddings
            self.pos_embed = get_sinusoid_encoding_table(
                num_patches, embed_dim)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)
               ]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(dim=embed_dim,
                  num_heads=num_heads,
                  mlp_ratio=mlp_ratio,
                  qkv_bias=qkv_bias,
                  qk_scale=qk_scale,
                  drop=drop_rate,
                  attn_drop=attn_drop_rate,
                  drop_path=dpr[i],
                  norm_layer=norm_layer,
                  init_values=init_values) for i in range(depth)
        ])
        #self.norm = norm_layer(embed_dim)
        # self.head = nn.Linear(
        #     embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        if use_learnable_pos_emb:
            trunc_normal_(self.pos_embed, std=.02)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}



    def forward_features(self, x, mask,  is_att = False):
        _, _, T, _, _ = x.shape
        x = self.patch_embed(x)

        x = x + self.pos_embed.type_as(x).to(x.device).clone().detach()

        B, _, C = x.shape
        x_vis = x[~mask].reshape(B, -1, C)  # ~mask means visible
        count = 0
        att_map = []
        v_list = []
        encoder_depth = 12
        if self.use_checkpoint:
            if is_att:
                for blk in self.blocks:
                    x_vis,att,v = checkpoint.checkpoint(blk, x_vis, is_att)
                    count = count + 1
                    if(count == encoder_depth):
                        att_map.append(att)
                        v_list.append(v)
            else:
                for blk in self.blocks:
                    count = count + 1
                    x_vis = checkpoint.checkpoint(blk, x_vis, is_att)
                                                   
        else:
            if is_att:
                for blk in self.blocks:
                    count = count + 1
                    x_vis,att,v = blk(x_vis, is_att)
                    if(count == encoder_depth):
                        att_map.append(att)
                        v_list.append(v)
                    
            else:
                for blk in self.blocks:
                    count = count + 1
                    x_vis = blk(x_vis, is_att)
                    


        #x_vis = self.norm(x_vis)
        if is_att:
            return x_vis, att_map, v_list
        else:
            return x_vis
        #return x_vis

    def forward(self, x, mask,  is_att = False):
        if is_att:
            x, att_map, v_list = self.forward_features(x, mask,is_att)
        else:
            x = self.forward_features(x, mask,is_att)
        #x = self.head(x)
        if is_att:
            return x, att_map, v_list
        else:
            return x



# class PretrainVisionTransformerDecoder(nn.Module):
#     """ Vision Transformer with support for patch or hybrid CNN input stage
#     """
#     def __init__(self,
#                  patch_size=16,
#                  num_classes=768,
#                  embed_dim=768,
#                  depth=12,
#                  num_heads=12,
#                  mlp_ratio=4.,
#                  qkv_bias=False,
#                  qk_scale=None,
#                  drop_rate=0.,
#                  attn_drop_rate=0.,
#                  drop_path_rate=0.,
#                  norm_layer=nn.LayerNorm,
#                  init_values=None,
#                  num_patches=196,
#                  tubelet_size=2,
#                  use_checkpoint=False):
#         super().__init__()
#         self.num_classes = num_classes
#         assert num_classes == 3 * tubelet_size * patch_size**2
#         self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
#         self.patch_size = patch_size
#         self.use_checkpoint = use_checkpoint

#         dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)
#                ]  # stochastic depth decay rule
#         self.blocks = nn.ModuleList([
#             Block(dim=embed_dim,
#                   num_heads=num_heads,
#                   mlp_ratio=mlp_ratio,
#                   qkv_bias=qkv_bias,
#                   qk_scale=qk_scale,
#                   drop=drop_rate,
#                   attn_drop=attn_drop_rate,
#                   drop_path=dpr[i],
#                   norm_layer=norm_layer,
#                   init_values=init_values) for i in range(depth)
#         ])
#         self.norm = norm_layer(embed_dim)
#         self.head = nn.Linear(
#             embed_dim, num_classes) if num_classes > 0 else nn.Identity()

#         self.apply(self._init_weights)

#     def _init_weights(self, m):
#         if isinstance(m, nn.Linear):
#             nn.init.xavier_uniform_(m.weight)
#             if isinstance(m, nn.Linear) and m.bias is not None:
#                 nn.init.constant_(m.bias, 0)
#         elif isinstance(m, nn.LayerNorm):
#             nn.init.constant_(m.bias, 0)
#             nn.init.constant_(m.weight, 1.0)

#     def get_num_layers(self):
#         return len(self.blocks)

#     @torch.jit.ignore
#     def no_weight_decay(self):
#         return {'pos_embed', 'cls_token'}

#     def get_classifier(self):
#         return self.head

#     def reset_classifier(self, num_classes, global_pool=''):
#         self.num_classes = num_classes
#         self.head = nn.Linear(
#             self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

#     def forward(self, x, return_token_num, module=None, mask_bool=None, stage=0, is_MasKD=False, is_feat=False, is_att =False, need_decoder = False):
#         count = 0
#         mask_layer = 2
#         f = []
#         att_map = []
#         v_list = []
#         if self.use_checkpoint:
#             for blk in self.blocks:
#                 if is_MasKD and count > mask_layer and stage == 1:
#                     mask_out = checkpoint.checkpoint(blk, mask_out)

#                 if is_att:
#                     x, att,v = checkpoint.checkpoint(blk, x,is_att)
#                     att_map.append(att)
#                     v_list.append(v)
#                 else:
#                     x = checkpoint.checkpoint(blk, x)
#                 if is_feat:              
#                     f.append(x)

#                 if is_MasKD and count == mask_layer:
#                     if stage ==1 :
#                         mask_out, mask_loss = module(x, mask_bool,stage)#torch.Size([2, 1568, 384])
#                     if stage ==2 :
#                         decoder_out = x
#                         mask_out = module(x, mask_bool,stage)#torch.Size([2, 1568, 384])
#                 count+=1
#         else:
#             for blk in self.blocks:
#                 if is_MasKD and count > mask_layer and stage == 1:
#                     mask_out = blk(mask_out)
#                 if is_att:
#                     x, att,v = blk(x, is_att)
#                     att_map.append(att)
#                     v_list.append(v)
#                 else:
#                      x = blk(x, is_att)
#                 if  is_feat:
#                     f.append(x)

#                 if is_MasKD and count == mask_layer :
#                     if stage ==1 :
#                         mask_out, mask_loss = module(x, mask_bool,stage)#torch.Size([2, 1568, 384])
#                     if stage ==2 :
#                         decoder_out = x
#                         mask_out = module(x, mask_bool,stage)#torch.Size([2, 1568, 384])
#                 count+=1

#         #x_before_head torch.Size([N, 1568, 384])
#         #all= x
#         if return_token_num > 0:
#             x = self.head(self.norm(x[:, -return_token_num:])#torch.Size([2, 1176, 1536])
#                           )  # only return the mask tokens predict pixels
#         else:
#             x = self.head(self.norm(x))

#         #x=self.head(self.norm(all))
#         if is_MasKD and stage == 1:
#                 mask_out = self.head(self.norm(mask_out))#torch.Size([2, 1568, 1536])
#                 #mask_out = self.head(self.norm(mask_out[:, -return_token_num:]))#torch.Size([2, 1568, 1536])


#         if not is_MasKD:
#             if not need_decoder or  not is_att or not is_feat:
#                 return x
#             else:
#                 return x,f,att_map,v_list   
#         else:     
#             if stage == 1:           
#                 return x, mask_out, mask_loss
#             elif stage == 2:
#                 return x, mask_out, decoder_out


class PretrainVisionTransformer(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(
            self,
            img_size=224,
            patch_size=16,
            encoder_in_chans=3,
            encoder_num_classes=0,
            encoder_embed_dim=768,
            encoder_depth=12,
            encoder_num_heads=12,
            decoder_num_classes=1536,  #  decoder_num_classes=768, 
            decoder_embed_dim=512,
            decoder_depth=8,
            decoder_num_heads=8,
            mlp_ratio=4.,
            qkv_bias=False,
            qk_scale=None,
            drop_rate=0.,
            attn_drop_rate=0.,
            drop_path_rate=0.,
            norm_layer=nn.LayerNorm,
            init_values=0.,
            use_learnable_pos_emb=False,
            use_checkpoint=False,
            tubelet_size=2,
            num_classes=0,  # avoid the error from create_fn in timm
            in_chans=0,  # avoid the error from create_fn in timm
    ):
        super().__init__()
        self.encoder = PretrainVisionTransformerEncoder(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=encoder_in_chans,
            num_classes=encoder_num_classes,
            embed_dim=encoder_embed_dim,
            depth=encoder_depth,
            num_heads=encoder_num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            norm_layer=norm_layer,
            init_values=init_values,
            tubelet_size=tubelet_size,
            use_checkpoint=use_checkpoint,
            use_learnable_pos_emb=use_learnable_pos_emb)

        # self.decoder = PretrainVisionTransformerDecoder(
        #     patch_size=patch_size,
        #     num_patches=self.encoder.patch_embed.num_patches,
        #     num_classes=decoder_num_classes,
        #     embed_dim=decoder_embed_dim,
        #     depth=decoder_depth,
        #     num_heads=decoder_num_heads,
        #     mlp_ratio=mlp_ratio,
        #     qkv_bias=qkv_bias,
        #     qk_scale=qk_scale,
        #     drop_rate=drop_rate,
        #     attn_drop_rate=attn_drop_rate,
        #     drop_path_rate=drop_path_rate,
        #     norm_layer=norm_layer,
        #     init_values=init_values,
        #     tubelet_size=tubelet_size,
        #     use_checkpoint=use_checkpoint)

        # self.encoder_to_decoder = nn.Linear(encoder_embed_dim,
        #                                     decoder_embed_dim,
        #                                     bias=False)

        #self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.pos_embed = get_sinusoid_encoding_table(
            self.encoder.patch_embed.num_patches, decoder_embed_dim)

        #trunc_normal_(self.mask_token, std=.02)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        #return {'pos_embed', 'cls_token', 'mask_token'}
        return {'pos_embed', 'cls_token'}

    def forward(self, x, mask, is_att =False):
        #_, _, T, _, _ = x.shape
        
        if is_att:
            x_vis,att_map,v_list = self.encoder(x, mask,  is_att = True)
        else:
            x_vis = self.encoder(x, mask)  # [B, N_vis, C_e]

        output_list = []
        if is_att:
            output_list.append(x_vis)
            output_list.append(att_map)
            output_list.append(v_list)  
        else:
            output_list.append(x_vis)
     
        return output_list



class PretrainVisionTransformerVAEDecoder(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
        Support VAE sampling
    """
    def __init__(self,
                 patch_size=16,
                 num_classes=768,
                 embed_dim=768,
                 depth=12,
                 num_heads=12,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 norm_layer=nn.LayerNorm,
                 init_values=None,
                 num_patches=196,
                 tubelet_size=2,
                 use_res=True,
                 vae_index=-1,
                 use_checkpoint=False):
        super().__init__()
        self.num_classes = num_classes
        assert num_classes == 3 * tubelet_size * patch_size**2
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.patch_size = patch_size
        self.vae_index = vae_index
        self.use_checkpoint = use_checkpoint

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)
               ]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(dim=embed_dim,
                  num_heads=num_heads,
                  mlp_ratio=mlp_ratio,
                  qkv_bias=qkv_bias,
                  qk_scale=qk_scale,
                  drop=drop_rate,
                  attn_drop=attn_drop_rate,
                  drop_path=dpr[i],
                  norm_layer=norm_layer,
                  init_values=init_values) for i in range(depth)
        ])
        self.use_res = use_res
        self.norm_mu = norm_layer(embed_dim)
        self.norm_var = norm_layer(embed_dim)
        self.head_mu = nn.Linear(embed_dim, embed_dim)
        self.head_var = nn.Linear(embed_dim, embed_dim)
        if self.use_res:
            self.head = nn.Linear(
                2 * embed_dim, num_classes)\
                if num_classes > 0 else nn.Identity()
            self.norm = norm_layer(2 * embed_dim)
        else:
            self.head = nn.Linear(
                embed_dim,
                num_classes) if num_classes > 0 else nn.Identity()
            self.norm = norm_layer(embed_dim)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(
            self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
    
    def reparameterize(self, mu, var):
        std = torch.exp(0.5*var)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward_vae(self, x):
        # Warning: history shouldn't generate variance and be trained.
        mu = self.head_mu(self.norm_mu(x))
        var = self.head_var(self.norm_var(x))
        z = self.reparameterize(mu, var)  # if self.training else mu
        return z, mu, var

    def forward(self, x, return_token_num):
        index = 0
        if index == self.vae_index:
            x, mu, var = self.forward_vae(x)

        if self.use_checkpoint:
            for blk in self.blocks:
                x = checkpoint.checkpoint(blk, x)
                index += 1
                if index == self.vae_index:
                    x, mu, var = self.forward_vae(x)
        else:
            for blk in self.blocks:
                x = blk(x)
                index += 1
                if index == self.vae_index:
                    x, mu, var = self.forward_vae(x)

        if self.vae_index == -1:
            z, mu, var = self.forward_vae(x)
            if self.use_res:
                x = torch.cat([x, z], dim=-1)
            else:
                x = z

        if return_token_num > 0:
            x = self.head(self.norm(x[:, -return_token_num:]))
        else:
            x = self.head(self.norm(x))

        return x, mu[:, -return_token_num:], var[:, -return_token_num:]


class PretrainVisionVAETransformer(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(
            self,
            img_size=224,
            patch_size=16,
            encoder_in_chans=3,
            encoder_num_classes=0,
            encoder_embed_dim=768,
            encoder_depth=12,
            encoder_num_heads=12,
            decoder_num_classes=1536,
            decoder_embed_dim=512,
            decoder_depth=8,
            decoder_num_heads=8,
            mlp_ratio=4.,
            qkv_bias=False,
            qk_scale=None,
            drop_rate=0.,
            attn_drop_rate=0.,
            drop_path_rate=0.,
            norm_layer=nn.LayerNorm,
            init_values=0.,
            use_learnable_pos_emb=False,
            use_checkpoint=False,
            tubelet_size=2,
            use_res=True,
            vae_index=-1,
            num_classes=0,  # avoid the error from create_fn in timm
            in_chans=0,  # avoid the error from create_fn in timm
    ):
        super().__init__()
        self.encoder = PretrainVisionTransformerEncoder(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=encoder_in_chans,
            num_classes=encoder_num_classes,
            embed_dim=encoder_embed_dim,
            depth=encoder_depth,
            num_heads=encoder_num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            norm_layer=norm_layer,
            init_values=init_values,
            tubelet_size=tubelet_size,
            use_checkpoint=use_checkpoint,
            use_learnable_pos_emb=use_learnable_pos_emb)

        self.decoder = PretrainVisionTransformerVAEDecoder(
            patch_size=patch_size,
            num_patches=self.encoder.patch_embed.num_patches,
            num_classes=decoder_num_classes,
            embed_dim=decoder_embed_dim,
            depth=decoder_depth,
            num_heads=decoder_num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            norm_layer=norm_layer,
            init_values=init_values,
            tubelet_size=tubelet_size,
            use_res=use_res,
            vae_index=vae_index,
            use_checkpoint=use_checkpoint)

        self.encoder_to_decoder = nn.Linear(encoder_embed_dim,
                                            decoder_embed_dim,
                                            bias=False)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.pos_embed = get_sinusoid_encoding_table(
            self.encoder.patch_embed.num_patches, decoder_embed_dim)

        trunc_normal_(self.mask_token, std=.02)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'mask_token'}

    def forward(self, x, mask, is_feat=False):
        _, _, T, _, _ = x.shape
        if is_feat:
            x_vis,f = self.encoder(x, mask, is_feat)  # [B, N_vis, C_e]
        else:
            x_vis = self.encoder(x, mask)  # [B, N_vis, C_e]
        

        x_vis = self.encoder_to_decoder(x_vis)  # [B, N_vis, C_d]
        B, N, C = x_vis.shape
        # we don't unshuffle the correct visible token order,
        # but shuffle the pos embedding accorddingly.
        # 1 * num_patches * c_d
        expand_pos_embed = self.pos_embed.expand(B, -1, -1).type_as(x).to(
            x.device).clone().detach()
        pos_emd_vis = expand_pos_embed[~mask].reshape(B, -1, C)
        pos_emd_mask = expand_pos_embed[mask].reshape(B, -1, C)
        # 1, 1, 384 + 1, 1176, 384
        x_full = torch.cat(
            [x_vis + pos_emd_vis, self.mask_token + pos_emd_mask],
            dim=1)  # [B, N, C_d]
        x, mu, var = self.decoder(x_full,
                         pos_emd_mask.shape[1])  # [B, N_mask, 3 * 16 * 16]
        loss_kl = get_kl(mu, var)
        #print(loss_kl)
        if is_feat:
            return x, loss_kl, f
        else:
            return x, loss_kl


class PretrainVisionCVAETransformer(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(
            self,
            img_size=224,
            patch_size=16,
            encoder_in_chans=3,
            encoder_num_classes=0,
            encoder_embed_dim=768,
            encoder_depth=12,
            encoder_num_heads=12,
            decoder_num_classes=1536,
            decoder_embed_dim=512,
            decoder_depth=8,
            decoder_num_heads=8,
            mlp_ratio=4.,
            qkv_bias=False,
            qk_scale=None,
            drop_rate=0.,
            attn_drop_rate=0.,
            drop_path_rate=0.,
            norm_layer=nn.LayerNorm,
            init_values=0.,
            use_learnable_pos_emb=False,
            use_checkpoint=False,
            tubelet_size=2,
            use_condition=False,
            num_classes=0,  # avoid the error from create_fn in timm
            in_chans=0,  # avoid the error from create_fn in timm
    ):
        super().__init__()
        self.use_contidion = use_condition
        self.encoder = PretrainVisionTransformerEncoder(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=encoder_in_chans,
            num_classes=encoder_num_classes,
            embed_dim=encoder_embed_dim,
            depth=encoder_depth,
            num_heads=encoder_num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            norm_layer=norm_layer,
            init_values=init_values,
            tubelet_size=tubelet_size,
            use_checkpoint=use_checkpoint,
            use_learnable_pos_emb=use_learnable_pos_emb)

        self.decoder = PretrainVisionTransformerDecoder(
            patch_size=patch_size,
            num_patches=self.encoder.patch_embed.num_patches,
            num_classes=decoder_num_classes,
            embed_dim=decoder_embed_dim,
            depth=decoder_depth,
            num_heads=decoder_num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            norm_layer=norm_layer,
            init_values=init_values,
            tubelet_size=tubelet_size,
            use_checkpoint=use_checkpoint)

        self.encoder_to_decoder = nn.Linear(encoder_embed_dim,
                                            decoder_embed_dim,
                                            bias=False)

        self.norm_mu = norm_layer(decoder_embed_dim)
        self.norm_var = norm_layer(decoder_embed_dim)
        self.head_mu = nn.Linear(decoder_embed_dim, decoder_embed_dim)
        self.head_var = nn.Linear(decoder_embed_dim, decoder_embed_dim)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.pos_embed = get_sinusoid_encoding_table(
            self.encoder.patch_embed.num_patches, decoder_embed_dim)

        trunc_normal_(self.mask_token, std=.02)

    def reparameterize(self, mu, var):
        std = torch.exp(0.5*var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'mask_token'}

    def forward(self, x, mask, is_feat=False):
        _, _, T, _, _ = x.shape
        x_vis = self.encoder(x, mask)  # [B, N_vis, C_e]
        x_vis = self.encoder_to_decoder(x_vis)  # [B, N_vis, C_d]

        # generate prior hidden vector for future sequence
        # to do: change mean to query format
        x_context = torch.mean(x_vis, dim=1)  # [B, C_e]
        mu = self.head_mu(self.norm_mu(x_context))
        var = self.head_var(self.norm_var(x_context))
        z = self.reparameterize(mu, var)

        B, N, C = x_vis.shape
        # we don't unshuffle the correct visible token order,
        # but shuffle the pos embedding accorddingly.
        # 1 * num_patches * c_d
        expand_pos_embed = self.pos_embed.expand(B, -1, -1).type_as(x).to(
            x.device).clone().detach()
        pos_emd_vis = expand_pos_embed[~mask].reshape(B, -1, C)
        pos_emd_mask = expand_pos_embed[mask].reshape(B, -1, C)
        # 1, 1, 384 + 1, 1176, 384

        x_full = torch.cat(
            [x_vis + pos_emd_vis,
             self.mask_token + pos_emd_mask + z.reshape(B, 1, C)],
            dim=1)  # [B, N, C_d]
        x = self.decoder(x_full,
                         pos_emd_mask.shape[1])  # [B, N_mask, 3 * 16 * 16]
        loss_kl = get_kl(mu, var)

        return x, loss_kl


class PretrainVisionCVAEv1Transformer(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(
            self,
            img_size=224,
            patch_size=16,
            encoder_in_chans=3,
            encoder_num_classes=0,
            encoder_embed_dim=768,
            encoder_depth=12,
            encoder_num_heads=12,
            decoder_num_classes=1536,
            decoder_embed_dim=512,
            decoder_depth=8,
            decoder_num_heads=8,
            mlp_ratio=4.,
            qkv_bias=False,
            qk_scale=None,
            drop_rate=0.,
            attn_drop_rate=0.,
            drop_path_rate=0.,
            norm_layer=nn.LayerNorm,
            init_values=0.,
            use_learnable_pos_emb=False,
            use_checkpoint=False,
            tubelet_size=2,
            num_classes=0,  # avoid the error from create_fn in timm
            in_chans=0,  # avoid the error from create_fn in timm
    ):
        super().__init__()
        self.encoder = PretrainVisionTransformerEncoder(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=encoder_in_chans,
            num_classes=encoder_num_classes,
            embed_dim=encoder_embed_dim,
            depth=encoder_depth,
            num_heads=encoder_num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            norm_layer=norm_layer,
            init_values=init_values,
            tubelet_size=tubelet_size,
            use_checkpoint=use_checkpoint,
            use_learnable_pos_emb=use_learnable_pos_emb)

        self.decoder = PretrainVisionTransformerDecoder(
            patch_size=patch_size,
            num_patches=self.encoder.patch_embed.num_patches,
            num_classes=decoder_num_classes,
            embed_dim=decoder_embed_dim,
            depth=decoder_depth,
            num_heads=decoder_num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            norm_layer=norm_layer,
            init_values=init_values,
            tubelet_size=tubelet_size,
            use_checkpoint=use_checkpoint)

        self.encoder_to_decoder = nn.Linear(encoder_embed_dim,
                                            decoder_embed_dim,
                                            bias=False)
        self.patch_size = tubelet_size * patch_size ** 2

        self.e_mlp_postprior = MLP(2 * encoder_embed_dim, [decoder_embed_dim,
                                                       decoder_embed_dim])

        self.e_mlp_prior = MLP(encoder_embed_dim, [decoder_embed_dim,
                                                   decoder_embed_dim])

        self.head_mu_prior = nn.Linear(decoder_embed_dim, decoder_embed_dim)
        self.head_var_prior = nn.Linear(decoder_embed_dim, decoder_embed_dim)

        self.head_mu_postprior = nn.Linear(decoder_embed_dim, decoder_embed_dim)
        self.head_var_postprior = nn.Linear(decoder_embed_dim, decoder_embed_dim)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.pos_embed = get_sinusoid_encoding_table(
            self.encoder.patch_embed.num_patches, decoder_embed_dim)

        trunc_normal_(self.mask_token, std=.02)

    def reparameterize(self, mu, var):
        std = torch.exp(0.5*var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def kl_loss(self, mean1, logvar1, mean2, logvar2):
        exponential = logvar1 - logvar2 - torch.pow(
            mean1 - mean2, 2) / logvar2.exp() - torch.exp(logvar1 - logvar2) + 1
        result = -0.5 * torch.sum(exponential, tuple(range(1, len(exponential.shape))))
        return result.mean()

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'mask_token'}

    def forward(self, x, mask, is_feat=False):
        B, C, T, H, W = x.shape
        patch_number = (T * H * W) // self.patch_size
        # mask_all = mask.reshape(B, -1, patch_number)[:, 0, :]
        mask_context = mask.reshape(B, -1, patch_number)[:, 1, :]
        mask_future = mask.reshape(B, -1, patch_number)[:, 2, :]
        x_context = self.encoder(x, mask_context)  # [B, N_vis, C_e]
        x_future = self.encoder(x, mask_future)  # [B, N_vis, C_e]

        x_vis = self.encoder_to_decoder(x_context)  # [B, N_vis, C_d]

        # generate prior hidden vector for future sequence
        # to do: change mean to query format
        h_context = torch.mean(x_context, dim=1)  # [B, C_e]
        h_future = torch.mean(x_future, dim=1)  # [B, C_e]
        h_postprior = self.e_mlp_postprior(torch.cat([h_context, h_future], dim=1))
        h_prior = self.e_mlp_prior(h_context)

        mu_postprior = self.head_mu_postprior(h_postprior)
        var_postprior = self.head_var_postprior(h_postprior)
        z_postprior = self.reparameterize(mu_postprior, var_postprior)

        mu_prior = self.head_mu_prior(h_prior)
        var_prior = self.head_var_prior(h_prior)
        z_prior = self.reparameterize(mu_prior, var_prior)

        B, N, C = x_vis.shape
        # we don't unshuffle the correct visible token order,
        # but shuffle the pos embedding accorddingly.
        # 1 * num_patches * c_d
        expand_pos_embed = self.pos_embed.expand(B, -1, -1).type_as(x).to(
            x.device).clone().detach()
        pos_emd_vis = expand_pos_embed[~mask_context].reshape(B, -1, C)
        pos_emd_mask = expand_pos_embed[mask_context].reshape(B, -1, C)
        # 1, 1, 384 + 1, 1176, 384

        x_full = torch.cat(
            [x_vis + pos_emd_vis,
             self.mask_token + pos_emd_mask + z_postprior.reshape(B, 1, C)],
            dim=1)  # [B, N, C_d]
        x = self.decoder(x_full,
                         pos_emd_mask.shape[1])  # [B, N_mask, 3 * 16 * 16]

        loss_kl = self.kl_loss(mu_postprior,
                               var_postprior,
                               mu_prior,
                               var_prior)

        return x, loss_kl


@register_model
def pretrain_videomae_small_patch16_224(pretrained=False, **kwargs):
    model = PretrainVisionTransformer(img_size=224,
                                      patch_size=16,
                                      encoder_embed_dim=384,
                                      encoder_depth=12,
                                      encoder_num_heads=6,
                                      encoder_num_classes=0,
                                      decoder_num_classes=1536,
                                      decoder_embed_dim=192,
                                      decoder_num_heads=3,
                                      mlp_ratio=4,
                                      qkv_bias=True,
                                      norm_layer=partial(nn.LayerNorm,
                                                         eps=1e-6),
                                      **kwargs)
    model.default_cfg = _cfg()
    if pretrained != '':
        assert os.path.exists(pretrained)
        checkpoint = torch.load(pretrained, map_location="cpu")
        load_pretrained(model, checkpoint["model"])
    return model


@register_model
def pretrain_videomae_base_patch16_224(pretrained=False, **kwargs):
    model = PretrainVisionTransformer(img_size=224,
                                      patch_size=16,
                                      encoder_embed_dim=768,
                                      encoder_depth=12,
                                      encoder_num_heads=12,
                                      encoder_num_classes=0,
                                      decoder_num_classes=1536,
                                      decoder_embed_dim=384,
                                      decoder_num_heads=6,
                                      mlp_ratio=4,
                                      qkv_bias=True,
                                      norm_layer=partial(nn.LayerNorm,
                                                         eps=1e-6),
                                      **kwargs)
    model.default_cfg = _cfg()
    if pretrained != '':
        assert os.path.exists(pretrained)
        checkpoint = torch.load(pretrained, map_location="cpu")
        load_pretrained(model, checkpoint["model"])
    return model


@register_model
def pretrain_videomae_large_patch16_224(pretrained=False, **kwargs):
    model = PretrainVisionTransformer(img_size=224,
                                      patch_size=16,
                                      encoder_embed_dim=1024,
                                      encoder_depth=24,
                                      encoder_num_heads=16,
                                      encoder_num_classes=0,
                                      decoder_num_classes=1536,
                                      decoder_embed_dim=512,
                                      decoder_num_heads=8,
                                      mlp_ratio=4,
                                      qkv_bias=True,
                                      norm_layer=partial(nn.LayerNorm,
                                                         eps=1e-6),
                                      **kwargs)
    model.default_cfg = _cfg()
    if pretrained != '':
        assert os.path.exists(pretrained)
        checkpoint = torch.load(pretrained, map_location="cpu")
        load_pretrained(model, checkpoint["model"])
    return model


@register_model
def pretrain_videomae_huge_patch16_224(pretrained=False, **kwargs):
    model = PretrainVisionTransformer(img_size=224,
                                      patch_size=16,
                                      encoder_embed_dim=1280,
                                      encoder_depth=32,
                                      encoder_num_heads=16,
                                      encoder_num_classes=0,
                                      decoder_num_classes=1536,
                                      decoder_embed_dim=640,
                                      decoder_num_heads=8,
                                      mlp_ratio=4,
                                      qkv_bias=True,
                                      norm_layer=partial(nn.LayerNorm,
                                                         eps=1e-6),
                                      **kwargs)
    model.default_cfg = _cfg()
    if pretrained != '':
        assert os.path.exists(pretrained)
        checkpoint = torch.load(pretrained, map_location="cpu")
        load_pretrained(model, checkpoint["model"])
    return model


@register_model
def pretrain_videomaevae_base_patch16_224(pretrained=False, **kwargs):
    model = PretrainVisionVAETransformer(
        img_size=224,
        patch_size=16,
        encoder_embed_dim=768,
        encoder_depth=12,
        encoder_num_heads=12,
        encoder_num_classes=0,
        decoder_num_classes=1536,
        decoder_embed_dim=384,
        decoder_num_heads=6,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        use_res=True,
        **kwargs)
    model.default_cfg = _cfg()
    if pretrained != '':
        assert os.path.exists(pretrained)
        checkpoint = torch.load(pretrained, map_location="cpu")
        load_pretrained(model, checkpoint["model"])
    return model

@register_model
def pretrain_videomaevaev1_base_patch16_224(pretrained=False, **kwargs):
    model = PretrainVisionVAETransformer(
        img_size=224,
        patch_size=16,
        encoder_embed_dim=768,
        encoder_depth=12,
        encoder_num_heads=12,
        encoder_num_classes=0,
        decoder_num_classes=1536,
        decoder_embed_dim=384,
        decoder_num_heads=6,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        use_res=False,
        **kwargs)
    model.default_cfg = _cfg()
    if pretrained != '':
        assert os.path.exists(pretrained)
        checkpoint = torch.load(pretrained, map_location="cpu")
        load_pretrained(model, checkpoint["model"])
    return model


@register_model
def pretrain_videomaevaev1_id1_base_patch16_224(pretrained=False, **kwargs):
    model = PretrainVisionVAETransformer(
        img_size=224,
        patch_size=16,
        encoder_embed_dim=768,
        encoder_depth=12,
        encoder_num_heads=12,
        encoder_num_classes=0,
        decoder_num_classes=1536,
        decoder_embed_dim=384,
        decoder_num_heads=6,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        use_res=False,
        vae_index=1,
        **kwargs)
    model.default_cfg = _cfg()
    if pretrained != '':
        assert os.path.exists(pretrained)
        checkpoint = torch.load(pretrained, map_location="cpu")
        load_pretrained(model, checkpoint["model"])
    return model


@register_model
def pretrain_videomaecvae_base_patch16_224(pretrained=False, **kwargs):
    model = PretrainVisionCVAETransformer(
        img_size=224,
        patch_size=16,
        encoder_embed_dim=768,
        encoder_depth=12,
        encoder_num_heads=12,
        encoder_num_classes=0,
        decoder_num_classes=1536,
        decoder_embed_dim=384,
        decoder_num_heads=6,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs)
    model.default_cfg = _cfg()
    if pretrained != '':
        assert os.path.exists(pretrained)
        checkpoint = torch.load(pretrained, map_location="cpu")
        load_pretrained(model, checkpoint["model"])
    return model


@register_model
def pretrain_videomaecvae_v1_base_patch16_224(pretrained=False, **kwargs):
    model = PretrainVisionCVAEv1Transformer(
        img_size=224,
        patch_size=16,
        encoder_embed_dim=768,
        encoder_depth=12,
        encoder_num_heads=12,
        encoder_num_classes=0,
        decoder_num_classes=1536,
        decoder_embed_dim=384,
        decoder_num_heads=6,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs)
    model.default_cfg = _cfg()
    if pretrained != '':
        assert os.path.exists(pretrained)
        checkpoint = torch.load(pretrained, map_location="cpu")
        load_pretrained(model, checkpoint["model"])
    return model


def load_pretrained(model, ckpt):
    state_dict = {}
    for key, param in model.named_parameters():
        if key in ckpt.keys():
            if param.shape == ckpt[key].shape:
                state_dict[key] = ckpt[key]
            else:
                print(f'{key} shape mismatched! {param.shape} in model; '
                      f'{ckpt[key].shape} in pretrained.')
        else:
            print(f'{key} not exists in pretrained! --> init randomly.')
    model.load_state_dict(state_dict, strict=False)


if __name__ == '__main__':
    # net = pretrain_videomaevae_base_patch16_224(pretrained='')
    # x = torch.randn(2, 3, 16, 224, 224)
    
    # vis, loss, feats= net(x=x,mask=False,is_feat =True)
    # #print(count)

    # for f in feats:
    #     print(f.shape)
    # net = pretrain_videomae_small_patch16_224(pretrained='')
    # x = torch.randn(2, 3, 16, 224, 224)
    
    # vis, feats= net(x=x,mask=False,is_feat =True)
    # #print(count)

    # for f in feats:
    #     print(f.shape)
    net = pretrain_videomae_base_patch16_224(pretrained='')
    x = torch.randn(2, 3, 16, 224, 224)
    
    vis, feats= net(x=x,mask=False,is_feat =True)
    #print(count)

    for f in feats:
        print(f.shape)


