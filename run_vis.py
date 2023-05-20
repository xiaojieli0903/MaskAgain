# -*- coding: utf-8 -*-
import argparse
import pdb
import math

import cv2
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from decord import VideoReader, cpu
from einops import rearrange
from PIL import Image
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models import create_model
from torchvision import transforms
from torchvision.transforms import ToPILImage

import modeling_pretrain_kd
import utils
from datasets import DataAugmentationForVideoMAE
from masking_generator import TubeMaskingGenerator, RandomMaskingGenerator, TimeMaskingGenerator
from transforms import *


import seaborn
import matplotlib.pyplot as plt
from collections import OrderedDict



class DataAugmentationForVideoMAE(object):
    def __init__(self, args):
        self.input_mean = [0.485, 0.456, 0.406]  # IMAGENET_DEFAULT_MEAN
        self.input_std = [0.229, 0.224, 0.225]  # IMAGENET_DEFAULT_STD
        normalize = GroupNormalize(self.input_mean, self.input_std)
        self.train_augmentation = GroupCenterCrop(args.input_size)
        self.transform = transforms.Compose([
            self.train_augmentation,
            Stack(roll=False),
            ToTorchFormatTensor(div=True),
            normalize,
        ])
        if args.mask_type == 'tube':
            self.masked_position_generator = TubeMaskingGenerator(
                args.window_size, args.mask_ratio)
        elif args.mask_type == 'random':
            self.masked_position_generator = RandomMaskingGenerator(
                args.window_size, args.mask_ratio)
        elif args.mask_type == 'time':
            self.masked_position_generator = TimeMaskingGenerator(
                args.window_size, args.mask_ratio, args.context_ratio)
        else:
            raise NotImplementedError(f'{self.mask_type} is not implemented')

    def __call__(self, images):
        process_data, _ = self.transform(images)
        return process_data, self.masked_position_generator()

    def __repr__(self):
        repr = "(DataAugmentationForVideoMAE,\n"
        repr += "  transform = %s,\n" % str(self.transform)
        repr += "  Masked position generator = %s,\n" % str(
            self.masked_position_generator)
        repr += ")"
        return repr


def get_args():
    parser = argparse.ArgumentParser(
        'VideoMAE visualization reconstruction script', add_help=False)
    parser.add_argument('img_path', type=str, help='input video path')
    parser.add_argument('save_path', type=str, help='save video path')
    parser.add_argument('--module_path', type=str, help='save module path')
    parser.add_argument('model_path',
                        type=str,
                        help='checkpoint path of model')
    parser.add_argument('--mask_type',
                        default='random',
                        choices=['random', 'tube', 'time'],
                        type=str,
                        help='masked strategy of video tokens/patches')
    parser.add_argument('--num_frames', type=int, default=16)
    parser.add_argument('--sampling_rate', type=int, default=4)
    parser.add_argument('--decoder_depth',
                        default=4,
                        type=int,
                        help='depth of decoder')
    parser.add_argument('--input_size',
                        default=224,
                        type=int,
                        help='videos input size for backbone')
    parser.add_argument('--device',
                        default='cuda:0',
                        help='device to use for training / testing')
    parser.add_argument('--imagenet_default_mean_and_std',
                        default=True,
                        action='store_true')
    parser.add_argument(
        '--mask_ratio',
        default=0.75,
        type=float,
        help='ratio of the visual tokens/patches need be masked')

    parser.add_argument(
        '--context_ratio',
        default=0.25,
        type=float,
        help='ratio of the visable context tokens/patches')

    parser.add_argument(
        '--times',
        default=1,
        type=int,
        help='forward times')

    # Model parameters
    parser.add_argument('--model',
                        default='pretrain_videomae_base_patch16_224',
                        type=str,
                        metavar='MODEL',
                        help='Name of model to vis')
    parser.add_argument('--drop_path',
                        type=float,
                        default=0.0,
                        metavar='PCT',
                        help='Drop path rate (default: 0.1)')
    #grass156 soccer118
    parser.add_argument(
        '--idx_f',
        default=0,
        type=int,
        help='frame index')
    parser.add_argument(
        '--idx',
        default=118,
        type=int,
        help='patch index')
    parser.add_argument(
        '--num_k',
        default=10,
        type=int,
        help='vis patch number')


    return parser.parse_args()


def get_model(args):
    print(f"Creating model: {args.model}")
    model = create_model(args.model,
                         pretrained='',
                         drop_path_rate=args.drop_path,
                         drop_block_rate=None,
                         decoder_depth=args.decoder_depth)

    return model


def main(args):
    print(args)
    seed = 0 + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device(args.device)
    cudnn.benchmark = True

    model = get_model(args)
    patch_size = model.encoder.patch_embed.patch_size
    print("Patch size = %s" % str(patch_size))
    args.window_size = (args.num_frames // 2, args.input_size // patch_size[0],
                        args.input_size // patch_size[1])
    args.patch_size = patch_size

    model.to(device)
    checkpoint = torch.load(args.model_path, map_location='cpu')
    print(checkpoint.keys())
    try:
        model.load_state_dict(checkpoint['model'],strict = False)
        #print(checkpoint['model'].keys())
    except:
        new_state_dict = OrderedDict()  
        
        for k, v in checkpoint['module'].items():
            name = 'encoder.'+k   
            new_state_dict[name] = v 
        #print(new_state_dict.keys())
        model.load_state_dict(new_state_dict,strict =False)


    model.eval()


    if args.save_path:
        Path(args.save_path).mkdir(parents=True, exist_ok=True)

    with open(args.img_path, 'rb') as f:
        vr = VideoReader(f, ctx=cpu(0))
    duration = len(vr)
    new_length = 1
    new_step = 1
    skip_length = new_length * new_step
    # frame_id_list = [1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45, 49, 53, 57, 61]

    tmp = np.arange(0, 32, 2) + 60
    frame_id_list = tmp.tolist()
    # average_duration = (duration - skip_length + 1) // args.num_frames
    # if average_duration > 0:
    #     frame_id_list = np.multiply(list(range(args.num_frames)),
    #                             average_duration)
    #     frame_id_list = frame_id_list + np.random.randint(average_duration,
    #                                             size=args.num_frames)

    video_data = vr.get_batch(frame_id_list).asnumpy()
    print(video_data.shape)
    img = [
        Image.fromarray(video_data[vid, :, :, :]).convert('RGB')
        for vid, _ in enumerate(frame_id_list)
    ]

    transforms = DataAugmentationForVideoMAE(args)
    img, bool_masked_pos = transforms((img, None))  # T*C,H,W
    img = img.view((args.num_frames, 3) + img.size()[-2:]).transpose(
        0, 1)  # T*C,H,W -> T,C,H,W -> C,T,H,W
    # img = img.view(( -1 , args.num_frames) + img.size()[-2:])
    bool_masked_pos = torch.from_numpy(bool_masked_pos)

    # img = img[None, :]
    # bool_masked_pos = bool_masked_pos[None, :]
    img = img.unsqueeze(0)

    bool_masked_pos = bool_masked_pos.unsqueeze(0)



    img = img.to(device, non_blocking=True)
    bool_masked_pos = bool_masked_pos.to(
        device, non_blocking=True).flatten(1).to(torch.bool)

    bool_nomask_pos = ~bool_masked_pos


    run_time = 0
    while run_time < args.times:
        print(f'run {run_time}-th times')
        with torch.no_grad():
            #outputs = model(img, bool_masked_pos)
            outputs, att_map_s,v_s,= model(img,bool_masked_pos, is_att = True)
           
            #save original video
            mean = torch.as_tensor(IMAGENET_DEFAULT_MEAN).to(device)[None, :, None,
                                                                     None, None]
            std = torch.as_tensor(IMAGENET_DEFAULT_STD).to(device)[None, :, None,
                                                                   None, None]
            ori_img = img * std + mean  # in [0, 1]
            

            print(ori_img.shape)
            img_squeeze = rearrange(
                ori_img,
                'b c (t p0) (h p1) (w p2) -> b (t h w) (p0 p1 p2) c',
                p0=2,
                p1=patch_size[0],
                p2=patch_size[0])
            print(img_squeeze.shape)
            img_norm = (img_squeeze - img_squeeze.mean(dim=-2, keepdim=True)) / (
                img_squeeze.var(dim=-2, unbiased=True, keepdim=True).sqrt() + 1e-6)
            img_patch = rearrange(img_norm, 'b n p c -> b n (p c)')
            return_token_num = bool_masked_pos.sum()
            vis_num = bool_nomask_pos.sum()

            mask = torch.ones_like(img_patch)
            mask[bool_masked_pos] = 0
            mask = rearrange(mask, 'b n (p c) -> b n p c', c=3)
            mask = rearrange(mask,
                             'b (t h w) (p0 p1 p2) c -> b c (t p0) (h p1) (w p2) ',
                             p0=2,
                             p1=patch_size[0],
                             p2=patch_size[1],
                             h=14,
                             w=14)
            ori_img = ori_img * mask
            imgs = [
                ToPILImage()(ori_img[0, :, vid, :, :].cpu())
                for vid, _ in enumerate(frame_id_list)
            ]
            for id, im in enumerate(imgs):
                if run_time > 0:
                    continue
                im.save(f"{args.save_path}/ori_img{id}.jpg")


            
            att1 =att_map_s[-1].clone()

            att = torch.nn.functional.softmax(att_map_s[-1], dim=-1)
            B, H, P, _, = att.shape
            
            
            
            att = att.reshape(B*H*P,P)
            margin_t = torch.topk(
                         abs(att), args.num_k, dim=-1
                     )[0][:, -1]
            bool_topk_pos_t = att >= margin_t.unsqueeze(-1)
            att = att * bool_topk_pos_t
            att = att.reshape(B,H,P,P)
            

            mask = torch.ones_like(img_patch)
            mask[bool_nomask_pos] = torch.repeat_interleave(abs(att[0,0,195*args.idx_f+args.idx,:]).unsqueeze(1),1536,dim=1)#195=1568/8-1
            mask = rearrange(mask, 'b n (p c) -> b n p c', c=3)
            mask = rearrange(mask,
                                'b (t h w) (p0 p1 p2) c -> b c (t p0) (h p1) (w p2) ',
                                p0=2,
                                p1=patch_size[0],
                                p2=patch_size[1],
                                h=14,
                                w=14)
            
            fig = plt.figure()
 
            img_heat = mask[0,0]
            for f in range(8):
                img_path = f"{args.save_path}/ori_img{2*f}.jpg"
                    
                img = plt.imread(img_path)
                    
                plt.imshow(img)
                plt.imshow(img_heat[2*f].cpu(), alpha=0.4, cmap='rainbow') 
                plt.colorbar()
                plt.clim(0, None)
                plt.axis('off') 
                plt.savefig(f'{args.save_path}/att_globle{f}.jpg', dpi=200)
                fig.clear()
            
            att1 = att1.reshape(B*H*P*8,P//8)
            att1 = torch.nn.functional.softmax(att1, dim=-1)
            margin_t = torch.topk(
                         abs(att1), args.num_k, dim=-1
                     )[0][:, -1]
            bool_topk_pos_t = att1 >= margin_t.unsqueeze(-1)
            att1 = att1 * bool_topk_pos_t
            att1 = att1.reshape(B,H,P,P)
            

            mask = torch.ones_like(img_patch)
            mask[bool_nomask_pos] = torch.repeat_interleave(abs(att1[0,0,192*args.idx_f+args.idx,:]).unsqueeze(1),1536,dim=1)
            mask = rearrange(mask, 'b n (p c) -> b n p c', c=3)
            mask = rearrange(mask,
                                'b (t h w) (p0 p1 p2) c -> b c (t p0) (h p1) (w p2) ',
                                p0=2,
                                p1=patch_size[0],
                                p2=patch_size[1],
                                h=14,
                                w=14)
            
            fig = plt.figure()
 
            img_heat = mask[0,0]
            for f in range(8):
                img_path = f"{args.save_path}/ori_img{2*f}.jpg"
                    
                img = plt.imread(img_path)
                    
                plt.imshow(img)
                plt.imshow(img_heat[2*f].cpu(), alpha=0.4, cmap='rainbow')  
                plt.colorbar()
                plt.clim(0, None) 
                plt.axis('off') 
                plt.savefig(f'{args.save_path}/att_frame{f}.jpg', dpi=200)
                fig.clear()
            

            abs_t = abs(outputs).sum(-1)
                
            margin_t = torch.topk(
                    abs_t, args.num_k, dim=-1
                )[0][:, -1]
            bool_topk_pos_t = abs_t >= margin_t.unsqueeze(-1)
            outputs = (outputs * bool_topk_pos_t.unsqueeze(-1))
            abs_t = abs(outputs).sum(-1)
            

            mask = torch.ones_like(img_patch)
            mask[bool_nomask_pos] = torch.repeat_interleave(abs_t.unsqueeze(-1),1536,dim=-1)
            mask = rearrange(mask, 'b n (p c) -> b n p c', c=3)
            mask = rearrange(mask,
                                'b (t h w) (p0 p1 p2) c -> b c (t p0) (h p1) (w p2) ',
                                p0=2,
                                p1=patch_size[0],
                                p2=patch_size[1],
                                h=14,
                                w=14)
            
            fig = plt.figure()
 
            img_heat = mask[0,0]
            for f in range(8):
                img_path = f"{args.save_path}/ori_img{2*f}.jpg"
                    
                img = plt.imread(img_path)
                    
                plt.imshow(img)
                plt.imshow(img_heat[2*f].cpu(), alpha=0.4, cmap='rainbow')  
                plt.colorbar()
                plt.clim(0, None) 
                plt.axis('off') 
                plt.savefig(f'{args.save_path}/feature{f}.jpg', dpi=200)
                fig.clear()
            

            run_time += 1
    print('done')


if __name__ == '__main__':
    opts = get_args()
    main(opts)
