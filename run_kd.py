#!/usr/bin/python
#coding=utf-8
import argparse
import datetime
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from timm.models import create_model

import modeling_pretrain_kd
import utils
from datasets import build_pretraining_dataset
#
from models.util import Mlp,Linear


from loops_mlp import train_distill as train
# import tensorboard_logger as tb_logger
#
from optim_factory import create_optimizer
from utils import NativeScalerWithGradNormCount as NativeScaler


def get_args():
    parser = argparse.ArgumentParser('VideoMAE pre-training script',
                                     add_help=False)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--save_ckpt_freq', default=50, type=int)
    #
    parser.add_argument('--init_epochs', type=int, default=30,
                        help='init training for two-stage methods')

    # Model parameters
    parser.add_argument('--model_s',
                        default='pretrain_videomae_base_patch16_224',
                        type=str,
                        metavar='MODEL_S',
                        # Metavar: It provides a different name for argsional argument in help messages.
                        help='Name of student model to train')

    parser.add_argument('--decoder_depth',
                        default=4,
                        type=int,
                        help='depth of decoder')

    parser.add_argument('--mask_type',
                        default='tube',
                        choices=['random', 'tube', 'time', 'timesplit','stusplit'],
                        type=str,
                        help='masked strategy of video tokens/patches')

    parser.add_argument(
        '--mask_ratio',
        default='0.75',
        type=str,
        help='ratio of the visual tokens/patches need be masked')

    parser.add_argument(
        '--context_ratio',
        default=0.25,
        type=float,
        help='ratio of the visable context tokens/patches')

    parser.add_argument('--input_size',
                        default=224,
                        type=int,
                        help='videos input size for backbone')

    parser.add_argument('--drop_path',
                        type=float,
                        default=0.0,
                        metavar='PCT',
                        help='Drop path rate (default: 0.1)')

    parser.add_argument('--normlize_target',
                        default=True,
                        type=bool,
                        help='normalized the target patch pixels')

    # optimizer parameters
    parser.add_argument('--opt',
                        default='adamw',
                        type=str,
                        metavar='OPTIMIZER',
                        help='optimizer (default: "adamw"')
    parser.add_argument('--opt_eps',
                        default=1e-8,
                        type=float,
                        metavar='EPSILON',
                        help='optimizer Epsilon (default: 1e-8)')
    parser.add_argument(
        '--opt_betas',
        default=None,
        type=float,
        nargs='+',
        metavar='BETA',
        help='optimizer Betas (default: None, use args default)')
    parser.add_argument('--clip_grad',
                        type=float,
                        default=None,
                        metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum',
                        type=float,
                        default=0.9,
                        metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight_decay',
                        type=float,
                        default=0.05,
                        help='weight decay (default: 0.05)')
    parser.add_argument('--weight_decay_end',
                        type=float,
                        default=None,
                        help="""Final value of the
        weight decay. We use a cosine schedule for WD. 
        (Set the same value with args.weight_decay to keep weight decay no change)"""
                        )

    parser.add_argument('--lr',
                        type=float,
                        default=1.5e-4,
                        metavar='LR',
                        help='learning rate (default: 1.5e-4)')
    parser.add_argument('--warmup_lr',
                        type=float,
                        default=1e-6,
                        metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument(
        '--min_lr',
        type=float,
        default=1e-5,
        metavar='LR',
        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')

    parser.add_argument('--warmup_epochs',
                        type=int,
                        default=40,
                        metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--warmup_steps',
                        type=int,
                        default=-1,
                        metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--use_checkpoint', action='store_true')
    parser.set_defaults(use_checkpoint=False)

    # Augmentation parameters
    parser.add_argument('--color_jitter',
                        type=float,
                        default=0.0,
                        metavar='PCT',
                        help='Color jitter factor (default: 0.4)')
    parser.add_argument(
        '--train_interpolation',
        type=str,
        default='bicubic',
        help=
        'Training interpolation (random, bilinear, bicubic default: "bicubic")'
    )

    # Dataset parameters
    parser.add_argument('--data_path',
                        default='',
                        type=str,
                        help='dataset path')
    parser.add_argument('--imagenet_default_mean_and_std',
                        default=True,
                        action='store_true')
    parser.add_argument('--num_frames', type=int, default=16)
    parser.add_argument('--sampling_rate', type=int, default=4)
    parser.add_argument('--output_dir',
                        default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir',
                        default='',
                        help='path where to tensorboard log')
    parser.add_argument('--device',
                        default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--pretrained', default='',
                        help='finetune from checkpoint')
    parser.add_argument('--auto_resume', action='store_true')
    parser.add_argument('--no_auto_resume',
                        action='store_false',
                        dest='auto_resume')
    parser.set_defaults(auto_resume=True)

    parser.add_argument('--start_epoch',
                        default=0,
                        type=int,
                        metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument(
        '--pin_mem',
        action='store_true',
        help=
        'Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.'
    )
    parser.add_argument('--no_pin_mem',
                        action='store_false',
                        dest='pin_mem',
                        help='')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size',
                        default=1,
                        type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url',
                        default='env://',
                        help='url used to set up distributed training')
    # teacher路径，k100上finetune精度是91.5%
    parser.add_argument('--path_t', type=str,
                        default='',
                        help='teacher model snapshot')
    # distillation
    parser.add_argument('--distill', type=str, default='hint',
                        choices=['rkd', 'hint','attention','similarity',
                        'RP','TF','TS','FCC','PCC','TSCC','PRBF','FRBF','TSRBF','BP','RC','TP','BTP',
                        'AM','kd','FD','MasKD','hinT','hinTop','deltaP','VTop','value','FrameTop','DeltaTop','PIR','hybrid','DC',
                        'deAtt','deF','onlymse'])
    parser.add_argument('-r', '--gamma', type=float, default=1,
                        help='weight for reconstruction')
    parser.add_argument('-a', '--alpha', type=float, default=0,
                        help='weight balance for KL')
    parser.add_argument('-b', '--beta', type=float, default=1,
                        help='weight balance for other KD losses')
    parser.add_argument('--beta_kd1', type=float, default=1,
                        help='weight balance for other KD1 losses')
    parser.add_argument('--beta_kd2', type=float, default=1,
                        help='weight balance for other KD2 losses')

    # KL distillation
    parser.add_argument('--kd_T', type=float, default=4,
                        help='temperature for KD distillation')
    parser.add_argument('--temperature_s', type=float, default=0.1,
                        help='student temperature for relation distillation')
    parser.add_argument('--temperature_t', type=float, default=0.05,
                        help='teacher temperature for relation distillation')
    parser.add_argument('--num_k', type=int, default=10,
                        help='topk for KD distillation')
    parser.add_argument('--num_f', type=int, default=10,
                        help='num feature for KD distillation')

    
    parser.add_argument('--load_decoder', type=bool, default=False, help='student load teacher decoder')


    parser.add_argument('--is_att', type=bool, default=False, help='need attention map')


    parser.add_argument('--onlykd', type=bool, default=False, help='only need kd')
    parser.add_argument('--ckpt_url', default="", help='for openI')




    args = parser.parse_args()



    return args



def get_teacher(args):
    print('==> creating teacher model')
    
    capacity = 'base'

    model = create_model('pretrain_videomae_' + capacity + '_patch16_224',
                            pretrained=args.pretrained,
                            decoder_depth=0,
                            use_checkpoint=True)

    return model


def get_model(args):
    print(f"Creating model: {args.model_s}")
    model = create_model(args.model_s,
                         pretrained=args.pretrained,
                         drop_path_rate=args.drop_path,
                         drop_block_rate=None,
                         decoder_depth=args.decoder_depth,
                         use_checkpoint=args.use_checkpoint)
    return model


def main(args):
    utils.init_distributed_mode(args)
    if args.mask_ratio.find(',') < 0:
        args.mask_ratio = float(args.mask_ratio)
    else:
        args.mask_ratio = [float(ratio)
                           for ratio in args.mask_ratio.split(',')]

    print(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    # 
    cudnn.benchmark = True

    model_t = get_teacher(args)
    model = get_model(args)

    patch_size = model.encoder.patch_embed.patch_size
    print("Student Patch size = %s" % str(patch_size))
    args.window_size = (args.num_frames // 2, args.input_size // patch_size[0],
                        args.input_size // patch_size[1])
    args.patch_size = patch_size

    # get dataset
    dataset_train = build_pretraining_dataset(args)

    num_tasks = utils.get_world_size()
    global_rank = utils.get_rank()
    sampler_rank = global_rank
    num_training_steps_per_epoch = len(
        dataset_train) // args.batch_size // num_tasks

    sampler_train = torch.utils.data.DistributedSampler(dataset_train,
                                                        num_replicas=num_tasks,
                                                        rank=sampler_rank,
                                                        shuffle=True)
    print("Sampler_train = %s" % str(sampler_train))

    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = utils.TensorboardLogger(log_dir=args.log_dir)
    else:
        log_writer = None

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
        worker_init_fn=utils.seed_worker)

    model.to(device)

    if args.load_decoder:
        teacher_checkpoint = torch.load(args.path_t, map_location='cpu')['model']
        decoder_dict = dict([(key,teacher_checkpoint[key]) for key in list(teacher_checkpoint.keys())[-1-4-4*13:-1]])
        model.load_state_dict(decoder_dict, strict=False)

            
        for parameter in model.decoder.parameters():
            parameter.requires_grad=False
        print('student load teacher decoder success')

    model_without_ddp = model
    module_list = torch.nn.ModuleList([])
    module_list_without_ddp = torch.nn.ModuleList([])
    module_list.append(model)
    module_list_without_ddp.append(model_without_ddp)

    dist_module_list = torch.nn.ModuleList([])

    capacity = args.model_s.split('_')[-3]

    if capacity == 'base':
        input_dim = 768
    elif capacity == 'small':
        input_dim = 384
    input_dim_t = 768

    Align_feature = Mlp(input_dim).to(device)
    Align_feature_without_ddp = Align_feature
    module_list.append(Align_feature)
    module_list_without_ddp.append(Align_feature_without_ddp)

    if args.distill in ['hinTop','VTop']:
        Align_valuesum = Linear(input_dim).to(device)
        Align_valuesum_without_ddp = Align_valuesum
        module_list.append(Align_valuesum)
        module_list_without_ddp.append(Align_valuesum_without_ddp)

        Align_valuesum_t = Linear(input_dim_t).to(device)
        Align_valuesum_without_ddp_t = Align_valuesum_t
        module_list.append(Align_valuesum_t)
        module_list_without_ddp.append(Align_valuesum_without_ddp_t)

        module_parameters = sum(p.numel() for p in Align_valuesum.parameters()
                       if p.requires_grad)

        print('One Linear module number of params: {} M'.format(module_parameters / 1e6))
    if args.distill in ['hybrid']:
        Align_valuesum = Linear(input_dim).to(device)
        Align_valuesum_without_ddp = Align_valuesum
        module_list.append(Align_valuesum)
        module_list_without_ddp.append(Align_valuesum_without_ddp)

        Align_valuesum_t = Linear(input_dim_t).to(device)
        Align_valuesum_without_ddp_t = Align_valuesum_t
        module_list.append(Align_valuesum_t)
        module_list_without_ddp.append(Align_valuesum_without_ddp_t)

        Align_valuesum1 = Linear(input_dim).to(device)
        Align_valuesum1_without_ddp = Align_valuesum1
        module_list.append(Align_valuesum1)
        module_list_without_ddp.append(Align_valuesum1_without_ddp)

        Align_valuesum1_t = Linear(input_dim_t).to(device)
        Align_valuesum1_without_ddp_t = Align_valuesum1_t
        module_list.append(Align_valuesum1_t)
        module_list_without_ddp.append(Align_valuesum1_without_ddp_t)

        module_parameters = sum(p.numel() for p in Align_valuesum.parameters()
                       if p.requires_grad)

        print('One Linear module number of params: {} M'.format(module_parameters / 1e6))
    


    n_parameters = sum(p.numel() for p in model.parameters()
                       if p.requires_grad)

    print('Student number of params: {} M'.format(n_parameters / 1e6))

    module_parameters = sum(p.numel() for p in Align_feature.parameters()
                       if p.requires_grad)

    print('One MLP module number of params: {} M'.format(module_parameters / 1e6))

    model_t.to(device)
    teacher_checkpoint = torch.load(args.path_t, map_location='cpu')
    print(teacher_checkpoint.keys())
    try:
        # torch.distributed.barrier()
        model_t.load_state_dict(teacher_checkpoint['model'], strict=False)
        print('load teacher success')
    except:
        model_t.load_state_dict(teacher_checkpoint['module'])



    total_batch_size = args.batch_size * utils.get_world_size()

    args.lr = args.lr * total_batch_size / 256
    args.min_lr = args.min_lr * total_batch_size / 256
    args.warmup_lr = args.warmup_lr * total_batch_size / 256
    print("LR = %.8f" % args.lr)
    print("Batch size = %d" % total_batch_size)
    print("Number of training steps = %d" % num_training_steps_per_epoch)
    print("Number of training examples per epoch = %d" %
          (total_batch_size * num_training_steps_per_epoch))

    if args.distributed:
        module_list_dist = []
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], find_unused_parameters=False)
        model_without_ddp = model.module
        # module_list[0] = model
        module_list_dist.append(model)
        dist_module_list.append(model_without_ddp)

        Align_feature = torch.nn.parallel.DistributedDataParallel(
            Align_feature, device_ids=[args.gpu], find_unused_parameters=False)
        Align_feature_without_ddp = Align_feature.module
        # module_list[0] = model
        module_list_dist.append(Align_feature)
        dist_module_list.append(Align_feature_without_ddp)

       
        if args.distill in ['hinTop','VTop']:
            Align_valuesum = torch.nn.parallel.DistributedDataParallel(
                Align_valuesum, device_ids=[args.gpu],
                find_unused_parameters=False)
            Align_valuesum_without_ddp =Align_valuesum.module
            module_list_dist.append(Align_valuesum)
            dist_module_list.append(Align_valuesum_without_ddp)

            Align_valuesum_t = torch.nn.parallel.DistributedDataParallel(
                Align_valuesum_t, device_ids=[args.gpu],
                find_unused_parameters=False)
            Align_valuesum_without_ddp_t =Align_valuesum_t.module
            module_list_dist.append(Align_valuesum_t)
            dist_module_list.append(Align_valuesum_without_ddp_t)
        if args.distill in ['hybrid']:
            Align_valuesum = torch.nn.parallel.DistributedDataParallel(
                Align_valuesum, device_ids=[args.gpu],
                find_unused_parameters=False)
            Align_valuesum_without_ddp =Align_valuesum.module
            module_list_dist.append(Align_valuesum)
            dist_module_list.append(Align_valuesum_without_ddp)

            Align_valuesum_t = torch.nn.parallel.DistributedDataParallel(
                Align_valuesum_t, device_ids=[args.gpu],
                find_unused_parameters=False)
            Align_valuesum_without_ddp_t =Align_valuesum_t.module
            module_list_dist.append(Align_valuesum_t)
            dist_module_list.append(Align_valuesum_without_ddp_t)

            Align_valuesum1 = torch.nn.parallel.DistributedDataParallel(
                Align_valuesum1, device_ids=[args.gpu],
                find_unused_parameters=False)
            Align_valuesum1_without_ddp =Align_valuesum1.module
            module_list_dist.append(Align_valuesum1)
            dist_module_list.append(Align_valuesum1_without_ddp)

            Align_valuesum1_t = torch.nn.parallel.DistributedDataParallel(
                Align_valuesum1_t, device_ids=[args.gpu],
                find_unused_parameters=False)
            Align_valuesum1_without_ddp_t =Align_valuesum1_t.module
            module_list_dist.append(Align_valuesum1_t)
            dist_module_list.append(Align_valuesum1_without_ddp_t)

        

       
    loss_scaler = NativeScaler()

    if args.distributed:
        optimizer = create_optimizer(args, dist_module_list)
        model_t = torch.nn.parallel.DistributedDataParallel(
            model_t, device_ids=[args.gpu], find_unused_parameters=False)
        module_list_dist.append(model_t)
    else:
        optimizer = create_optimizer(args, module_list)
        module_list.append(model_t)

    print("Use step level LR & WD scheduler!")
    lr_schedule_values = utils.cosine_scheduler(
        args.lr,
        args.min_lr,
        args.epochs,
        num_training_steps_per_epoch,
        warmup_epochs=args.warmup_epochs,
        warmup_steps=args.warmup_steps,
    )
    if args.weight_decay_end is None:
        args.weight_decay_end = args.weight_decay
    wd_schedule_values = utils.cosine_scheduler(args.weight_decay,
                                                args.weight_decay_end,
                                                args.epochs,
                                                num_training_steps_per_epoch)
    print("Max WD = %.7f, Min WD = %.7f" %
          (max(wd_schedule_values), min(wd_schedule_values)))
    

    if args.distributed:
        utils.auto_load_module(args=args,
                          model_without_ddp=dist_module_list,
                          optimizer=optimizer,
                          loss_scaler=loss_scaler)

    else:

        utils.auto_load_module(args=args,
                          model_without_ddp=module_list_without_ddp,
                          optimizer=optimizer,
                          loss_scaler=loss_scaler)

    



    torch.cuda.empty_cache()

    if args.output_dir and utils.is_main_process():
        with open(os.path.join(args.output_dir, "log.json"),
                  mode="a",
                  encoding="utf-8") as f:
            f.write(json.dumps(vars(args), indent=2) + "\n")

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        if log_writer is not None:
            log_writer.set_step(epoch * num_training_steps_per_epoch)
        if args.distributed:
            train_stats = train(
                module_list=module_list_dist,
                data_loader=data_loader_train,
                optimizer=optimizer,
                device=device,
                epoch=epoch,
                loss_scaler=loss_scaler,
                max_norm=args.clip_grad,
                log_writer=log_writer,
                start_steps=epoch * num_training_steps_per_epoch,
                lr_schedule_values=lr_schedule_values,
                wd_schedule_values=wd_schedule_values,
                patch_size=patch_size[0],
                normlize_target=args.normlize_target,
                opt=args
            )
        else:
            train_stats = train(
                module_list=module_list,
                data_loader=data_loader_train,
                optimizer=optimizer,
                device=device,
                epoch=epoch,
                loss_scaler=loss_scaler,
                max_norm=args.clip_grad,
                log_writer=log_writer,
                start_steps=epoch * num_training_steps_per_epoch,
                lr_schedule_values=lr_schedule_values,
                wd_schedule_values=wd_schedule_values,
                patch_size=patch_size[0],
                normlize_target=args.normlize_target,
                opt=args
            )
        if args.output_dir:
            if (epoch +
                1) % args.save_ckpt_freq == 0 or epoch + 1 == args.epochs:
                if args.distributed:
                    utils.save_module(args=args,
                                 model_without_ddp=dist_module_list,
                                 optimizer=optimizer,
                                 loss_scaler=loss_scaler,
                                 epoch=epoch)

                else:
                    utils.save_module(args=args,
                                 model_without_ddp=module_list_without_ddp,
                                 optimizer=optimizer,
                                 loss_scaler=loss_scaler,
                                 epoch=epoch)

        

        log_stats = {
            **{f'train_{k}': v
               for k, v in train_stats.items()}, 'epoch': epoch,
            'n_parameters': n_parameters
        }

        if args.output_dir and utils.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.json"),
                      mode="a",
                      encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")


    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    argss = get_args()
    if argss.output_dir:
        Path(argss.output_dir).mkdir(parents=True, exist_ok=True)
    main(argss)
