import math
import sys
from typing import Iterable

import torch
import torch.nn as nn


import utils
#import torch.nn.functional as F

from distiller_zoo import HinTop,VTop

def train_distill(module_list,
                    data_loader: Iterable,
                    optimizer: torch.optim.Optimizer,
                    device: torch.device,
                    epoch: int,
                    loss_scaler,
                    opt,
                    max_norm: float = 0,
                    patch_size: int = 16,
                    normlize_target: bool = True,
                    log_writer=None,
                    lr_scheduler=None,
                    start_steps=None,
                    lr_schedule_values=None,
                    wd_schedule_values=None
                    ):
    """One epoch distillation"""
    # set modules as train()
    for module in module_list:
        module.train()
    # set teacher as eval()
    
    module_list[-1].eval()
    model_t = module_list[-1]
    model = module_list[0]


    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter(
        'lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter(
        'min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    loss_func = nn.MSELoss()

    for step, batch in enumerate(
            metric_logger.log_every(data_loader, print_freq, header)):
        # assign learning rate & weight decay for each step
        it = start_steps + step  # global training iteration
        if lr_schedule_values is not None or wd_schedule_values is not None:
            for i, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None:
                    param_group["lr"] = lr_schedule_values[it] * param_group[
                        "lr_scale"]
                if wd_schedule_values is not None and param_group[
                        "weight_decay"] > 0:
                    param_group["weight_decay"] = wd_schedule_values[it]
        videos, bool_masked_pos = batch
        videos = videos.to(device, non_blocking=True)
        bool_masked_pos = bool_masked_pos.to(
            device, non_blocking=True).flatten(1).to(torch.bool)

    
        with torch.no_grad():
            
            if opt.distill in ['VTop','hinTop','hybrid']:
                opt.is_att = True
            elif opt.distill == 'onlymse':
                pass            
            

            output_list_t = model_t(videos, bool_masked_pos,is_att = opt.is_att)
            
            if opt.distill in ['VTop','hinTop','hybrid']:
                output_t,  att_map_t, v_list_t =  output_list_t
                att_map_t = [att.detach() for att in att_map_t]
                v_list_t = [v.detach() for v in v_list_t]
            elif opt.distill == 'onlymse':
                output_t =  output_list_t[0]
        
        with torch.cuda.amp.autocast():
            #import pdb; pdb.set_trace()
            output_list = model(videos, bool_masked_pos, is_att = opt.is_att)

            if opt.distill in ['VTop','hinTop','hybrid']:
                outputs, att_map_s,v_list_s = output_list
            elif opt.distill == 'onlymse':
                outputs =  output_list[0]                     
    

            if opt.onlykd:
                pass
            else:
                batchsize, patch, _ = outputs.shape

                abs_t = abs(output_t).sum(-1)
                
                margin_t = torch.topk(
                    abs_t, opt.num_f, dim=-1
                )[0][:, -1]
                bool_topk_pos_t = abs_t >= margin_t.unsqueeze(-1)
                
                if opt.alpha != 0:
                    temperature_s = opt.temperature_s#0.1
                    temperature_t = opt.temperature_t#0.05
                    len = bool_topk_pos_t.sum()
                    sd = outputs[bool_topk_pos_t]
                    td = output_t[bool_topk_pos_t]
                    # Angle loss
                    with torch.no_grad():
                        td = torch.nn.functional.normalize(td, p=2, dim=-1)
                        td = torch.einsum('cd, ed->ce',
                                td,
                                td)

                        td[range(len), range(len)] = -1000
                    # sd = student
                    sd = torch.nn.functional.normalize(sd, p=2, dim=-1)
                    sd = torch.einsum('cd, ed->ce',
                                        sd,
                                        sd)
                    sd[range(len), range(len)] = -1000
                    p_s = torch.nn.functional.log_softmax(sd/temperature_s, dim=-1)#be
                    p_t = torch.nn.functional.softmax(td/temperature_t, dim=-1)#to be
                    loss_div = torch.nn.functional.kl_div(p_s, p_t, reduction='sum') / (batchsize * opt.num_f )#MM_relation


                output_t = (output_t * bool_topk_pos_t.unsqueeze(-1))
                outputs = (outputs * bool_topk_pos_t.unsqueeze(-1))
                loss_mse = loss_func(output_t,outputs) * patch / opt.num_f#MM_hint


            
            if opt.distill == 'hinTop':
                criterion_kd = HinTop(num_k=opt.num_k)
                att_s = att_map_s[-1]
                att_t = att_map_t[-1]
                v_s = v_list_s[-1]
                v_t = v_list_t[-1]
                loss_kd = criterion_kd(att_s, att_t, v_s,v_t,t=opt.kd_T)
            elif opt.distill == 'VTop':
                criterion_kd = VTop(num_k=opt.num_k)
                att_s = att_map_s[-1]
                att_t = att_map_t[-1]
                v_s = v_list_s[-1]
                v_t = v_list_t[-1]
                loss_kd = criterion_kd(att_s, att_t, v_s,v_t,t=opt.kd_T)
            elif opt.distill == 'hybrid':
                criterion_kd1 = HinTop(num_k=opt.num_k)
                criterion_kd2= VTop(num_k=opt.num_k)
                att_s1 = att_map_s[-1]
                att_t1 = att_map_t[-1]
                v_s = v_list_s[-1]
                v_t = v_list_t[-1]
                att_s2 = att_map_s[-1].clone()
                att_t2 = att_map_t[-1].clone()

                loss_kd1 = criterion_kd1(att_s1, att_t1, v_s,v_t,t=opt.kd_T)                
                loss_kd2 = criterion_kd2(att_s2, att_t2, v_s,v_t,t=opt.kd_T)

                loss_kd = opt.beta_kd1 * loss_kd1 + opt.beta_kd2 * loss_kd2#MM_hidden

            elif opt.distill == 'onlymse':
                pass
            else:
                raise NotImplementedError(opt.distill)
        
        if opt.distill == 'onlymse' and opt.alpha != 0:
                loss_all = opt.gamma * loss_mse + opt.alpha * loss_div
        elif opt.distill == 'onlymse':
                loss_all = opt.gamma * loss_mse 
        elif opt.onlykd:
            loss_all = opt.beta * loss_kd
        elif opt.alpha != 0:
            loss_all = opt.gamma * loss_mse  + opt.beta * loss_kd + opt.alpha * loss_div 
        else:
            loss_all = opt.gamma * loss_mse  + opt.beta * loss_kd

        loss_value = loss_all.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()
        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(
            optimizer, 'is_second_order') and optimizer.is_second_order
        grad_norm = loss_scaler(loss_all,
                                optimizer,
                                clip_grad=max_norm,
                                parameters=model.parameters(),
                                create_graph=is_second_order)


        loss_scale_value = loss_scaler.state_dict()["scale"]

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        metric_logger.update(loss_scale=loss_scale_value)
        if opt.onlykd:
            pass
        else:
            metric_logger.update(loss_mse=loss_mse.item())
        if opt.distill == 'onlymse':
            pass
        elif opt.distill == 'hybrid':
            metric_logger.update(loss_kd=loss_kd.item())
            metric_logger.update(loss_kd1=loss_kd1.item())
            metric_logger.update(loss_kd2=loss_kd2.item())
        else:
            metric_logger.update(loss_kd=loss_kd.item())
        if opt.alpha != 0:
            metric_logger.update(loss_div=loss_div.item())
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)
        metric_logger.update(min_lr=min_lr)
        weight_decay_value = None
        for group in optimizer.param_groups:
            if group["weight_decay"] > 0:
                weight_decay_value = group["weight_decay"]
        metric_logger.update(weight_decay=weight_decay_value)
        metric_logger.update(grad_norm=grad_norm)

        if log_writer is not None:
            log_writer.update(loss=loss_value, head="loss")
            if opt.onlykd:
                pass
            else:
                log_writer.update(loss=loss_mse.item(), head="loss_mse")


            if opt.alpha != 0:
                log_writer.update(loss=loss_div.item(), head="loss_div")


            if opt.distill == 'onlymse':
                pass
            elif opt.distill == 'hybrid':
                log_writer.update(loss=loss_kd.item(), head="loss_kd")
                log_writer.update(loss=loss_kd1.item(), head="loss_kd1")
                log_writer.update(loss=loss_kd2.item(), head="loss_kd2")

            else:
                log_writer.update(loss=loss_kd.item(), head="loss_kd")
            
            log_writer.update(loss_scale=loss_scale_value, head="opt")
            log_writer.update(lr=max_lr, head="opt")
            log_writer.update(min_lr=min_lr, head="opt")
            log_writer.update(weight_decay=weight_decay_value, head="opt")
            log_writer.update(grad_norm=grad_norm, head="opt")
            log_writer.set_step()

        if lr_scheduler is not None:
            lr_scheduler.step_update(start_steps + step)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

