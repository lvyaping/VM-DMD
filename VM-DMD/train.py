# -*- coding: utf-8 -*-
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import timm
from datasets.dataset import NPY_datasets
from tensorboardX import SummaryWriter
from models.vmunet.vmunet import VMUNet
from distillation.distill_manager import RelationalDistillationManager
from tools.init_stage0_from_swin import (
    initialize_stage0_with_swin,
    initialize_stage3_with_swin,
    initialize_stages_with_swin,
)

from engine import *
import os
import sys

from utils import *
from configs.config_setting import setting_config

import warnings
warnings.filterwarnings("ignore")


def main(config):

    is_distributed = 'RANK' in os.environ and 'WORLD_SIZE' in os.environ
    
    if is_distributed:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
        
        dist.init_process_group(backend='nccl', init_method='env://')
        torch.cuda.set_device(local_rank)
        config.local_rank = local_rank
        config.distributed = True
        
        print(f"[Rank {rank}] Distributed training: rank={rank}, world_size={world_size}, local_rank={local_rank}")
    else:
        config.distributed = False
        config.local_rank = -1
        os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu_id
        local_rank = -1
    
    if not is_distributed or config.local_rank == 0:
        print(f'#----------Creating logger----------#')
    sys.path.append(config.work_dir + '/')
    log_dir = os.path.join(config.work_dir, 'log')
    checkpoint_dir = os.path.join(config.work_dir, 'checkpoints')
    resume_model = os.path.join(checkpoint_dir, 'latest.pth')
    outputs = os.path.join(config.work_dir, 'outputs')

    if not is_distributed or config.local_rank == 0:
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        if not os.path.exists(outputs):
            os.makedirs(outputs)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
    
    if is_distributed:
        dist.barrier()
    
    global logger
    logger = get_logger('train', log_dir, rank=config.local_rank if is_distributed else -1)
    global writer
    if not is_distributed or config.local_rank == 0:
        writer = SummaryWriter(config.work_dir + 'summary')
    else:
        writer = None
    
    if not is_distributed or config.local_rank == 0:
        log_config_info(config, logger)

    if not is_distributed or config.local_rank == 0:
        print('#----------GPU init----------#')
    gpu_ids = [int(id) for id in config.gpu_id.split(',')] if isinstance(config.gpu_id, str) else [int(config.gpu_id)]
    gpus_num = torch.cuda.device_count() if is_distributed else len(gpu_ids)
    
    set_seed(config.seed)
    if is_distributed:
        torch.cuda.manual_seed_all(config.seed)
    torch.cuda.empty_cache()

    if not is_distributed or config.local_rank == 0:
        print('#----------Preparing dataset----------#')
    train_dataset = NPY_datasets(config.data_path, config, train=True)
    train_sampler = DistributedSampler(train_dataset, shuffle=True) if is_distributed else None
    train_loader = DataLoader(train_dataset,
                                batch_size=config.batch_size // gpus_num if is_distributed else config.batch_size,
                                shuffle=(train_sampler is None),
                                sampler=train_sampler,
                                pin_memory=True,
                                num_workers=config.num_workers)
    
    val_dataset = NPY_datasets(config.data_path, config, train=False)
    val_sampler = DistributedSampler(val_dataset, shuffle=False) if is_distributed else None
    val_loader = DataLoader(val_dataset,
                                batch_size=1,
                                shuffle=False,
                                sampler=val_sampler,
                                pin_memory=True, 
                                num_workers=config.num_workers,
                                drop_last=True)

    if not is_distributed or config.local_rank == 0:
        print('#----------Preparing Model----------#')
    model_cfg = config.model_config
    if config.network == 'vmunet':
        model = VMUNet(
            num_classes=model_cfg['num_classes'],
            input_channels=model_cfg['input_channels'],
            depths=model_cfg['depths'],
            depths_decoder=model_cfg['depths_decoder'],
            drop_path_rate=model_cfg['drop_path_rate'],
            load_ckpt_path=model_cfg['load_ckpt_path'],
        )
        model.load_from()

        stages_to_init = getattr(config, 'init_stages_from_swin', None)
        if stages_to_init:
            ckpt_path = getattr(
                config,
                'init_stages_swin_ckpt',
                getattr(config, 'init_stage0_swin_ckpt', None)
            )
            use_projection = getattr(config, 'init_stages_use_projection', True)
            if not is_distributed or config.local_rank == 0:
                print(f'#----------Initializing stages {stages_to_init} from Swin----------#')
            initialize_stages_with_swin(
                model,
                ckpt_path,
                stages=stages_to_init,
                use_projection=use_projection
            )
            if not is_distributed or config.local_rank == 0:
                print('#----------Stage weight initialization finished----------#')
        else:
            if getattr(config, 'init_stage0_from_swin', False):
                if not is_distributed or config.local_rank == 0:
                    print('#----------Initializing Stage 0 weights from Swin----------#')
                initialize_stage0_with_swin(model, getattr(config, 'init_stage0_swin_ckpt', None))
                if not is_distributed or config.local_rank == 0:
                    print('#----------Stage 0 weight initialization finished----------#')
            
            if getattr(config, 'init_stage3_from_swin', False):
                if not is_distributed or config.local_rank == 0:
                    print('#----------Initializing Stage 3 (last stage) weights from Swin----------#')
                initialize_stage3_with_swin(model, getattr(config, 'init_stage3_swin_ckpt', None))
                if not is_distributed or config.local_rank == 0:
                    print('#----------Stage 3 weight initialization finished----------#')
        
    else: raise Exception('network in not right!')
    
    if is_distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).cuda()
        model = DDP(model, device_ids=[config.local_rank], output_device=config.local_rank, find_unused_parameters=False)

        distill_manager = RelationalDistillationManager(config.relational_kd)
    else:
        if gpus_num > 1:
            model = torch.nn.DataParallel(model.cuda(), device_ids=gpu_ids, output_device=gpu_ids[0])
        else:
            model = model.cuda()
        distill_manager = RelationalDistillationManager(config.relational_kd)

    if not is_distributed or config.local_rank == 0:
        cal_params_flops(model.module if hasattr(model, 'module') else model, 256, logger)

    if not is_distributed or config.local_rank == 0:
        print('#----------Preparing loss, opt, sch and amp----------#')
    criterion = config.criterion
    optimizer = get_optimizer(config, model)
    scheduler = get_scheduler(config, optimizer)

    if not is_distributed or config.local_rank == 0:
        print('#----------Set other params----------#')
    max_dice = 0.0 
    start_epoch = 1
    best_epoch = 1 
    
    early_stop_patience = 50 
    val_loss_history = [] 
    best_val_loss = float('inf')  
    best_val_loss_epoch = 1 
    patience_counter = 0

    if config.only_test_and_save_figs:
        checkpoint = torch.load(config.best_ckpt_path, map_location=torch.device('cpu'))
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint
        
        filtered_state_dict = {k: v for k, v in state_dict.items() 
                              if not (k.endswith('total_ops') or k.endswith('total_params'))}
        if len(state_dict) != len(filtered_state_dict):
            filtered_keys = len(state_dict) - len(filtered_state_dict)
            if not is_distributed or config.local_rank == 0:
                print(f'[Test] Filtered out {filtered_keys} keys (total_ops/total_params) from checkpoint')
        
        model_to_load = model.module if hasattr(model, 'module') else model
        model_to_load.load_state_dict(filtered_state_dict, strict=False)
        
        config.work_dir = config.img_save_path
        if not is_distributed or config.local_rank == 0:
            if not os.path.exists(config.work_dir + 'outputs/'):
                os.makedirs(config.work_dir + 'outputs/')
        if is_distributed:
            dist.barrier()
        
        loss = test_one_epoch(
                val_loader,
                model,
                criterion,
                logger,
                config,
            )
        return

    if os.path.exists(resume_model):

        if not is_distributed or config.local_rank == 0:
            print('#----------Resume Model and Other params----------#')
        checkpoint = torch.load(resume_model, map_location=torch.device('cpu'))
        model_to_load = model.module if hasattr(model, 'module') else model
        state_dict = checkpoint['model_state_dict']
        filtered_state_dict = {k: v for k, v in state_dict.items() 
                              if not (k.endswith('total_ops') or k.endswith('total_params'))}
        if len(state_dict) != len(filtered_state_dict):
            filtered_keys = len(state_dict) - len(filtered_state_dict)
            if not is_distributed or config.local_rank == 0:
                print(f'[Resume] Filtered out {filtered_keys} keys (total_ops/total_params) from checkpoint')
        model_to_load.load_state_dict(filtered_state_dict, strict=False)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        saved_epoch = checkpoint['epoch']
        start_epoch += saved_epoch
        if 'max_dice' in checkpoint:
            max_dice, best_epoch = checkpoint['max_dice'], checkpoint['best_epoch']
        elif 'min_loss' in checkpoint:
            max_dice = 0.0
            best_epoch = checkpoint.get('min_epoch', 1)
        else:
            max_dice = 0.0
            best_epoch = 1
        loss = checkpoint.get('loss', 0.0)
        
        val_loss_history = checkpoint.get('val_loss_history', [])
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        best_val_loss_epoch = checkpoint.get('best_val_loss_epoch', 1)
        patience_counter = checkpoint.get('patience_counter', 0)

        log_info = f'resuming model from {resume_model}. resume_epoch: {saved_epoch}, max_dice: {max_dice:.4f}, best_epoch: {best_epoch}, loss: {loss:.4f}'
        logger.info(log_info)

    step = 0
    if not is_distributed or config.local_rank == 0:
        print('#----------Training----------#')
    for epoch in range(start_epoch, config.epochs + 1):
        
        if is_distributed:
            train_sampler.set_epoch(epoch)

        step = train_one_epoch(
            train_loader,
            model,
            criterion,
            optimizer,
            scheduler,
            epoch,
            step,
            logger,
            config,
            writer,
            distill_manager=distill_manager,
            local_rank=config.local_rank if is_distributed else 0
        )

        dice_score, val_loss = val_one_epoch(
                val_loader,
                model,
                criterion,
                epoch,
                logger,
                config,
                local_rank=config.local_rank if is_distributed else 0
            )

        should_stop = False
        if not is_distributed or config.local_rank == 0:
            val_loss_history.append(val_loss)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_loss_epoch = epoch
                patience_counter = 0  
                if not is_distributed or config.local_rank == 0:
                    logger.info(f'[Early Stop] Validation loss improved: {best_val_loss:.4f} at epoch {epoch}, reset patience counter')
            else:
                patience_counter += 1
                if not is_distributed or config.local_rank == 0:
                    logger.info(f'[Early Stop] Validation loss did not improve. Patience: {patience_counter}/{early_stop_patience} (best: {best_val_loss:.4f} at epoch {best_val_loss_epoch})')

            if patience_counter >= early_stop_patience:
                should_stop = True
                stop_msg = f'[Early Stop] Training stopped at epoch {epoch}. Validation loss did not improve for {early_stop_patience} consecutive epochs.'
                print(stop_msg)
                logger.info(stop_msg)
                logger.info(f'[Early Stop] Best validation loss: {best_val_loss:.4f} at epoch {best_val_loss_epoch}')
        
        if is_distributed:
            stop_tensor = torch.tensor(1 if should_stop else 0, dtype=torch.int32).cuda()
            dist.broadcast(stop_tensor, src=0)
            should_stop = (stop_tensor.item() == 1)
        
        if should_stop:
            if not is_distributed or config.local_rank == 0:
                print(f'[Early Stop] Stopping training at epoch {epoch}')
                logger.info(f'[Early Stop] Training stopped early at epoch {epoch}')
            break

        if not is_distributed or config.local_rank == 0:
            if dice_score > max_dice:
                model_to_save = model.module if hasattr(model, 'module') else model
                try:
                    torch.save(model_to_save.state_dict(), os.path.join(checkpoint_dir, 'best.pth'))
                    max_dice = dice_score
                    best_epoch = epoch
                    print(f'[Best Model] Epoch {epoch}: Dice = {dice_score:.4f}')
                except (RuntimeError, OSError) as e:
                    print(f'[WARNING] Failed to save best model at epoch {epoch}: {e}')
                    print('[WARNING] This may be due to disk space issues. Training will continue but best model was not saved.')
                    logger.warning(f'Failed to save best model: {e}')

            model_to_save = model.module if hasattr(model, 'module') else model
            try:
                torch.save(
                    {
                        'epoch': epoch,
                        'max_dice': max_dice,
                        'best_epoch': best_epoch,
                        'loss': val_loss, 
                        'model_state_dict': model_to_save.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'val_loss_history': val_loss_history,
                        'best_val_loss': best_val_loss,
                        'best_val_loss_epoch': best_val_loss_epoch,
                        'patience_counter': patience_counter,
                    }, os.path.join(checkpoint_dir, 'latest.pth'))
            except (RuntimeError, OSError) as e:
                print(f'[WARNING] Failed to save latest checkpoint at epoch {epoch}: {e}')
                print('[WARNING] This may be due to disk space issues. Training will continue but checkpoint was not saved.')
                logger.warning(f'Failed to save latest checkpoint: {e}') 

    if (not is_distributed or config.local_rank == 0) and os.path.exists(os.path.join(checkpoint_dir, 'best.pth')):
        if not is_distributed or config.local_rank == 0:
            print('#----------Testing----------#')
        best_weight = torch.load(config.work_dir + 'checkpoints/best.pth', map_location=torch.device('cpu'))
        model_to_test = model.module if hasattr(model, 'module') else model
        model_to_test.load_state_dict(best_weight)
        loss = test_one_epoch(
                val_loader,
                model_to_test,
                criterion,
                logger,
                config,
            )
        os.rename(
            os.path.join(checkpoint_dir, 'best.pth'),
            os.path.join(checkpoint_dir, f'best-epoch{best_epoch}-dice{max_dice:.4f}.pth')
        )
    
    if is_distributed:
        dist.destroy_process_group()


if __name__ == '__main__':
    config = setting_config
    main(config)
