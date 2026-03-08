import argparse
import copy
import os
import os.path as osp
import time

import mmcv
import torch
from mmcv.cnn import get_model_complexity_info
from mmcv.runner import get_dist_info, init_dist
from mmcv.utils import Config, DictAction, get_git_hash

from mmseg import __version__
from mmseg.apis import set_random_seed, train_segmentor
from mmseg.datasets import build_dataset
from mmseg.models import build_segmentor
from mmseg.utils import collect_env, get_root_logger


def parse_args():
    parser = argparse.ArgumentParser(description='Train a segmentor')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--load-from', help='the checkpoint file to load weights from')
    parser.add_argument(
        '--resume-from', help='the checkpoint file to resume from')
    parser.add_argument(
        '--no-validate',
        action='store_true',
        help='whether not to evaluate the checkpoint during training')
    group_gpus = parser.add_mutually_exclusive_group()
    group_gpus.add_argument(
        '--gpus',
        type=int,
        help='number of gpus to use '
        '(only applicable to non-distributed training)')
    group_gpus.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        help='ids of gpus to use '
        '(only applicable to non-distributed training)')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--options', nargs='+', action=DictAction, help='custom options')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    if args.options is not None:
        cfg.merge_from_dict(args.options)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])
    if args.load_from is not None:
        cfg.load_from = args.load_from
    if args.resume_from is not None:
        cfg.resume_from = args.resume_from
    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids
    else:
        cfg.gpu_ids = range(1) if args.gpus is None else range(args.gpus)

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # create work_dir
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    # dump config (with error handling for yapf compatibility)
    try:
    cfg.dump(osp.join(cfg.work_dir, osp.basename(args.config)))
    except TypeError as e:
        # Handle yapf version incompatibility (yapf 0.43+ doesn't support verify parameter)
        if 'verify' in str(e):
            import shutil
            # Just copy the config file instead of dumping
            config_dest = osp.join(cfg.work_dir, osp.basename(args.config))
            shutil.copy2(args.config, config_dest)
            print(f'Warning: Config dump failed due to yapf incompatibility. Copied config file instead.')
        else:
            raise
    # init the logger before other steps
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)

    # init the meta dict to record some important information such as
    # environment info and seed, which will be logged
    meta = dict()
    # log env info
    env_info_dict = collect_env()
    env_info = '\n'.join([f'{k}: {v}' for k, v in env_info_dict.items()])
    dash_line = '-' * 60 + '\n'
    logger.info('Environment info:\n' + dash_line + env_info + '\n' +
                dash_line)
    meta['env_info'] = env_info

    # log some basic info
    logger.info(f'Distributed training: {distributed}')
    logger.info(f'Config:\n{cfg.pretty_text}')

    # set random seeds
    if args.seed is not None:
        logger.info(f'Set random seed to {args.seed}, deterministic: '
                    f'{args.deterministic}')
        set_random_seed(args.seed, deterministic=args.deterministic)
    cfg.seed = args.seed
    meta['seed'] = args.seed
    meta['exp_name'] = osp.basename(args.config)

    model = build_segmentor(
        cfg.model,
        train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg'))

    # Disable model structure printing to avoid cluttering the output
    # __repr__ is now overridden in EncoderDecoder class itself
    # logger.info(model)  # Already disabled

    # Print model complexity (Params and FLOPs) before training
    # Only print on rank 0 to avoid duplicate output
    rank, _ = get_dist_info()
    if rank == 0:
        try:
            # Get input shape from config (try crop_size first, then img_scale)
            input_shape = (3, 224, 224)  # default
            if hasattr(cfg.data, 'train') and hasattr(cfg.data.train, 'pipeline'):
                for transform in cfg.data.train.pipeline:
                    if isinstance(transform, dict):
                        if 'crop_size' in transform:
                            crop_size = transform['crop_size']
                            if isinstance(crop_size, (list, tuple)):
                                input_shape = (3, crop_size[0], crop_size[1])
                            elif isinstance(crop_size, int):
                                input_shape = (3, crop_size, crop_size)
                            break
                        elif 'img_scale' in transform:
                            img_scale = transform['img_scale']
                            if isinstance(img_scale, (list, tuple)):
                                if isinstance(img_scale[0], (list, tuple)):
                                    # img_scale can be [(h1, w1), (h2, w2)]
                                    img_scale = img_scale[0]
                                input_shape = (3, img_scale[0], img_scale[1])
                            break
            
            # Check if model has forward_dummy method (required for FLOPs calculation)
            # Note: We create a temporary copy of the model for FLOPs calculation
            # to avoid modifying the original model's forward method
            if hasattr(model, 'forward_dummy'):
                # Save original state
                original_training = model.training
                
                # Create a deep copy for FLOPs calculation to avoid modifying original model
                model_for_flops = copy.deepcopy(model)
                model_for_flops.eval()
                
                # Move model to GPU if available (required for SyncBatchNorm)
                if torch.cuda.is_available():
                    model_for_flops = model_for_flops.cuda()
                
                # Use forward_dummy for FLOPs calculation on the copy
                model_for_flops.forward = model_for_flops.forward_dummy
                
                try:
                    flops, params = get_model_complexity_info(model_for_flops, input_shape)
                finally:
                    # Clean up the temporary model copy
                    if torch.cuda.is_available():
                        model_for_flops = model_for_flops.cpu()
                    del model_for_flops
                    # Ensure original model is in training mode
                    model.train(original_training)
                
                # Parse FLOPs and Params from string format (e.g., "1429.68 GMac", "48.98 M")
                flops_str = str(flops)
                params_str = str(params)
                
                # Extract numeric values from FLOPs
                try:
                    flops_str_clean = flops_str.strip()
                    # Handle formats: "44.6 GFLOPs", "1429.68 GMac", "1429.68G", etc.
                    if 'GFLOPs' in flops_str_clean or 'GFLOP' in flops_str_clean:
                        # Format: "44.6 GFLOPs" or "44.6 GFLOP"
                        flops_val = float(flops_str_clean.replace('GFLOPs', '').replace('GFLOP', '').strip())
                    elif 'GMac' in flops_str_clean:
                        # Format: "1429.68 GMac" or "1429.68GMac"
                        flops_val = float(flops_str_clean.replace('GMac', '').strip())
                    elif 'MMac' in flops_str_clean:
                        # Format: "1429.68 MMac" -> convert to G
                        flops_val = float(flops_str_clean.replace('MMac', '').strip()) / 1000.0
                    elif 'MFLOPs' in flops_str_clean or 'MFLOP' in flops_str_clean:
                        # Format: "1429.68 MFLOPs" -> convert to G
                        flops_val = float(flops_str_clean.replace('MFLOPs', '').replace('MFLOP', '').strip()) / 1000.0
                    elif flops_str_clean.endswith('G'):
                        # Format: "1429.68G"
                        flops_val = float(flops_str_clean[:-1].strip())
                    elif flops_str_clean.endswith('M'):
                        # Format: "1429.68M" -> convert to G
                        flops_val = float(flops_str_clean[:-1].strip()) / 1000.0
                    else:
                        # Try to parse as number and assume it's in the base unit
                        # Split by space and take first part
                        parts = flops_str_clean.split()
                        if len(parts) > 0:
                            flops_val = float(parts[0]) / 1e9  # Assume base unit, convert to G
                        else:
                            flops_val = 0.0
                except Exception as e:
                    logger.warning(f'Failed to parse FLOPs string "{flops_str}": {e}')
                    print(f'Warning: Failed to parse FLOPs string "{flops_str}": {e}')
                    flops_val = 0.0
                
                # Extract numeric values from Params
                try:
                    params_str_clean = params_str.strip()
                    # Handle formats: "48.98 M", "48.98M", etc.
                    if params_str_clean.endswith('M'):
                        # Format: "48.98 M" or "48.98M"
                        params_val = float(params_str_clean[:-1].strip())
                    elif params_str_clean.endswith('K'):
                        # Format: "48.98 K" -> convert to M
                        params_val = float(params_str_clean[:-1].strip()) / 1000.0
                    else:
                        # Try to parse as number
                        parts = params_str_clean.split()
                        if len(parts) > 0:
                            params_val = float(parts[0]) / 1e6  # Assume base unit, convert to M
                        else:
                            params_val = 0.0
                except Exception as e:
                    logger.warning(f'Failed to parse Params string "{params_str}": {e}')
                    print(f'Warning: Failed to parse Params string "{params_str}": {e}')
                    params_val = 0.0
                
                # Print in formatted table
                logger.info('=' * 70)
                logger.info('Model Complexity:')
                logger.info('-' * 70)
                logger.info(f'Input shape: {input_shape}')
                logger.info(f'Params (M): {params_val:.2f}')
                logger.info(f'FLOPs (G): {flops_val:.2f}')
                logger.info('=' * 70)
                
                # Also print to console
                print('=' * 70)
                print('Model Complexity:')
                print('-' * 70)
                print(f'Input shape: {input_shape}')
                print(f'Params (M): {params_val:.2f}')
                print(f'FLOPs (G): {flops_val:.2f}')
                print('=' * 70)
            else:
                # If model doesn't support FLOPs calculation, just count parameters
                total_params = sum(p.numel() for p in model.parameters())
                trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                params_m = total_params / 1e6
                
                logger.info('=' * 70)
                logger.info('Model Parameters:')
                logger.info('-' * 70)
                logger.info(f'Total Params (M): {params_m:.2f}')
                logger.info(f'Trainable Params (M): {trainable_params / 1e6:.2f}')
                logger.info('=' * 70)
                
                print('=' * 70)
                print('Model Parameters:')
                print('-' * 70)
                print(f'Total Params (M): {params_m:.2f}')
                print(f'Trainable Params (M): {trainable_params / 1e6:.2f}')
                print('=' * 70)
        except Exception as e:
            logger.warning(f'Failed to calculate model complexity: {e}')
            print(f'Warning: Failed to calculate model complexity: {e}')

    datasets = [build_dataset(cfg.data.train)]
    if len(cfg.workflow) == 2:
        val_dataset = copy.deepcopy(cfg.data.val)
        val_dataset.pipeline = cfg.data.train.pipeline
        datasets.append(build_dataset(val_dataset))
    if cfg.checkpoint_config is not None:
        # save mmseg version, config file content and class names in
        # checkpoints as meta data
        cfg.checkpoint_config.meta = dict(
            mmseg_version=f'{__version__}+{get_git_hash()[:7]}',
            config=cfg.pretty_text,
            CLASSES=datasets[0].CLASSES,
            PALETTE=datasets[0].PALETTE)
    # add an attribute for visualization convenience
    model.CLASSES = datasets[0].CLASSES
    train_segmentor(
        model,
        datasets,
        cfg,
        distributed=distributed,
        validate=(not args.no_validate),
        timestamp=timestamp,
        meta=meta)


if __name__ == '__main__':
    main()
