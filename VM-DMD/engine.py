import numpy as np
from tqdm import tqdm
import torch
import torch.distributed as dist
from torch.cuda.amp import autocast as autocast
from sklearn.metrics import confusion_matrix
from utils import save_imgs


def get_adaptive_loss_weights(epoch):

    if epoch <= 25:
        alpha = 1.0  
        beta = 0  
    elif epoch <= 100:
        alpha = 0.95
        beta = 0.05
    elif epoch <= 200:
        alpha = 0.8
        beta = 0.2
    else:
        alpha = 0.7
        beta = 0.3
    
    return alpha, beta


def train_one_epoch(train_loader,
                    model,
                    criterion, 
                    optimizer, 
                    scheduler,
                    epoch, 
                    step,
                    logger, 
                    config,
                    writer,
                    distill_manager=None,
                    local_rank=0):
    '''
    train model for one epoch
    '''
    model.train() 
 
    loss_list = []
    rel_loss_list = []

    for iter, data in enumerate(train_loader):
        step += iter
        optimizer.zero_grad()
        images, targets = data
        images, targets = images.cuda(non_blocking=True).float(), targets.cuda(non_blocking=True).float()

        model_to_use = model.module if hasattr(model, 'module') else model
        
        if distill_manager is not None and distill_manager.enabled:
            out, student_feats, dir_feats = model_to_use(
                images,
                return_rel_features=True,
                return_directions=True,
            )
            rel_loss = distill_manager.compute_loss(
                images,
                student_feats,
                model_to_use,
                dir_feats=dir_feats,
            )
        else:
            out = model_to_use(images)
            rel_loss = None

        task_loss = criterion(out, targets)
        
        has_distill = rel_loss is not None
        if has_distill:
            alpha, beta = get_adaptive_loss_weights(epoch)
            loss = alpha * task_loss + beta * rel_loss
        else:
            loss = task_loss
            alpha, beta = 1.0, 0.0
        
        loss_value = loss.item()
        task_loss_value = task_loss.item()
        loss_list.append(loss_value)
        if has_distill:
            rel_loss_value = rel_loss.item()
            rel_loss_list.append(rel_loss_value)
        else:
            rel_loss_value = None

        loss.backward()
        optimizer.step()
        
        del out

        if distill_manager is not None and distill_manager.enabled:
            del student_feats
            del dir_feats
        
        del loss, task_loss
        if has_distill:
            del rel_loss

        del images, targets
        
        
        now_lr = optimizer.state_dict()['param_groups'][0]['lr']

        if writer is not None:
            writer.add_scalar('loss', loss_value, global_step=step)
            writer.add_scalar('loss_task', task_loss_value, global_step=step)
            if has_distill and rel_loss_value is not None:
                writer.add_scalar('loss_rel', rel_loss_value, global_step=step)
                writer.add_scalar('loss_weight_alpha', alpha, global_step=step)
                writer.add_scalar('loss_weight_beta', beta, global_step=step)

        if iter % config.print_interval == 0:
            rel_info = f', rel_kd: {np.mean(rel_loss_list):.4f}' if rel_loss_list else ''
            weight_info = f', alpha: {alpha:.2f}, beta: {beta:.2f}' if has_distill else ''
            log_info = f'train: epoch {epoch}, iter:{iter}, loss: {np.mean(loss_list):.4f}, task_loss: {task_loss_value:.4f}{rel_info}{weight_info}, lr: {now_lr}'

            is_distributed = dist.is_initialized() if dist.is_available() else False
            if not is_distributed or local_rank == 0:
                print(log_info)
                logger.info(log_info)
    dir_weight_stats = []
    if distill_manager is not None and distill_manager.enabled:
        dir_weight_stats = distill_manager.pop_dir_weight_stats()
        if writer is not None:
            for stage_idx, avg_weights in dir_weight_stats:
                if not avg_weights:
                    continue
                for dir_idx, value in enumerate(avg_weights):
                    writer.add_scalar(
                        f'dir_weight/stage{stage_idx}/dir{dir_idx}',
                        value,
                        global_step=epoch
                    )

        formatted = []
        for stage_idx, avg_weights in dir_weight_stats:
            if not avg_weights:
                continue
            formatted.append(
                f'Stage{stage_idx}: [{", ".join(f"{w:.4f}" for w in avg_weights)}]'
            )
        if formatted:
            log_msg = f'DirWeights epoch {epoch} -> ' + ' | '.join(formatted)
            if logger is not None:
                logger.info(log_msg)
            is_distributed = dist.is_initialized() if dist.is_available() else False
            if not is_distributed or local_rank == 0:
                print(log_msg)

    scheduler.step() 
    return step


def val_one_epoch(test_loader,
                    model,
                    criterion, 
                    epoch, 
                    logger,
                    config,
                    local_rank=0):
    # switch to evaluate mode
    model.eval()
    
    is_distributed = dist.is_initialized() if dist.is_available() else False
    
    preds = []
    gts = []
    loss_list = []
    total_loss = 0.0
    num_samples = 0
    
    with torch.no_grad():
        for data in tqdm(test_loader, disable=(is_distributed and local_rank != 0)):
            img, msk = data
            img, msk = img.cuda(non_blocking=True).float(), msk.cuda(non_blocking=True).float()

            out = model(img)
            loss = criterion(out, msk)

            batch_size = img.size(0)
            total_loss += loss.item() * batch_size
            num_samples += batch_size
            
            loss_list.append(loss.item())
            
            gts.append(msk.squeeze(1).cpu().detach().numpy())
            if type(out) is tuple:
                out = out[0]
            out = out.squeeze(1).cpu().detach().numpy()
            preds.append(out)

    if is_distributed:

        loss_tensor = torch.tensor([total_loss, num_samples], dtype=torch.float32).cuda()
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        total_loss = loss_tensor[0].item()
        num_samples = int(loss_tensor[1].item())
        avg_loss = total_loss / num_samples if num_samples > 0 else 0.0

        preds_list = [None] * dist.get_world_size()
        gts_list = [None] * dist.get_world_size()
        
        preds_flat = np.array(preds).reshape(-1)
        gts_flat = np.array(gts).reshape(-1)

        preds_length = len(preds_flat)
        gts_length = len(gts_flat)
        
        length_tensor = torch.tensor([preds_length, gts_length], dtype=torch.int32).cuda()
        gathered_lengths = [torch.zeros_like(length_tensor) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_lengths, length_tensor)
        
        max_preds_len = max([l[0].item() for l in gathered_lengths])
        max_gts_len = max([l[1].item() for l in gathered_lengths])
        
        if len(preds_flat) < max_preds_len:
            preds_padded = np.pad(preds_flat, (0, max_preds_len - len(preds_flat)), mode='constant')
        else:
            preds_padded = preds_flat
        
        if len(gts_flat) < max_gts_len:
            gts_padded = np.pad(gts_flat, (0, max_gts_len - len(gts_flat)), mode='constant')
        else:
            gts_padded = gts_flat
        
        preds_tensor = torch.from_numpy(preds_padded).float().cuda()
        gts_tensor = torch.from_numpy(gts_padded).float().cuda()
        
        gathered_preds = [torch.zeros_like(preds_tensor) for _ in range(dist.get_world_size())]
        gathered_gts = [torch.zeros_like(gts_tensor) for _ in range(dist.get_world_size())]
        
        dist.all_gather(gathered_preds, preds_tensor)
        dist.all_gather(gathered_gts, gts_tensor)
        
        if local_rank == 0:
            all_preds_list = []
            all_gts_list = []
            
            for rank_idx in range(dist.get_world_size()):
                pred_len = gathered_lengths[rank_idx][0].item()
                gt_len = gathered_lengths[rank_idx][1].item()
                
                all_preds_list.append(gathered_preds[rank_idx][:pred_len].cpu().numpy())
                all_gts_list.append(gathered_gts[rank_idx][:gt_len].cpu().numpy())
            
            preds_np = np.concatenate(all_preds_list, axis=0)
            gts_np = np.concatenate(all_gts_list, axis=0)
        else:
            preds_np = None
            gts_np = None
        
        dist.barrier()
    else:
        avg_loss = total_loss / num_samples if num_samples > 0 else np.mean(loss_list)
        preds_np = np.array(preds).reshape(-1)
        gts_np = np.array(gts).reshape(-1)

    dice_score = 0.0  
    if not is_distributed or local_rank == 0:
        if preds_np is not None and gts_np is not None:
            y_pre = np.where(preds_np>=config.threshold, 1, 0)
            y_true = np.where(gts_np>=0.5, 1, 0)

            confusion = confusion_matrix(y_true, y_pre)
            TN, FP, FN, TP = confusion[0,0], confusion[0,1], confusion[1,0], confusion[1,1] 

            accuracy = float(TN + TP) / float(np.sum(confusion)) if float(np.sum(confusion)) != 0 else 0
            sensitivity = float(TP) / float(TP + FN) if float(TP + FN) != 0 else 0
            specificity = float(TN) / float(TN + FP) if float(TN + FP) != 0 else 0
            precision = float(TP) / float(TP + FP) if float(TP + FP) != 0 else 0
            f1_or_dsc = float(2 * TP) / float(2 * TP + FP + FN) if float(2 * TP + FP + FN) != 0 else 0
            miou = float(TP) / float(TP + FP + FN) if float(TP + FP + FN) != 0 else 0
            dice_score = f1_or_dsc

            print('=' * 70)
            print(f'Validation Metrics (Epoch {epoch}):')
            print('-' * 70)
            header = f"{'Miou':<10} {'Dsc':<10} {'Acc':<10} {'Spe':<10} {'Sen':<10} {'Pre':<10}"
            values = f"{miou:<10.4f} {f1_or_dsc:<10.4f} {accuracy:<10.4f} {specificity:<10.4f} {sensitivity:<10.4f} {precision:<10.4f}"
            print(header)
            print(values)
            print('=' * 70)
            
            log_info = f'val epoch: {epoch}, loss: {avg_loss:.4f}, miou: {miou:.4f}, dsc: {f1_or_dsc:.4f}, accuracy: {accuracy:.4f}, \
                    specificity: {specificity:.4f}, sensitivity: {sensitivity:.4f}, precision: {precision:.4f}, confusion_matrix: {confusion}'
            logger.info(log_info)
        else:
            log_info = f'val epoch: {epoch}, loss: {avg_loss:.4f}'
            print(log_info)
            logger.info(log_info)
    
    if is_distributed:
        dice_tensor = torch.tensor(dice_score, dtype=torch.float32).cuda()
        loss_tensor = torch.tensor(avg_loss, dtype=torch.float32).cuda()
        dist.broadcast(dice_tensor, src=0)
        dist.broadcast(loss_tensor, src=0)
        dice_score = dice_tensor.item()
        avg_loss = loss_tensor.item()
    
    return dice_score, avg_loss


def test_one_epoch(test_loader,
                    model,
                    criterion,
                    logger,
                    config,
                    test_data_name=None):
    # switch to evaluate mode
    model.eval()
    preds = []
    gts = []
    loss_list = []
    with torch.no_grad():
        for i, data in enumerate(tqdm(test_loader)):
            img, msk = data
            img, msk = img.cuda(non_blocking=True).float(), msk.cuda(non_blocking=True).float()

            out = model(img)
            loss = criterion(out, msk)

            loss_list.append(loss.item())
            msk = msk.squeeze(1).cpu().detach().numpy()
            gts.append(msk)
            if type(out) is tuple:
                out = out[0]
            out = out.squeeze(1).cpu().detach().numpy()
            preds.append(out) 
            if i % config.save_interval == 0:
                save_imgs(img, msk, out, i, config.work_dir + 'outputs/', config.datasets, config.threshold, test_data_name=test_data_name)

        preds = np.array(preds).reshape(-1)
        gts = np.array(gts).reshape(-1)

        y_pre = np.where(preds>=config.threshold, 1, 0)
        y_true = np.where(gts>=0.5, 1, 0)

        confusion = confusion_matrix(y_true, y_pre)
        TN, FP, FN, TP = confusion[0,0], confusion[0,1], confusion[1,0], confusion[1,1] 

        accuracy = float(TN + TP) / float(np.sum(confusion)) if float(np.sum(confusion)) != 0 else 0
        sensitivity = float(TP) / float(TP + FN) if float(TP + FN) != 0 else 0
        specificity = float(TN) / float(TN + FP) if float(TN + FP) != 0 else 0
        precision = float(TP) / float(TP + FP) if float(TP + FP) != 0 else 0
        f1_or_dsc = float(2 * TP) / float(2 * TP + FP + FN) if float(2 * TP + FP + FN) != 0 else 0
        miou = float(TP) / float(TP + FP + FN) if float(TP + FP + FN) != 0 else 0

        print('=' * 70)
        if test_data_name is not None:
            print(f'Test Metrics ({test_data_name}):')
        else:
            print('Test Metrics:')
        print('-' * 70)
        header = f"{'Miou':<10} {'Dsc':<10} {'Acc':<10} {'Spe':<10} {'Sen':<10} {'Pre':<10}"
        values = f"{miou:<10.4f} {f1_or_dsc:<10.4f} {accuracy:<10.4f} {specificity:<10.4f} {sensitivity:<10.4f} {precision:<10.4f}"
        print(header)
        print(values)
        print('=' * 70)
        
        if test_data_name is not None:
            log_info = f'test_datasets_name: {test_data_name}'
            logger.info(log_info)
        log_info = f'test of best model, loss: {np.mean(loss_list):.4f}, miou: {miou:.4f}, dsc: {f1_or_dsc:.4f}, accuracy: {accuracy:.4f}, \
                specificity: {specificity:.4f}, sensitivity: {sensitivity:.4f}, precision: {precision:.4f}, confusion_matrix: {confusion}'
        logger.info(log_info)

    return np.mean(loss_list)
