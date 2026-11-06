import logging
import time
import torch
from utils.meter import AverageMeter
from utils.metrics import Evaluator
from utils.comm import get_rank, synchronize
from torch.utils.tensorboard import SummaryWriter
from prettytable import PrettyTable
import random


def do_train(start_epoch, args, model, train_loader, evaluator, optimizer,
             scheduler, checkpointer):

    log_period = args.log_period
    eval_period = args.eval_period
    device = "cuda"
    num_epoch = args.num_epoch
    arguments = {}
    arguments["num_epoch"] = num_epoch
    arguments["iteration"] = 0

    logger = logging.getLogger("IRRA.train")
    logger.info('start training')

    meters = {
        "loss": AverageMeter(),
        "sdm_loss": AverageMeter(),
        "itc_loss": AverageMeter(),
        "id_loss": AverageMeter(),
        "mlm_loss": AverageMeter(),
        "view_loss": AverageMeter(),
        "ortho_loss": AverageMeter(),
        "img_acc": AverageMeter(),
        "txt_acc": AverageMeter(),
        "mlm_acc": AverageMeter()
    }

    tb_writer = SummaryWriter(log_dir=args.output_dir)

    best_top1 = 0.0

    # train
    for epoch in range(start_epoch, num_epoch + 1):
        start_time = time.time()
        for meter in meters.values():
            meter.reset()
        model.train()

        for n_iter, batch in enumerate(train_loader):
            ##随机挑选一个caption
            length =  len(batch['caption_ids'])
            batch_list =[]
            mask_batch_list =[]
            
            for l in range(length):
                index_select = random.randrange(batch['caption_ids'][l].shape[0])
                caption_select = batch['caption_ids'][l][index_select]
                mlm_token_select = batch['mask_tokens_list'][l][index_select]
                batch_list.append(caption_select)
                mask_batch_list.append(mlm_token_select)
            batch['caption_ids'] = torch.stack(batch_list) 
            batch['mask_tokens_list'] = torch.stack(mask_batch_list)
            #
            batch = {k: v.to(device) for k, v in batch.items()}
            ret = model(batch)
            total_loss = sum([v for k, v in ret.items() if "loss" in k])

            # batch_size = batch['images'].shape[0]
            # 使用RGB图像计算batch_size（因为三个模态的batch_size相同）
            batch_size = batch['rgb_images'].shape[0]
            meters['loss'].update(total_loss.item(), batch_size)
            meters['sdm_loss'].update(ret.get('sdm_loss', 0), batch_size)
            meters['itc_loss'].update(ret.get('itc_loss', 0), batch_size)
            meters['id_loss'].update(ret.get('id_loss', 0), batch_size)
            meters['mlm_loss'].update(ret.get('mlm_loss', 0), batch_size)
            meters['view_loss'].update(ret.get('view_loss', 0), batch_size)
            meters['ortho_loss'].update(ret.get('ortho_loss', 0), batch_size)

            meters['img_acc'].update(ret.get('img_acc', 0), batch_size)
            meters['txt_acc'].update(ret.get('txt_acc', 0), batch_size)
            meters['mlm_acc'].update(ret.get('mlm_acc', 0), 1)

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            synchronize()

            if (n_iter + 1) % log_period == 0:
                info_str = f"Epoch[{epoch}] Iteration[{n_iter + 1}/{len(train_loader)}]"
                # log loss and acc info
                for k, v in meters.items():
                    if v.avg > 0:
                        info_str += f", {k}: {v.avg:.4f}"
                info_str += f", Base Lr: {scheduler.get_lr()[0]:.2e}"
                logger.info(info_str)
        
        tb_writer.add_scalar('lr', scheduler.get_lr()[0], epoch)
        tb_writer.add_scalar('temperature', ret['temperature'], epoch)
        for k, v in meters.items():
            if v.avg > 0:
                tb_writer.add_scalar(k, v.avg, epoch)


        scheduler.step()
        if get_rank() == 0:
            end_time = time.time()
            time_per_batch = (end_time - start_time) / (n_iter + 1)
            logger.info(
                "Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]"
                .format(epoch, time_per_batch,
                        train_loader.batch_size / time_per_batch))
        if epoch % eval_period == 0:
            if get_rank() == 0:
                logger.info("Validation Results - Epoch: {}".format(epoch))
                # if args.distributed:
                #     top1 = evaluator.eval(model.module.eval())
                # else:
                #     top1 = evaluator.eval(model.eval())

                if args.distributed:
                    # 使用三个evaluator分别评估三个模态
                    results = evaluator.eval_multimodal(model.module.eval())
                else:
                    results = evaluator.eval_multimodal(model.eval())

                torch.cuda.empty_cache()
                
                # 提取三个模态的结果
                # rgb_results = results['rgb']
                # nir_results = results['nir']
                # tir_results = results['tir']
                mm_results = results['mm']
                
                # 使用三个模态的平均R1值作为最佳模型判断标准
                # avg_top1 = (rgb_results['R1'] + nir_results['R1'] + tir_results['R1']) / 3.0
                avg_top1 = mm_results['R1'] 
                if best_top1 < avg_top1:
                    best_top1 = avg_top1
                    arguments["epoch"] = epoch
                    # arguments["rgb_results"] = rgb_results
                    # arguments["nir_results"] = nir_results
                    # arguments["tir_results"] = tir_results
                    arguments["mm_results"] = mm_results
                    checkpointer.save("best", **arguments)
                    
                    logger.info(f"New best model saved! Avg R1: {avg_top1:.3f}")
                    # logger.info(f"RGB: R1={rgb_results['R1']:.3f}, R5={rgb_results['R5']:.3f}, R10={rgb_results['R10']:.3f}, mAP={rgb_results['mAP']:.3f}, mINP={rgb_results['mINP']:.3f}")
                    # logger.info(f"NIR: R1={nir_results['R1']:.3f}, R5={nir_results['R5']:.3f}, R10={nir_results['R10']:.3f}, mAP={nir_results['mAP']:.3f}, mINP={nir_results['mINP']:.3f}")
                    # logger.info(f"TIR: R1={tir_results['R1']:.3f}, R5={tir_results['R5']:.3f}, R10={tir_results['R10']:.3f}, mAP={tir_results['mAP']:.3f}, mINP={tir_results['mINP']:.3f}")
                    logger.info(f"MM:  R1={mm_results['R1']:.3f}, R5={mm_results['R5']:.3f}, R10={mm_results['R10']:.3f}, mAP={mm_results['mAP']:.3f}, mINP={mm_results['mINP']:.3f}")
                
                
                # # 提取三个模态的结果
                # multimodal_results = results['multimodal']
                
                # # 使用三个模态的平均R1值作为最佳模型判断标准
                # avg_top1 = multimodal_results['R1'] 
                # if best_top1 < avg_top1:
                #     best_top1 = avg_top1
                #     arguments["epoch"] = epoch
                #     arguments["multimodal_results"] = multimodal_results
                #     checkpointer.save("best", **arguments)
                    
                #     logger.info(f"New best model saved! Avg R1: {avg_top1:.3f}")
                #     logger.info(f"Fusion: R1={multimodal_results['R1']:.3f}, R5={multimodal_results['R5']:.3f}, R10={multimodal_results['R10']:.3f}, mAP={multimodal_results['mAP']:.3f}, mINP={multimodal_results['mINP']:.3f}")
                    
    
    if get_rank() == 0:
        logger.info(f"Training completed! Best avg R1: {best_top1:.3f} at epoch {arguments['epoch']}")
        logger.info("Best model results:")
        # logger.info(f"RGB: R1={arguments['rgb_results']['R1']:.3f}, R5={arguments['rgb_results']['R5']:.3f}, R10={arguments['rgb_results']['R10']:.3f}, mAP={arguments['rgb_results']['mAP']:.3f}, mINP={arguments['rgb_results']['mINP']:.3f}")
        # logger.info(f"NIR: R1={arguments['nir_results']['R1']:.3f}, R5={arguments['nir_results']['R5']:.3f}, R10={arguments['nir_results']['R10']:.3f}, mAP={arguments['nir_results']['mAP']:.3f}, mINP={arguments['nir_results']['mINP']:.3f}")
        # logger.info(f"TIR: R1={arguments['tir_results']['R1']:.3f}, R5={arguments['tir_results']['R5']:.3f}, R10={arguments['tir_results']['R10']:.3f}, mAP={arguments['tir_results']['mAP']:.3f}, mINP={arguments['tir_results']['mINP']:.3f}")
        logger.info(f"MM:  R1={arguments['mm_results']['R1']:.3f}, R5={arguments['mm_results']['R5']:.3f}, R10={arguments['mm_results']['R10']:.3f}, mAP={arguments['mm_results']['mAP']:.3f}, mINP={arguments['mm_results']['mINP']:.3f}")

def do_inference(model, test_img_loader, test_txt_loader):

    logger = logging.getLogger("IRRA.test")
    logger.info("Enter inferencing")

    evaluator = Evaluator(test_img_loader, test_txt_loader)
    top1 = evaluator.eval(model.eval())

    return top1



