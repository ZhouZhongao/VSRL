from prettytable import PrettyTable
import torch
import numpy as np
import os
import torch.nn.functional as F
import logging


def rank(similarity, q_pids, g_pids, max_rank=10, get_mAP=True):
    if get_mAP:
        indices = torch.argsort(similarity, dim=1, descending=True)
    else:
        # acclerate sort with topk
        _, indices = torch.topk(
            similarity, k=max_rank, dim=1, largest=True, sorted=True
        )  # q * topk
    pred_labels = g_pids[indices.cpu()]  # q * k
    matches = pred_labels.eq(q_pids.view(-1, 1))  # q * k

    all_cmc = matches[:, :max_rank].cumsum(1) # cumulative sum
    all_cmc[all_cmc > 1] = 1
    all_cmc = all_cmc.float().mean(0) * 100
    # all_cmc = all_cmc[topk - 1]

    if not get_mAP:
        return all_cmc, indices

    num_rel = matches.sum(1)  # q
    tmp_cmc = matches.cumsum(1)  # q * k

    inp = [tmp_cmc[i][match_row.nonzero()[-1]] / (match_row.nonzero()[-1] + 1.) for i, match_row in enumerate(matches)]
    mINP = torch.cat(inp).mean() * 100

    tmp_cmc = [tmp_cmc[:, i] / (i + 1.0) for i in range(tmp_cmc.shape[1])]
    tmp_cmc = torch.stack(tmp_cmc, 1) * matches
    AP = tmp_cmc.sum(1) / num_rel  # q
    mAP = AP.mean() * 100

    return all_cmc, mAP, mINP, indices


class Evaluator():
    def __init__(self, img_loader, txt_loader):
        self.img_loader = img_loader # gallery
        self.txt_loader = txt_loader # query
        self.logger = logging.getLogger("IRRA.eval")
        
        # 添加属性来保存最后一次评估的详细结果
        self.last_r1 = 0.0
        self.last_r5 = 0.0
        self.last_r10 = 0.0
        self.last_map = 0.0
        self.last_minp = 0.0

    def _compute_embedding(self, model):
        model = model.eval()
        device = next(model.parameters()).device

        qids, gids, qfeats, gfeats = [], [], [], []
        # text
        for pid, caption in self.txt_loader:
            caption = caption.to(device)
            with torch.no_grad():
                text_feat = model.encode_text(caption)
                #用于检索concat
                # text_feat = torch.concat((text_feat,text_feat,text_feat),dim =-1)
            qids.append(pid.view(-1)) # flatten 
            qfeats.append(text_feat)
        qids = torch.cat(qids, 0)
        qfeats = torch.cat(qfeats, 0)

        # image
        for pid, img in self.img_loader:
            if not isinstance(img, list):
                img = img.to(device)
                with torch.no_grad():
                    img_feat = model.encode_image(img)
                    img_feat = torch.concat((img_feat,img_feat,img_feat),dim =-1)
            else:
                rgb_img,nir_img,tir_img = img
                rgb_img = rgb_img.to(device)
                nir_img = nir_img.to(device)
                tir_img = tir_img.to(device)
                with torch.no_grad():
                    rgb_img_feat = model.encode_image(rgb_img)
                    nir_img_feat = model.encode_image(nir_img)
                    tir_img_feat = model.encode_image(tir_img)
                    img_feat = (rgb_img_feat + nir_img_feat + tir_img_feat)/3
                    #用于检索concat
                    # img_feat = torch.concat((rgb_img_feat,nir_img_feat,tir_img_feat),dim =-1)
            gids.append(pid.view(-1)) # flatten 
            gfeats.append(img_feat)
        gids = torch.cat(gids, 0)
        gfeats = torch.cat(gfeats, 0)

        return qfeats, gfeats, qids, gids
    
    def eval(self, model, i2t_metric=False):

        qfeats, gfeats, qids, gids = self._compute_embedding(model)

        qfeats = F.normalize(qfeats, p=2, dim=1) # text features
        gfeats = F.normalize(gfeats, p=2, dim=1) # image features

        similarity = qfeats @ gfeats.t()

        t2i_cmc, t2i_mAP, t2i_mINP, _ = rank(similarity=similarity, q_pids=qids, g_pids=gids, max_rank=10, get_mAP=True)
        t2i_cmc, t2i_mAP, t2i_mINP = t2i_cmc.numpy(), t2i_mAP.numpy(), t2i_mINP.numpy()
        
        # 保存详细结果
        self.last_r1 = t2i_cmc[0]
        self.last_r5 = t2i_cmc[4]
        self.last_r10 = t2i_cmc[9]
        self.last_map = t2i_mAP
        self.last_minp = t2i_mINP
        
        table = PrettyTable(["task", "R1", "R5", "R10", "mAP", "mINP"])
        table.add_row(['t2i', t2i_cmc[0], t2i_cmc[4], t2i_cmc[9], t2i_mAP, t2i_mINP])

        if i2t_metric:
            i2t_cmc, i2t_mAP, i2t_mINP, _ = rank(similarity=similarity.t(), q_pids=gids, g_pids=qids, max_rank=10, get_mAP=True)
            i2t_cmc, i2t_mAP, i2t_mINP = i2t_cmc.numpy(), i2t_mAP.numpy(), i2t_mINP.numpy()
            table.add_row(['i2t', i2t_cmc[0], i2t_cmc[4], i2t_cmc[9], i2t_mAP, i2t_mINP])
        
        table.custom_format["R1"] = lambda f, v: f"{v:.3f}"
        table.custom_format["R5"] = lambda f, v: f"{v:.3f}"
        table.custom_format["R10"] = lambda f, v: f"{v:.3f}"
        table.custom_format["mAP"] = lambda f, v: f"{v:.3f}"
        table.custom_format["mINP"] = lambda f, v: f"{v:.3f}"
        self.logger.info('\n' + str(table))
        
        return t2i_cmc[0]


class MultimodalEvaluator():
    """
    多模态评估器，分别评估text到RGB、NIR、TIR三个模态的检索性能
    """
    def __init__(self, rgb_loader, nir_loader, tir_loader, mm_loader,txt_loader):
        # self.rgb_loader = rgb_loader
        # self.nir_loader = nir_loader
        # self.tir_loader = tir_loader
        self.mm_loader = mm_loader
        self.txt_loader = txt_loader
        self.logger = logging.getLogger("IRRA.multimodal_eval")

    def eval_multimodal(self, model):
        """
        分别评估三个模态的性能
        
        Returns:
            dict: 包含三个模态的评估结果
        """
        self.logger.info("Starting multimodal evaluation...")
        
        # 创建三个独立的评估器
        # evaluator_rgb = Evaluator(self.rgb_loader, self.txt_loader)
        # evaluator_nir = Evaluator(self.nir_loader, self.txt_loader)
        # evaluator_tir = Evaluator(self.tir_loader, self.txt_loader)
        evaluator_mm = Evaluator(self.mm_loader, self.txt_loader)
        
        # 分别评估三个模态
        logger = logging.getLogger("IRRA.eval")
        
        # 评估RGB模态
        # logger.info("Evaluating RGB modality...")
        # top1_rgb = evaluator_rgb.eval(model.eval())
        
        # # 评估NIR模态
        # logger.info("Evaluating NIR modality...")
        # top1_nir = evaluator_nir.eval(model.eval())
        
        # # 评估TIR模态
        # logger.info("Evaluating TIR modality...")
        # top1_tir = evaluator_tir.eval(model.eval())
        
        # 评估多模态模态
        logger.info("Evaluating Multimodal modality...")
        top1_mm = evaluator_mm.eval(model.eval())
        # 收集所有结果
        results = {
            # 'rgb': {
            #     'R1': top1_rgb,
            #     'R5': evaluator_rgb.last_r5,
            #     'R10': evaluator_rgb.last_r10,
            #     'mAP': evaluator_rgb.last_map,
            #     'mINP': evaluator_rgb.last_minp
            # },
            # 'nir': {
            #     'R1': top1_nir,
            #     'R5': evaluator_nir.last_r5,
            #     'R10': evaluator_nir.last_r10,
            #     'mAP': evaluator_nir.last_map,
            #     'mINP': evaluator_nir.last_minp
            # },
            # 'tir': {
            #     'R1': top1_tir,
            #     'R5': evaluator_tir.last_r5,
            #     'R10': evaluator_tir.last_r10,
            #     'mAP': evaluator_tir.last_map,
            #     'mINP': evaluator_tir.last_minp
            # },
            'mm': {
                'R1': top1_mm,
                'R5': evaluator_mm.last_r5,
                'R10': evaluator_mm.last_r10,
                'mAP': evaluator_mm.last_map,
                'mINP': evaluator_mm.last_minp
            }
        }
        
        # 打印汇总结果
        self.logger.info("Multimodal Evaluation Results:")
        # self.logger.info(f"RGB:   R1={results['rgb']['R1']:.3f}, R5={results['rgb']['R5']:.3f}, R10={results['rgb']['R10']:.3f}, mAP={results['rgb']['mAP']:.3f}, mINP={results['rgb']['mINP']:.3f}")
        # self.logger.info(f"NIR:   R1={results['nir']['R1']:.3f}, R5={results['nir']['R5']:.3f}, R10={results['nir']['R10']:.3f}, mAP={results['nir']['mAP']:.3f}, mINP={results['nir']['mINP']:.3f}")
        # self.logger.info(f"TIR:   R1={results['tir']['R1']:.3f}, R5={results['tir']['R5']:.3f}, R10={results['tir']['R10']:.3f}, mAP={results['tir']['mAP']:.3f}, mINP={results['tir']['mINP']:.3f}")
        # avg_r1 = (results['rgb']['R1'] + results['nir']['R1'] + results['tir']['R1']) / 3.0
        # self.logger.info(f"Average_single R1: {avg_r1:.3f}")
        self.logger.info(f"MM:    R1={results['mm']['R1']:.3f}, R5={results['mm']['R5']:.3f}, R10={results['mm']['R10']:.3f}, mAP={results['mm']['mAP']:.3f}, mINP={results['mm']['mINP']:.3f}")
        
        return results

class MultimodalEvaluator_Fusion():
    """
    多模态评估器，分别评估text到RGB、NIR、TIR三个模态的检索性能
    """
    def __init__(self, val_multimodal_loader, txt_loader):
        self.val_multimodal_loader = val_multimodal_loader
        self.txt_loader = txt_loader
        self.logger = logging.getLogger("IRRA.multimodal_eval")

    def eval_multimodal(self, model):
        """
        分别评估三个模态的性能
        
        Returns:
            dict: 包含三个模态的评估结果
        """
        self.logger.info("Starting multimodal evaluation...")
        
        # 创建三个独立的评估器
        evaluator_multimodal = Evaluator(self.val_multimodal_loader, self.txt_loader)
        
        # 分别评估三个模态
        logger = logging.getLogger("IRRA.eval")
        
        # 评估三模态融合
        logger.info("Evaluating fusion modality...")
        top1_multimodal = evaluator_multimodal.eval(model.eval())
        
        
        # 收集所有结果
        results = {
            'multimodal': {
                'R1': top1_multimodal,
                'R5': evaluator_multimodal.last_r5,
                'R10': evaluator_multimodal.last_r10,
                'mAP': evaluator_multimodal.last_map,
                'mINP': evaluator_multimodal.last_minp
            }
        }
        
        # 打印汇总结果
        self.logger.info("Multimodal Evaluation Results:")
        self.logger.info(f"Multimodal:   R1={results['multimodal']['R1']:.3f}, R5={results['multimodal']['R5']:.3f}, R10={results['multimodal']['R10']:.3f}, mAP={results['multimodal']['mAP']:.3f}, mINP={results['multimodal']['mINP']:.3f}")
        
        
        return results