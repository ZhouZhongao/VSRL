from model import objectives
from .clip_model import Transformer, QuickGELU, LayerNorm, build_CLIP_from_openai_pretrained, convert_weights
import numpy as np
import torch
import torch.nn as nn
from collections import OrderedDict
from .SFTS import SFTS
from .Frequency import Frequency_based_Token_Selection
from .vit_pytorch import BlockMask
import torch.nn.functional as F


class IRRA(nn.Module):
    def __init__(self, args, num_classes=11003):
        super().__init__()
        self.args = args
        self.num_classes = num_classes
        self._set_task()

        self.base_model, base_cfg = build_CLIP_from_openai_pretrained(args.pretrain_choice, args.img_size, args.stride_size)
        self.embed_dim = base_cfg['embed_dim']

        self.logit_scale = torch.ones([]) * (1 / args.temperature) 
        
        self.view_gate = nn.Linear(self.embed_dim, 2)
        nn.init.normal_(self.view_gate.weight.data, std=0.001)
        nn.init.constant_(self.view_gate.bias.data, val=0.0)
        
        self.modality_gate = nn.Linear(self.embed_dim, 3)
        nn.init.normal_(self.modality_gate.weight.data, std=0.001)
        nn.init.constant_(self.modality_gate.bias.data, val=0.0)
        self.inverseNet = Mlp(in_features=512, hidden_features=512 * 4, out_features=512, drop=0.1)
        if 'id' in args.loss_names:
            self.classifier = nn.Linear(self.embed_dim, self.num_classes)
            nn.init.normal_(self.classifier.weight.data, std=0.001)
            nn.init.constant_(self.classifier.bias.data, val=0.0)

        # if 'mlm' in args.loss_names:
        if True:
            self.cross_attn = nn.MultiheadAttention(self.embed_dim,
                                                    self.embed_dim // 64,
                                                    batch_first=True)
            self.cross_modal_transformer = Transformer(width=self.embed_dim,
                                                       layers=args.cmt_depth,
                                                       heads=self.embed_dim //
                                                       64)
            scale = self.cross_modal_transformer.width**-0.5
            
            self.ln_pre_t = LayerNorm(self.embed_dim)
            self.ln_pre_i = LayerNorm(self.embed_dim)
            self.ln_post = LayerNorm(self.embed_dim)

            proj_std = scale * ((2 * self.cross_modal_transformer.layers)**-0.5)
            attn_std = scale
            fc_std = (2 * self.cross_modal_transformer.width)**-0.5
            for block in self.cross_modal_transformer.resblocks:
                nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
                nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
                nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
                nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

            # init cross attn
            nn.init.normal_(self.cross_attn.in_proj_weight, std=attn_std)
            nn.init.normal_(self.cross_attn.out_proj.weight, std=proj_std)

            self.mlm_head = nn.Sequential(
                OrderedDict([('dense', nn.Linear(self.embed_dim, self.embed_dim)),
                            ('gelu', QuickGELU()),
                            ('ln', LayerNorm(self.embed_dim)),
                            ('fc', nn.Linear(self.embed_dim, self.num_classes))])) #nn.Linear(self.embed_dim, args.vocab_size)
            
            # init mlm head
            nn.init.normal_(self.mlm_head.dense.weight, std=fc_std)
            nn.init.normal_(self.mlm_head.fc.weight, std=proj_std)
        if 'grpo' in args.loss_names:
            self.reference_model = None
            self.reward_func = None
    def _set_task(self):
        loss_names = self.args.loss_names
        self.current_task = [l.strip() for l in loss_names.split('+')]
        print(f'Training Model with {self.current_task} tasks')
    
    
    def cross_former(self, q, k, v):
        x = self.cross_attn(
                self.ln_pre_t(q),
                self.ln_pre_i(k),
                self.ln_pre_i(v),
                need_weights=False)[0]
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.cross_modal_transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x)
        return x

    def encode_image(self, image):
        x = self.base_model.encode_image(image)
        if isinstance(x,tuple):
            return x[0][:, 0, :].float()
        else:
            return x[:, 0, :].float()
        # return x.float() # for CLIP ResNet visual model
    def encode_text(self, text):
        x = self.base_model.encode_text(text)
        return x[torch.arange(x.shape[0]), text.argmax(dim=-1)].float()

    def forward(self, batch):
        ret = dict()
        # images = batch['images']
        # caption_ids = batch['caption_ids']

        # 获取三个模态的图像
        rgb_images = batch['rgb_images']
        nir_images = batch['nir_images']
        tir_images = batch['tir_images']
        caption_ids = batch['caption_ids']
        pids = batch['pids']
        mask_tokens = batch['mask_tokens_list']
        # image_feats, text_feats = self.base_model(images, caption_ids)
        # i_feats = image_feats[:, 0, :].float()
        # 编码三个模态的图像
        rgb_feats, text_feats, _  = self.base_model(rgb_images, caption_ids)
        nir_feats, _, _  = self.base_model(nir_images, caption_ids)
        tir_feats, _, _  = self.base_model(tir_images, caption_ids)
      
        t_feats = text_feats[torch.arange(text_feats.shape[0]), caption_ids.argmax(dim=-1)].float()
        
        # 融合三个模态的图像特征（简单平均）
        rgb_i_feats = rgb_feats[:, 0, :].float()
        nir_i_feats = nir_feats[:, 0, :].float()
        tir_i_feats = tir_feats[:, 0, :].float()

        i_feats = (rgb_i_feats + nir_i_feats + tir_i_feats) / 3.0
        

        logit_scale = self.logit_scale
        ret.update({'temperature': 1 / logit_scale})
        
        rgb_view_feats = rgb_feats[:, -1, :]
        nir_view_feats = nir_feats[:, -1, :]
        tir_view_feats = tir_feats[:, -1, :]
        
        ### view_loss + ortho_loss
        rgb_view_logits = self.view_gate(rgb_view_feats).float()
        nir_view_logits = self.view_gate(nir_view_feats).float()
        tir_view_logits = self.view_gate(tir_view_feats).float()
        ce = nn.CrossEntropyLoss(reduction="mean")
        view_loss = ce(rgb_view_logits, batch['view_ids']) + ce(nir_view_logits, batch['view_ids']) + ce(tir_view_logits, batch['view_ids'])
        ret.update({'view_loss': view_loss  * self.args.view_loss_weight})
        
        ortho_loss = 3.0 + F.cosine_similarity(rgb_i_feats,  rgb_view_feats, dim=-1).abs().mean() + \
        F.cosine_similarity(nir_i_feats, nir_view_feats, dim=-1).abs().mean() + \
        F.cosine_similarity(tir_i_feats, tir_view_feats, dim=-1).abs().mean() - \
        F.cosine_similarity(rgb_view_feats, nir_view_feats + tir_view_feats, dim=-1).mean() - \
        F.cosine_similarity(nir_view_feats, rgb_view_feats + tir_view_feats, dim=-1).mean() - \
        F.cosine_similarity(tir_view_feats, rgb_view_feats + nir_view_feats, dim=-1).mean()
    
        ret.update({'ortho_loss': ortho_loss  * self.args.ortho_loss_weight})
       
        ## modality fusion
        mask_text_feats = self.base_model.encode_text(mask_tokens)
        mask_t_feats = mask_text_feats [torch.arange(mask_text_feats.shape[0]), mask_tokens.argmax(dim=-1)].float()
        
        t_feats_inverse = self.inverseNet(t_feats.half())
        mask_t_feats_inverse = self.inverseNet(mask_t_feats.half())
        
        mask_text_rgb = torch.cat((mask_t_feats_inverse.unsqueeze(1),rgb_feats[:, 1:, :]),dim =1)
        text_nir = torch.cat((t_feats_inverse.unsqueeze(1),nir_feats[:, 1:, :]),dim =1)
        text_tir = torch.cat((t_feats_inverse.unsqueeze(1),tir_feats[:, 1:, :]),dim =1)
        text_multimodal = torch.cat((t_feats_inverse.unsqueeze(1),(rgb_feats[:, 1:, :]+ nir_feats[:, 1:, :]+tir_feats[:, 1:, :])/ 3.0),dim =1)
        fusion_rgb_features = self.cross_former(mask_text_rgb,mask_text_rgb,mask_text_rgb)
        fusion_nir_features = self.cross_former(text_nir,text_nir,text_nir)
        fusion_tir_features = self.cross_former(text_tir,text_tir,text_tir)
        fusion_multimodal_features = self.cross_former(text_multimodal,text_multimodal,text_multimodal)
        
        fusion_rgb_feats = fusion_rgb_features[:,0,:]
        fusion_nir_feats = fusion_nir_features[:,0,:]
        fusion_tir_feats = fusion_tir_features[:,0,:]
        fusion_multimodal_feats =fusion_multimodal_features[:,0,:]
        # ret.update({'itc_loss':(objectives.compute_itc(fusion_rgb_feats.float(), t_feats, logit_scale) +\
        #     objectives.compute_itc(fusion_nir_feats.float(),t_feats, logit_scale)+\
        #     objectives.compute_itc(fusion_tir_feats.float(),t_feats, logit_scale))*self.args.mask_loss_weight
        #     })
        
        ret.update({'itc_loss':(objectives.compute_itc(fusion_rgb_feats.float(), fusion_multimodal_feats.float(), logit_scale) +\
            objectives.compute_itc(fusion_nir_feats.float(),fusion_multimodal_feats.float(), logit_scale)+\
            objectives.compute_itc(fusion_tir_feats.float(),fusion_multimodal_feats.float(), logit_scale))*self.args.mask_loss_weight
            })
        
        
        # rgb_aug_logits = self.classifier(fusion_rgb_feats.half()).float()
        # nir_aug_logits = self.classifier(fusion_nir_feats.half()).float()
        # tir_aug_logits = self.classifier(fusion_tir_feats.half()).float()
        
        # ret.update({'itc_loss':(ce(rgb_aug_logits, batch['pids'])+ce(nir_aug_logits, batch['pids'])+ce(tir_aug_logits, batch['pids'])) *self.args.mask_loss_weight })
        # 用itc 或 id
    
        # mse = nn.MSELoss()
        # ret.update({'itc_loss':(mse(rgb_text_cross,nir_text_cross)+mse(rgb_text_cross,tir_text_cross)) *self.args.mask_loss_weight }
        # )
        # ret.update({'itc_loss':(ce(nir2rgb_text_cross_logits, batch['pids'])+ce(tir2rgb_text_cross_logits, batch['pids'])) *self.args.mask_loss_weight }
        # )
        # 拉近 rgb_text_cross + mask_t_feats   #拉近与t_feats +nir_text_cross
        # 拉近 rgb_text_cross + mask_t_feats   #拉近与t_feats +tir_text_cross
        
        # rgb_aug_feats = rgb_text_cross.mean(dim=1).float() 
        # nir_aug_feats = nir_text_cross.mean(dim=1).float() 
        # tir_aug_feats = tir_text_cross.mean(dim=1).float() 
        # ret.update({'itc_loss':(objectives.compute_itc(rgb_aug_feats, nir_aug_feats, logit_scale) +\
        #     objectives.compute_itc(rgb_aug_feats, tir_aug_feats, logit_scale))*self.args.mask_loss_weight
        #     })
        
        # ret.update({'itc_loss':objectives.compute_sdm(rgb_aug_feats, nir_aug_feats,batch['pids'], logit_scale) *self.args.mask_loss_weight +\
        #     objectives.compute_sdm(rgb_aug_feats, tir_aug_feats, batch['pids'],logit_scale)*self.args.mask_loss_weight
        #     })
        
        # ret.update({'itc_loss':(ce(rgb_aug_logits, batch['pids'])+ce(nir_aug_logits, batch['pids'])+ce(tir_aug_logits, batch['pids'])) *self.args.mask_loss_weight })
        if 'itc' in self.current_task:
            ret.update({'itc_loss':objectives.compute_itc(i_feats, t_feats, logit_scale)})
            
    
        if 'sdm' in self.current_task:
            ret.update({'sdm_loss':objectives.compute_sdm(i_feats, t_feats, batch['pids'], logit_scale)})

        if 'cmpm' in self.current_task:
            ret.update({'cmpm_loss':objectives.compute_cmpm(i_feats, t_feats, batch['pids'])})
        
        if 'id' in self.current_task:
            image_logits = self.classifier(i_feats.half()).float()
            text_logits = self.classifier(t_feats.half()).float()
            ret.update({'id_loss':objectives.compute_id(image_logits, text_logits, batch['pids'])*self.args.id_loss_weight})
                                    
            
            # image_pred = torch.argmax(rgb_image_logits, dim=1)
            image_pred = torch.argmax(image_logits, dim=1)
            text_pred = torch.argmax(text_logits, dim=1)

            image_precision = (image_pred == batch['pids']).float().mean()
            text_precision = (text_pred == batch['pids']).float().mean()
            ret.update({'img_acc': image_precision})
            ret.update({'txt_acc': text_precision})
        
        if 'mlm' in self.current_task:
            mlm_ids = batch['mlm_ids']

            mlm_feats = self.base_model.encode_text(mlm_ids)

            # x = self.cross_former(mlm_feats, image_feats, image_feats)
            
            # 使用融合特征特征进行MLM
            x = self.cross_former(mlm_feats, mm_feats, mm_feats)

            x = self.mlm_head(x)  # [batch_size, text_len, num_colors]

            scores = x.float().reshape(-1, self.args.vocab_size)
            mlm_labels = batch['mlm_labels'].reshape(-1)
            ret.update({'mlm_loss': objectives.compute_mlm(scores, mlm_labels)*self.args.mlm_loss_weight})

            pred = scores.max(1)[1]
            mlm_label_idx = torch.nonzero(mlm_labels)
            acc = (pred[mlm_label_idx] == mlm_labels[mlm_label_idx]).float().mean()
            ret.update({'mlm_acc': acc})
        

        return ret


def build_model(args, num_classes=11003):
    model = IRRA(args, num_classes)
    # covert model to fp16
    convert_weights(model)
    return model



class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x