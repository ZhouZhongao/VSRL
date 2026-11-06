import logging
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader
from datasets.sampler import RandomIdentitySampler
from datasets.sampler_ddp import RandomIdentitySampler_DDP
from torch.utils.data.distributed import DistributedSampler

from utils.comm import get_world_size

from .bases import ImageDataset,ImageDataset_Multimodal, TextDataset, ImageTextDataset, ImageTextMLMDataset

from .cuhkpedes import CUHKPEDES
from .icfgpedes import ICFGPEDES
from .rstpreid import RSTPReid
from .MARS_Text import MARS_Text
from .MARS_Text_Group import MARS_Text_Group
from .RGBNT201_Text import RGBNT201_Text

__factory = {'CUHK-PEDES': CUHKPEDES, 'ICFG-PEDES': ICFGPEDES, 'RSTPReid': RSTPReid , 'MARS':MARS_Text,'MARS_Group':MARS_Text_Group,'RGBNT201':RGBNT201_Text}


def build_transforms(img_size=(384, 128), aug=False, is_train=True):
    height, width = img_size

    mean = [0.48145466, 0.4578275, 0.40821073]
    std = [0.26862954, 0.26130258, 0.27577711]

    if not is_train:
        transform = T.Compose([
            T.Resize((height, width)),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])
        return transform

    # transform for training
    if aug:
        transform = T.Compose([
            T.Resize((height, width)),
            T.RandomHorizontalFlip(0.5),
            T.Pad(10),
            T.RandomCrop((height, width)),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
            T.RandomErasing(scale=(0.02, 0.4), value=mean),
        ])
    else:
        transform = T.Compose([
            T.Resize((height, width)),
            T.RandomHorizontalFlip(0.5),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])
    return transform


def collate(batch):
    keys = set([key for b in batch for key in b.keys()])
    # turn list of dicts data structure to dict of lists data structure
    dict_batch = {k: [dic[k] if k in dic else None for dic in batch] for k in keys}

    batch_tensor_dict = {}
    for k, v in dict_batch.items():
        if k == "caption_ids":
            batch_tensor_dict.update({k: v})
        else:
            if isinstance(v[0], int):
                batch_tensor_dict.update({k: torch.tensor(v)})
            elif torch.is_tensor(v[0]):
                batch_tensor_dict.update({k: torch.stack(v)})
            else:
                raise TypeError(f"Unexpect data type: {type(v[0])} in a batch.")

    return batch_tensor_dict

def build_dataloader(args, tranforms=None):
    logger = logging.getLogger("IRRA.dataset")

    num_workers = args.num_workers
    dataset = __factory[args.dataset_name](root=args.root_dir)
    num_classes = len(dataset.train_id_container)
    
    if args.training:
        train_transforms = build_transforms(img_size=args.img_size,
                                            aug=args.img_aug,
                                            is_train=True)
        val_transforms = build_transforms(img_size=args.img_size,
                                          is_train=False)

        if args.MLM:
            train_set = ImageTextMLMDataset(dataset.train,
                                     train_transforms,
                                     text_length=args.text_length)
        else:
            train_set = ImageTextDataset(dataset.train,
                                     train_transforms,
                                     text_length=args.text_length)

        if args.sampler == 'identity':
            if args.distributed:
                logger.info('using ddp random identity sampler')
                logger.info('DISTRIBUTED TRAIN START')
                mini_batch_size = args.batch_size // get_world_size()
                # TODO wait to fix bugs
                data_sampler = RandomIdentitySampler_DDP(
                    dataset.train, args.batch_size, args.num_instance)
                batch_sampler = torch.utils.data.sampler.BatchSampler(
                    data_sampler, mini_batch_size, True)

            else:
                logger.info(
                    f'using random identity sampler: batch_size: {args.batch_size}, id: {args.batch_size // args.num_instance}, instance: {args.num_instance}'
                )
                train_loader = DataLoader(train_set,
                                          batch_size=args.batch_size,
                                          sampler=RandomIdentitySampler(
                                              dataset.train, args.batch_size,
                                              args.num_instance),
                                          num_workers=num_workers,
                                          collate_fn=collate)
        elif args.sampler == 'random':
            # TODO add distributed condition
            logger.info('using random sampler')
            train_loader = DataLoader(train_set,
                                      batch_size=args.batch_size,
                                      shuffle=True,
                                      num_workers=num_workers,
                                      collate_fn=collate)
        else:
            logger.error('unsupported sampler! expected softmax or triplet but got {}'.format(args.sampler))

        # use test set as validate set
        # ds = dataset.val if args.val_dataset == 'val' else dataset.test
        ####------- 用query_mm中的 text当qeury去检索test中的三模态#########
        # ds_query_mm = dataset.val
        ds_train = dataset.train
        ds_test = dataset.test
         # 创建三个模态的图像数据集
        val_rgb_set = ImageDataset(ds_test['image_pids'], ds_test['rgb_paths'], val_transforms)
        val_nir_set = ImageDataset(ds_test['image_pids'], ds_test['nir_paths'], val_transforms)
        val_tir_set = ImageDataset(ds_test['image_pids'], ds_test['tir_paths'], val_transforms)
        val_mm_set = ImageDataset_Multimodal(ds_test['image_pids'], ds_test['rgb_paths'],ds_test['nir_paths'],ds_test['tir_paths'], val_transforms)
        
        val_txt_set = TextDataset(ds_test['caption_pids'],
                                  ds_test['captions'],
                                  text_length=args.text_length)
        val_rgb_loader = DataLoader(val_rgb_set,
                                    batch_size=args.batch_size,
                                    shuffle=False,
                                    num_workers=num_workers)
        val_nir_loader = DataLoader(val_nir_set,
                                    batch_size=args.batch_size,
                                    shuffle=False,
                                    num_workers=num_workers)
        val_tir_loader = DataLoader(val_tir_set,
                                    batch_size=args.batch_size,
                                    shuffle=False,
                                    num_workers=num_workers)
        val_mm_loader = DataLoader(val_mm_set,
                                    batch_size=args.batch_size,
                                    shuffle=False,
                                    num_workers=num_workers)
        val_txt_loader = DataLoader(val_txt_set,
                                    batch_size=args.batch_size,
                                    shuffle=False,
                                    num_workers=num_workers)
        # val_img_loader = DataLoader(val_img_set,
        #                             batch_size=args.batch_size,
        #                             shuffle=False,
        #                             num_workers=num_workers)
        # val_txt_loader = DataLoader(val_txt_set,
        #                             batch_size=args.batch_size,
        #                             shuffle=False,
        #                             num_workers=num_workers)

        # return train_loader, val_img_loader, val_txt_loader, num_classes
        return train_loader, val_rgb_loader, val_nir_loader, val_tir_loader, val_mm_loader, val_txt_loader, num_classes


    else:
        # build dataloader for testing
        if tranforms:
            test_transforms = tranforms
        else:
            test_transforms = build_transforms(img_size=args.img_size,
                                               is_train=False)

        ds = dataset.test
        # test_img_set = ImageDataset(ds['image_pids'], ds['img_paths'],
        #                             test_transforms)
        # 创建三个模态的图像数据集
        test_rgb_set = ImageDataset(ds['image_pids'], ds['rgb_paths'], test_transforms)
        test_nir_set = ImageDataset(ds['image_pids'], ds['nir_paths'], test_transforms)
        test_tir_set = ImageDataset(ds['image_pids'], ds['tir_paths'], test_transforms)
        
        test_txt_set = TextDataset(ds['caption_pids'],
                                   ds['captions'],
                                   text_length=args.text_length)

        # test_img_loader = DataLoader(test_img_set,
        #                              batch_size=args.test_batch_size,
        #                              shuffle=False,
        #                              num_workers=num_workers)
        test_rgb_loader = DataLoader(test_rgb_set,
                                     batch_size=args.test_batch_size,
                                     shuffle=False,
                                     num_workers=num_workers)
        test_nir_loader = DataLoader(test_nir_set,
                                     batch_size=args.test_batch_size,
                                     shuffle=False,
                                     num_workers=num_workers)
        test_tir_loader = DataLoader(test_tir_set,
                                     batch_size=args.test_batch_size,
                                     shuffle=False,
                                     num_workers=num_workers)
        test_txt_loader = DataLoader(test_txt_set,
                                     batch_size=args.test_batch_size,
                                     shuffle=False,
                                     num_workers=num_workers)
        # return test_img_loader, test_txt_loader, num_classes
        return test_rgb_loader, test_nir_loader, test_tir_loader, test_txt_loader, num_classes
