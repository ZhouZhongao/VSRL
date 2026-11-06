from typing import List
from torch.utils.data import Dataset
import os.path as osp
import logging
import torch
from utils.iotools import read_image
from utils.simple_tokenizer import SimpleTokenizer
from prettytable import PrettyTable
import random
import regex as re
import copy


class BaseDataset(object):
    """
    Base class of text to image reid dataset
    """
    logger = logging.getLogger("IRRA.dataset")

    def show_dataset_info(self):
        num_train_pids, num_train_imgs, num_train_captions = len(
            self.train_id_container), len(self.train_annos), len(self.train)
        num_test_pids, num_test_imgs, num_test_captions = len(
            self.test_id_container), len(self.test_annos), len(
                self.test['captions'])
        # num_val_pids, num_val_imgs, num_val_captions = len(
        #     self.val_id_container), len(self.val_annos), len(
        #         self.val['captions'])

        # TODO use prettytable print comand line table

        self.logger.info(f"{self.__class__.__name__} Dataset statistics:")
        table = PrettyTable(['subset', 'ids', 'images', 'captions'])
        table.add_row(
            ['train', num_train_pids, num_train_imgs, num_train_captions])
        table.add_row(
            ['test', num_test_pids, num_test_imgs, num_test_captions])
        # table.add_row(['val', num_val_pids, num_val_imgs, num_val_captions])
        self.logger.info('\n' + str(table))


def tokenize(caption: str, tokenizer, text_length=77, truncate=True) -> torch.LongTensor:
    sot_token = tokenizer.encoder["<|startoftext|>"]
    eot_token = tokenizer.encoder["<|endoftext|>"]
    tokens = [sot_token] + tokenizer.encode(caption) + [eot_token]

    result = torch.zeros(text_length, dtype=torch.long)
    if len(tokens) > text_length:
        if truncate:
            tokens = tokens[:text_length]
            tokens[-1] = eot_token
        else:
            raise RuntimeError(
                f"Input {caption} is too long for context length {text_length}"
            )
    result[:len(tokens)] = torch.tensor(tokens)
    return result


class ImageTextDataset(Dataset):
    def __init__(self,
                 dataset,
                 transform=None,
                 text_length: int = 77,
                 truncate: bool = True):
        self.dataset = dataset
        self.transform = transform
        self.text_length = text_length
        self.truncate = truncate
        self.tokenizer = SimpleTokenizer()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        # 修改为支持三模态：pid, image_id, rgb_path, nir_path, tir_path, caption
        pid, image_id, rgb_path, nir_path, tir_path, captions , view_id= self.dataset[index]
        # pid, image_id, img_path, caption = self.dataset[index]
        # img = read_image(img_path)
        # if self.transform is not None:
        #     img = self.transform(img)
        # 读取三个模态的图像
        rgb_img = read_image(rgb_path)
        nir_img = read_image(nir_path)
        tir_img = read_image(tir_path)
        
        if self.transform is not None:
            rgb_img = self.transform(rgb_img)
            nir_img = self.transform(nir_img)
            tir_img = self.transform(tir_img)
        tokens_list =[]
        mask_tokens_list= []
        for caption in captions:
            tokens = tokenize(caption, tokenizer=self.tokenizer, text_length=self.text_length, truncate=self.truncate)
            tokens_list.append(tokens)
            caption_mask = self._mask_color(caption)
            mask_tokens = tokenize(caption_mask, tokenizer=self.tokenizer, text_length=self.text_length, truncate=self.truncate)
            mask_tokens_list.append(mask_tokens)
            
        tokens_list = torch.stack(tokens_list)
        mask_tokens_list = torch.stack(mask_tokens_list)
        
        
        
        # ret = {
        #     'pids': pid,
        #     'image_ids': image_id,
        #     'images': img,
        #     'caption_ids': tokens,
        # }

        ret = {
            'pids': pid,
            'image_ids': image_id,
            'rgb_images': rgb_img,
            'nir_images': nir_img,
            'tir_images': tir_img,
            'caption_ids': tokens_list,
            "view_ids" : view_id,
            "mask_tokens_list": mask_tokens_list ,
        }

        return ret
    def _mask_color(self, caption):
        """
        Masking some random tokens for Language Model task with probabilities as in the original BERT paper.
        :param tokens: list of int, tokenized sentence.
        :return: (list of int, list of int), masked tokens and related labels for MLM prediction
        """
        color_words = ["red", "blue", "green", "yellow", "black", "white",
                        "brown", "pink", "purple", "orange", "gray", "grey",
                        "blonde", "golden", "beige", "khaki", "dark", "camouflage"]
        pattern = r"\b(" + "|".join(color_words) + r")\b"
        masked_caption = re.sub(pattern, "<|mask|>", caption, flags=re.IGNORECASE)
        return masked_caption

class ImageDataset(Dataset):
    def __init__(self, image_pids, img_paths, transform=None):
        self.image_pids = image_pids
        self.img_paths = img_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_pids)

    def __getitem__(self, index):
        pid, img_path = self.image_pids[index], self.img_paths[index]
        img = read_image(img_path)
        if self.transform is not None:
            img = self.transform(img)
        return pid, img

class ImageDataset_Multimodal(Dataset):
    def __init__(self, image_pids, rgb_img_paths,nir_img_paths,tir_img_paths, transform=None):
        self.image_pids = image_pids
        self.rgb_img_paths = rgb_img_paths
        self.nir_img_paths = nir_img_paths
        self.tir_img_paths = tir_img_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_pids)

    def __getitem__(self, index):
        pid, rgb_img_path, nir_img_path, tir_img_path = self.image_pids[index], self.rgb_img_paths[index], self.nir_img_paths[index], self.tir_img_paths[index]
        rgb_img = read_image(rgb_img_path)
        nir_img = read_image(nir_img_path)
        tir_img = read_image(tir_img_path)
        if self.transform is not None:
            rgb_img = self.transform(rgb_img)
            nir_img = self.transform(nir_img)
            tir_img = self.transform(tir_img)
        return pid,(rgb_img,nir_img,tir_img)

class TextDataset(Dataset):
    def __init__(self,
                 caption_pids,
                 captions,
                 text_length: int = 77,
                 truncate: bool = True):
        self.caption_pids = caption_pids
        self.captions = captions
        self.text_length = text_length
        self.truncate = truncate
        self.tokenizer = SimpleTokenizer()

    def __len__(self):
        return len(self.caption_pids)

    def __getitem__(self, index):
        pid, caption = self.caption_pids[index], self.captions[index]

        caption = tokenize(caption, tokenizer=self.tokenizer, text_length=self.text_length, truncate=self.truncate)

        return pid, caption


class ImageTextMLMDataset(Dataset):
    def __init__(self,
                 dataset,
                 transform=None,
                 text_length: int = 77,
                 truncate: bool = True):
        self.dataset = dataset
        self.transform = transform
        self.text_length = text_length
        self.truncate = truncate

        self.tokenizer = SimpleTokenizer()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        # 修改为支持三模态：pid, image_id, rgb_path, nir_path, tir_path, caption
        pid, image_id, rgb_path, nir_path, tir_path, caption = self.dataset[index]
        # pid, image_id, img_path, caption = self.dataset[index]
        # img = read_image(img_path)
        # if self.transform is not None:
        #     img = self.transform(img)
        # 读取三个模态的图像
        rgb_img = read_image(rgb_path)
        nir_img = read_image(nir_path)
        tir_img = read_image(tir_path)
        
        if self.transform is not None:
            rgb_img = self.transform(rgb_img)
            nir_img = self.transform(nir_img)
            tir_img = self.transform(tir_img)
        
        caption_tokens = tokenize(caption, tokenizer=self.tokenizer, text_length=self.text_length, truncate=self.truncate)

        mlm_tokens, mlm_labels = self._build_random_masked_tokens_and_labels(caption_tokens.cpu().numpy())


        # ret = {
        #     'pids': pid,
        #     'image_ids': image_id,
        #     'images': img,
        #     'caption_ids': caption_tokens,
        #     'mlm_ids': mlm_tokens,
        #     'mlm_labels': mlm_labels
        # }
        ret = {
            'pids': pid,
            'image_ids': image_id,
            'rgb_images': rgb_img,
            'nir_images': nir_img,
            'tir_images': tir_img,
            'caption_ids': caption_tokens,
            'mlm_ids': mlm_tokens,
            'mlm_labels': mlm_labels
        }

        return ret

    def _build_random_masked_tokens_and_labels(self, tokens):
        """
        Masking some random tokens for Language Model task with probabilities as in the original BERT paper.
        :param tokens: list of int, tokenized sentence.
        :return: (list of int, list of int), masked tokens and related labels for MLM prediction
        """
        mask = self.tokenizer.encoder["<|mask|>"]
        token_range = list(range(1, len(self.tokenizer.encoder)-3)) # 1 ~ 49405
        
        labels = []
        for i, token in enumerate(tokens):
            if 0 < token < 49405:
                prob = random.random()
                # mask token with 15% probability
                if prob < 0.15:
                    prob /= 0.15 #此时prob会被放大，但仍然小于100%

                    # 80% randomly change token to mask token
                    if prob < 0.8:
                        tokens[i] = mask

                    # 10% randomly change token to random token
                    elif prob < 0.9:
                        tokens[i] = random.choice(token_range)

                    # -> rest 10% randomly keep current token

                    # append current token to output (we will predict these later)
                    labels.append(token)
                else:
                    # no masking token (will be ignored by loss function later)
                    labels.append(0)
            else:
                labels.append(0)
        
        if all(l == 0 for l in labels):
            # at least mask 1
            labels[1] = tokens[1]
            tokens[1] = mask

        return torch.tensor(tokens), torch.tensor(labels)