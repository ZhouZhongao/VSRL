import os.path as op
import glob
import re
import json
from typing import List, Dict, Tuple, Set

from .bases import BaseDataset
import random


class MARS_Text_Group(BaseDataset):
    """
    三模态文本-图像检索数据集（TIReID）
    支持RGB、NIR、TIR三个模态，使用RGB的文本描述

    - dataset_dir: marslite
    - 使用 text_update/{train|test}_annotations.json
    - 支持三个模态：RGB、NIR、TIR
    - 输出格式对齐 RSTPReid，但扩展为三模态
    """
    dataset_dir = 'mars_1000'
    # dataset_dir = 'mars1000_update'

    def __init__(self, root: str = '', verbose: bool = True):
        super(MARS_Text_Group, self).__init__()

        self.dataset_dir = op.join(root, self.dataset_dir)
        # 三个模态的图像目录
        self.train_img_dir_rgb = op.join(self.dataset_dir, 'train', 'RGB_topFinal_min4')
        self.train_img_dir_nir = op.join(self.dataset_dir, 'train', 'IR_topFinal_min4')
        self.train_img_dir_tir = op.join(self.dataset_dir, 'train', 'Thermal_topFinal_min4')
        
        self.test_img_dir_rgb = op.join(self.dataset_dir, 'test', 'RGB_topFinal_min4')
        self.test_img_dir_nir = op.join(self.dataset_dir, 'test', 'IR_topFinal_min4')
        self.test_img_dir_tir = op.join(self.dataset_dir, 'test', 'Thermal_topFinal_min4')

        # 标注文件（仅使用RGB的文本描述）
        self.text_dir = op.join(self.dataset_dir, 'text_group')
        self.train_anno_path = op.join(self.text_dir, 'train_annotations_group.json')
        self.test_anno_path = op.join(self.text_dir, 'test_annotations_group.json')

        self._check_before_run()

        # 加载RGB的文本标注
        train_caption_map = self._load_caption_map(self.train_anno_path)
        test_caption_map = self._load_caption_map(self.test_anno_path)

        # 构建三模态的标注
        self.train_annos = self._build_multimodal_annos(
            self.train_img_dir_rgb, self.train_img_dir_nir, self.train_img_dir_tir, 
            train_caption_map
        )
        self.test_annos = self._build_multimodal_annos(
            self.test_img_dir_rgb, self.test_img_dir_nir, self.test_img_dir_tir, 
            test_caption_map
        )
       

        # 基于三模态标注生成训练/测试/验证数据
        self.train, self.train_id_container = self._process_multimodal_anno(
            self.train_annos, training=True
        )
        self.test, self.test_id_container = self._process_multimodal_anno(
            self.test_annos, training=False
        )


        if verbose:
            self.logger.info("=> MARSLITE (MARS_Text) Three-Modal Images and Captions are loaded")
            self.show_dataset_info()

    def _check_before_run(self):
        if not op.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        for modal_dir in [self.train_img_dir_rgb, self.train_img_dir_nir, self.train_img_dir_tir]:
            if not op.exists(modal_dir):
                raise RuntimeError("'{}' is not available".format(modal_dir))
        for modal_dir in [self.test_img_dir_rgb, self.test_img_dir_nir, self.test_img_dir_tir]:
            if not op.exists(modal_dir):
                raise RuntimeError("'{}' is not available".format(modal_dir))

        if not op.exists(self.train_anno_path):
            raise RuntimeError("'{}' is not available".format(self.train_anno_path))
        if not op.exists(self.test_anno_path):
            raise RuntimeError("'{}' is not available".format(self.test_anno_path))

    def _load_caption_map(self, anno_path: str) -> Dict[str, List[str]]:
        """
        将 JSON 标注加载为 {basename: [captions...]}
        仅使用存在 captions 且非空的项。
        """
        with open(anno_path, 'r') as f:
            annos = json.load(f)

        caption_map: Dict[str, List[str]] = {}
        for item in annos:
            if 'captions' in item and item['captions']:
                # basename = op.basename(item.get('img_path', ''))
                id = op.basename(item.get('id', ''))
                if id:
                    caption_map[id] = list(item['captions'])
        
        self.logger.info(f"Loaded {len(caption_map)} captions from {anno_path}")
        return caption_map

    def _collect_img_list(self, img_dir: str) -> List[str]:
        return sorted(glob.glob(op.join(img_dir, '*.jpg')))

    def _extract_pid_from_name(self, filename: str) -> str:
        """
        通过文件名解析 PID。示例：m1_c2_m1_f000044 -> m1
        返回原始 pid 字符串（不做 int 转换）。
        """
        pattern = re.compile(r'([a-z\d]+)_c\d+_m\d+_f\d+', re.IGNORECASE)
        m = pattern.search(filename)
        pid_str = m.group(1)
        return pid_str

    def _build_multimodal_annos(self, rgb_dir: str, nir_dir: str, tir_dir: str, 
                                caption_map: Dict[str, List[str]]) -> List[dict]:
        """
        构建三模态的标注列表，每个样本包含RGB、NIR、TIR三个模态的图像路径
        """
        # 收集RGB模态的图像（作为主要参考）
        rgb_imgs = self._collect_img_list(rgb_dir)
        
        # 添加调试信息
        self.logger.info(f"Found {len(rgb_imgs)} RGB images in {rgb_dir}")
        # 构建文件名到路径的映射
        rgb_map = {op.basename(p): p for p in rgb_imgs}

        # 使用RGB文件名构建PID标签（因为文本描述基于RGB）
        pid2label = self._build_pid_labels(rgb_imgs)
        self.logger.info(f"Found {len(pid2label)} unique PIDs")

        annos: List[dict] = []
        valid_count = 0
        total_count = 0
        
        for rgb_path in rgb_imgs:
            total_count += 1
            basename = op.basename(rgb_path)
            pid_raw = self._extract_pid_from_name(basename)
            if not pid_raw:
                self.logger.debug(f"Skipping {basename}: no PID extracted")
                continue
            
            # 根据RGB文件名生成对应的NIR和TIR文件名
            nir_basename = self._convert_modal_in_filename(basename, 'm2')
            tir_basename = self._convert_modal_in_filename(basename, 'm3')
            
            # 检查其他两个模态的图像文件是否存在
            # 注意：这里应该使用完整的模态目录路径
            nir_path = op.join(nir_dir, nir_basename)  # nir_dir 已经是 /path/to/IR
            tir_path = op.join(tir_dir, tir_basename)  # tir_dir 已经是 /path/to/Thermal
            
            if not op.exists(nir_path):
                self.logger.debug(f"Skipping {basename}: NIR file not found: {nir_path}")
                continue
            if not op.exists(tir_path):
                self.logger.debug(f"Skipping {basename}: TIR file not found: {tir_path}")
                continue
            
            pid_label = pid2label[pid_raw]
            # caps = caption_map.get(basename, [])
            caps = caption_map.get(pid_raw, [])
            if not caps:
                self.logger.debug(f"Skipping {basename}: no captions found")
                continue
            # cap = random.choice(caps)
            annos.append({
                'id': int(pid_label),
                'rgb_path': basename,
                'nir_path': nir_basename,
                'tir_path': tir_basename,
                'captions': list(caps),
            })
            valid_count += 1
        
        self.logger.info(f"Built {len(annos)} valid annotations from {total_count} RGB images")
        return annos

    def _build_pid_labels(self, img_paths: List[str]) -> Dict[str, int]:
        """
        从图像路径中收集原始 pid，并构建 pid -> label 的映射，label 从 0 开始。
        """
        pid_set: Set[str] = set()
        for p in img_paths:
            pid_raw = self._extract_pid_from_name(op.basename(p))
            if pid_raw:
                pid_set.add(pid_raw)
        sorted_pids = sorted(pid_set)
        pid2label = {pid: idx for idx, pid in enumerate(sorted_pids)}
        return pid2label

    def _process_multimodal_anno(self, annos: List[dict], training=False):
        """
        处理三模态标注，生成训练或评估数据
        """
        pid_container: Set[int] = set()
        
        if training:
            dataset = []
            image_id = 0
            for anno in annos:
                pid = int(anno['id'])
                pid_container.add(pid)
                
                # 构建三个模态的完整路径
                rgb_path = op.join(self.train_img_dir_rgb, anno['rgb_path'])
                nir_path = op.join(self.train_img_dir_nir, anno['nir_path'])
                tir_path = op.join(self.train_img_dir_tir, anno['tir_path'])
                
                captions = anno['captions']
                view_id = get_camera_view(rgb_path)
                dataset.append((pid, image_id, rgb_path, nir_path, tir_path, captions,view_id))
                image_id += 1
            
            # 训练集 id 连续性检查和重新编码
            sorted_pids = sorted(pid_container)
            if sorted_pids != list(range(len(sorted_pids))):
                # PID不连续，需要重新编码
                pid_mapping = {old_pid: new_idx for new_idx, old_pid in enumerate(sorted_pids)}
                # 重新构建数据集
                dataset = []
                image_id = 0
                for anno in annos:
                    old_pid = int(anno['id'])
                    new_pid = pid_mapping[old_pid]
                    rgb_path = op.join(self.train_img_dir_rgb, anno['rgb_path'])
                    nir_path = op.join(self.train_img_dir_nir, anno['nir_path'])
                    tir_path = op.join(self.train_img_dir_tir, anno['tir_path'])
                    
                    captions = anno['captions']
                    view_id = get_camera_view(rgb_path)
                    dataset.append((new_pid, image_id, rgb_path, nir_path, tir_path, captions,view_id))
                    image_id += 1
                # 更新PID容器
                pid_container = set(range(len(sorted_pids)))
            
            return dataset, pid_container
        else:
            # 测试/验证集：分别存储三个模态的图像信息
            rgb_paths: List[str] = []
            nir_paths: List[str] = []
            tir_paths: List[str] = []
            captions: List[str] = []
            image_pids: List[int] = []
            caption_pids: List[int] = []
            
            current_pid = 0 
            for anno in annos:
                pid = int(anno['id'])
                pid_container.add(pid)
                
                # 构建三个模态的完整路径
                # if 'test' in self.test_img_dir_rgb:  # 判断是测试集还是验证集
                #     rgb_path = op.join(self.test_img_dir_rgb, anno['rgb_path'])
                #     nir_path = op.join(self.test_img_dir_nir, anno['nir_path'])
                #     tir_path = op.join(self.test_img_dir_tir, anno['tir_path'])
                # else:
                #     rgb_path = op.join(self.val_img_dir_rgb, anno['rgb_path'])
                #     nir_path = op.join(self.val_img_dir_nir, anno['nir_path'])
                #     tir_path = op.join(self.val_img_dir_tir, anno['tir_path'])
                rgb_path = op.join(self.test_img_dir_rgb, anno['rgb_path'])
                nir_path = op.join(self.test_img_dir_nir, anno['nir_path'])
                tir_path = op.join(self.test_img_dir_tir, anno['tir_path'])
                rgb_paths.append(rgb_path)
                nir_paths.append(nir_path)
                tir_paths.append(tir_path)
                image_pids.append(pid)
                if current_pid == pid :
                    for caption in anno['captions']:
                        captions.append(caption)
                        caption_pids.append(pid)
                    current_pid +=1
                else:
                    continue

            dataset = {
                "image_pids": image_pids,
                "rgb_paths": rgb_paths,
                "nir_paths": nir_paths,
                "tir_paths": tir_paths,
                "caption_pids": caption_pids,
                "captions": captions
            }
            return dataset, pid_container

    def _convert_modal_in_filename(self, filename: str, target_modal: str) -> str:
        """
        将文件名中的模态部分转换为目标模态
        
        Args:
            filename: 原始文件名，如 'z93_c2_m1_f008562'
            target_modal: 目标模态，'m1'(RGB), 'm2'(NIR), 'm3'(TIR)
        
        Returns:
            转换后的文件名，如 'z93_c2_m2_f008562'
        """
        # 使用正则表达式替换第三个m数字部分
        pattern = r'([a-z\d]+_c\d+_m)\d+(_f\d+)'
        replacement = r'\g<1>' + target_modal[-1] + r'\g<2>'
        converted = re.sub(pattern, replacement, filename)
        return converted
    
def get_camera_view(filename: str) -> int:
    match = re.search(r'_(c\d+)_', filename)
    if not match:
        return -1  # 无法匹配相机信息时返回 -1
    
    camera = match.group(1)
    cam_id = int(camera[1:])  # 提取数字部分，如 c1 -> 1
    
    # c1-c5: 返回0（地面视角），c6-c7: 返回1（空中视角）
    if 1 <= cam_id <= 5:
        return 0
    elif 6 <= cam_id <= 7:
        return 1
    else:
        return -1