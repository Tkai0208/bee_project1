# -*- coding: utf-8 -*-
"""数据格式转换功能"""

import os
import json
from collections import defaultdict
from tqdm import tqdm

def coco_to_yolo_single(coco_json, out_dir):
    """单文件稳健转换"""
    os.makedirs(out_dir, exist_ok=True)

    with open(coco_json, 'r', encoding='utf-8') as f:
        coco = json.load(f)

    # 1. 建立索引
    id2img = {i['id']: i for i in coco['images']}
    cat2cls = {c['id']: 0 for c in coco['categories']}  # 蜜蜂单类
    id2anns = defaultdict(list)
    for ann in coco['annotations']:
        if ann.get('ignore', 0) == 1:
            continue
        id2anns[ann['image_id']].append(ann)

    ok_img = 0
    ok_box = 0
    skip_box = 0

    # 2. 逐图转换
    for img_id, img_info in tqdm(id2img.items(), desc=os.path.basename(coco_json)):
        fname = img_info['file_name']
        img_w = img_info['width']
        img_h = img_info['height']
        txt_name = os.path.splitext(fname)[0] + '.txt'
        txt_path = os.path.join(out_dir, txt_name)

        anns = id2anns.get(img_id, [])
        lines = []
        for ann in anns:
            x, y, w, h = ann['bbox']
            # 过滤异常
            if w <= 0 or h <= 0:
                skip_box += 1
                continue
            # 裁剪到图像内
            x1, y1, x2, y2 = x, y, x + w, y + h
            x1 = max(0, min(x1, img_w))
            y1 = max(0, min(y1, img_h))
            x2 = max(0, min(x2, img_w))
            y2 = max(0, min(y2, img_h))
            w, h = x2 - x1, y2 - y1
            if w <= 0 or h <= 0:
                skip_box += 1
                continue

            xc = (x1 + x2) / 2.0 / img_w
            yc = (y1 + y2) / 2.0 / img_h
            w = w / img_w
            h = h / img_h
            cls = cat2cls.get(ann['category_id'], 0)
            lines.append(f"{cls} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
            ok_box += 1

        # 无标注也写空文件，YOLOv8 不报错
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
        ok_img += 1

    print(f"【{os.path.basename(coco_json)}】"
          f" 图象 {ok_img} 张 | 框 {ok_box} 个 | 跳过异常框 {skip_box} 个")
    return ok_img, ok_box, skip_box


def process_all_coco_annotations(datasets_path):
    """处理所有 COCO 标注文件"""
    splits = ['train', 'valid', 'test']
    total_img = total_box = total_skip = 0

    for split in splits:
        json_file = os.path.join(datasets_path, 'original_annotations', split, '_annotations.coco.json')
        if not os.path.exists(json_file):
            print(f"⚠️  未找到 {json_file}，跳过")
            continue
        out_dir = os.path.join(datasets_path, 'labels', split)
        img_cnt, box_cnt, skip_cnt = coco_to_yolo_single(json_file, out_dir)
        total_img += img_cnt
        total_box += box_cnt
        total_skip += skip_cnt

    print("\n===== 转换汇总 =====")
    print(f"总计 图像: {total_img} 张 | 有效框: {total_box} 个 | 跳过异常框: {total_skip} 个")
    print("标签输出目录:", os.path.abspath(os.path.join(datasets_path, 'labels')))

    return total_img, total_box, total_skip