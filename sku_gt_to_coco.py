#!/usr/bin/env python3
"""
SKU-110K : labels YOLO du split test -> COCO.

Produit le fichier que baselines.py lit tel quel, sans
reimplementer la regle d'appariement.

Les dimensions viennent du CSV officiel, pas d'une relecture
des images.

Sortie : ../sku110k-audit/test/_annotations.coco.json

Usage, depuis SKU110K_fixed/ :
    python sku_gt_to_coco.py
"""

import csv
import glob
import json
import os
from collections import defaultdict

ANN_CSV = 'annotations/annotations_test.csv'
LABELS = 'yolo/labels/test'
OUT_DIR = '../sku110k-audit/test'
OUT = os.path.join(OUT_DIR, '_annotations.coco.json')

EXPECTED_IMAGES = 2936
EXPECTED_BOXES = 431546


def load_dims():
    """image_name -> (width, height), depuis le CSV officiel."""
    dims = {}
    with open(ANN_CSV, newline='', encoding='utf-8') as f:
        for r in csv.reader(f):
            if len(r) < 8:
                continue
            try:
                dims[r[0]] = (int(r[6]), int(r[7]))
            except ValueError:
                continue
    return dims


def main():
    if not os.path.isdir(LABELS):
        raise SystemExit('%s introuvable : faire la conversion YOLO d abord.' % LABELS)

    dims = load_dims()
    print('dimensions lues pour %d images' % len(dims))

    images, annotations = [], []
    img_id, ann_id = 1, 1
    sans_dim = 0

    for path in sorted(glob.glob(os.path.join(LABELS, '*.txt'))):
        stem = os.path.splitext(os.path.basename(path))[0]
        fname = stem + '.jpg'

        if fname not in dims:
            sans_dim += 1
            continue
        W, H = dims[fname]

        images.append({
            'id': img_id,
            'file_name': fname,
            'width': W,
            'height': H,
        })

        for line in open(path, encoding='utf-8'):
            parts = line.split()
            if len(parts) < 5:
                continue
            cx, cy, nw, nh = (float(v) for v in parts[1:5])
            w = nw * W
            h = nh * H
            x = (cx * W) - w / 2.0
            y = (cy * H) - h / 2.0
            annotations.append({
                'id': ann_id,
                'image_id': img_id,
                'category_id': 1,
                'bbox': [x, y, w, h],
                'area': w * h,
                'iscrowd': 0,
            })
            ann_id += 1

        img_id += 1

    out = {
        'info': {'description': 'SKU-110K fixed, split test, converti depuis YOLO'},
        'licenses': [],
        'images': images,
        'annotations': annotations,
        'categories': [
            {'id': 0, 'name': 'Items', 'supercategory': 'none'},
            {'id': 1, 'name': 'Product', 'supercategory': 'Items'},
        ],
    }

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(OUT, 'w', encoding='utf-8') as f:
        json.dump(out, f)

    print('images ecrites : %d' % len(images))
    print('boites ecrites : %d' % len(annotations))
    if sans_dim:
        print('images sans dimension dans le CSV : %d' % sans_dim)
    print('sortie : %s' % OUT)

    assert len(images) == EXPECTED_IMAGES, \
        'attendu %d images, obtenu %d' % (EXPECTED_IMAGES, len(images))
    assert len(annotations) == EXPECTED_BOXES, \
        'attendu %d boites, obtenu %d' % (EXPECTED_BOXES, len(annotations))
    print('\nControles passes : %d images, %d boites.' % (EXPECTED_IMAGES, EXPECTED_BOXES))


if __name__ == '__main__':
    main()