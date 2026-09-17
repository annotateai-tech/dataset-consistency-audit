#!/usr/bin/env python3
"""
SKU-110K : overlay des desaccords, sur les vingt images tirees
a l'avance.

L'appariement n'est PAS recalcule ici. match() est importee de
baselines.py, exactement comme pour le tableau. Ce fichier ne
fait que du dessin : il ne peut donc pas changer un chiffre.

A 150 boites par image, tout dessiner est illisible. Donc :
  appariees        gris fin, 1 px, ou rien avec --sans-appariees
  label seul       rouge plein, epais
  prediction seule vert plein, epais

Sortie en pleine resolution, pas redimensionnee : c'est en
zoomant sur un rayon qu'on lit une regle.

Usage, depuis sku110k-audit/ :
    python sku110k_overlay.py
    python sku110k_overlay.py --sans-appariees
    python sku110k_overlay.py test_1053.jpg
"""

import glob
import json
import os
import sys

from PIL import Image, ImageDraw

from baselines import match, load_coco, IOU_MATCH

Image.MAX_IMAGE_PIXELS = None

COCO = 'test/_annotations.coco.json'
PREDS = 'predictions'
IMAGES = '../SKU110K_fixed/yolo/images/test'
OUT_DIR = 'overlay'
READ_SAMPLE = 'sku110k_test_read_sample.txt'

GREY = (140, 140, 140)
RED = (255, 60, 60)
GREEN = (0, 230, 118)


def draw_boxes(d, boxes, colour, width):
    for (x, y, w, h) in boxes:
        d.rectangle([x, y, x + w, y + h], outline=colour, width=width)


def main():
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    sans_appariees = '--sans-appariees' in sys.argv

    if not os.path.exists(COCO):
        raise SystemExit('Introuvable : ' + COCO)
    if not os.path.isdir(IMAGES):
        raise SystemExit('Images introuvables : ' + IMAGES)

    if args:
        names = args
    elif os.path.exists(READ_SAMPLE):
        names = [l.strip() for l in open(READ_SAMPLE, encoding='utf-8') if l.strip()]
    else:
        raise SystemExit('%s absent et aucune image donnee en argument.' % READ_SAMPLE)

    gt_all, sizes = load_coco(COCO)
    os.makedirs(OUT_DIR, exist_ok=True)

    print('Appariement importe de baselines.py, IoU %.2f' % IOU_MATCH)
    print('Appariees : %s' % ('masquees' if sans_appariees else 'gris fin'))
    print()

    for fname in names:
        stem = os.path.splitext(fname)[0]
        img_path = os.path.join(IMAGES, fname)
        pred_path = os.path.join(PREDS, stem + '.json')

        if not os.path.isfile(img_path):
            print('  %-16s image absente' % fname)
            continue
        if not os.path.isfile(pred_path):
            print('  %-16s prediction absente' % fname)
            continue

        raw = json.load(open(pred_path, encoding='utf-8'))
        pred = [
            (d['bbox']['x'], d['bbox']['y'], d['bbox']['width'], d['bbox']['height'])
            for d in raw.get('detections', [])
        ]
        gt = gt_all.get(fname, [])

        matched_gt, used_pred = match(gt, pred, IOU_MATCH)

        gt_only = [g for i, g in enumerate(gt) if i not in matched_gt]
        pred_only = [p for i, p in enumerate(pred) if i not in used_pred]
        matched = [g for i, g in enumerate(gt) if i in matched_gt]

        im = Image.open(img_path).convert('RGB')
        d = ImageDraw.Draw(im)

        # Epaisseur proportionnelle a l'image : lisible en zoom.
        thick = max(3, int(round(min(im.size) / 500.0)))

        if not sans_appariees:
            draw_boxes(d, matched, GREY, 1)
        draw_boxes(d, gt_only, RED, thick)
        draw_boxes(d, pred_only, GREEN, thick)

        suffix = '_desaccords' if sans_appariees else '_overlay'
        out = os.path.join(OUT_DIR, stem + suffix + '.jpg')
        im.save(out, quality=92)

        print('  %-16s  gt %4d  pred %4d  |  rouge %3d  vert %3d  ->  %s'
              % (fname, len(gt), len(pred), len(gt_only), len(pred_only),
                 os.path.basename(out)))

    print()
    print('rouge = label du dataset sans prediction')
    print('vert  = prediction sans label')
    if not sans_appariees:
        print('gris  = apparie, IoU >= %.2f' % IOU_MATCH)
    print()
    print('Sortie : %s/' % OUT_DIR)


if __name__ == '__main__':
    main()