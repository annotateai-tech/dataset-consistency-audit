#!/usr/bin/env python3
"""
SKU-110K : predictions YOLO -> JSON au format lu par baselines.py.

Schema reproduit champ pour champ depuis les fichiers de
retail-audit/predictions/ :

    {"success": true,
     "image": {"width": W, "height": H, "filename": "..."},
     "detections": [{"class": "...", "confidence": 0.92,
                     "bbox": {"x": ..., "y": ..., "width": ..., "height": ...}}]}

x et y sont le coin haut-gauche en pixels, pas le centre.
Ultralytics ecrit "class cx cy w h conf", normalise : on
denormalise et on decale de la demi-largeur.

Une image sans aucune detection ne produit pas de .txt cote
Ultralytics ; elle recoit ici un JSON avec detections vide,
pour que le compte d'images reste complet.

Sortie : ../sku110k-audit/predictions/*.json

Usage, depuis SKU110K_fixed/ :
    python sku_pred_to_json.py
"""

import csv
import glob
import json
import os

ANN_CSV = 'annotations/annotations_test.csv'
PRED_LABELS = 'yolo/predict_test/labels'
OUT_DIR = '../sku110k-audit/predictions'

CLASS_NAME = 'Product'


def load_dims():
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
    if not os.path.isdir(PRED_LABELS):
        raise SystemExit('%s introuvable.' % PRED_LABELS)

    dims = load_dims()
    os.makedirs(OUT_DIR, exist_ok=True)

    with_pred = set()
    total_boxes = 0
    ecrits = 0

    for path in sorted(glob.glob(os.path.join(PRED_LABELS, '*.txt'))):
        stem = os.path.splitext(os.path.basename(path))[0]
        fname = stem + '.jpg'
        if fname not in dims:
            continue
        W, H = dims[fname]
        with_pred.add(fname)

        detections = []
        for line in open(path, encoding='utf-8'):
            parts = line.split()
            if len(parts) < 5:
                continue
            cx, cy, nw, nh = (float(v) for v in parts[1:5])
            conf = float(parts[5]) if len(parts) >= 6 else None
            w = nw * W
            h = nh * H
            det = {
                'class': CLASS_NAME,
                'bbox': {
                    'x': (cx * W) - w / 2.0,
                    'y': (cy * H) - h / 2.0,
                    'width': w,
                    'height': h,
                },
            }
            if conf is not None:
                det['confidence'] = round(conf, 4)
            detections.append(det)

        total_boxes += len(detections)

        with open(os.path.join(OUT_DIR, stem + '.json'), 'w', encoding='utf-8') as f:
            json.dump({
                'success': True,
                'image': {'width': W, 'height': H, 'filename': fname},
                'detections': detections,
            }, f)
        ecrits += 1

    # Images du test sans aucune detection : JSON vide, pour ne pas
    # les faire disparaitre du tableau.
    vides = []
    for fname, (W, H) in sorted(dims.items()):
        if fname in with_pred:
            continue
        stem = os.path.splitext(fname)[0]
        with open(os.path.join(OUT_DIR, stem + '.json'), 'w', encoding='utf-8') as f:
            json.dump({
                'success': True,
                'image': {'width': W, 'height': H, 'filename': fname},
                'detections': [],
            }, f)
        vides.append(fname)
        ecrits += 1

    print('JSON ecrits        : %d' % ecrits)
    print('avec detections    : %d' % len(with_pred))
    print('sans detection     : %d' % len(vides))
    print('boites predites    : %d' % total_boxes)
    if vides:
        print('\nImages du test ou le modele n a rien detecte :')
        for v in vides:
            print('  ' + v)
    assert ecrits == 2936, 'attendu 2936 JSON, obtenu %d' % ecrits
    print('\nControle passe : 2936 JSON ecrits.')
    print('sortie : %s' % OUT_DIR)


if __name__ == '__main__':
    main()