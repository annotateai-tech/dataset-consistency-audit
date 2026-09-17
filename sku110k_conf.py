#!/usr/bin/env python3
"""
SKU-110K : distribution de confiance des verts.

Un vert est une prediction que le matching n'a pas appariee.
A conf=0.25, une part est du bruit de bord de decision ; au-dessus
de 0,6, chacun merite le regard. Ce script dit combien il y en a
dans chaque tranche, AVANT de les regarder un par un.

L'appariement est importe de baselines.py, jamais recalcule.

Usage, depuis sku110k-audit/ :
    python sku110k_conf.py                  # les 20 images tirees
    python sku110k_conf.py --tout           # les 2936 images
    python sku110k_conf.py test_1053.jpg
"""

import glob
import json
import os
import sys

from baselines import match, load_coco, IOU_MATCH

COCO = 'test/_annotations.coco.json'
PREDS = 'predictions'
READ_SAMPLE = 'sku110k_test_read_sample.txt'

BINS = [(0.25, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.01)]


def bin_of(c):
    for i, (lo, hi) in enumerate(BINS):
        if lo <= c < hi:
            return i
    return len(BINS) - 1


def main():
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    tout = '--tout' in sys.argv

    gt_all, _ = load_coco(COCO)

    if args:
        names = args
    elif tout:
        names = [os.path.basename(p)[:-5] + '.jpg'
                 for p in sorted(glob.glob(os.path.join(PREDS, '*.json')))]
    else:
        names = [l.strip() for l in open(READ_SAMPLE, encoding='utf-8') if l.strip()]

    head = '%-16s %6s %6s %8s' % ('image', 'gt', 'pred', 'apparies')
    for lo, hi in BINS:
        head += ' %10s' % ('%.2f-%.2f' % (lo, hi))
    head += ' %8s' % 'verts'
    print(head)
    print('-' * len(head))

    tot = [0] * len(BINS)
    tot_verts = tot_app = tot_gt = tot_pred = 0

    for fname in names:
        stem = os.path.splitext(fname)[0]
        pf = os.path.join(PREDS, stem + '.json')
        if not os.path.isfile(pf):
            continue

        raw = json.load(open(pf, encoding='utf-8'))
        dets = raw.get('detections', [])
        pred = [(d['bbox']['x'], d['bbox']['y'], d['bbox']['width'], d['bbox']['height'])
                for d in dets]
        confs = [d.get('confidence', 0.0) for d in dets]
        gt = gt_all.get(fname, [])

        matched_gt, used_pred = match(gt, pred, IOU_MATCH)

        counts = [0] * len(BINS)
        for i, c in enumerate(confs):
            if i not in used_pred:
                counts[bin_of(c)] += 1

        verts = sum(counts)
        line = '%-16s %6d %6d %8d' % (fname, len(gt), len(pred), len(matched_gt))
        for c in counts:
            line += ' %10d' % c
        line += ' %8d' % verts
        print(line)

        for i, c in enumerate(counts):
            tot[i] += c
        tot_verts += verts
        tot_app += len(matched_gt)
        tot_gt += len(gt)
        tot_pred += len(pred)

    print('-' * len(head))
    line = '%-16s %6d %6d %8d' % ('TOTAL', tot_gt, tot_pred, tot_app)
    for c in tot:
        line += ' %10d' % c
    line += ' %8d' % tot_verts
    print(line)

    if tot_verts:
        line = '%-16s %6s %6s %8s' % ('part des verts', '', '', '')
        for c in tot:
            line += ' %9.1f%%' % (100.0 * c / tot_verts)
        print(line)

    print()
    haut = tot[2] + tot[3]
    print('Verts au-dessus de 0,60 : %d sur %d  (%.1f%%)'
          % (haut, tot_verts, 100.0 * haut / tot_verts if tot_verts else 0))
    print('Ce sont ceux a regarder en premier : a ce niveau de confiance,')
    print('le modele affirme voir un produit la ou aucun label n existe.')
    print()
    print('Verts entre 0,25 et 0,40 : %d  (%.1f%%)'
          % (tot[0], 100.0 * tot[0] / tot_verts if tot_verts else 0))
    print('Bruit de bord de decision probable. A compter a part.')


if __name__ == '__main__':
    main()