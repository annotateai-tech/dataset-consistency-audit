#!/usr/bin/env python3
"""
SKU-110K : desaccords par image, geometrie separee.

match(), iou() et load_coco() sont importees de baselines.py, jamais
reecrites. pair_geometry() est une regle de SECOND niveau, definie
ici : elle s'applique aux listes que match() a deja produites et ne
peut pas modifier un appariement.

Une paire est dite "geometrie" si un label sans prediction et une
prediction sans label se recouvrent avec un IoU dans [0.20, 0.50[,
et si chacun est le meilleur recouvrement de l'autre. C'est le meme
objet dessine deux fois : pas un desaccord sur ce qu'est un objet,
un desaccord sur ou s'arrete sa boite.

Seuil bas choisi une fois, avant de lire les chiffres : 0.20.

Sortie :
    desaccords_par_image.csv
    et un resume a l'ecran

Usage, depuis sku110k-audit/ :
    python sku110k_audit.py
"""

import csv
import glob
import json
import os
import sys
import time

from baselines import iou, match, load_coco, IOU_MATCH

COCO = 'test/_annotations.coco.json'
PREDS = 'predictions'
OUT_CSV = 'desaccords_par_image.csv'
READ_SAMPLE = 'sku110k_test_read_sample.txt'

GEOM_LO = 0.20


def pair_geometry(gt_only, pred_only, lo=GEOM_LO, hi=IOU_MATCH):
    """Paires 'meme objet, boxe autrement'. Appariement mutuel."""
    if not gt_only or not pred_only:
        return []

    best_g = {}
    for gi, g in enumerate(gt_only):
        b, bi = 0.0, -1
        for pi, p in enumerate(pred_only):
            s = iou(g, p)
            if s > b:
                b, bi = s, pi
        if lo <= b < hi:
            best_g[gi] = bi

    best_p = {}
    for pi, p in enumerate(pred_only):
        b, bi = 0.0, -1
        for gi, g in enumerate(gt_only):
            s = iou(g, p)
            if s > b:
                b, bi = s, gi
        if lo <= b < hi:
            best_p[pi] = bi

    return [(gi, pi) for gi, pi in best_g.items() if best_p.get(pi) == gi]


def main():
    if not os.path.exists(COCO):
        raise SystemExit('Introuvable : ' + COCO)

    print('Lecture de la verite terrain...')
    gt_all, sizes = load_coco(COCO)
    print('  %d images, %d boites' % (len(gt_all), sum(len(v) for v in gt_all.values())))

    pred_files = sorted(glob.glob(os.path.join(PREDS, '*.json')))
    print('  %d fichiers de prediction' % len(pred_files))
    print()
    print('Appariement : match() importee, IoU %.2f' % IOU_MATCH)
    print('Geometrie   : paires mutuelles, IoU dans [%.2f, %.2f[' % (GEOM_LO, IOU_MATCH))
    print()

    rows = []
    t0 = time.time()
    total_gt = total_pred = total_matched = 0

    for n, pf in enumerate(pred_files, 1):
        stem = os.path.basename(pf)[:-5]
        fname = stem + '.jpg'

        raw = json.load(open(pf, encoding='utf-8'))
        pred = [
            (d['bbox']['x'], d['bbox']['y'], d['bbox']['width'], d['bbox']['height'])
            for d in raw.get('detections', [])
        ]
        W, H = raw['image']['width'], raw['image']['height']
        gt = gt_all.get(fname, [])

        matched_gt, used_pred = match(gt, pred, IOU_MATCH)

        gt_only = [g for i, g in enumerate(gt) if i not in matched_gt]
        pred_only = [p for i, p in enumerate(pred) if i not in used_pred]

        n_geom = len(pair_geometry(gt_only, pred_only))
        vrais_gt = len(gt_only) - n_geom
        vrais_pred = len(pred_only) - n_geom

        total_gt += len(gt)
        total_pred += len(pred)
        total_matched += len(matched_gt)

        rows.append({
            'image': fname,
            'width': W,
            'height': H,
            'gt': len(gt),
            'pred': len(pred),
            'matched': len(matched_gt),
            'gt_only': len(gt_only),
            'pred_only': len(pred_only),
            'geometrie': n_geom,
            'label_sans_pred': vrais_gt,
            'pred_sans_label': vrais_pred,
            'vrais_desaccords': vrais_gt + vrais_pred,
            'desaccords_bruts': len(gt_only) + len(pred_only),
        })

        if n % 200 == 0 or n == len(pred_files):
            sys.stdout.write('\r  %d / %d   %.0fs' % (n, len(pred_files), time.time() - t0))
            sys.stdout.flush()

    print('\n')

    with open(OUT_CSV, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    def s(k):
        return sum(r[k] for r in rows)

    bruts, geom, vrais = s('desaccords_bruts'), s('geometrie'), s('vrais_desaccords')

    print('=' * 62)
    print('SKU-110K, split test complet')
    print('=' * 62)
    print('images                          %10d' % len(rows))
    print('boites verite terrain           %10d' % total_gt)
    print('boites predites                 %10d' % total_pred)
    print('appariees (IoU >= %.2f)         %10d' % (IOU_MATCH, total_matched))
    print('-' * 62)
    print('desaccords bruts                %10d' % bruts)
    print('  paires geometrie              %10d' % geom)
    print('  boites concernees             %10d   (%.1f%% des bruts)'
          % (geom * 2, 100.0 * geom * 2 / bruts if bruts else 0))
    print('-' * 62)
    print('VRAIS DESACCORDS                %10d' % vrais)
    print('  labels sans prediction        %10d   (%.2f%% du GT)'
          % (s('label_sans_pred'), 100.0 * s('label_sans_pred') / total_gt))
    print('  predictions sans label        %10d   (%.2f%% des pred)'
          % (s('pred_sans_label'), 100.0 * s('pred_sans_label') / total_pred))
    r = (s('pred_sans_label') / s('label_sans_pred')) if s('label_sans_pred') else 0
    print('  ratio pred/label              %10.2f' % r)
    print('-' * 62)
    print('moyenne vrais par image         %10.1f' % (vrais / len(rows)))
    med = sorted(x['vrais_desaccords'] for x in rows)[len(rows) // 2]
    print('mediane vrais par image         %10d' % med)
    print('=' * 62)

    if os.path.exists(READ_SAMPLE):
        sample = set(l.strip() for l in open(READ_SAMPLE, encoding='utf-8') if l.strip())
        sub = sorted([r for r in rows if r['image'] in sample], key=lambda r: r['image'])
        if sub:
            print('\nLes vingt images tirees a l avance :')
            print('  %-16s %5s %5s %6s %6s %7s %7s'
                  % ('image', 'gt', 'pred', 'brut', 'geom', 'l_sans', 'p_sans'))
            for r in sub:
                print('  %-16s %5d %5d %6d %6d %7d %7d'
                      % (r['image'], r['gt'], r['pred'], r['desaccords_bruts'],
                         r['geometrie'], r['label_sans_pred'], r['pred_sans_label']))
            print('  %-16s %5d %5d %6d %6d %7d %7d'
                  % ('TOTAL',
                     sum(r['gt'] for r in sub), sum(r['pred'] for r in sub),
                     sum(r['desaccords_bruts'] for r in sub),
                     sum(r['geometrie'] for r in sub),
                     sum(r['label_sans_pred'] for r in sub),
                     sum(r['pred_sans_label'] for r in sub)))

    print('\nDetail complet : %s' % OUT_CSV)


if __name__ == '__main__':
    main()