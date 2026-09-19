#!/usr/bin/env python3
"""
SKU-110K : les trois nombres qui tranchent, sans rouvrir une image.

1. Enclosures : combien de boites en contiennent une autre, cote GT
   et cote modele. Sur Retail Shelf, le GT etait a 4,5 %. Si le GT de
   SKU-110K est proche de zero, la regle "une boite par emplacement"
   est appliquee ; si le modele est nettement au-dessus, il ne l'a
   pas apprise.

2. Les predictions sans label : leur centre tombe-t-il dans une boite
   du GT (une unite de plus dans un emplacement deja labellise) ou en
   dehors de tout (un produit hors perimetre) ?

3. Les labels sans prediction : centre dans une boite du modele
   (geometrie ou lot contre unite) ou hors de tout (capacite).

match() est importee de baselines.py, jamais reecrite.

Usage, depuis sku110k-audit/ :
    python sku110k_mesures.py
"""

import glob
import json
import os
import sys
import time

from baselines import iou, match, load_coco, IOU_MATCH, enclosed_boxes, ENCLOSE_RATIO

COCO = 'test/_annotations.coco.json'
PREDS = 'predictions'

# Une boite en enclot une autre si l'aire de l'intersection couvre
# au moins ce ratio de la petite. Seuil choisi avant de lire les
# chiffres.
ENCLOSE = ENCLOSE_RATIO
GEOM_LO = 0.20


def pair_geometry(gt_only, pred_only, lo=GEOM_LO, hi=IOU_MATCH):
    """Meme regle de second niveau que sku110k_audit.py."""
    if not gt_only or not pred_only:
        return set(), set()
    best_g = {}
    for gi, g in enumerate(gt_only):
        b, bi = 0.0, -1
        for pi, p in enumerate(pred_only):
            sc = iou(g, p)
            if sc > b:
                b, bi = sc, pi
        if lo <= b < hi:
            best_g[gi] = bi
    best_p = {}
    for pi, p in enumerate(pred_only):
        b, bi = 0.0, -1
        for gi, g in enumerate(gt_only):
            sc = iou(g, p)
            if sc > b:
                b, bi = sc, gi
        if lo <= b < hi:
            best_p[pi] = bi
    gset, pset = set(), set()
    for gi, pi in best_g.items():
        if best_p.get(pi) == gi:
            gset.add(gi); pset.add(pi)
    return gset, pset


def centre(b):
    return b[0] + b[2] / 2.0, b[1] + b[3] / 2.0


def centre_inside(pt, b):
    x, y = pt
    return b[0] <= x <= b[0] + b[2] and b[1] <= y <= b[1] + b[3]


def _count_enclosures_ancienne(boxes):
    """Nombre de boites qui contiennent le centre d'une autre boite
    plus petite, et dont l'intersection couvre >= ENCLOSE de celle-ci.
    """
    n = len(boxes)
    if n < 2:
        return 0
    areas = [b[2] * b[3] for b in boxes]
    order = sorted(range(n), key=lambda i: -areas[i])
    enclosing = set()
    for ii, i in enumerate(order):
        bi = boxes[i]
        for j in order[ii + 1:]:
            bj = boxes[j]
            if not centre_inside(centre(bj), bi):
                continue
            ix0 = max(bi[0], bj[0])
            iy0 = max(bi[1], bj[1])
            ix1 = min(bi[0] + bi[2], bj[0] + bj[2])
            iy1 = min(bi[1] + bi[3], bj[1] + bj[3])
            inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
            if areas[j] > 0 and inter / areas[j] >= ENCLOSE:
                enclosing.add(i)
                break
    return len(enclosing)


def main():
    gt_all, _ = load_coco(COCO)
    files = sorted(glob.glob(os.path.join(PREDS, '*.json')))

    gt_total = gt_enc = pr_total = pr_enc = 0
    v_dedans = v_dehors = 0
    r_dedans = r_dehors = 0
    images_gt_enc = []
    n_geom = 0

    t0 = time.time()
    for n, pf in enumerate(files, 1):
        fname = os.path.basename(pf)[:-5] + '.jpg'
        raw = json.load(open(pf, encoding='utf-8'))
        pred = [(d['bbox']['x'], d['bbox']['y'], d['bbox']['width'], d['bbox']['height'])
                for d in raw.get('detections', [])]
        gt = gt_all.get(fname, [])

        e = len(enclosed_boxes(gt))
        if e:
            images_gt_enc.append((fname, e, len(gt)))
        gt_enc += e
        gt_total += len(gt)

        pr_enc += len(enclosed_boxes(pred))
        pr_total += len(pred)

        matched_gt, used_pred = match(gt, pred, IOU_MATCH)

        gt_only = [g for i, g in enumerate(gt) if i not in matched_gt]
        pred_only = [p for i, p in enumerate(pred) if i not in used_pred]
        g_geom, p_geom = pair_geometry(gt_only, pred_only)
        n_geom += len(g_geom)

        for i, p in enumerate(pred_only):
            if i in p_geom:
                continue
            c = centre(p)
            if any(centre_inside(c, g) for g in gt):
                v_dedans += 1
            else:
                v_dehors += 1

        for i, g in enumerate(gt_only):
            if i in g_geom:
                continue
            c = centre(g)
            if any(centre_inside(c, p) for p in pred):
                r_dedans += 1
            else:
                r_dehors += 1

        if n % 200 == 0 or n == len(files):
            sys.stdout.write('\r  %d / %d   %.0fs' % (n, len(files), time.time() - t0))
            sys.stdout.flush()
    print('\n')

    print('=' * 60)
    print('1. BOITES ENCLOSES (>= %d%% de leur aire dans une plus grande)' % (ENCLOSE * 100))
    print('=' * 60)
    print('GT      %8d / %8d   %6.2f %%' % (gt_enc, gt_total, 100.0 * gt_enc / gt_total))
    print('modele  %8d / %8d   %6.2f %%' % (pr_enc, pr_total, 100.0 * pr_enc / pr_total))
    print('(comparaison Retail Shelf : voir enclosures_meme_regle.py)')
    print('images du GT avec au moins une enclosure : %d / %d'
          % (len(images_gt_enc), len(files)))
    if images_gt_enc:
        top = sorted(images_gt_enc, key=lambda x: -x[1])[:15]
        print('\nles quinze plus fortes, cote GT :')
        for f, e, t in top:
            print('  %-16s %4d enclosures / %4d boites' % (f, e, t))

    print('\npaires geometrie exclues des mesures 2 et 3 : %d' % n_geom)
    tv = v_dedans + v_dehors
    print()
    print('=' * 60)
    print('2. PREDICTIONS SANS LABEL, geometrie exclue')
    print('=' * 60)
    print('dans une boite du GT    %8d   %5.1f %%   unite de plus dans un emplacement'
          % (v_dedans, 100.0 * v_dedans / tv if tv else 0))
    print('hors de toute boite GT  %8d   %5.1f %%   produit hors perimetre'
          % (v_dehors, 100.0 * v_dehors / tv if tv else 0))
    print('total                   %8d' % tv)

    tr = r_dedans + r_dehors
    print()
    print('=' * 60)
    print('3. LABELS SANS PREDICTION, geometrie exclue')
    print('=' * 60)
    print('dans une boite du modele %7d   %5.1f %%   geometrie, ou lot contre unite'
          % (r_dedans, 100.0 * r_dedans / tr if tr else 0))
    print('hors de toute prediction %7d   %5.1f %%   capacite du modele'
          % (r_dehors, 100.0 * r_dehors / tr if tr else 0))
    print('total                    %7d' % tr)
    print('=' * 60)


if __name__ == '__main__':
    main()