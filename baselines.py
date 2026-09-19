#!/usr/bin/env python3
"""
Compte les desaccords par cluster, sur les donnees actuelles.

Ces chiffres sont les BASELINES du pre-enregistrement. Ils sont geles
avant la livraison de Carol. Les seuils absolus en decoulent
mecaniquement : baseline / 2, arrondi.

Regions definies en pourcentage : (gauche, haut, droite, bas)
Regle d'attribution : un desaccord appartient a un cluster si le CENTRE
de sa boite tombe dans la region.

Usage:
    python baselines.py
"""

import os
import json
import glob

# ---- REGIONS, GELEES LE 25 AOUT 2026 ----
# stem partiel -> {nom_cluster: (l, t, r, b) en %}
CLUSTERS = {
    '029': {
        'kiwi_punnets':  (34,  2, 76, 16),
        'berry_punnets': (20, 19, 50, 36),
    },
    '032': {
        'bottles_standing': (20,  0, 78, 62),
        'bottles_lying':    (20, 63, 62, 98),
    },
    '019': {
        'fridge_depth':    (0,  25, 100, 88),
        'fp_sign':         (44,  0,  60,   6),
        'fp_dhl_and_text': (64, 16,  97,  28),
        'fp_floor':        (58, 91,  92, 100),
    },
    '007': {
        'citrus_mesh_bags': (55, 55, 100, 100),
    },
}

# Faux positifs indiscutables, comptes a part (region 019 signage)
SCATTERED = [('019', 'fp_sign'), ('019', 'fp_dhl_and_text'), ('019', 'fp_floor')]

IOU_MATCH = 0.5


def iou(a, b):
    ax0, ay0, ax1, ay1 = a[0], a[1], a[0] + a[2], a[1] + a[3]
    bx0, by0, bx1, by1 = b[0], b[1], b[0] + b[2], b[1] + b[3]
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0, ix1 - ix0), max(0, iy1 - iy0)
    inter = iw * ih
    if inter == 0:
        return 0.0
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


def match(gt, pred, thr):
    """Appariement glouton, dans l'ordre d'iteration de gt.

    Cet ordre EST la regle. Ne pas l'optimiser : un appariement
    optimal global donnerait d'autres chiffres, et les baselines
    du 25 aout 2026 ont ete produites avec celui-ci.

    Renvoie (matched_gt, used_pred), deux ensembles d'indices.
    """
    used_pred = set()
    matched_gt = set()
    for gi, g in enumerate(gt):
        best, bi = 0.0, -1
        for pi, p in enumerate(pred):
            if pi in used_pred:
                continue
            s = iou(g, p)
            if s > best:
                best, bi = s, pi
        if best >= thr:
            used_pred.add(bi)
            matched_gt.add(gi)
    return matched_gt, used_pred


ENCLOSE_RATIO = 0.90


def enclosed_boxes(boxes, ratio=ENCLOSE_RATIO):
    """Indices des boites ENCLOSES : celles dont au moins `ratio` de
    l'aire tombe dans une autre boite de la meme image, plus grande.

    Definition unique, appelee par tous les scripts. On compte la
    boite en trop, pas la boite qui contient : un lot de quatre
    unites compte quatre, ce qui est ce que le GT a reellement pose.
    """
    n = len(boxes)
    if n < 2:
        return set()
    areas = [b[2] * b[3] for b in boxes]
    out = set()
    for j in range(n):
        if areas[j] <= 0:
            continue
        bj = boxes[j]
        cx, cy = bj[0] + bj[2] / 2.0, bj[1] + bj[3] / 2.0
        for i in range(n):
            if i == j or areas[i] <= areas[j]:
                continue
            bi = boxes[i]
            if not (bi[0] <= cx <= bi[0] + bi[2] and bi[1] <= cy <= bi[1] + bi[3]):
                continue
            ix = max(0.0, min(bi[0] + bi[2], bj[0] + bj[2]) - max(bi[0], bj[0]))
            iy = max(0.0, min(bi[1] + bi[3], bj[1] + bj[3]) - max(bi[1], bj[1]))
            if ix * iy / areas[j] >= ratio:
                out.add(j)
                break
    return out


def centre_in(box, region, W, H):
    """box = (x, y, w, h) en pixels ; region = (l,t,r,b) en %"""
    cx = box[0] + box[2] / 2
    cy = box[1] + box[3] / 2
    l = W * region[0] / 100
    t = H * region[1] / 100
    r = W * region[2] / 100
    b = H * region[3] / 100
    return l <= cx <= r and t <= cy <= b


def load_coco(path):
    d = json.load(open(path, encoding='utf-8'))
    names = {im['id']: im['file_name'] for im in d['images']}
    sizes = {im['file_name']: (im['width'], im['height']) for im in d['images']}
    out = {}
    for a in d['annotations']:
        n = names.get(a['image_id'])
        if n:
            out.setdefault(n, []).append(tuple(a['bbox']))
    return out, sizes


def main():
    coco_path = 'test/_annotations.coco.json'
    if not os.path.exists(coco_path):
        print('Introuvable : ' + coco_path)
        return

    gt_all, sizes = load_coco(coco_path)

    print()
    print('BASELINES PAR CLUSTER — geles le 25 aout 2026')
    print('Regle : un desaccord appartient au cluster si le CENTRE')
    print('de sa boite tombe dans la region.')
    print()
    print('{:<22} {:>10} {:>10} {:>10} {:>9}'.format(
        'CLUSTER', 'GT_ONLY', 'PRED_ONLY', 'TOTAL', 'SEUIL'))
    print('-' * 66)

    grand_total = 0

    for pred_file in sorted(glob.glob('predictions/*.json')):
        stem = os.path.basename(pred_file).replace('.json', '')
        key = stem[:3]
        if key not in CLUSTERS:
            continue

        raw = json.load(open(pred_file, encoding='utf-8'))
        pred = [
            (d['bbox']['x'], d['bbox']['y'], d['bbox']['width'], d['bbox']['height'])
            for d in raw.get('detections', [])
        ]
        W = raw['image']['width']
        H = raw['image']['height']

        # retrouver le fichier image correspondant dans le COCO
        fname = None
        for candidate in gt_all:
            if candidate.startswith(key):
                fname = candidate
                break
        gt = gt_all.get(fname, [])

        matched_gt, used_pred = match(gt, pred, IOU_MATCH)

        gt_only = [g for i, g in enumerate(gt) if i not in matched_gt]
        pred_only = [p for i, p in enumerate(pred) if i not in used_pred]

        for cname, region in CLUSTERS[key].items():
            a = sum(1 for b in gt_only if centre_in(b, region, W, H))
            c = sum(1 for b in pred_only if centre_in(b, region, W, H))
            total = a + c
            grand_total += total

            if (key, cname) in SCATTERED:
                seuil = 'pm 2'
            else:
                seuil = '<= ' + str(total // 2)

            print('{:<22} {:>10} {:>10} {:>10} {:>9}'.format(
                key + '/' + cname, a, c, total, seuil))

    print('-' * 66)
    print('{:<22} {:>32}'.format('TOTAL CLUSTERS', grand_total))
    print()
    print('Lecture : le pre-enregistrement dit "drop by at least half"')
    print('pour les clusters couverts, et "move by less than +/-2" pour')
    print('les erreurs dispersees. La colonne SEUIL en est la traduction')
    print('mecanique. Tout seuil qui ne se deduit pas du texte enregistre')
    print('est une deviation et se publie comme telle.')
    print()


if __name__ == '__main__':
    main()