#!/usr/bin/env python3
"""
La meme regle d'enclosure sur les deux datasets.

Retail Shelf avait ete mesure avec count_convention.py : contenant
defini par une aire au-dessus de la mediane, inclusion par le centre.
SKU-110K a ete mesure avec un recouvrement a 90 %. Deux regles, deux
chiffres qui ne vont pas dans la meme phrase.

Ce script applique la regle des 90 % aux deux, pour que la comparaison
soit publiable.

Usage, depuis sku110k-audit/ :
    python enclosures_meme_regle.py
"""

import json
import os

ENCLOSE = 0.90

DATASETS = [
    ('SKU-110K, test',   'test/_annotations.coco.json'),
    ('Retail Shelf, test', os.path.expanduser('~/Desktop/retail-audit/test/_annotations.coco.json')),
    ('Retail Shelf, Carol', os.path.expanduser('~/Desktop/retail-audit-carol/test/_annotations.coco.json')),
]


def load(path):
    d = json.load(open(path, encoding='utf-8'))
    names = {im['id']: im['file_name'] for im in d['images']}
    out = {}
    for a in d['annotations']:
        n = names.get(a['image_id'])
        if n:
            out.setdefault(n, []).append(tuple(a['bbox']))
    return out


def centre(b):
    return b[0] + b[2] / 2.0, b[1] + b[3] / 2.0


def inside(pt, b):
    x, y = pt
    return b[0] <= x <= b[0] + b[2] and b[1] <= y <= b[1] + b[3]


def count_enclosures(boxes):
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
            if not inside(centre(bj), bi):
                continue
            ix0, iy0 = max(bi[0], bj[0]), max(bi[1], bj[1])
            ix1 = min(bi[0] + bi[2], bj[0] + bj[2])
            iy1 = min(bi[1] + bi[3], bj[1] + bj[3])
            inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
            if areas[j] > 0 and inter / areas[j] >= ENCLOSE:
                enclosing.add(i)
                break
    return len(enclosing)


def main():
    print('Regle unique : une boite est dite enclosante si elle contient')
    print('le centre d une plus petite ET couvre >= %d %% de son aire.' % (ENCLOSE * 100))
    print()
    print('%-22s %10s %10s %9s %8s' % ('dataset', 'boites', 'enclosantes', '%', 'images'))
    print('-' * 64)

    for nom, path in DATASETS:
        if not os.path.exists(path):
            print('%-22s   introuvable : %s' % (nom, path))
            continue
        data = load(path)
        tot = enc = img_enc = 0
        for boxes in data.values():
            e = count_enclosures(boxes)
            tot += len(boxes)
            enc += e
            if e:
                img_enc += 1
        print('%-22s %10d %10d %8.2f %% %6d/%d'
              % (nom, tot, enc, 100.0 * enc / tot if tot else 0, img_enc, len(data)))

    print('-' * 64)


if __name__ == '__main__':
    main()