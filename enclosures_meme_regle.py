#!/usr/bin/env python3
"""
La meme definition d'enclosure sur les trois jeux de labels.

enclosed_boxes() est importee de baselines.py : une seule
definition, un seul seuil, aucune copie locale.

Usage, depuis sku110k-audit/ :
    python enclosures_meme_regle.py
"""

import json
import os

from baselines import load_coco, enclosed_boxes, ENCLOSE_RATIO

DATASETS = [
    ('SKU-110K, test', 'test/_annotations.coco.json'),
    ('Retail Shelf, test',
     os.path.expanduser('~/Desktop/retail-audit/test/_annotations.coco.json')),
    ('Retail Shelf, Carol',
     'retail-audit-carol-labels.json'),
]


def main():
    print('Definition unique, importee de baselines.py :')
    print('une boite est ENCLOSE si au moins %d %% de son aire tombe'
          % (ENCLOSE_RATIO * 100))
    print('dans une AUTRE boite plus grande de la meme image.')
    print('Chaque boite est comptee une seule fois.')
    print()
    print('%-22s %10s %10s %8s %9s'
          % ('dataset', 'boites', 'encloses', '%', 'images'))
    print('-' * 64)

    for nom, path in DATASETS:
        if not os.path.exists(path):
            print('%-22s   introuvable : %s' % (nom, path))
            continue
        data = load_coco(path)[0]
        tot = enc = img_enc = 0
        for boxes in data.values():
            e = len(enclosed_boxes(boxes))
            tot += len(boxes)
            enc += e
            if e:
                img_enc += 1
        print('%-22s %10d %10d %7.2f %% %5d/%d'
              % (nom, tot, enc, 100.0 * enc / tot if tot else 0, img_enc, len(data)))

    print('-' * 64)


if __name__ == '__main__':
    main()
