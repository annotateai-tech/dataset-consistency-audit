#!/usr/bin/env python3
"""
Conversion CSV SKU-110K -> YOLO.

  train : uniquement les 800 images de sku110k_train_subset.txt
  val   : les 588 images du split val officiel (selection de best.pt)
  test  : les 2936 images du split test, jamais touchees avant
          la prediction finale de l'audit

Precautions imposees par SKU-110K, connu pour ses boites sales :
  - coordonnees clippees a [0, width] et [0, height]
  - boites degenerees rejetees : x2 <= x1, y2 <= y1, ou aire quasi nulle
  - les dimensions du CSV sont verifiees contre celles du fichier image
    sur l'echantillon train (assert)

Tout ce qui est rejete est compte et ecrit dans le README du sous-ensemble.

Sortie :
    yolo/images/{train,val,test}/*.jpg   (copies)
    yolo/labels/{train,val,test}/*.txt
    yolo/data.yaml
    yolo/CONVERSION.md

Usage, depuis SKU110K_fixed/ :
    python convertir_yolo.py
"""

import csv
import os
import shutil
from collections import defaultdict, Counter

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

IMAGES = 'images'
ANN = 'annotations'
OUT = 'yolo'
SUBSET_FILE = 'sku110k_train_subset.txt'
MIN_AREA_PX = 4.0   # sous ce seuil, la boite n'est pas annotable


def load_csv(path):
    """image_name -> liste de (x1, y1, x2, y2, w, h)."""
    rows = defaultdict(list)
    with open(path, newline='', encoding='utf-8') as f:
        for r in csv.reader(f):
            if len(r) < 8:
                continue
            name = r[0]
            try:
                x1, y1, x2, y2 = (float(v) for v in r[1:5])
                w, h = int(r[6]), int(r[7])
            except ValueError:
                continue
            rows[name].append((x1, y1, x2, y2, w, h))
    return rows


def convert(rows, names, split, stats):
    img_dir = os.path.join(OUT, 'images', split)
    lbl_dir = os.path.join(OUT, 'labels', split)
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)

    checked_dims = 0

    for name in names:
        src = os.path.join(IMAGES, name)
        if not os.path.isfile(src):
            stats['images_absentes'] += 1
            continue

        boxes = rows.get(name, [])
        if not boxes:
            stats['images_sans_annotation'] += 1

        # Doublons exacts : meme (x1, y1, x2, y2) plusieurs fois sur l'image.
        seen = set()
        for b in boxes:
            key = b[:4]
            if key in seen:
                stats['boites_dupliquees'] += 1
            seen.add(key)

        # Dimensions annoncees par le CSV contre le fichier reel :
        # une seule ouverture par image, sur les 40 premieres du split.
        if HAS_PIL and boxes and checked_dims < 40:
            w_csv, h_csv = boxes[0][4], boxes[0][5]
            with Image.open(src) as im:
                if im.size != (w_csv, h_csv):
                    raise SystemExit(
                        '%s : le CSV annonce %dx%d, le fichier fait %dx%d'
                        % (name, w_csv, h_csv, im.size[0], im.size[1])
                    )
            checked_dims += 1

        lines = []
        for (x1, y1, x2, y2, w, h) in boxes:
            stats['boites_lues'] += 1

            if w <= 0 or h <= 0:
                stats['rejet_dimensions_image'] += 1
                continue

            cx1, cy1 = max(0.0, min(x1, w)), max(0.0, min(y1, h))
            cx2, cy2 = max(0.0, min(x2, w)), max(0.0, min(y2, h))
            if (cx1, cy1, cx2, cy2) != (x1, y1, x2, y2):
                stats['boites_clippees'] += 1

            bw, bh = cx2 - cx1, cy2 - cy1
            if bw <= 0 or bh <= 0:
                stats['rejet_degeneree'] += 1
                continue
            if bw * bh < MIN_AREA_PX:
                stats['rejet_aire_nulle'] += 1
                continue

            cx = (cx1 + cx2) / 2.0 / w
            cy = (cy1 + cy2) / 2.0 / h
            nw, nh = bw / w, bh / h
            lines.append('0 %.6f %.6f %.6f %.6f' % (cx, cy, nw, nh))
            stats['boites_gardees'] += 1

        dst = os.path.join(img_dir, name)
        if not os.path.exists(dst):
            shutil.copy2(src, dst)

        stem = os.path.splitext(name)[0]
        with open(os.path.join(lbl_dir, stem + '.txt'), 'w', encoding='utf-8') as f:
            if lines:
                f.write('\n'.join(lines) + '\n')

        stats['images_converties'] += 1

    return checked_dims


def main():
    if not os.path.isdir(ANN):
        raise SystemExit('Lancer depuis le dossier SKU110K_fixed/')
    if not os.path.isfile(SUBSET_FILE):
        raise SystemExit('%s introuvable : faire le tirage d abord.' % SUBSET_FILE)

    subset = [l.strip() for l in open(SUBSET_FILE, encoding='utf-8') if l.strip()]
    print('Sous-ensemble train : %d images' % len(subset))

    print('Lecture des annotations...')
    train_rows = load_csv(os.path.join(ANN, 'annotations_train.csv'))
    val_rows = load_csv(os.path.join(ANN, 'annotations_val.csv'))
    test_rows = load_csv(os.path.join(ANN, 'annotations_test.csv'))
    print('  train : %d images annotees' % len(train_rows))
    print('  val   : %d images annotees' % len(val_rows))
    print('  test  : %d images annotees' % len(test_rows))

    stats = {'train': Counter(), 'val': Counter(), 'test': Counter()}

    print('\nConversion train (%d)...' % len(subset))
    checked = convert(train_rows, subset, 'train', stats['train'])
    print('  dimensions verifiees sur %d images' % checked)

    print('Conversion val (%d)...' % len(val_rows))
    convert(val_rows, sorted(val_rows.keys()), 'val', stats['val'])

    print('Conversion test (%d)...' % len(test_rows))
    convert(test_rows, sorted(test_rows.keys()), 'test', stats['test'])

    # Chemin absolu : Ultralytics resout 'path' contre son propre datasets_dir.
    with open(os.path.join(OUT, 'data.yaml'), 'w', encoding='utf-8') as f:
        f.write('path: %s\n' % os.path.abspath(OUT).replace('\\', '/'))
        f.write('train: images/train\n')
        f.write('val: images/val\n')
        f.write('test: images/test\n')
        f.write('nc: 1\n')
        f.write("names: ['object']\n")

    def block(title, s_):
        total = s_['boites_lues']
        kept = s_['boites_gardees']
        rejected = total - kept
        pct = (100.0 * rejected / total) if total else 0.0
        out = ['## %s' % title, '']
        out.append('| | |')
        out.append('|---|---|')
        out.append('| images converties | %d |' % s_['images_converties'])
        out.append('| boites lues | %d |' % total)
        out.append('| boites gardees | %d |' % kept)
        out.append('| boites rejetees | %d (%.3f%%) |' % (rejected, pct))
        out.append('| dont degenerees (x2<=x1 ou y2<=y1) | %d |' % s_['rejet_degeneree'])
        out.append('| dont aire < %g px | %d |' % (MIN_AREA_PX, s_['rejet_aire_nulle']))
        out.append('| dont dimensions image invalides | %d |' % s_['rejet_dimensions_image'])
        out.append('| boites clippees aux bords | %d |' % s_['boites_clippees'])
        out.append('| doublons exacts (meme x1,y1,x2,y2) | %d |' % s_['boites_dupliquees'])
        out.append('| images sans annotation | %d |' % s_['images_sans_annotation'])
        out.append('| images absentes du disque | %d |' % s_['images_absentes'])
        out.append('')
        return '\n'.join(out)

    readme = ['# SKU-110K - conversion CSV vers YOLO', '']
    readme.append('Sous-ensemble train tire avec `random.seed(42)`, liste figee dans')
    readme.append('`sku110k_train_subset.txt` et commitee avant toute mesure.')
    readme.append('')
    readme.append('Le split `val` officiel sert a la selection de `best.pt`.')
    readme.append('Le split `test` n est jamais lu avant la prediction finale :')
    readme.append('c est celui que l audit examine.')
    readme.append('')
    readme.append('Regles appliquees, identiques sur les trois splits :')
    readme.append('')
    readme.append('- coordonnees clippees a `[0, width]` et `[0, height]`')
    readme.append('- boite rejetee si `x2 <= x1` ou `y2 <= y1` apres clipping')
    readme.append('- boite rejetee si son aire est inferieure a %g pixels' % MIN_AREA_PX)
    readme.append('- normalisation par le `width,height` de la ligne CSV, verifie')
    readme.append('  contre les dimensions reelles sur les 40 premieres images')
    readme.append('  de chaque split')
    readme.append('- les doublons exacts sont comptes mais conserves : ce sont')
    readme.append('  les labels du dataset, pas une erreur de conversion')
    readme.append('')
    readme.append('Un filtre different fera bouger les chiffres de l audit : c est')
    readme.append('pourquoi les rejets sont comptes ici.')
    readme.append('')
    readme.append(block('Train - sous-ensemble de %d' % len(subset), stats['train']))
    readme.append(block('Val - split officiel', stats['val']))
    readme.append(block('Test - split complet, audite', stats['test']))

    with open(os.path.join(OUT, 'CONVERSION.md'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(readme))

    print()
    print('=' * 60)
    for title in ('train', 'val', 'test'):
        s_ = stats[title]
        total, kept = s_['boites_lues'], s_['boites_gardees']
        print('%-6s lues %-8d gardees %-8d rejetees %d'
              % (title.upper(), total, kept, total - kept))
        print('       degenerees %d, aire nulle %d, clippees %d, doublons %d'
              % (s_['rejet_degeneree'], s_['rejet_aire_nulle'],
                 s_['boites_clippees'], s_['boites_dupliquees']))
    print('=' * 60)
    print('\nDetail : %s/CONVERSION.md' % OUT)


if __name__ == '__main__':
    main()