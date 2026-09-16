#!/usr/bin/env python3
"""
Tirage reproductible des sous-ensembles SKU-110K.

seed 42, fixee avant toute mesure.
  800 images du split train  -> entrainement
   20 images du split test   -> lecture manuelle des overlays

Les deux listes sont ecrites AVANT l'entrainement et AVANT
d'avoir vu le moindre chiffre. C'est ce qui rend l'audit
reproductible et empeche de choisir les images spectaculaires
apres coup.

Usage, depuis SKU110K_fixed/ :
    python tirage_sku110k.py
"""

import os
import random
import glob

SEED = 42
N_TRAIN = 800
N_TEST_READ = 20

IMAGES = 'images'
OUT_TRAIN = 'sku110k_train_subset.txt'
OUT_TEST = 'sku110k_test_read_sample.txt'


def main():
    if not os.path.isdir(IMAGES):
        raise SystemExit('Lancer depuis le dossier SKU110K_fixed/')

    for path in (OUT_TRAIN, OUT_TEST):
        if os.path.exists(path):
            raise SystemExit(
                '%s existe deja. Le tirage ne doit pas etre refait : '
                'le supprimer volontairement si c est intentionnel.' % path
            )

    train = sorted(os.path.basename(p) for p in glob.glob(os.path.join(IMAGES, 'train_*.jpg')))
    test = sorted(os.path.basename(p) for p in glob.glob(os.path.join(IMAGES, 'test_*.jpg')))

    print('train disponibles : %d' % len(train))
    print('test disponibles  : %d' % len(test))

    if len(train) < N_TRAIN or len(test) < N_TEST_READ:
        raise SystemExit('Pas assez d images pour le tirage demande.')

    # Un seul generateur, seed fixee, deux tirages successifs.
    rng = random.Random(SEED)
    subset_train = sorted(rng.sample(train, N_TRAIN))
    subset_test = sorted(rng.sample(test, N_TEST_READ))

    with open(OUT_TRAIN, 'w', encoding='utf-8') as f:
        f.write('\n'.join(subset_train) + '\n')
    with open(OUT_TEST, 'w', encoding='utf-8') as f:
        f.write('\n'.join(subset_test) + '\n')

    print()
    print('seed              : %d' % SEED)
    print('train tire        : %d  -> %s' % (len(subset_train), OUT_TRAIN))
    print('test a lire       : %d  -> %s' % (len(subset_test), OUT_TEST))
    print()
    print('Les 20 images de test a lire, fixees avant toute mesure :')
    for name in subset_test:
        print('  ' + name)
    print()
    print('Commiter ces deux fichiers dans le repo AVANT de convertir')
    print('ou d entrainer quoi que ce soit.')


if __name__ == '__main__':
    main()