# SKU-110K - conversion CSV vers YOLO

Sous-ensemble train tire avec `random.seed(42)`, liste figee dans
`sku110k_train_subset.txt` et commitee avant toute mesure.

Archive auditee : `SKU110K_fixed.tar.gz`, la version corrigee du dataset,
pas la publication originale de 2019. Les boites degenerees et les doublons
y ont deja ete nettoyes en amont, ce que confirment les compteurs a zero
ci-dessous.

Le split `val` officiel sert a la selection de `best.pt`.
Le split `test` n est jamais lu avant la prediction finale :
c est celui que l audit examine.

Regles appliquees, identiques sur les trois splits :

- coordonnees clippees a `[0, width]` et `[0, height]`
- boite rejetee si `x2 <= x1` ou `y2 <= y1` apres clipping
- boite rejetee si son aire est inferieure a 4 pixels
- normalisation par le `width,height` de la ligne CSV, verifie
  contre les dimensions reelles sur les 40 premieres images
  de chaque split
- les doublons exacts sont comptes mais conserves : ce sont
  les labels du dataset, pas une erreur de conversion

Un filtre different fera bouger les chiffres de l audit : c est
pourquoi les rejets sont comptes ici.

## Train - sous-ensemble de 800

| | |
|---|---|
| images converties | 800 |
| boites lues | 118207 |
| boites gardees | 118207 |
| boites rejetees | 0 (0.000%) |
| dont degenerees (x2<=x1 ou y2<=y1) | 0 |
| dont aire < 4 px | 0 |
| dont dimensions image invalides | 0 |
| boites clippees aux bords | 1423 |
| doublons exacts (meme x1,y1,x2,y2) | 0 |
| images sans annotation | 0 |
| images absentes du disque | 0 |

## Val - split officiel

| | |
|---|---|
| images converties | 588 |
| boites lues | 90968 |
| boites gardees | 90968 |
| boites rejetees | 0 (0.000%) |
| dont degenerees (x2<=x1 ou y2<=y1) | 0 |
| dont aire < 4 px | 0 |
| dont dimensions image invalides | 0 |
| boites clippees aux bords | 1031 |
| doublons exacts (meme x1,y1,x2,y2) | 0 |
| images sans annotation | 0 |
| images absentes du disque | 0 |

## Test - split complet, audite

| | |
|---|---|
| images converties | 2936 |
| boites lues | 431546 |
| boites gardees | 431546 |
| boites rejetees | 0 (0.000%) |
| dont degenerees (x2<=x1 ou y2<=y1) | 0 |
| dont aire < 4 px | 0 |
| dont dimensions image invalides | 0 |
| boites clippees aux bords | 5211 |
| doublons exacts (meme x1,y1,x2,y2) | 0 |
| images sans annotation | 0 |
| images absentes du disque | 0 |
