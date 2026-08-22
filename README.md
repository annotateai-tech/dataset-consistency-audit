# dataset-consistency-audit

Find annotation rules your dataset never wrote down.

Train a small detector on a dataset's **own** training split, then measure
where its predictions disagree with that dataset's **own** test labels.
Scattered disagreements are model noise. Clustered disagreements are usually
a specification that was never written.

The model is a probe, not a judge. Its absolute quality is beside the point —
what matters is *where* it disagrees.

## Install

```bash
pip install ultralytics pillow
```

## Use

Train a probe on the dataset's own training split:

```bash
yolo detect train data=path/to/data.yaml model=yolo11s.pt \
    epochs=200 imgsz=1280 batch=2 patience=0
```

Then audit the test split:

```bash
# YOLO format
python audit.py \
    --weights runs/detect/train/weights/best.pt \
    --images  path/to/test/images \
    --labels  path/to/test/labels \
    --format  yolo

# COCO format (Roboflow's default export)
python audit.py \
    --weights runs/detect/train/weights/best.pt \
    --images  test \
    --labels  test/_annotations.coco.json \
    --format  coco
```

## Output

```
IMAGE                     THEIRS   MODEL   MATCH    DIFF
----------------------------------------------------------
007_jpg.rf.c5aa2ae43         377     392     254     +15
014_jpg.rf.a70c0e851         146     115      65     -31
019_jpg.rf.1ce9a5af2         138     181     114     +43
029_jpg.rf.9757c7c2b         409     330     219     -79
032_jpg.rf.3525a56f5         305     174     146    -131
----------------------------------------------------------
TOTAL                       1375    1192     798

Recall on their labels : 58.0 %
Precision vs theirs    : 66.9 %
Agreement (F1)         : 62.2 %
```

Plus overlays in `./overlay/` — red for the published labels, green for the
model.

## Reading the result

The **diff** column is the wrong number. The **match** column is the right
one. On the first row above, the counts differ by fifteen out of 377 — which
looks like agreement — while only two thirds of the boxes actually match.

Watch the sign of the diff across images. A dataset following one consistent
rule, even a bad one, makes the model drift in a single direction. Drift in
both directions is the signature of a dataset that contradicts itself.

Then read the disagreements one by one, looking for rules rather than errors:

- a cluster on one product type is a **granularity** rule nobody wrote down
  (is a sealed punnet one object, or one per item inside?)
- a cluster at the back of shelves is an **occlusion threshold**
- a cluster on rotated objects is a **definition that depends on pose**
- a cluster on partially visible objects is the **modal/amodal** question:
  do you box the visible fragment or the full extent?

None of these is fixed by hiring better annotators.

## Options

```
--conf     0.25   prediction confidence threshold
--iou      0.5    matching threshold between a label and a prediction
--imgsz    1280   inference size; raise it for small dense objects
--max-det  1000   YOLO defaults to 300, which silently truncates dense scenes
--out      overlay
```

`--max-det` matters more than it looks. At the default of 300, two of the
images above returned exactly 300 predictions — the model wanted more and was
cut off.

## Related

Different from label *error* detection. Confident learning
([Northcutt et al., 2021](https://arxiv.org/abs/1911.00068), and the Cleanlab
line of work) finds labels that are **wrong** — the image says cat, the label
says dog. This finds labels that are **incompatible with each other**: each
one defensible on its own, no two agreeing on what an object is.

## Write-up

A worked example on a public CC BY dataset, with figures:
https://www.annotateai.tech/blog/four-contradictory-rules

## Licence

MIT.