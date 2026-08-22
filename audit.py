#!/usr/bin/env python3
"""
dataset-consistency-audit

Train a small detector on a dataset's own training split, then measure where
its predictions disagree with the dataset's own test labels. Clustered
disagreements usually point at annotation rules that were never written down.

The model is a probe, not a judge. Its absolute quality is not the point.

Usage
-----
    pip install ultralytics pillow

    # 1. train a probe on the dataset's own training split
    yolo detect train data=path/to/data.yaml model=yolo11s.pt \\
        epochs=200 imgsz=1280 batch=2 patience=0

    # 2. audit
    python audit.py \\
        --weights runs/detect/train/weights/best.pt \\
        --images  path/to/test/images \\
        --labels  path/to/test/labels \\
        --format  yolo

    # COCO datasets (Roboflow default) put images and one json together:
    python audit.py \\
        --weights best.pt \\
        --images  test \\
        --labels  test/_annotations.coco.json \\
        --format  coco

Output
------
    A per-image table: their boxes, model boxes, matched, difference.
    Overall recall, precision and F1 against the published labels.
    Overlays in ./overlay/ — red = published labels, green = model.

What to do with it
------------------
    Sort the disagreements by image and look at where they cluster. Read
    them one by one, looking for rules rather than errors. A cluster on one
    product type is a granularity rule nobody wrote down. A cluster at the
    back of shelves is an occlusion threshold. A cluster on rotated objects
    is a definition that depends on pose.

MIT licence.
"""

import os
import glob
import json
import argparse
from PIL import Image, ImageDraw

Image.MAX_IMAGE_PIXELS = None

RED = (255, 60, 60)
GREEN = (0, 230, 118)


def iou(a, b):
    """Boxes as (x, y, w, h) in pixels."""
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


def load_yolo(labels_dir, stem, W, H):
    path = os.path.join(labels_dir, stem + '.txt')
    boxes = []
    if not os.path.exists(path):
        return boxes
    for line in open(path):
        parts = line.split()
        if len(parts) < 5:
            continue
        cx, cy, w, h = [float(v) for v in parts[1:5]]
        boxes.append(((cx - w / 2) * W, (cy - h / 2) * H, w * W, h * H))
    return boxes


def load_coco(coco_path):
    """Returns {filename: [(x, y, w, h), ...]}."""
    d = json.load(open(coco_path, encoding='utf-8'))
    names = {im['id']: im['file_name'] for im in d['images']}
    out = {}
    for a in d['annotations']:
        name = names.get(a['image_id'])
        if name:
            out.setdefault(name, []).append(tuple(a['bbox']))
    return out


def match(gt, pred, threshold):
    """Greedy assignment. Returns the number of matched pairs."""
    used = set()
    matched = 0
    for g in gt:
        best, best_i = 0.0, -1
        for i, p in enumerate(pred):
            if i in used:
                continue
            s = iou(g, p)
            if s > best:
                best, best_i = s, i
        if best >= threshold:
            used.add(best_i)
            matched += 1
    return matched


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--weights', required=True, help='trained .pt file')
    ap.add_argument('--images', required=True, help='directory of test images')
    ap.add_argument('--labels', required=True,
                    help='YOLO labels directory, or a COCO json file')
    ap.add_argument('--format', choices=['yolo', 'coco'], default='yolo')
    ap.add_argument('--conf', type=float, default=0.25)
    ap.add_argument('--iou', type=float, default=0.5, help='matching threshold')
    ap.add_argument('--imgsz', type=int, default=1280)
    ap.add_argument('--max-det', type=int, default=1000)
    ap.add_argument('--out', default='overlay')
    args = ap.parse_args()

    try:
        from ultralytics import YOLO
    except ImportError:
        print("ultralytics is missing. Run: pip install ultralytics")
        return

    model = YOLO(args.weights)
    os.makedirs(args.out, exist_ok=True)

    coco = load_coco(args.labels) if args.format == 'coco' else None

    images = []
    for ext in ('*.jpg', '*.jpeg', '*.png'):
        images.extend(glob.glob(os.path.join(args.images, ext)))
    images.sort()

    if not images:
        print("No images found in " + args.images)
        return

    print()
    print("{:<24} {:>7} {:>7} {:>7} {:>7}".format(
        'IMAGE', 'THEIRS', 'MODEL', 'MATCH', 'DIFF'))
    print('-' * 58)

    tot_gt = tot_pred = tot_match = 0

    for path in images:
        fname = os.path.basename(path)
        stem = os.path.splitext(fname)[0]

        img = Image.open(path).convert('RGB')
        W, H = img.size

        if coco is not None:
            gt = coco.get(fname, [])
        else:
            gt = load_yolo(args.labels, stem, W, H)

        r = model.predict(path, conf=args.conf, imgsz=args.imgsz,
                          max_det=args.max_det, verbose=False)[0]
        pred = []
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            pred.append((x1, y1, x2 - x1, y2 - y1))

        matched = match(gt, pred, args.iou)

        draw = ImageDraw.Draw(img)
        sw = max(2, int(min(W, H) / 500))
        for b in gt:
            draw.rectangle([b[0], b[1], b[0] + b[2], b[1] + b[3]],
                           outline=RED, width=sw)
        for b in pred:
            draw.rectangle([b[0], b[1], b[0] + b[2], b[1] + b[3]],
                           outline=GREEN, width=sw)

        out_img = img.copy()
        out_img.thumbnail((1800, 1800), Image.LANCZOS)
        out_name = '{}_theirs{}_model{}.jpg'.format(stem[:24], len(gt), len(pred))
        out_img.save(os.path.join(args.out, out_name), quality=88, optimize=True)

        print("{:<24} {:>7} {:>7} {:>7} {:>+7}".format(
            stem[:22], len(gt), len(pred), matched, len(pred) - len(gt)))

        tot_gt += len(gt)
        tot_pred += len(pred)
        tot_match += matched

    print('-' * 58)
    print("{:<24} {:>7} {:>7} {:>7}".format('TOTAL', tot_gt, tot_pred, tot_match))
    print()

    precision = tot_match / tot_pred if tot_pred else 0
    recall = tot_match / tot_gt if tot_gt else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0

    print("Recall on their labels : {:.1f} %".format(recall * 100))
    print("Precision vs theirs    : {:.1f} %".format(precision * 100))
    print("Agreement (F1)         : {:.1f} %".format(f1 * 100))
    print()
    print("The diff column is the wrong number. The match column is the right one.")
    print("Overlays in ./{}/ — red = published labels, green = model.".format(args.out))
    print()


if __name__ == '__main__':
    main()