#!/usr/bin/env python3
"""Run the official scrollprize/ink_9um (hybrid_3d2d) model on a (planes,H,W) uint8 render.

Uses the pinned Villa implementation (make_model / normalize_flat_patch /
logits_to_probabilities), 17 central planes, 128x128 tiles, Hann-weighted
blending. Writes a PNG with value = round(255*p) (same convention as Hecate).

  python ink9um_infer.py render.npy out.png --checkpoint step-075000.pth [--reverse] [--stride 64]
"""
import argparse, contextlib, io
import numpy as np, torch
from PIL import Image
from vesuvius.ink_detection.models.checkpoint import config_from_checkpoint, select_inference_weights
from vesuvius.ink_detection.models.model import make_model
from vesuvius.ink_detection.inference.infer import normalize_flat_patch, compute_importance_map_2d, logits_to_probabilities

ap = argparse.ArgumentParser()
ap.add_argument('render'); ap.add_argument('out')
ap.add_argument('--checkpoint', required=True)
ap.add_argument('--reverse', action='store_true')
ap.add_argument('--stride', type=int, default=64)
ap.add_argument('--batch', type=int, default=64)
ap.add_argument('--device', default='cuda')
a = ap.parse_args()

payload = torch.load(a.checkpoint, map_location='cpu', weights_only=True)
cfg = config_from_checkpoint(payload)
_, state = select_inference_weights(payload)
with contextlib.redirect_stdout(io.StringIO()):
    model = make_model(cfg)
model.load_state_dict(state, strict=True)
model.eval().to(a.device)

vol = np.load(a.render, mmap_mode='r')
c = vol.shape[0] // 2
idx = np.arange(c - 8, c + 9)          # 17 central planes
if a.reverse:
    idx = idx[::-1]
valid = np.asarray(vol[c]) > 0
H, W = valid.shape
win = compute_importance_map_2d(patch_size=(128, 128), mode='hann').numpy()
S = np.zeros((H, W), np.float32); Wt = np.zeros((H, W), np.float32)
ys = list(range(0, max(H - 128, 0) + 1, a.stride)); xs = list(range(0, max(W - 128, 0) + 1, a.stride))
if ys[-1] != max(H - 128, 0): ys.append(max(H - 128, 0))
if xs[-1] != max(W - 128, 0): xs.append(max(W - 128, 0))
tiles = [(y, x) for y in ys for x in xs if valid[y:y + 128, x:x + 128].mean() > 0.5]
print('tiles', len(tiles), flush=True)
for i in range(0, len(tiles), a.batch):
    chunk = tiles[i:i + a.batch]
    batch = np.stack([normalize_flat_patch(np.ascontiguousarray(vol[idx, y:y + 128, x:x + 128]), 'tifxyz_robust') for y, x in chunk])
    with torch.inference_mode():
        lg = model(torch.from_numpy(batch)[:, None].to(a.device))['ink']
        pr = logits_to_probabilities(lg, image_hw=(128, 128))[:, 0].float().cpu().numpy()
    for (y, x), p in zip(chunk, pr):
        S[y:y + 128, x:x + 128] += p * win; Wt[y:y + 128, x:x + 128] += win
P = np.where(Wt > 1e-3, S / np.maximum(Wt, 1e-6), 0)
P[~valid] = 0
Image.fromarray(np.uint8(np.rint(np.clip(P, 0, 1) * 255))).save(a.out)
print('done', a.out, flush=True)
