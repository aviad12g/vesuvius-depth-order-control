#!/usr/bin/env python3
"""Cross-model agreement diagnostic.

Given probability maps from two independently trained ink models on the SAME
render (same pixel grid), report:

  global_r          Pearson r over valid pixels (minus --edge-margin at the render boundary)
  local_r_p99/max   distribution of Pearson r in sliding windows (default 256 px
                    = 2.46 mm at 9.6 um, i.e. letter scale) that contain some
                    response in both maps
  joint_hot         fraction of valid pixels where BOTH models exceed 0.5

and the same numbers for a depth-shuffled control pair (both models run on the
identical shuffled input). Agreement that is no higher than the shuffled pair's
is not evidence of ink.

usage: cross_model_agreement.py --valid V.npy|V.png \
          --real A_rev.png B_rev.png [--real A_fwd.png B_fwd.png] \
          --shuffled A_shuf.png B_shuf.png --out result.json
"""
import argparse, json
import numpy as np
from PIL import Image
from scipy.ndimage import binary_erosion


def load(p):
    return np.asarray(Image.open(p), np.float32) / 255


def load_valid(p):
    return (np.load(p) if p.endswith('.npy') else np.asarray(Image.open(p)) > 0).astype(bool)


def local_r(a, b, v, win, step, min_active=0.02):
    H, W = a.shape
    out = []
    for y in range(0, max(H - win, 0) + 1, step):
        for x in range(0, max(W - win, 0) + 1, step):
            m = v[y:y + win, x:x + win]
            if m.mean() < 0.8:
                continue
            aa, bb = a[y:y + win, x:x + win][m], b[y:y + win, x:x + win][m]
            if (aa > 0.5).mean() < min_active or (bb > 0.5).mean() < min_active:
                continue
            if aa.std() < 1e-6 or bb.std() < 1e-6:
                continue
            out.append((float(np.corrcoef(aa, bb)[0, 1]), y, x))
    return out


def stats(a, b, v, win, step):
    g = float(np.corrcoef(a[v], b[v])[0, 1])
    loc = local_r(a, b, v, win, step)
    rs = np.array([r for r, _, _ in loc]) if loc else np.array([np.nan])
    best = max(loc) if loc else (np.nan, -1, -1)
    return {'global_r': round(g, 4), 'joint_hot': round(float(((a > .5) & (b > .5))[v].mean()), 5),
            'n_windows': len(loc), 'local_r_p99': round(float(np.nanpercentile(rs, 99)), 4) if loc else None,
            'local_r_max': round(float(best[0]), 4) if loc else None, 'best_window_yx': [int(best[1]), int(best[2])]}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--valid', required=True)
    ap.add_argument('--real', nargs=2, action='append', required=True, metavar=('MODEL_A', 'MODEL_B'))
    ap.add_argument('--shuffled', nargs=2, required=True, metavar=('MODEL_A', 'MODEL_B'))
    ap.add_argument('--window', type=int, default=256)
    ap.add_argument('--step', type=int, default=64)
    ap.add_argument('--edge-margin', type=int, default=64,
                    help='ignore pixels this close to the render boundary (both models misbehave there)')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    v = load_valid(a.valid)
    if a.edge_margin > 0:
        v = binary_erosion(v, iterations=a.edge_margin, border_value=0)
    res = {'window_px': a.window, 'edge_margin_px': a.edge_margin, 'real': [], 'shuffled': None}
    for pa, pb in a.real:
        res['real'].append({'a': pa, 'b': pb, **stats(load(pa), load(pb), v, a.window, a.step)})
    res['shuffled'] = {'a': a.shuffled[0], 'b': a.shuffled[1], **stats(load(a.shuffled[0]), load(a.shuffled[1]), v, a.window, a.step)}
    json.dump(res, open(a.out, 'w'), indent=2)
    print(json.dumps(res))


if __name__ == '__main__':
    main()
