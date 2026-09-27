#!/usr/bin/env python3
"""Depth-order control for surface ink models.

A surface-conditioned ink model should respond to ink because of the physical
order of CT planes through the sheet. This tool runs a model on the same render
under three conditions and reports how much of its positive response survives
when that order is destroyed:

  forward    planes as rendered
  reverse    planes reversed (the other physical normal orientation)
  shuffled   the central K planes randomly permuted (fixed seed), then run in the
             reverse orientation.  Same voxels, same per-pixel intensity
             histogram through depth, but no physical depth order.

Output (in --out-dir):
  <name>-{forward,reverse,shuffled}.png   probability maps, value = round(255*p)
  <name>-control.json                     positive fractions and densest windows
  <name>-control.png                      raw / forward / reverse / shuffled panel

Reading the result: if the shuffled positive fraction is similar to or larger
than both real orientations, the model's blobs on this surface are not specific
to physically ordered depth and should not be treated as ink evidence. A real
signal should shrink or lose its shape under shuffling (see README for a
worked example on PHerc1447).

Runner: the released Hecate CLI (huggingface.co/scrollprize/hecate). Any other
model can be plugged in with --cmd, a template containing {input}, {output}
and {reverse} (replaced by the flag string or '').
"""
import argparse, json, os, shlex, subprocess
import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import uniform_filter


def run(cmd_tmpl, inp, out, reverse, rev_flag):
    cmd = cmd_tmpl.format(input=shlex.quote(inp), output=shlex.quote(out), reverse=rev_flag if reverse else '')
    print('+', cmd, flush=True)
    subprocess.run(cmd, shell=True, check=True)


def densest(p, valid, win, thr):
    pos = uniform_filter(((p > thr) & valid).astype(np.float32), win)
    cov = uniform_filter(valid.astype(np.float32), win)
    pos = np.where(cov > 0.8, pos, -1)
    y, x = np.unravel_index(np.argmax(pos), pos.shape)
    return int(y), int(x), float(pos[y, x])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('render', help='(planes,H,W) uint8 .npy from render_surface.py')
    ap.add_argument('--valid', help='(H,W) bool .npy; default <render>-valid.npy')
    ap.add_argument('--out-dir', default='control-out')
    ap.add_argument('--name')
    ap.add_argument('--hecate', help='path to hecate.py')
    ap.add_argument('--checkpoint', help='path to hecate_9.6um.pth')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--cmd', help='custom model command template (overrides --hecate)')
    ap.add_argument('--reverse-flag', default='--reverse')
    ap.add_argument('--shuffle-planes', type=int, default=16)
    ap.add_argument('--seed', type=int, default=20260927)
    ap.add_argument('--threshold', type=float, default=0.5)
    ap.add_argument('--window-um', type=float, default=9600, help='densest-window size (um)')
    ap.add_argument('--px', type=float, default=9.6)
    a = ap.parse_args()

    name = a.name or os.path.splitext(os.path.basename(a.render))[0]
    os.makedirs(a.out_dir, exist_ok=True)
    vol = np.load(a.render)
    valid = np.load(a.valid or a.render.replace('.npy', '') + '-valid.npy')
    if a.cmd:
        tmpl = a.cmd
    else:
        if not (a.hecate and a.checkpoint):
            ap.error('give --hecate and --checkpoint, or --cmd')
        tmpl = (f'python {shlex.quote(a.hecate)} --checkpoint {shlex.quote(a.checkpoint)} --spacing-um {a.px} '
                f'--device {a.device} --input {{input}} --output {{output}} {{reverse}}')

    c = vol.shape[0] // 2
    idx = np.arange(c - a.shuffle_planes // 2, c + a.shuffle_planes // 2)
    perm = idx.copy()
    np.random.default_rng(a.seed).shuffle(perm)
    shuffled = vol.copy()
    shuffled[idx] = vol[perm]
    shuf_path = os.path.join(a.out_dir, name + '-shuffled-input.npy')
    np.save(shuf_path, shuffled)

    outs = {k: os.path.join(a.out_dir, f'{name}-{k}.png') for k in ('forward', 'reverse', 'shuffled')}
    run(tmpl, a.render, outs['forward'], False, a.reverse_flag)
    run(tmpl, a.render, outs['reverse'], True, a.reverse_flag)
    run(tmpl, shuf_path, outs['shuffled'], True, a.reverse_flag)
    os.remove(shuf_path)

    P = {k: np.asarray(Image.open(v), np.float32) / 255 for k, v in outs.items()}
    win = max(8, int(round(a.window_um / a.px)))
    res = {'name': name, 'valid_cm2': float(valid.sum() * (a.px * 1e-4) ** 2), 'threshold': a.threshold,
           'shuffle': {'planes': idx.tolist(), 'permutation': perm.tolist(), 'seed': a.seed},
           'positive_fraction': {k: float((p[valid] > a.threshold).mean()) for k, p in P.items()}}
    real = max(res['positive_fraction']['forward'], res['positive_fraction']['reverse'])
    res['shuffled_over_best_real'] = (res['positive_fraction']['shuffled'] / real) if real > 0 else None
    res['densest_window'] = {}
    for k in ('forward', 'reverse'):
        y, x, d = densest(P[k], valid, win, a.threshold)
        dsh = float(uniform_filter(((P['shuffled'] > a.threshold) & valid).astype(np.float32), win)[y, x])
        res['densest_window'][k] = {'center_yx': [y, x], 'size_px': win, 'positive_fraction': d,
                                    'shuffled_same_window': dsh}
    # heuristic flag only; always inspect the panels
    res['order_specific_heuristic'] = (res['shuffled_over_best_real'] is not None and res['shuffled_over_best_real'] < 0.5)
    json.dump(res, open(os.path.join(a.out_dir, name + '-control.json'), 'w'), indent=2)

    raw = vol[c - 2:c + 2].astype(np.float32).mean(0)
    lo, hi = np.percentile(raw[valid], [1, 99]) if valid.any() else (0, 255)
    panels = [np.uint8(np.clip((raw - lo) / max(hi - lo, 1), 0, 1) * 255)] + \
             [np.uint8(np.clip(P[k] / 0.8, 0, 1) * 255) for k in ('forward', 'reverse', 'shuffled')]
    labels = ['raw CT (central planes)', 'forward', 'reverse', 'shuffled depth (control)']
    h, w = raw.shape
    s = min(1.0, 800 / max(h, w))
    W, H = max(1, int(w * s)), max(1, int(h * s))
    canvas = Image.new('L', (2 * W + 10, 2 * H + 60), 255)
    d = ImageDraw.Draw(canvas)
    for i, (im, lab) in enumerate(zip(panels, labels)):
        x0, y0 = (i % 2) * (W + 10), (i // 2) * (H + 30) + 20
        canvas.paste(Image.fromarray(im).resize((W, H)), (x0, y0))
        frac = '' if i == 0 else f"  p>{a.threshold}: {res['positive_fraction'][lab.split()[0]]:.3f}"
        d.text((x0 + 4, y0 - 16), f'{name}: {lab}{frac}', fill=0)
    canvas.save(os.path.join(a.out_dir, name + '-control.png'))
    print(json.dumps({k: res[k] for k in ('name', 'valid_cm2', 'positive_fraction', 'shuffled_over_best_real', 'order_specific_heuristic')}))


if __name__ == '__main__':
    main()
