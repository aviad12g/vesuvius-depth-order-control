#!/usr/bin/env python3
"""Build isotropic surface-conditioned renders (Z, Y, X uint8) for 9.6 um ink models.

Two input kinds are supported:

  tifxyz          a tifxyz mesh (x.tif, y.tif, z.tif, meta.json) plus its CT volume
                  (level-0 zarr v2). Planes are sampled along the mesh normals.
  surface-volume  an existing surface-volume zarr (e.g. the official
                  PHerc0800 `surface-volumes/*.zarr`), resampled to the target pitch.

Output: <out>.npy  (planes, H, W) uint8, planes centred on the surface, spacing
        --px um in all three axes; <out>-valid.npy (H, W) bool.

Sources may be local paths or http(s) URLs (e.g. the Vesuvius open-data bucket).
Only the chunks a surface actually touches are downloaded; they are cached on disk.
Depth direction follows the mesh normal (u x v); run models in BOTH depth orders.
"""
import argparse, concurrent.futures as cf, io, json, math, os, time, urllib.request
import numpy as np
import tifffile
from scipy.ndimage import map_coordinates


def read(src, tries=5):
    """Bytes from a local path or URL; None if missing."""
    if not src.startswith(('http://', 'https://')):
        return open(src, 'rb').read() if os.path.exists(src) else None
    for t in range(tries):
        try:
            return urllib.request.urlopen(src, timeout=120).read()
        except Exception as e:  # noqa: BLE001
            if '404' in str(e):
                return None
            time.sleep(1 + t)
    raise RuntimeError('fetch failed: ' + src)


class ZarrV2:
    """Minimal read-only zarr v2 array (any numcodecs compressor, '/' or '.' separator)."""

    def __init__(self, root, cache):
        self.root = root.rstrip('/') + '/'
        m = json.loads(read(self.root + '.zarray'))
        self.shape, self.chunks, self.dtype = m['shape'], m['chunks'], np.dtype(m['dtype'])
        self.sep = m.get('dimension_separator', '.')
        self.fill = m.get('fill_value') or 0
        self.codec = None
        if m.get('compressor'):
            import numcodecs
            self.codec = numcodecs.get_codec(m['compressor'])
        self.cache = cache
        os.makedirs(cache, exist_ok=True)

    def _path(self, k):
        return os.path.join(self.cache, '_'.join(map(str, k)))

    def fetch(self, keys, workers=64):
        todo = [k for k in keys if not os.path.exists(self._path(k))]

        def one(k):
            d = read(self.root + self.sep.join(map(str, k)))
            open(self._path(k), 'wb').write(d or b'')

        with cf.ThreadPoolExecutor(workers) as ex:
            list(ex.map(one, todo))

    def _chunk(self, k):
        d = open(self._path(k), 'rb').read()
        if not d:
            return None
        if self.codec is not None:
            d = self.codec.decode(d)
        a = np.frombuffer(d, self.dtype)
        return a.reshape(self.chunks) if a.size == np.prod(self.chunks) else None

    def keys(self, lo, hi):
        c = self.chunks
        return [(a, b, d) for a in range(lo[0] // c[0], (hi[0] - 1) // c[0] + 1)
                for b in range(lo[1] // c[1], (hi[1] - 1) // c[1] + 1)
                for d in range(lo[2] // c[2], (hi[2] - 1) // c[2] + 1)]

    def block(self, lo, hi):
        """Dense [lo, hi) ZYX block; unavailable chunks are filled with fill_value."""
        ks = self.keys(lo, hi)
        self.fetch(ks)
        out = np.full([h - l for l, h in zip(lo, hi)], self.fill, self.dtype)
        c = self.chunks
        for k in ks:
            a = self._chunk(k)
            if a is None:
                continue
            o = [k[i] * c[i] for i in range(3)]
            s = [slice(max(lo[i], o[i]), min(hi[i], o[i] + c[i])) for i in range(3)]
            out[tuple(slice(s[i].start - lo[i], s[i].stop - lo[i]) for i in range(3))] = \
                a[tuple(slice(s[i].start - o[i], s[i].stop - o[i]) for i in range(3))]
        return out


def render_tifxyz(a):
    vol = ZarrV2(a.volume, os.path.join(a.cache, 'volume'))
    sp = a.tifxyz.rstrip('/') + '/'
    meta = json.loads(read(sp + 'meta.json'))
    X, Y, Z = [tifffile.imread(io.BytesIO(read(sp + f'{c}.tif'))).astype(np.float64) for c in 'xyz']
    if a.crop:
        r0, r1, c0, c1 = a.crop
        X, Y, Z = X[r0:r1, c0:c1], Y[r0:r1, c0:c1], Z[r0:r1, c0:c1]
    V = (X > 0) & (Y > 0) & (Z > 0)
    P = np.where(V[..., None], np.stack([Z, Y, X], -1), np.nan)  # ZYX voxel coords

    def cdiff(arr, ax):
        g = np.full(arr.shape, np.nan)
        if ax == 0:
            g[1:-1] = (arr[2:] - arr[:-2]) / 2
        else:
            g[:, 1:-1] = (arr[:, 2:] - arr[:, :-2]) / 2
        return g

    du = np.stack([cdiff(P[..., i], 1) for i in range(3)], -1)
    dv = np.stack([cdiff(P[..., i], 0) for i in range(3)], -1)
    N = np.cross(du, dv)
    N /= np.linalg.norm(N, axis=-1, keepdims=True)
    good = V & np.isfinite(N).all(-1)
    f = (1.0 / float(meta['scale'][0])) * a.voxel_um / a.px  # output px per stored grid step
    H, W = int((P.shape[0] - 1) * f) + 1, int((P.shape[1] - 1) * f) + 1
    offs = (np.arange(a.planes) - (a.planes - 1) / 2) * a.px / a.voxel_um
    out = np.zeros((a.planes, H, W), np.uint8)
    valid = np.zeros((H, W), bool)
    T = 512
    tiles = [(y, x) for y in range(0, H, T) for x in range(0, W, T)]
    print(f'stored grid {P.shape[:2]} -> output {(H, W)}, {len(tiles)} tiles', flush=True)
    for ti, (y, x) in enumerate(tiles):
        yy, xx = np.mgrid[y:min(y + T, H), x:min(x + T, W)] / f
        pts = np.stack([map_coordinates(np.nan_to_num(P[..., i]), [yy, xx], order=1) for i in range(3)], -1)
        nv = np.stack([map_coordinates(np.nan_to_num(N[..., i]), [yy, xx], order=1) for i in range(3)], -1)
        ok = map_coordinates(good.astype(np.float32), [yy, xx], order=1) > 0.999
        nn = np.linalg.norm(nv, axis=-1)
        ok &= nn > 0.5
        if not ok.any():
            continue
        q, n = pts[ok], nv[ok] / nn[ok][:, None]
        ends = np.concatenate([q + n * offs.min(), q + n * offs.max()])
        lo = np.maximum(np.floor(ends.min(0)) - 2, 0).astype(int)
        hi = np.minimum(np.ceil(ends.max(0)) + 3, vol.shape).astype(int)
        if len(vol.keys(lo, hi)) > a.max_chunks_per_tile:
            print(f'skip tile {ti}: folded/steep geometry ({len(vol.keys(lo, hi))} chunks)', flush=True)
            continue
        blk = vol.block(lo, hi).astype(np.float32)
        for k, o in enumerate(offs):
            s = (q + n * o - lo).T
            plane = out[k, y:y + ok.shape[0], x:x + ok.shape[1]]
            plane[ok] = np.rint(np.clip(map_coordinates(blk, s, order=1), 0, 255)).astype(np.uint8)
        valid[y:y + ok.shape[0], x:x + ok.shape[1]] = ok
        if ti % 10 == 0:
            print(f'tile {ti + 1}/{len(tiles)}', flush=True)
    out[:, ~valid] = 0
    return out, valid


def render_surface_volume(a):
    z = ZarrV2(a.zarr, os.path.join(a.cache, 'surface'))
    S = z.shape
    arr = z.block((0, 0, 0), tuple(S)).astype(np.float32)
    ctr = (S[0] - 1) / 2
    f = a.px / a.voxel_um
    H, W = int((S[1] - 1) / f) + 1, int((S[2] - 1) / f) + 1
    yy, xx = np.mgrid[0:H, 0:W] * f
    out = np.zeros((a.planes, H, W), np.uint8)
    for k in range(a.planes):
        zc = ctr + (k - (a.planes - 1) / 2) * a.px / a.voxel_um
        if not 0 <= zc <= S[0] - 1:
            raise SystemExit(f'requested depth exceeds the {S[0]}-plane surface volume')
        out[k] = np.rint(np.clip(map_coordinates(arr, [np.full_like(yy, zc), yy, xx], order=1), 0, 255))
    valid = out.min(0) > 0
    out[:, ~valid] = 0
    return out, valid


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='kind', required=True)
    t = sub.add_parser('tifxyz')
    t.add_argument('--tifxyz', required=True, help='directory/URL containing x.tif y.tif z.tif meta.json')
    t.add_argument('--volume', required=True, help='level-0 zarr v2 array of the CT volume (path or URL)')
    t.add_argument('--crop', type=int, nargs=4, metavar=('R0', 'R1', 'C0', 'C1'), help='crop on the stored grid')
    t.add_argument('--max-chunks-per-tile', type=int, default=400)
    s = sub.add_parser('surface-volume')
    s.add_argument('--zarr', required=True, help='level-0 zarr v2 array of a surface volume')
    for q in (t, s):
        q.add_argument('--voxel-um', type=float, required=True, help='source voxel size (um)')
        q.add_argument('--px', type=float, default=9.6, help='output spacing in all axes (um)')
        q.add_argument('--planes', type=int, default=24)
        q.add_argument('--cache', default='./chunk-cache')
        q.add_argument('--out', required=True)
    a = p.parse_args()
    out, valid = render_tifxyz(a) if a.kind == 'tifxyz' else render_surface_volume(a)
    np.save(a.out, out)
    np.save(a.out.replace('.npy', '') + '-valid.npy', valid)
    print(json.dumps({'shape': out.shape, 'valid_cm2': float(valid.sum() * (a.px * 1e-4) ** 2)}))


if __name__ == '__main__':
    main()
