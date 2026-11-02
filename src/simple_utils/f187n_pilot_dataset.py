"""Strict dataset conventions for the controlled one-channel F187N pilot."""
import hashlib
import json
from pathlib import Path

import numpy as np

SCHEMA = 'f187n_grouped_pilot_v1'
VARIANTS = ('smooth_control', 'clumpy_cone', 'full')


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def grouped_splits(count, seed):
    if count < 3:
        raise ValueError('Need at least three independent groups')
    n = max(1, int(count*.2))
    order = np.random.default_rng(seed).permutation(count)
    return {int(index): ('val' if rank<n else 'test' if rank<2*n else 'train')
            for rank, index in enumerate(order)}


def normalize_image(image):
    image = np.asarray(image, dtype=np.float64)
    if image.ndim != 2 or not np.isfinite(image).all():
        raise ValueError('Expected finite 2D F187N image')
    softening = .03*float(np.percentile(np.abs(image), 99))
    if softening <= 0:
        raise ValueError('Image has no usable flux')
    return np.arcsinh(image/softening).astype(np.float32), softening


def validate_manifest(root, verify_hashes=True):
    root = Path(root)
    manifest = json.loads((root/'manifest.json').read_text())
    if (manifest.get('schema') != SCHEMA or manifest.get('band') != 'F187N' or
            manifest.get('shape') != [1,256,256] or
            manifest.get('projection') != 'circular_cone_envelope_v2' or
            manifest.get('normalization') != 'asinh(image/(0.03*p99(abs(image))))'):
        raise ValueError('Wrong band, projection, shape, normalization, or dataset schema')
    groups, ids, digests, mask_hashes, seeds = {}, set(), {}, {}, {}
    for row in manifest['samples']:
        group, split = row['group'], row['split']
        if split not in ('train','val','test') or row['variant'] not in VARIANTS:
            raise ValueError('Invalid split or variant')
        if row['id'] in ids or (group in groups and groups[group] != split):
            raise ValueError('Duplicate sample or physical group leaked across splits')
        ids.add(row['id']); groups[group] = split
        if row['seed'] in seeds and seeds[row['seed']] != group:
            raise ValueError('One realization seed is assigned to multiple groups')
        seeds[row['seed']] = group
        if row['softening'] <= 0 or not np.isfinite(row['softening']):
            raise ValueError('Invalid image normalization')
        if group in mask_hashes and mask_hashes[group] != row['mask_sha256']:
            raise ValueError('Matched variants must retain identical masks')
        mask_hashes[group] = row['mask_sha256']
        for kind in ('image','mask'):
            path = Path(row[kind])
            if path.is_absolute() or '..' in path.parts:
                raise ValueError('Dataset paths must stay within dataset root')
            if verify_hashes and sha256(root/path) != row[kind+'_sha256']:
                raise ValueError(f'Changed {kind}: {path}')
        digest = row['image_sha256']
        if digest in digests and digests[digest] != split:
            raise ValueError('Identical input appears across data splits')
        digests[digest] = split
    for variant in VARIANTS:
        rows = [r for r in manifest['samples'] if r['variant']==variant]
        if len(rows) != len(groups) or {r['group'] for r in rows} != set(groups):
            raise ValueError('Each group requires exactly one of every variant')
        if {r['split'] for r in rows} != {'train','val','test'}:
            raise ValueError('Every variant requires nonempty train/val/test splits')
    if len(seeds) != len(groups):
        raise ValueError('Each group must have one consistent realization seed')
    return manifest


def fit_mask_angles(mask, east, north, valid):
    """Fit a bicone to a binary mask in a fixed 0.3--1.5 arcsec annulus.

    Candidate full widths 30..150 deg, axes 0..177 deg, both in 3-deg steps.
    No target geometry is supplied. Reject if no wedge beats empty/full masks.
    This descriptive fit is not an uncertainty or a 3D angle measurement.
    """
    radius = np.hypot(east,north)
    aperture = valid & (radius>.3) & (radius<1.5)
    angles = np.rad2deg(np.arctan2(east,north)) % 180
    bins = np.minimum((angles[aperture]/3).astype(int),59)
    positive = np.bincount(bins, weights=mask[aperture].astype(float), minlength=60)
    total = np.bincount(bins, minlength=60)
    centers = np.arange(60)*3+1.5
    best = float(min(positive.sum(),total.sum()-positive.sum()))
    result = None
    for axis in np.arange(0.,180.,3.):
        distance = abs((centers-axis+90)%180-90)
        for width in np.arange(30.,151.,3.):
            inside = distance <= width/2
            error = float((total[inside]-positive[inside]).sum()+positive[~inside].sum())
            if error < best:
                best = error
                result = {'axis_pa_mod180_deg':float(axis), 'projected_full_opening_deg':float(width)}
    return result
