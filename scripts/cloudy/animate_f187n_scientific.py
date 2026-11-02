#!/usr/bin/env python3
"""Image-only, common-stretch presentations of the existing F187N model."""
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import binary_erosion

import animate_f187n_outreach as render

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'results/cloudy/f187n_scientific_gifs_v1'
SIZE, FPS = 900, 5
render.OUT, render.W, render.H, render.FPS = OUT, SIZE, SIZE, FPS


def frame(array, title, label, note):
    im = Image.fromarray(array).resize((SIZE, SIZE), Image.Resampling.LANCZOS).convert('RGBA')
    overlay = Image.new('RGBA', im.size)
    d = ImageDraw.Draw(overlay)
    d.rectangle((0, 0, SIZE, 124), fill=(0, 0, 0, 195))
    d.rectangle((0, SIZE-116, SIZE, SIZE), fill=(0, 0, 0, 195))
    d.text((25, 12), title, font=render.font(39, 'SemiBold'), fill='white')
    d.text((25, 65), label, font=render.font(31), fill='white')
    d.text((25, SIZE-107), note, font=render.font(27), fill='white')
    d.text((25, SIZE-65), 'Synthetic F187N | fixed asinh stretch', font=render.font(25), fill='white')
    # Angular bar uses the saved detector pixel scale, not assumed orientation.
    length = SIZE/256/PIXEL_SCALE
    x, y = SIZE-35-length, SIZE-29
    d.line((int(x), y, SIZE-35, y), fill='white', width=4)
    d.text((int(x), y-38), '1 arcsec', font=render.font(24), fill='white')
    return Image.alpha_composite(im, overlay).convert('RGB')


def main():
    global PIXEL_SCALE
    if OUT.exists():
        raise FileExistsError('Use a fresh output directory to preserve prior exports.')
    OUT.mkdir(parents=True)
    p = np.load(render.SOURCE)
    meta = json.loads(str(p['metadata']))
    PIXEL_SCALE = meta['pixel_scale_arcsec']
    groups = [sum(p[f'{prefix}_{kind}'] for kind in ('pa_alpha','other_lines','nebular_continuum'))
              for prefix in ('cone','host','starforming')]
    groups += [p['stellar_continuum'], p['nuclear_continuum']]
    labels = ['AGN-ionized gas', 'Host gas (low-ionization AGN model)',
              'Stellar-ionized H II gas', 'Stellar continuum', 'Nuclear continuum']
    total = sum(groups)
    np.testing.assert_allclose(total, p['relative_clean']*meta['empirical_scale'], rtol=1e-12)
    fractions = [float(a.sum()/total.sum()) for a in groups]
    ceiling = float(np.percentile(total, 99.6))
    # No frame-by-frame normalization and no crossfades between components.
    frames = []
    for i, (a, label) in enumerate(zip(groups+[total], labels+['All components combined'])):
        fraction = fractions[i] if i < 5 else 1.
        f = frame(render.rgb(a, ceiling), 'ISOLATED COMPONENTS', label,
                  f'Model signal: {100*fraction:.1f}% of the total')
        frames.extend([f]*20)
    render.export('01_isolated_components', frames)
    frames = []
    cumulative = np.zeros_like(total)
    order = [3, 1, 2, 0, 4]
    for i, j in enumerate(order):
        cumulative = cumulative+groups[j]
        note = 'Only stellar continuum' if i == 0 else f'Previous components + {labels[j]}'
        # Long host label is shortened only in the explanatory addition text.
        note = note.replace('Host gas (low-ionization AGN model)', 'host gas')
        f = frame(render.rgb(cumulative, ceiling), 'CUMULATIVE F187N SIGNAL',
                  f'Step {i+1}/5: '+('start with ' if i==0 else 'add ')+labels[j].replace(' (low-ionization AGN model)',''),
                  note)
        frames.extend([f]*20)
    np.testing.assert_allclose(cumulative, total, rtol=1e-12)
    render.export('02_cumulative_components', frames)
    mask = (p['mask']>.5)[::-1]
    edge = mask ^ binary_erosion(mask)
    full = render.rgb(total, ceiling)
    frames = []
    for i in range(70):
        alpha = float(render.ease((i-10)/12)*(1-render.ease((i-48)/12)))
        a = full.astype(float)
        tint = np.array([70, 235, 245])
        a[mask] = (1-.10*alpha)*a[mask]+.10*alpha*tint
        a[edge] = (1-alpha)*a[edge]+alpha*tint
        frames.append(frame(np.uint8(a), 'F187N MODEL + CONE MASK',
            'Cyan: nominal projected cone boundary', 'Geometric training label; not a brightness threshold'))
    render.export('03_cone_mask', frames)
    provenance = {
        'source': str(render.SOURCE.relative_to(ROOT)),
        'source_sha256': hashlib.sha256(render.SOURCE.read_bytes()).hexdigest(),
        'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'dimensions': [SIZE, SIZE], 'fps': FPS,
        'component_labels': labels, 'integrated_model_signal_fractions': fractions,
        'display_ceiling': ceiling, 'display': 'One fixed magma/asinh brightness mapping for all three movies; image-only labels and angular scale. Static held stages, no component crossfades.',
        'signal': 'Saved empirically scaled, PSF-convolved source component maps; excludes added baseline and noise. Fractions are assumed model contributions, not measured NGC 1068 population fractions.',
        'physics': meta['limitations'],
        'checks': 'Component sum equals empirical_scale * relative_clean; cumulative last stage equals full sum; mask uses exact saved binary array.',
        'not_physical_time': True}
    (OUT/'manifest.json').write_text(json.dumps(provenance, indent=2)+'\n')
    (OUT/'README.md').write_text('''# Scientific F187N component animations

Image-only 900 x 900 animations, with large overlaid labels and a one-arcsecond scale bar. All use the SAME fixed asinh stretch and false-color mapping. Isolated components are shown individually (four seconds each); cumulative stages retain everything already added (four seconds each). Weak components may be faint at this honest common scale. The mask loop lasts 14 seconds. MP4 copies and posters are included.

The data are the existing soft-gas model, not a new CLOUDY run or an observed galaxy. No spectral or component weights were changed. These are already band-integrated, PSF-convolved source maps; added baseline and approximate noise are excluded to explain the components. They are not a full JWST Level 3 simulation. Signal fractions are integrated over the modeled field and describe the assumed model, not measured population fractions in NGC 1068. The host is low-ionization AGN gas, not the independent H II population. AGN grid convergence, incomplete line inventory, assumed continua, geometry and amplitude limitations remain unchanged. See manifest.json.

Regenerate with `venv/bin/python scripts/cloudy/animate_f187n_scientific.py`, choosing a fresh OUT directory in the script to preserve previous exports.
''')


if __name__ == '__main__':
    main()
