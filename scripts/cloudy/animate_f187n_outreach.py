#!/usr/bin/env python3
"""Render outreach animations from saved F187N components, without new physics.

Frames show explanatory crossfades/addition, never physical time evolution.
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import binary_erosion
os.environ.setdefault('MPLCONFIGDIR', '/tmp/agn-outreach-matplotlib')
from matplotlib import colormaps

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'results/cloudy/ngc1068_f187n_soft_gas_v1/image_mask.npz'
OUT = ROOT / 'results/cloudy/f187n_outreach_v3'
W, H, SIDE, FPS = 1200, 800, 608, 10
BG = (5, 9, 19)
WHITE, MUTED, CYAN = '#eff5ff', '#9baac1', '#6fe7ec'


def font(size, weight='Regular'):
    return ImageFont.truetype(f'/usr/share/fonts/opentype/inter/InterDisplay-{weight}.otf', size)


def ease(x):
    x = np.clip(x, 0, 1)
    return x*x*(3-2*x)


def rgb(data, ceiling, cmap='magma'):
    a = np.arcsinh(22*np.maximum(data, 0)/ceiling)/np.arcsinh(22)
    return np.uint8(255*colormaps[cmap](np.clip(a, 0, 1))[..., :3])[::-1].copy()


def canvas(title, subtitle, array, section, heading, lines, progress, accent=CYAN):
    im = Image.new('RGB', (W, H), BG)
    d = ImageDraw.Draw(im)
    d.text((42, 23), 'LIGHT FROM AN ACTIVE GALAXY', font=font(13, 'SemiBold'), fill=CYAN)
    d.text((42, 48), title, font=font(36, 'Light'), fill=WHITE)
    d.text((44, 97), subtitle, font=font(15), fill=MUTED)
    tile = Image.fromarray(array).resize((SIDE, SIDE), Image.Resampling.LANCZOS)
    im.paste(tile, (42, 135))
    d.rectangle((41, 134, 650, 743), outline='#233047')
    d.text((694, 162), section.upper(), font=font(13, 'SemiBold'), fill=accent)
    y = 198
    for s in heading:
        d.text((692, y), s, font=font(36, 'Light'), fill=WHITE)
        y += 44
    y += 25
    for line in lines:
        d.text((694, y), line, font=font(18), fill=MUTED)
        y += 29
    d.line((694, 610, 1148, 610), fill='#253149', width=1)
    d.text((694, 633), 'JWST / NIRCam / F187N', font=font(16, 'SemiBold'), fill=WHITE)
    d.text((694, 661), 'NGC 1068-inspired synthetic model', font=font(15), fill=MUTED)
    d.text((694, 687), '2D field: approximately 7.9 arcseconds', font=font(15), fill=MUTED)
    d.text((42, 768), 'SIMULATION  /  Illustrative, not an observed image or a fitted galaxy model', font=font(13), fill=MUTED)
    d.rectangle((694, 734, 1148, 737), fill='#253149')
    d.rectangle((694, 734, int(694+454*progress), 737), fill=accent)
    return im


def export(name, frames):
    # A single palette prevents colors flickering from frame to frame.
    sheet = Image.new('RGB', (240*8, 160*6))
    for j, frame in enumerate(frames[::max(1, len(frames)//48)][:48]):
        sheet.paste(frame.resize((240, 160)), ((j % 8)*240, (j//8)*160))
    palette = sheet.quantize(colors=256)
    quantized = [f.quantize(palette=palette, dither=Image.Dither.NONE) for f in frames]
    quantized[0].save(OUT/f'{name}.gif', save_all=True, append_images=quantized[1:],
                      duration=1000//FPS, loop=0, disposal=2, optimize=False)
    frames[len(frames)//2].save(OUT/f'{name}_poster.png')
    # MP4 uses the original RGB frames, giving gradients suitable for projection.
    p = subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo',
        '-pixel_format', 'rgb24', '-video_size', f'{W}x{H}', '-framerate', str(FPS),
        '-i', '-', '-an', '-c:v', 'libx264', '-crf', '18', '-pix_fmt', 'yuv420p',
        '-movflags', '+faststart', str(OUT/f'{name}.mp4')], stdin=subprocess.PIPE)
    for frame in frames:
        p.stdin.write(frame.tobytes())
    p.stdin.close()
    if p.wait():
        raise RuntimeError('ffmpeg export failed')
    print(name, len(frames), 'frames', flush=True)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT/'manifest.json').exists():
        raise FileExistsError('Preserve previous exports; choose a fresh OUT path.')
    p = np.load(SOURCE)
    groups = [sum(p[f'{prefix}_{line}'] for line in ('pa_alpha', 'other_lines', 'nebular_continuum'))
              for prefix in ('cone', 'host', 'starforming')]
    groups += [p['stellar_continuum'], p['nuclear_continuum']]
    total = sum(groups)
    metadata = json.loads(str(p['metadata']))
    np.testing.assert_allclose(total, p['relative_clean']*metadata['empirical_scale'], rtol=1e-12)
    ceiling = float(np.percentile(total, 99.6))
    full = rgb(total, ceiling)
    labels = ['AGN-ionized gas', 'Host galaxy gas', 'Star-forming gas', 'Stellar light', 'Bright nucleus']
    copy = [
        (['A soft cone', 'of glowing gas'], ['Gas emission follows an irregular,', 'clumpy illumination pattern.', '', 'Gas spectra are supplied by CLOUDY;', 'the projected structure is prescribed.']),
        (['Gas beyond', 'the cone'], ['A separate host component can', 'overlap the cone in projection.', '', 'This gas uses lower-ionization', 'AGN photoionization models.']),
        (['Young stars', 'light up gas'], ['A separate stellar-ionized H II', 'component adds diffuse structure.', '', 'Its placement is independent of', 'the cone mask.']),
        (['The underlying', 'stellar glow'], ['A smooth, assumed stellar', 'continuum contributes to F187N.', '', 'It is part of the same image,', 'alongside the gas emission.']),
        (['A luminous', 'central source'], ['An assumed hot-dust plus power-law', 'continuum illuminates the center.', '', 'A point source and compact', 'surroundings form this component.']),
        (['Many sources.', 'One filter.'], ['F187N collects emission lines', 'and continuum together.', '', 'The saved maps include the', 'instrument point-spread function.'])]
    portraits = [rgb(a, max(float(np.percentile(a, 99.7)), np.finfo(float).tiny)) for a in groups] + [full]
    # Tour: hold each component and gently dissolve to the next; periodic endpoint.
    frames = []
    for i in range(6):
        for k in range(20):
            t = float(ease((k-13)/6))
            a = np.uint8((1-t)*portraits[i].astype(float)+t*portraits[(i+1)%6])
            active = i if t < .5 else (i+1)%6
            info = copy[active]
            frames.append(canvas('Anatomy of an active galaxy',
                'A tour of the components contributing light to one infrared filter', a,
                f'{active+1:02d} / 06   '+(labels+['Combined F187N image'])[active], info[0],
                info[1]+['', 'Each isolated view is stretched separately.'], (i+k/20)/6))
    export('01_component_tour', frames)
    # Build: additive maps in linear intensity, with one common nonlinear display.
    frames = []
    order = [3, 1, 2, 0, 4]
    descriptions = ['Stellar continuum', 'Add host galaxy gas', 'Add star-forming gas', 'Add AGN-ionized gas', 'Add the bright nucleus']
    for i in range(120):
        phase = i/10
        weights = np.array([float(ease((phase-j*1.65)/1.1)) for j in range(5)])
        if phase > 10:
            weights *= 1-float(ease((phase-10)/1.9))
        view = sum(w*groups[j] for w, j in zip(weights, order))
        current = min(int(phase/1.65), 4)
        lines = [('+  ' if w > .99 else '-  ')+labels[j] for w, j in zip(weights, order)]
        lines += ['', 'One fixed display stretch throughout.', 'Transitions show addition, not evolution.']
        frames.append(canvas('Building the infrared image',
            'Independent emission components add together before the display stretch', rgb(view, ceiling),
            'ASSEMBLING THE SIGNAL', [descriptions[current].replace('Add ', ''), 'in F187N'], lines, i/119))
    export('02_build_the_image', frames)
    # Mask: exact binary geometry, with contour; no physical changes to the image.
    mask = (p['mask'] > .5)[::-1]
    edge = mask ^ binary_erosion(mask)
    frames = []
    for i in range(100):
        alpha = float(ease((i-15)/22)*(1-ease((i-65)/22)))
        a = full.astype(float).copy()
        tint = np.array([80, 230, 238])
        a[mask] = (1-.16*alpha)*a[mask]+.16*alpha*tint
        a[edge] = (1-alpha)*a[edge]+alpha*tint
        frames.append(canvas('Where is the ionization cone?',
            'Soft emission and a sharp training label describe different things', np.uint8(a),
            'THE SEGMENTATION TARGET', ['Soft gas.', 'Sharp geometry.'],
            ['Cyan marks the nominal cone mask.', '', 'The mask is a geometric label;', 'it is not a brightness threshold.', '', 'Gas can glow outside the boundary.', 'Other light can appear inside it.', '', 'Overlay fades; the model stays fixed.'], i/99))
    export('03_cone_mask_reveal', frames)
    manifest = {
        'source': str(SOURCE.relative_to(ROOT)),
        'source_sha256': hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'dimensions': [W, H], 'fps': FPS,
        'animations': 'Explanatory crossfades, additive assembly and label overlays; not physical time evolution.',
        'display': 'Magma false color, asinh stretch. Tour isolated maps independently stretched; build and mask use fixed total-image stretch. No artificial stars or emission added.',
        'physics': 'Existing illustrative 2D model. Cached AGN grid has first-iteration warnings; H II gas has converged CLOUDY spectra. Assumed geometry, continua and amplitudes. Not a fit to NGC 1068, and not the new spectral-fit pilot.',
        'checks': 'Independent component sum agrees with saved relative_clean times empirical_scale; exact saved binary mask used.'}
    (OUT/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    (OUT/'README.md').write_text('''# F187N outreach animations

Three looping animations derived directly from the latest saved soft-gas F187N model:

1. **Component tour:** separately stretched gas and continuum components, followed by their combined image.
2. **Build the image:** linear addition of existing components under one fixed display stretch.
3. **Cone mask reveal:** the saved sharp binary mask over fixed, softer emission.

GIFs loop automatically. MP4 copies preserve smoother gradients for projection; set your presentation player to loop. Each is 1200 × 800 pixels. Posters are included.

These are explanatory transitions, not time evolution, measured visible colors, or a new simulation run. False colors encode brightness in one infrared filter. CLOUDY supplies gas spectra; morphology and continua are prescribed. The source is an illustrative NGC 1068-inspired 2D model, not an observed image or an observationally fitted galaxy. It retains the source model's incomplete line inventory and old AGN-grid convergence limitations. New spectral-comparison fits have not been substituted into these images. See manifest.json for provenance and display choices.

Reproduce from the repository root: `venv/bin/python scripts/cloudy/animate_f187n_outreach.py`. Existing completed exports are protected; choose a fresh output path before rerunning.
''')


if __name__ == '__main__':
    main()
