"""Compute line/filter associations and velocity sensitivity from cached NIRCam curves.

Associations are predictions of filter transmission, not detections or claims
that a line dominates an observed image. No Cloudy models or cubes are changed.
"""

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.jwst_utils.nircam import load_filters
from src.simple_utils.synthetic_photometry import HC_ERG_ANGSTROM, photon_bandpass_weights

TARGETS = [
    dict(name='NGC1068', redshift=0.003793, redshift_type='existing project systemic',
         observed_line_filter='F187N', reference='https://www.stsci.edu/jwst/phase2-public/3707.pdf'),
    dict(name='JADES182930', redshift=3.03, redshift_type='spectroscopic',
         ra_deg=53.161524, dec_deg=-27.85607, observed_line_filter='F210M', continuum_filter='F182M',
         reference='https://arxiv.org/html/2501.05512v2'),
    dict(name='JADES188852', redshift=2.73, redshift_type='photometric',
         ra_deg=53.173974, dec_deg=-27.843388, observed_line_filter='F182M', continuum_filter='F210M',
         reference='https://arxiv.org/html/2501.05512v2'),
    dict(name='JADES207632', redshift=2.57, redshift_type='photometric',
         ra_deg=53.134037, dec_deg=-27.781039, observed_line_filter='F182M', continuum_filter='F210M',
         reference='https://arxiv.org/html/2501.05512v2'),
]

# Illustrative lab/Cloudy wavelengths. Adopt a consistent air/vacuum convention
# and exact Cloudy line identifiers when constructing a production line list.
LINES = {
    'H_beta': dict(rest_um=0.486132, tracer='hydrogen recombination'),
    'OIII_4959': dict(rest_um=0.495891, tracer='ionized gas; doublet'),
    'OIII_5007': dict(rest_um=0.500684, tracer='ionized gas; cone tracer'),
    'H_alpha': dict(rest_um=0.656280, tracer='hydrogen recombination'),
    'HeI_1083': dict(rest_um=1.08302, tracer='helium emission'),
    'Pa_beta': dict(rest_um=1.28181, tracer='hydrogen recombination'),
    'FeII_1644': dict(rest_um=1.64355, tracer='partially ionized gas; shock sensitivity'),
    'Pa_alpha': dict(rest_um=1.87510, tracer='ionized gas; AGN and star formation'),
    'SiVI_1963': dict(rest_um=1.963, tracer='coronal gas; approximate wavelength'),
    'H2_2122': dict(rest_um=2.12183, tracer='warm molecular gas'),
    'Br_gamma': dict(rest_um=2.16612, tracer='hydrogen recombination'),
    'SiVII_2483': dict(rest_um=2.483, tracer='coronal gas; approximate wavelength'),
    'Br_alpha': dict(rest_um=4.05115, tracer='hydrogen recombination'),
}


def profile_response(rest_um, redshift, velocity_kms, sigma_kms, curve):
    c = 299792.458
    center = rest_um*1e4*(1+redshift)*(1+velocity_kms/c)
    sigma = rest_um*1e4*(1+redshift)*sigma_kms/c
    wave = np.linspace(center-8*sigma, center+8*sigma, 1601)
    profile = np.exp(-0.5*((wave-center)/sigma)**2)/(sigma*np.sqrt(2*np.pi))
    fw, throughput = curve.T
    weights = photon_bandpass_weights(wave, fw, throughput)
    return float(weights@profile*HC_ERG_ANGSTROM/center)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=Path('results/cloudy/nircam_reference_audit'))
    args = p.parse_args()
    filters = load_filters()
    args.output.mkdir(parents=True, exist_ok=True)
    associations = []
    for target in TARGETS:
        for line, info in LINES.items():
            observed = info['rest_um']*(1+target['redshift'])
            for name, entry in filters.items():
                fw, throughput = entry['curve'].T
                response = float(np.interp(observed*1e4, fw, throughput, left=0, right=0))
                if response < 0.001:
                    continue
                associations.append(dict(target=target['name'], redshift=target['redshift'],
                                         redshift_type=target['redshift_type'], line=line,
                                         rest_um=info['rest_um'], observed_um=observed, filter=name,
                                         monochromatic_throughput=response,
                                         fraction_of_peak=response/throughput.max(),
                                         filter_class='narrow' if name.endswith('N') else 'wide' if 'W' in name else 'medium',
                                         known_observed_line_filter=name==target['observed_line_filter']))
    with (args.output/'line_filter_matches.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(associations[0]))
        writer.writeheader()
        writer.writerows(associations)
    velocity_rows = []
    cases = [('NGC1068', 'Pa_alpha', 'F187N'), ('NGC1068', 'FeII_1644', 'F164N'),
             ('NGC1068', 'Br_alpha', 'F405N'), ('JADES182930', 'OIII_5007', 'F210M'),
             ('JADES182930', 'H_beta', 'F210M')]
    for target_name, line, name in cases:
        target = next(t for t in TARGETS if t['name']==target_name)
        for velocity in (-900., 0., 900.):
            value = profile_response(LINES[line]['rest_um'], target['redshift'], velocity, 85., filters[name]['curve'])
            velocity_rows.append(dict(target=target_name, line=line, filter=name, velocity_kms=velocity,
                                      line_sigma_kms=85., profile_averaged_response=value,
                                      fraction_of_filter_peak=value/filters[name]['curve'][:, 1].max()))
    with (args.output/'velocity_response.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(velocity_rows[0]))
        writer.writeheader()
        writer.writerows(velocity_rows)
    report = dict(targets=TARGETS, lines=LINES,
                  interpretation='Transmission predictions, not detection or line dominance; continuum and other lines must be modeled.',
                  wavelength_note='Illustrative wavelengths; use consistent air/vacuum convention for production.',
                  sources=['https://jwst-docs.stsci.edu/jwst-near-infrared-camera/nircam-instrumentation/nircam-filters',
                           'https://academic.oup.com/mnras/article/430/3/2411/1751793',
                           'https://arxiv.org/html/2501.05512v2'],
                  velocity_assumption='Existing toy-cone limits +/-900 km/s, Gaussian sigma 85 km/s; not fitted measurements')
    (args.output/'line_audit.json').write_text(json.dumps(report, indent=2)+'\n')
    for row in velocity_rows:
        print(row['target'], row['line'], row['filter'], row['velocity_kms'],
              f"response={row['profile_averaged_response']:.4f}, peak fraction={row['fraction_of_filter_peak']:.3f}")
    print(f'Saved {len(associations)} line/filter associations in {args.output}')


if __name__ == '__main__':
    main()
