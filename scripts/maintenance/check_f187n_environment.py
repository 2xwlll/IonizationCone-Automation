#!/usr/bin/env python3
"""Report environment/input readiness; parses on Python 3.6 to explain upgrades."""
import argparse
import importlib
import json
from pathlib import Path
import platform
import sys


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs',action='store_true')
    parser.add_argument('--require-cuda',action='store_true')
    args=parser.parse_args()
    report={'python':sys.version,'executable':sys.executable,'platform':platform.platform(),
            'libc':platform.libc_ver(),'packages':{},'errors':[]}
    if sys.version_info < (3,11):
        report['errors'].append('Use the dedicated Python 3.12 environment; this scientific stack needs Python >=3.11.')
    else:
        for name in ('numpy','scipy','astropy','matplotlib','torch'):
            try:report['packages'][name]=str(importlib.import_module(name).__version__)
            except Exception as exc:report['errors'].append('{}: {}'.format(name,exc))
        if 'torch' in report['packages']:
            import torch
            report['cuda_available']=torch.cuda.is_available()
            report['torch_cuda_build']=torch.version.cuda
            if args.require_cuda:
                if not torch.cuda.is_available():report['errors'].append('GPU unavailable; use an allocated GPU node and matching PyTorch CUDA wheel.')
                else:
                    try:
                        report['gpu']=torch.cuda.get_device_name(0)
                        report['gpu_capability']=list(torch.cuda.get_device_capability(0))
                        layer=torch.nn.Conv2d(1,2,3).cuda()
                        value=layer(torch.ones(1,1,16,16,device='cuda')).square().mean()
                        value.backward();torch.cuda.synchronize()
                        if not torch.isfinite(value):raise RuntimeError('Nonfinite GPU output')
                        report['gpu_forward_backward']='passed'
                    except Exception as exc:report['errors'].append('GPU kernel check: {}'.format(exc))
    if args.inputs:
        root=Path(__file__).resolve().parents[2]
        cfg=json.loads((root/'configs/cloudy/ngc1068_f187n_soft_gas_projection_v2.json').read_text())
        names=[cfg[k] for k in ('grid','reference','psf')]+[cfg['star_forming']['grid']]
        names += [str(Path(cfg['grid']).parent/'manifest.json'),
                  str(Path(cfg['star_forming']['grid']).parent/'manifest.json')]
        report['inputs']={name:(root/name).is_file() for name in names}
        report['errors']+=['Missing input: '+p for p,exists in report['inputs'].items() if not exists]
        hii_manifest=root/Path(cfg['star_forming']['grid']).parent/'manifest.json'
        if hii_manifest.is_file():
            try:
                if not json.loads(hii_manifest.read_text())['status'].startswith('completed stellar-ionized'):
                    report['errors'].append('H II grid manifest does not report completed stellar-ionized calculations')
            except Exception as exc:report['errors'].append('Invalid H II manifest: {}'.format(exc))
    print(json.dumps(report,indent=2))
    return 1 if report['errors'] else 0


if __name__=='__main__':sys.exit(main())
