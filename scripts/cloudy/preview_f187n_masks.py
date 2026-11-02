"""Display saved F187N image/mask pairs without altering the scientific arrays."""
import argparse
import json
import os
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR','/tmp/ioncone-matplotlib')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import AsinhNorm


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--population',type=Path,default=Path('results/cloudy/f187n_population_pilot_v3'))
    args=parser.parse_args()
    samples=json.loads((args.population/'manifest.json').read_text())['samples']
    output=args.population/'mask_examples'
    output.mkdir(exist_ok=True)
    with np.load(args.population/'reference_comparison.npz') as f:
        top=float(np.percentile(f['real'][f['valid']],99.5))
    norm=AsinhNorm(linear_width=.03*top,vmin=0,vmax=top)

    def row(axes,image,mask,name):
        for ax in (axes[0],axes[2]):
            ax.imshow(image,origin='lower',cmap='magma',norm=norm,interpolation='nearest')
        axes[1].imshow(mask,origin='lower',cmap='gray',vmin=0,vmax=1,interpolation='nearest')
        tint=np.zeros((*mask.shape,4))
        tint[...,:3]=[0,1,1]
        tint[...,3]=.12*mask
        axes[2].imshow(tint,origin='lower',interpolation='nearest')
        axes[2].contour(mask,levels=[.5],colors=['cyan'],linewidths=1.3,origin='lower')
        for ax,title in zip(axes,('Synthetic F187N','Cone mask (white = cone)','Cone boundary + shaded interior')):
            ax.set_title(title,fontsize=11)
            ax.set_xticks([]);ax.set_yticks([])
        axes[0].set_ylabel(name,fontsize=12)

    fig,axes=plt.subplots(len(samples),3,figsize=(11,3.6*len(samples)),squeeze=False,constrained_layout=True)
    for i,sample in enumerate(samples):
        with np.load(Path(sample['output'])/'image_mask.npz') as f:
            image,mask=f['input'],f['mask']
        assert image.shape==mask.shape and np.isfinite(image).all()
        assert np.isin(mask,[0,1]).all()
        row(axes[i],image,mask,sample['name'])
        single,panels=plt.subplots(1,3,figsize=(12,4),constrained_layout=True)
        row(panels,image,mask,sample['name'])
        single.suptitle('Synthetic example: geometric cone label, not a brightness threshold',fontsize=12)
        single.savefig(output/(sample['name']+'.png'),dpi=160)
        plt.close(single)
    fig.suptitle('F187N examples with known synthetic cone masks\nCyan marks the projected AGN cone; host emission can overlap it',fontsize=13)
    fig.savefig(output/'all_examples.png',dpi=150)
    plt.close(fig)
    print(output/'all_examples.png')


if __name__=='__main__':
    main()
