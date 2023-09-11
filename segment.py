import numpy as np
import sys
import suite2p
from suite2p import run_s2p
from suite2p import registration
import caiman as cm
from pathlib import Path
import h5py

# Figure Style settings for notebook.
import matplotlib as mpl

mpl.rcParams.update({
    'axes.spines.left': False,
    'axes.spines.bottom': False,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'legend.frameon': False,
    'figure.subplot.wspace': .01,
    'figure.subplot.hspace': .01,
    'figure.figsize': (18, 13),
    'ytick.major.left': False,
})
jet = mpl.cm.get_cmap('jet')
jet.set_bad(color='k')


def get_params():
    return {
        "nplanes": 1,
        "nchannels": 1,
        "tau": 3.0,
        "fs": 6.5,
    }


def mmap_to_h5(mmap_path):
    datapath = Path(mmap_path)
    newpath = datapath / 'h5'
    newpath.mkdir(exist_ok=True)
    files = sorted(datapath.glob('*.mmap'))

    for file in files:
        file = str(file)
        yr, dims, T = cm.load_memmap(file)
        y_new = np.reshape(yr, dims + (T,), order='F')

        new_file_path = newpath / (Path(file).stem + '.h5')
        with h5py.File(new_file_path, 'w') as f:
            f.create_dataset('data', data=y_new)


if __name__ == "__main__":
    datapath = Path.home() / "data" / "lbm" / "planes" / "h5" / 'test'
    binpath = Path.home() / 'data' / 'lbm' / 'bin'
    binpath.mkdir(exist_ok=True)
    convert = False
    # right now this includes both rigid + nonrigid mapped files, but we really only want the NR ones
    # we also want to avoid this conversion altogether, but that requires some changes to CaIman
    # or at least this should probably be done with software more suited to file management operations like C
    if convert:
        mmap_to_h5(datapath)

    ops = suite2p.default_ops()
    db = {
        'h5py_key': 'data',
        'look_one_level_down': False,  # whether to look in ALL subfolders when searching for tiffs
        'data_path': [datapath],
        'fast_disk': binpath,  # string which specifies where the binary file will be stored (should be an SSD)
    }
    opsEnd = suite2p.run_s2p(ops=ops, db=db)
