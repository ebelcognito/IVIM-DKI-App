"""Headless check of the fitting engine: `IVIM-DKI.exe --selftest [data.nii.gz ...]`.

Fits the bundled SNR-60 simulations (or the given files) with the published settings over the whole volume and
reports the RMSE of the varied parameter against the ground truth. Writes selftest_result.txt next to the app."""
import glob
import os
import sys

import numpy as np
import nibabel as nib

import engine

VARIED = {'Simulation-I': 0, 'Simulation-II': 1, 'Simulation-III': 2, 'Simulation-IV': 3}


def app_dir():
    return os.path.dirname(sys.executable) if getattr(sys, 'frozen', False) else os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main(paths):
    paths = paths or sorted(glob.glob(os.path.join(app_dir(), 'test_data', '*_SNR60.nii.gz')))
    lines = []
    for path in paths:
        stem = path[:-7]
        img = nib.load(path)
        data = img.get_fdata()
        b = np.loadtxt(stem + '.bval').ravel()
        maps, info = engine.fit(data, b, np.ones(data.shape[:3], bool), img.header.get_zooms()[:3],
                                engine.INITIAL['Simulation'], engine.LOWER['Simulation (D* floor 0.0001)'],
                                engine.PAPER_UPPER)
        line = f"{os.path.basename(path)}: {info['status']}, {info['iterations']} iterations, {info['seconds']:.1f} s ({info['backend']})"
        truth_path = stem + '_ground_truth.nii.gz'
        name = os.path.basename(stem).replace('_SNR60', '')
        if os.path.exists(truth_path) and name in VARIED:
            j = VARIED[name]
            t = nib.load(truth_path).get_fdata()[..., j]
            rmse = 100 * np.sqrt(np.mean((maps[..., j] - t) ** 2)) / t.mean()
            line += f"; RMSE of {engine.NAMES[j]} = {rmse:.2f}%"
        print(line, flush=True)
        lines.append(line)
    with open(os.path.join(app_dir(), 'selftest_result.txt'), 'w') as f:
        f.write('\n'.join(lines) + '\n')
    return 0 if lines else 1


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
