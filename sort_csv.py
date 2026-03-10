#!/usr/bin/env python3
"""
Split using local mean Y per z/c bin to handle cambered aerofoils.
At each chordwise position the local midpoint (≈ camber line) is
estimated from neighbouring points, so the split works even where
upper/lower Y values converge near the trailing edge.
"""

import csv, os, sys, argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('aoa', type=int, help='Angle of attack (e.g. 0, 15)')
args = parser.parse_args()
AOA = args.aoa

RAW_DIR    = 'raw_data'
SORTED_DIR = 'sorted_data'

os.makedirs(SORTED_DIR, exist_ok=True)  # create if absent

CP_FILES = {
    'ke':     os.path.join(RAW_DIR, f'aoa{AOA}_ke_cp_vs_zoverc.csv'),
    'komega': os.path.join(RAW_DIR, f'aoa{AOA}_komega_cp_vs_zoverc.csv'),
    'rst':    os.path.join(RAW_DIR, f'aoa{AOA}_rst_cp_vs_zoverc.csv'),
}
WSS_FILES = {
    'ke':     os.path.join(RAW_DIR, f'aoa{AOA}_ke_wss_vs_zoverc.csv'),
    'komega': os.path.join(RAW_DIR, f'aoa{AOA}_komega_wss_vs_zoverc.csv'),
    'rst':    os.path.join(RAW_DIR, f'aoa{AOA}_rst_wss_vs_zoverc.csv'),
}

Z_COL=0; VAL_COL=1; Y_COL=3
N_BINS = 50   # number of z/c bins for local camber estimation


def read_csv(filepath):
    rows = []
    with open(filepath, newline='') as f:
        reader = csv.reader(f)
        header = next(reader)
        if len(header) < 4:
            print(f"ERROR: {filepath} needs 4 columns."); sys.exit(1)
        for row in reader:
            try:
                rows.append((float(row[Z_COL]), float(row[Y_COL]), float(row[VAL_COL])))
            except (ValueError, IndexError):
                pass
    return rows


def split_and_normalise(rows, n_bins=N_BINS):
    zs  = np.array([r[0] for r in rows])
    ys  = np.array([r[1] for r in rows])
    vals= np.array([r[2] for r in rows])

    z_min, z_max = zs.min(), zs.max()
    z_norm = (zs - z_min) / (z_max - z_min)

    # Build local camber line: mean Y in each z/c bin
    bins = np.linspace(0, 1, n_bins + 1)
    local_mean_y = np.zeros(len(rows))
    for i in range(n_bins):
        mask = (z_norm >= bins[i]) & (z_norm < bins[i+1])
        if mask.sum() > 0:
            local_mean_y[mask] = ys[mask].mean()
        else:
            # empty bin — fill with global mean (rare)
            local_mean_y[mask] = ys.mean()

    # Handle last bin boundary
    mask_last = z_norm == 1.0
    if mask_last.sum() > 0:
        local_mean_y[mask_last] = ys[mask_last].mean()

    # Classify relative to local camber Y
    upper_mask = ys >= local_mean_y
    lower_mask = ~upper_mask

    def extract_sorted(mask):
        pts = sorted(zip(z_norm[mask], vals[mask]), key=lambda r: r[0])
        return pts

    return extract_sorted(upper_mask), extract_sorted(lower_mask), z_min, z_max


def write_csv(rows, filepath, value_name):
    with open(filepath, 'w', newline='') as f:
        f.write(f"z_over_c,{value_name}\n")
        for z, val in rows:
            f.write(f"{z:.8f},{val:.8f}\n")


def process(files, value_name):
    print(f"\n── {value_name} ──────────────────────────")
    for label, fname in files.items():
        if not os.path.exists(fname):
            print(f"  WARNING: {fname} not found, skipping."); continue
        rows = read_csv(fname)
        upper, lower, z_min, z_max = split_and_normalise(rows)
        upper_out = os.path.join(SORTED_DIR, f"upper_{label}_{value_name.lower()}_aoa{AOA}.csv")
        lower_out = os.path.join(SORTED_DIR, f"lower_{label}_{value_name.lower()}_aoa{AOA}.csv")
        write_csv(lower, lower_out, value_name)
        print(f"  {label}: {len(upper)} upper, {len(lower)} lower"
              f"  |  z raw [{z_min:.3f}, {z_max:.3f}] → [0,1]"
              f"  →  {upper_out}, {lower_out}")

process(CP_FILES,  "Cp")
process(WSS_FILES, "WSS")
print("\nDone. Run plots.gp")
