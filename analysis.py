#!/usr/bin/env python3
"""
analyse.py — Two-stage CFD aerofoil post-processor.

Stage 1: Scan raw_data/ for all known CSVs, sort upper/lower surfaces,
         write sorted_data/ CSVs, and build a master pandas DataFrame.

Stage 2: Read PLOT_CONFIGS declarative dict and render each plot as a PNG.

Usage:
    python analyse.py          # runs both stages
    python analyse.py --stage 1
    python analyse.py --stage 2
"""

import os, sys, csv, argparse
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────

RAW_DIR    = 'raw_data'
SORTED_DIR = 'sorted_data'
PLOT_DIR   = 'plots'

AOAS    = [0, 5, 7.5, 10, 12.5, 15]
MODELS  = ['ke', 'komega', 'rst']

VARIABLES = {
    'cp': {
        # Case-insensitive substrings to match column headers.
        # First column whose header contains any listed keyword wins.
        'z_keywords':   ['z_over_c', 'z/c'],
        'y_keywords':   ['position[y]', 'y (m)', 'y(m)', 'coord y'],
        'val_keywords': ['pressure coefficient', 'cp', 'c_p'],
    },
    'wss': {
        'z_keywords':   ['z_over_c', 'z/c'],
        'y_keywords':   ['position[y]', 'y (m)', 'y(m)', 'coord y'],
        'val_keywords': ['wall shear stress', 'wss', 'shear stress'],
    },
}

# Number of chordwise bins for camber-line estimation.
# Higher = finer resolution; the Gaussian smooth below handles noise.
N_BINS = 200

# Gaussian smoothing sigma (in bin units) applied to the camber line estimate.
# Increase if upper/lower are still misclassified near LE/TE.
CAMBER_SMOOTH_SIGMA = 5

# ── Colours ────────────────────────────────────
MODEL_COLOURS = {
    'ke':     '#E63946',
    'komega': '#457B9D',
    'rst':    '#2D6A4F',
}

MODEL_LABELS = {
    'ke':     r'$k$-$\varepsilon$',
    'komega': r'$k$-$\omega$',
    'rst':    'RST',
}

AOA_COLOURS = {
    0:    '#03045e',
    5:    '#0077b6',
    7.5:  '#00b4d8',
    10:   '#f4a261',
    12.5: '#e76f51',
    15:   '#9b2226',
}


# ─────────────────────────────────────────────
# STAGE 2 — PLOT CONFIGURATION (built programmatically)
# ─────────────────────────────────────────────
#
# Each entry defines one PNG.  Keys:
#   output     – filename saved into PLOT_DIR
#   variable   – 'cp' or 'wss'
#   title      – plot title
#   ylabel     – y-axis label
#   invert_y   – True for Cp
#   yrange     – (lo, hi) or None for auto  [None boundary = auto on that side]
#   upper_only – if True, only the suction surface is plotted (no dashed lower)
#   series     – list of dicts:
#                    model   – 'ke', 'komega', 'rst'
#                    aoa     – numeric AoA
#                    label   – legend label
#                    colour  – optional hex; falls back to MODEL_COLOURS / AOA_COLOURS

def _build_plot_configs():
    cfgs = {}

    # ── Per-AOA: all three models ───────────────────────────────────────────
    # Both surfaces shown; colour distinguishes turbulence model.
    for aoa in AOAS:
        aoa_label = f'{aoa:g}'

        cfgs[f'cp_all_models_aoa{aoa_label}'] = {
            'output':     f'cp_all_models_aoa{aoa_label}.png',
            'variable':   'cp',
            'title':      rf'$C_p$ — all turbulence models, $\alpha = {aoa_label}°$',
            'ylabel':     r'$C_p$',
            'invert_y':   True,
            'yrange':     None,
            'upper_only': False,
            'series': [
                {'model': 'ke',     'aoa': aoa, 'label': MODEL_LABELS['ke']},
                {'model': 'komega', 'aoa': aoa, 'label': MODEL_LABELS['komega']},
                {'model': 'rst',    'aoa': aoa, 'label': MODEL_LABELS['rst']},
            ],
        }

        cfgs[f'wss_all_models_aoa{aoa_label}'] = {
            'output':     f'wss_all_models_aoa{aoa_label}.png',
            'variable':   'wss',
            'title':      rf'WSS — all turbulence models, $\alpha = {aoa_label}°$',
            'ylabel':     'Wall Shear Stress (Pa)',
            'invert_y':   False,
            'yrange':     (0, None),
            'upper_only': False,
            'series': [
                {'model': 'ke',     'aoa': aoa, 'label': MODEL_LABELS['ke']},
                {'model': 'komega', 'aoa': aoa, 'label': MODEL_LABELS['komega']},
                {'model': 'rst',    'aoa': aoa, 'label': MODEL_LABELS['rst']},
            ],
        }

    # ── Per-model: all AOAs ─────────────────────────────────────────────────
    # Suction surface only (upper_only=True); colour distinguishes AoA.
    for model in MODELS:
        model_label = MODEL_LABELS[model]

        cfgs[f'cp_{model}_all_aoa'] = {
            'output':     f'cp_{model}_all_aoa.png',
            'variable':   'cp',
            'title':      rf'$C_p$ — {model_label}, all $\alpha$',
            'ylabel':     r'$C_p$',
            'invert_y':   True,
            'yrange':     None,
            'upper_only': True,
            'series': [
                {'model': model, 'aoa': aoa,
                 'label': rf'$\alpha = {aoa:g}°$',
                 'colour': AOA_COLOURS[aoa]}
                for aoa in AOAS
            ],
        }

        cfgs[f'wss_{model}_all_aoa'] = {
            'output':     f'wss_{model}_all_aoa.png',
            'variable':   'wss',
            'title':      rf'WSS — {model_label}, all $\alpha$',
            'ylabel':     'Wall Shear Stress (Pa)',
            'invert_y':   False,
            'yrange':     (0, None),
            'upper_only': True,
            'series': [
                {'model': model, 'aoa': aoa,
                 'label': rf'$\alpha = {aoa:g}°$',
                 'colour': AOA_COLOURS[aoa]}
                for aoa in AOAS
            ],
        }

    return cfgs

PLOT_CONFIGS = _build_plot_configs()


# ─────────────────────────────────────────────
# STAGE 1 HELPERS
# ─────────────────────────────────────────────

def aoa_str(aoa):
    """Format AoA for filenames: 12.5 → '12.5', 0 → '0'."""
    return str(aoa).rstrip('0').rstrip('.') if '.' in str(aoa) else str(aoa)


def raw_filename(variable, model, aoa):
    return os.path.join(RAW_DIR,
                        f'aoa{aoa_str(aoa)}_{model}_{variable}_vs_zoverc.csv')


def sorted_filename(surface, model, variable, aoa):
    return os.path.join(SORTED_DIR,
                        f'{surface}_{model}_{variable}_aoa{aoa_str(aoa)}.csv')


def find_col(headers_lower, keywords):
    """Return index of first header containing any keyword (case-insensitive)."""
    for i, h in enumerate(headers_lower):
        for kw in keywords:
            if kw in h:
                return i
    return None


def read_raw_csv(filepath, z_keywords, val_keywords, y_keywords):
    rows = []
    with open(filepath, newline='') as f:
        reader = csv.reader(f)
        raw_header = next(reader)
        headers_lower = [h.strip().lower() for h in raw_header]

        col_z = find_col(headers_lower, z_keywords)
        col_y = find_col(headers_lower, y_keywords)

        # Value column: first match not already claimed by z or y
        col_val = None
        for i, h in enumerate(headers_lower):
            if i in (col_z, col_y):
                continue
            for kw in val_keywords:
                if kw in h:
                    col_val = i
                    break
            if col_val is not None:
                break

        if None in (col_z, col_y, col_val):
            print(f'  ERROR: {filepath} — could not identify columns.')
            print(f'    headers : {raw_header}')
            print(f'    z→{col_z}, y→{col_y}, val→{col_val}')
            return None

        for row in reader:
            try:
                rows.append((
                    float(row[col_z]),
                    float(row[col_y]),
                    float(row[col_val]),
                ))
            except (ValueError, IndexError):
                pass
    return rows


def split_and_normalise(rows, n_bins=N_BINS, smooth_sigma=CAMBER_SMOOTH_SIGMA):
    """
    Split points into upper/lower surface using a smoothed local camber line.

    The camber line is estimated as the mean Y in each chordwise bin, then
    smoothed with a Gaussian filter.  This is robust to:
      - Cambered aerofoils at non-zero AoA
      - Sparse bins near LE/TE
      - Noisy surface meshes (e.g. aoa5 intermediate AoAs)
    """
    zs   = np.array([r[0] for r in rows])
    ys   = np.array([r[1] for r in rows])
    vals = np.array([r[2] for r in rows])

    z_min, z_max = zs.min(), zs.max()
    z_norm = (zs - z_min) / (z_max - z_min)

    # Per-bin mean Y
    bins = np.linspace(0, 1, n_bins + 1)
    bin_mean_y = np.full(n_bins, np.nan)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (z_norm >= lo) & (z_norm < hi)
        if i == n_bins - 1:
            mask |= (z_norm == 1.0)
        if mask.sum() > 0:
            bin_mean_y[i] = ys[mask].mean()

    # Fill empty bins by linear interpolation
    nan_mask = np.isnan(bin_mean_y)
    if nan_mask.any() and (~nan_mask).sum() >= 2:
        x_idx = np.arange(n_bins)
        bin_mean_y[nan_mask] = np.interp(
            x_idx[nan_mask], x_idx[~nan_mask], bin_mean_y[~nan_mask]
        )
    elif nan_mask.all():
        bin_mean_y[:] = ys.mean()

    # Smooth the camber line estimate
    smoothed = gaussian_filter1d(bin_mean_y, sigma=smooth_sigma, mode='nearest')

    # Map each point to its bin's smoothed camber Y
    bin_indices = np.clip(
        np.searchsorted(bins[1:], z_norm, side='right'), 0, n_bins - 1
    )
    local_camber_y = smoothed[bin_indices]

    upper_mask = ys >= local_camber_y

    def extract(mask):
        return sorted(zip(z_norm[mask], vals[mask]), key=lambda r: r[0])

    return extract(upper_mask), extract(~upper_mask)


def write_sorted_csv(pts, filepath, value_name):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w', newline='') as f:
        f.write(f'z_over_c,{value_name}\n')
        for z, val in pts:
            f.write(f'{z:.8f},{val:.8f}\n')


# ─────────────────────────────────────────────
# STAGE 1 — BUILD DATAFRAME
# ─────────────────────────────────────────────

def build_dataframe():
    """
    Scan all (aoa x model x variable) combinations.
    For each found file: sort upper/lower, write sorted CSVs,
    append rows to master list.
    Returns a DataFrame with columns:
        aoa, model, variable, surface, z_over_c, value
    """
    os.makedirs(SORTED_DIR, exist_ok=True)
    records = []
    missing = []

    for variable, vcfg in VARIABLES.items():
        z_keywords   = vcfg['z_keywords']
        y_keywords   = vcfg['y_keywords']
        val_keywords = vcfg['val_keywords']

        for aoa in AOAS:
            for model in MODELS:
                fpath = raw_filename(variable, model, aoa)

                if not os.path.exists(fpath):
                    missing.append(fpath)
                    continue

                rows = read_raw_csv(fpath, z_keywords, val_keywords, y_keywords)
                if rows is None or len(rows) == 0:
                    print(f'  WARNING: {fpath} empty or unreadable — skipping.')
                    continue

                upper, lower = split_and_normalise(rows)

                write_sorted_csv(upper,
                                 sorted_filename('upper', model, variable, aoa),
                                 variable.upper())
                write_sorted_csv(lower,
                                 sorted_filename('lower', model, variable, aoa),
                                 variable.upper())

                for z, val in upper:
                    records.append((aoa, model, variable, 'upper', z, val))
                for z, val in lower:
                    records.append((aoa, model, variable, 'lower', z, val))

                print(f'  ✓  {fpath}  →  {len(upper)} upper, {len(lower)} lower')

    if missing:
        print(f'\n  Gracefully skipped {len(missing)} missing file(s):')
        for m in missing:
            print(f'    {m}')

    df = pd.DataFrame(records,
                      columns=['aoa', 'model', 'variable', 'surface',
                                'z_over_c', 'value'])
    return df


# ─────────────────────────────────────────────
# STAGE 2 — PLOTTING
# ─────────────────────────────────────────────

def resolve_colour(series_cfg):
    if 'colour' in series_cfg:
        return series_cfg['colour']
    model = series_cfg.get('model')
    aoa   = series_cfg.get('aoa')
    return MODEL_COLOURS.get(model, AOA_COLOURS.get(aoa, '#333333'))


def render_plots(df):
    os.makedirs(PLOT_DIR, exist_ok=True)

    for plot_key, cfg in PLOT_CONFIGS.items():
        fig, ax = plt.subplots(figsize=(10, 6))
        legend_handles = []
        any_data = False
        upper_only = cfg.get('upper_only', False)
        surfaces   = ('upper',) if upper_only else ('upper', 'lower')

        for series in cfg['series']:
            model    = series['model']
            aoa      = series['aoa']
            label    = series['label']
            colour   = resolve_colour(series)
            variable = cfg['variable']

            for surface in surfaces:
                subset = df[
                    (df['aoa']      == aoa)      &
                    (df['model']    == model)    &
                    (df['variable'] == variable) &
                    (df['surface']  == surface)
                ].sort_values('z_over_c')

                if subset.empty:
                    continue

                linestyle = '-' if surface == 'upper' else '--'
                ax.plot(
                    subset['z_over_c'],
                    subset['value'],
                    color=colour,
                    linestyle=linestyle,
                    linewidth=2.0,
                )
                any_data = True

            legend_handles.append(
                mlines.Line2D([], [], color=colour, linestyle='-',
                              linewidth=2.0, label=label)
            )

        if not any_data:
            print(f'  WARNING: no data for plot "{plot_key}" — skipping.')
            plt.close(fig)
            continue

        # Surface convention legend (only when both surfaces are shown)
        if not upper_only:
            legend_handles.append(
                mlines.Line2D([], [], color='grey', linestyle='-',
                              linewidth=1.5, label='Suction surface')
            )
            legend_handles.append(
                mlines.Line2D([], [], color='grey', linestyle='--',
                              linewidth=1.5, label='Pressure surface')
            )

        ax.set_xlabel('x/c', fontsize=13)
        ax.set_ylabel(cfg['ylabel'], fontsize=13)
        ax.set_title(cfg['title'], fontsize=14, pad=12)
        ax.set_xlim(0, 1)

        if cfg['yrange'] is not None:
            ax.set_ylim(cfg['yrange'])
            if cfg['invert_y'] and ax.get_ylim()[0] < ax.get_ylim()[1]:
                ax.invert_yaxis()
        elif cfg['invert_y']:
            ax.invert_yaxis()

        ax.grid(True, color='#cccccc', linewidth=0.7)
        ax.legend(handles=legend_handles, fontsize=11,
                  loc='best', framealpha=0.9)

        fig.tight_layout()
        out_path = os.path.join(PLOT_DIR, cfg['output'])
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f'  ✓  {out_path}')


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='CFD aerofoil post-processor')
    parser.add_argument('--stage', type=int, choices=[1, 2],
                        help='Run only stage 1 (ingest) or stage 2 (plot). '
                             'Default: run both.')
    args = parser.parse_args()

    df = None

    if args.stage in (None, 1):
        print('\n── Stage 1: ingesting raw data ──────────────────────────')
        df = build_dataframe()
        print(f'\n  DataFrame: {len(df)} rows across '
              f'{df["aoa"].nunique()} AoAs, '
              f'{df["model"].nunique()} models, '
              f'{df["variable"].nunique()} variables.')

    if args.stage in (None, 2):
        print('\n── Stage 2: rendering plots ─────────────────────────────')
        print(f'  {len(PLOT_CONFIGS)} plots configured.')
        if df is None:
            print('  (stage 1 not run — reading sorted_data/ to rebuild DataFrame)')
            df = rebuild_from_sorted()
        render_plots(df)

    print('\nDone.')


def rebuild_from_sorted():
    """Reconstruct the master DataFrame by reading sorted_data/ CSVs."""
    records = []
    if not os.path.isdir(SORTED_DIR):
        print(f'  ERROR: {SORTED_DIR}/ not found. Run stage 1 first.')
        sys.exit(1)

    for fname in os.listdir(SORTED_DIR):
        if not fname.endswith('.csv'):
            continue
        # Pattern: {surface}_{model}_{variable}_aoa{aoa}.csv
        parts = fname.replace('.csv', '').split('_')
        if len(parts) < 4:
            continue
        surface  = parts[0]
        model    = parts[1]
        variable = parts[2]
        aoa_part = parts[3].replace('aoa', '')
        try:
            aoa = float(aoa_part)
        except ValueError:
            continue

        fpath = os.path.join(SORTED_DIR, fname)
        with open(fpath, newline='') as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                try:
                    records.append((aoa, model, variable, surface,
                                    float(row[0]), float(row[1])))
                except (ValueError, IndexError):
                    pass

    return pd.DataFrame(records,
                        columns=['aoa', 'model', 'variable', 'surface',
                                 'z_over_c', 'value'])


if __name__ == '__main__':
    main()
