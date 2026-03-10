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

N_BINS = 50   # chordwise bins for camber-line estimation

# ── Colours for each turbulence model ──────────
MODEL_COLOURS = {
    'ke':     '#E63946',
    'komega': '#457B9D',
    'rst':    '#2D6A4F',
}

# ── Human-readable labels ──────────────────────
MODEL_LABELS = {
    'ke':     r'$k$-$\varepsilon$',
    'komega': r'$k$-$\omega$',
    'rst':    'RST',
}

AOA_COLOURS = {
    0:    '#1b4332',
    5:    '#40916c',
    10:   '#f4a261',
    12.5: '#e76f51',
    15:   '#9b2226',
}


# ─────────────────────────────────────────────
# STAGE 2 — PLOT CONFIGURATION
# ─────────────────────────────────────────────
# Each entry in PLOT_CONFIGS defines one output PNG.
#
# Keys:
#   output    – filename (saved into PLOT_DIR)
#   variable  – 'cp' or 'wss'
#   title     – plot title string
#   ylabel    – y-axis label
#   invert_y  – True for Cp (conventional aerofoil presentation)
#   yrange    – (min, max) or None for auto
#   series    – list of dicts, each with:
#                   model   – 'ke', 'komega', 'rst'
#                   aoa     – numeric AoA value
#                   label   – legend label (upper surface entry)
#                   colour  – hex colour (optional; falls back to
#                             MODEL_COLOURS then AOA_COLOURS)
#
# Upper surface is always solid; lower always dashed.
# Missing data is silently skipped.

PLOT_CONFIGS = {

    # ── Cp: all models at AoA = 0 ──────────────
    'cp_all_models_aoa0': {
        'output':   'cp_all_models_aoa0.png',
        'variable': 'cp',
        'title':    r'$C_p$ — all turbulence models, $\alpha = 0°$',
        'ylabel':   r'$C_p$',
        'invert_y': True,
        'yrange':   (-1.6, 1.2),
        'series': [
            {'model': 'ke',     'aoa': 0, 'label': r'$k$-$\varepsilon$'},
            {'model': 'komega', 'aoa': 0, 'label': r'$k$-$\omega$'},
            {'model': 'rst',    'aoa': 0, 'label': 'RST'},
        ],
    },

    # ── Cp: all models at AoA = 15 ─────────────
    'cp_all_models_aoa15': {
        'output':   'cp_all_models_aoa15.png',
        'variable': 'cp',
        'title':    r'$C_p$ — all turbulence models, $\alpha = 15°$',
        'ylabel':   r'$C_p$',
        'invert_y': True,
        'yrange':   (-2.5, 1.2),
        'series': [
            {'model': 'ke',     'aoa': 15, 'label': r'$k$-$\varepsilon$'},
            {'model': 'komega', 'aoa': 15, 'label': r'$k$-$\omega$'},
            {'model': 'rst',    'aoa': 15, 'label': 'RST'},
        ],
    },

    # ── Cp: RST across all AoAs ─────────────────
    'cp_rst_all_aoa': {
        'output':   'cp_rst_all_aoa.png',
        'variable': 'cp',
        'title':    r'$C_p$ — RST model, all $\alpha$',
        'ylabel':   r'$C_p$',
        'invert_y': True,
        'yrange':   None,
        'series': [
            {'model': 'rst', 'aoa': 0,    'label': r'$\alpha = 0°$',    'colour': AOA_COLOURS[0]},
            {'model': 'rst', 'aoa': 5,    'label': r'$\alpha = 5°$',    'colour': AOA_COLOURS[5]},
            {'model': 'rst', 'aoa': 10,   'label': r'$\alpha = 10°$',   'colour': AOA_COLOURS[10]},
            {'model': 'rst', 'aoa': 12.5, 'label': r'$\alpha = 12.5°$', 'colour': AOA_COLOURS[12.5]},
            {'model': 'rst', 'aoa': 15,   'label': r'$\alpha = 15°$',   'colour': AOA_COLOURS[15]},
        ],
    },

    # ── WSS: all models at AoA = 0 ─────────────
    'wss_all_models_aoa0': {
        'output':   'wss_all_models_aoa0.png',
        'variable': 'wss',
        'title':    r'WSS — all turbulence models, $\alpha = 0°$',
        'ylabel':   'Wall Shear Stress (Pa)',
        'invert_y': False,
        'yrange':   (0, None),
        'series': [
            {'model': 'ke',     'aoa': 0, 'label': r'$k$-$\varepsilon$'},
            {'model': 'komega', 'aoa': 0, 'label': r'$k$-$\omega$'},
            {'model': 'rst',    'aoa': 0, 'label': 'RST'},
        ],
    },

    # ── WSS: RST across all AoAs ────────────────
    'wss_rst_all_aoa': {
        'output':   'wss_rst_all_aoa.png',
        'variable': 'wss',
        'title':    r'WSS — RST model, all $\alpha$',
        'ylabel':   'Wall Shear Stress (Pa)',
        'invert_y': False,
        'yrange':   (0, None),
        'series': [
            {'model': 'rst', 'aoa': 0,    'label': r'$\alpha = 0°$',    'colour': AOA_COLOURS[0]},
            {'model': 'rst', 'aoa': 5,    'label': r'$\alpha = 5°$',    'colour': AOA_COLOURS[5]},
            {'model': 'rst', 'aoa': 7.5,  'label': r'$\alpha = 7.5°$',  'colour': AOA_COLOURS[7.5]},
            {'model': 'rst', 'aoa': 10,   'label': r'$\alpha = 10°$',   'colour': AOA_COLOURS[10]},
            {'model': 'rst', 'aoa': 12.5, 'label': r'$\alpha = 12.5°$', 'colour': AOA_COLOURS[12.5]},
            {'model': 'rst', 'aoa': 15,   'label': r'$\alpha = 15°$',   'colour': AOA_COLOURS[15]},
        ],
    },
}


# ─────────────────────────────────────────────
# STAGE 1 HELPERS
# ─────────────────────────────────────────────

def aoa_str(aoa):
    """Format AoA as string suitable for filenames: 12.5 → '12.5', 0 → '0'."""
    return str(aoa).rstrip('0').rstrip('.') if '.' in str(aoa) else str(aoa)


def raw_filename(variable, model, aoa):
    template = f'aoa{aoa_str(aoa)}_{model}_{variable}_vs_zoverc.csv'
    return os.path.join(RAW_DIR, template)


def sorted_filename(surface, model, variable, aoa):
    fname = f'{surface}_{model}_{variable}_aoa{aoa_str(aoa)}.csv'
    return os.path.join(SORTED_DIR, fname)


def find_col(headers_lower, keywords):
    """Return index of first header containing any keyword, or None."""
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

        col_z   = find_col(headers_lower, z_keywords)
        col_y   = find_col(headers_lower, y_keywords)
        # For val, skip any column already claimed by z or y
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


def split_and_normalise(rows, n_bins=N_BINS):
    zs   = np.array([r[0] for r in rows])
    ys   = np.array([r[1] for r in rows])
    vals = np.array([r[2] for r in rows])

    z_min, z_max = zs.min(), zs.max()
    z_norm = (zs - z_min) / (z_max - z_min)

    bins = np.linspace(0, 1, n_bins + 1)
    local_mean_y = np.zeros(len(rows))

    for i in range(n_bins):
        mask = (z_norm >= bins[i]) & (z_norm < bins[i + 1])
        local_mean_y[mask] = ys[mask].mean() if mask.sum() > 0 else ys.mean()

    mask_last = z_norm == 1.0
    if mask_last.sum() > 0:
        local_mean_y[mask_last] = ys[mask_last].mean()

    upper_mask = ys >= local_mean_y
    lower_mask = ~upper_mask

    def extract(mask):
        return sorted(zip(z_norm[mask], vals[mask]), key=lambda r: r[0])

    return extract(upper_mask), extract(lower_mask)


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
    Scan all (aoa × model × variable) combinations.
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

                # Write sorted CSVs
                write_sorted_csv(
                    upper,
                    sorted_filename('upper', model, variable, aoa),
                    variable.upper(),
                )
                write_sorted_csv(
                    lower,
                    sorted_filename('lower', model, variable, aoa),
                    variable.upper(),
                )

                # Append to master records
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

        for series in cfg['series']:
            model    = series['model']
            aoa      = series['aoa']
            label    = series['label']
            colour   = resolve_colour(series)
            variable = cfg['variable']

            for surface in ('upper', 'lower'):
                subset = df[
                    (df['aoa']      == aoa)    &
                    (df['model']    == model)  &
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

            # One legend entry per series (solid line; label describes the series)
            legend_handles.append(
                mlines.Line2D([], [], color=colour, linestyle='-',
                              linewidth=2.0, label=label)
            )

        if not any_data:
            print(f'  WARNING: no data for plot "{plot_key}" — skipping.')
            plt.close(fig)
            continue

        # Surface style legend entries (shared across all series)
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
        if df is None:
            # Re-build from sorted CSVs if running stage 2 standalone
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
        # Filename pattern: {surface}_{model}_{variable}_aoa{aoa}.csv
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
            next(reader)  # skip header
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
