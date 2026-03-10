# CFD Project 2 Plots

- Author: Jed Nelson <jn1422@ic.ac.uk>
- Date: 2026-03-10

For AOA study.

## Usage:

To run the code to generate the plots afresh:

1. Make sure you have `python` on your system with `numpy`, `scipy`, `matplotlib` and `pandas` installed.
2. Run `python analysis.py`
3. Profit
4. Check `plots/` for the plots.

## Modification

Each plot corresponds to a config in the `cfgs` dictionary inside
`_build_plot_configs()` for a declarative way to specify the plots and what
should be displayed.  Just add a new dictionary entry to `cfgs` and it'll be
spit out.  Or ask Claude to do it for you!
