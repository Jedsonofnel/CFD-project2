# ============================================================
# PLOT 1: Cp vs z/c — all three turbulence models
# Uses upper_*/lower_* files from sort_csv.py
# ============================================================

if (!exists("aoa")) aoa = 0

set terminal pngcairo size 1400,900 enhanced font 'Arial,14'

set output sprintf('cp_comparison_aoa%d.png', aoa)
set title sprintf('Pressure Coefficient Distribution ..., {/Symbol a} = %d{/Symbol \260}', aoa) \
          font 'Arial,18'

set xlabel 'x/c'  font 'Arial,16'
set ylabel 'C_p'  font 'Arial,16'

set xrange [0:1]
set yrange [1.2:-1.6]    # inverted y-axis — conventional for Cp plots
set grid lc rgb '#cccccc'
set datafile separator ','
set key top right box opaque

# Skip header
set key autotitle columnheader

# Each model gets one colour; upper and lower share it.
# Upper surface labelled in legend, lower surface unlabelled (notitle).

plot \
  sprintf('upper_ke_cp_aoa%d.csv', aoa) using 1:2 with lines lw 2 lc rgb '#E63946' \
      title 'k-{/Symbol e}', \
  sprintf('lower_ke_cp_aoa%d.csv', aoa) using 1:2 with lines lw 2 lc rgb '#E63946' \
      notitle, \
  sprintf('upper_komega_cp_aoa%d.csv',aoa) using 1:2 with lines lw 2 lc rgb '#457B9D' \
      title 'k-{/Symbol w}', \
  sprintf('lower_komega_cp_aoa%d.csv', aoa) using 1:2 with lines lw 2 lc rgb '#457B9D' \
      notitle, \
  sprintf('upper_rst_cp_aoa%d.csv', aoa) using 1:2 with lines lw 2 lc rgb '#2D6A4F' \
      title 'RST', \
  sprintf('lower_rst_cp%d.csv', aoa)    using 1:2 with lines lw 2 lc rgb '#2D6A4F' \
      notitle


# ============================================================
# PLOT 2: WSS vs z/c — all three turbulence models
# WSS is always positive so no upper/lower split needed in
# the plot itself, but we still plot both surfaces as separate
# series so lines don't cross between them
# ============================================================

set output 'wss_comparison.png'

set title 'Wall Shear Stress Distribution at Span Station x = 10 m, {/Symbol a} = 0{/Symbol \260}' \
          font 'Arial,18'
set xlabel 'x/c'  font 'Arial,16'
set ylabel 'Wall Shear Stress (Pa)'  font 'Arial,16'

set xrange [0:1]
set yrange [0:*]    # not inverted
set grid lc rgb '#cccccc'
set key top right box opaque

plot \
  'upper_ke_wss.csv'     using 1:2 with lines lw 2 lc rgb '#E63946' \
      title 'k-{/Symbol e} (suction surface)', \
  'lower_ke_wss.csv'     using 1:2 with lines lw 2 lc rgb '#E63946' dt 2 \
      title 'k-{/Symbol e} (pressure surface)', \
  'upper_komega_wss.csv' using 1:2 with lines lw 2 lc rgb '#457B9D' \
      title 'k-{/Symbol w} (suction surface)', \
  'lower_komega_wss.csv' using 1:2 with lines lw 2 lc rgb '#457B9D' dt 2 \
      title 'k-{/Symbol w} (pressure surface)', \
  'upper_rst_wss.csv'    using 1:2 with lines lw 2 lc rgb '#2D6A4F' \
      title 'RST (suction surface)', \
  'lower_rst_wss.csv'    using 1:2 with lines lw 2 lc rgb '#2D6A4F' dt 2 \
      title 'RST (pressure surface)'
