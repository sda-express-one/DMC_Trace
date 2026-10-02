# Maps of the data written by `test_lk_sphere --write FILE` over the unit sphere
# (x = azimuth phi, y = cos(theta): an equal-area map).
#   gnuplot -e "data='FILE'; out='lk_sphere.png'" tests/tools/lk_sphere.gp
# columns: 1 phi 2 cos_theta 3 lambda0 4 lambda1 5 lambda2 6 IBZ sector
if (!exists("data")) data = "lk_sphere.dat"
if (!exists("out")) out = "lk_sphere.png"
set terminal pngcairo size 1500,950 font "sans,10"
set output out
set view map
set pm3d map corners2color c1
set xrange [0:360]; set yrange [-1:1]
set xtics 45; set ytics 0.5
set xlabel "phi [deg]"; set ylabel "cos(theta)"
unset key
set multiplot layout 2,2 title "diagonalizeLKHamiltonian on the unit k sphere" font ",13"

set palette rgbformulae 33,13,10
set title "lambda_0 (as returned)"; splot data u 1:2:3 with pm3d
set title "lambda_1 (as returned)"; splot data u 1:2:4 with pm3d
set title "lambda_2 (as returned)"; splot data u 1:2:5 with pm3d

set palette maxcolors 48
set palette rgbformulae 3,11,6
set cbrange [-0.5:47.5]
set title "IBZ fold chosen (8*permutation + sign bits)"; splot data u 1:2:6 with pm3d

unset multiplot
