### Settings
reset
set print "-"
set term png truecolor size 1500,500
set output "lorenz_2D.png"
set grid
set style fill transparent solid 0.25 noborder
set style line 1  lt 1  lw 2 pt 3 ps 0.5
set style line 2  lt 2  lw 2 pt 3 ps 0.5

### Global png
set multiplot layout 2,1

# Plot x, y, z
set arrow from 0,-20 to 0,30 nohead lw 2
plot "ref.dat"    u 1:2 w l ls 1 t "x baseline"
plot "lorenz.dat" u 1:2 w l ls 2 t "x controlled"