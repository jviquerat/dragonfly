### Retrieve arguments
baseline   = ARG1
controlled = ARG2

### Settings
reset
set print "-"
set term png truecolor size 1500,350
set output "lorenz_compare.png"
set grid
set style fill transparent solid 0.25 noborder
set style line 1  lt 1  lw 2 pt 3 ps 0.5
set style line 2  lt 2  lw 2 pt 3 ps 0.5

### Global png
set multiplot layout 2,1
set arrow from 0, graph 0 to 0, graph 1 nohead lt rgb "black" lw 2

# Plot x
plot baseline u 1:2 w l ls 1 t "x baseline"
plot controlled u 1:2 w l ls 2 t "x controlled"