### Retrieve arguments
path = ARG1

### Settings
reset
set print "-"
set term png truecolor size 1500,500
set output "ppo.png"
set grid
set style fill transparent solid 0.25 noborder
set style line 1  lt 1  lw 3 pt 3 ps 0.5
set style line 2  lt 2  lw 3 pt 3 ps 0.5
set style line 3  lt 3  lw 3 pt 3 ps 0.5
set style line 4  lt 4  lw 3 pt 3 ps 0.5
set style line 5  lt 5  lw 3 pt 3 ps 0.5
set style line 6  lt 6  lw 3 pt 3 ps 0.5
set style line 7  lt 7  lw 3 pt 3 ps 0.5
set style line 8  lt 8  lw 3 pt 3 ps 0.5
set style line 9  lt 9  lw 3 pt 3 ps 0.5
set style line 10 lt 10 lw 3 pt 3 ps 0.5

file = path."/ppo_avg.dat"

### Global png
set multiplot layout 1,3

# Plot score
plot file u 1:6:7 w filledc lt 1 notitle, \
     file u 1:5   w l ls 1 t "score"

# Plot episode length
plot file u 1:12:13 w filledc lt 2 notitle, \
     file u 1:11    w l ls 2 t "ep length"

# Plot entropy
plot file u 1:24:25 w filledc lt 5 notitle, \
     file u 1:23    w l ls 5 t "entropy"