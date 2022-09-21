### Retrieve arguments
path = ARG1

### Settings
reset
set print "-"
set term png truecolor size 500,500
set output "ppo.png"
set grid
set style fill transparent solid 0.25 noborder
set style line 1  lt 1  lw 3 pt 3 ps 0.5
set style line 2  lt 2  lw 3 pt 3 ps 0.5

file = path."/ppo_avg.dat"

### Global png

# Plot score
plot file u 1:6:7 w filledc lt 1 notitle, \
     file u 1:5   w l ls 1 t "score"