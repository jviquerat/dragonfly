### Settings
reset
set print "-"
set term png truecolor size 1000,500
set output "lorenz_3D.png"
set style fill transparent solid 0.25 noborder
set style line 1  lt 1  lw 2 pt 3 ps 0.5
set style line 2  lt 2  lw 2 pt 3 ps 0.5
set style line 3  lt 3  lw 2 pt 3 ps 0.5

### Animated gif
unset grid
unset xtics
unset ytics
unset ztics
unset colorbox
unset border
set xrange [-25:25]
set yrange [-25:25]
set zrange [0:60]

splot "ref.dat" u 2:3:4:2 w l palette lw 1.5 notitle