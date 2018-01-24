
include("sphere-evaluation.jl")
using PlotlyJS


n = 10

θ = [0;(0.5:n-0.5)/n;1]
φ = [(0:2n-2)*2/(2n-1);2]
x = [cospi(φ)*sinpi(θ) for θ in θ, φ in φ]
y = [sinpi(φ)*sinpi(θ) for θ in θ, φ in φ]
z = [cospi(θ) for θ in θ, φ in φ]

N = 3
f = 1:(N+1)^2
w = size(x)[1]
h = size(x)[2]
RHSF = zeros(w, h) + 0im
for i in 1:w
    for j in 1:h
        RHSF[i,j] = funcEval(f, x[i,j], y[i,j], z[i,j])
    end
end
RHSF = abs2.(RHSF)

s = surface(x=x, y=y, z=z, colorscale = "Viridis", surfacecolor = RHSF, cmin = minimum(RHSF), cmax = maximum(RHSF), showscale = false)
ax = attr(visible = false)
cam = attr(up = attr(x=0,y=0,z=1), center = attr(x=0,y=0,z=0), eye = attr(x=0.75,y=0.75,z=0.75))
layout = Layout(width = 500, height = 500, autosize = false, margin = attr(l = 0, r = 0, b = 0, t = 0), scene = attr(xaxis = ax, yaxis = ax, zaxis = ax, camera = cam))
p = plot(s, layout)
savefig(p, "plot.pdf")
