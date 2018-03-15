
include("simulation.jl")

using Makie, GeometryTypes

function sphere_streamline(linebuffer, ∇ˢf, pt, h=0.01f0, n=5)
    push!(linebuffer, pt)
    df = normalize(∇ˢf(pt[1], pt[2], pt[3]))
    push!(linebuffer, normalize(pt .+ h*df))
    for k=2:n
        cur_pt = last(linebuffer)
        push!(linebuffer, cur_pt)
        df = normalize(∇ˢf(cur_pt...))
        println(cur_pt...)
        push!(linebuffer, normalize(cur_pt .+ h*df))
    end
    return
end

function streamlines(
        scene, ∇ˢf, pts::AbstractVector{T};
        h=0.01f0, n=5, color = :black, linewidth = 1
    ) where T
    linebuffer = T[]
    sub = Scene(
        scene,
        h = h, n = 5, color = :black, linewidth = 1
    )
    lines = lift_node(to_node(∇ˢf), to_node(pts), sub[:h], sub[:n]) do ∇ˢf, pts, h, n
        empty!(linebuffer)
        for point in pts
            sphere_streamline(linebuffer, ∇ˢf, point, h, n)
        end
        linebuffer
    end
    linesegment(sub, lines, color = sub[:color], linewidth = sub[:linewidth])
    sub
end

# needs to be in a function for ∇ˢf to be fast and inferable
function test()
    n = 20
    f   = (x,y,z) -> x*exp(cos(y)*z)
    ∇f  = (x,y,z) -> Point3f0(exp(cos(y)*z), -sin(y)*z*x*exp(cos(y)*z), x*cos(y)*exp(cos(y)*z))
    ∇ˢf = (x,y,z) -> ∇f(x,y,z) - Point3f0(x,y,z)*dot(Point3f0(x,y,z), ∇f(x,y,z))
    θ = [0;(0.5:n-0.5)/n;1]
    φ = [(0:2n-2)*2/(2n-1);2]
    x = [cospi(φ)*sinpi(θ) for θ in θ, φ in φ]
    y = [sinpi(φ)*sinpi(θ) for θ in θ, φ in φ]
    z = [cospi(θ) for θ in θ, φ in φ]

    pts = vec(Point3f0.(x, y, z))
    scene = Scene()
    lns = streamlines(scene, ∇ˢf, pts)
    # those can be changed interactively:
    lns[:color] = :black
    lns[:h] = 0.06
    lns[:linewidth] = 1.0
    for i = linspace(0.01, 0.1, 100)
        lns[:h] = i
        yield()
    end
end
test()

using GeometryTypes, Colors, Makie

function arrows(
        parent, points::AbstractVector{Pair{Point3f0, Point3f0}};
        arrowhead = Pyramid(Point3f0(0, 0, -0.5), 1f0, 1f0), arrowtail = nothing, arrowsize = 0.3,
        linecolor = :black, arrowcolor = linecolor, linewidth = 1,
        linestyle = nothing, scale = Vec3f0(1)
    )
    linesegment(parent, points, color = linecolor, linewidth = linewidth, linestyle = linestyle, scale = scale)
    rotations = map(points) do p
        p1, p2 = p
        dir = p2 .- p1
        GLVisualize.rotation_between(Vec3f0(0, 0, 1), Vec3f0(dir))
    end
    meshscatter(
        last.(points), marker = arrowhead, markersize = arrowsize, color = arrowcolor,
        rotations = rotations, scale = scale
    )
end
n = 20
f   = (x,y,z) -> x*exp(cos(y)*z)
∇f  = (x,y,z) -> Point3f0(exp(cos(y)*z), -sin(y)*z*x*exp(cos(y)*z), x*cos(y)*exp(cos(y)*z))
∇ˢf = (x,y,z) -> ∇f(x,y,z) - Point3f0(x,y,z)*dot(Point3f0(x,y,z), ∇f(x,y,z))

θ = [0;(0.5:n-0.5)/n;1]
φ = [(0:2n-2)*2/(2n-1);2]
x = [cospi(φ)*sinpi(θ) for θ in θ, φ in φ]
y = [sinpi(φ)*sinpi(θ) for θ in θ, φ in φ]
z = [cospi(θ) for θ in θ, φ in φ]

pts = vec(Point3f0.(x, y, z))
∇ˢF = vec(∇ˢf.(x, y, z))

arr = map(pts, ∇ˢF) do p, dir
    p => p .+ (dir .* 0.2f0)
end
scene = Scene();
surface(scene , x, y, z)
arrows(scene, arr, arrowsize = 0.03, linecolor = :gray, linewidth = 3)






# using PlotlyJS
#
#
# n = 10
#
# θ = [0;(0.5:n-0.5)/n;1]
# φ = [(0:2n-2)*2/(2n-1);2]
# x = [cospi(φ)*sinpi(θ) for θ in θ, φ in φ]
# y = [sinpi(φ)*sinpi(θ) for θ in θ, φ in φ]
# z = [cospi(θ) for θ in θ, φ in φ]
#
# N = 3
# f = 1:(N+1)^2
# w = size(x)[1]
# h = size(x)[2]
# RHSF = zeros(w, h) + 0im
# for i in 1:w
#     for j in 1:h
#         RHSF[i,j] = funcEval(f, x[i,j], y[i,j], z[i,j])
#     end
# end
# RHSF = abs2.(RHSF)
#
# s = surface(x=x, y=y, z=z, colorscale = "Viridis", surfacecolor = RHSF, cmin = minimum(RHSF), cmax = maximum(RHSF), showscale = false)
# ax = attr(visible = false)
# cam = attr(up = attr(x=0,y=0,z=1), center = attr(x=0,y=0,z=0), eye = attr(x=0.75,y=0.75,z=0.75))
# layout = Layout(width = 500, height = 500, autosize = false, margin = attr(l = 0, r = 0, b = 0, t = 0), scene = attr(xaxis = ax, yaxis = ax, zaxis = ax, camera = cam))
# p = plot(s, layout)
# savefig(p, "plot.pdf")
