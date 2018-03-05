

include("SphericalHarmonics.jl")
include("SphericalHarmonicsGrad.jl")
# using ApproxFun
# using PlotlyJS


let

    # Plotting function for a function (coeff vector) f
    global function plot_on_sphere(f, filename="plot.pdf")

        # Create arrays of (x,y,z) points
        n = 50
        θ = [0;(0.5:n-0.5)/n;1]
        φ = [(0:2n-2)*2/(2n-1);2]
        x = [cospi(φ)*sinpi(θ) for θ in θ, φ in φ]
        y = [sinpi(φ)*sinpi(θ) for θ in θ, φ in φ]
        z = [cospi(θ) for θ in θ, φ in φ]

        # For each (x,y,z) point, evaluate the function f
        w = size(x)[1]
        h = size(x)[2]
        RHSF = zeros(w, h) + 0im
        for i in 1:w
            for j in 1:h
                RHSF[i,j] = funcEval(f, x[i,j], y[i,j], z[i,j])
            end
        end
        RHSF = abs2.(RHSF)

        # Plot
        s = surface(x=x, y=y, z=z, colorscale = "Viridis", surfacecolor = RHSF, cmin = minimum(RHSF), cmax = maximum(RHSF), showscale = false)
        ax = attr(visible = false)
        cam = attr(up = attr(x=0,y=0,z=1), center = attr(x=0,y=0,z=0), eye = attr(x=0.75,y=0.75,z=0.75))
        layout = Layout(width = 500, height = 500, autosize = false, margin = attr(l = 0, r = 0, b = 0, t = 0), scene = attr(xaxis = ax, yaxis = ax, zaxis = ax, camera = cam))
        p = plot(s, layout)
        savefig(p, filename)

    end

    # Heat equation simulation
    global function heat_eqn(u0, h, maxits)

        M = length(u0)
        N = round(Int, sqrt(M) - 1)
        @assert (M > 0 && sqrt(M) - 1 == N) "invalid length of u0"

        # Execute the backward Euler method
        A = Diagonal(ones(M)) - h*laplacian_sh(N)
        u = copy(u0)
        for it=1:maxits
            u = A\u
        end

        plot_on_sphere(u)
        return u

    end

    # Offset heat equation simulation
    global function offset_heat(u0, h, maxits)

        M = length(u0)
        N = round(Int, sqrt(M) - 1)
        @assert (M > 0 && sqrt(M) - 1 == N) "invalid length of u0"

        # Create the matrix polynomial for V(x,y,z) = x^2 + y^2 + z as
        # V(Jx,Jy,Jz) = Jx^2 + Jy^2 + Jz^2
        V = Jx(N)^2 + Jy(N)^2 + Jz(N)

        # Execute the backward Euler method
        A = Diagonal(ones(M)) - h*(laplacian_sh(N) + V)
        u = copy(u0)
        for it=1:maxits
            u = A\u
        end

        plot_on_sphere(u)
        return u

    end


end
