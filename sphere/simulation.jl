

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
        s = surface(x=x, y=y, z=z, colorscale = "Viridis", surfacecolor = RHSF,
                    cmin = minimum(RHSF), cmax = maximum(RHSF),
                    showscale = false)
        ax = attr(visible = false)
        cam = attr(up = attr(x=0,y=0,z=1), center = attr(x=0,y=0,z=0),
                    eye = attr(x=0.75,y=0.75,z=0.75))
        layout = Layout(width = 500, height = 500, autosize = false,
                        margin = attr(l = 0, r = 0, b = 0, t = 0),
                        scene = attr(xaxis = ax, yaxis = ax, zaxis = ax,
                        camera = cam))
        p = plot(s, layout)
        savefig(p, filename)

    end


    # Operator matrix K corresponding to the cross product of k acting on u,
    # vector of coeffs for expansion in grad (tangent space) basis.
    global function outward_normal_cross_product(N)
        l,u = 0,0          # block bandwidths
        λ,μ = 1,1          # sub-block bandwidths: the bandwidths of each block
        cols = rows = 2:4:2(2N+1)  # block sizes
        K = BandedBlockBandedMatrix(0.0I, (rows,cols), (l,u), (λ,μ))
        for n = 0:N
            r = sum(rows[1:n])
            for i=r+1:2:r+rows[n+1]
                K[i,i+1] = -1.0
                K[i+1,i] = 1.0
            end
        end
        return K
    end

    # Operator matrix corresponding to the divergence of u, vector of coeffs
    # for expansion in grad (tangent space) basis.
    global function div_sh(N)
        l,u = 0,0          # block bandwidths
        λ,μ = 0,2N          # sub-block bandwidths: the bandwidths of each block
        rows = 1:2:(2N+1)  # block sizes
        cols = 2*rows
        D = BandedBlockBandedMatrix(zeros(sum(rows),sum(cols)), (rows,cols), (l,u), (λ,μ))
        for n=0:N
            c = sum(cols[1:n])
            i = sum(rows[1:n]) + 1
            for j=c+1:2:c+cols[n+1]
                D[i,j] = -n*(n+1)
                i += 1
            end
        end
        return D
    end

    global function grad_sh_2(N)
        l,u = 0,0          # block bandwidths
        λ,μ = 2N,0          # sub-block bandwidths: the bandwidths of each block
        cols = 1:2:(2N+1)  # block sizes
        rows = 2*cols
        G = BandedBlockBandedMatrix(zeros(sum(rows),sum(cols)), (rows,cols), (l,u), (λ,μ))
        for n=0:N
            r = sum(rows[1:n])
            j = sum(cols[1:n]) + 1
            for i=r+1:2:r+rows[n+1]
                G[i,j] = 1.0
                j += 1
            end
        end
        return G
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

    # Linearised Shallow Water Equations.
    # u0, h0 should be given as vectors containing coefficients of their
    # expansion in the tangent basis (∇Y, ∇⟂Y)
    # We start with h ≡ h0 := 1
    global function linear_SWE(u0, h0, dt, maxits)

        # Our eqn is u_t + f k x u = 0 where k is the outward normal vector.

        M1 = length(h0)
        M2 = length(u0)
        N = round(Int, sqrt(M1) - 1)
        @assert (M1 > 0 && sqrt(M1) - 1 == N) "invalid length of u0 and/or h0"
        @assert M2 == 2M1 "length of u0 should be double that of h0"

        # Parameters
        f = 1
        H = 1

        # Operator matrices
        K = outward_normal_cross_product(N)
        D = div_sh(N)
        G = grad_sh_2(N)

        # Execute the backward Euler method
        A = dt*f*K + dt^2*H*G*D
        B = dt*H*D
        C = dt*G
        u = copy(u0)
        h = copy(h0)
        for it=1:maxits
            u = A\(u + C*h)
            h -= B*u
        end

        # plot_on_sphere(u)
        return u, h

    end


end
