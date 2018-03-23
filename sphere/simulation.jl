

include("SphericalHarmonics.jl")
include("SphericalHarmonicsGrad.jl")
# using ApproxFun
# using PlotlyJS
using Makie, GeometryTypes


let

    function sphere_streamline(linebuffer, ∇ˢf, pt, h=0.01f0, n=5)
        push!(linebuffer, pt)
        ∇ˢfeval = abs.(tangent_func_eval(∇ˢf, pt[1], pt[2], pt[3]))
        mag = norm(∇ˢfeval)
        df = ∇ˢfeval/mag
        push!(linebuffer, normalize(pt .+ h*df))
        for k=2:n
            cur_pt = last(linebuffer)
            push!(linebuffer, cur_pt)
            df = normalize(abs.(tangent_func_eval(∇ˢf, cur_pt...)))
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

    # Plotting function for a function (coeff vector) f
    global function plot_on_sphere(f, as_streamlines=true)

        # Create arrays of (x,y,z) points
        n = 20
        θ = [0;(0.5:n-0.5)/n;1]
        φ = [(0:2n-2)*2/(2n-1);2]
        x = [cospi(φ)*sinpi(θ) for θ in θ, φ in φ]
        y = [sinpi(φ)*sinpi(θ) for θ in θ, φ in φ]
        z = [cospi(θ) for θ in θ, φ in φ]
        pts = vec(Point3f0.(x, y, z))

        if as_streamlines
            # Plot
            scene = Scene()
            lns = streamlines(scene, f, pts)
            # those can be changed interactively:
            lns[:color] = :black
            lns[:h] = 0.06
            lns[:linewidth] = 1.0
            for i = linspace(0.01, 0.1, 100)
                lns[:h] = i
                yield()
            end
        else
            # Scalar valued function (density type plot)
            gridw = size(x)[1]
            gridh = size(x)[2]
            F = zeros(gridw, gridh) + 0im
            for i in 1:gridw
                for j in 1:gridh
                    F[i,j] = funcEval(f, x[i,j], y[i,j], z[i,j])
                end
            end
            F = abs2.(F)
            scene = Scene()
            s = Makie.surface(x, y, z, image = F, colormap = :viridis, colornorm = (-1.0, 1.0))
        end

    end


    # Operator matrix K corresponding to the cross product of k (outward unit
    # normal vector, (x,y,z)) acting on u given as a vector of coeffs for
    # expansion in grad (tangent space) basis.
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

    #=
    Operator matrix corresponding to the gradient of h, where h is given as a
    vector of coeffs for expansion in the SH basis. The output matrix is given
    so that the action of G*h yeilds the vector of ceofficients in the
    expansion in the tangent basis.
    =#
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

    #=
    Operator matrix corresponding to the coriolis frequency f=2Ωcos(θ)=2Ωz.
    We use the gradient jacobi operator for z here, as f acts on u which is
    given in the tangent space.
    =#
    global function coriolis_freq(N)
        T = 60*60*24
        return 2*(2*pi/T)*grad_Jz(N)
    end


    # Heat equation simulation
    global function heat_eqn(u0, h, maxits, plot=false)

        M = length(u0)
        N = round(Int, sqrt(M) - 1)
        @assert (M > 0 && sqrt(M) - 1 == N) "invalid length of u0"

        # Execute the backward Euler method
        A = Diagonal(ones(M)) - h*laplacian_sh(N)
        u = copy(u0)
        for it=1:maxits
            u = A\u
        end

        if plot
            plot_on_sphere(u)
        end
        return u

    end

    # Offset heat equation simulation
    global function offset_heat(u0, h, maxits, plot=false)

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

        if plot
            plot_on_sphere(u)
        end
        return u

    end

    # Linearised Shallow Water Equations.
    # u0, h0 should be given as vectors containing coefficients of their
    # expansion in the tangent basis (∇Y, ∇⟂Y)
    global function linear_SWE(u0, h0, dt, maxits, plot=false)

        # Our eqn is u_t + f k x u = 0 where k is the outward normal vector.

        M1 = length(h0)
        M2 = length(u0)
        N = round(Int, sqrt(M1) - 1)
        @assert (M1 > 0 && sqrt(M1) - 1 == N) "invalid length of u0 and/or h0"
        @assert M2 == 2M1 "length of u0 should be double that of h0"

        # Constants
        H = norm(abs.(funcEval(h0, 1, 0, 0))) # Base/reference height

        # Operator matrices
        K = outward_normal_cross_product(N)
        D = div_sh(N)
        G = grad_sh_2(N)
        F = coriolis_freq(N)

        # Execute the backward Euler method
        A = I + dt*F*K + dt^2*H*G*D
        B = dt*H*D
        C = dt*G
        u = copy(u0)
        h = copy(h0)
        for it=1:maxits
            u = A\(u + C*h)
            h -= B*u
        end

        if plot
            plot_on_sphere(h, false)
        end
        return u, h

    end


end
