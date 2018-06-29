

using SphericalHarmonics
# using ApproxFun
# using PlotlyJS
using BlockBandedMatrices, Makie, GeometryTypes


let

    function sphere_streamline(linebuffer, ∇ˢf, DT, a, b, pt, h=0.01f0, n=5)
        push!(linebuffer, pt)
        df = normalize(Float32.(abs2.(tangent_func_eval(∇ˢf, pt[1], pt[2], pt[3], DT, a, b))))
        push!(linebuffer, normalize(pt .+ h*df))
        for k=2:n
            cur_pt = last(linebuffer)
            push!(linebuffer, cur_pt)
            df = normalize(Float32.(abs2.(tangent_func_eval(∇ˢf, cur_pt..., DT, a, b))))
            push!(linebuffer, normalize(cur_pt .+ h*df))
        end
        return
    end

    function streamlines(
            scene, ∇ˢf, DT, a, b, pts::AbstractVector{T};
            h=0.1f0, n=5, color = :black, linewidth = 1
        ) where T
        linebuffer = T[]
        sub = Scene(
            scene, ∇ˢf = to_node(∇ˢf),
            h = h, n = 5, color = :black, linewidth = 1
        )
        lines = lift_node(sub[:∇ˢf], to_node(pts), sub[:h], sub[:n]) do ∇ˢf, pts, h, n
            empty!(linebuffer)
            # for point in pts
            #     sphere_streamline(linebuffer, ∇ˢf, DT, a, b, point, h, n)
            # end
            for i=1:length(pts)
                point = pts[i]
                check = true
                if abs(point[3]) > 0.9
                    for j=1:length(pts)
                        if i == j
                            continue
                        else
                            if norm(abs2.(point - pts[j])) < 2e-4
                                check = false
                                break
                            end
                        end
                    end
                end
                if check
                    sphere_streamline(linebuffer, ∇ˢf, DT, a, b, point, h, n)
                end
            end
            linebuffer
        end
        linesegment(sub, lines, color = sub[:color], linewidth = sub[:linewidth])
        sub
    end

    # Plotting function for a function (coeff vector) f
    global function plot_on_sphere(f, DT, a, b)
        # Plot the streamlines on a solid sphere

        # Create arrays of (x,y,z) points
        n = 50
        θ = [0;(0.5:n-0.5)/n;1]
        φ = [(0:2n-2)*2/(2n-1);2]
        x = [cospi(φ)*sinpi(θ) for θ in θ, φ in φ]
        y = [sinpi(φ)*sinpi(θ) for θ in θ, φ in φ]
        z = [cospi(θ) for θ in θ, φ in φ]
        scene = Scene()
        s = Makie.surface(x, y, z, colormap = :viridis, colornorm = (-1.0, 1.0))

        n = 20
        θ = [0;(0.5:n-0.5)/n;1]
        φ = [(0:2n-2)*2/(2n-1);2]
        x = [cospi(φ)*sinpi(θ) for θ in θ, φ in φ]
        y = [sinpi(φ)*sinpi(θ) for θ in θ, φ in φ]
        z = [cospi(θ) for θ in θ, φ in φ]
        pts = vec(Point3f0.(x, y, z))
        lns = streamlines(scene, f, DT, a, b, pts)
        lns[:color] = :black
        return scene, lns

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
        return 2*(2*pi/T)*grad_Jz(N).'
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

        # Create the matrix polynomial for V(x,y,z) = x^2 + y^2 + z
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
    global function linear_SWE(u0, h0, H, dt, maxits, plot=false, filename="")

        M1 = length(h0)
        M2 = length(u0)
        N = round(Int, sqrt(M1) - 1)
        @assert (M1 > 0 && sqrt(M1) - 1 == N) "invalid length of u0 and/or h0"
        @assert M2 == 2M1 "length of u0 should be double that of h0"

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
        if plot
            # Get the clenshaw matrices now, to avoid calculating them repeatedly
            DT, a, b = get_clenshaw_matrices(N)
            scene, lns = plot_on_sphere(u, DT, a, b)
            # record a video
            center!(scene)
            io = VideoStream(scene, ".", filename)
            recordframe!(io)
            for it=1:maxits
                if it % 50 == 0
                    println("Iteration no. : ", it)
                end
                u = A\(u + C*h)
                h -= B*u
                lns[:∇ˢf] = to_node(u)
                yield()
                recordframe!(io)
                sleep(1/30)
            end
            finish(io, "mp4") # could also be gif, webm or mkv
        else
            for it=1:maxits
                u = A\(u + C*h)
                h -= B*u
            end
        end

        return u, h

    end


end
