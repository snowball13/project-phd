
include("sphere-evaluation.jl")
using ApproxFun
using PlotlyJS


let

    global function betaVal(l, r)
        if r < 0
            return (l+1)*sqrt(l/(2l+1))
        elseif r > 0
            return l*sqrt((l+1)/(2l+1))
        else
            return im*sqrt(l*(l+1))
        end
    end

    # function clebsch_gordan_coeff_calc(j1, m1, j2, m2, J, M)
    #     if M == m1+m2
    #         cg = (2J+1)*factorial(J+j1-j2)*factorial(J-j1+j2)*factorial(-J+j1+j2)/factorial(J+j1+j2+1)
    #         cg *= factorial(J+M)*factorial(J-M)*factorial(j1-m1)*factorial(j1+m1)*factorial(j2-m2)*factorial(j2+m2)
    #         cg = sqrt(cg)
    #         S = 0.0
    #         check = 0
    #         k = 0
    #         while check < 5
    #             if j1+j2-J-k<0 || j1-m1-k<0 || j2+m2-k<0 || J-j2+m1+k<0 || J-j1-m2+k<0
    #                 check += 1
    #                 k += 1
    #                 continue
    #             end
    #             S += ((-1.0)^k)/(factorial(k)*factorial(j1+j2-J-k)
    #                              *factorial(j1-m1-k)*factorial(j2+m2-k)
    #                              *factorial(J-j2+m1+k)*factorial(J-j1-m2+k))
    #             k += 1
    #         end
    #         return S*cg
    #     end
    #     return 0.0
    # end
    #
    # global function clebsch_gordan_coeff(j1, m1, j2, m2, J, M)
    #     # A Wiki page with a (long) formula for C-G coeffs exists that I assume
    #     # is correct
    #     # return 1.0
    #     if M < 0 && j1 < j2
    #         return clebsch_gordan_coeff_calc(j2,-m2,j1,-m1,J,-M) * (-1.0)^(J-j1-j2)
    #     elseif M < 0
    #         return clebsch_gordan_coeff_calc(j1,-m1,j2,-m2,J,-M) * (-1.0)^(J-j1-j2)
    #     elseif j1 < j2
    #         return clebsch_gordan_coeff_calc(j2,m2,j1,m1,J,M) * (-1.0)^(J-j1-j2)
    #     else
    #         return clebsch_gordan_coeff_calc(j1,m1,j2,m2,J,M)
    #     end
    # end

    #=
    Function that returns the Clebsch-Gordan coefficient
        <L, M-ms; 1, ms | J, M>
    =#
    global function clebsch_gordan_coeff(J, M, L, ms)
        out = 0.0
        if ms == 1
            if J == L+1
                out = sqrt((L+M)*(L+M+1) / ((2L+1)*(2L+2)))
            elseif J == L
                out = - sqrt((L+M)*(L-M+1) / (2L*(L+1)))
            elseif J == L-1
                out = sqrt((L-M)*(L-M+1) / (2L*(2L+1)))
            end
        elseif ms == 0
            if J == L+1
                out = sqrt((L-M+1)*(L+M+1) / ((L+1)*(2L+1)))
            elseif J == L
                out = M / sqrt(L*(L+1))
            elseif J == L-1
                out = - sqrt((L-M)*(L+M) / (L*(2L+1)))
            end
        elseif ms == -1
            if J == L+1
                out = sqrt((L-M)*(L-M+1) / ((2L+1)*(2L+2)))
            elseif J == L
                out = sqrt((L-M)*(L+M+1) / (2L*(L+1)))
            elseif J == L-1
                out = sqrt((L+M)*(L+M+1) / (2L*(2L+1)))
            end
        end
        return out
    end

    function spin_func(r)
        if r == 1 || r == -1
            return [-r, -im, 0] / sqrt(2)
        elseif r == 0
            return [0, 0, 1]
        else
            return [0, 0, 0]
        end
    end

    global function coeff_a(l, m)
        ms = 0
        a = coeff_A(l+1, m-ms) * betaVal(l, 1) * clebsch_gordan_coeff(l, m, l+1, ms)
        a /= (betaVal(l+1, 1) * clebsch_gordan_coeff(l+1, m+1, l+2, ms))
        return a
    end

    global function coeff_b(l, m)
        b = 0
        diff = l - m
        if l > 2 && diff > 1
            ms = -1
            if diff == 2
                ms = 1
            elseif diff == 3
                ms = 0
            end
            b = coeff_B(l-1, m-ms) * betaVal(l, -1) * clebsch_gordan_coeff(l, m, l-1, ms)
            b /= (betaVal(l-1, -1) * clebsch_gordan_coeff(l-1, m+1, l-2, ms))
        end
        return b
    end

    global function coeff_d(l, m)
        ms = 0
        d = coeff_D(l+1, m-ms) * betaVal(l, 1) * clebsch_gordan_coeff(l, m, l+1, ms)
        d /= (betaVal(l+1, 1) * clebsch_gordan_coeff(l+1, m-1, l+2, ms))
        return d
    end

    global function coeff_e(l, m)
        e = 0
        diff = l + m
        if l > 1 && diff > 1
            ms = 1
            if diff == 2
                ms = -1
            elseif diff == 3
                ms = 0
            end
            e = coeff_E(l-1, m-ms) * betaVal(l, -1) * clebsch_gordan_coeff(l, m, l-1, ms)
            e /= (betaVal(l-1, -1) * clebsch_gordan_coeff(l-1, m-1, l-2, ms))
        end
        return e
    end

    global function coeff_f(l, m)
        ms = 0
        f = coeff_F(l+1, m-ms) * betaVal(l, 1) * clebsch_gordan_coeff(l, m, l+1, ms)
        f /= (betaVal(l+1, 1) * clebsch_gordan_coeff(l+1, m, l+2, ms))
        return f
    end

    global function coeff_g(l, m)
        g = 0
        diff = l - m
        if l > 2 && diff > 0 && l+m > 0
            ms = -1
            if diff == 1
                ms = 1
            elseif diff == 2
                ms = 0
            end
            g = coeff_G(l-1, m-ms) * betaVal(l, -1) * clebsch_gordan_coeff(l, m, l-1, ms)
            g /= (betaVal(l-1, -1) * clebsch_gordan_coeff(l-1, m, l-2, ms))
        end
        return g
    end

    global function coeff_h(l, m)
        h = 0
        if !(l == m || (l == 1 && m == 0))
            ms = -1
            if m > 0 && l - m == 1
                ms = 0
            end
            h = coeff_A(l-1,m-ms)*betaVal(l,-1)*clebsch_gordan_coeff(l,m,l-1,ms)
            h += coeff_B(l+1,m-ms)*betaVal(l,1)*clebsch_gordan_coeff(l,m,l+1,ms)
            h -= coeff_a(l,m)*betaVal(l+1,-1)*clebsch_gordan_coeff(l+1,m+1,l,ms)
            h -= coeff_b(l,m)*betaVal(l-1,1)*clebsch_gordan_coeff(l-1,m+1,l,ms)
            h /= betaVal(l,0)*clebsch_gordan_coeff(l,m+1,l,ms)
        end
        return h
    end

    global function coeff_j(l, m)
        j = 0
        if !(l == -m || (l == 1 && m == 0))
            ms = 1
            if m < 0 && l + m == 1
                ms = 0
            end
            j = coeff_D(l-1,m-ms)*betaVal(l,-1)*clebsch_gordan_coeff(l,m,l-1,ms)
            j += coeff_E(l+1,m-ms)*betaVal(l,1)*clebsch_gordan_coeff(l,m,l+1,ms)
            j -= coeff_d(l,m)*betaVal(l+1,-1)*clebsch_gordan_coeff(l+1,m-1,l,ms)
            j -= coeff_e(l,m)*betaVal(l-1,1)*clebsch_gordan_coeff(l-1,m-1,l,ms)
            j /= betaVal(l,0)*clebsch_gordan_coeff(l,m-1,l,ms)
        end
        return j
        # mss = [10,10,10]
        # for ms = 0:1
        #     j = coeff_D(l-1,m-ms)*betaVal(l,-1)*clebsch_gordan_coeff(l,m,l-1,ms)
        #     j += coeff_E(l+1,m-ms)*betaVal(l,1)*clebsch_gordan_coeff(l,m,l+1,ms)
        #     j -= coeff_d(l,m)*betaVal(l+1,-1)*clebsch_gordan_coeff(l+1,m-1,l,ms)
        #     j -= coeff_e(l,m)*betaVal(l-1,1)*clebsch_gordan_coeff(l-1,m-1,l,ms)
        #     j /= betaVal(l,0)*clebsch_gordan_coeff(l,m-1,l,ms)
        #     if abs(j) > 1e-12
        #         mss[ms+2] = ms
        #     end
        # end
        # return mss
    end

    global function coeff_k(l, m)
        k = 0
        if m != 0
            ms = sign(m)
            k = coeff_F(l-1,m-ms)*betaVal(l,-1)*clebsch_gordan_coeff(l,m,l-1,ms)
            k += coeff_G(l+1,m-ms)*betaVal(l,1)*clebsch_gordan_coeff(l,m,l+1,ms)
            k -= coeff_f(l,m)*betaVal(l+1,-1)*clebsch_gordan_coeff(l+1,m,l,ms)
            k -= coeff_g(l,m)*betaVal(l-1,1)*clebsch_gordan_coeff(l-1,m,l,ms)
            k /= betaVal(l,0)*clebsch_gordan_coeff(l,m,l,ms)
        end
        return k
    end

    global function perp_coeff_a(l, m)
        return coeff_a(l, m)'
    end

    global function perp_coeff_b(l, m)
        return coeff_b(l, m)'
    end

    global function perp_coeff_d(l, m)
        return coeff_d(l, m)'
    end

    global function perp_coeff_e(l, m)
        return coeff_e(l, m)'
    end

    global function perp_coeff_f(l, m)
        return coeff_f(l, m)'
        # ms = sign(m)
        # f = betaVal(l, 0) * clebsch_gordan_coeff(l, m, l, ms) * coeff_F(l, m-ms)
        # f -= perp_coeff_k(l, m) * betaVal(l, 1) * clebsch_gordan_coeff(l, m, l+1, ms)
        # f /= (betaVal(l+1, 0) * clebsch_gordan_coeff(l+1, m, l+1, ms))
        # return f
    end

    global function perp_coeff_g(l, m)
        return coeff_g(l, m)'
    end

    global function perp_coeff_h(l, m)
        return coeff_h(l, m)'
    end

    global function perp_coeff_j(l, m)
        return coeff_j(l, m)'
    end

    global function perp_coeff_k(l, m)
        return coeff_k(l, m)'
        # k = 0
        # if m != 0
        #     ms = 0
        #     mr = sign(m)
        #     k = coeff_F(l,m-ms)*betaVal(l,0)*clebsch_gordan_coeff(l,m,l,ms)
        #     k -= (coeff_F(l,m-mr)*betaVal(l,0)*clebsch_gordan_coeff(l+1,m,l+1,ms)
        #             *clebsch_gordan_coeff(l,m,l,mr)
        #             /clebsch_gordan_coeff(l+1,m,l+1,mr))
        #     k /= (betaVal(l,1)*clebsch_gordan_coeff(l,m,l+1,ms)
        #             -(betaVal(l,1)*clebsch_gordan_coeff(l+1,m,l+1,ms)
        #                 *clebsch_gordan_coeff(l,m,l+1,mr)
        #                 /clebsch_gordan_coeff(l+1,m,l+1,mr)))
        # end
        # return k
    end


    function grad_coeff_A(l, m, k)
        if l == 0
            return 0
        else
            return betaVal(l, 1) * clebsch_gordan_coeff(l, m, l+1, -1) * spin_func(-1)[k]
        end
    end

    function grad_coeff_B(l, m, k)
        return betaVal(l, -1) * clebsch_gordan_coeff(l, m, l-1, -1) * spin_func(-1)[k]
    end

    function grad_coeff_D(l, m, k)
        return betaVal(l, 1) * clebsch_gordan_coeff(l, m, l+1, 1) * spin_func(1)[k]
    end

    function grad_coeff_E(l, m, k)
        return betaVal(l, -1) * clebsch_gordan_coeff(l, m, l-1, 1) * spin_func(1)[k]
    end

    function grad_coeff_F(l, m, k)
        return betaVal(l, 1) * clebsch_gordan_coeff(l, m, l+1, 0) * spin_func(0)[k]
    end

    function grad_coeff_G(l, m, k)
        return betaVal(l, -1) * clebsch_gordan_coeff(l, m, l-1, 0) * spin_func(0)[k]
    end

    function grad_perp_coeff_A(l, m, k)
        return betaVal(l, 0) * clebsch_gordan_coeff(l, m, l, -1) * spin_func(-1)[k]
    end

    function grad_perp_coeff_D(l, m, k)
        return betaVal(l, 0) * clebsch_gordan_coeff(l, m, l, 1) * spin_func(1)[k]
    end

    function grad_perp_coeff_F(l, m, k)
        if l == 0
            return 0.0
        end
        return betaVal(l, 0) * clebsch_gordan_coeff(l, m, l, 0) * spin_func(0)[k]
    end

    function grad_matrix_Ak(n, k)
        zerosVec = zeros(2*n + 1) + 0.0im
        if k == 3
            d = copy(zerosVec)
            for j = -n:n
                d[j+n+1] = grad_coeff_F(n, j, k)
            end
            return [zeros(2*n+1, 1) Diagonal(d) zeros(2*n+1, 1)]
        else
            leftdiag = copy(zerosVec)
            rightdiag = copy(zerosVec)
            for j = -n:n
                leftdiag[j+n+1] = grad_coeff_D(n, j, k)
                rightdiag[j+n+1] = grad_coeff_A(n, j, k)
            end
            left = [Diagonal(leftdiag) zeros(2*n+1, 2)]
            right = [zeros(2*n+1, 2) Diagonal(rightdiag)]
            return left + right
        end
    end

    function grad_matrix_Ck(n, k)
        zerosVec = zeros(2*n - 1) + 0.0im
        if k == 3
            d = copy(zerosVec)
            for j = -n:n-2
                d[j+n+1] = grad_coeff_G(n, j+1, k)
            end
            return [zerosVec'; Diagonal(d); zerosVec']
        else
            upperdiag = copy(zerosVec)
            lowerdiag = copy(zerosVec)
            for j = -n:n-2
                upperdiag[j+n+1] = grad_coeff_B(n, j, k)
                lowerdiag[j+n+1] = grad_coeff_E(n, j+2, k)
            end
            upper = Diagonal(upperdiag)
            lower = Diagonal(lowerdiag)
            return [upper; zerosVec'; zerosVec'] + [zerosVec'; zerosVec'; lower]
        end
    end

    function grad_perp_matrix_Bk(n, k)
        if n == 0
            return grad_perp_coeff_F(0, 0, k)
        end
        upperdiag = zeros(2n) + 0.0im
        lowerdiag = copy(upperdiag)
        d = zeros(2n+1) + 0.0im
        for j = -n:n-1
            upperdiag[j+n+1] = grad_perp_coeff_A(n, j, k)
            d[j+n+1] = grad_perp_coeff_F(n, j, k)
            lowerdiag[j+n+1] = grad_perp_coeff_D(n, j+1, k)
        end
        d[end] = grad_perp_coeff_F(n, n, k)
        return Tridiagonal(lowerdiag, d, upperdiag)
    end

    #=
    Outputs the matrix corresponding to the operator ∂Y/∂x_k where k=1,2,3
    and Y is a spherical harmonic.
    The maths behind this stems from the gradient of a spherical harmonic (SH)
    is the sum of two vector spherical harmoncs (VSH). A VSH can be written as a
    sum of SHs with weights as Clebsch-Gorden coeffs times a vector (an
    eigenvector of S3).
    http://scipp.ucsc.edu/~haber/ph216/clebsch.pdf
    =#
    global function grad_sh(N, k)
        l,u = 1,1          # block bandwidths
        λ,μ = 2,2          # sub-block bandwidths: the bandwidths of each block
        cols = rows = 1:2:2N+1  # block sizes
        J = BandedBlockBandedMatrix(0.0im*I, (rows,cols), (l,u), (λ,μ))
        if N == 0
            return J
        end
        J[1,2:4] = grad_matrix_Ak(0,k)
        J[2:4,1] = grad_matrix_Ck(1,k)
        if N == 1
            return J
        end
        for n = 2:N
            J[(n-1)^2+1:n^2,n^2+1:(n+1)^2] = grad_matrix_Ak(n-1,k)
            J[n^2+1:(n+1)^2,(n-1)^2+1:n^2] = grad_matrix_Ck(n,k)
        end
        return J
    end

    #=
    Outputs the matrix corresponding to the operator that yields the kth entry
    of the vector ∇^⟂(Y_l^m), k=1,2,3 where Y^l_m is the (l,m) spherical
    harmonic.
    The maths behind this stems from the perpendicular gradient of a spherical
    harmonic (SH) is a vector spherical harmoncs (VSH). A VSH can be written as
    a sum of SHs with weights as Clebsch-Gorden coeffs times a vector (an
    eigenvector of S3).
    http://scipp.ucsc.edu/~haber/ph216/clebsch.pdf
    =#
    global function grad_perp_sh(N, k)
        l,u = 0,0          # block bandwidths
        λ,μ = 1,1          # sub-block bandwidths: the bandwidths of each block
        cols = rows = 1:2:2N+1  # block sizes
        J = BandedBlockBandedMatrix(0.0im*I, (rows,cols), (l,u), (λ,μ))
        for n = 0:N
            entries = n^2+1:(n+1)^2
            J[entries,entries] = grad_perp_matrix_Bk(n,k)
        end
        return J
    end

    # Matrix representing the Laplacian operator on the vector of the OPs
    # (shperical harmonic polynomials) up to order N
    global function laplacian_sh(N)
        M = (N+1)^2
        D = speye(M)
        for l=0:N
            entries = l^2+1:(l+1)^2
            D[entries, entries] *= -l*(l+1)
        end
        return D
    end

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





tol = 1e-12



N = 6
Dx = grad_sh(N, 1)
Dy = grad_sh(N, 2)
Dz = grad_sh(N, 3)
# I would expect this to just be diagonal (like the Laplacian)
D2 = Dx^2 + Dy^2 + Dz^2
Lap = laplacian_sh(N)
# D2 then matches Lap (ignoring the last 2N+1 rows/cols, since the matrices for
# Dx are not true representations for derivatives at these entries)
B = abs.(D2 - Lap)[1:end-(2N+1), 1:end-(2N+1)]
@test count(i->i>tol, B) == 0

DPerpx = grad_perp_sh(N, 1)
DPerpy = grad_perp_sh(N, 2)
DPerpz = grad_perp_sh(N, 3)
D2Perp = DPerpx^2 + DPerpy^2 + DPerpz^2
# D2 then matches Lap (ignoring the last 2N+1 rows/cols, since the matrices for
# Dx are not true representations for derivatives at these entries)
B = abs.(D2Perp - Lap)[1:end-(2N+1), 1:end-(2N+1)]
@test count(i->i>tol, B) == 0

######

l, m = 3,3
x, y = 0.5, 0.1
z = sqrt(1 - x^2 - y^2)
Y = opEval(N, x, y, z)
DxY = Dx*Y
DPerpxY = DPerpx*Y
@test abs(x*DxY[l^2+l+1+m] - (coeff_a(l, m)*DxY[(l+1)^2+l+1+1+m+1]
                                + coeff_b(l, m)*DxY[(l-1)^2+l-1+1+m+1]
                                + coeff_d(l, m)*DxY[(l+1)^2+l+1+1+m-1]
                                + coeff_e(l, m)*DxY[(l-1)^2+l-1+1+m-1]
                                + coeff_h(l, m)*DPerpxY[l^2+l+1+m+1]
                                + coeff_j(l, m)*DPerpxY[l^2+l+1+m-1])
    ) < tol
@test abs(y*DxY[l^2+l+1+m] - (- im * coeff_a(l, m)*DxY[(l+1)^2+l+1+1+m+1]
                                - im * coeff_b(l, m)*DxY[(l-1)^2+l-1+1+m+1]
                                + im * coeff_d(l, m)*DxY[(l+1)^2+l+1+1+m-1]
                                + im * coeff_e(l, m)*DxY[(l-1)^2+l-1+1+m-1]
                                - im * coeff_h(l, m)*DPerpxY[l^2+l+1+m+1]
                                + im * coeff_j(l, m)*DPerpxY[l^2+l+1+m-1])
    ) < tol
@test abs(z*DxY[l^2+l+1+m] - (coeff_f(l, m)*DxY[(l+1)^2+l+1+1+m]
                                + coeff_g(l, m)*DxY[(l-1)^2+l-1+1+m]
                                + coeff_k(l, m)*DPerpxY[l^2+l+1+m])
    ) < tol

@test abs(x*DPerpxY[l^2+l+1+m] - (perp_coeff_a(l, m)*DPerpxY[(l+1)^2+l+1+1+m+1]
                                + perp_coeff_b(l, m)*DPerpxY[(l-1)^2+l-1+1+m+1]
                                + perp_coeff_d(l, m)*DPerpxY[(l+1)^2+l+1+1+m-1]
                                + perp_coeff_e(l, m)*DPerpxY[(l-1)^2+l-1+1+m-1]
                                + perp_coeff_h(l, m)*DxY[l^2+l+1+m+1]
                                + perp_coeff_j(l, m)*DxY[l^2+l+1+m-1])
    ) < tol
@test abs(y*DPerpxY[l^2+l+1+m] - (- im * perp_coeff_a(l, m)*DPerpxY[(l+1)^2+l+1+1+m+1]
                                    - im * perp_coeff_b(l, m)*DPerpxY[(l-1)^2+l-1+1+m+1]
                                    + im * perp_coeff_d(l, m)*DPerpxY[(l+1)^2+l+1+1+m-1]
                                    + im * perp_coeff_e(l, m)*DPerpxY[(l-1)^2+l-1+1+m-1]
                                    - im * perp_coeff_h(l, m)*DxY[l^2+l+1+m+1]
                                    + im * perp_coeff_j(l, m)*DxY[l^2+l+1+m-1])
    ) < tol
@test abs(z*DPerpxY[l^2+l+1+m] - (perp_coeff_f(l, m)*DPerpxY[(l+1)^2+l+1+1+m]
                                    + perp_coeff_g(l, m)*DPerpxY[(l-1)^2+l-1+1+m]
                                    + perp_coeff_k(l, m)*DxY[l^2+l+1+m])
    ) < tol
