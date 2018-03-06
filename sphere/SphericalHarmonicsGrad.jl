#=

Functions for determining the gradient and perpendicular gradient of spherical
harmonics (OPs) on the unit sphere.

The maths behind this stems from the gradient of a spherical harmonic (SH)
is the sum of two vector spherical harmoncs (VSH). A VSH can be written as a
sum of SHs with weights as Clebsch-Gorden coeffs times a vector (an
eigenvector of S3).
http://scipp.ucsc.edu/~haber/ph216/clebsch.pdf

=#

include("SphericalHarmonics.jl")


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


    #=
    Coefficients from the relations of multiplication by x,y,z etc on
    grad(Y_l^m) and grad⟂(Y_l^m). Used as Entries in the Jx, Jy, Jz jacobi
    matrices
    =#

    global function coeff_a(l, m)
        ms = 0
        a = coeff_A(l+1, m-ms) * betaVal(l, 1) * clebsch_gordan_coeff(l, m, l+1, ms)
        a /= (betaVal(l+1, 1) * clebsch_gordan_coeff(l+1, m+1, l+2, ms))
        return a
    end

    global function coeff_b(l, m)
        b = 0
        diff = l - m
        if l > 1 && diff > 1
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
        if l > 1 && l != abs(m)
            diff = l - m
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
        if l != m
            ms = -1
            if m == 0 || (m > 0 && l - m == 1)
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
        if l != -m
            ms = 1
            if m == 0 || (m < 0 && l + m == 1)
                ms = 0
            end
            j = coeff_D(l-1,m-ms)*betaVal(l,-1)*clebsch_gordan_coeff(l,m,l-1,ms)
            j += coeff_E(l+1,m-ms)*betaVal(l,1)*clebsch_gordan_coeff(l,m,l+1,ms)
            j -= coeff_d(l,m)*betaVal(l+1,-1)*clebsch_gordan_coeff(l+1,m-1,l,ms)
            j -= coeff_e(l,m)*betaVal(l-1,1)*clebsch_gordan_coeff(l-1,m-1,l,ms)
            j /= betaVal(l,0)*clebsch_gordan_coeff(l,m-1,l,ms)
        end
        return j
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
    end


    #=
    Sub-Matrices for the Jacobi operator matrices
    =#
    global function grad_jacobi_Ax(n)
        len = 3(2n+1)
        leftdiag = zeros(len) + 0.0im
        rightdiag = copy(leftdiag)
        # Gather non-zero entries
        index = 1
        for k = -n:n
            leftdiag[index:index+2] = coeff_d(n, k)
            rightdiag[index:index+2] = coeff_a(n, k)
            index += 3
        end
        # Create the sub-sub-matrix
        left = [Diagonal(leftdiag) zeros(len, 6)]
        right = [zeros(len, 6) Diagonal(rightdiag)]
        A = left + right
        # Assemble full sub matrix, exploiting the symmetry of the system
        zerosMatrix = zeros(size(A))
        return [A zerosMatrix; zerosMatrix conj(A)]
    end

    global function grad_jacobi_Ay(n)
        len = 3(2n+1)
        leftdiag = zeros(len)
        rightdiag = copy(leftdiag)
        # Gather non-zero entries
        index = 1
        for k = -n:n
            leftdiag[index:index+2] = coeff_d(n, k)
            rightdiag[index:index+2] = coeff_a(n, k)
            index += 3
        end
        # Create the sub-sub-matrix
        left = [Diagonal(leftdiag) zeros(len, 6)]
        right = [zeros(len, 6) Diagonal(rightdiag)]
        A = im * (left - right)
        # Assemble full sub matrix, exploiting the symmetry of the system
        zerosMatrix = zeros(size(A))
        return [A zerosMatrix; zerosMatrix conj(A)]
    end

    global function grad_jacobi_Az(n)
        len = 3(2n+1)
        d = zeros(len)
        # Gather non-zero entries
        index = 1
        for k = -n:n
            d[index:index+2] = coeff_f(n, k)
            index += 3
        end
        # Create the sub-sub-matrix
        A = [zeros(len,3) Diagonal(d) zeros(len,3)]
        # Assemble full sub matrix, exploiting the symmetry of the system
        zerosMatrix = zeros(size(A))
        return [A zerosMatrix; zerosMatrix conj(A)]
    end

    global function grad_jacobi_Bx(n)
        if n == 0
            return zeros(6,6)+0.0im
        end
        len = 3(2n)
        superdiag = zeros(len)+0.0im
        subdiag = copy(superdiag)
        # Gather non-zero entries
        superdiag[1:3] = coeff_h(n, -n)
        index = 1
        for k = -n+1:n-1
            superdiag[index+3:index+5] = coeff_h(n, k)
            subdiag[index:index+2] = coeff_j(n, k)
            index += 3
        end
        subdiag[end-2:end] = coeff_j(n, n)
        # Create the sub-sub-matrix
        super = [zeros(len,3) Diagonal(superdiag); zeros(3,len+3)]
        sub = [zeros(3,len+3); Diagonal(subdiag) zeros(len,3)]
        B = super + sub
        # Assemble full sub matrix, exploiting the symmetry of the system
        zerosMatrix = zeros(size(B))
        return [zerosMatrix B; conj(B) zerosMatrix]
    end

    global function grad_jacobi_By(n)
        len = 3(2n)
        superdiag = zeros(len)
        subdiag = copy(superdiag)
        # Gather non-zero entries
        superdiag[1:3] = coeff_h(n, -n)
        index = 1
        for k = -n+1:n-1
            superdiag[index+3:index+5] = coeff_h(n, k)
            subdiag[index:index+2] = coeff_j(n, k)
            index += 3
        end
        subdiag[end-2:end] = coeff_j(n, n)
        # Create the sub-sub-matrix
        super = [zeros(len,3) Diagonal(superdiag); zeros(3,len+3)]
        sub = [zeros(3,len+3); Diagonal(subdiag) zeros(len,3)]
        B = im * (-super + sub)
        # Assemble full sub matrix, exploiting the symmetry of the system
        zerosMatrix = zeros(size(B))
        return [zerosMatrix B; conj(B) zerosMatrix]
    end

    global function grad_jacobi_Bz(n)
        len = 3(2n)
        d = zeros(len)
        # Gather non-zero entries
        index = 1
        for k = -n:n
            d[index:index+2] = coeff_k(n, k)
            index += 3
        end
        # Create the sub-sub-matrix
        B = Diagonal(d)
        # Assemble full sub matrix, exploiting the symmetry of the system
        zerosMatrix = zeros(size(B))
        return [zerosMatrix B; conj(B) zerosMatrix]
    end

    global function grad_jacobi_Cx(n)
        len = 3(2n-1)
        upperdiag = zeros(len)
        lowerdiag = copy(upperdiag)
        # Gather non-zero entries
        index = 1
        for k = -n:n-2
            upperdiag[index:index+2] = coeff_b(n, k)
            lowerdiag[index:index+2] = coeff_e(n, k+2)
            index += 3
        end
        # Create the sub-sub-matrix
        upper = [Diagonal(upperdiag); zeros(6, len)]
        lower = [zeros(6, len); Diagonal(lowerdiag)]
        C = upper + lower
        # Assemble full sub matrix, exploiting the symmetry of the system
        zerosMatrix = zeros(size(C))
        return [C zerosMatrix; zerosMatrix conj(C)]
    end

    global function grad_jacobi_Cy(n)
        len = 3(2n-1)
        upperdiag = zeros(len)
        lowerdiag = copy(upperdiag)
        # Gather non-zero entries
        index = 1
        for k = -n:n-2
            upperdiag[index:index+2] = coeff_b(n, k)
            lowerdiag[index:index+2] = coeff_e(n, k+2)
            index += 3
        end
        # Create the sub-sub-matrix
        upper = [Diagonal(upperdiag); zeros(6, len)]
        lower = [zeros(6, len) Diagonal(lowerdiag)]
        C = im * (-upper + lower)
        # Assemble full sub matrix, exploiting the symmetry of the system
        zerosMatrix = zeros(size(C))
        return [C zerosMatrix; zerosMatrix conj(C)]
    end

    global function grad_jacobi_Cz(n)
        len = 3(2n-1)
        d = zeros(len)
        # Gather non-zero entries
        index = 1
        for k = -n+1:n-1
            d[index:index+2] = coeff_g(n, k)
            index += 3
        end
        # Create the sub-sub-matrix
        C = [zeros(3,len); Diagonal(d); zeros(3,len)]
        # Assemble full sub matrix, exploiting the symmetry of the system
        zerosMatrix = zeros(size(C))
        return [C zerosMatrix; zerosMatrix conj(C)]
    end

    #=
    Jacobi matrices for the tangent space vector basis
    =#
    global function grad_Jx(N)
        l,u = 1,1          # block bandwidths
        cols = rows = 6:12:6(2N+1)  # block sizes
        J = BlockBandedMatrix(0.0im*I, (rows,cols), (l,u))
        J[1:rows[1], 1:cols[1]] = grad_jacobi_Bx(0)
        if N > 0
            for n = 1:N
                row = sum(rows[1:n])
                col = sum(cols[1:n])
                J[row-rows[n]+1:row, col+1:col+cols[n+1]] = grad_jacobi_Ax(n-1)
                J[row+1:row+rows[n+1], col+1:col+cols[n+1]] = grad_jacobi_Bx(n)
                J[row+1:row+rows[n+1], col-cols[n]+1:col] = grad_jacobi_Cx(n)
            end
        end
        return J
    end


    #=
    Entries (coefficients) in the Sub-Matrices used in the gradient matrix
    operator functions
    =#

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


    #=
    Sub-Matrices for use in the gradient matrix operator functions
    =#

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

    #=
    Matrix representing the Laplacian operator on the vector of the OPs
    (shperical harmonic polynomials) up to order N
    =#
    global function laplacian_sh(N)
        M = (N+1)^2
        D = speye(M)
        for l=0:N
            entries = l^2+1:(l+1)^2
            D[entries, entries] *= -l*(l+1)
        end
        return D
    end


    #=
    Ouputs the tangent basis vectors (∇Y, ∇⟂Y) up to order N evaluated at a
    point x,y,z on the unit sphere.Returns the vector [∇P_0,...,∇P_N]
    where ∇P_l = [∇Y_l^-l, ∇⟂Y_l^-l,...,∇Y_l^l, ∇⟂Y_l^l].
    =#
    global function tangent_basis_eval(N, x, y, z)
        Dx = grad_sh(N+1, 1)
        Dy = grad_sh(N+1, 2)
        Dz = grad_sh(N+1, 3)
        DPerpx = grad_perp_sh(N+1, 1)
        DPerpy = grad_perp_sh(N+1, 2)
        DPerpz = grad_perp_sh(N+1, 3)
        Y = opEval(N+1, x, y, z)

        # entries = 1:6:(6(N+1)^2)
        # out = Vector{Complex{Float64}}(6*(N+1)^2)
        # out[entries] = (Dx*Y)[1:end-(2N+3)]
        # out[entries+1] = (Dy*Y)[1:end-(2N+3)]
        # out[entries+2] = (Dz*Y)[1:end-(2N+3)]
        # out[entries+3] = (DPerpx*Y)[1:end-(2N+3)]
        # out[entries+4] = (DPerpy*Y)[1:end-(2N+3)]
        # out[entries+5] = (DPerpz*Y)[1:end-(2N+3)]

        DxY = Dx*Y
        DyY = Dy*Y
        DzY = Dz*Y
        DPerpxY = DPerpx*Y
        DPerpyY = DPerpy*Y
        DPerpzY = DPerpz*Y
        out = Vector{Complex{Float64}}(6*(N+1)^2)
        index = 1
        for l = 0:N
            for m = -l:l
                out[index] = DxY[l^2+l+m+1]
                out[index+1] = DyY[l^2+l+m+1]
                out[index+2] = DzY[l^2+l+m+1]
                index += 3
            end
            for m = -l:l
                out[index] = DPerpxY[l^2+l+m+1]
                out[index+1] = DPerpyY[l^2+l+m+1]
                out[index+2] = DPerpzY[l^2+l+m+1]
                index += 3
            end
        end

        return out
    end

end




#====#
# Testing


tol = 1e-10



N = 10
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

x, y = 0.5, 0.1
z = sqrt(1 - x^2 - y^2)
Y = opEval(N, x, y, z)
DxY = Dx*Y
DPerpxY = DPerpx*Y
for l = 1:6
    for m = -l:l
        @test abs(x*DxY[l^2+l+1+m] - (coeff_a(l, m)*DxY[(l+1)^2+l+1+1+m+1]
                                        + coeff_b(l, m)*DxY[(l-1)^2+l-1+1+m+1]
                                        + coeff_d(l, m)*DxY[(l+1)^2+l+1+1+m-1]
                                        + coeff_e(l, m)*DxY[max(1,(l-1)^2+l-1+1+m-1)]
                                        + coeff_h(l, m)*DPerpxY[l^2+l+1+m+1]
                                        + coeff_j(l, m)*DPerpxY[l^2+l+1+m-1])
            ) < tol
        @test abs(y*DxY[l^2+l+1+m] - (- im * coeff_a(l, m)*DxY[(l+1)^2+l+1+1+m+1]
                                        - im * coeff_b(l, m)*DxY[(l-1)^2+l-1+1+m+1]
                                        + im * coeff_d(l, m)*DxY[(l+1)^2+l+1+1+m-1]
                                        + im * coeff_e(l, m)*DxY[max(1,(l-1)^2+l-1+1+m-1)]
                                        - im * coeff_h(l, m)*DPerpxY[l^2+l+1+m+1]
                                        + im * coeff_j(l, m)*DPerpxY[l^2+l+1+m-1])
            ) < tol
        @test abs(z*DxY[l^2+l+1+m] - (coeff_f(l, m)*DxY[(l+1)^2+l+1+1+m]
                                        + coeff_g(l, m)*DxY[max(1, (l-1)^2+l-1+1+m)]
                                        + coeff_k(l, m)*DPerpxY[l^2+l+1+m])
            ) < tol

        @test abs(x*DPerpxY[l^2+l+1+m] - (perp_coeff_a(l, m)*DPerpxY[(l+1)^2+l+1+1+m+1]
                                        + perp_coeff_b(l, m)*DPerpxY[(l-1)^2+l-1+1+m+1]
                                        + perp_coeff_d(l, m)*DPerpxY[(l+1)^2+l+1+1+m-1]
                                        + perp_coeff_e(l, m)*DPerpxY[max(1,(l-1)^2+l-1+1+m-1)]
                                        + perp_coeff_h(l, m)*DxY[l^2+l+1+m+1]
                                        + perp_coeff_j(l, m)*DxY[l^2+l+1+m-1])
            ) < tol
        @test abs(y*DPerpxY[l^2+l+1+m] - (- im * perp_coeff_a(l, m)*DPerpxY[(l+1)^2+l+1+1+m+1]
                                            - im * perp_coeff_b(l, m)*DPerpxY[(l-1)^2+l-1+1+m+1]
                                            + im * perp_coeff_d(l, m)*DPerpxY[(l+1)^2+l+1+1+m-1]
                                            + im * perp_coeff_e(l, m)*DPerpxY[max(1,(l-1)^2+l-1+1+m-1)]
                                            - im * perp_coeff_h(l, m)*DxY[l^2+l+1+m+1]
                                            + im * perp_coeff_j(l, m)*DxY[l^2+l+1+m-1])
            ) < tol
        @test abs(z*DPerpxY[l^2+l+1+m] - (perp_coeff_f(l, m)*DPerpxY[(l+1)^2+l+1+1+m]
                                            + perp_coeff_g(l, m)*DPerpxY[max(1, (l-1)^2+l-1+1+m)]
                                            + perp_coeff_k(l, m)*DxY[l^2+l+1+m])
            ) < tol
    end
end

#####

a = x*tangent_basis_eval(N,x,y,z)
b = grad_Jx(N)*tangent_basis_eval(N,x,y,z)
c = abs.(a[1:6N^2] - b[1:6N^2])
@test count(i->i>tol, c) == 0
