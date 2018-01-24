# Script to obtain a point evaluation of the Nth set of OPs for the unit sphere

using ApproxFun
using Base.Test
using Base.LinAlg


let

    #=
    Functions to obtain the coefficients used in the matrices for
    J^x, J^y, J^z
    =#

    # Function outputting the constant for the (l,m) spherical harmonic
    # polynomial
    global function alphaVal(l, m)
        c = sqrt((2*l + 1) * gamma(l - m + 1) / (4*pi*gamma(l + m + 1)))
        ctilde = 1.0
        if m < 0
            m = abs(m)
            ctilde = (-1.0)^m * gamma(l - m + 1) / gamma(l + m + 1)
        end
        chat = gamma(l + m + 1) / (gamma(l + 1) * (-2.0)^m)
        return c * chat * ctilde
    end

    global function AtildeVal(l, m)
        A = (m + 0.5) / (l + 0.5)
        A *= (l + m + 2) / (l + 1)
        A *= (l + m + 1) / (2*m + 2)
        A *= (m + 1) / (2*m + 1)
        return A
    end

    global function BtildeVal(l, m)
        B = - (m + 0.5) / (l + 0.5)
        B *= l / (2*m + 2)
        B *= (m + 1) / (2*m + 1)
        return B
    end

    global function DtildeVal(l, m)
        D = - ((l - m + 1) * (l - m + 2)) / (2 * (m - 0.5) * (l + 0.5))
        D *= (2*m - 1) / (l + 1)
        return D
    end

    global function EtildeVal(l, m)
        E = ((l + m - 1) * (l + m)) / (2 * (m - 0.5) * (l + 0.5))
        E *= (l * (2*m - 1)) / ((l + m) * (l + m - 1))
        return E
    end

    global function FtildeVal(l, m)
        return (l - m + 1) * (l + m + 1) / ((2*l + 1) * (l + 1))
    end

    global function GtildeVal(l, m)
        G = 1.0
        if (abs(m) <= l - 1)
            G = l / (2*l + 1)
        else
            G = 0.0
        end
        return G
    end

    # The following coeff functions give the coefficients used in the
    # relations for x*Y^m_l, y*Y^m_l and z*Y^m_l where Y^m_l is the l,m
    # spherical harmonic polynomial. These are then used as the non-zero entries
    # in our system matrices.
    global function coeff_A(l, m)
        A = 1.0
        if (m >= 0)
            A = AtildeVal(l, m)
        else
            A = DtildeVal(l, abs(m))
        end
        A *= alphaVal(l, m) / (2 * alphaVal(l+1, m+1))
        return A
    end

    global function coeff_B(l, m)
        B = 1.0
        if (m >= 0)
            if (l - abs(m) - 2 >= 0)
                B = BtildeVal(l, m)
            else
                B = 0.0
            end
        else
            B = EtildeVal(l, abs(m))
        end
        B *= alphaVal(l, m) / (2 * alphaVal(l-1, m+1))
        return B
    end

    global function coeff_D(l, m)
        D = 1.0
        if (m > 0)
            D = DtildeVal(l, m)
        else
            D = AtildeVal(l, abs(m))
        end
        D *= alphaVal(l, m) / (2 * alphaVal(l+1, m-1))
        return D
    end

    global function coeff_E(l, m)
        E = 1.0
        if (m > 0)
            E = EtildeVal(l, m)
        else
            if (l - abs(m) - 2 >= 0)
                E = BtildeVal(l, abs(m))
            else
                E = 0.0
            end
        end
        E *= alphaVal(l, m) / (2 * alphaVal(l-1, m-1))
        return E
    end

    global function coeff_F(l, m)
        return FtildeVal(l, abs(m)) * alphaVal(l, m) / alphaVal(l+1, m)
    end

    global function coeff_G(l, m)
        return GtildeVal(l, abs(m)) * alphaVal(l, m) / alphaVal(l-1, m)
    end


    #===#


    #=
    Functions to obtain the matrices used in the 3-term relation for evaluating
    the OPs on the sphere
    =#

    global function systemMatrix_A(n)
        A = 0
        if n == 0
            d_00 = coeff_D(0, 0)
            a_00 = coeff_A(0, 0)
            f_00 = coeff_F(0, 0)
            A = [d_00 0 a_00; im*d_00 0 -im*a_00; 0 f_00 0]
        else
            # We proceed by creating diagonal matrices using the coefficients of
            # the 3-term relations for the spherical harmonic OPs, and then combining
            zerosVec = zeros(2*n + 1)
            leftdiag = copy(zerosVec)
            rightdiag = copy(zerosVec)
            lowerdiag = copy(zerosVec)
            for k = -n:n
                leftdiag[k+n+1] = coeff_D(n, k)
                rightdiag[k+n+1] = coeff_A(n, k)
                lowerdiag[k+n+1] = coeff_F(n, k)
            end
            left = [Diagonal(leftdiag) zeros(2*n+1, 2)]
            right = [zeros(2*n+1, 2) Diagonal(rightdiag)]
            lower = [zeros(2*n+1, 1) Diagonal(lowerdiag) zeros(2*n+1, 1)]
            A = [left + right; -im*(-left + right); lower]
        end
        return A
    end

    global function systemMatrix_B(n)
        return zeros(3*(2*n + 1), 2*n + 1)
    end

    global function systemMatrix_C(n)
        if n == 0
            return zeros(3, 1)
        elseif n == 1
            b_11 = coeff_B(1, -1)
            e_11 = coeff_E(1, 1)
            g_10 = coeff_G(1, 0)
            return [b_11; 0; e_11; -im*b_11; 0; im*e_11; 0; g_10; 0]
        end
        # We proceed by creating diagonal matrices using the coefficients of
        # the 3-term relations for the spherical harmonic OPs, and then combining
        zerosVec = zeros(2*n - 1)
        upperdiag = copy(zerosVec)
        lowerdiag = copy(zerosVec)
        diag_z = copy(zerosVec)
        for k = -n:n-2
            upperdiag[k+n+1] = coeff_B(n, k)
            lowerdiag[k+n+1] = coeff_E(n, k+2)
            diag_z[k+n+1] = coeff_G(n, k+1)
        end
        upper = Diagonal(upperdiag)
        lower = Diagonal(lowerdiag)
        C_x = [upper; zerosVec'; zerosVec'] + [zerosVec'; zerosVec'; lower]
        C_y = [upper; zerosVec'; zerosVec'] - [zerosVec'; zerosVec'; lower]
        C_y *= -im
        C_z = [zerosVec'; Diagonal(diag_z); zerosVec']
        return [C_x; C_y; C_z]
    end

    global function systemMatrix_G(n, x, y, z)
        Iden = eye(2*n + 1)
        return [x*Iden; y*Iden; z*Iden]
    end

    global function systemMatrix_DT(n)
        # Note DT_n is a right inverse matrix of A_n
        DT = 0
        if n == 0
            d_00 = coeff_D(0, 0)
            a_00 = coeff_A(0, 0)
            f_00 = coeff_F(0, 0)
            DT = [1./(2*d_00) -im/(2*d_00) 0; 0 0 1./f_00; 1./(2*a_00) im/(2*a_00) 0]
        else
            # We proceed by creating diagonal matrices using the coefficients of
            # the 3-term relations for the spherical harmonic OPs, and then combining
            upperdiag = zeros(2*n + 1)
            for k = -n:n
                upperdiag[k+n+1] = 1./(2 * coeff_D(n, k))
            end
            upper = [Diagonal(upperdiag) Diagonal(-im*upperdiag) zeros(2*n+1, 2*n+1)]
            lower = im*zeros(2, 3*(2*n+1))
            lower[1, 2*n] = 1./(2 * coeff_A(n, n-1))
            lower[2, 2*n+1] = 1./(2 * coeff_A(n, n))
            lower[1, 2*(2*n+1)-1] = im*lower[1, 2*n]
            lower[2, 2*(2*n+1)] = im*lower[2, 2*n+1]
            DT = [upper; lower]
        end
        return DT
    end


    #====#


    #=
    Functions to obtain the matrices corresponding to multiplication of the OPs
    by x, y and z respectively (called J^x, J^y and J^z)
    =#

    global function systemMatrix_Ax(n)
        zerosVec = zeros(2*n + 1)
        leftdiag = copy(zerosVec)
        rightdiag = copy(zerosVec)
        for k = -n:n
            leftdiag[k+n+1] = coeff_D(n, k)
            rightdiag[k+n+1] = coeff_A(n, k)
        end
        left = [Diagonal(leftdiag) zeros(2*n+1, 2)]
        right = [zeros(2*n+1, 2) Diagonal(rightdiag)]
        return left + right
    end

    global function systemMatrix_Ay(n)
        zerosVec = zeros(2*n + 1)
        leftdiag = copy(zerosVec)
        rightdiag = copy(zerosVec)
        for k = -n:n
            leftdiag[k+n+1] = coeff_D(n, k)
            rightdiag[k+n+1] = coeff_A(n, k)
        end
        left = [Diagonal(leftdiag) zeros(2*n+1, 2)]
        right = [zeros(2*n+1, 2) Diagonal(rightdiag)]
        return -im*(left + right)
    end

    global function systemMatrix_Az(n)
        zerosVec = zeros(2*n + 1)
        d = copy(zerosVec)
        for k = -n:n
            d[k+n+1] = coeff_F(n, k)
        end
        return [zeros(2*n+1, 1) Diagonal(d) zeros(2*n+1, 1)]
    end

    global function systemMatrix_Bx(n)
        return zeros(2*n+1, 2*n+1)
    end

    global function systemMatrix_By(n)
        return zeros(2*n+1, 2*n+1)
    end

    global function systemMatrix_Bz(n)
        return zeros(2*n+1, 2*n+1)
    end

    global function systemMatrix_Cx(n)
        zerosVec = zeros(2*n - 1)
        upperdiag = copy(zerosVec)
        lowerdiag = copy(zerosVec)
        for k = -n:n-2
            upperdiag[k+n+1] = coeff_B(n, k)
            lowerdiag[k+n+1] = coeff_E(n, k+2)
        end
        upper = Diagonal(upperdiag)
        lower = Diagonal(lowerdiag)
        return [upper; zerosVec'; zerosVec'] + [zerosVec'; zerosVec'; lower]
    end

    global function systemMatrix_Cy(n)
        zerosVec = zeros(2*n - 1)
        upperdiag = copy(zerosVec)
        lowerdiag = copy(zerosVec)
        for k = -n:n-2
            upperdiag[k+n+1] = coeff_B(n, k)
            lowerdiag[k+n+1] = coeff_E(n, k+2)
        end
        upper = Diagonal(upperdiag)
        lower = Diagonal(lowerdiag)
        return - im * ([upper; zerosVec'; zerosVec'] - [zerosVec'; zerosVec'; lower])
    end

    global function systemMatrix_Cz(n)
        zerosVec = zeros(2*n - 1)
        d = copy(zerosVec)
        for k = -n:n-2
            d[k+n+1] = coeff_G(n, k+1)
        end
        return [zerosVec'; Diagonal(d); zerosVec']
    end

    global function Jx(N)
        M = (N+1)^2
        J = sparse(zeros(M,M))
        J[1,2:4] = systemMatrix_Ax(0)
        for n = 1:N-1
            rows = n^2+1:(n+1)^2
            J[rows, (n-1)^2+1:n^2] = systemMatrix_Cx(n)
            J[rows, (n+1)^2+1:(n+2)^2] = systemMatrix_Ax(n)
        end
        J[N^2+1:end,(N-1)^2+1:N^2] = systemMatrix_Cx(N)
        return J
    end

    global function Jy(N)
        M = (N+1)^2
        J = sparse(zeros(M,M))
        J[1,2:4] = systemMatrix_Ay(0)
        for n = 1:N-1
            rows = n^2+1:(n+1)^2
            J[rows, (n-1)^2+1:n^2] = systemMatrix_Cy(n)
            J[rows, (n+1)^2+1:(n+2)^2] = systemMatrix_Ay(n)
        end
        J[N^2+1:end,(N-1)^2+1:N^2] = systemMatrix_Cy(N)
        return J
    end

    global function Jz(N)
        M = (N+1)^2
        J = sparse(zeros(M,M))
        J[1,2:4] = systemMatrix_Az(0)
        for n = 1:N-1
            rows = n^2+1:(n+1)^2
            J[rows, (n-1)^2+1:n^2] = systemMatrix_Cz(n)
            J[rows, (n+1)^2+1:(n+2)^2] = systemMatrix_Az(n)
        end
        J[N^2+1:end,(N-1)^2+1:N^2] = systemMatrix_Cz(N)
        return J
    end


    #=
    Function to obtain the point evaluation of the Nth set of OPs (order N) at
    the point on the unit sphere (x, y, z)
    =#
    global function opEval(N, x, y, z)

        # Check that x and y are on the unit circle
        delta = 0.001
        @assert (x^2 + y^2 + z^2 < 1 + delta &&  x^2 + y^2 + z^2 > 1 - delta) "the point (x, y) must be on unit circle"

        # Check that N is a non-negative integer
        @assert N >= 0 "the argument N should be a non-negative integer"

        # We initialise P_(-1) = 0, P_0 = 1, and an empty vector for P_1
        P_nminus1 = 0
        P_n = alphaVal(0, 0)
        P_nplus1 = 0

        for n = 0:N-1
            # Define the matrices in the 3-term relation
            B_n = systemMatrix_B(n)
            C_n = systemMatrix_C(n)
            G_n = systemMatrix_G(n, x, y, z)
            DT_n = systemMatrix_DT(n)

            # Calculate the next set of OPs
            P_nplus1 = - DT_n * (B_n - G_n) * P_n - DT_n * C_n * P_nminus1

            # Re-label for the next step
            P_nminus1 = copy(P_n)
            P_n = copy(P_nplus1)
        end

        return P_n

    end


    #====#


    #=
    Function to obtain a point evaluation of a function f(x,y,z) where f is input as
    the coefficients of its expansion in the basis of the OPs for the sphere, i.e.
        f(x, y) = sum(vecdot(f_n, P_n))
    where the {P_n} are the OPs on the sphere (spherical harmonics)

    Uses the Clenshaw Algorithm.
    =#
    global function funcEval(f, x, y, z)

        # Check that x and y are on the unit circle
        delta = 0.001
        @assert (x^2 + y^2 + z^2 < 1 + delta &&  x^2 + y^2 + z^2 > 1 - delta) "the point (x, y, z) must be on unit sphere"

        M = length(f)
        N = round(Int, sqrt(M) - 1)
        @assert (M > 0 && sqrt(M) - 1 == N) "invalid length of f"

        # Complete the reverse recurrance to gain gamma_1, gamma_2
        # Note that gamma_(N+1) = 0, gamma_(N+2) = 0
        gamma_nplus2 = zeros((N+3)^2-(N+2)^2)
        gamma_nplus1 = zeros((N+2)^2-(N+1)^2)
        gamma_n = 0.0
        for n = N:-1:1
            a = - (systemMatrix_DT(n) * (systemMatrix_B(n) - systemMatrix_G(n, x, y, z))).'
            b = - (systemMatrix_DT(n+1) * systemMatrix_C(n+1)).'
            gamma_n = view(f, n^2+1:(n+1)^2) + a * gamma_nplus1 + b * gamma_nplus2
            gamma_nplus2 = copy(gamma_nplus1)
            gamma_nplus1 = copy(gamma_n)
        end

        # Calculate the evaluation of f using gamma_1, gamma_2
        # f(x,y,z) = P_0*f_0 + gamma_1^T * P_1 - (DT_1*C_1)^T * gamma_2
        b = - (systemMatrix_DT(1) * systemMatrix_C(1)).'
        P_1 = opEval(1, x, y, z)
        P_0 = opEval(0, x, y, z)
        #return P_0 * f[1] + vecdot(gamma_nplus1, P_1) + P_0 * b * gamma_nplus2
        return P_0 * f[1] + (P_1.' * gamma_nplus1)[1] + P_0 * b * gamma_nplus2

    end

end

#-----
# Testing

x = 0.1
y = 0.8
z = sqrt(1 - x^2 - y^2)
N = 3
p = opEval(N, x, y, z)

p_actual = alphaVal(3, -1) * (x - im*y) * ((z-1)^2 * 15/4 + (z-1) * 15/2 + 3)
# p_actual = alphaVal(2,2) * (x + im*y)^2
@test p[N] ≈ p_actual

N = 10
f = 1:(N+1)^2
fxyz = funcEval(f, x, y, z)
fxyz_actual = 0.0
for k = 0:N
    fxyz_actual += vecdot(view(f, k^2+1:(k+1)^2), opEval(k, x, y, z))
end
@test fxyz ≈ fxyz_actual
