# Script to obtain a point evaluation of the Nth set of OPs for the unit circle

using ApproxFun
using Base.Test
using Base.LinAlg


let

    #= Functions to obtain the coefficients used in the matrices for
    J^x, J^y, J^z
    =#

    global function alphaVal(l, m)
        c = sqrt((2*l + 1) * gamma(l - m + 1) / (4*pi*gamma(l + m + 1)))
        ctilde = 1.0
        if m < 0
            m = abs(m)
            ctilde = (-1.0)^m * gamma(l - m + 1) / gamma(l + m + 1)
        end
        chat = gamma(l + m + 1) / (gamma(l + 2) * (-2.0)^m)
        return c * chat * ctilde
    end

    global function bhatVal(l, m)
        return gamma(l+1) * gamma(2*m+1) / (gamma(l+m+1) * gamma(m+1))
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
        D = (l - m + 1) / (2*m + 0.5)
        D *= (l - m + 2) / (l + 1.5)
        D *= (2*m - 1) / (l + 1)
        return D
    end

    global function EtildeVal(l, m)
        E = (l + m + 1) / (2*m + 0.5)
        E *= (l + m + 2) / (l + 1.5)
        E *= (2*m - 1) / (l + m)
        E *= l / (l + m - 1)
        return E
    end

    global function FtildeVal(l, m)
        return (l - m + 1) * (l + m + 1) / ((2*l + 1) * (l + 1))
    end

    global function GtildeVal(l, m)
        G = 1.0
        if (abs(m) < l - 1)
            G = l / (2*l + 1)
        else
            G = 0.0
        end
        return G
    end

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
        if (l - abs(m) - 2 > 0)
            if (m >= 0)
                B = BtildeVal(l, m)
            else
                B = EtildeVal(l, abs(m))
            end
            B *= alphaVal(l, m) / (2 * alphaVal(l-1, m+1))
        else
            B = 0.0
        end
        return B
    end

    global function coeff_D(l, m)
        D = 1.0
        if (m >= 0)
            D = DtildeVal(l, m)
        else
            D = AtildeVal(l, abs(m))
        end
        D *= alphaVal(l, m) / (2 * alphaVal(l+1, m-1))
        return D
    end

    global function coeff_E(l, m)
        E = 1.0
        if (m >= 0)
            E = EtildeVal(l, m)
        else
            E = BtildeVal(l, abs(m))
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
            return [b_11; 0; e_11; b_11; 0; -e_11; 0; g_10; 0]
        end
        # We proceed by creating diagonal matrices using the coefficients of
        # the 3-term relations for the spherical harmonic OPs, and then combining
        zerosVec = zeros(2*n - 1)
        upperdiag = zerosVec
        lowerdiag = zerosVec
        diag_z = zerosVec
        for k = -n:n-2
            upperdiag[k+n+1] = coeff_B(n, k)
            lowerdiag[k+n+1] = coeff_E(n, k+2)
            diag_z[k+n+1] = coeff_G(n, k+1)
        end
        upper = Diagonal(upperdiag)
        lower = Diagonal(lowerdiag)
        C_x = [upper; zerosVec'; zerosVec'] + [zerosVec'; zerosVec'; lower]
        C_y = [upper; zerosVec'; zerosVec'] - [zerosVec'; zerosVec'; lower]
        C_z = [zerosVec'; Diagonal(diag_z); zerosVec']
        return [C_x; C_y; C_z]
    end

    global function systemMatrix_G(n, x, y, z)
        Iden = eye(2*n + 1)
        return [x*Iden; y*Iden; z*Iden]
    end

    global function systemMatrix_DT(n)
        DT = 0
        if n == 0
            d_00 = coeff_D(0, 0)
            a_00 = coeff_A(0, 0)
            f_00 = coeff_F(0, 0)
            DT = [1./(2*d_00) -1./(2*d_00) 0; 0 0 1./f_00; 1./(2*a_00) 1./(2*a_00) 0]
        else
            # We proceed by creating diagonal matrices using the coefficients of
            # the 3-term relations for the spherical harmonic OPs, and then combining
            upperdiag = zeros(2*n + 1)
            for k = -n:n
                upperdiag[k+n+1] = 1./(2 * coeff_D(n, k))
            end
            upper = [Diagonal(upperdiag) Diagonal(-upperdiag) zeros(2*n+1, 2*n+1)]
            lower = zeros(2, 3*(2*n+1))
            lower[1, 2*n] = 1./(2 * coeff_A(n, n-1))
            lower[2, 2*n+1] = 1./(2 * coeff_A(n, n))
            lower[1, 2*(2*n+1)-1] = lower[1, 2*n]
            lower[2, 2*(2*n+1)] = lower[2, 2*n+1]
            DT = [upper; lower]
        end
        return DT
    end


    #====#


    #=
    Function to obtain the point evaluation of the Nth OP at the point on the
    unit sphere (x, y, z)
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
        P_nplus1 = [0; 0]

        for n = 0:N-1
            # Define the matrices in the 3-term relation
            B_n = systemMatrix_B(n)
            C_n = systemMatrix_C(n)
            G_n = systemMatrix_G(n, x, y, z)
            DT_n = systemMatrix_DT(n)

            # Calculate the next set of OPs
            P_nplus1 = - DT_n * (B_n - G_n) * P_n - DT_n * C_n * P_nminus1

            # Re-label for the next step
            P_nminus1 = P_n
            P_n = P_nplus1
        end

        return P_n

    end

end



#=
Function to obtain a point evaluation of a function f(x,y) where f is input as
the coefficients of its expansion in the basis of the OPs for the circle, i.e.
    f(x, y) = sum(vecdot(f_n, P_n))
where the {P_n} are the OPs on the circle, P_0 = 1 and P_n in R^2 for n>0.

We note the structure of the input f is then a vector of length M=2N+1 where N is
the number of OP sets that are used in the expansion, i.e. n = 0,...,N, and where
f_0=f[1] and f_n=f[2*n : 2*n+1] in R^2 for n>0.

Uses the Clenshaw Algorithm.
=#
let

    global function funcEval(f, x, y)

        # Check that x and y are on the unit circle
        delta = 0.001
        @assert (x^2 + y^2 < 1 + delta &&  x^2 + y^2 > 1 - delta) "the point (x, y) must be on unit circle"

        M = length(f)
        N = (M - 1) / 2
        @assert (M % 2 == 1) "invalid length of f - should be odd number"

        # Define the matrices used in our 3-term relation
        G_0 = [x; y]
        G_n = [x 0; 0 x; y 0; 0 y]
        alpha = - DT_n * (B_n - G_n)
        beta = - DT_n * C_n

        # Define a zeros vector to store the gammas.
        # Note that we add in gamma_(N+1) = gamma_(N+2) = [0;0]
        gamma = zeros(M+(2*2))
        # Complete the reverse recurrance to gain gamma_1, gamma_2
        for n = M:-2:2
            gamma[n-1 : n] = view(f, n-1:n) + alpha * view(gamma, n+1:n+2) + beta * view(gamma, n+3:n+4)
        end

        # Calculate the evaluation of f using gamma_1, gamma_2
        beta = - DT_n * C_1
        P_1 = [y; x]
        return f[1] + vecdot(P_1, view(gamma, 2:3)) + vecdot(beta, view(gamma, 4:5))

    end

end

#-----
# Testing

x = 0.1
y = 0.8
z = sqrt(1 - x^2 - y^2)
N = 1
p = opEval(N, x, y, z)
println(p)

theta = acos(z)
phi = acos(x / sin(theta))
# p_actual = alphaVal(2, 0) * 0.5 * (3*z^2 - 1)
p_actual = alphaVal(1, 0) * z
@test p[N+1] ≈ p_actual

# N = 5
# f = 1:(2*N+1)
# fxy = funcEval(f, x, y)
# fxy_actual = zeros(length(f))
# fxy_actual = f[1]
# for i = 1:N
#     fxy_actual += vecdot(view(f, 2*i:2*i+1), OPeval(i, x, y))
# end
# @test fxy ≈ fxy_actual
