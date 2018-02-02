# Script to obtain a point evaluation of the Nth set of OPs for the unit circle

# using ApproxFun
using Base.Test
using BlockArrays
using BlockBandedMatrices


#=
Function to obtain a point evaluation of the Nth set of OPs for the unit circle
at the point (x, y) = (cos(theta), sin(theta)).

The nth set of OPs are : [sin(n*theta), cos(n*theta)], n non-negative integer
=#
let

    # Define the matrices used in our 3-term relations
    const DT_0 = [0 1; 1 0]
    const DT_n = [2 0 0 0; 0 2 0 0]
    const B_0 = [0; 0]
    const B_n = [0 0; 0 0; 0 0; 0 0]
    const C_1 = [0; 0.5; 0.5; 0]
    const C_n = [0.5 0; 0 0.5; 0 0.5; -0.5 0]

    global function OPeval(N, x, y)

        # Check that x and y are on the unit circle
        delta = 0.001
        @assert (x^2 + y^2 < 1 + delta &&  x^2 + y^2 > 1 - delta) "the point (x, y) must be on unit circle"

        # Check that N is a non-negative integer
        @assert N >= 0 "the argument N should be a non-negative integer"

        # Define the matrices used in our 3-term relations
        G_0 = [x; y]
        G_n = [x 0; 0 x; y 0; 0 y]

        # # We initialise P_(-1) = 0, P_0 = 1, and an empty vector for P_1
        # P_nminus1 = 0
        # P_n = 1
        # P_nplus1 = [0; 0]
        #
        # for n = 0:N-1
        #     if n == 0
        #         P_nplus1 = - DT_0 * (B_0 - G_0) * P_n
        #     elseif n == 1
        #         P_nplus1 = - DT_n * (B_n - G_n) * P_n - DT_n * C_1 * P_nminus1
        #     else
        #         P_nplus1 = - DT_n * (B_n - G_n) * P_n - DT_n * C_n * P_nminus1
        #     end
        #     P_nminus1 = P_n
        #     P_n = P_nplus1
        # end

        # Declare empty array to store the OP values
        P = zeros(2*N+1)
        # We initialise P_0 = 1, and treat P_(-1) = 0
        P[1] = 1
        # Begin loop, storing OP values
        for n = 0:N-1
            if n == 0
                P[2*(n+1) : 2*(n+1)+1] = - DT_0 * (B_0 - G_0) * P[1]
            elseif n == 1
                P[2*(n+1) : 2*(n+1)+1] = - DT_n * (B_n - G_n) * view(P, 2*n:2*n+1) - DT_n * C_1 * P[1]
            else
                P[2*(n+1) : 2*(n+1)+1] = - DT_n * (B_n - G_n) * view(P, 2*n:2*n+1) - DT_n * C_n * view(P, 2*(n-1):2*(n-1)+1)
            end
        end

        return view(P, 2*N:2*N+1)

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

    # Define the matrices used in our 3-term relation
    const DT_0 = [0 1; 1 0]
    const DT_n = [2 0 0 0; 0 2 0 0]
    const B_0 = [0; 0]
    const B_n = [0 0; 0 0; 0 0; 0 0]
    const C_1 = [0; 0.5; 0.5; 0]
    const C_n = [0.5 0; 0 0.5; 0 0.5; -0.5 0]

    const Ax_0 = [0 1]
    const Ax_n = [0.5 0; 0 0.5]
    const Cx_1 = [0; 0.5]
    const Cx_n = [0.5 0; 0 0.5]

    const Ay_0 = [1 0]
    const Ay_n = [0 -0.5; 0.5 0]
    const Cy_1 = [0.5; 0]
    const Cy_n = [0 0.5; -0.5 0]

    global function Jx(N)
        l,u = 1,1          # block bandwidths
        λ,μ = 1,1          # sub-block bandwidths: the bandwidths of each block
        cols = rows = [1; round.(Int,2*ones(N))]  # block sizes
        J = BandedBlockBandedMatrix(0.0I, (rows,cols), (l,u), (λ,μ))
        if N == 0
            return J
        end
        J[1,2:3] = Ax_0
        J[2:3,1] = Cx_1
        if N == 1
            return J
        end
        rows = 4:5
        cols = 4:5
        for n = 2:N
            J[rows-2,cols] = Ax_n
            J[rows,cols-2] = Cx_n
            rows += 2
            cols += 2
        end
        return J
    end

    global function Jy(N)
        l,u = 1,1          # block bandwidths
        λ,μ = 1,1          # sub-block bandwidths: the bandwidths of each block
        cols = rows = [1; round.(Int,2*ones(N))]  # block sizes
        J = BandedBlockBandedMatrix(0.0I, (rows,cols), (l,u), (λ,μ))
        if N == 0
            return J
        end
        J[1,2:3] = Ay_0
        J[2:3,1] = Cy_1
        if N == 1
            return J
        end
        rows = 4:5
        cols = 4:5
        for n = 2:N
            J[rows-2,cols] = Ay_n
            J[rows,cols-2] = Cy_n
            rows += 2
            cols += 2
        end
        return J
    end

    global function funcEval(f, x, y)

        # Check that x and y are on the unit circle
        delta = 0.001
        @assert (x^2 + y^2 < 1 + delta &&  x^2 + y^2 > 1 - delta) "the point (x, y) must be on unit circle"

        M = length(f)
        N = (M - 1) / 2
        @assert (M % 2 == 1) "invalid length of f - should be odd number"

        # Define the matrices used in our 3-term relation
        G_0 = [x, y]   # [Jx ; Jy]
        G_n = [x 0; 0 x; y 0; 0 y]   # [Jx 0I ; 0I Jx ; Jy 0; 0 Jy]
        alpha = - DT_n * (B_n - G_n)
        beta = - DT_n * C_n

        # Define a zeros vector to store the gammas.
        # Note that we add in gamma_(N+1) = gamma_(N+2) = [0;0]
        gamma = Vector{Float64}(M+(2*2))    # OPERATOR VERSION: Vector{Matrix{Float64}}(M+(2*2)) or PseudoBlockArray(uninitialized,
        gamma[end-4:4] = 0                  # OPERATOR VERSION: gamma[end-4:4] = zeros(Jx)
        # Complete the reverse recurrance to gain gamma_1, gamma_2
        for n = M:-2:2
            gamma[n-1 : n] = view(f, n-1:n) + alpha * view(gamma, n+1:n+2) + beta * view(gamma, n+3:n+4)  # Operator version view(f, n-1:n).*I
        end

        # Calculate the evaluation of f using gamma_1, gamma_2
        beta = - DT_n * C_1
        P_1 = [y; x]  #[Jy ; Jx]
        return f[1] + vecdot(P_1, view(gamma, 2:3)) + vecdot(beta, view(gamma, 4:5))

    end


    global function funcOperatorEval(f)

        M = length(f)
        N = round(Int, (M - 1) / 2)
        @assert (M % 2 == 1) "invalid length of f - should be odd number"

        # Define the Jacobi operator matrices
        J_x = Jx(N)
        J_y = Jy(N)

        # Define the matrices used in our 3-term relation
        G_n = Matrix{Matrix{Float64}}(4,2)
        for kj in eachindex(G_n)
            G_n[kj] = zeros(J_x)
        end
        G_n[1,1] = G_n[2,2] = J_x
        G_n[3,1] = G_n[4,2] = J_y
        a = - DT_n * (B_n - G_n)
        b = - DT_n * C_n

        # Define a zeros vector to store the gammas.
        # Note that we add in gamma_(N+1) = gamma_(N+2) = [0;0]
        gamma_nplus2 = Vector{Matrix{Float64}}(2)
        gamma_nplus1 = Vector{Matrix{Float64}}(2)
        gamma_n = Vector{Matrix{Float64}}(2)
        for k in eachindex(gamma_nplus2)
            gamma_nplus2[k] = zeros(J_x)
            gamma_nplus1[k] = zeros(J_x)
            gamma_n[k] = zeros(J_x)
        end

        # Complete the reverse recurrance to gain gamma_1, gamma_2
        for n = N:-1:1
            gamma_n = view(f, 2n:2n+1).*I + a * gamma_nplus1 + b * gamma_nplus2
            gamma_nplus2 = copy(gamma_nplus1)
            gamma_nplus1 = copy(gamma_n)
        end

        # Calculate the evaluation of f using gamma_1, gamma_2
        b = - DT_n * C_1
        P_1 = [J_y, J_x]
        return f[1].*I + P_1.' * gamma_nplus1 + b.' * gamma_nplus2

    end

end

N = 5
f = 1:(2*N+1)
fj = funcOperatorEval(f)



#
# using BlockArrays#, GPUArrays
#
# x = rand(3,3)
# y = rand(3,3)
#
# G_0 = [x, y]
# G_n = Matrix{Matrix{Float64}}(4,2)
# for kj in eachindex(G_n)
#         G_n[kj] = zeros(x)
# end
# G_n[1,1] = G_n[2,2] = x
# G_n[3,1] = G_n[4,2] = y
# G_n
#
# B_n = Matrix{typeof(I)}(4,2)
#     B_n .= 0I
#
# DT_n = [2 0 0 0;
#         0 2 0 0]
#
# alpha = - DT_n * (B_n - G_n)
#
#
#
# v = Vector{Matrix{Float64}}(2)
#     for k in eachindex(v) v[k] = rand(3,3) end
# v
#
#
# @which G_n * v
# A = G_n; x = v; TS = Matrix{Float64}
# @which A_mul_B!(similar(x,TS,size(A,1)),A,x)
#
# y = similar(x,TS,size(A,1))
# @which Base.LinAlg.generic_matvecmul!(y, 'N', A, x)
#
# G_n * v
#
#
# gamma
#
#
#
# [1,2,3,4,5,6,7]
#
# n = 3 # size(Jx,1)
#
# PseudoBlockArray{Float64}(uninitialized, [n;fill(2n,3)], [n])
#
# @which PseudoBlockArray([1,2,3,4,5,6,7], [1;fill(2,3)])
#
# M = 10
#
# [x ; y] # concats
# x = [1,2]
# y = [3,4]
# [x ; y]
#
#
#
#
#
# [x,y]
#
#
#
# [[1,2], [3,4]]
#
# A = rand(2,2)
#
# A*[[1 2; 3 4], [3 4; 5 6]]
#
#
# I + [3 3; 4 4]
#
#
#
# [3,4] .*I
#
#
# A*[[1 2; 3 4], [3 4; 5 6]]    + [3I,4I]
#
#
# #-----
# # Testing
#
# x = 0.1
# y = sqrt(1 - x^2)
# N = 1000
# p = OPeval(N, x, y)
#
# theta = acos(x)
# p_actual = [sin(N*theta); cos(N*theta)]
# @test p ≈ p_actual
#
# N = 5
# f = 1:(2*N+1)
# fxy = funcEval(f, x, y)
# fxy_actual = zeros(length(f))
# fxy_actual = f[1]
# for i = 1:N
#     fxy_actual += vecdot(view(f, 2*i:2*i+1), OPeval(i, x, y))
# end
# @test fxy ≈ fxy_actual
