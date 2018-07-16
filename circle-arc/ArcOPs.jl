#=
Setup
=#
function innerTh(T, U, h)
    xmin = h
    xmax = 1
    x = Fun(identity, xmin..xmax)
    w = (1/sqrt(1-x^2))
    return sum(w*T*U)
end

function innerUh(T, U, h)
    xmin = h
    xmax = 1
    x = Fun(identity, xmin..xmax)
    w = sqrt(1-x^2)
    return sum(w*T*U)
end

function initialise_arc_ops(N, h)
    # @assert (N > 1)

    # Get recurrance relation coefficients for the 1D OPs T_k^h(x), U_k^h(x)
    xmin = h
    xmax = 1
    x = Fun(identity, xmin..xmax)
    w = (1/sqrt(1-x^2))
    Tkh,α,β = lanczos(w,N+1)
    w = sqrt(1-x^2)
    Ukh,γ,δ = lanczos(w,N+1)

    # Obtain the relationship/conversion coefficients between T_k^h and U_k^h:
    #   T_k^h = a_k*U_k^h + b_k*U_(k-1)^h + c_k*U_(k-2)^h
    #   y^2*U_(k-1)^h = (1-x^2)*U_(k-1)^h = d_k*U_(k-1)^h + e_k*U_k^h + f_k*U_(k+1)^h
    a = zeros(N+1)
    b = copy(a)
    c = copy(a)
    d = copy(a)
    e = copy(a)
    f = copy(a)
    a[1] = innerUh(Tkh[1], Ukh[1], h)
    a[2] = innerUh(Tkh[2], Ukh[2], h)
    b[2] = innerUh(Tkh[2], Ukh[1], h)
    for k = 3:N+1
        a[k] = innerUh(Tkh[k], Ukh[k], h)
        b[k] = innerUh(Tkh[k], Ukh[k-1], h)
        c[k] = innerUh(Tkh[k], Ukh[k-2], h)
    end
    for k=1:N-1
        d[k] = innerUh(Ukh[k], Tkh[k], h)
        e[k] = innerUh(Ukh[k], Tkh[k+1], h)
        f[k] = innerUh(Ukh[k], Tkh[k+2], h)
    end
    d[N] = innerUh(Ukh[N], Tkh[N], h)
    e[N] = innerUh(Ukh[N], Tkh[N+1], h)
    d[N+1] = innerUh(Ukh[N+1], Tkh[N+1], h)

    # We return all the coefficients
    return α, β, γ, δ, a, b, c, d, e, f
end

#=
Functions to construct the operators corresponding to multiplication by x and y
of the vector P given by P = [P_0,P_1,...] where P_k = [T_k^h(x),y*U_k^h(x)]
(the arc OPs).
=#
function Jx(N, α, β, γ, δ)
    rows = cols = [1]
    append!(rows, [2 for i=2:N+1])
    J = BandedBlockBandedMatrix(zeros(sum(rows), sum(cols)), (rows,cols), (1,1), (0,0))
    view(J, Block(1,1)) .= [α[1]]
    view(J, Block(1,2)) .= [β[1] 0.0]
    view(J, Block(2,1)) .= [β[1]; 0.0]
    for i = 2:N
        view(J, Block(i,i)) .= [α[i] 0; 0 γ[i-1]]
        subblock = [β[i] 0; 0 δ[i-1]]
        view(J, Block(i,i+1)) .= subblock
        view(J, Block(i+1,i)) .= subblock
    end
    view(J, Block(N+1,N+1)) .= [α[N+1] 0; 0 γ[N]]
end

function Jy(N, a, b, c, d, e, f)
    rows = cols = [1]
    append!(rows, [2 for i=2:N+1])
    J = BandedBlockBandedMatrix(zeros(sum(rows), sum(cols)), (rows,cols), (1,1), (1,1))
    view(J, Block(1,2)) .= [0 a[1]]
    view(J, Block(2,1)) .= [0; d[1]]
    for i = 2:N
        view(J, Block(i,i)) .= [0 b[i]; e[i-1] 0]
        view(J, Block(i,i+1)) .= [0 a[i]; f[i-1] 0]
        view(J, Block(i+1,i)) .= [0 c[i+1]; d[i] 0]
    end
    view(J, Block(N+1,N+1)) .= [0 b[N+1]; e[N] 0]
    return J
end

#=
Functions to output the matrices used in the Clenshaw Algorithm for
evaluation of a function given by its coefficients of its expansion in the
arc OPs.
=#
function clenshaw_matrix_B(n, α, γ, b, e)
    if n == 0
        return [α[1]; 0]
    else
        return [α[n+1] 0; 0 γ[n]; 0 b[n+1]; e[n] 0]
    end
end

function clenshaw_matrix_G(n, x, y)
    if n == 0
        return [x; y]
    else
        iden = speye(2)
        return [x*iden; y*iden]
    end
end

function clenshaw_matrix_Gtilde(n, x, y)
    if n == 0
        return [-y; x]
    else
        iden = speye(2)
        return [-y*iden; x*iden]
    end
end

function clenshaw_matrix_C(n, β, δ, c, d)
    if n == 1
        out = zeros(4,1)
        out[1] = β[1]
        out[4] = d[1]
        return out
    else
        return [β[n] 0; 0 δ[n-1]; 0 c[n]; d[n-1] 0]
    end
end

function clenshaw_matrix_DT(n, β, δ, a)
    if n == 0
        return [1./β[1] 0; 0 1./a[1]]
    else
        return [1./β[n+1] 0 0 0; 0 1./δ[n] 0 0]
    end
end

#=
Function to obtain for storage the coefficient matrices used in the Clenshaw
Algorithm when using N iterations. Once we have these, we can call the
arc_func_eval() method passing in the arrays of matrices to access each
time, rather than calculating them each time we evaluate.
=#
function get_arc_clenshaw_matrices(N, α, β, γ, δ, a, b, c, d, e, f)
    DT = Array{Matrix{Float64}}(N-1)
    K = copy(DT)
    L = copy(DT)
    for n = N-1:-1:1
        DT[n] = clenshaw_matrix_DT(n, β, δ, a)
        K[n] = DT[n] * clenshaw_matrix_B(n, α, γ, b, e)
        L[n] = DT[n] * clenshaw_matrix_C(n, β, δ, c, d)
    end
    return DT, K, L
end

#=
Function to evaluate the order N OP set at the point (x,y) that lies on the
unit circle arc
=#
function arc_op_eval(N, h, x, y)
    # Check that x and y are on the unit circle
    delta = 1e-5
    @assert (h >= 0 && h < 1 && x^2 + y^2 < 1 + delta &&  x^2 + y^2 > 1 - delta && x >= h && x <= 1)
    "the point (x, y) must be on unit circle on the arc defined by the x coord being in the range [h,1]"

    X = Fun(identity, h..1)
    w = (1/sqrt(1-X^2))
    Tkh,α,β = lanczos(w,1)
    if N == 0
        return Tkh[1](x)
    elseif N == 1
        w = sqrt(1-X^2)
        Ukh,a,b = lanczos(w,1)
        return [Tkh[2](x); y*Ukh[1](x)]
    else
        α, β, γ, δ, a, b, c, d, e, f = initialise_arc_ops(N, h)
        DT, K, L = get_arc_clenshaw_matrices(N, α, β, γ, δ, a, b, c, d, e, f)
        P0 = (-clenshaw_matrix_DT(0, β, δ, a)
        *(clenshaw_matrix_B(0, α, γ, b, e)-clenshaw_matrix_G(0,x,y))*Tkh[1](x))
        P1 = - (K[1] - DT[1]*clenshaw_matrix_G(1, x, y))*P0 - L[1]*Tkh[1](x)
        P2 = zeros(2)
        for n = 2:N-1
            P2 = - (K[n] - DT[n]*clenshaw_matrix_G(n, x, y))*P1 - L[n]*P0
            P0 = copy(P1)
            P1 = copy(P2)
        end
        return P1
    end
end

#=
Function to evaluate the derivative of the order n OP at the point (x,y) on
the unit circle arc - ∂/∂s(P_n).
=#
function arc_op_derivative_eval(N, h, x, y)
    # Check that x and y are on the unit circle
    delta = 1e-5
    @assert (h >= 0 && h < 1 && x^2 + y^2 < 1 + delta &&  x^2 + y^2 > 1 - delta && x >= h && x <= 1)
    "the point (x, y) must be on unit circle on the arc defined by the x coord being in the range [h,1]"

    if N == 0
        return 0.0
    end

    X = Fun(identity, h..1)
    w = (1/sqrt(1-X^2))
    Tkh,α,β = lanczos(w,1)
    T0 = Tkh[1](x)

    α, β, γ, δ, a, b, c, d, e, f = initialise_arc_ops(N, h)
    DT, K, L = get_arc_clenshaw_matrices(N, α, β, γ, δ, a, b, c, d, e, f)
    DT0 = clenshaw_matrix_DT(0, β, δ, a)
    B0 = clenshaw_matrix_B(0, α, γ, b, e)

    # n=0
    Pnm1 = 0
    Pn = T0
    dPn = 0
    dPnp1 = DT0*clenshaw_matrix_Gtilde(0,x,y)*T0
    Pnm2 = copy(Pnm1)
    Pnm1 = copy(Pn)
    dPnm1 = copy(dPn)
    dPn = copy(dPnp1)
    # n=1
    if N == 1
        return dPn
    end
    Pn = - DT0*(B0-clenshaw_matrix_G(0,x,y))*Pnm1
    dPnp1 = (-K[1]+DT[1]*clenshaw_matrix_G(1,x,y))*dPn + DT[1]*clenshaw_matrix_Gtilde(1,x,y)*Pn
    Pnm2 = copy(Pnm1)
    Pnm1 = copy(Pn)
    dPnm1 = copy(dPn)
    dPn = copy(dPnp1)
    for n = 2:N-1
        Pn = (-K[n-1]+DT[n-1]*clenshaw_matrix_G(n-1,x,y))*Pnm1 - L[n-1]*Pnm2
        dPnp1 = ((-K[n]+DT[n]*clenshaw_matrix_G(n,x,y))*dPn
        - L[n]*dPnm1
        + DT[n]*clenshaw_matrix_Gtilde(n,x,y)*Pn)
        Pnm2 = copy(Pnm1)
        Pnm1 = copy(Pn)
        dPnm1 = copy(dPn)
        dPn = copy(dPnp1)
    end
    return dPn
end


#=
Function to obtain the evaluation of a function f(x,y)
where f is input as the coefficients of its expansion in the arc space
basis of the circle, i.e.
f(x, y) = sum(vecdot(f_n, P_n))
where the {P_n} are the order n OPs for the arc.

Uses the Clenshaw Algorithm.
=#
function arc_func_eval(F, h, α, β, γ, δ, a, b, c, d, e, f, x, y)
    # Check that x and y are on the unit circle
    delta = 1e-5
    @assert (h >= 0 && h < 1 && x^2 + y^2 < 1 + delta &&  x^2 + y^2 > 1 - delta && x >= h && x <= 1)
    "the point (x, y) must be on unit circle on the arc defined by the x coord being in the range [h,1]"

    M = length(F)
    N = Int((M - 1) / 2)
    @assert (M % 2 == 1) "invalid length of f - should be odd number"

    # Complete the reverse recurrance to gain gamma_1, gamma_2
    # Note that gamma_(N+1) = 0, gamma_(N+2) = 0
    T0 = arc_op_eval(0, h, x, y)
    DT, K, L = get_arc_clenshaw_matrices(N, α, β, γ, δ, a, b, c, d, e, f)
    if N == 0
        return T0*F
    elseif N == 1
        gamma_nplus2 = zeros(2)
        gamma_nplus1 = view(f, 2N:2N+1)
    else
        gamma_nplus2 = view(F, 2N:2N+1)
        gamma_nplus1 = (view(F, 2(N-1):2N-1)
        - (K[N-1] - DT[N-1]*clenshaw_matrix_G(N-1, x, y)) * gamma_nplus2)
        for n = N-2:-1:1
            gamma_n = (view(F, 2n:2n+1)
            - (K[n] - DT[n]*clenshaw_matrix_G(n, x, y)) * gamma_nplus1
            - L[n+1] * gamma_nplus2)
            gamma_nplus2 = copy(gamma_nplus1)
            gamma_nplus1 = copy(gamma_n)
        end
    end

    # Calculate the evaluation of f using gamma_1, gamma_2
    P_1 = arc_op_eval(1, h, x, y)
    return F[1]*T0 + vecdot(P_1, gamma_nplus1) - T0*vecdot(L[1], gamma_nplus2)
end

function Q_func_eval(F, h, x, y)
    # Check that x and y are on the unit circle
    delta = 1e-5
    @assert (h >= 0 && h < 1 && x^2 + y^2 < 1 + delta &&  x^2 + y^2 > 1 - delta && x >= h && x <= 1)
    "the point (x, y) must be on unit circle on the arc defined by the x coord being in the range [h,1]"

    M = length(F)
    N = Int((M - 1) / 2)
    @assert (M % 2 == 1) "invalid length of f - should be odd number"

    xmin = h
    xmax = 1
    X = Fun(identity, xmin..xmax)
    w = X*sqrt(1-X^2)
    Ũkh,_,_ = lanczos(w,N+1)
    w = X/sqrt(1-X^2)
    T̃kh,_,_ = lanczos(w,N+1)
    Q = zeros(2N+1)
    Q[1] = T̃kh[1](x)
    for n = 1:N
        view(Q, 2n:2n+1) .= T̃kh[n+1](x), y*Ũkh[n](x)
    end
    return Q.'*F
end


#=
Gather the coefficients of the expansion of an ApproxFun.Fun in the arc OP
basis.
=#
function func_to_coeffs(F, N, h)
    xmin = h
    xmax = 1
    X = Fun(identity, xmin..xmax)
    w = sqrt(1-X^2)
    Ukh,_,_ = lanczos(w,N+1)
    w = (1/sqrt(1-X^2))
    Tkh,_,_ = lanczos(w,N+1)

    # Y = sqrt(1-X^2)
    # Fe = (F.(X,Y) + F.(X,-Y))/2
    # Fo = (F.(X,Y) - F.(X,-Y))/2
    Fe = Fun( x -> (F(x,sqrt(1-x^2)) + F(x, -sqrt(1-x^2)))/2, Chebyshev(xmin..xmax), 50)
    Fo = Fun( x -> (F(x,sqrt(1-x^2)) - F(x, -sqrt(1-x^2)))/2, JacobiWeight(0,1/2,Chebyshev(xmin..xmax)), 50)

    Fcoeffs = zeros(2N+1)
    Fcoeffs[1] = sum(w*Tkh[1]*Fe)
    for k = 1:N
        Fcoeffs[2k] = sum(w*Tkh[k+1]*Fe)
        Fcoeffs[2k+1] = sum(Ukh[k]*Fo)
    end
    return Fcoeffs
end

#=
Gather the coefficients of the expansion of an ApproxFun.Fun in the arc "Q"
OP basis.
=#
function func_to_Q_coeffs(F, N, h)
    xmin = h
    xmax = 1
    X = Fun(identity, xmin..xmax)
    w = X*sqrt(1-X^2)
    Ũkh,_,_ = lanczos(w,N+1)
    w = X/sqrt(1-X^2)
    T̃kh,_,_ = lanczos(w,N+1)

    # Y = sqrt(1-X^2)
    # Fe = (F.(X,Y) + F.(X,-Y))/2
    # Fo = (F.(X,Y) - F.(X,-Y))/2
    Fe = Fun( x -> (F(x,sqrt(1-x^2)) + F(x, -sqrt(1-x^2)))/2, Chebyshev(xmin..xmax), 50)
    Fo = Fun( x -> (F(x,sqrt(1-x^2)) - F(x, -sqrt(1-x^2)))/2, JacobiWeight(0,1/2,Chebyshev(xmin..xmax)), 50)

    Fcoeffs = zeros(2N+1)
    Fcoeffs[1] = sum(w*T̃kh[1]*Fe)
    for k = 1:N
        Fcoeffs[2k] = sum(w*T̃kh[k+1]*Fe)
        Fcoeffs[2k+1] = sum(X*Ũkh[k]*Fo)
    end
    return Fcoeffs
end

#=
Gather the matrix coefficients of the expansion of the Nth arc OP in the arc "Q"
OP basis.
=#
# function get_op_in_Q_basis_coeff_mats(N, h)
#     if N == 0
#         A = Vector{Matrix{Float64}}(1)
#         P1 = (x,y)->arc_op_eval(0, h, x, y)
#         P1c = func_to_Q_coeffs(P1, 0, h)
#         A[1] = zeros(1,1)
#         A[1][1] = P1c[1]
#         return A
#     end
#     A = Vector{Matrix{Float64}}(2)
#     P1 = (x,y)->arc_op_eval(N, h, x, y)[1]
#     P2 = (x,y)->arc_op_eval(N, h, x, y)[2]
#     P1c = func_to_Q_coeffs(P1, N, h)
#     P2c = func_to_Q_coeffs(P2, N, h)
#     if N == 1
#         A[1] = zeros(2,1)
#         A[1][1] = P1c[1]
#         A[1][2] = P2c[1]
#         A[2] = zeros(2,2)
#         A[2] = [P1c[2] P1c[3]; P2c[2] P2c[3]]
#     else
#         j = 1
#         for n = N-1:N
#             A[j] = zeros(2,2)
#             A[j] = [P1c[2n] P1c[2n+1]; P2c[2n] P2c[2n+1]]
#             j += 1
#         end
#     end
#     return A
# end
function get_op_in_Q_basis_coeff_mats(N, h)
    xmin = h
    xmax = 1
    X = Fun(identity, xmin..xmax)
    w = sqrt(1-X^2)
    Ukh,_,_ = lanczos(w,N+1)
    w = 1/sqrt(1-X^2)
    Tkh,_,_ = lanczos(w,N+1)
    if N == 0
        A = Vector{Matrix{Float64}}(1)
        P1 = (x,y)->Tkh[1](x)
        P1c = func_to_Q_coeffs(P1, 0, h)
        A[1] = zeros(1,1)
        A[1][1] = P1c[1]
        return A
    end
    A = Vector{Matrix{Float64}}(2)
    P1 = (x,y)->Tkh[N+1](x)
    P2 = (x,y)->Ukh[N](x)*y
    P1c = func_to_Q_coeffs(P1, N, h)
    P2c = func_to_Q_coeffs(P2, N, h)
    if N == 1
        A[1] = zeros(2,1)
        A[1][1] = P1c[1]
        A[1][2] = P2c[1]
        A[2] = zeros(2,2)
        A[2] = [P1c[2] P1c[3]; P2c[2] P2c[3]]
    else
        j = 1
        for n = N-1:N
            A[j] = zeros(2,2)
            A[j] = [P1c[2n] P1c[2n+1]; P2c[2n] P2c[2n+1]]
            j += 1
        end
    end
    return A
end

#=
Gather the matrix coefficients of the expansion of the Nth "Q" OP in the arc P
OP basis.
=#
function get_Q_in_op_basis_coeff_mats(N, h)
    xmin = h
    xmax = 1
    X = Fun(identity, xmin..xmax)
    w = X*sqrt(1-X^2)
    Ũkh,_,_ = lanczos(w,N+1)
    w = X/sqrt(1-X^2)
    T̃kh,_,_ = lanczos(w,N+1)
    A = Vector{Matrix{Float64}}(N+1)
    if N == 0
        Q1 = (x,y)->T̃kh[1](x)
        Q1c = func_to_coeffs(Q1, 0, h)
        A[1] = zeros(1,1)
        A[1][1] = Q1c[1]
        return A
    end
    Q1 = (x,y)->T̃kh[N+1](x)
    Q2 = (x,y)->Ũkh[N](x)*y
    Q1c = func_to_coeffs(Q1, N, h)
    Q2c = func_to_coeffs(Q2, N, h)
    A[1] = zeros(2,1)
    A[1][1] = Q1c[1]
    A[1][2] = Q2c[1]
    j = 2
    for n = 1:N
        A[n+1] = zeros(2,2)
        A[n+1] = [Q1c[j] Q1c[j+1]; Q2c[j] Q2c[j+1]]
        j += 2
    end
    return A
end

# Gain coeffs of ∂/∂s(P_N) in OP basis (coeffs are matrices)
function get_derivative_op_basis_coeff_mats(N, h)
    xmin = h
    xmax = 1
    X = Fun(identity, xmin..xmax)
    w = sqrt(1-X^2)
    Ukh,_,_ = lanczos(w,N+1)
    w = 1/sqrt(1-X^2)
    Tkh,_,_ = lanczos(w,N+1)
    dP1 = (x,y)->arc_op_derivative_eval(N,h,x,y)[1]
    dP2 = (x,y)->arc_op_derivative_eval(N,h,x,y)[2]
    dP1e = Fun( x -> (dP1(x,sqrt(1-x^2)) + dP1(x, -sqrt(1-x^2)))/2, Chebyshev(xmin..xmax), 10)
    dP1o = Fun( x -> (dP1(x,sqrt(1-x^2)) - dP1(x, -sqrt(1-x^2)))/2, JacobiWeight(0,1/2,Chebyshev(xmin..xmax)), 50)
    dP2e = Fun( x -> (dP2(x,sqrt(1-x^2)) + dP2(x, -sqrt(1-x^2)))/2, Chebyshev(xmin..xmax), 50)
    dP2o = Fun( x -> (dP2(x,sqrt(1-x^2)) - dP2(x, -sqrt(1-x^2)))/2, JacobiWeight(0,1/2,Chebyshev(xmin..xmax)), 10)
    Ae = Vector{Matrix{Float64}}(N+1)
    Ae[1] = zeros(2,1)
    Ao = copy(Ae)
    Ae[1][1] = sum(w*Tkh[1]*dP1e)
    Ae[1][2] = sum(w*Tkh[1]*dP2e)
    for j = 1:N
        n = j+1
        Ae[n] = zeros(2,2)
        Ao[n] = zeros(2,2)
        Ae[n][1,1] = sum(w*Tkh[n]*dP1e)
        Ae[n][2,1] = sum(w*Tkh[n]*dP2e)
        Ao[n][1,2] = sum(Ukh[n-1]*dP1o)
        Ao[n][2,2] = sum(Ukh[n-1]*dP2o)
    end
    A = Ae .+ Ao
    return A
end

#=
Find 2x2 matrices A_N, B_N s.t. ∂P_N/∂s = A_N*Q_N, B_N*Q_{N-1}.
Note Q_k(x,y) = [T̃_k^h(x); y*Ũ_{k-1}^h(x)]
=#
function get_derivative_op_Q_coeff_mats(N, h)
    xmin = h
    xmax = 1
    X = Fun(identity, xmin..xmax)
    w = X * sqrt(1-X^2)
    Ũkh,_,_ = lanczos(w,N+1)
    w = X / sqrt(1-X^2)
    T̃kh,_,_ = lanczos(w,N+1)
    dP1 = (x,y)->arc_op_derivative_eval(N,h,x,y)[1]
    dP2 = (x,y)->arc_op_derivative_eval(N,h,x,y)[2]
    # dP1e = Fun( x -> (dP1(x,sqrt(1-x^2)) + dP1(x, -sqrt(1-x^2)))/2, Chebyshev(xmin..xmax), 10)
    dP1o = Fun( x -> (dP1(x,sqrt(1-x^2)) - dP1(x, -sqrt(1-x^2)))/2, JacobiWeight(0,1/2,Chebyshev(xmin..xmax)), 50)
    dP2e = Fun( x -> (dP2(x,sqrt(1-x^2)) + dP2(x, -sqrt(1-x^2)))/2, Chebyshev(xmin..xmax), 50)
    # dP2o = Fun( x -> (dP2(x,sqrt(1-x^2)) - dP2(x, -sqrt(1-x^2)))/2, JacobiWeight(0,1/2,Chebyshev(xmin..xmax)), 10)
    A = Vector{Matrix{Float64}}(2)
    if N == 1
        A[1] = zeros(2,1)
        # A[1][1] = sum(w*T̃kh[1]*dP1e)
        A[1][2] = sum(w*T̃kh[1]*dP2e)
        A[2] = zeros(2,2)
        A[2][2,1] = sum(w*T̃kh[2]*dP2e)
        A[2][1,2] = sum(X*Ũkh[1]*dP1o)
    else
        j = 1
        for n = N-1:N
            A[j] = zeros(2,2)
            # A[n][1,1] = sum(w*T̃kh[n]*dP1e)
            A[j][2,1] = sum(w*T̃kh[n+1]*dP2e)
            A[j][1,2] = sum(X*Ũkh[n]*dP1o)
            # A[n][2,2] = sum(X*Ũkh[n-1]*dP2o)
            j += 1
        end
    end
    return A
end

# Operator matrix for conversion from P basis to Q basis
function arc2Q(N, h)
    cols = rows = [1; round.(Int,2*ones(N))]  # block sizes
    l,u = 0,1
    λ,μ = 1,1
    C = BandedBlockBandedMatrix(0.0*I, (rows,cols), (l,u), (λ,μ))
    view(C, Block(1,1)) .= get_op_in_Q_basis_coeff_mats(0, h)[1]
    for k=1:N
        println(k)
        A = get_op_in_Q_basis_coeff_mats(k, h)
        view(C, Block(k,k+1)) .= A[1].'
        view(C, Block(k+1,k+1)) .= A[2].'
    end
    return C
end

# Operator matrix for conversion from Q basis to P basis
function Q2arc(N, h)
    cols = rows = [1; round.(Int,2*ones(N))]  # block sizes
    l,u = 0,N
    λ,μ = 1,1
    C = BandedBlockBandedMatrix(0.0I, (rows,cols), (l,u), (λ,μ))
    view(C, Block(1,1)) .= get_Q_in_op_basis_coeff_mats(0, h)[1]
    for k=1:N
        println(k)
        A = get_Q_in_op_basis_coeff_mats(k, h)
        for n = 1:k+1
            view(C, Block(n,k+1)) .= A[n].'
        end
    end
    return C
end

# Operator matrix for ∂/∂s. Acts on arc OP coeffs and results in arc OP coeffs.
function arc_derivative_operator(N, h)
    cols = rows = [1; round.(Int,2*ones(N))]  # block sizes
    l,u = 0,N          # block bandwidths
    λ,μ = 1,1          # sub-block bandwidths: the bandwidths of each block
    Ds = BandedBlockBandedMatrix(0.0I, (rows,cols), (l,u), (λ,μ))
    for k=1:N
        println(k)
        A = get_derivative_op_basis_coeff_mats(k, h)
        for n = 1:k
            view(Ds, Block(n+1,k+1)) .= A[n+1].'
        end
    end
    return Ds
end

# Operator matrix for ∂/∂s. Acts on arc OP coeffs and results in Q coeffs.
function arc_derivative_operator_in_Q(N, h)
    cols = rows = [1; round.(Int,2*ones(N))]  # block sizes
    l,u = 0,1          # block bandwidths
    λ,μ = 1,1          # sub-block bandwidths: the bandwidths of each block
    D̃s = BandedBlockBandedMatrix(0.0I, (rows,cols), (l,u), (λ,μ))
    for k=1:N
        println(k)
        A = get_derivative_op_Q_coeff_mats(k, h)
        j = 1
        for n = k-1:k
            view(D̃s, Block(n+1,k+1)) .= A[j].'
            j += 1
        end
    end
    return D̃s
end

# #=======#
#
# N, h = 5, 0
# α, β, γ, δ, a, b, c, d, e, f = initialise_arc_ops(N, h)
# x = 0.8
# y = sqrt(1-x^2)
#
# #=======#
#
# # Test for function evaluation method
# F = rand(2N+1)
# Feval = arc_func_eval(F, h, α, β, γ, δ, a, b, c, d, e, f, x, y)
# xmin = h
# xmax = 1
# X = Fun(identity, xmin..xmax)
# w = (1/sqrt(1-X^2))
# Tkh,_,_ = lanczos(w,N+1)
# w = sqrt(1-X^2)
# Ukh,_,_ = lanczos(w,N+1)
# Factual = F[1]*Tkh[1](x)
# for n = 1:N
#     Factual += F[2n]*Tkh[n+1](x) + F[2n+1]*y*Ukh[n](x)
# end
# @test Factual ≈ Feval
#
# #=======#
#
# # Test for ApproxFun.Fun to coefficients method
# F = Fun((x,y)->exp(x+y))
# Fcoeffs = func_to_coeffs(F, N, h)
# tol = 1e-4
# @test abs(arc_func_eval(Fcoeffs, h, α, β, γ, δ, a, b, c, d, e, f, x, y) - F(x,y)) < tol
#
# #=======#
#
# # Check the derivative evaluation function
# dP1 = (x,y)->arc_op_derivative_eval(N,h,x,y)[1]
# dP2 = (x,y)->arc_op_derivative_eval(N,h,x,y)[2]
# θ = acos(x); dt = 1e-9; findiff = (arc_op_eval(N,h,cos(θ+dt),sin(θ+dt))-arc_op_eval(N,h,x,y))/dt
# tol = 1e-5
# @test (abs(findiff[1] - dP1(x,y)) < tol && abs(findiff[2] - dP2(x,y)) < tol)
#
# #=======#
#
# # Gain the coeffs vec for a function in Q basis
# u = Fun((x,y)->exp(x+y))
# uc = func_to_Q_coeffs(u, N, h)
# ueval = Q_func_eval(uc, h, x, y)
# @test abs(ueval - u(x,y)) < 1e-5
#
# #=======#
#
# # Operator matrix for conversion from Q basis to P basis (and back)
# Q2P = Q2arc(N, h)
# u = Fun((x,y)->exp(x+y))
# uc = func_to_coeffs(u, N, h)
# ucQ = func_to_Q_coeffs(u, N, h)
# tol = 1e-4
# @test norm(Q2P*ucQ - uc) < tol
# # P2Q = arc2Q(N, h)
# # @test norm(P2Q*uc - ucQ) < tol
#
# #=======#
#
# # Example
# D̃s = arc_derivative_operator_in_Q(N, h)
# u = Fun((x,y)->exp(x+y))
# dsu = (x,y)->(x-y)*exp(x+y)
# uc = func_to_coeffs(u, N, h)
# dsuc = Q2P*D̃s*uc
# dsu_eval = arc_func_eval(dsuc, h, α, β, γ, δ, a, b, c, d, e, f, x, y)
# dsu(x,y)
# tol = 1e-4
# @test abs(dsu(x,y) - dsu_eval) < tol
