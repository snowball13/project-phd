using ApproxFun
using BlockBandedMatrices
using BlockArrays
using Base.Test

let

    #=
    Setup
    =#
    global function innerTh(T, U, h)
        xmin = h
        xmax = 1
        x = Fun(identity, xmin..xmax)
        w = (1/sqrt(1-x^2))
        return sum(w*T*U)
    end

    global function innerUh(T, U, h)
        xmin = h
        xmax = 1
        x = Fun(identity, xmin..xmax)
        w = sqrt(1-x^2)
        return sum(w*T*U)
    end

    global function initialise_arc_ops(N, h)
        @assert (N > 1)

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
    global function Jx(N, α, β, γ, δ)
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

    global function Jy(N, a, b, c, d, e, f)
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
    global function get_arc_clenshaw_matrices(N, α, β, γ, δ, a, b, c, d, e, f)
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
    global function arc_op_eval(N, h, x, y)
        # We only deal with N = 0,1 for now (useful for Clenshaw)
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
    Function to obtain the evaluation of a function f(x,y)
    where f is input as the coefficients of its expansion in the arc space
    basis of the circle, i.e.
        f(x, y) = sum(vecdot(f_n, P_n))
    where the {P_n} are the order n OPs for the arc.

    Uses the Clenshaw Algorithm.
    =#
    global function arc_func_eval(F, h, α, β, γ, δ, a, b, c, d, e, f, x, y)
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


    #=
    Gather the coefficients of the expansion of an ApproxFun.Fun in the arc OP
    basis.
    =#
    global function func_to_coeffs(F, N, h)
        xmin = h
        xmax = 1
        X = Fun(identity, xmin..xmax)
        w = sqrt(1-X^2)
        Ukh,γ,δ = lanczos(w,N+1)
        w = (1/sqrt(1-X^2))
        Tkh,α,β = lanczos(w,N+1)

        Y = sqrt(1-X^2)
        Fe = (F.(X,Y) + F.(X,-Y))/2
        Fo = (F.(X,Y) - F.(X,-Y))/2

        Fcoeffs = zeros(2N+1)
        Fcoeffs[1] = innerTh(Tkh[1],Fe,h)
        for k = 1:N
            Fcoeffs[2k] = sum(w*Tkh[k+1]*Fe)
            Fcoeffs[2k+1] = sum(Ukh[k]*Fo)
        end
        return Fcoeffs
    end

end

#=======#

N, h = 30, 0
α, β, γ, δ, a, b, c, d, e, f = initialise_arc_ops(N, h)
x = 0.8
y = sqrt(1-x^2)

#=======#

# Test for function evaluation method
F = ones(2N+1)
Feval = arc_func_eval(F, h, α, β, γ, δ, a, b, c, d, e, f, x, y)
xmin = h
xmax = 1
X = Fun(identity, xmin..xmax)
w = (1/sqrt(1-X^2))
Tkh,_,_ = lanczos(w,N+1)
w = sqrt(1-X^2)
Ukh,_,_ = lanczos(w,N+1)
Factual = Tkh[1](x)
for n = 1:N
    Factual += F[2n]*Tkh[n+1](x) + F[2n+1]*y*Ukh[n](x)
end
@test Factual ≈ Feval

#=======#

# Test for ApproxFun.Fun to coefficients method

F = Fun((x,y)->exp(x+y))
Fcoeffs = func_to_coeffs(F, N, h)
@test arc_func_eval(Fcoeffs, h, α, β, γ, δ, a, b, c, d, e, f, x, y) ≈ F(x,y)
