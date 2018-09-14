#=
hemisphere
=#

using ApproxFun
    using SphericalHarmonics
    using BlockBandedMatrices
    using BlockArrays
    using Base.Test

# We define the hemisphere by x ∈ [0,1], y ∈ [-1,1], z = sqrt(1-x^2-y^2).

function sh_eval_real(l, x, y, z)
    Y = sh_eval(l, x, y, z)[l^2+1:end]
    Yre = zeros(2l+1)
    rt2 = sqrt(2)
    for m = -l:l
        if m < 0
            Yre[l+m+1] = im*(Y[l+m+1] - Y[l-m+1]*(-1)^m)/rt2
        elseif m > 0
            Yre[l+m+1] = (Y[l-m+1] + Y[l+m+1]*(-1)^m)/rt2
        else
            Yre[l+m+1] = real(Y[l+m+1])
        end
    end
    return Yre
end

function getTanonfunc(n, k, pn, qn, ρ)
    return (x,y)->pn[k+1][n-k+1](x)*ρ(x)^k*qn[k+1](y/ρ(x))
end
function getUanonfunc(n, k, pn, qn, ρ)
    return (x,y)->pn[k+1][n-k+1](x)*ρ(x)^k*qn[k+1](y/ρ(x))
end

function getOPanonfuncT(n, k, Tn)
    return (x,y,z)->Tn[n+1][k+1](x,y)
end
function getOPanonfuncU(n, k, Un)
    return (x,y,z)->z*Un[n][k+1](x,y)
end

#=====#

function getOPanonfunc(N, kind='P')
    # T_n (μ = 0)
    μ = 0
    X1 = Fun(identity, 0..1)
    X2 = Fun(identity, -1..1)
    w2 = 1/sqrt(1-X2^2)
    ρ = sqrt(1-X1^2)
    qnT,_,_ = lanczos(w2,N+1)
    pnT = Vector{Vector{ApproxFun.Fun}}(N+1)
    if kind == 'P'
        for k = 0:N
            pnT[k+1],_,_ = lanczos((1-X1^2)^(k+μ), N+1)
        end
    elseif kind == 'Q'
        for k = 0:N
            pnT[k+1],_,_ = lanczos(X1*(1-X1^2)^(k+μ), N+1)
        end
    else
        error("'kind' value not valid - must be 'P' or 'Q'")
        return
    end
    Tn = Vector{Vector{Function}}(N+1)
    for n = 0:N
        Tn[n+1] = Vector{Function}(n+1)
        for k = 0:n
            Tn[n+1][k+1] = getTanonfunc(n, k, pnT, qnT, ρ)
        end
    end

    # U_n (μ = 1)
    μ = 1
    X1 = Fun(identity, 0..1)
    X2 = Fun(identity, -1..1)
    w2 = sqrt(1-X2^2)
    ρ = sqrt(1-X1^2)
    qnU,_,_ = lanczos(w2,N+1)
    pnU = Vector{Vector{ApproxFun.Fun}}(N+1)
    if kind == 'P'
        for k = 0:N
            pnU[k+1],_,_ = lanczos((1-X1^2)^(k+μ), N+1)
        end
    elseif kind == 'Q'
        for k = 0:N
            pnU[k+1],_,_ = lanczos(X1*(1-X1^2)^(k+μ), N+1)
        end
    else
        error("'kind' value not valid - must be 'P' or 'Q'")
        return
    end
    Un = Vector{Vector{Function}}(N+1)
    for n = 0:N
        Un[n+1] = Vector{Function}(n+1)
        for k = 0:n
            Un[n+1][k+1] = getUanonfunc(n, k, pnU, qnU, ρ)
        end
    end

    # Put together : Pn = [Tn(x,y); z*Un(x,y)]
    P = Vector{Vector{Function}}(N+1)
    for n = 0:N
        P[n+1] = Vector{Function}(2n+1)
        j = 1
        for k = 0:n
            # Rn[n+1][j] = (x,y,z)->Pn[n+1][k+1](x,y)
            P[n+1][j] = getOPanonfuncT(n, k, Tn)
            j += 1
        end
        for k = 0:n-1
            # Rn[n+1][j] = (x,y,z)->z*Qn[n][k+1](x,y)
            P[n+1][j] = getOPanonfuncU(n, k, Un)
            j += 1
        end
    end
    return P
end

# OPs (P basis) wrt uniform weight
function hemisphere_op_eval(N, x, y, z)
    P = getOPanonfunc(N)
    out = zeros((N+1)^2)
    j = 1
    for l = 0:N
        for m = 1:2l+1
            out[j] = P[l+1][m](x,y,z)
            j += 1
        end
    end
    return out
end

# Alt OPs (Q basis) wrt weight x*uniform
function hemisphere_op_eval_Q(N, x, y, z)
    Q = getOPanonfunc(N, kind = 'Q')
    out = zeros((N+1)^2)
    j = 1
    for l = 0:N
        for m = 1:2l+1
            out[j] = Q[l+1][m](x,y,z)
            j += 1
        end
    end
    return out
end

#=
The following coeff functions give the coefficients used in the
relations for mult by x, y, z of the OPs (or Q basis OPs)
=#
function get1dops(N, kind = 'P')
    # T_n (μ = 0)
    μ = 0
    Xp = Fun(identity, 0..1)
    Xq = Fun(identity, -1..1)
    wqT = 1/sqrt(1-Xq^2)
    qT, a, b = lanczos(wqT, N+1)
    δ = b[1] # noted as is not 0.5 like the rest of the vector b
    pT = Vector{Vector{ApproxFun.Fun}}(N+1)
    αT = Vector{Vector{Vector{Float64}}}(N+1)
    wpT = Vector{ApproxFun.Fun}(N+1)
    if kind == 'P'
        for k = 0:N
            αT[k+1] = Vector{Vector{Float64}}(2)
            wpT[k+1] = (1-Xp^2)^(k+μ)
            pT[k+1], αT[k+1][1], αT[k+1][2] = lanczos(wpT[k+1], N+1)
        end
    elseif kind == 'Q'
        for k = 0:N
            αT[k+1] = Vector{Vector{Float64}}(2)
            wpT[k+1] = Xp * (1-Xp^2)^(k+μ)
            pT[k+1], αT[k+1][1], αT[k+1][2] = lanczos(wpT[k+1], N+1)
        end
    end

    # U_n (μ = 1)
    μ = 1
    wqU = sqrt(1-Xq^2)
    qU, _, _ = lanczos(wqU,N+1)
    pU = Vector{Vector{ApproxFun.Fun}}(N+1)
    αU = Vector{Vector{Vector{Float64}}}(N+1)
    wpU = Vector{ApproxFun.Fun}(N+1)
    if kind == 'P'
        for k = 0:N
            αU[k+1] = Vector{Vector{Float64}}(2)
            wpU[k+1] = (1-Xp^2)^(k+μ)
            pU[k+1], αU[k+1][1], αU[k+1][2] = lanczos(wpU[k+1], N+1)
        end
    elseif kind == 'Q'
        for k = 0:N
            αU[k+1] = Vector{Vector{Float64}}(2)
            wpU[k+1] = Xp * (1-Xp^2)^(k+μ)
            pU[k+1], αU[k+1][1], αU[k+1][2] = lanczos(wpU[k+1], N+1)
        end
    end

    return pT, wpT, qT, wqT, pU, wpU, qU, wqU, Xp, Xq, αT, αU, δ
end

function coeffAlpha(n, k, m, α)
    return α[k+1][m][n-k+1]
end

function coeffBetaT(n, k, m, p, wp, Xp, δ)
    fac = 0.5
    int = 0.0
    if m == 1
        int = sum(wp[k+1] * p[k+1][n-k+1] * p[k][n-k+1])
        if k == 1
            fac = δ
        end
    elseif m == 2
        int = sum(wp[k+1] * p[k+1][n-k+1] * (1-Xp^2) * p[k+2][n-k-1])
        if k == 0
            fac = δ
        end
    elseif m == 3
        int = sum(wp[k+1] * p[k+1][n-k+1] * p[k][n-k+2])
        if k == 1
            fac = δ
        end
    elseif m == 4
        int = sum(wp[k+1] * p[k+1][n-k+1] * (1-Xp^2) * p[k+2][n-k])
        if k == 0
            fac = δ
        end
    elseif m == 5
        int = sum(wp[k+1] * p[k+1][n-k+1] * p[k][n-k+3])
        if k == 1
            fac = δ
        end
    else
        int = sum(wp[k+1] * p[k+1][n-k+1] * (1-Xp^2) * p[k+2][n-k+1])
        if k == 0
            fac = δ
        end
    end
    return fac * int
end

function coeffBetaU(n, k, m, p, wp, Xp)
    fac = 0.5
    int = 0.0
    if m == 1
        int = sum(wp[k+1] * p[k+1][n-k+1] * p[k][n-k+1])
    elseif m == 2
        int = sum(wp[k+1] * p[k+1][n-k+1] * (1-Xp^2) * p[k+2][n-k-1])
    elseif m == 3
        int = sum(wp[k+1] * p[k+1][n-k+1] * p[k][n-k+2])
    elseif m == 4
        int = sum(wp[k+1] * p[k+1][n-k+1] * (1-Xp^2) * p[k+2][n-k])
    elseif m == 5
        int = sum(wp[k+1] * p[k+1][n-k+1] * p[k][n-k+3])
    else
        int = sum(wp[k+1] * p[k+1][n-k+1] * (1-Xp^2) * p[k+2][n-k+1])
    end
    return fac * int
end

function coeffGammaT(n, k, m, Z)
    pT, wpT, qT, wqT, pU, wpU, qU, wqU, Xp, Xq, αT, αU, δ = Z
    if m == 1
        return (sum(wqU * qT[k+1] * qU[k-1])
                * sum(wpU[k-1] * pT[k+1][n-k+1] * (1-Xp^2) * pU[k-1][n-k+1]))
    elseif m == 2
        return (sum(wqU * qT[k+1] * qU[k+1])
                * sum(wpU[k+1] * pT[k+1][n-k+1] * pU[k+1][n-k-1]))
    elseif m == 3
        return (sum(wqU * qT[k+1] * qU[k-1])
                * sum(wpU[k-1] * pT[k+1][n-k+1] * (1-Xp^2) * pU[k-1][n-k+2]))
    elseif m == 4
        return (sum(wqU * qT[k+1] * qU[k+1])
                * sum(wpU[k+1] * pT[k+1][n-k+1] * pU[k+1][n-k]))
    elseif m == 5
        return (sum(wqU * qT[k+1] * qU[k-1])
                * sum(wpU[k-1] * pT[k+1][n-k+1] * (1-Xp^2) * pU[k-1][n-k+3]))
    else
        return (sum(wqU * qT[k+1] * qU[k+1])
                * sum(wpU[k+1] * pT[k+1][n-k+1] * pU[k+1][n-k+1]))
    end
end

function coeffGammaU(n, k, m, Z)
    if m == 1
        return coeffGammaT(n, k, 7-m, Z)
    elseif m == 2
        return coeffGammaT(n, k+2, 7-m, Z)
    elseif m == 3
        return coeffGammaT(n+1, k, 7-m, Z)
    elseif m == 4
        return coeffGammaT(n+1, k+2, 7-m, Z)
    elseif m == 5
        return coeffGammaT(n+2, k, 7-m, Z)
    else
        return coeffGammaT(n+2, k+2, 7-m, Z)
    end
end

#=
Functions to obtain the matrices corresponding to multiplication of the OPs
by x, y and z respectively (i.e. the Jacobi operators J^x, J^y and J^z)
=#
function jacobiAx(n, Z)
    pT, wpT, qT, wqT, pU, wpU, qU, wqU, Xp, Xq, αT, αU, δ = Z
    m = 2
    if n == 0
        return coeffAlpha(0, 0, m, αT), 0, 0
    end
    A = zeros(2n+1, 2n+3)
    for k = 0:n-1
        view(A, k+1, k+1) .= coeffAlpha(n, k, m, αT)
        view(A, n+k+2, n+k+3) .= coeffAlpha(n-1, k, m, αU)
    end
    view(A, n+1, n+1) .= coeffAlpha(n, n, m, αT)
    return A
end

function jacobiBx(n, Z)
    pT, wpT, qT, wqT, pU, wpU, qU, wqU, Xp, Xq, αT, αU, δ = Z
    m = 1
    if n == 0
        return coeffAlpha(0, 0, m, αT)
    end
    D = zeros(2n+1)
    for k = 0:n-1
        view(D, k+1) .= coeffAlpha(n, k, m, αT)
        view(D, n+k+2) .= coeffAlpha(n-1, k, m, αU)
    end
    view(D, n+1) .= coeffAlpha(n, n, m, αT)
    return Diagonal(D)
end

function jacobiCx(n, Z)
    pT, wpT, qT, wqT, pU, wpU, qU, wqU, Xp, Xq, αT, αU, δ = Z
    m = 2
    if n == 1
        return coeffAlpha(0, 0, m, αT), 0, 0
    end
    C = zeros(2n+1, 2n-1)
    for k = 0:n-2
        view(C, k+1, k+1) .= coeffAlpha(n-1, k, m, αT)
        view(C, n+k+2, n+k+1) .= coeffAlpha(n-2, k, m, αU)
    end
    view(C, n, n) .= coeffAlpha(n-1, n-1, m, αT)
    return C
end

function jacobiAy(n, Z)
    pT, wpT, qT, wqT, pU, wpU, qU, wqU, Xp, Xq, αT, αU, δ = Z
    upperm = 6
    lowerm = 5
    if n == 0
        return 0, coeffBetaT(0, 0, upperm, pT, wpT, Xp, δ), 0
    end
    A = zeros(2n+1, 2n+3)
    view(A, 1, 2) .= coeffBetaT(n, 0, upperm, pT, wpT, Xp, δ)
    view(A, n+2, n+4) .= coeffBetaU(n-1, 0, upperm, pU, wpU, Xp)
    for k = 1:n-1
        view(A, k+1, k:k+2) .= coeffBetaT(n, k, lowerm, pT, wpT, Xp, δ),
                                0,
                                coeffBetaT(n, k, upperm, pT, wpT, Xp, δ)
        view(A, n+k+2, n+k+2:n+k+4) .= coeffBetaU(n-1, k, lowerm, pU, wpU, Xp),
                                        0,
                                        coeffBetaU(n-1, k, upperm, pU, wpU, Xp)
    end
    view(A, n+1, n:n+2) .= coeffBetaT(n, n, lowerm, pT, wpT, Xp, δ),
                            0,
                            coeffBetaT(n, n, upperm, pT, wpT, Xp, δ)
    return A
end

function jacobiBy(n, Z)
    pT, wpT, qT, wqT, pU, wpU, qU, wqU, Xp, Xq, αT, αU, δ = Z
    if n == 0
        return 0
    end
    upperD = zeros(2n)
    lowerD = zeros(2n)
    upperm = 4
    lowerm = 3
    if n == 1
        view(upperD, 1:2) .= coeffBetaT(n, 0, upperm, pT, wpT, Xp, δ), 0
        view(lowerD, 1:2) .= coeffBetaT(n, 1, lowerm, pT, wpT, Xp, δ), 0
        return Tridiagonal(lowerD, zeros(3), upperD)
    end
    view(upperD, 1) .= coeffBetaT(n, 0, upperm, pT, wpT, Xp, δ)
    view(upperD, n+2) .= coeffBetaU(n-1, 0, upperm, pU, wpU, Xp)
    for k = 1:n-2
        view(upperD, k+1) .= coeffBetaT(n, k, upperm, pT, wpT, Xp, δ)
        view(upperD, n+k+2) .= coeffBetaU(n-1, k, upperm, pU, wpU, Xp)
        view(lowerD, k) .= coeffBetaT(n, k, lowerm, pT, wpT, Xp, δ)
        view(lowerD, n+k+1) .= coeffBetaU(n-1, k, lowerm, pU, wpU, Xp)
    end
    view(lowerD, n-1) .= coeffBetaT(n, n-1, lowerm, pT, wpT, Xp, δ)
    view(lowerD, 2n) .= coeffBetaU(n-1, n-1, lowerm, pU, wpU, Xp)
    view(upperD, n) .= coeffBetaT(n, n-1, upperm, pT, wpT, Xp, δ)
    view(lowerD, n) .= coeffBetaT(n, n, lowerm, pT, wpT, Xp, δ)
    return Tridiagonal(lowerD, zeros(2n+1), upperD)
end

function jacobiCy(n, Z)
    pT, wpT, qT, wqT, pU, wpU, qU, wqU, Xp, Xq, αT, αU, δ = Z
    upperm = 2
    lowerm = 1
    if n == 1
        return 0, coeffBetaT(1, 1, lowerm, pT, wpT, Xp, δ), 0
    end
    C = zeros(2n+1, 2n-1)
    view(C, 2, 1) .= coeffBetaT(n, 1, lowerm, pT, wpT, Xp, δ)
    view(C, n+3, n+1) .= coeffBetaU(n-1, 1, lowerm, pU, wpU, Xp)
    for k = 0:n-3
        view(C, k+1:k+3, k+2) .= coeffBetaT(n, k, upperm, pT, wpT, Xp, δ),
                                    0,
                                    coeffBetaT(n, k+2, lowerm, pT, wpT, Xp, δ)
        view(C, n+k+2:n+k+4, n+k+2) .= coeffBetaU(n-1, k, upperm, pU, wpU, Xp),
                                        0,
                                        coeffBetaU(n-1, k+2, lowerm, pU, wpU, Xp)
    end
    view(C, n-1:n+1, n) .= coeffBetaT(n, n-2, upperm, pT, wpT, Xp, δ),
                            0,
                            coeffBetaT(n, n, lowerm, pT, wpT, Xp, δ)
    return C
end

function jacobiAz(n, Z)
    upperm = 6
    lowerm = 5
    if n == 0
        return 0, 0, coeffGammaT(0, 0, upperm, Z)
    end
    A = zeros(2n+1, 2n+3)
    if n == 1
        view(A, 1, 4) .= coeffGammaT(n, 0, upperm, Z)
        view(A, 2, 5) .= coeffGammaT(n, 1, upperm, Z)
        view(A, 3, 1:3) .= coeffGammaU(n-1, 0, lowerm, Z), 0, coeffGammaU(n-1, 0, upperm, Z)
        return A
    end
    for k = 0:1
        view(A, k+1, n+k+3) .= coeffGammaT(n, k, upperm, Z)
        view(A, n+k+2, k+1:k+3) .= coeffGammaU(n-1, k, lowerm, Z), 0, coeffGammaU(n-1, k, upperm, Z)
    end
    for k = 2:n-1
        view(A, k+1, n+k+1:n+k+3) .= coeffGammaT(n, k, lowerm, Z), 0, coeffGammaT(n, k, upperm, Z)
        view(A, n+k+2, k+1:k+3) .= coeffGammaU(n-1, k, lowerm, Z), 0, coeffGammaU(n-1, k, upperm, Z)
    end
    view(A, n+1, 2n+1:2n+3) .= coeffGammaT(n, n, lowerm, Z), 0, coeffGammaT(n, n, upperm, Z)
    return A
end

function jacobiBz(n, Z)
    upperm = 4
    lowerm = 3
    if n == 0
        return 0
    end
    B = zeros(2n+1, 2n+1)
    if n == 1
        view(B, 1, 3) .= coeffGammaT(n, 0, upperm, Z)
        view(B, 3, 1) .= coeffGammaU(n-1, 0, lowerm, Z)
        return B
    end
    for k = 0:1
        view(B, n+k+2, k+1) .= coeffGammaU(n-1, k, lowerm, Z)
    end
    for k = 0:n-3
        view(B, k+1:k+3, n+k+2) .= coeffGammaT(n, k, upperm, Z), 0, coeffGammaT(n, k+2, lowerm, Z)
        view(B, n+k+2:n+k+4, k+3) .= coeffGammaU(n-1, k, upperm, Z), 0, coeffGammaU(n-1, k+2, lowerm, Z)
    end
    view(B, n-1:n+1, 2n) .= coeffGammaT(n, n-2, upperm, Z), 0, coeffGammaT(n, n, lowerm, Z)
    view(B, n, 2n+1) .= coeffGammaT(n, n-1, upperm, Z)
    view(B, 2n, n+1) .= coeffGammaU(n-1, n-2, upperm, Z)
    return B
end

function jacobiCz(n, Z)
    upperm = 2
    lowerm = 1
    if n == 1
        return 0, 0, coeffGammaU(0, 0, lowerm, Z)
    end
    C = zeros(2n+1, 2n-1)
    view(C, n+2, 1) .= coeffGammaU(n-1, 0, lowerm, Z)
    view(C, n+3, 2) .= coeffGammaU(n-1, 1, lowerm, Z)
    for k = 0:n-3
        view(C, k+1:k+3, n+k+1) .= coeffGammaT(n, k, upperm, Z), 0, coeffGammaT(n, k+2, lowerm, Z)
        view(C, n+k+2:n+k+4, k+3) .= coeffGammaU(n-1, k, upperm, Z), 0, coeffGammaU(n-1, k+2, lowerm, Z)
    end
    view(C, n-1:n+1, 2n-1) .= coeffGammaT(n, n-2, upperm, Z), 0, coeffGammaT(n, n, lowerm, Z)
    return C
end

#=
Jacobi matrix for mult by x on the P basis
=#
function hemJx(N)
    Z = get1dops(N)
    l,u = 1, 1          # block bandwidths
    λ,μ = 1, 1          # sub-block bandwidths: the bandwidths of each block
    cols = rows = 1:2:(2N+1)  # block sizes
    J = BandedBlockBandedMatrix(0.0I, (rows,cols), (l,u), (λ,μ))
    J[1] = jacobiBx(0, Z)
    if N == 0
        return J
    end
    view(J, 1, 2:4) .= jacobiAx(0, Z)
    view(J, 2:4, 2:4) .= jacobiBx(1, Z)
    view(J, 2:4, 1) .= jacobiCx(1, Z)
    if N == 1
        return J
    end
    for n = 2:N
        view(J, (n-1)^2+1:n^2, n^2+1:(n+1)^2) .= jacobiAx(n-1, Z)
        view(J, n^2+1:(n+1)^2, n^2+1:(n+1)^2) .= jacobiBx(n, Z)
        view(J, n^2+1:(n+1)^2, (n-1)^2+1:n^2) .= jacobiCx(n, Z)
    end
    return J
end

#=
Jacobi matrix for mult by y on the P basis
=#
function hemJy(N)
    Z = get1dops(N)
    l,u = 1, 1          # block bandwidths
    λ,μ = 2, 2          # sub-block bandwidths: the bandwidths of each block
    cols = rows = 1:2:(2N+1)  # block sizes
    J = BandedBlockBandedMatrix(0.0I, (rows,cols), (l,u), (λ,μ))
    J[1] = jacobiBy(0, Z)
    if N == 0
        return J
    end
    view(J, 1, 2:4) .= jacobiAy(0, Z)
    view(J, 2:4, 2:4) .= jacobiBy(1, Z)
    view(J, 2:4, 1) .= jacobiCy(1, Z)
    if N == 1
        return J
    end
    for n = 2:N
        view(J, (n-1)^2+1:n^2, n^2+1:(n+1)^2) .= jacobiAy(n-1, Z)
        view(J, n^2+1:(n+1)^2, n^2+1:(n+1)^2) .= jacobiBy(n, Z)
        view(J, n^2+1:(n+1)^2, (n-1)^2+1:n^2) .= jacobiCy(n, Z)
    end
    return J
end

#=
Jacobi matrix for mult by z on the P basis
=#
function hemJz(N)
    Z = get1dops(N)
    l,u = 1, 1          # block bandwidths
    λ,μ = 2N+1, 2N+1          # sub-block bandwidths: the bandwidths of each block
    cols = rows = 1:2:(2N+1)  # block sizes
    J = BandedBlockBandedMatrix(0.0I, (rows,cols), (l,u), (λ,μ))
    J[1] = jacobiBz(0, Z)
    if N == 0
        return J
    end
    view(J, 1, 2:4) .= jacobiAz(0, Z)
    view(J, 2:4, 2:4) .= jacobiBz(1, Z)
    view(J, 2:4, 1) .= jacobiCz(1, Z)
    if N == 1
        return J
    end
    for n = 2:N
        view(J, (n-1)^2+1:n^2, n^2+1:(n+1)^2) .= jacobiAz(n-1, Z)
        view(J, n^2+1:(n+1)^2, n^2+1:(n+1)^2) .= jacobiBz(n, Z)
        view(J, n^2+1:(n+1)^2, (n-1)^2+1:n^2) .= jacobiCz(n, Z)
    end
    return J
end


# Methods to obtain the matrices used in the Clenshaw Algorithm
function clenshawDT(n, Z)
    pT, wpT, qT, wqT, pU, wpU, qU, wqU, Xp, Xq, αT, αU, δ = Z
    if n == 0
        return Diagonal([1/coeffAlpha(0, 0, 2, αT);
                         1/coeffBetaT(0, 0, 6, pT, wpT, Xp, δ);
                         1/coeffGammaT(0, 0, 6, Z)])
    end
    DT = zeros(2n+3, 3(2n+1))
    for k = 0:n
        view(DT, k+1, k+1) .= 1/coeffAlpha(n, k, 2, αT)
    end
    b = 0.5/coeffBetaT(n, n, 6, pT, wpT, Xp, δ)
    view(DT, n+2, 3n+2) .= b
    view(DT, n+2, 3(2n+1)) .= - b
    for k = 0:n-1
        view(DT, n+k+3, n+k+2) .= 1/coeffAlpha(n-1, k, 2, αU)
    end
    b = 0.5/coeffBetaU(n-1, n-1, 6, pU, wpU, Xp)
    view(DT, 2n+3, 2(2n+1)) .= b
    view(DT, 2n+3, 5n+3) .= b
    return DT
end

function clenshawB(n, Z)
    B = zeros(3(2n+1), 2n+1)
    view(B, 1:2n+1, 1:2n+1) .= jacobiBx(n, Z)
    view(B, 2n+2:2(2n+1), 1:2n+1) .= jacobiBy(n, Z)
    view(B, 4n+3:3(2n+1), 1:2n+1) .= jacobiBz(n, Z)
    return B
end

function clenshawC(n, Z)
    C = zeros(3(2n+1), 2n-1)
    view(C, 1:2n+1, 1:2n-1) .= jacobiCx(n, Z)
    view(C, 2n+2:2(2n+1), 1:2n-1) .= jacobiCy(n, Z)
    view(C, 4n+3:3(2n+1), 1:2n-1) .= jacobiCz(n, Z)
    return C
end

function clenshawG(n, x, y, z)
    G = zeros(3(2n+1), 2n+1)
    view(G, 1:2n+1, 1:2n+1) .= Diagonal(x * ones(2n+1))
    view(G, 2n+2:2(2n+1), 1:2n+1) .= Diagonal(y * ones(2n+1))
    view(G, 4n+3:3(2n+1), 1:2n+1) .= Diagonal(z * ones(2n+1))
    return G
end

# Obtain the submatrices used in the Clenshaw Alg
function get_hemisphere_clenshaw_matrices(N, Z)
    DT = Array{Matrix{Float64}}(N-1)
    K = copy(DT)
    L = copy(DT)
    for n = N-1:-1:1
        DT[n] = clenshawDT(n, Z)
        K[n] = DT[n] * clenshawB(n, Z)
        L[n] = DT[n] * clenshawC(n, Z)
    end
    return DT, K, L
end

#=
Function to obtain the evaluation of a function f(x,y,z)
where f is input as the coefficients of its expansion in the "P" hemisphere
basis, i.e.
f(x, y, z) = sum(vecdot(f_n, P_n(x,y,z)))
where the {P_n} are the order n OPs for the hemisphere.

Uses the Clenshaw Algorithm.
=#
function hemisphere_fun_eval(f, Z, x, y, z)
    # Check that (x, y, z) lies are on the unit hemisphere
    delta = 1e-5
    @assert (x^2 + y^2 + z^2 < 1 + delta &&  x^2 + y^2 + z^2 > 1 - delta && x >= 0 && x <= 1)
        "the point (x, y, z) must be on unit hemisphere defined by the x coord being in the range [0,1]"

    M = length(f)
    N = round(Int, sqrt(M) - 1)
    @assert (M > 0 && sqrt(M) - 1 == N) "invalid length of f"

    # Complete the reverse recurrance to gain gamma_1, gamma_2
    # Note that gamma_(N+1) = 0, gamma_(N+2) = 0
    P0 = hemisphere_op_eval(0, x, y, z)
    if N == 0
        return P0*f
    elseif N == 1
        P1 = hemisphere_op_eval(1, x, y, z)
        return (f[1] * P0 + view(f, N^2+1:(N+1)^2).' * view(P1, 2:4))[1]
    end

    DT, K, L = get_hemisphere_clenshaw_matrices(N, Z)
    xi = Vector{Array{Float64}}(3)
    xi[3] = view(f, N^2+1:(N+1)^2).'
    xi[2] = (view(f, (N-1)^2+1:N^2).'
             - xi[3] * (K[N-1] - DT[N-1] * clenshawG(N-1, x, y, z)))
    for n = N-2:-1:1
        xi[1] = (view(f, n^2+1:(n+1)^2).'
                    - xi[2] * (K[n] - DT[n] * clenshawG(n, x, y, z))
                    - xi[3] * L[n+1])
        xi[3] = copy(xi[2])
        xi[2] = copy(xi[1])
    end
    # Calculate the evaluation of f using gamma_1, gamma_2
    P1 = hemisphere_op_eval(1, x, y, z)
    return (f[1] * P0 + xi[2] * view(P1, 2:4) - P0 * xi[3] * L[1])[1]
end

function hemisphere_Q_fun_eval(f, x, y, z)
    # Check that (x, y, z) lies are on the unit hemisphere
    delta = 1e-5
    @assert (x^2 + y^2 + z^2 < 1 + delta &&  x^2 + y^2 + z^2 > 1 - delta && x >= 0 && x <= 1)
        "the point (x, y, z) must be on unit hemisphere defined by the x coord being in the range [0,1]"

    M = length(f)
    N = round(Int, sqrt(M) - 1)
    @assert (M > 0 && sqrt(M) - 1 == N) "invalid length of f"

    Q = getOPanonfunc(N, 'Q')
    out = 0.0
    for n = 0:N
        for k = 1:2n+1
            out += f[n^2+k] * Q[n+1][k](x, y, z)
        end
    end
    return out
end

# Operator for conversion from P to Q basis
function getHem2QCoeffMats(n, ZP, ZQ)
    pT, wpT, qT, wqT, pU, wpU, qU, wqU, Xp, Xq, αT, αU, δ = ZP
    pTQ, _, _, _, pUQ, _, _, _, _, _, _, _, _ = ZQ
    if n == 0
        A = Vector{Matrix{Float64}}(1)
        A[1] = zeros(1,1)
        A[1][1] = sum(pT[1][1] * Xp * pTQ[1][1] * wpT[1])
        return A
    end
    A = Vector{Matrix{Float64}}(2)
    A[1] = zeros(2n+1, 2n+1)
    A[2] = zeros(2n+1, 2n-1)
    m1 = n
    m2 = n - 1
    for i = 1:n
        j = i
        view(A[1], i, j) .= sum(pT[i][n-i+2] * Xp * pTQ[j][m1-j+2] * wpT[i])
        view(A[2], i, j) .= sum(pT[i][n-i+2] * Xp * pTQ[j][m2-j+2] * wpT[i])
    end
    i = n + 1
    j = i
    view(A[1], i, j) .= sum(pT[i][n-i+2] * Xp * pTQ[j][m1-j+2] * wpT[i])
    for i = n+2:2n
        j = i
        view(A[1], i, j) .= sum(pU[i-n-1][2n-i+2] * Xp * pUQ[j-m1-1][2m1-j+2] * wpU[i-n-1])
        j = i - 1
        view(A[2], i, j) .= sum(pU[i-n-1][2n-i+2] * Xp * pUQ[j-m2-1][2m2-j+2] * wpU[i-n-1])
    end
    i = 2n + 1
    j = i
    view(A[1], i, j) .= sum(pU[i-n-1][2n-i+2] * Xp * pUQ[j-m1-1][n+m1-j+2] * wpU[i-n-1])
    return A
end

function hem2Q(N, ZP, ZQ)
    cols = rows = 1:2:2N+1  # block sizes
    l,u = 0,1
    λ,μ = 0,1
    C = BandedBlockBandedMatrix(0.0*I, (rows,cols), (l,u), (λ,μ))
    view(C, Block(1, 1)) .= getHem2QCoeffMats(0, ZP, ZQ)[1]
    for n = 1:N
        A = getHem2QCoeffMats(n, ZP, ZQ)
        view(C, Block(n+1, n+1)) .= A[1].'
        view(C, Block(n, n+1)) .= A[2].'
    end
    return C
end

#=====#

N = 10
tol = 1e-12
x = 0.3; y = 0.8; z = sqrt(1-x^2-y^2)
P = hemisphere_op_eval(N, x, y, z)
J = hemJx(N)
norm((x*P-J*P)[1:N^2])
J = hemJy(N)
norm((y*P-J*P)[1:N^2])
J = hemJz(N)
norm((z*P-J*P)[1:N^2])

M = (N+1)^2
f = ones(M)
ZP = get1dops(N)
out = 0.0
for k = 0:N
    out += vecdot(view(f, k^2+1:(k+1)^2), view(P, k^2+1:(k+1)^2))
end
out - hemisphere_fun_eval(f, ZP, x, y, z)

ZQ = get1dops(N, 'Q')
C = hem2Q(N, ZP, ZQ)
res = zeros(M)
inds = []
for j = 1:M
    c = zeros(M)
    c[j] = 1.0
    res[j] = abs(hemisphere_Q_fun_eval(C*c, x, y, z) - hemisphere_fun_eval(c, ZP, x, y, z))
    if res[j] > tol
        append!(inds, j)
    end
end
@test length(inds) == 0


#=====================#



pT, wpT, qT, wqT, pU, wpU, qU, wqU, Xp, Xq, αT, αU, δ = ZP
pTQ, wpTQ, _, _, pUQ, wpUQ, _, _, _, _, αTQ, αUQ, _ = ZQ
n = 8; k = 1; dx = 1e-9
k * qU[k](x) - (qT[k+1](x+dx) - qT[k+1](x)) / dx
(k+2) * qV[k](x) - (qU[k+1](x+dx) - qU[k+1](x)) / dx
x/2 * qU[k+1](x) - (1-x)/2 * qU[k+1](x) - x * (1-x) * (qU[k+1](x+dx) - qU[k+1](x)) / dx
(k+1) * qT[k+2](x)
pT[k+1][n-k+1](x) - pU[k][n-k+1](x)
pU[k+1][n-k+1](x) - pV[k][n-k+1](x)



# Contstuction of the 1D hierarchy of OPs
function OPSpace(N)
    an = 5 # Number of "a" parameter vals
    bn = 20 # Number of "b" parameter vals
    cn = 5 # Number of "c" parameter vals
    H = Matrix{Vector{ApproxFun.Fun}}(an, bn)
    P = Vector{Vector{ApproxFun.Fun}}(bn)
    b = -1.0
    X = Fun(identity, 0..1)
    Y = Fun(identity, -1..1)
    ρ = sqrt(1 - X^2)
    for j = 1:bn
        if round(b/2) == b/2
            wH = (1 - Y^2)^(Int(b/2))
        else
            wH = (1 - Y^2)^(b/2)
        end
        P[j] = lanczos(wP, N+1)[1]
        a = 0
        for i = 1:an
            if round(b/2) == b/2
                wH = X^a * (1 - X^2)^(Int(b/2))
            else
                wH = X^a * (1 - X^2)^(b/2)
            end
            lanczos(wH, N+1)[1]
            H[i, j] = lanczos(wH, N+1)[1]
            a += 1
        end
        b += 1
    end
    return H, P
end

function getOPH(H, a, b)
    return H[a+1, Int(b+2)]
end

function getOPP(P, b)
    return P[Int(2b+2)]
end

N = 5
H, P = OPSpace(N)

# Coeff of dP_n^(b,b)/dy = c * P_{n-1}^(b+1,b+1)
a = 0; b = 0.5
P0 = getOPP(P, b)
P1 = getOPP(P, b+1)
n = 2
Y = Fun(identity, -1..1)
dP0 = Fun(x->((P0[n+1](x+dx) - P0[n+1](x)) / dx), Chebyshev(-1..1), 10)
c = sum(dP0 * P1[n] * (1 - Y^2)^(b+1))
dP0(x) - c * P1[n](x)

# Coeffs of dH_n^(a,b)/dx = c1 * H_{n-1}^(a+1,b+1) + c2 * H_{n-2}^(a+1,b+1)
a = 0; b = -0.5
H0 = getOPH(H, a, b)
H1 = getOPH(H, a+1, b+1)
n = 4
X = Fun(identity, 0..1)
dH0 = Fun(x->((H0[n+1](x+dx) - H0[n+1](x)) / dx), Chebyshev(0..1), 10)
c1 = sum(dH0 * H1[n] * (1 - X^2)^(b+1) * X^(a+1))
c2 = sum(dH0 * H1[n-1] * (1 - X^2)^(b+1) * X^(a+1))
dH0(x) - (c1 * H1[n](x) + c2 * H1[n-1](x))



N = 5
H, P = OPSpace(N)

X = Fun(identity, -1..1)
b = 1.5; n = 3
dx = 1e-8
P0 = getOPP(P, b)
dP = Fun(x->((P0[n+1](x+dx) - P0[n+1](x)) / dx), Chebyshev(-1..1), 20)
β = b + 1
w = (1 - X^2)^β
for m = 0:N+1
    P1 = getOPP(P, β)
    println("beta=", β, " m=", m, " int=", sum(dP * w * P1[m+1]))
end

m = n - 1
# c = 0.5 * (n + 2b + 1)
P1 = getOPP(P, β)
c = sum(dP * (1-X^2)^β * P1[m+1])
x = rand(1)[1]
c * P1[m+1](x) - dP(x)

a = 1; b = 1.5; n = 3
X = Fun(identity, 0..1)
dx = 1e-8
H0 = getOPH(H, a, b)
dH = Fun(x->((H0[n+1](x+dx) - H0[n+1](x)) / dx), Chebyshev(0..1), 10)
α = a + 1
β = b + 1
w = X^α * (1 - X^2)^β
for m = 0:N+1
    # f = m * X^(m-1+α-a) * (1-X^2)^(β-b) + α * X^(m+α-a-1) * (1-X^2)^(β-b) - 2 * β * X^(m+α-a+1) * (1-X^2)^(β-b-1)
    # f = m * X^(m-1+α-a) * (1-X^2)^(β-b+0.5) + α * X^(m+α-a-1) * (1-X^2)^(β-b+0.5) - 2 * (β-b+0.5) * X^(m+α-a+1) * (1-X^2)^(β-b-0.5)
    H1 = getOPH(H, α, β)
    println("alpha=", α, " beta=", β, " m=", m, " int=", sum(dH * w * H1[m+1]))
end

H1 = getOPH(H, α, β)
c1 = sum(dH * w * H1[n])
c2 = sum(dH * w * H1[n-1])
x = rand(1)[1]
c1 * H1[n](x) + c2 * H1[n-1](x) - dH(x)


N = 5
H, P = OPSpace(N)
X = Fun(identity, 0..1)
ρ = sqrt(1 - X^2)
a = 2; b = 2; n = 4
H0 = getOPH(H, a, b)
α = a
β = b - 2
if round(b/2) == b/2
    w = X^a * (1 - X^2)^(Int(b/2))
else
    w = X^a * (1 - X^2)^(b/2)
end
H1 = getOPH(H, α, β)
x = rand(1)[1]
out = 0.0
cvec = []
for m = 0:n+2
    c = sum(H0[n+1] * w * H1[m+1])
    append!(cvec, c)
    out += c * H1[m+1](x)
end
out - H0[n+1](x) * (1-x^2)
cvec


N = 5
Y = Fun(identity, -1..1)
a = 2; b = 1; n = 3
P0,_,_ = lanczos((1-Y^2)^b, N+1)
P1,_,_ = lanczos((1-Y^2)^(b+1), N+1)
Q,_,_ = lanczos((1-Y)^(b+1) * (1+Y)^(b-1), N+1)
dy = 1e-9; dP = Fun(y->((P0[n+1](y + dy) - P0[n+1](y))/dy), Chebyshev(-1..1), 50)
# L5starPn = - b * P0[n+1] + (1 - Y) * 0.5 * (n + 2b + 1) * P1[n]
L6P = b * P0[n+1] + (1 + Y) * 0.5 * (n + 2b + 1) * P1[n]
w = (1-Y^2)^(b+1)
out = 0.0
cvec = []
y = rand(1)[1]
if rand(1)[1] > 0.5
    y *= -1
end
for m = 0:n-1
    c = sum(w * dP * P1[m+1])
    append!(cvec, c)
    out += c * P1[m+1](y)
end
out - dP(y)
cvec[end] * P1[n](y) - (P0[n+1](y + dy) - P0[n+1](y))/dy
cvec
0.5 * (n + 2b + 1) - cvec[end]
0.5 * (n + 2b + 1) * P1[n](y)
(P0[n+1](y + dy) - P0[n+1](y))/dy
