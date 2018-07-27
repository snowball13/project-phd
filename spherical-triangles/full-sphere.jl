using ApproxFun
    using SphericalHarmonics

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

function getTAnonFunc(n, k, pn, qn, ρ)
    return (x,y)->pn[k+1][n-k+1](x)*ρ(x)^k*qn[k+1](y/ρ(x))
end
function getUAnonFunc(n, k, pn, qn, ρ)
    return (x,y)->pn[k+1][n-k+1](x)*ρ(x)^k*qn[k+1](y/ρ(x))
end

function getOPAnonFuncT(n, k, Tn)
    return (x,y,z)->Tn[n+1][k+1](x,y)
end
function getOPAnonFuncU(n, k, Un)
    return (x,y,z)->z*Un[n][k+1](x,y)
end

function getOPAnonFunc(N)
    # T_n (μ = 0)
    μ = 0
    X = Fun(identity, -1..1)
    # w1 = (1-X)^μ/sqrt(1-X^2)
    w2 = 1/sqrt(1-X^2)
    ρ = sqrt(1-X^2)
    qnT,_,_ = lanczos(w2,N+1)
    pnT = Vector{Vector{ApproxFun.Fun}}(N+1)
    for k = 0:N
        pnT[k+1],_,_ = lanczos((1-X^2)^(k+μ), N+1)
    end
    Tn = Vector{Vector{Function}}(N+1)
    for n = 0:N
        Tn[n+1] = Vector{Function}(n+1)
        for k = 0:n
            Tn[n+1][k+1] = getTAnonFunc(n, k, pnT, qnT, ρ)
        end
    end

    # U_n (μ = 1)
    μ = 1
    X = Fun(identity, -1..1)
    # w1 = (1-X)^μ/sqrt(1-X^2)
    w2 = sqrt(1-X^2)
    ρ = sqrt(1-X^2)
    qnU,_,_ = lanczos(w2,N+1)
    pnU = Vector{Vector{ApproxFun.Fun}}(N+1)
    for k = 0:N
        pnU[k+1],_,_ = lanczos((1-X^2)^(k+μ), N+1)
    end
    Un = Vector{Vector{Function}}(N+1)
    for n = 0:N
        Un[n+1] = Vector{Function}(n+1)
        for k = 0:n
        Un[n+1][k+1] = getUAnonFunc(n, k, pnU, qnU, ρ)
        end
    end

    # Put together : Pn = [Tn(x,y); z*Un(x,y)]
    P = Vector{Vector{Function}}(N+1)
    for n = 0:N
        P[n+1] = Vector{Function}(2n+1)
        j = 1
        for k = 0:n
            # Rn[n+1][j] = (x,y,z)->Pn[n+1][k+1](x,y)
            P[n+1][j] = getOPAnonFuncT(n, k, Tn)
            j += 1
        end
        for k = 0:n-1
            # Rn[n+1][j] = (x,y,z)->z*Qn[n][k+1](x,y)
            P[n+1][j] = getOPAnonFuncU(n, k, Un)
            j += 1
        end
    end
    return P
end

function opEval(N, x, y, z)
    P = getOPAnonFunc(N)
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


# Does the span of these match the span of the SHs? We expect the V matrix cols
# from the SVD for each Y and P matrices to be equal up to sign.
n = 2
m = 10
Y = zeros(2n+1, m); P = zeros(2n+1, m)
x = 0.8rand(m); y = 0.5rand(m); z = zeros(m)
for i=1:m
    z[i] = sqrt(1-x[i]^2-y[i]^2)
    Y[:, i] = sh_eval_real(n, x[i], y[i], z[i])
    P[:, i] = opEval(n, x[i], y[i], z[i])[n^2+1:(n+1)^2]
end
UY, SY, VY = svd(Y)
UP, SP, VP = svd(P)
VY - VP
VY + VP
