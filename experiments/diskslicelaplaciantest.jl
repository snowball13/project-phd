# Run this setup code to get the recurrence coefficients for the 1D OPs

using Revise
using ApproxFun
using Test, Profile, StaticArrays, LinearAlgebra, BlockArrays,
        BlockBandedMatrices, SparseArrays
using OrthogonalPolynomialFamilies
import OrthogonalPolynomialFamilies: points, pointswithweights, getopptseval,
                    opevalatpts, inner2, getopindex, getnk, resizecoeffs!,
                    transformparamsoperator, weightedpartialoperatorx,
                    weightedpartialoperatory, partialoperatorx, partialoperatory,
                    laplaceoperator, derivopevalatpts, getderivopptseval, getopnorms,
                    getopnorm, operatorclenshaw, weight, biharmonicoperator,
                    getPspace, getRspace, differentiatespacex, differentiatespacey,
                    differentiateweightedspacex, differentiateweightedspacey,
                    resizedata!, resizedataonedimops!
using JLD

f = Fun()
@which Fun()
x = 0.9
S = JacobiWeight(-0.5,-0.5)
S.space
p = points(JacobiWeight(-0.5,-0.5), 20)
@which chebyshevpoints(float(real(eltype(Float64))), 20, kind=1)
x(0.9)

B = BigFloat; T = Float64
n = 100
X = Fun(B(-1)..1); J = OrthogonalPolynomialFamily(T, 1-X^2)
s, ws = pointswithweights(B, J(B(-0.5)), n) # Quad for W = 1/sqrt(1-s^2)
X = Fun(-1..1); J = OrthogonalPolynomialFamily(T, 1-X^2)
s, ws = pointswithweights(J(-0.5), n) # Quad for W = 1/sqrt(1-s^2)
reverse!(s)
p - s
ws[1] - π / n


D = DiskSliceFamily(); a, b = 1.0, 1.0; ap, bp = 1.0, 2.0
S1 = D(a, b); S2 = D(ap, bp)
k = 2
R1 = getRspace(S1, k)
R2 = getRspace(S2, k)
pts, w = pointswithweights(D.R(S2.params[1], S2.params[2] + k - 0.5), 50)
getopptseval(R1, 20, pts); getopptseval(R2, 20, pts)
n = 5; m = 20
Float64(inner2(R1, opevalatpts(R1, n-k+1, pts), opevalatpts(R2, m-k+1, pts), w))
ret

function getreccoeffsP!(S::DiskSliceSpace{<:Any, B, T, <:Any}, N) where {B,T}
    @show "getreccoeffsP!"
    P = getPspace(S)
    resize!(P.a, N+1); resize!(P.b, N+1)
    c = Int(P.params[1])
    if c == 0
        P.a[:] = load("experiments/saved/diskslice-alpha=0.2-beta=0.8-P00-rec-coeffs-a-N=1000-BF.jld", "a")[1:N+1]
        P.b[:] = load("experiments/saved/diskslice-alpha=0.2-beta=0.8-P00-rec-coeffs-b-N=1000-BF.jld", "b")[1:N+1]
    else
        P.a[:] = load("experiments/saved/diskslice-alpha=0.2-beta=0.8-P11-rec-coeffs-a-N=1000-BF.jld", "a")[1:N+1]
        P.b[:] = load("experiments/saved/diskslice-alpha=0.2-beta=0.8-P11-rec-coeffs-b-N=1000-BF.jld", "b")[1:N+1]
    end
    resize!(P.ops, 2); resize!(P.weight, 1)
    P.weight[1] = prod(P.family.factors.^(P.params))
    X = Fun(identity, domain(P.weight[1]))
    P.ops[1] = Fun(1/sqrt(sum(P.weight[1])),space(X))
    v = X * P.ops[1] - P.a[1] * P.ops[1]
    P.ops[2] = v / P.b[1]
    for k = 2:length(P.a)
        v = X * P.ops[2] - P.b[k-1] * P.ops[1] - P.a[k] * P.ops[2]
        P.ops[1] = P.ops[2]
        P.ops[2] = v / P.b[k]
    end
    S
end

function solutionblocknorms(D::DiskSliceFamily, f, A, N)
    ucfs = sparse(A)[1:getopindex(N+D.nparams-1,N+D.nparams-1), 1:getopindex(N,N)] \ resizecoeffs!(f, N+D.nparams-1)
    ucfspsa = PseudoBlockArray(ucfs, [i+1 for i=0:N])
    unorms = [norm(view(ucfspsa, Block(i))) for i = 1:N+1]
    unorms
end
function plotnorms(unorms, label; append=false, line=:solid)
    if append
        Plots.plot!(unorms, line=(3, line), label=label, xscale=:log10, yscale=:log10, legend=:bottomleft)
    else
        Plots.plot(unorms, line=(3, :solid), label=label, xscale=:log10, yscale=:log10, legend=:bottomleft)
        Plots.xlabel!("Block")
        Plots.ylabel!("Norm")
    end
end

#=======================#

# Half Disk / End Disk Slice

α, β = 0.2, 0.8
a, b = 1.0, 1.0
HD = DiskSliceFamily(Float64, typeof(α), 0.0, 1.0, -1.0, 1.0)
S = HD(a, b); S0 = HD(a - 1, b - 1)
isindomain(x, D::DiskSliceFamily) = D.α ≤ x[1] ≤ D.β && D.γ*D.ρ(x[1]) ≤ x[2] ≤ D.δ*D.ρ(x[1])
x, y = BigFloat(1)/3, -BigFloat(1)/10; z = [x;y]; isindomain(z, HD)

N = 530
getreccoeffsP!(S, N)
getreccoeffsP!(S0, N)
resizedataonedimops!(S, N)
resizedataonedimops!(S0, N)

N = 500
LL = laplaceoperator(S, S, N; square=false, weighted=true)
LLold = load("experiments/saved/laplacian-w11-array.jld", "Lw11")

maximum(abs, sparse(LL)[1:getopindex(N,N), 1:getopindex(N,N)] - sparse(LLold))

f1 = Fun((x,y)->1.0, S); f1.coefficients
f2 = Fun((x,y)->(1 - HD.α^2 - y^2), S); f2.coefficients
f3 = Fun((x,y)->weight(S, x, y)^2, S); f3.coefficients
f4 = Fun((x,y)->exp(-1000((x-0.5)^2+(y-0.5)^2)), S, 20000); f4.coefficients
N = 198
unorms1 = solutionblocknorms(HD, f1, LL, N)
unorms2 = solutionblocknorms(HD, f2, LL, N)
unorms3 = solutionblocknorms(HD, f3, LL, N)
unorms4 = solutionblocknorms(HD, f4, LL, N)
using Plots
plotnorms(unorms1, "f(x,y) = 1"; append=false)
plotnorms(unorms2, "f(x,y) = (1-alpha^2-y^2)"; append=true, line=:dash)
plotnorms(unorms3, "f(x,y) = W{(1,1,1)}^2"; append=true, line=:dashdot)
plotnorms(unorms4, "f(x,y) = exp(-1000((x-0.5)^2+(y-0.5)^2))"; append=true, line=:dot)
Plots.savefig("experiments/images/solutionblocknorms-poisson-halfdisk-N=$N-f64.pdf")

#=======================#

# Disk Slice
a, b, c = 1.0, 1.0, 1.0
α, β = 0.2, 0.8
DD = DiskSliceFamily(Float64, typeof(α), α, β, -1.0, 1.0)
S = DD(a, b, c); S0 = DD(a - 1, b - 1, c - 1)
D = DiskSliceFamily(α, β)
S = D(a, b, c); S0 = D(a - 1, b - 1, c - 1)
isindomain(x, D::DiskSliceFamily) = D.α ≤ x[1] ≤ D.β && D.γ*D.ρ(x[1]) ≤ x[2] ≤ D.δ*D.ρ(x[1])
x, y = BigFloat(1)/3, -BigFloat(1)/10; z = [x;y]; isindomain(z, D)

N = 999
getreccoeffsP!(S, N)
resizedataonedimops!(S, N)
getreccoeffsP!(S0, N)
resizedataonedimops!(S0, N)

N = 990
LLds = laplaceoperator(S, S, N; square=false, weighted=true)
save("experiments/saved/diskslice-alpha=0.2-beta=0.8-laplacian-111-N=$N-f64.jld", "Lw111", LLds)
# N = 990
# Δw111 = load("experiments/saved/diskslice-alpha=0.2-beta=0.8-laplacian-111-N=$N.jld", "Lw111")


f1 = Fun((x,y)->1.0, S); f1.coefficients
f2 = Fun((x,y)->(1 - DD.α^2 - y^2)*(1 - DD.β^2 - y^2), S); f2.coefficients
f3 = Fun((x,y)->weight(S, x, y)^3, S); f3.coefficients
f4 = Fun((x,y)->exp(-1000((x-0.5)^2+(y-0.5)^2)), S, 20000); f4.coefficients

N = 950
unorms1 = solutionblocknorms(f1, LLds, N)
unorms2 = solutionblocknorms(f2, LLds, N)
unorms3 = solutionblocknorms(f3, LLds, N)
unorms4 = solutionblocknorms(f4, LLds, N)
using Plots
plotnorms(unorms1, "f(x,y) = 1"; append=false)
plotnorms(unorms2, "f(x,y) = (1-alpha^2-y^2)(1-beta^2-y^2)"; append=true, line=:dash)
plotnorms(unorms3, "f(x,y) = W{(1,1,1)}^3"; append=true, line=:dashdot)
plotnorms(unorms4, "f(x,y) = exp(-1000((x-0.5)^2+(y-0.5)^2))"; append=true, line=:dot)
Plots.savefig("experiments/images/solutionblocknorms-poisson-diskslice-alpha=$α-beta=$β-N=$N-f64.pdf")

D.R

T = Float64
v = Vector{SArray{Tuple{3},T,1,3}}()
resize!(v, 3)


#=======================#


"""
Solve `A x = b` for `x` using iterative improvement
"""
function iterimprove(A::SparseMatrixCSC{T}, b::Vector{T};
                        iters=5, verbose=true) where T
    eps(T) < eps(Float64) || throw(ArgumentError("wrong implementation"))
    A0 = SparseMatrixCSC{Float64}(A)
    F = factorize(A0)
    x = zeros(T, length(b))
    r = copy(b)
    for iter = 1:iters
        y = F \ Vector{Float64}(r)
        for i in eachindex(x)
            x[i] += y[i]
        end
        r = b - A * x
        if verbose
            @show "at iter %d resnorm = %.3g\n" iter norm(r)
        end
    end
    x
end

rhoval(x) = sqrt(1-x^2)
dxrhoval(x) = -x/sqrt(1-x^2)
d2xrhoval(x) = - (1-x^2)^(-0.5) - x^2 * (1-x^2)^(-1.5)
function dxweight(S::DiskSliceSpace{<:Any, B, T, <:Any}, x, y) where {B,T}
    D = S.family
    if D.nparams == 2
        a, b = T.(S.params)
        ret = a * weight(D(a-1, b), x, y)
        ret += 2 * rhoval(x) * dxrhoval(x) * b * weight(D(a, b-1), x, y)
        ret
    else
        a, b, c = T.(S.params)
        ret = -a * weight(D(a-1, b, c), x, y)
        ret += b * weight(D(a, b-1, c), x, y)
        ret += 2 * rhoval(x) * dxrhoval(x) * c * weight(D(a, b, c-1), x, y)
        B(ret)
    end
end
dxweight(S::DiskSliceSpace, z) = dxweight(S, z[1], z[2])
function dyweight(S::DiskSliceSpace{<:Any, B, T, <:Any}, x, y) where {B,T}
    ret = - 2 * S.params[end] * y * weight(differentiateweightedspacey(S), x, y)
    B(ret)
end
dyweight(S::DiskSliceSpace, z) = dyweight(S, z[1], z[2])
function d2xweight(S::DiskSliceSpace{<:Any, B, T, <:Any}, x, y) where {B,T}
    D = S.family
    if D.nparams == 2
        a, b = T.(S.params)
        ret1 = a * ((a - 1) * weight(D(a-2, b), x, y)
                                + 2 * rhoval(x) * dxrhoval(x) * b * weight(D(a-1, b-1), x, y))
        ret2 = (2 * rhoval(x) * dxrhoval(x) * b * (a * weight(D(a-1, b-1), x, y)
                                    + 2 * rhoval(x) * dxrhoval(x) * (b-1) * weight(D(a, b-2), x, y))
                    + 2 * b * (rhoval(x) * d2xrhoval(x) + dxrhoval(x)^2) * weight(D(a, b-1), x, y))
        ret1 + ret2
    else
        a, b, c = T.(S.params)
        ret1 = a * ((a-1)*weight(D(a-2,b,c),x,y) - b*weight(D(a-1,b-1,c),x,y) - 2*c*rhoval(x)*dxrhoval(x)*weight(D(a-1,b,c-1),x,y))
        ret2 = b * (-a*weight(D(a-1,b-1,c),x,y) + (b-1)*weight(D(a,b-2,c),x,y) + 2*c*rhoval(x)*dxrhoval(x)*weight(D(a,b-1,c-1),x,y))
        ret3 = 2*c*rhoval(x)*dxrhoval(x) * (-a*weight(D(a-1,b,c-1),x,y) + b*weight(D(a,b-1,c-1),x,y)
                                                    + 2*(c-1)*rhoval(x)*dxrhoval(x)*weight(D(a,b,c-2),x,y))
        ret4 = 2*c*(dxrhoval(x)^2 + rhoval(x)*d2xrhoval(x))*weight(D(a,b,c-1),x,y)
        B(ret1+ret2+ret3+ret4)
    end
end
d2xweight(S::DiskSliceSpace, z) = d2xweight(S, z[1], z[2])
function d2yweight(S::DiskSliceSpace{<:Any, B, T, <:Any}, x, y) where {B,T}
    D = S.family
    ret = - 2 * S.params[end] * weight(differentiateweightedspacey(S), x, y)
    ret += 4 * S.params[end] * (S.params[end] - 1) * y^2 * weight(differentiateweightedspacey(differentiateweightedspacey(S)), x, y)
    B(ret)
end
d2yweight(S::DiskSliceSpace, z) = d2yweight(S, z[1], z[2])



# # R rec coefss initialisation
# R00, R11 = getRspace(S0, Int(-S0.params[end])), getRspace(S, Int(-S.params[end]))
# r00a, r00b, r11a, r11b = R00.a, R00.b, R11.a, R11.b
# r00ops, r11ops = R00.ops, R11.ops

R00 = getRspace(S0, Int(-S0.params[end]))
resize!(R00.a, length(r00a)); R00.a[:] = r00a[:]
resize!(R00.b, length(r00b)); R00.b[:] = r00b[:]
resize!(R00.ops, length(r00ops)); R00.ops[:] = r00ops[:]
R00 = getRspace(S, Int(-S.params[end]))
resize!(R00.a, length(r11a)); R00.a[:] = r11a[:]
resize!(R00.b, length(r11b)); R00.b[:] = r11b[:]
resize!(R00.ops, length(r11ops)); R00.ops[:] = r11ops[:]

# resize!(R00.ops, 2); resize!(R00.weight, 1)
# R00.weight[1] = prod(R00.family.factors.^(R00.params))
# Z = Fun(identity, domain(R00.weight[1]))
# R00.ops[1] = Fun(1/sqrt(sum(R00.weight[1])),space(Z))
# v = Z * R00.ops[1] - R00.a[1] * R00.ops[1]
# R00.ops[2] = v / R00.b[1]
# for k = 2:length(R00.a)
#     v = Z * R00.ops[2] - R00.b[k-1] * R00.ops[1] - R00.a[k] * R00.ops[2]
#     R00.ops[1] = R00.ops[2]
#     R00.ops[2] = v / R00.b[k]
# end



# # Operators
# N = 200
# HD.nparams
# A = partialoperatorx(differentiateweightedspacex(S), N+HD.nparams)
# save("experiments/saved/diskslice-alpha=0.2-beta=1.0-operators-N=$N-dx-bf.jld", "dx", A)
# B = weightedpartialoperatorx(S, N)
# save("experiments/saved/diskslice-alpha=0.2-beta=1.0-operators-N=$N-wdx-bf.jld", "wdx", B)
# C = transformparamsoperator(differentiatespacey(HD(S.params .- 1)), S, N+HD.nparams-1)
# save("experiments/saved/diskslice-alpha=0.2-beta=1.0-operators-N=$N-t-bf.jld", "t", C)
# E = partialoperatory(HD(S.params .- 1), N+HD.nparams)
# save("experiments/saved/diskslice-alpha=0.2-beta=1.0-operators-N=$N-dy-bf.jld", "dy", E)
# F = transformparamsoperator(differentiateweightedspacey(S), HD(S.params .- 1), N+1, weighted=true)
# save("experiments/saved/diskslice-alpha=0.2-beta=1.0-operators-N=$N-wt-bf.jld", "wt", F)
# G = weightedpartialoperatory(S, N)
# save("experiments/saved/diskslice-alpha=0.2-beta=1.0-operators-N=$N-wdy-bf.jld", "wdy", G)
# AAl, AAu = A.l + B.l, A.u + B.u
# BBl, BBu = C.l + E.l + F.l + G.l, C.u + E.u + F.u + G.u
# AAλ, AAμ = A.λ + B.λ, A.μ + B.μ
# BBλ, BBμ = C.λ + E.λ + F.λ + G.λ, C.μ + E.μ + F.μ + G.μ
# AA = sparse(A) * sparse(B)
# BB = sparse(C) * sparse(E) * sparse(F) * sparse(G)
# L = BandedBlockBandedMatrix(AA + BB, (1:nblocks(A)[1], 1:nblocks(B)[2]),
#                             (max(AAl,BBl),max(AAu,BBu)), (max(AAλ,BBλ),max(AAμ,BBμ)))
# save("experiments/saved/diskslice-alpha=0.2-beta=1.0-laplacian-11-nonsquare-N=$N-bf.jld", "Lw111", L)
