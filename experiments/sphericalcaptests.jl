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
                    resizedata!,
                    resizedataonedimops!, getnki, convertcoeffsvecorder,
                    diffoperatorphi, diffoperatortheta, diffoperatortheta2,
                    differentiatespacephi, increasedegreeoperator
using JLD

# Useful functions for testing
function converttopseudo(S::SphericalCapSpace, cfs; converttobydegree=true)
    N = getnki(S, length(cfs))[1]
    if converttobydegree
        PseudoBlockArray(convertcoeffsvecorder(S, cfs), [2n+1 for n=0:N])
    else
        PseudoBlockArray(cfs, [2n+1 for n=0:N])
    end
end
isindomain(pt, D::SphericalCapFamily) = D.α ≤ pt[3] ≤ D.β && norm(pt) == 1.0
isindomain(pt, S::SphericalCapSpace) = isindomain(pt, S.family)
"""
Solve `A x = b` for `x` using iterative improvement
(for BigFloat sparse matrix and vector)
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

# Setup
T = Float64; B = T# BigFloat
α = 0.2
DSF = DiskSliceFamily(α)# DiskSliceFamily(T, T, α, 1.0, -1.0, 1.0); a, b = 1.0, 1.0; ap, bp = 1.0, 2.0
SCF = SphericalCapFamily(B, T, α)
a = 1.0
S = SCF(a, 0.0); S2 = DSF(a, 0.0)

y, z = B(-0.234), B(0.643); x = sqrt(1 - z^2 - y^2); p = [x; y; z]; isindomain(p, SCF)
θ = atan(y / x)
resizedata!(S, 10)

#===#

# Test transform
n = 200
f = (x,y,z)->cos(y)
pts, w = pointswithweights(S, n)
vals = [f(pt...) for pt in pts]
cfs = transform(S, vals)
N = getnki(S, length(cfs))[1]
cfs2 = PseudoBlockArray(convertcoeffsvecorder(S, cfs), [2n+1 for n=0:N])
F = Fun(S, cfs)
F(p)
f(p...)
T(F(p) - f(p...))
@test T(F(p)) ≈ T(f(p...))
vals2 = itransform(S, cfs); pts2 = points(S, length(vals2)/2); @test T.(vals2) ≈ T.([f(pt...) for pt in pts2])
F = Fun(f, S, 200); F.coefficients
@test T(F(p)) ≈ T(f(p...))

# ∂/∂θ operator
f = (x,y,z)->x^2 + y^4 # (cos2t + sin4t*ρ2(z))ρ2(z)
dθf = (x,y,z)->-2x * y + 4y^3 * x
F = Fun(f, S, 100); F.coefficients
N = getnki(S, ncoefficients(F))[1]
dθ = diffoperatortheta(S, N)
dθF = Fun(S, dθ * F.coefficients)
@test dθF(p) ≈ dθf(p...)

# ∂²/∂θ² operator
f = (x,y,z)->x^2 + y^4 # (cos2t + sin4t*ρ2(z))ρ2(z)
d2θf = (x,y,z)->2y^2 - 2x^2 + 12x^2*y^2 - 4y^4
F = Fun(f, S, 100); F.coefficients
N = getnki(S, ncoefficients(F))[1]
dθ2 = diffoperatortheta2(S, N; weighted=false)
d2θF = Fun(S, dθ2 * F.coefficients)
@test d2θF(p) ≈ d2θf(p...)

# ρ(z)∂/∂ϕ operator
# f = (x,y,z)->x^a y^b z^c
# df = (x,y,z)->x^a y^b z^(c-1) [(a+b)z^2 - cρ(z)^2]
inds = [4, 3, 5]; sum(inds)
f = (x,y,z)->x^inds[1] * y^inds[2] * z^inds[3]
df = (x,y,z)->((inds[1] + inds[2]) * z^2 - inds[3] * S.family.ρ(z)^2) * x^inds[1] * y^inds[2] * z^(inds[3]-1) # deg = degf + 1
F = Fun(f, S, 2*(sum(inds)+1)^2); F.coefficients
N = getnki(S, ncoefficients(F))[1]
dϕ = diffoperatorphi(S, N; weighted=false)
cfs = dϕ * F.coefficients
dF = Fun(differentiatespacephi(S), cfs)
@test dF(p) ≈ df(p...)

# ρ(z)∂/∂ϕ weighted operator
# f = (x,y,z)->x^a y^b z^c
# df = (x,y,z)->x^a y^b z^(c-1) [(a+b)z^2 - cρ(z)^2]
inds = [4, 3, 5]; sum(inds)
f = (x,y,z)->x^inds[1] * y^inds[2] * z^inds[3]
df = (x,y,z)->((z - S.family.α) * ((inds[1] + inds[2]) * z^2 - inds[3] * S.family.ρ(z)^2) * x^inds[1] * y^inds[2] * z^(inds[3]-1)
                - S.params[1] * S.family.ρ(z)^2 * f(x,y,z)) # deg = degf + 2
F = Fun(f, S, 2*(sum(inds)+1)^2); F.coefficients
N = getnki(S, ncoefficients(F))[1]
dϕ = diffoperatorphi(S, N; weighted=true)
cfs = dϕ * F.coefficients
dF = Fun(differentiatespacephi(S; weighted=true), cfs)
@test dF(p) ≈ df(p...)

# non-weighted transform params operator
S0 = differentiatespacephi(S; weighted=true)
inds = [3, 4, 5]; sum(inds)
f = (x,y,z)->x^inds[1] * y^inds[2] * z^inds[3]
F0 = Fun(f, S0, 2*(sum(inds)+1)^2); F0.coefficients
N = getnki(S0, ncoefficients(F0))[1]
t = transformparamsoperator(S0, S, N)
F = Fun(S, t * F0.coefficients); F.coefficients
@test F(p) ≈ F0(p)

# weighted transform params operator
S0 = differentiatespacephi(S; weighted=true)
inds = [7, 2, 3]; sum(inds)
f = (x,y,z)->x^inds[1] * y^inds[2] * z^inds[3]
F = Fun(f, S, 2*(sum(inds)+1)^2); F.coefficients
N = getnki(S, ncoefficients(F))[1]
t = transformparamsoperator(S, S0, N; weighted=true)
F0 = Fun(S0, t * F.coefficients); F0.coefficients
@test F(p) * weight(S, p) ≈ F0(p) * weight(S0, p)

# increase degree
inds = [3, 1, 5]; sum(inds)
f = (x,y,z)->cos(x+y)# x^inds[1] * y^inds[2] * z^inds[3]
F = Fun(f, S, 2*(sum(inds)+1)^2); F.coefficients
N = getnki(S, ncoefficients(F))[1]
t = increasedegreeoperator(S, N, N+2)
Fi = Fun(S, t * F.coefficients); Fi.coefficients
@test F(p) == Fi(p)
F(p)
Fi(p)

# Laplacian
N = 2
A, B, C, E, F, G = laplaceoperator(S, S, N; square=true)
U = (x,y,z)->1.0
rho2f = (x,y,z)->-2 * z * (1-z^2) # 2 * ρ * ρ' * ρ^2
F = Fun(rho2f, S, 2*(N+1)^2); F.coefficients
ucfs = sparse(Δ)[1:(N+1)^2, 1:(N+1)^2] \ F.coefficients

A
B
C
E
F
G

Array(A * B)
Array(C * E * F * G)



#==============#
# Saving to disk

# Clenshaw mats
this
sb = load("experiments/saved/sphericalcap/sphericalcap-alpha=0.2-clenshawmats-B-BF.jld", "B"); resize!(S.B, length(sb)); S.B[:] = sb[:]
sc = load("experiments/saved/sphericalcap/sphericalcap-alpha=0.2-clenshawmats-C-BF.jld", "C", S.C); resize!(S.C, length(sc)); S.C[:] = sc[:]
sdt = load("experiments/saved/sphericalcap/sphericalcap-alpha=0.2-clenshawmats-DT-BF.jld", "DT", S.DT); resize!(S.DT, length(sdt)); S.DT[:] = sdt[:]
