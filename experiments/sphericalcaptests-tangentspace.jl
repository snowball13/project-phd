using Revise
using ApproxFun
using Test, Profile, StaticArrays, LinearAlgebra, BlockArrays,
        BlockBandedMatrices, SparseArrays
using OrthogonalPolynomialFamilies
import OrthogonalPolynomialFamilies: points, pointswithweights, getopptseval,
                    opevalatpts, inner2, getopindex, getnk, resizecoeffs!,
                    transformparamsoperator, weightedpartialoperatorx,
                    weightedpartialoperatory, partialoperatorx, partialoperatory,
                    derivopevalatpts, getderivopptseval, getopnorms,
                    getopnorm, operatorclenshaw, weight, biharmonicoperator,
                    getPspace, getRspace, differentiatespacex, differentiatespacey,
                    differentiateweightedspacex, differentiateweightedspacey,
                    resizedata!,
                    resizedataonedimops!, getnki, convertcoeffsvecorder,
                    diffoperatorphi, diffoperatortheta, diffoperatortheta2,
                    differentiatespacephi, increasedegreeoperator,
                    getptsevalforop, getderivptsevalforop, laplacianoperator,
                    rho2laplacianoperator, gettangentspace,
                    rhogradoperator, coriolisoperator, rho2operator,
                    rhodivminusgradrhodotoperator, gradrhooperator, weightoperator
using JLD


isindomain(pt, D::SphericalCapFamily) = D.α ≤ pt[3] ≤ D.β && norm(pt) == 1.0
isindomain(pt, S::SphericalCapSpace) = isindomain(pt, S.family)
rhoval(z) = sqrt(1 - z^2)


# Setup
T = Float64; B = T#BigFloat
α = 0.2
DSF = DiskSliceFamily(B, T, α, 1.0, -1.0, 1.0)
SCF = SphericalCapFamily(B, T, B(α * 1000) / 1000)
a = 1.0
S = SCF(a, 0.0); S2 = DSF(a, a); S0 = SCF(0.0, 0.0)
ST = gettangentspace(SCF)

y, z = B(-234)/1000, B(643)/1000; x = sqrt(1 - z^2 - y^2); p = [x; y; z]; isindomain(p, SCF)
θ = atan(y / x)
resizedata!(S, 20)
resizedata!(S0, 20)

function gettangentspacecoeffsvec(S::SphericalCapSpace, fc::AbstractVector{T},
                                    gc::AbstractVector{T}) where T
    m = length(fc)
    @assert m == length(gc) "Coeffs must be same length"
    ret = zeros(T, 2m)
    it = 1
    for j = 1:m
        ret[it] = fc[j]; ret[it+1] = gc[j]
        it += 2
    end
    ret
end
gettangentspacecoeffsvec(f::Fun, g::Fun) =
    gettangentspacecoeffsvec(f.space, f.coefficients, g.coefficients)
function getscspacecoeffsvecs(ST::SphericalCapTangentSpace, Fc::AbstractVector{T}) where T
    m = Int(length(Fc) / 2)
    u, v = zeros(T, m), zeros(T, m)
    it = 1
    for j = 1:m
        u[j] = Fc[it]; v[j] = Fc[it+1]
        it += 2
    end
    u, v
end
getscspacecoeffsvecs(F::Fun) = getscspacecoeffsvecs(F.space, F.coefficients)

# Useful functions for testing
function converttopseudo(S::SphericalCapSpace, cfs; converttobydegree=true)
    N = getnki(S, length(cfs))[1]
    @assert (length(cfs) == (N+1)^2) "Invalid coeffs length"
    if converttobydegree
        PseudoBlockArray(convertcoeffsvecorder(S, cfs), [2n+1 for n=0:N])
    else
        PseudoBlockArray(cfs, [2n+1 for n=0:N])
    end
end

# Solve `A x = b` for `x` using iterative improvement
# (for BigFloat sparse matrix and vector)
function iterimprove(A::SparseMatrixCSC{T}, b::Vector{T};
                        iters=5, verbose=true) where T
    if eps(T) ≥ eps(Float64)
        # throw(ArgumentError("wrong implementation"))
        return A \ b
    end
    A0 = SparseMatrixCSC{Float64}(A)
    F = factorize(A0)
    x = zeros(T, size(A)[2]) # x = zeros(T, length(b))
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


#=======#


# Test of tangent clenshaw
basisvecs = [cos(θ) * z; sin(θ) * z; - rhoval(z)], [-sin(θ); cos(θ); 0]
N = 5
for n=0:N, k=0:n, i=0:min(1,k), j=0:1
cfs = zeros(B, getopindex(ST, N, N, 1, 1)); cfs[getopindex(ST, n, k, i, j; bydegree=false, N=N)] = 1; cfs
ret = OrthogonalPolynomialFamilies.clenshaw(cfs, ST, p)
cfs0 = zeros(B, getopindex(S, N, N, 1)); cfs0[getopindex(S, n, k, i; bydegree=false, N=N)] = 1; cfs0
retactual = Fun(S0, cfs0)(p) * basisvecs[j+1]
res = maximum(abs, ret - retactual)
if res > 1e-10
@show n,k,i,j,res
end
@test ret ≈ retactual
end


# Linear SWE operator tests
basisvecs = [cos(θ) * z; sin(θ) * z; - rhoval(z)], [-sin(θ); cos(θ); 0]
inds = [10, 3, 7]; N = sum(inds)
f = Fun((x,y,z)->x^inds[1] * y^inds[2] * z^inds[3], S, 2(N+1)^2)
dϕf = (x,y,z)->((inds[1] + inds[2]) * z^2 - inds[3] * S.family.ρ(z)^2) * x^inds[1] * y^inds[2] * z^(inds[3]-1) # ρ*∂f/∂ϕ, deg = degf + 1
dθf = (x,y,z)->z^inds[3] * y^(inds[2] - 1) * x^(inds[1] - 1) * (inds[2] * x^2 - inds[1] * y^2) # ∂f/∂θ
indsϕ = [10, 7, 7]; Nϕ = sum(indsϕ)
indsθ = [10, 3, 7]; Nθ = sum(indsθ); Nt = max(Nϕ, Nθ)
Fϕ = Fun((x,y,z)->x^indsϕ[1] * y^indsϕ[2] * z^indsϕ[3], S0, 2(Nt+1)^2); Fϕ.coefficients
Fθ = Fun((x,y,z)->x^indsθ[1] * y^indsθ[2] * z^indsθ[3], S0, 2(Nt+1)^2); Fθ.coefficients
dϕFϕ = (x,y,z)->((indsϕ[1] + indsϕ[2]) * z^2 - indsϕ[3] * S.family.ρ(z)^2) * x^indsϕ[1] * y^indsϕ[2] * z^(indsϕ[3]-1) # ρ*∂f/∂ϕ, deg = degf + 1
dθFθ = (x,y,z)->z^indsθ[3] * y^(indsθ[2] - 1) * x^(indsθ[1] - 1) * (indsθ[2] * x^2 - indsθ[1] * y^2) # ∂f/∂θ
# G
Gw = rhogradoperator(S, N)
cfs0 = Gw * f.coefficients
ret = Fun(ST, cfs0)(p)
retactual = ((weight(S, p) * dϕf(p...) - f(p) * S.family.ρ(z)^2) * basisvecs[1]
        + weight(S, p) * dθf(p...) * basisvecs[2])
maximum(abs, ret - retactual)
@test ret ≈ retactual
G = rhogradoperator(S, N; weighted=false)
cfs2 = G * f.coefficients
S2 = S.family(2.0, 0.0)
cfs2ϕ = [cfs2[2i - 1] for i = 1:(N+2)^2]
cfs2θ = [cfs2[2i] for i = 1:(N+2)^2]
Fun(S2, cfs2ϕ)(p) - dϕf(p...)
@test Fun(S2, cfs2ϕ)(p) ≈ dϕf(p...)
Fun(S2, cfs2θ)(p) - dθf(p...)
@test Fun(S2, cfs2θ)(p) ≈ dθf(p...)
# D
D = rhodivminusgradrhodotoperator(ST, Nt)
cfs0 = gettangentspacecoeffsvec(Fϕ, Fθ)
cfs = D * cfs0
ret = Fun(S, cfs)(p)
retactual = dϕFϕ(p...) + dθFθ(p...)
ret - retactual
@test ret ≈ retactual
# F
F = coriolisoperator(ST, Nt; square=false)
cfs0 = gettangentspacecoeffsvec(Fϕ, Fθ)
cfs = F * cfs0
ret = Fun(ST, cfs)(p)
Ω = B(1)# B(72921) / 1e9
retactual = Fun(ST, gettangentspacecoeffsvec(-Fθ, Fϕ) * 2 * Ω)(p) * p[3]
maximum(abs, ret - retactual)
@test ret ≈ retactual
# P (scalar and tangent space)
Ps = rho2operator(S, S, N)
cfs = Ps * f.coefficients
ret = Fun(S, cfs)(p)
retactual = f(p) * weight(S, p) * S.family.ρ(z)^2
ret - retactual
@test ret ≈ retactual
P = rho2operator(ST, Nt)
cfs0 = gettangentspacecoeffsvec(Fϕ, Fθ)
cfs = P * cfs0
fϕ, fθ = getscspacecoeffsvecs(ST, cfs)
@test Fϕ(p) * S.family.ρ(z)^2 ≈ Fun(S0, fϕ)(p)
Fϕ(p) * S.family.ρ(z)^2 - Fun(S0, fϕ)(p)
@test Fθ(p) * S.family.ρ(z)^2 ≈ Fun(S0, fθ)(p)
cfs = transformparamsoperator(ST, Nt+2) * P * cfs0
fϕ, fθ = getscspacecoeffsvecs(ST, cfs)
@test Fϕ(p) * S.family.ρ(z)^2 ≈ Fun(S2, fϕ)(p)
Fϕ(p) * S.family.ρ(z)^2 - Fun(S2, fϕ)(p)
@test Fθ(p) * S.family.ρ(z)^2 ≈ Fun(S2, fθ)(p)
# R (mult by ∇ρ to scalar fun)
R = gradrhooperator(S, N)
cfs0 = R * f.coefficients
cfs, shouldbezero = getscspacecoeffsvecs(ST, cfs0)
@test maximum(abs, shouldbezero) == 0
@test Fun(S2, cfs)(p) ≈ f(p) * p[3]
Fun(S2, cfs)(p) - f(p) * p[3]
# W (weight operator)
aa, bb = 2, 1
W = weightoperator(ST, aa, bb, Nt)
cfs0 = gettangentspacecoeffsvec(Fϕ, Fθ)
ret = Fun(ST, W * cfs0)(p)
retactual = Fun(ST, cfs0)(p) * (p[3] - S.family.α)^aa * (1 - p[3]^2)^bb
maximum(abs, ret - retactual)
@test ret ≈ retactual
# V (mult scalar by ϕ̲̂)
V = OrthogonalPolynomialFamilies.unitvecphioperator(S0, Nt)
ret = Fun(ST, V * Fϕ.coefficients)(p)
retactual = Fϕ(p) * basisvecs[1]
maximum(abs, ret - retactual)
@test ret ≈ retactual
