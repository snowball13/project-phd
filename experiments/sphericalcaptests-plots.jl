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
using Makie


isindomain(pt, D::SphericalCapFamily) = D.α ≤ pt[3] ≤ D.β && norm(pt) == 1.0
isindomain(pt, S::SphericalCapSpace) = isindomain(pt, S.family)
rhoval(z) = sqrt(1 - z^2)


# Setup
T = Float64; B = T#BigFloat
α = 0.2
SCF = SphericalCapFamily(B, T, B(α * 1000) / 1000)
a = 1.0
S = SCF(a, 0.0); S2 = SCF(2.0, a); S0 = SCF(0.0, 0.0)
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

function squareoperator(S::SphericalCapSpace, A::BandedBlockBandedMatrix, N::Int)
    """ this is to square the given operator, by reassigning the non-zero
        entries correctly for given N value
    """
    @assert (N < nblocks(A)[1] - 1 && N == nblocks(A)[2] - 1) "Invalid size of A or incorrect N val"
    C = BandedBlockBandedMatrix(Zeros{B}((N+1)^2, (N+1)^2),
                                ([N+1; 2N:-2:1], [N+1; 2N:-2:1]),
                                (A.l, A.u), (A.λ, A.μ))
    for k = 1:N
        inds = 1:2(N-k+1)
        view(C, Block(k+1, k+1)) .= view(A, Block(k+1, k+1))[inds, inds]
    end
    k = 0
    view(C, Block(k+1, k+1)) .= view(A, Block(k+1, k+1))[1:N+1, 1:N+1]
    C
end

#===#
# Plotting functions
function plot_on_domain_scalar!(scene, S, u, ϕ, θ; weighted=true)
    x = [cospi(θ)*sinpi(ϕ) for ϕ in ϕ, θ in θ]
    y = [sinpi(θ)*sinpi(ϕ) for ϕ in ϕ, θ in θ]
    z = [cospi(ϕ) for ϕ in ϕ, θ in θ]
    if weighted
        upts = [(weight(S, cospi(θ)*sinpi(ϕ), sinpi(θ)*sinpi(ϕ), cospi(ϕ))
                    * u(cospi(θ)*sinpi(ϕ), sinpi(θ)*sinpi(ϕ), cospi(ϕ))) for ϕ in ϕ, θ in θ]
    else
        upts = [u(cospi(θ)*sinpi(ϕ), sinpi(θ)*sinpi(ϕ), cospi(ϕ)) for ϕ in ϕ, θ in θ]
    end
    s = Makie.surface!(scene, x, y, z, color = upts, colormap = :viridis)#, colorrange = (-1.0, 1.0))
    scene, s
end


#=======#
# Poisson
using SpecialFunctions
# f = (x,y,z) -> 1 + erf(5(1 - 10((x - p[1])^2 + (y - p[2])^2 + (z - p[3])^2)))
# F = Fun(f, S, 2*(N+1)^2)
N = 60
Δw = laplacianoperator(S, N; square=true)
f = (x,y,z) -> (- 2exp(x)*y*z*(2+x)
                + (weight(S,x,y,z)
                    * exp(x)
                    * (y^3 + z^2*y - 4x*y - 2y))) # Exact solution to Δu = f, where u = wR10(z)*y*exp(x)
F2 = Fun(f, S, 2*(N+1)^2)
F2(p) - f(p...)
ucfs = sparse(Δw) \ F2.coefficients
u = Fun(S, ucfs)
u(p) - exp(p[1]) * p[2]
n = 100
ϕmax = (acos(S.family.α)/π)
ϕ = [0; (0.5:n*ϕmax-0.5)/n; ϕmax]
θ = [(0:2n-2) * 2 / (2n-1); 2]
scene = Scene()
scene, s = plot_on_domain_scalar!(scene, S, u, ϕ, θ)
center!(scene)
wdth, hght = 40, 700
cl = colorlegend(scene[end], raw=true, camera=campixel!, width=(wdth,hght))
vs = vbox(scene, cl)
Makie.save("experiments/saved/sphericalcap/images/poisson-u=wyexpx-N=$N.png", vs)

# Helmholtz
N = 60
k = 100
z1 = S.family.α; x1 = 0.7; vpt = [x1; sqrt(1 - x1^2 - z1^2); z1]
v = (x,y,z) -> 1 - (3(x - vpt[1])^2 + 5(y - vpt[2])^2 + 2(z - vpt[3])^2)
V = operatorclenshaw(Fun(v, S, 100), S, N)
Δw = laplacianoperator(S, N; square=true)
tw = transformparamsoperator(S, S0, N; weighted=true)
t = transformparamsoperator(S0, S, N+1; weighted=false)
A = sparse(Δw) + k^2 * V * sparse(squareoperator(S, t * tw, N))
f = (x,y,z) -> y * weight(S, x, y, z) * exp(x)
F = Fun(f, S, 2(N+1)^2); F.coefficients
F(p) - f(p...)
ucfs = A \ F.coefficients
u = Fun(S, ucfs)
u(p)
n = 200
ϕmax = (acos(S.family.α)/π)
ϕ = [0; (0.5:n*ϕmax-0.5)/n; ϕmax]
θ = [(0:2n-2) * 2 / (2n-1); 2]
scene = Scene()
scene, s = plot_on_domain_scalar!(scene, S, u, ϕ, θ)
center!(scene)
wdth, hght = 40, 700
cl = colorlegend(scene[end], raw=true, camera=campixel!, width=(wdth,hght))
vs = vbox(scene, cl)
Makie.save("experiments/saved/sphericalcap/images/helmholtz-f=wyzexpx-N=$N-n=$n.png", vs)


# Biharmonic
N = 100
S2 = S.family((B(2), B(0)))
Lw = squareoperator(S, rho2laplacianoperator(S2, S0, N; weighted=true), N)
L = squareoperator(S, rho2laplacianoperator(S0, S2, N; weighted=false), N)
Jz = OrthogonalPolynomialFamilies.jacobiz(S2, N)
t = transformparamsoperator(S, S2, N; weighted=false)
Dϕ = squareoperator(S, diffoperatorphi(S0, N), N)
t2 = transformparamsoperator(S0, S2, N; weighted=false)
v = Fun((x,y,z)->2z^2 - 2S.family.ρ(z)^2 + 3z, S2, 20)
V = operatorclenshaw(v.coefficients, S2, N)
Bw = sparse(L * Lw - Jz * t * Dϕ * Lw) - V * sparse(t2 * Lw)

f = (x,y,z)->(1 + erf(5(1 - 10((x - 0.5)^2 + y^2)))) * (1 - z^2)
F = Fun(f, S0, 2(N+1)^2)
F.coefficients
F(p) - f(p...)
ucfs = Bw \ F.coefficients
u = Fun(S, ucfs)
u(p)
n = 50
ϕmax = (acos(S.family.α)/π)
ϕ = [0; (0.5:n*ϕmax-0.5)/n; ϕmax]
θ = [(0:2n-2) * 2 / (2n-1); 2]
scene = Scene()
scene, s = plot_on_domain_scalar!(scene, S2, u, ϕ, θ)
center!(scene)
wdth, hght = 40, 700
cl = colorlegend(scene[end], raw=true, camera=campixel!, width=(wdth,hght))
vs = vbox(scene, cl)
Makie.save("experiments/saved/sphericalcap/images/biharmonic-f=erf-N=$N-n=$n.png", vs)
