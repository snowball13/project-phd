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
                    rhodivminusgradrhodotoperator, gradrhooperator, weightoperator,
                    squareoperator
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


#====#

#=
Sparsity of Operators
=#
using Plots
function getspymat(A)
    B = sparse(A)
    m1, m2 = size(A)
    for i = 1:m1, j = 1:m2
        B[i, j] = A[m1-i+1, m2-j+1]
    end
    B
end
maxm = sum(1:20+1)
N = 30

# Laplacian
Δw = squareoperator(S, laplacianoperator(S, N), N)
    Plots.spy(sparse(getspymat(Δw)), marker = (:square, 1), legend=nothing)
    Plots.savefig("experiments/saved/sphericalcap/images/sparsity-of-laplacian.png")

# Biharmonic
Lw = squareoperator(S, rho2laplacianoperator(S2, S0, N; weighted=true), N)
    L = squareoperator(S, rho2laplacianoperator(S0, S2, N; weighted=false), N)
    Jz = OrthogonalPolynomialFamilies.jacobiz(S2, N)
    t = transformparamsoperator(S, S2, N; weighted=false)
    Dϕ = squareoperator(S, diffoperatorphi(S0, N), N)
    t2 = transformparamsoperator(S0, S2, N; weighted=false)
    v = Fun((x,y,z)->2z^2 - 2S.family.ρ(z)^2 + 3z, S2, 20)
    V = operatorclenshaw(v.coefficients, S2, N)
    Bw = sparse(L * Lw - Jz * t * Dϕ * Lw) - V * sparse(t2 * Lw)
Plots.spy(sparse(getspymat(Bw)), marker = (:square, 1), legend=nothing)
    Plots.savefig("experiments/saved/sphericalcap/images/sparsity-of-biharmonic.png")

# Helmholtz
k = 20
    z1 = S.family.α; x1 = 0.7; vpt = [x1; sqrt(1 - x1^2 - z1^2); z1]
    v = (x,y,z) -> 1 - (3(x - vpt[1])^2 + 5(y - vpt[2])^2 + 2(z - vpt[3])^2)
    V = operatorclenshaw(Fun(v, S, 100), S, N)
    Δw = squareoperator(S, laplacianoperator(S, N), N)
    tw = transformparamsoperator(S, S0, N; weighted=true)
    t = transformparamsoperator(S0, S, N+1; weighted=false)
    A = Δw + k^2 * V * squareoperator(S, t * tw, N)
Plots.spy(sparse(getspymat(A)), marker = (:square, 1), legend=nothing)
Plots.savefig("experiments/saved/sphericalcap/images/sparsity-of-helmholtz.png")



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
A = Δw + k^2 * V * squareoperator(S, t * tw, N)
f = (x,y,z) -> y * weight(S, x, y, z) * exp(x)
F = Fun(f, S, 2(N+1)^2); F.coefficients
@test v(p...) * f(p...) ≈ Fun(S, V * F.coefficients)(p)
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



#=====#
#=
Plotting the norms of each block of coeffs for solutions for different RHSs
=#
function getsolutionblocknorms(N::Int, S::SphericalCapSpace, A, f1, f2, f3, f4)
    u1 = Fun(S, A \ f1)
    @show "1"
    u2 = Fun(S, A \ f2)
    @show "2"
    u3 = Fun(S, A \ f3)
    @show "3"
    u4 = Fun(S, A \ f4)
    u1coeffs = converttopseudo(S, u1.coefficients; converttobydegree=true)
    u2coeffs = converttopseudo(S, u2.coefficients; converttobydegree=true)
    u3coeffs = converttopseudo(S, u3.coefficients; converttobydegree=true)
    u4coeffs = converttopseudo(S, u4.coefficients; converttobydegree=true)
    u1norms = zeros(N+1)
    u2norms = zeros(N+1)
    u3norms = zeros(N+1)
    u4norms = zeros(N+1)
    for i = 1:N+1
        u1norms[i] = norm(view(u1coeffs, Block(i)))
        u2norms[i] = norm(view(u2coeffs, Block(i)))
        u3norms[i] = norm(view(u3coeffs, Block(i)))
        u4norms[i] = norm(view(u4coeffs, Block(i)))
    end
    u1norms, u2norms, u3norms, u4norms
end

# solutionblocknorms - Poisson
N = 100
Δw = squareoperator(S, laplacianoperator(S, N), N)
z1 = S.family.α; x1 = 0.7; vpt = [x1; sqrt(1 - x1^2 - z1^2); z1]
f1 = Fun((x,y,z)->1.0, S, 10); f1.coefficients
f2 = Fun((x,y,z)->weight(S, x, y, z)^2, S, 100); f2.coefficients
f3 = Fun((x,y,z)->weight(S, x, y, z) * S.family.ρ(z)^2, S, 300); f3.coefficients
f4anon = (x,y,z) -> exp(-100((x-vpt[1])^2 + (y-vpt[2])^2 + (z-vpt[3])^2))
f4 = Fun(f4anon, S, 40000); f4.coefficients
resize!(S.reccoeffsa, 0)
using JLD
save("experiments/saved/sphericalcap/jld/f4cfs-a=1-N=200.jld", "f4cfs", f4.coefficients)
f4cfs = load("experiments/saved/sphericalcap/jld/f4cfs-a=1-N=200.jld", "f4cfs")
f4 = Fun(S, f4cfs)
f4(p)
f4anon(p...)
f4(p) - f4anon(p...)
u1norms, u2norms, u3norms, u4norms = getsolutionblocknorms(N, S, Δw, resizecoeffs!(S, f1, N),
                resizecoeffs!(S, f2, N), resizecoeffs!(S, f3, N), resizecoeffs!(S, f4, N))
using Plots
Plots.plot(u1norms, line=(3, :solid), label="f(x,y,z) = 1", xscale=:log10, yscale=:log10, legend=:bottomleft)
Plots.plot!(u2norms, line=(3, :dash), label="f(x,y,z) = W{(1)}^2")
Plots.plot!(u3norms, line=(3, :dashdot), label="f(x,y,z) = (1-z^2) * W{(1)}")
Plots.plot!(u4norms, line=(3, :dot), label = "f(x,y,z) = exp(-100(||x-p||^2))")
Plots.xlabel!("Block")
Plots.ylabel!("Norm")
Plots.savefig("experiments/saved/sphericalcap/images/solutionblocknorms-poisson-N=$N.pdf")

# Helmholtz
N = 100
k = 20
    z1 = S.family.α; x1 = 0.7; vpt = [x1; sqrt(1 - x1^2 - z1^2); z1]
    v = (x,y,z) -> 1 - (3(x - vpt[1])^2 + 5(y - vpt[2])^2 + 2(z - vpt[3])^2)
    V = operatorclenshaw(Fun(v, S, 100), S, N)
    Δw = squareoperator(S, laplacianoperator(S, N), N)
    tw = transformparamsoperator(S, S0, N; weighted=true)
    t = transformparamsoperator(S0, S, N+1; weighted=false)
    A = Δw + k^2 * V * squareoperator(S, t * tw, N)
u1norms, u2norms, u3norms, u4norms = getsolutionblocknorms(N, S, A, resizecoeffs!(S, f1, N),
                resizecoeffs!(S, f2, N), resizecoeffs!(S, f3, N), resizecoeffs!(S, f4, N))
using Plots
Plots.plot(u1norms, line=(3, :solid), label="f(x,y,z) = 1", xscale=:log10, yscale=:log10, legend=:bottomleft)
Plots.plot!(u2norms, line=(3, :dash), label="f(x,y,z) = W{(1)}^2")
Plots.plot!(u3norms, line=(3, :dashdot), label="f(x,y,z) = (1-z^2) * W{(1)}")
Plots.plot!(u4norms, line=(3, :dot), label = "f(x,y,z) = exp(-100(||x-p||^2))")
Plots.xlabel!("Block")
Plots.ylabel!("Norm")
Plots.savefig("experiments/saved/sphericalcap/images/solutionblocknorms-helmholtz-N=$N.pdf")

# Biharmonic
N = 100
Bw = biharmonicoperator(S2, N)
z1 = S.family.α; x1 = 0.7; vpt = [x1; sqrt(1 - x1^2 - z1^2); z1]
f1 = Fun((x,y,z)->1.0, S2, 10); f1.coefficients
f2 = Fun((x,y,z)->weight(S, x, y, z)^2, S2, 100); f2.coefficients
f3 = Fun((x,y,z)->weight(S, x, y, z) * S.family.ρ(z)^2, S2, 300); f3.coefficients
f4anon = (x,y,z) -> exp(-10((x-vpt[1])^2 + (y-vpt[2])^2 + (z-vpt[3])^2))
f4 = Fun(f4anon, S2, 3000); f4.coefficients
f4(p)
f4anon(p...)
f4(p) - f4anon(p...)
u1norms, u2norms, u3norms, u4norms = getsolutionblocknorms(N, S2, Bw,
                    resizecoeffs!(S2, f1, N), resizecoeffs!(S2, f2, N),
                    resizecoeffs!(S2, f3, N), resizecoeffs!(S2, f4, N))
using Plots
Plots.plot(u1norms, line=(3, :solid), label="f(x,y,z) = 1", xscale=:log10, yscale=:log10, legend=:bottomleft)
Plots.plot!(u2norms, line=(3, :dash), label="f(x,y,z) = W{(1)}^2")
Plots.plot!(u3norms, line=(3, :dashdot), label="f(x,y,z) = (1-z^2) * W{(1)}")
Plots.plot!(u4norms, line=(3, :dot), label = "f(x,y,z) = exp(-10(||x-p||^2))")
Plots.xlabel!("Block")
Plots.ylabel!("Norm")
Plots.savefig("experiments/saved/sphericalcap/images/solutionblocknorms-biharmonic-N=$N.pdf")


#====#

"""
# TODO:
Sort out Makie
Work out if Biharmonic operator is correct
    - what RHSs should I use?
Add to paper
"""
