using Plots
using BandedMatrices
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
                    coriolisoperator, rho2operator, gradrhooperator,
                    weightoperator, squareoperator
using LinearAlgebra
using JLD
# using Makie


isindomain(pt, D::SphericalCapFamily) = D.α ≤ pt[3] ≤ D.β && norm(pt) == 1.0
isindomain(pt, S::SphericalCapSpace) = isindomain(pt, S.family)
rhoval(z) = sqrt(1 - z^2)


# Setup
T = Float64; B = T#BigFloat
α = 0.2
SCF = SphericalCapFamily(B, T, B(α * 1000) / 1000)
a = 1.0
S = SCF(a, 0.0); S2 = SCF(2.0, 0.0); S0 = SCF(0.0, 0.0)
# ST = gettangentspace(SCF)

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
    B = sparse(zeros(size(A)))
    m1, m2 = size(A)
    for i = 1:m1, j = 1:m2
        aa = A[m1-i+1, m2-j+1]
        if abs(aa) > 1e-17
            B[i, j] = aa
        end
    end
    B
end
maxm = sum(1:20+1)
N = 30

# Laplacian
Δw = squareoperator(S, laplacianoperator(S, S, N), N)
Plots.spy(sparse(getspymat(Δw)), marker = (:square, 1), legend=nothing)
Plots.savefig("experiments/saved/sphericalcap/images/sparsity-of-laplacian.png")

Δ2w = squareoperator(S, laplacianoperator(S2, S0, N; weighted=true), N)
Plots.spy(sparse(getspymat(Δw)), marker = (:square, 1), legend=nothing)
Plots.savefig("experiments/saved/sphericalcap/images/sparsity-of-laplacian-w2.png")

Δ2 = laplacianoperator(S0, S2, N; weighted=false)
Plots.spy(sparse(getspymat(Δw)), marker = (:square, 1), legend=nothing)
Plots.savefig("experiments/saved/sphericalcap/images/sparsity-of-laplacian-2.png")

# Biharmonic
Bw = biharmonicoperator(S2, N; square=true)
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
    A = sparse(Δw) + k^2 * sparse(V) * sparse(squareoperator(S, t * tw, N))
Plots.spy(sparse(getspymat(A)), marker = (:square, 1), legend=nothing)
Plots.savefig("experiments/saved/sphericalcap/images/sparsity-of-helmholtz.png")

# ρ^2-Laplacian
Lw = squareoperator(S, rho2laplacianoperator(S, S, N), N)
Plots.spy(sparse(getspymat(Lw)), marker = (:square, 1), legend=nothing)
Plots.savefig("experiments/saved/sphericalcap/images/sparsity-of-rho2laplacian.png")

a = randn(10)
b = randn(10)
c = randn(10)
d = randn(10)
(a' * b) * (c' * d) - a' * (b * d') * c


#=======#
# Examples

# Poisson
using SpecialFunctions
# f = (x,y,z) -> 1 + erf(5(1 - 10((x - p[1])^2 + (y - p[2])^2 + (z - p[3])^2)))
N = 80
Δw = laplacianoperator(S, N; square=true)
# f = (x,y,z) -> 1 + erf(5(1 - 10((x - p[1])^2 + (y - p[2])^2 + (z - p[3])^2)))
f = (x,y,z)->(1 + erf(5(1 - 10((x - 0.5)^2 + y^2)))) * (1 - z^2)
F = Fun(f, S, 10000)
resizecoeffs!(S, F, N)
# f = (x,y,z) -> (- 2exp(x)*y*z*(2+x)
#                 + (weight(S,x,y,z)
#                     * exp(x)
#                     * (y^3 + z^2*y - 4x*y - 2y))) # Exact solution to Δu = f, where u = wR10(z)*y*exp(x)
# F2 = Fun(f, S, 1500); resizecoeffs!(S, F, N)
F(p) - f(p...)
ucfs = sparse(Δw) \ F.coefficients
u = Fun(S, ucfs)
u(p) - exp(p[1]) * p[2]
n = 50
ϕmax = (acos(S.family.α)/π)
ϕ = [0; (0.5:n*ϕmax-0.5)/n; ϕmax]
θ = [(0:2n-2) * 2 / (2n-1); 2]
scene = Scene()
scene, s = plot_on_domain_scalar!(scene, S, u, ϕ, θ)
center!(scene)
wdth, hght = 40, 700
cl = colorlegend(scene[end], raw=true, camera=campixel!, width=(wdth,hght))
vs = vbox(scene, cl)
Makie.save("experiments/saved/sphericalcap/images/poisson-f=erf-N=$N-n=$n.png", vs)

# Helmholtz
N = 100
k = 100
z1 = S.family.α; x1 = 0.7; vpt = [x1; sqrt(1 - x1^2 - z1^2); z1]
v = (x,y,z) -> 1 - (3(x - vpt[1])^2 + 5(y - vpt[2])^2 + 2(z - vpt[3])^2)
V = operatorclenshaw(Fun(v, S, 100), S, N)
Δw = squareoperator(S, laplacianoperator(S, S, N), N)
tw = transformparamsoperator(S, S0, N; weighted=true)
t = transformparamsoperator(S0, S, N+1; weighted=false)
A = Δw + k^2 * V * squareoperator(S, t * tw, N)
f = (x,y,z) -> y * weight(S, x, y, z) * exp(x)
F = Fun(f, S, 2(N+1)^2); F.coefficients
@test v(p...) * f(p...) ≈ Fun(S, V * F.coefficients)(p)
ucfs = A \ F.coefficients
u = Fun(S, ucfs)
u(p)
n = 600
ϕmax = (acos(S.family.α)/π)
ϕ = [0; (0.5:n*ϕmax-0.5)/n; ϕmax]
θ = [(0:2n-2) * 2 / (2n-1); 2]
using Makie
scene = Scene()
scene, s = plot_on_domain_scalar!(scene, S, u, ϕ, θ)
center!(scene)
wdth, hght = 40, 700
cl = colorlegend(scene[end], raw=true, camera=campixel!, width=(wdth,hght))
vs = vbox(scene, cl)
Makie.save("experiments/saved/sphericalcap/images/helmholtz-f=wyexpx-N=$N-n=$n.png", vs)


# Biharmonic
N = 80
Bw = biharmonicoperator(S2, N; square=false)
using SpecialFunctions
f = (x,y,z)->(1 + erf(5(1 - 10((x - 0.5)^2 + y^2))))
F = Fun(f, S2, 10000)
resizecoeffs!(S2, F, N+2)
# f = (x,y,z) -> (- 2exp(x)*y*z*(2+x)
#                 + (weight(S,x,y,z)
#                     * exp(x)
#                     * (y^3 + z^2*y - 4x*y - 2y))) # Exact solution to Δu = f, where u = wR10(z)*y*exp(x)
# F = Fun(f, S2, 1000); resizecoeffs!(S2, F, N+6)
F(p)
f(p...)
F(p) - f(p...)
ucfs = sparse(Bw) \ resizecoeffs!(S2, F, N+2)
u = Fun(S2, ucfs)
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
function getsolutionblocknorms(N::Int, S::SphericalCapSpace, A::BandedBlockBandedMatrix,
                                fs; makesparse=false)
    if makesparse
        AA = sparse(A)
    else
        AA = A
    end
    unorms = Vector{Vector{B}}(undef, length(fs))
    for j = 1:length(fs)
        @show j, length(fs)
        u = Fun(S, AA \ fs[j])
        ucoeffs = converttopseudo(S, u.coefficients; converttobydegree=true)
        unorms[j] = [norm(view(ucoeffs, Block(i))) for i=1:N+1]
    end
    unorms
end

# solutionblocknorms - Poisson
N = 100
Δw = squareoperator(S, laplacianoperator(S, S, N), N)

ϵ = 0.1; f5anon = (x,y,z)->norm([x; y; z] .- (1.0 * (1 / sqrt(3) + ϵ)))
f5 = Fun(f5anon, S, 2(N+1)^2); f5.coefficients
f5(p) - f5anon(p...)
ϵ = 1; f1anon = (x,y,z)->norm([x; y; z] .- (1.0 * (1 / sqrt(3) + ϵ)))
f1 = Fun(f1anon, S, 500*2); f1.coefficients
f1(p) - f1anon(p...)
ϵ = 2; f2anon = (x,y,z)->norm([x; y; z] .- (1.0 * (1 / sqrt(3) + ϵ)))
f2 = Fun(f2anon, S, 400*2); f2.coefficients
f2(p) - f2anon(p...)
ϵ = 3; f3anon = (x,y,z)->norm([x; y; z] .- (1.0 * (1 / sqrt(3) + ϵ)))
f3 = Fun(f3anon, S, 282*2); f3.coefficients
f3(p) - f3anon(p...)
ϵ = 10; f4anon = (x,y,z)->norm([x; y; z] .- (1.0 * (1 / sqrt(3) + ϵ)))
f4 = Fun(f4anon, S, 141*2); f4.coefficients
f4(p) - f4anon(p...)
# f4 = Fun(f4anon, S, 2(N+1)^2); f4.coefficients
# f4cfs = f4.coefficients
# save("experiments/saved/sphericalcap/jld/f4cfs-a=1-N=$N.jld", "f4cfs", f4.coefficients)
# f4anon = (x,y,z) -> exp(-100((x-vpt[1])^2 + (y-vpt[2])^2 + (z-vpt[3])^2))
# f4cfs = load("experiments/saved/sphericalcap/jld/f4cfs-a=1-N=100.jld", "f4cfs")

unorms = getsolutionblocknorms(N, S, Δw,
                                (resizecoeffs!(S, f5, N), resizecoeffs!(S, f1, N),
                                    resizecoeffs!(S, f2, N), resizecoeffs!(S, f3, N),
                                    resizecoeffs!(S, f4, N)))
using Plots
Plots.plot(unorms[1], line=(3, :solid), label="ϵ = 0.1", yscale=:log10, legend=:bottomleft)
Plots.plot!(unorms[2], line=(3, :dash), label="ϵ = 1")
Plots.plot!(unorms[3], line=(3, :dashdot), label="ϵ = 2")
Plots.plot!(unorms[4], line=(3, :dot), label = "ϵ = 3")
Plots.plot!(unorms[5], line=(3, :solid), label = "ϵ = 10")
Plots.xlabel!("Block")
Plots.ylabel!("Norm")
Plots.savefig("experiments/saved/sphericalcap/images/solutionblocknorms-poisson-epsilonfun-N=$N.pdf")

# Helmholtz
f1 = Fun((x,y,z)->1.0, S, 2(1+1)^2); f1.coefficients
f2 = Fun((x,y,z)->weight(S, x, y, z) * S.family.ρ(z)^2, S, 2(3+1)^2); f2.coefficients
N = 200 # TODO Run with N larger
Jx = OrthogonalPolynomialFamilies.jacobix(S, N)
Jy = OrthogonalPolynomialFamilies.jacobiy(S, N)
Jz = OrthogonalPolynomialFamilies.jacobiz(S, N)
Δw = squareoperator(S, laplacianoperator(S, S, N), N)
    tw = transformparamsoperator(S, S0, N; weighted=true)
    t = transformparamsoperator(S0, S, N+1; weighted=false)
z1 = S.family.α; x1 = 0.7; vpt = [x1; sqrt(1 - x1^2 - z1^2); z1]
    v = (x,y,z) -> 1 - (3(x - vpt[1])^2 + 5(y - vpt[2])^2 + 2(z - vpt[3])^2)
    V = operatorclenshaw(Fun(v, S, 13*2).coefficients, S, N, Jx, Jy, Jz)

ks = (1, 10, 20, 50) # , 100, 200)
styles = (:solid, :dash, :dashdot, :dot, :dash, :solid)
for j = 1:length(ks)
    k = ks[j]
    @show k
    A = Δw + k^2 * V * squareoperator(S, t * tw, N)
    unorms = getsolutionblocknorms(N, S, A, (resizecoeffs!(S, f1, N),))
    if j == 1
        Plots.plot(unorms[1], line=(3, styles[j]), label="k = $k", yscale=:log10, legend=:bottomleft)
    else
        Plots.plot!(unorms[1], line=(3, styles[j]), label="k = $k", yscale=:log10, legend=:bottomleft)
    end
    # Plots.plot!(unorms[2], line=(3, :dash), label="k = $k, f = (1-z^2) * (z - α)")
end
Plots.xlabel!("Block")
Plots.ylabel!("Norm")
Plots.savefig("experiments/saved/sphericalcap/images/solutionblocknorms-helmholtz-varyingk-N=$N.pdf")

# Biharmonic
N = 200
Bw = biharmonicoperator(S2, N)
Bw = squareoperator(S2, Bw, N)

z1 = S.family.α; x1 = 0.7; vpt = [x1; sqrt(1 - x1^2 - z1^2); z1]

ϵ = 10; f1anon = (x,y,z) -> exp(-ϵ*((x-vpt[1])^2 + (y-vpt[2])^2 + (z-vpt[3])^2))
    f1 = Fun(f1anon, S2, 2500); f1.coefficients
    f1(p) - f1anon(p...)
ϵ = 50; f2anon = (x,y,z) -> exp(-ϵ*((x-vpt[1])^2 + (y-vpt[2])^2 + (z-vpt[3])^2))
    f2 = Fun(f2anon, S2, 10000); f2.coefficients
    f2(p) - f2anon(p...)
ϵ = 100; f3anon = (x,y,z) -> exp(-ϵ*((x-vpt[1])^2 + (y-vpt[2])^2 + (z-vpt[3])^2))
    f3cfs = load("experiments/saved/sphericalcap/jld/f4cfs-a=1-N=100-S2.jld", "f4cfs")
    f3 = Fun(S2, f3cfs); f3.coefficients
    f3(p) - f3anon(p...)
ϵ = 200; f4anon = (x,y,z) -> exp(-ϵ*((x-vpt[1])^2 + (y-vpt[2])^2 + (z-vpt[3])^2))
    f4 = Fun(f4anon, S2, 2(150+1)^2); f4.coefficients
    f4(p) - f4anon(p...)

save("experiments/saved/sphericalcap/jld/biharmonic-f4cfs-N=150.jld", "f4cfs", f4.coefficients)
unorms = getsolutionblocknorms(N, S2, Bw,
                    (resizecoeffs!(S2, f1, N), resizecoeffs!(S2, f2, N),
                     resizecoeffs!(S2, f3, N), resizecoeffs!(S2, f4, N)))
using Plots
Plots.plot(unorms[1], line=(3, :solid), label="ϵ = 10", yscale=:log10, legend=:bottomleft)
Plots.plot!(unorms[2], line=(3, :dash), label="ϵ = 50")
Plots.plot!(unorms[3], line=(3, :dashdot), label="ϵ = 100")
Plots.plot!(unorms[4], line=(3, :dot), label = "ϵ = 200")
Plots.xlabel!("Block")
Plots.ylabel!("Norm")
Plots.savefig("experiments/saved/sphericalcap/images/solutionblocknorms-biharmonic-expfun-N=$N.pdf")



#====#

# Complexity plots for building differential operators
using BenchmarkTools, Plots
function testcomplexityofdifferentialoperator(S::SphericalCapSpace, N::Int, v::Fun)
    Jz = OrthogonalPolynomialFamilies.jacobiz(S, N)
    Jx = BandedBlockBandedMatrix(Zeros{B}((N+1)^2, (N+1)^2),
                                    [N+1; 2N:-2:1], [N+1; 2N:-2:1],
                                    (0, 0), (0, 0))
    Jy = Jx
    # @show "done Jacobi mats"
    V = operatorclenshaw(v.coefficients, S, N, Jx, Jy, Jz; rotationalinvariant=true)
    L = laplacianoperator(S, S, N; square=true)
    L + V
end
# function testcomplexityofdifferentialoperator(S::SphericalCapSpace, N::Int, v::Fun)
#     V = operatorclenshaw(v.coefficients, S, N)
#     L = laplacianoperator(S, S, N; square=true)
#     L + V
# end
function testcomplexityofdifferentialoperator2(N::Int)
    laplacianoperator(S, S, N; square=true)
end
function factorizeblockdiagonal(A::BandedBlockBandedMatrix{T}, N::Int) where T
    qrvec = [factorize(A[Block(k+1, k+1)]) for k = 0:N]
end
function factorizeblockdiagonalsolve(qrvec::Vector{L}, N::Int, cfs::Vector{T}) where {L,T}
    # qrvec = [factorize(A[Block(k+1, k+1)]) for k = 0:N]
    # NOTE assumes square matrix
    ret = copy(cfs)
    k = 0
    view(ret, 1:N+1) .= qrvec[k+1] \ view(cfs, 1:N+1)
    ind = N+1
    for k = 1:N
        view(ret, (ind+1):(ind + 2(N - k + 1))) .= qrvec[k+1] \ view(ret, (ind+1):(ind+ 2(N - k + 1)))
        ind += 2(N - k + 1)
    end
    ret
end
function testcomplexity!(S::SphericalCapSpace, v::Fun, f::Fun, Ns, tbuild, tfact, tsolve)
    # # TODO create an operatorclenshaw method for rotational invariant V,
    # # i.e. v(x,y,z) ≡ v(z), where the result V is Block-Diagonal and we dont
    # # need to account for the Jx, Jy bits.
    for k = 1:length(Ns)
        N = Ns[k]
        @show "TEST COMPLEXITY", k, N

        # belapsed
        tbuild[k] = @belapsed testcomplexityofdifferentialoperator($(S), $(N), $(v))
        A = testcomplexityofdifferentialoperator(S, N, v)
        tfact[k] = @belapsed factorizeblockdiagonal($(A), $(N)) # factorize
        Afact = factorizeblockdiagonal(A, N) # factorize(A)
        cfs = resizecoeffs!(S, f, N)
        tsolve[k] = @belapsed factorizeblockdiagonalsolve($(Afact), $(N), $(cfs)) # $(A) \ resizecoeffs!($(S), $(f), $(N))

        # # elapsed
        # # tbuild[k] = @elapsed testcomplexityofdifferentialoperator(S, N, v, Jx, Jy, Jz) # we have v=v(z)
        # # A = testcomplexityofdifferentialoperator(S, N, v, Jx, Jy, Jz)
        # tbuild[k] = @elapsed testcomplexityofdifferentialoperator(S, N, v) # We have v=v(x,y,z)
        # A = testcomplexityofdifferentialoperator(S, N, v)
        # tfact[k] = @elapsed factorize(A)
        # Afact = factorize(A)
        # tsolve[k] = @elapsed Afact \ resizecoeffs!(S, f, N)
    end
    tbuild, tfact, tsolve
end





N = 300
A = testcomplexityofdifferentialoperator(S, N, v)
@time AA = BandedMatrices.qr(A); 1
@time AA = factorize(A); 1
@time qrvec = [factorize(A[Block(k+1, k+1)]) for k = 0:N]; 1
@time ret = AA \ resizecoeffs!(S, f, N)
@time ret2 = factorizeblockdiagonalsolve(qrvec, N, resizecoeffs!(S, f, N))
ret ≈ ret2


# NOTE that the operatorclenshaw as implemented is basically unable to handle any large N>100.
# Try sifting through zero coeff vals and thus limiting the arithmetic in the operatorclenshaw methods

Nend = 1000; Ns = 100:100:Nend
tbuild = Array{Float64}(undef, length(Ns))
tfact, tsolve = copy(tbuild), copy(tbuild)
vanon = (x,y,z) -> cos(z) # (x,y,z) -> x * cos(z * y)
v = Fun(vanon, S, 500)
v(p) - vanon(p...)
v.coefficients
# Exact solution to Δu = f, where u = wR10(z)*y*exp(x)
fanon = (x,y,z) -> (- 2exp(x)*y*z*(2+x) + (weight(S,x,y,z) * exp(x) * (y^3 + z^2*y - 4x*y - 2y)))
f = Fun(fanon, S, 2*21^2)
f(p) - fanon(p...)
resizedataonedimops!(S, 20)
resizedataonedimops!(S, Nend+5)

testcomplexity!(S, v, f, Ns, tbuild, tfact, tsolve)
tbuild, tfact, tsolve
save("experiments/saved/sphericalcap/jld/complexity-belapsed-new-Nend=$Nend.jld", "tbuild", tbuild, "tfact", tfact, "tsolve", tsolve)
# tbuildold = load("experiments/saved/sphericalcap/jld/complexity2-Nend=60.jld", "tbuild")
# tfactold = load("experiments/saved/sphericalcap/jld/complexity-Nend=60.jld", "tfact")
# tsolveold = load("experiments/saved/sphericalcap/jld/complexity-Nend=60.jld", "tsolve")

tb = vcat(tbuildold, tbuild)
tf = vcat(tfactold, tfact)
ts = vcat(tsolveold, tsolve)


using Plots
Plots.plot(Ns, tbuild; line=(3, :dash), yscale=:log10, xscale=:log10, legend=:bottomright, label="Building")
Plots.plot!(Ns, tfact; line=(3, :dot), label="Factorisation")
Plots.plot!(Ns, tsolve; line=(3, :dashdot), label="Solve")
Plots.plot!(Ns, tbuild + tfact + tsolve; line=(3, :solid), label="Total")
Plots.plot!(Ns, Ns.^2; line=(3, :solid), label="Total")
Plots.xlabel!("Degree")
Plots.ylabel!("Time")
savefig("experiments/saved/sphericalcap/images/complexity-belapsed-new-Nend=$Nend.pdf")



#====#


T = Float64; B = T#BigFloat
α = 0.2
SCF = SphericalCapFamily(B, T, B(α * 1000) / 1000)
a = 1.0
S = SCF(a, 0.0); S2 = SCF(2.0, 0.0); S0 = SCF(0.0, 0.0)

#=
Condition numbers of Laplacian
=#

using Plots

# For given N
N = 200
Δw = squareoperator(S, laplacianoperator(S, S, N), N)
P = Diagonal([1/Δw[i,i] for i=1:(N+1)^2])
PΔw = P * Δw
conditionnumbers_un = zeros(N+1)
conditionnumbers_pre = zeros(N+1)
for m = 0:N
    conditionnumbers_un[m+1] = LinearAlgebra.cond(view(Δw, Block(m+1, m+1)))
    conditionnumbers_pre[m+1] = LinearAlgebra.cond(view(PΔw, Block(m+1, m+1)))
end
conditionnumbers_un, conditionnumbers_pre
Plots.plot([i for i=0:N], conditionnumbers_un; line=(3, :dash),
            legend=:topright, label="Unconditioned")
Plots.plot!([i for i=0:N], conditionnumbers_pre; line=(3, :dot), label="Preconditioned")
Plots.xlabel!("Fourier mode block")
Plots.ylabel!("Condition number")
savefig("experiments/saved/sphericalcap/images/condition-numbers-laplacian-N=$N.pdf")

# Max block cond no. as we increase N
function conditionnumberplot(S::SphericalCapSpace, Ns)
    """ Returns the (max) condition numbers for the Laplacian and preconditioned
    Laplacian operator matrix blocks for increasing N's.
    """
    maxconditionnumber_un = zeros(length(Ns))
    maxconditionnumber_pre = zeros(length(Ns))
    i = 1
    for N in Ns
        @show N
        Δw = squareoperator(S, laplacianoperator(S, S, N), N)
        P = Diagonal([1/Δw[i,i] for i=1:(N+1)^2])
        PΔw = P * Δw
        conditionnumbers_un = zeros(N+1)
        conditionnumbers_pre = zeros(N+1)
        for m = 0:N
            conditionnumbers_un[m+1] = LinearAlgebra.cond(view(Δw, Block(m+1, m+1)))
            conditionnumbers_pre[m+1] = LinearAlgebra.cond(view(PΔw, Block(m+1, m+1)))
        end
        maxconditionnumber_un[i] = maximum(conditionnumbers_un)
        maxconditionnumber_pre[i] = maximum(conditionnumbers_pre)
        i += 1
    end
    maxconditionnumber_un, maxconditionnumber_pre
end
Ns = 10:10:200
maxconditionnumber_un, maxconditionnumber_pre = conditionnumberplot(S, Ns)
Plots.plot(Ns, maxconditionnumber_un; line=(3, :dash),
            legend=:topleft, label="Unconditioned")
Plots.plot!(Ns, maxconditionnumber_pre; line=(3, :dot), label="Preconditioned")
Plots.xlabel!("Degree")
Plots.ylabel!("Max condition number of all blocks")
savefig("experiments/saved/sphericalcap/images/condition-numbers-max-laplacian.pdf")







#--------------------------------------


#====#
# SH stuff

function muval(l, r)
    if r < 0
        return (l+1)*sqrt(l/(2l+1))
    elseif r > 0
        return l*sqrt((l+1)/(2l+1))
    else
        return im*sqrt(l*(l+1))
    end
end

#=
Function that returns the Clebsch-Gordan coefficient
    <L, M-ms; 1, ms | J, M>
=#
function clebsch_gordan_coeff(J, M, L, ms)
    out = 0.0
    if ms == 1
        if J == L+1
            out = sqrt((L+M)*(L+M+1) / ((2L+1)*(2L+2)))
        elseif J == L
            out = - sqrt((L+M)*(L-M+1) / (2L*(L+1)))
        elseif J == L-1
            out = sqrt((L-M)*(L-M+1) / (2L*(2L+1)))
        end
    elseif ms == 0
        if J == L+1
            out = sqrt((L-M+1)*(L+M+1) / ((L+1)*(2L+1)))
        elseif J == L
            out = M / sqrt(L*(L+1))
        elseif J == L-1
            out = - sqrt((L-M)*(L+M) / (L*(2L+1)))
        end
    elseif ms == -1
        if J == L+1
            out = sqrt((L-M)*(L-M+1) / ((2L+1)*(2L+2)))
        elseif J == L
            out = sqrt((L-M)*(L+M+1) / (2L*(L+1)))
        elseif J == L-1
            out = sqrt((L+M)*(L+M+1) / (2L*(2L+1)))
        end
    end
    return out
end
function shcoeff(l, m)
    α = 0
    if m == 0
        α = sqrt((2l+1)/(4pi))
    elseif m > 0
        α = sqrt((2l+1)/(4pi) * factorial(big(l+m)) * factorial(big(l-m))) / (factorial(big(l)) * (-2.0)^m)
    else
        m = abs(m)
        α = sqrt((2l+1)/(4pi) * factorial(big(l+m)) * factorial(big(l-m))) / (factorial(big(l)) * (2.0)^m)
    end
    Float64(α)
end
function shreccoeff(l, m, j; type::String="x")
    if type == "z"
        if j == 1
            if l - m < 1
                return 0
            end
            ret = (l / 2l+1) * (shcoeff(l, m) / shcoeff(l-1, m))
        elseif j == 2
            ret = (l - m + 1) * (l + m + 1) / ((2l + 1) * (l + 1)) * (shcoeff(l, m) / shcoeff(l+1, m))
        else
            error("invalid j")
        end
    else
        if j == 1
            # Do we have a non-zero coeff?
            if m ≤ 0 && l + m < 2
                return 0.0
            end
            if m > 0
                ret = 2l / (2l + 1)
            else
                ret = - l / (2 * (2l + 1))
            end
            ret *= shcoeff(l, m) / (2 * shcoeff(l-1, m-1))
        elseif j == 2
            if m ≥ 0 && l - m < 2
                return 0.0
            end
            if m < 0
                ret = 2l / (2l + 1)
            else
                ret = - l / (2 * (2l + 1))
            end
            ret *= shcoeff(l, m) / (2 * shcoeff(l-1, m+1))
        elseif j == 3
            if m > 0
                ret = - 2(l - m + 2) * (l - m + 1) / ((2l + 1) * (l + 1))
            else
                ret = (l - m + 2) * (l - m + 1) / (2 * (2l + 1) * (l + 1))
            end
            ret *= shcoeff(l, m) / (2 * shcoeff(l+1, m-1))
        elseif j == 4
            if m < 0
                ret = - 2(l + m + 2) * (l + m + 1) / ((2l + 1) * (l + 1))
            else
                ret = (l + m + 2) * (l + m + 1) / (2 * (2l + 1) * (l + 1))
            end
            ret *= shcoeff(l, m) / (2 * shcoeff(l+1, m+1))
        end
        if type == "y"
            ret *= (-1)^(j+1) * im
        end
    end
    ret
end


for l = 1:10, m = -l:l
    if l - 2 ≥ abs(m)
        aa = clebsch_gordan_coeff(l, m, l-1, -1) * shreccoeff(l-1, m+1, 1; type="y")
        bb = clebsch_gordan_coeff(l, m, l-1, 1) * shreccoeff(l-1, m-1, 2; type="y")
        ret = abs(aa - bb)
        if ret > 1e-16
            @show l, m, ret
        end
    end
end

for l = 1:10, m = -l:l
    aa = clebsch_gordan_coeff(l, m, l+1, -1) * shreccoeff(l+1, m+1, 3; type="y")
    bb = clebsch_gordan_coeff(l, m, l+1, 1) * shreccoeff(l+1, m-1, 4; type="y")
    ret = abs(aa - bb)
    if ret > 1e-15
        @show l, m, ret
    end
end

l, m = 4, -1
for l = 1:20, m = -l:l
    aa = (- clebsch_gordan_coeff(l, m, l-1, -1) * shreccoeff(l-1, m+1, 3; type="y")
            + clebsch_gordan_coeff(l, m, l-1, 1) * shreccoeff(l-1, m-1, 4; type="y"))
    bb = (- clebsch_gordan_coeff(l, m, l+1, -1) * shreccoeff(l+1, m+1, 1; type="y")
            + clebsch_gordan_coeff(l, m, l+1, 1) * shreccoeff(l+1, m-1, 2; type="y"))
    cc = clebsch_gordan_coeff(l, m, l, 0)
    ret = abs(cc * muval(l, 0) - 2(aa * muval(l, -1) + bb * muval(l, 1)) / sqrt(2))
    if ret > 1e-14
        @show l, m, ret
    end
end
