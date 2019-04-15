using Revise
using ApproxFun
using Test, Profile
using StaticArrays, LinearAlgebra, BlockArrays, BlockBandedMatrices,
        SparseArrays
# using Makie
using OrthogonalPolynomialFamilies
import OrthogonalPolynomialFamilies: points, pointswithweights, getopptseval,
                    opevalatpts, inner2, getopindex, getnk, resizecoeffs!,
                    transformparamsoperator, weightedpartialoperatorx,
                    weightedpartialoperatory, partialoperatorx, partialoperatory,
                    laplaceoperator, derivopevalatpts, getderivopptseval, getopnorms,
                    getopnorm, operatorclenshaw, weight, biharmonicoperator
using SpecialFunctions

#===#

# Set up for experiments
precision(BigFloat)
setprecision(800)
D = HalfDiskFamily()
a, b = 1.0, 1.0; S = D(a, b)
x, y = 0.42, -0.2; z = [x; y]
N = 25 # Sparsity plots
@time bihar = biharmonicoperator(D(2.0, 2.0), N)
N = 100 # Solution plots
@time bihar = biharmonicoperator(D(2.0, 2.0), N)
@time errfuncfs = Fun((x,y)->(1 + erf(5(1 - 10((x - 0.5)^2 + (y)^2)))), S, N*(N+1)).coefficients
N = 200 # solutionblocknorms plot
ff = (x,y)->exp(-1000((x-0.2)^2 + (y-0.2)^2))
pts = pointswithweights(S, N*(N+1))[1]
vals = [ff(pt...) for pt in pts]
HDTP = OrthogonalPolynomialFamilies.HalfDiskTransformPlan(S, vals)
@time f4cfs = HDTP * vals
S



using JLD
B = BigFloat
# save("experiments/saved/laplacian-w11.jld", "Lw11", Δw)
# save("experiments/saved/biharmonic.jld", "bihar", bihar)
# save("experiments/saved/halfdiskspace11.jld", "opnorms", S.opnorms, "A", S.A, "B", S.B, "C", S.C, "DT", S.DT)
# save("experiments/saved/Hfamily-11p5.jld", "Ha", S.family.H(B(1.0),B(1.5)).a, "Hb", S.family.H(B(1.0),B(1.5)).b)
# save("experiments/saved/Pfamily-11.jld", "Pa", S.family.P(B(1.0),B(1.0)).a, "Pb", S.family.P(B(1.0),B(1.0)).b)
# save("experiments/saved/errfuncfs.jld", "erfuncfs", errfuncfs)
# save("experiments/saved/f4cfs.jld", "f4cfs", f4cfs)
Δw = load("experiments/saved/laplacian-w11.jld", "Lw11")
bihar = load("experiments/saved/biharmonic.jld", "bihar")
errfuncfs = load("experiments/saved/errfuncfs.jld", "erfuncfs")
f4cfs = load("experiments/saved/f4cfs.jld", "f4cfs")
D = HalfDiskFamily()
    d = load("experiments/saved/halfdiskspace11.jld")
    S = D(a, b)
    resize!(S.A, length(d["A"]))
    S.A[:] = d["A"][:]
    resize!(S.B, length(d["B"]))
    S.B[:] = d["B"][:]
    resize!(S.C, length(d["C"]))
    S.C[:] = d["C"][:]
    resize!(S.DT, length(d["DT"]))
    S.DT[:] = d["DT"][:]
    resize!(S.opnorms, length(d["opnorms"]))
    S.opnorms[:] = d["opnorms"][:]
    d = load("experiments/saved/Hfamily-11p5.jld")
    H = D.H(B(1.0),B(1.5)); P = D.P(B(1.0),B(1.0))
    resize!(H.a, length(d["Ha"]))
    H.a[:] = d["Ha"][:]
    resize!(H.b, length(d["Hb"]))
    H.b[:] = d["Hb"][:]
    d = load("experiments/saved/Pfamily-11.jld")
    resize!(P.a, length(d["Pa"]))
    P.a[:] = d["Pa"][:]
    resize!(P.b, length(d["Pb"]))
    P.b[:] = d["Pb"][:]
    # x * P[n](x) == (b[n] * P[n+1](x) + a[n] * P[n](x) + b[n-1] * P[n-1](x))
    # => P[n+1](x) == 1/b[n] * (x * P[n](x) - a[n] * P[n](x) - b[n-1] * P[n-1](x))
    resize!(H.ops, 2)
    n = length(H.a)
    w = H.weight
    x = Fun(identity, domain(w))
    H.ops[1] = Fun(1/sqrt(sum(w)),space(x)); H.ops[2] = ((x - H.a[1]) * H.ops[1]) / H.b[1]
    for k = 2:n
        p = ((x - H.a[k]) * H.ops[2] - H.b[k-1] * H.ops[1]) / H.b[k]
        H.ops[1] = H.ops[2]
        H.ops[2] = p
    end
    resize!(P.ops, 2)
    n = length(P.a)
    w = P.weight
    x = Fun(identity, domain(w))
    P.ops[1] = Fun(1/sqrt(sum(w)),space(x)); P.ops[2] = ((x - P.a[1]) * P.ops[1]) / P.b[1]
    for k = 2:n
        p = ((x - P.a[k]) * P.ops[2] - P.b[k-1] * P.ops[1]) / P.b[k]
        P.ops[1] = P.ops[2]
        P.ops[2] = p
    end
    S


#====#



#=
Sparsity of laplace operator
=#
using PyPlot
# a, b = 1.0, 1.0; D = HalfDiskFamily(); S = D(a, b)
# N = 30
maxm = sum(1:20+1)
PyPlot.clf()
    PyPlot.spy(Array(sparse(Δw[1:maxm, 1:maxm])))
    PyPlot.axis(xmin=0, xmax=maxm, ymin=maxm, ymax=0)
    savefig("experiments/sparsityoflaplacian-w11")
PyPlot.clf()
    PyPlot.spy(Array(sparse(bihar[1:maxm, 1:maxm])))
    PyPlot.axis(xmin=0, xmax=maxm, ymin=maxm, ymax=0)
    savefig("experiments/sparsityofbiharmonic")
PyPlot.clf()
    N = 25
    k = 200
    DD = HalfDiskFamily(); v = Fun((x,y)->(1 - (3(x-1)^2 + 5y^2)), DD(0.0, 0.0))
    V = operatorclenshaw(v, D(0.0, 0.0), N+3)
    T = transformparamsoperator(D(0.0, 0.0), S, N+3)
    Tw = transformparamsoperator(S, D(0.0, 0.0), N, weighted=true)
    A = Δw + (k^2 * T * V * Tw)[1:getopindex(N,N), 1:getopindex(N,N)]
    PyPlot.spy(Array(sparse(A[1:maxm, 1:maxm])))
    PyPlot.axis(xmin=0, xmax=maxm, ymin=maxm, ymax=0)
    savefig("experiments/sparsityofhelmholtz")

#=
Examples
=#

using PyPlot
evalu(u, z) = isindomain(z, u.space.family) ? u(z) : NaN
isindomain(x, D::HalfDiskFamily) = 0 ≤ x[1] ≤ 1 && -sqrt(1-x[1]^2) ≤ x[2] ≤ sqrt(1-x[1]^2)

#=
Poisson
Solution of Δu = f, where f(x,y) = 1 + erf(5(1 - 10((x - 0.5)^2 + (y)^2)))
=#
errfun.coefficients
N = getnk(ncoefficients(errfun))[1]
u = Fun(S, sparse(Δw) \ errfun.coefficients)
# Create arrays of (x,y) points
n = 500
x = LinRange(0, 1, n)
y = LinRange(-1, 1, n)
sln = zeros(length(x), length(y))
for i = 1:length(x)
    for j = 1:length(y)
        val = evalu(u, [x[i]; y[j]])
        sln[j,i] = OrthogonalPolynomialFamilies.weight(S, x[i], y[j]) * val
    end
    @show i
end
PyPlot.clf()
w, h = plt[:figaspect](2.0)
PyPlot.figure(figsize=(w,h))
PyPlot.pcolor(x, y, sln)
savefig("experiments/solution-poisson")
#=
Variable coefficient Helmholtz
Solution of Δu + k²vu = f
=#
N = 100
k = 200
f = Fun((x,y)->exp(x)*x*y, S, N*(N+1))
DD = HalfDiskFamily(); v = Fun((x,y)->(1 - (3(x-1)^2 + 5y^2)), DD(0.0, 0.0))
V = operatorclenshaw(v, D(0.0, 0.0), N+3)
T = transformparamsoperator(D(0.0, 0.0), S, N+3)
Tw = transformparamsoperator(S, D(0.0, 0.0), N, weighted=true)
# save("experiments/saved/helmholtz.jld", "fcfs", f.coefficients, "T", T, "Tw", Tw)
save("experiments/saved/helmholtz-V.jld", "V", V)
fcfs = load("experiments/saved/helmholtz.jld", "fcfs")
T = load("experiments/saved/helmholtz.jld", "T")
Tw = load("experiments/saved/helmholtz.jld", "Tw")
V = load("experiments/saved/helmholtz-V.jld", "V")
A = Δw[1:getopindex(N,N), 1:getopindex(N,N)] + (k^2 * T * V * Tw)[1:getopindex(N,N), 1:getopindex(N,N)]
u = Fun(S, sparse(A) \ f.coefficients)
# Create arrays of (x,y) points
n = 1500
x = LinRange(0, 1, n)
y = LinRange(-1, 1, n)
sln = zeros(length(x), length(y))
for i = 1:length(x)
    for j = 1:length(y)
        val = evalu(u, [x[i]; y[j]])
        sln[j,i] = OrthogonalPolynomialFamilies.weight(S, x[i], y[j]) * val
    end
    @show i
end
PyPlot.clf()
w, h = plt[:figaspect](2.0)
PyPlot.figure(figsize=(w,h))
PyPlot.pcolor(x, y, sln)
PyPlot.savefig("experiments/solution-helmholtz")
#=
Biharmonic
Solution of Δ²u = f, where f(x,y) = 1 + erf(5(1 - 10((x - 0.5)^2 + (y)^2)))
=#
N = getnk(ncoefficients(errfun))[1]
T = transformparamsoperator(S, D(2.0, 2.0), N, weighted=false)
u = Fun(D(2.0, 2.0), sparse(bihar) \ (T * errfun.coefficients))
# Create arrays of (x,y) points
n = 500
x = LinRange(0, 1, n)
y = LinRange(-1, 1, n)
sln = zeros(length(x), length(y))
for i = 1:length(x)
    for j = 1:length(y)
        val = evalu(u, [x[i]; y[j]])
        sln[j,i] = val
    end
    @show i
end
PyPlot.clf()
w, h = plt[:figaspect](2.0)
PyPlot.figure(figsize=(w,h))
PyPlot.pcolor(x, y, sln)
savefig("experiments/solution-biharmonic")

#=
Plotting the norms of each block of coeffs for solutions of Poisson for
different RHSs
=#
N = 200
S
f1 = Fun((x,y)->1.0, S)
f2 = Fun((x,y)->x^2 + y^2 - 1, S)
f3 = Fun((x,y)->x^2*y^2*(1-x^2-y^2)^2, S)
u1 = Fun(S, sparse(Δw) \ resizecoeffs!(f1, N))
u2 = Fun(S, sparse(Δw) \ resizecoeffs!(f2, N))
u3 = Fun(S, sparse(Δw) \ resizecoeffs!(f3, N))
u4 = Fun(S, sparse(Δw) \ f4cfs)
u1coeffs = PseudoBlockArray(u1.coefficients, [i+1 for i=0:N])
u2coeffs = PseudoBlockArray(u2.coefficients, [i+1 for i=0:N])
u3coeffs = PseudoBlockArray(u3.coefficients, [i+1 for i=0:N])
u4coeffs = PseudoBlockArray(u4.coefficients, [i+1 for i=0:N])
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
using Plots
Plots.plot(u1norms, label="f(x,y) = 1", xscale=:log10, yscale=:log10, legend=:bottomleft)
Plots.plot!(u2norms, label="f(x,y) = x^2 + y^2 - 1")
Plots.plot!(u3norms, label="f(x,y) = x^2 * y^2 * (1-x^2-y^2)^2")
Plots.plot!(u4norms, label = "f(x,y) = exp(-1000((x-0.2)^2+(y-0.2)^2)+(y+0.2)^2))")
Plots.xlabel!("Block")
Plots.ylabel!("Norm")
Plots.savefig("experiments/solutionblocknorms")






#====#

# Model Problem: Δ(u*w)(x,y) = f(x,y) in Ω=halfdisk; u(x,y) ≡ 0 on ∂Ω.
#   where w(x,y) = x*(1-x^2-y^2) is the weight of the D(1.0,1.0) basis.

# 1) f(x,y) = -8x => u(x,y) ≡ 1
N = 1 # degree of f
c = rand(1)[1]; f = Fun((x,y)->-c*8x, S)
resizecoeffs!(f, N)
u = Fun(S, sparse(Δw) \ resizecoeffs!(f, N))
@test u(z) ≈ c # Result u(x,y) where Δ(u*w)(x,y) = f(x,y)
# plot(x->u(x,y))
# plot(y->u(x,y))

# 2) f(x,y) = 2 - 12xy - 14x^2 - 2y^2 => u(x,y) = x + y
U = Fun((x,y)->x+y, S)
N = 2 # degree of f
f = Fun((x,y)->(2 - 12x*y - 14x^2 - 2y^2), S)
u = Fun(S, sparse(Δw) \ resizecoeffs!(f, N))
@test u(z) ≈ U(z)
u.coefficients

# 3) f(x,y) = y*exp(x)*[2-11x-6x^2-x^3-2y^2-xy^2] => u(x,y) = y*exp(x)
U = Fun((x,y)->y*exp(x), S)
N = 12 # degree of f
m = Int((N+1)*(N+2))
f = Fun((x,y)->y*exp(x)*(2-11x-6x^2-x^3-2y^2-x*y^2), S, m)
u = Fun(S, sparse(Δw) \ f.coefficients[1:size(Δ)[1]])
@test u(z) ≈ U(z)
u(z) - U(z)



#====#

#=
Model problem: Helmholtz eqn:
    Δ(u(x,y)) + k²u(x,y) = f(x,y) in Ω
    u(x,y) = c on ∂Ω
where k is the wavenumber, u is the amplitude at a point (x,y) on the halfdisk,
and W(x,y) is the P^{(1,1)} weight.

i.e. with constant non-zero BCs

We frame this as u = v + c where v satisfies
    Δ(v(x,y)) + k²v(x,y) = f(x,y) - k²c
    v(x,y) = 0 on ∂Ω
=#
k, c = rand(2)

# 1) f(x,y) = k²c - 8αx + W(x,y)*α*k^2
#   => v(x,y) = W(x,y)*α
#   => u(x,y) = W(x,y)*α + c
α = rand(1)[1]
f = Fun((x,y)->(- 8α * x + x^Float64(S.a) * (1-x^2-y^2)^Float64(S.b) * α * k^2), S)
n = 5
C = transformparamsoperatorsquare(S, (S.family)(S.a-1, S.b-1), n, weighted=true)
T = transformparamsoperator((S.family)(S.a-1, S.b-1), S, n)
vc = T * C * (sparse(Δw[1:getopindex(n,n), 1:getopindex(n,n)] + k^2 * T * C) \ resizecoeffs!(f, n))
uc = vc + [c; zeros(length(vc)-1)]
u = Fun(S, uc)
@test u(z) ≈ OrthogonalPolynomialFamilies.weight(S, z) * α + c

# 2) f(x,y) = k²c + W(x,y)*k²(x+y) + (2 - 12xy - 14x² - 2y²)
#   => v(x,y) = W(x,y) * (x + y)
#   => u(x,y) = W(x,y) * (x + y) + c
U = Fun((x,y)->OrthogonalPolynomialFamilies.weight(S, x, y) * (x+y) + c, S)
n = 5 # degree of f
f = Fun((x,y)->(OrthogonalPolynomialFamilies.weight(S, x, y)*(x+y)*k^2 + 2 - 12x*y - 14x^2 - 2y^2), S)
C = transformparamsoperatorsquare(S, (S.family)(S.a-1, S.b-1), n, weighted=true)
T = transformparamsoperator((S.family)(S.a-1, S.b-1), S, n)
vc = T * C * (sparse(Δw[1:getopindex(n,n), 1:getopindex(n,n)] + k^2 * T * C) \ resizecoeffs!(f, n))
uc = vc + [c; zeros(length(vc)-1)]
u = Fun(S, uc)
@test u(z) ≈ U(z)




#====#
#=
Model problem: Heat eqn
    ∂u/∂t(x,y) = Δ(W*u)(x,y)
=>  (Back Euler)
    u1 - u0 = h * Δ(W*u1)
=>  (I - hΔ)\u0 = u1
=#
using Plots
a, b = 1.0, 1.0; D = HalfDiskFamily(); S = D(a, b)
x, y = 0.4, -0.2; z = [x; y] # Test point
h = 1e-2
N = 15
Δ = laplacesquare(D, N)
T = increaseparamsoperator((S.family)(S.a-1, S.b-1), N)
C = convertweightedtononoperatorsquare(S, N)
u0 = Fun((x,y)->sin(x+y)*y, S, 200)
u0.coefficients

û = OrthogonalPolynomialFamilies.resizecoeffs!(u0, N)
plt = plot()
maxits = 100
res = []
nplots = 5
for it = 1:maxits
    u = T * C * (sparse(T * C - h * Δ) \ û)
    if round(nplots*it/maxits) == nplots*it/maxits
        plot!(plt, abs.(u), label="t=$it")
    end
    append!(res, maximum(abs.(u-û)))
    global û = copy(u)
end
res
plt
savefig("experiments/example2timesteppingcovnvergence")


#===#

# NOTE: call getopptseval() before this function
function recβ2(S::HalfDiskSpace{<:Any, <:Any, T}, pts, w, n, k, j) where T
    # We get the norms of the 2D OPs
    m = Int((n+2)*(n+3) / 2)
    if length(S.opnorms) < m
        getopnorms(S, m)
    end

    H1 = (S.family.H)(S.a, S.b + k - 0.5)
    H2 = (S.family.H)(S.a, S.b + k + 0.5)
    H3 = (S.family.H)(S.a, S.b + k + 1.5)
    P = (S.family.P)(S.b, S.b)
    δ = getopnorm(P) * (isodd(j) ? OrthogonalPolynomialFamilies.recγ(T, P, k+1) : OrthogonalPolynomialFamilies.recβ(T, P, k+1))

    if j == 1
        val = inner2(H2, opevalatpts(H2, n-k+1, pts),
                        (-pts.^2 .+ 1).^k .* opevalatpts(H1, n-k+1, pts), w)
        val * δ / S.opnorms[getopindex(n-1, k-1)]
    elseif j == 2
        val = inner2(H2, opevalatpts(H2, n-k+1, pts),
                        (-pts.^2 .+ 1).^(k+1) .* opevalatpts(H3, n-k-1, pts), w)
        val * δ / S.opnorms[getopindex(n-1, k+1)]
    elseif j == 3
        val = inner2(H2, opevalatpts(H2, n-k+1, pts),
                        (-pts.^2 .+ 1).^k .* opevalatpts(H1, n-k+2, pts), w)
        val * δ / S.opnorms[getopindex(n, k-1)]
    elseif j == 4
        val = inner2(H2, opevalatpts(H2, n-k+1, pts),
                        (-pts.^2 .+ 1).^(k+1) .* opevalatpts(H3, n-k, pts), w)
        val * δ / S.opnorms[getopindex(n, k+1)]
    elseif j == 5
        val = inner2(H2, opevalatpts(H2, n-k+1, pts),
                        (-pts.^2 .+ 1).^k .* opevalatpts(H1, n-k+3, pts), w)
        val * δ / S.opnorms[getopindex(n+1, k-1)]
    elseif j == 6
        val = inner2(H2, opevalatpts(H2, n-k+1, pts),
                        (-pts.^2 .+ 1).^(k+1) .* opevalatpts(H3, n-k+1, pts), w)
        val * δ / S.opnorms[getopindex(n+1, k+1)]
    else
        error("Invalid entry to function")
    end
end
function resizedata2!(S::HalfDiskSpace, N)
    # N is the max degree of the OPs
    N₀ = length(S.B)
    N ≤ N₀ - 2 && return S
    pts, w = pointswithweights(S.family.H(S.a, S.b+0.5), N+1)
    for k = 0:N+1
        getopptseval(S.family.H(S.a, S.b+k+0.5), N, pts)
    end
    resize!(S.A, N + 2)
    resize!(S.B, N + 2)
    resize!(S.C, N + 2)
    resize!(S.DT, N + 2)
    getBs!(S, N, N₀)
    getCs!(S, N, N₀)
    getAs!(S, N, N₀)
    getDTs!(S, N, N₀)
    S
end


#===#
# General 2D Family/Space code

import Base: in

# R should be Float64
abstract type AbstractOrthogonalPolynomialFamily2D{R} end

struct OrthogonalPolynomialDomain2D{R} <: Domain{SVector{2,R}} end

OrthogonalPolynomialDomain2D() = OrthogonalPolynomialDomain2D{Float64}()

struct OrthogonalPolynomialSpace2D{FAM, R, FA, F} <: Space{OrthogonalPolynomialDomain2D{R}, R}
    family::FAM # Pointer back to the family
    a::R # OP parameter
    b::R # OP parameter
    opnorms::Vector{R} # squared norms
    A::Vector{SparseMatrixCSC{R}} # Storage for matrices for clenshaw
    B::Vector{SparseMatrixCSC{R}}
    C::Vector{SparseMatrixCSC{R}}
    DT::Vector{SparseMatrixCSC{R}}
end

# TODO
function OrthogonalPolynomialSpace2D(fam::AbstractOrthogonalPolynomialFamily2D{R},
                                        a::R, b::R) where R
    OrthogonalPolynomialSpace2D{typeof(fam), typeof(a)}(fam, a, b,
            Vector{R}(), Vector{SparseMatrixCSC{R}}(),
            Vector{SparseMatrixCSC{R}}(), Vector{SparseMatrixCSC{R}}(),
            Vector{SparseMatrixCSC{R}}())
end

# TODO
in(x::SVector{2}, D::OrthogonalPolynomialDomain2D) =
    (0 ≤ x[1] ≤ 1 && -sqrt(1-x[1]^2) ≤ x[2] ≤ sqrt(1-x[1]^2))

spacescompatible(A::OrthogonalPolynomialSpace2D, B::OrthogonalPolynomialSpace2D) =
    (A.a == B.a && A.b == B.b)

# R should be Float64
struct OrthogonalPolynomialFamily2D{R, IN, FN, OPF} <: AbstractOrthogonalPolynomialFamily2D{R}
    α::IN
    β::IN
    γ::IN
    ρ::Fun
    w1a::Fun
    w1b::Fun
    w2::Fun
    H::OPF
    P::OPF
    spaces::Dict{NTuple{2,R}, OrthogonalPolynomialSpace2D}
end

function OrthogonalPolynomialFamily2D(α, β, γ, ρ::Fun, w1a::Fun,
                                        w1b::Fun, w2::Fun)
    R = Float64
    spaces = Dict{NTuple{2,R}, OrthogonalPolynomialSpace2D}()
    H = OrthogonalPolynomialFamilies.OrthogonalPolynomialFamily(w1a, w1b, ρ)
    P = OrthogonalPolynomialFamilies.OrthogonalPolynomialFamily(w2)
    OrthogonalPolynomialFamily2D{R, typeof(α), typeof(ρ), typeof(H)}(α, β, γ, ρ, w1a, w1b,
                                                            w2, H, P, spaces)
end

function (D::OrthogonalPolynomialFamily2D{R})(a::R, b::R) where R
    haskey(D.spaces,(a,b)) && return D.spaces[(a,b)]
    D.spaces[(a,b)] = OrthogonalPolynomialSpace2D(D, a, b)
end

(D::OrthogonalPolynomialFamily2D{R})(a, b) where R = D(convert(R,a), convert(R,b))

α = 0
β = 1
γ = 1
X = Fun(identity, α..β)
Y = Fun(identity, -γ..γ)
ρ = sqrt(1-X^2)
w1a = X
w1b = (1-X^2)
w2 = (1-Y^2)
D = OrthogonalPolynomialFamily2D(α, β, γ, ρ, w1a, w1b, w2)

R = Float64
spaces = Dict{NTuple{2,R}, OrthogonalPolynomialSpace2D}()
H = OrthogonalPolynomialFamilies.OrthogonalPolynomialFamily(w1a, w1b, ρ)
P = OrthogonalPolynomialFamilies.OrthogonalPolynomialFamily(w2)
OrthogonalPolynomialFamily2D{R, typeof(α), typeof(ρ), typeof(H)}(α, β, γ, ρ, w1a, w1b,
                                                        w2, H, P, spaces)
