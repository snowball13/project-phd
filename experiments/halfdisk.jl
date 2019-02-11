using OrthogonalPolynomialFamilies, ApproxFun
using Test, Profile
using StaticArrays, LinearAlgebra, BlockArrays, BlockBandedMatrices,
        SparseArrays
# using Makie

import OrthogonalPolynomialFamilies: points, pointswithweights, getopptseval,
                    opevalatpts, inner2, laplace, getopindex, getnk, resizecoeffs!,
                    convertweightedtononoperator, increaseparamsoperator,
                    transformoperator, laplacesquare, convertweightedtononoperatorsquare

#===#
# Speed up the code! Add storage of the basis evaluated at the points from
# points(), and use in a new basiseval method (using 3-term recurrence)

a, b = 1.0, 1.0; D = HalfDiskFamily(); S = D(a, b)
x, y = 0.4, -0.2; z = [x; y]
N = 10
laplace(D, N)

N = 30
S = D(a, b)
D
@time W = OrthogonalPolynomialFamilies.weightedpartialoperatorx(S, N)
@time W = OrthogonalPolynomialFamilies.weightedpartialoperatory(S, N)
@time A = OrthogonalPolynomialFamilies.partialoperatorx(S, N+2)
@time A = OrthogonalPolynomialFamilies.partialoperatory(S, N+2)
S = D(1, 0)
@time T = transformoperator(S, N+1)
S = D(0, 1)
@time T = transformoperator(S, N+1)
S = D(0, 2)
@time T = transformoperator(S, N+1)

D = HalfDiskFamily(); N = 15
@time laplace(D, N)

#====#
#=
Sparsity of laplace operator
=#
using PyPlot
a, b = 1.0, 1.0; D = HalfDiskFamily(); S = D(a, b)
N = 30
Δ = laplacesquare(D, N)
DDD = copy(Δ)
PyPlot.clf()
PyPlot.spy(Array(sparse(Δ)))
PyPlot.axis(xmin=0, xmax=500, ymin=500, ymax=0)
PyPlot.title("Sparsity of Δ operator")
savefig("experiments/sparsityoflaplacian")

#=
Solution of Δu = f, where f(x,y) = 1 + erf(5(1 - 10((x - 0.5)^2 + (y)^2)))
                        # f = y*exp(x)*(2-11x-6x^2-x^3-2y^2-x*y^2)
=#
using SpecialFunctions
using Makie, FileIO
evalu(u, z) = isindomain(z, u.space.family) ? u(z) : 0.0
isindomain(x, D::HalfDiskFamily) = 0 ≤ x[1] ≤ 1 && -sqrt(1-x[1]^2) ≤ x[2] ≤ sqrt(1-x[1]^2)
function getvalidpoint()
    x, y = rand(1)[1], rand(1)[1]
    while true
        if isindomain([x, y])
            break
        else
            x, y = rand(1)[1], rand(1)[1]
        end
    end
    x, y
end

N = 30
# Δ = laplacesquare(D, N)
m = 500
f = Fun((x,y)->(1 + erf(5(1 - 10((x - 0.5)^2 + (y)^2)))), S, m)
f.coefficients
# u = Fun(S, sparse(Δ) \ f.coefficients[1:size(Δ)[1]])
u = Fun(S, sparse(Δ) \ resizecoeffs!(f, N))

# Create arrays of (x,y) points
n = 100
x = LinRange(0, 1, n)
y = LinRange(-1, 1, n)
sln = zeros(length(x), length(y))
for i = 1:length(x)
    for j = 1:length(y)
        val = evalu(u, [x[i]; y[j]])
        sln[i,j] = OrthogonalPolynomialFamilies.weight(S, x[i], y[j]) * val
    end
end
sln
scene = Scene()
s = Makie.image(x, y, sln, colormap = :viridis)
n = 20
θ = [0;(0.5:n-0.5)/(2n);0.5]
x = [cospi(θ) for θ in θ]
y = [sinpi(θ) for θ in θ]
pts = vec(Point2f0.(x, y))
Makie.lines!(s, pts, linewidth = 2, color = :black)
pts = vec(Point2f0.(x, -y))
Makie.lines!(s, pts, linewidth = 2, color = :black)
FileIO.save("experiments/solution.png", s)


#=
Plotting the norms of each block of coeffs for solutions of Poisson for
different RHSs
=#
m = 800
f4 = Fun((x,y)->exp(-1000((x-0.2)^2+(y-0.2)^2)), S, m)
N = getnk(length(f4.coefficients))[1]
Δc = Δ[1:getopindex(N, N), 1:getopindex(N, N)] # laplacesquare(D, N)
f1 = Fun((x,y)->1.0, S)
f2 = Fun((x,y)->x^2 + y^2 - 1, S)
f3 = Fun((x,y)->x^2*y^2*(1-x^2-y^2)^2, S)
u1 = Fun(S, sparse(Δc) \ resizecoeffs!(f1, N))
u2 = Fun(S, sparse(Δc) \ resizecoeffs!(f2, N))
u3 = Fun(S, sparse(Δc) \ resizecoeffs!(f3, N))
u4 = Fun(S, sparse(Δc) \ f4.coefficients)
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
# scene = Scene()
# s = lines(f1norms, xaxis=:log)# , color=:blue)
# lines!(f2norms, color=:red)
# lines!(f3norms, color=:green)
# lines!(f4norms, color=:purple)
using Plots
Plots.plot(u1norms, label="f(x,y) = 1", xscale=:log10, yscale=:log10, legend=:bottomleft)
Plots.plot!(u2norms, label="f(x,y) = x^2 + y^2 - 1")
Plots.plot!(u3norms, label="f(x,y) = x^2 * y^2 * (1-x^2-y^2)^2")
Plots.plot!(u4norms, label = "f(x,y) = exp(-1000((x-0.2)^2+(y-0.2)^2))")
Plots.xlabel!("Block")
Plots.ylabel!("Norm")
savefig("experiments/solutionblocknorms")






#====#

# Model Problem: Δ(u*w)(x,y) = f(x,y) in Ω=halfdisk; u(x,y) ≡ 0 on ∂Ω.
#   where w(x,y) = x*(1-x^2-y^2) is the weight of the D(1.0,1.0) basis.
a, b = 1.0, 1.0; D = HalfDiskFamily(); S = D(a, b)
x, y = 0.4, -0.2; z = [x; y] # Test point

# 1) f(x,y) = -8x => u(x,y) ≡ 1
N = 1 # degree of f
c = rand(1)[1]; f = Fun((x,y)->-c*8x, S)
resizecoeffs!(f, N)
Δ = laplacesquare(D, N)
u = Fun(S, sparse(Δ) \ resizecoeffs!(f, N))
@test u(z) ≈ c # Result u(x,y) where Δ(u*w)(x,y) = f(x,y)
# plot(x->u(x,y))
# plot(y->u(x,y))

# 2) f(x,y) = 2 - 12xy - 14x^2 - 2y^2 => u(x,y) = x + y
U = Fun((x,y)->x+y, S)
N = 2 # degree of f
f = Fun((x,y)->(2 - 12x*y - 14x^2 - 2y^2), S)
Δ = laplacesquare(D, N)
u = Fun(S, sparse(Δ) \ resizecoeffs!(f, N))
@test u(z) ≈ U(z)
u.coefficients

# 3) f(x,y) = y*exp(x)*[2-11x-6x^2-x^3-2y^2-xy^2] => u(x,y) = y*exp(x)
U = Fun((x,y)->y*exp(x), S)
N = 12 # degree of f
m = Int((N+1)*(N+2))
f = Fun((x,y)->y*exp(x)*(2-11x-6x^2-x^3-2y^2-x*y^2), S, m)
Δ = laplacesquare(D, N)
u = Fun(S, sparse(Δ) \ f.coefficients[1:size(Δ)[1]])
@test u(z) ≈ U(z)
u(z) - U(z)

# 4) f(x,y) = 2zcos(x) - 8xsin(x) - xzsin(x) - 4x^2cos(x) - 12xy => u(x,y) = sin(x) + y
# (where z = 1-x^2-y^2)
using PyPlot
Uc = resizecoeffs!(Fun((x,y)->sin(x) + y, S), 15)
m = 200
f = Fun((x,y)->(2*(1-x^2-y^2)*cos(x) - 8*x*sin(x) - x*(1-x^2-y^2)*sin(x) - 4*x^2*cos(x) - 12*x*y),
        S, m)
resizecoeffs!(f, 30)
N = 2
Δ = laplace(D, N-1)
u = Fun(S, Δ \ f.coefficients[1:size(Δ)[1]])
res = abs.(resizecoeffs!(u, 15) - Uc)
plt = plot(res, yaxis=:log, label="N=$N", xlim=[0,125], ylim=[1e-20,1])
for N = 4:2:15
    Δ = laplace(D, N-1)
    u = Fun(S, Δ \ f.coefficients[1:size(Δ)[1]])
    res = abs.(resizecoeffs!(u, 15) - Uc)
    plot!(plt, res, label="N=$N")
end
plt
title("A sinusoidally modulated sinusoid")
savefig("experiments/example4coeffs")


#====#
#=
Model problem: Helmholtz eqn:
    Δ(u(x,y)) + k²u(x,y) = -f(x,y)
where k is the wavenumber, u is the amplitude at a point (x,y) on the halfdisk,
and W(x,y) is the P^{(1,1)} weight.
=#
a, b = 1.0, 1.0; D = HalfDiskFamily(); S = D(a, b)
k = rand(1)[1]

# 1) f(x,y) = 8cx - W(x,y)*c*k^2 => u(x,y) = W(x,y)*c
c = rand(1)[1]; f = Fun((x,y)->(8c * x - x^(S.a) * (1-x^2-y^2)^(S.b) * c * k^2), S)
N = 3
C = convertweightedtononoperatorsquare(S, N)
T = increaseparamsoperator((S.family)(S.a-1, S.b-1), N)
Δ = laplacesquare(D, N)
uc = - T * C * (sparse(Δ + k^2 * T * C) \ resizecoeffs!(f, N))
u = Fun(S, uc)
@test u(z) ≈ OrthogonalPolynomialFamilies.weight(S, z)*c

# 2) f(x,y) = - W(x,y)*k^2(x+y) - (2 - 12xy - 14x^2 - 2y^2) => u(x,y) = W(x,y) * (x + y)
U = Fun((x,y)->OrthogonalPolynomialFamilies.weight(S, x, y) * (x+y), S)
N = 5 # degree of f
f = Fun((x,y)->(-U(x,y)*k^2 - 2 + 12x*y + 14x^2 + 2y^2), S)
f.coefficients
C = convertweightedtononoperatorsquare(S, N)
T = increaseparamsoperator(D(S.a-1, S.b-1), N)
Δ = laplacesquare(D, N)
uc = - T * C * (sparse(Δ + k^2 * T * C) \ resizecoeffs!(f, N))
u = Fun(S, uc)
@test u(z) ≈ U(z)

# 2b) f(x,y) = - W(x,y)*k^2(x+y) - (2 - 12xy - 14x^2 - 2y^2) => u(x,y) = W(x,y) * (x + y)
U = Fun((x,y)->(x+y), S) # No need to convert back to non-weighted in this example
N = 5 # degree of f
f = Fun((x,y)->(-OrthogonalPolynomialFamilies.weight(S, x, y) * U(x,y)*k^2 - 2 + 12x*y + 14x^2 + 2y^2), S)
f.coefficients
C = convertweightedtononoperatorsquare(S, N)
T = increaseparamsoperator(D(S.a-1, S.b-1), N)
Δ = laplacesquare(D, N)
uc = - (sparse(Δ + k^2 * T * C) \ resizecoeffs!(f, N))
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
