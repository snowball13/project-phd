using OrthogonalPolynomialFamilies, ApproxFun
using Test, Profile
using StaticArrays, LinearAlgebra, BlockArrays, BlockBandedMatrices,
        SparseArrays
# using Makie

import OrthogonalPolynomialFamilies: points, pointswithweights, getopptseval,
                    opevalatpts, inner2, laplace, getopindex, getnk

#===#
# Speed up the code! Add storage of the basis evaluated at the points from
# points(), and use in a new basiseval method (using 3-term recurrence)

a, b = 1.0, 1.0; D = HalfDiskFamily(); S = D(a, b)
N = 15
@time W = OrthogonalPolynomialFamilies.weightedpartialoperatorx(S, N)
@time W = OrthogonalPolynomialFamilies.weightedpartialoperatory(S, N)
@time A = OrthogonalPolynomialFamilies.partialoperatorx(S, N+2)
@time A = OrthogonalPolynomialFamilies.partialoperatory(S, N+2)
S = D(1, 0)
@time T = OrthogonalPolynomialFamilies.transformoperator(S, N+1)
S = D(0, 1)
@time T = OrthogonalPolynomialFamilies.transformoperator(S, N+1)
S = D(0, 2)
@time T = OrthogonalPolynomialFamilies.transformoperator(S, N+1)

D = HalfDiskFamily(); N = 15
@time laplace(D, N)



#====#

# Model Problem: Δ(u*w)(x,y) = f(x,y) in Ω=halfdisk; u(x,y) ≡ 0 on ∂Ω.
#   where w(x,y) = x*(1-x^2-y^2) is the weight of the D(1.0,1.0) basis.
a, b = 1.0, 1.0; D = HalfDiskFamily(); S = D(a, b)
x, y = 0.4, -0.2; z = [x; y] # Test point

# 1) f(x,y) = -8x => u(x,y) ≡ 1
N = 1 # degree of f
c = rand(1)[1]; f = Fun((x,y)->-c*8x, S)
OrthogonalPolynomialFamilies.resizecoeffs!(f, N)
Δ = OrthogonalPolynomialFamilies.laplace(D, N-1)
Δ \ f.coefficients
u = Fun(S, Δ \ f.coefficients)
@test u(z) ≈ c # Result u(x,y) where Δ(u*w)(x,y) = f(x,y)
# plot(x->u(x,y))
# plot(y->u(x,y))

# 2) f(x,y) = 2 - 12xy - 14x^2 - 2y^2 => u(x,y) = x + y
U = Fun((x,y)->x+y, S)
N = 2 # degree of f
f = Fun((x,y)->(2 - 12x*y - 14x^2 - 2y^2), S)
Δ = OrthogonalPolynomialFamilies.laplace(D, N-1)
u = Fun(S, Δ \ OrthogonalPolynomialFamilies.resizecoeffs!(f, N))
@test u(z) ≈ U(z)

# 3) f(x,y) = y*exp(x)*[2-11x-6x^2-x^3-2y^2-xy^2] => u(x,y) = y*exp(x)
U = Fun((x,y)->y*exp(x), S)
N = 7 # degree of f
m = Int((N+1)*(N+2))
f = Fun((x,y)->y*exp(x)*(2-11x-6x^2-x^3-2y^2-x*y^2), S, m)
Δ = OrthogonalPolynomialFamilies.laplace(D, N-1)
u = Fun(S, Δ \ f.coefficients[1:size(Δ)[1]])
@test u(z) ≈ U(z) atol=10.0^(-N)

# 4) f(x,y) = 2zcos(x) - 8xsin(x) - xzsin(x) - 4x^2cos(x) - 12xy => u(x,y) = sin(x) + y
# (where z = 1-x^2-y^2)
using Plots
Uc = Fun((x,y)->sin(x), S).coefficients
m = 200
f = Fun((x,y)->(2*(1-x^2-y^2)*cos(x) - 8*x*sin(x) - x*(1-x^2-y^2)*sin(x) - 4*x^2*cos(x) - 12*x*y),
        S, m)
N = 2
û = zeros(1)
Δ = laplace(D, N-1)
u = Δ \ f.coefficients[1:size(Δ)[1]]
res = abs.(u - append!(û, zeros(size(u)[1]-size(û)[1])))
plt = plot(res, yaxis=:log, label="N=$N")
û = copy(u)
for N = 3:10 # degree of estimation
    Δ = laplace(D, N-1)
    u = Δ \ f.coefficients[1:size(Δ)[1]]
    res = abs.(u - append!(û, zeros(size(u)[1]-size(û)[1])))
    plot!(plt, res, label="N=$N")
    global û = copy(u)
end
plt
savefig("experiments/example4coeffs")


#====#
#=
Model problem: Helmholtz eqn:
    Δ(W*u)(x,y) + k²(W*u)(x,y) = -f(x,y)
where k is the wavenumber, u is the amplitude at a point (x,y) on the halfdisk,
and W(x,y) is the P^{(1,1)} weight.
=#
a, b = 1.0, 1.0; D = HalfDiskFamily(); S = D(a, b)
wavenumberoperator(N, k) = sparse([sparse(I,sum(1:N),sum(1:N)) * k^2; zeros(N+1, sum(1:N))])
k = rand(1)[1]

# 1) f(x,y) = 8cx - W(x,y)*c*k^2 => u(x,y) ≡ c
c = rand(1)[1]; f = Fun((x,y)->(8*c*x - x^(S.a) * (1-x^2-y^2)^(S.b)*c*k^2), S)
N = 4
C = OrthogonalPolynomialFamilies.convertweightedtononoperator(S, N)
Δ = OrthogonalPolynomialFamilies.laplacesquare(D, N-1)
uc = - C * (sparse(Δ + C*k^2) \ OrthogonalPolynomialFamilies.resizecoeffs!(f, N))
u = Fun(S, uc)
@show u(z)
c

# 2) f(x,y) = -k^2(x+y) - (2 - 12xy - 14x^2 - 2y^2) => u(x,y) = x + y
U = Fun((x,y)->x+y, S)
N = 2 # degree of f
f = Fun((x,y)->(-(x+y)*k^2 - 2 + 12x*y + 14x^2 + 2y^2), S)
Δ = OrthogonalPolynomialFamilies.laplace(D, N-1)
K = wavenumberoperator(N, k)
u = Fun(S, - sparse(Δ + K) \ OrthogonalPolynomialFamilies.resizecoeffs!(f, N))
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
N = 5
Δ = OrthogonalPolynomialFamilies.laplacesquare(D, N-1)
û = OrthogonalPolynomialFamilies.resizecoeffs!(Fun((x,y)->x*(1-x^2)*y, S), N)
plt = plot()
maxits = 100
nplots = 5
for it = 1:99
    u = sparse(I - h * Δ) \ û
    if round(nplots*it/maxits) == nplots*it/maxits
        # plot!(plt, abs.(u-û))
    end
    global û = copy(u)
end
for it = 100:maxits+100
    u = sparse(I - h * Δ) \ û
    if round(nplots*it/maxits) == nplots*it/maxits
        plot!(plt, abs.(u), label="t=$it")
    end
    global û = copy(u)
end
plt
savefig("experiments/example2timesteppingcovnvergence")
U = Fun(S, û)
û
U(z)

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
