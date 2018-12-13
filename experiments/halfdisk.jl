using OrthogonalPolynomialFamilies, ApproxFun
using StaticArrays, LinearAlgebra, Test, Profile, BlockArrays,
        BlockBandedMatrices, SparseArrays
using Makie

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
Δ = OrthogonalPolynomialFamilies.laplace(D, N-1)
u = Δ \ f.coefficients[1:size(Δ)[1]]
res = abs.(u - append!(û, zeros(size(u)[1]-size(û)[1])))
plt = plot(res, yaxis=:log)
û = copy(u)
for N = 3:10 # degree of estimation
    Δ = OrthogonalPolynomialFamilies.laplace(D, N-1)
    u = Δ \ f.coefficients[1:size(Δ)[1]]
    res = abs.(u - append!(û, zeros(size(u)[1]-size(û)[1])))
    plot!(plt, res)
    global û = copy(u)
end
plt
savefig("example4coeffs")


#====#
#=
Model problem: Helmholtz eqn:
    Δ(W*u)(x,y) + k²u(x,y) = -f(x,y)
where k is the wavenumber, u is the amplitude at a point (x,y) on the halfdisk,
and W(x,y) is the P^{(1,1)} weight.
=#
a, b = 1.0, 1.0; D = HalfDiskFamily(); S = D(a, b)
wavenumberoperator(N, k) = sparse([sparse(I,sum(1:N),sum(1:N)) * k^2; zeros(N+1, sum(1:N))])
k = rand(1)[1]

# 1) f(x,y) = 8cx - ck^2 => u(x,y) ≡ c
c = rand(1)[1]; f = Fun((x,y)->(8*c*x - c*k^2), S)
N = 2
K = wavenumberoperator(N, k)
Δ = OrthogonalPolynomialFamilies.laplace(D, N-1)
uc = sparse(Δ + K) \ (-OrthogonalPolynomialFamilies.resizecoeffs!(f, N))
u = Fun(S, uc)
@test u(z) ≈ c

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
a, b = 1.0, 1.0; D = HalfDiskFamily(); S = D(a, b)
x, y = 0.4, -0.2; z = [x; y] # Test point
h = 1e-3
N = 5
Δ = OrthogonalPolynomialFamilies.laplacesquare(D, N-1)
û = OrthogonalPolynomialFamilies.resizecoeffs!(Fun((x,y)->x, S), N)
plt = plot()
maxits = 1000
nplots = 10
for it = 1:1000
    u = sparse(I - h * Δ) \ û
    if round(nplots*it/maxits) == nplots*it/maxits
        # plot!(plt, abs.(u-û))
    end
    global û = copy(u)
end
for it = 1001:maxits+1001
    u = sparse(I - h * Δ) \ û
    if round(nplots*it/maxits) == nplots*it/maxits
        plot!(plt, abs.(u-û))
    end
    global û = copy(u)
end
plt
u = Fun(S, uc)
u(z)

function operatorclenshawG(n, Jx, Jy)
    G = Matrix{SparseMatrixCSC{Float64}}(undef, 2(n+1), n+1)
    for i = 1:n+1
        for j = 1:n+1
            if i == j
                G[i,j] = Jx
                G[i+n+1,j] = Jy
            else
                G[i,j] = zeros(size(Jx))
                G[i+n+1,j] = zeros(size(Jy))
            end
        end
    end
    G
end
function converttooperatorclenshawmatrix(A, Jx)
    nn = size(Jx)
    B = Array{SparseMatrixCSC{Float64}}(undef, size(A))
    for ij = 1:length(A)
        B[ij] = sparse(I, nn) * A[ij]
    end
    B
end

function operatorclenshaw(cfs, S::HalfDiskSpace)
    # TODO: ρ(Jx) doesnt work, how to implement (i.e. get operator version of P_1)
    m̃ = length(cfs)
    N = -1 + Int(round(sqrt(1+2(m̃-1))))
    OrthogonalPolynomialFamilies.resizedata!(S, N+1)
    m = Int((N+1)*(N+2)/2)
    Jx = OrthogonalPolynomialFamilies.jacobix(S, N)
    Jy = OrthogonalPolynomialFamilies.jacobiy(S, N)
    if m̃ < m
        resize!(cfs, m)
        cfs[m̃+1:end] .= 0.0
    end
    P0 = 1.0
    id = sparse(I, size(Jx))
    if N == 0
        return cfs[1] * id * P0
    end
    ρx = S.ρ(z[1])
    P1 = converttooperatorclenshawmatrix([Fun(S.H(S.a, S.b+0.5), [0, 1])(z[1]);
          ρx * Fun(S.P(S.b, S.b), [0, 1])(z[2]/ρx)], Jx)
    if N == 1
        return cfs[1] * P0 + dot(view(cfs, 2:3), P1)
    end
    inds2 = m-N:m
    γ2 = converttooperatorclenshawmatrix(view(cfs, inds2), Jx)'
    inds1 = (m-2N):(m-N-1)
    γ1 = (converttooperatorclenshawmatrix(view(cfs, inds1), Jx)'
        - γ2 * converttooperatorclenshawmatrix(S.DT[N], Jx) * (converttooperatorclenshawmatrix(S.B[N], Jx)
                                                                - operatorclenshawG(N-1, Jx, Jy)))
    for n = N-2:-1:1
        ind = sum(1:n)
        γ = (converttooperatorclenshawmatrix(view(cfs, ind+1:ind+n+1), Jx)'
             - γ1 * converttooperatorclenshawmatrix(S.DT[n+1], Jx) * (converttooperatorclenshawmatrix(S.B[n+1], Jx) - operatorclenshawG(n, Jx, Jy))
             - γ2 * converttooperatorclenshawmatrix(S.DT[n+2] * S.C[n+2], Jx))
        γ2 = copy(γ1)
        γ1 = copy(γ)
    end
    cfs[1] * P0 * id, (gg[1] * P1[1] + gg[2] * P1[2]), - sparse((P0 * γ2 * converttooperatorclenshawmatrix(S.DT[2] * S.C[2], Jx))[1])
    # cfs[1] * P0 + γ1 * P1 - (P0 * γ2 * S.DT[2] * S.C[2])[1]
end

cfs = Fun((x,y)->x*y, S).coefficients
aa, bb, cc = operatorclenshaw(cfs, S)


#===#
# General 2D Family/Space code

# R should be Float64
abstract type AbstractOrthogonalPolynomialFamily2{R} end

struct OrthogonalPolynomialDomain2D{R} <: Domain{SVector{2,R}} end

HalfDisk() = HalfDisk{Float64}()

struct OrthogonalPolynomialSpace2D{FAM, R, FA, F} <: Space{OrthogonalPolynomialDomain2D{R}, R}
    family::FAM # Pointer back to the family
    a::R # OP parameter
    b::R # OP parameter
    H::FA # OPFamily in [xlim[1],xlim[2]]
    P::FA # OPFamily in [-max(ρ(x)),max(ρ(x))] for x in [xlim[1],xlim[2]]
    ρ::F # Fun of sqrt(1-X^2) in [0,1]
    opnorms::Vector{R} # squared norms
    A::Vector{SparseMatrixCSC{R}} # Storage for matrices for clenshaw
    B::Vector{SparseMatrixCSC{R}}
    C::Vector{SparseMatrixCSC{R}}
    DT::Vector{SparseMatrixCSC{R}}
end

#TODO
function (fam::OrthogonalPolynomialSpace2D{R})(α::R, β::R, ρ::Fun) where R
    X = Fun(identity, 0..1)
    Y = Fun(identity, -1..1)
    H = OrthogonalPolynomialFamily(X, 1-X^2)
    P = OrthogonalPolynomialFamily(1+Y, 1-Y)
    ρ = sqrt(1 - X^2)
    OrthogonalPolynomialSpace2D{typeof(fam), typeof(a), typeof(H), typeof(ρ)}(fam,
            a, b, H, P, ρ, Vector{R}(), Vector{SparseMatrixCSC{R}}(),
            Vector{SparseMatrixCSC{R}}(), Vector{SparseMatrixCSC{R}}(),
            Vector{SparseMatrixCSC{R}}())
end
OrthogonalPolynomialSpace2D() = OrthogonalPolynomialSpace2D(0.5, 0.5)

in(x::SVector{2}, D::OrthogonalPolynomialSpace2D) =
    (0 ≤ x[1] ≤ 1 && -sqrt(1-x[1]^2) ≤ x[2] ≤ sqrt(1-x[1]^2))

spacescompatible(A::OrthogonalPolynomialSpace2D, B::OrthogonalPolynomialSpace2D) =
    (A.a == B.a && A.b == B.b)

# R should be Float64
struct OrthogonalPolynomialFamily2D{R} <: AbstractOrthogonalPolynomialFamily2D{R}
    spaces::Dict{NTuple{2,R}, HalfDiskSpace}
end

function OrthogonalPolynomialFamily2D()
    WW = Fun
    R = Float64
    spaces = Dict{NTuple{2,R}, HalfDiskSpace}()
    OrthogonalPolynomialFamily2D{R}(spaces)
end

function (D::OrthogonalPolynomialFamily2D{R})(a::R, b::R) where R
    haskey(D.spaces,(a,b)) && return D.spaces[(a,b)]
    D.spaces[(a,b)] = HalfDiskSpace(D, a, b)
end

(D::OrthogonalPolynomialFamily2D{R})(a, b) where R = D(convert(R,a), convert(R,b))
