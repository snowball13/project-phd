using ApproxFun
    import ApproxFun: evaluate, PolynomialSpace, recα, recβ, recγ, recA, recB, recC, domain,
                        domainspace, rangespace, bandwidths, lanczos, prectype, canonicaldomain, tocanonical,
                        spacescompatible, points, transform, itransform, AbstractProductSpace, tensorizer,
                        columnspace
    import Base: getindex, in
using StaticArrays
using FastGaussQuadrature
using LinearAlgebra
using SingularIntegralEquations
using Test


# Finds the OPs and recurrence for weight w, having already found N₀ OPs
function lanczos!(w, P, β, γ, N₀=0)

    # x * P[n](x) == (γ[n] * P[n+1](x) + β[n] * P[n](x) + γ[n-1] * P[n-1](x))

    N = length(β)
    x = Fun(identity,space(w))

    if N₀ <= 0
        N₀ = 1
        f1 = Fun(1/sqrt(sum(w)),space(x))
        P[1] = f1
        v = x*P[1]
        β[1] = sum(w*v*P[1])
        v = v - β[1]*P[1]
        γ[1] = sqrt(sum(w*v^2))
        P[2] = v/γ[1]
    end

    for k = N₀+1:N
        v = x*P[k] - γ[k-1]*P[k-1]
        β[k] = sum(w*v*P[k])
        v = v - β[k]*P[k]
        γ[k] = sqrt(sum(w*v^2))
        P[k+1] = v/γ[k]
    end

    P,β,γ
end

# Finds the OPs and recurrence for weight w
function lanczos(w, N)
    # x * P[n](x) == (γ[n] * P[n+1](x) + β[n] * P[n](x) + γ[n-1] * P[n-1](x))
    P = Array{Fun}(undef, N + 1)
    β = Array{eltype(w)}(undef, N)
    γ = Array{eltype(w)}(undef, N)
    lanczos!(w, P, β, γ)
end


abstract type SpaceFamily{D,R} end

struct OrthogonalPolynomialSpace{F,WW,D,R,FN} <: PolynomialSpace{D,R}
    family::F # Pointer back to the family
    weight::WW # The full product weight
    a::Vector{R} # Diagonal recurrence coefficients
    b::Vector{R} # Off diagonal recurrence coefficients
    ops::Vector{FN} # Cache the Funs given back from lanczos
end

domain(S::OrthogonalPolynomialSpace) = domain(S.weight)
canonicaldomain(S::OrthogonalPolynomialSpace) = domain(S)
tocanonical(S::OrthogonalPolynomialSpace, x) = x


OrthogonalPolynomialSpace(fam::SpaceFamily{D,R}, w::Fun{<:Space{D,R}}) where {D,R,N} =
    OrthogonalPolynomialSpace{typeof(fam),typeof(w),D,R,Fun}(fam, w, Vector{R}(), Vector{R}(), Vector{Fun}())

function resizedata!(S::OrthogonalPolynomialSpace, n)
    N₀ = length(S.a) - 1
    n ≤ N₀ + 1 && return S
    resize!(S.a, n)
    resize!(S.b, n)
    resize!(S.ops, n + 1)
    S.ops[:], S.a[:], S.b[:] = lanczos!(S.weight, S.ops, S.a, S.b, N₀)
    S
end

# R is range-type, which should be Float64.
struct OrthogonalPolynomialFamily{FF,WW,D,R,N} <: SpaceFamily{D,R}
    factors::FF
    spaces::Dict{NTuple{N,R}, OrthogonalPolynomialSpace}
end

function OrthogonalPolynomialFamily(w::Vararg{Fun{<:Space{D,R}},N}) where {D,R,N}
    all(domain.(w) .== Ref(domain(first(w)))) || throw(ArgumentError("domains incompatible"))
    WW =  Fun
    spaces = Dict{NTuple{N,R}, OrthogonalPolynomialSpace{OrthogonalPolynomialFamily{typeof(w),WW,D,R,N},WW,D,R}}()
    OrthogonalPolynomialFamily{typeof(w),WW,D,R,N}(w, spaces)
end

function (P::OrthogonalPolynomialFamily{<:Any,<:Any,<:Any,R,N})(α::Vararg{R,N}) where {R,N}
    haskey(P.spaces,α) && return P.spaces[α]
    P.spaces[α] = OrthogonalPolynomialSpace(P, prod(P.factors.^α))
end


#####
# recα/β/γ are given by
#       x p_{n-1} = γ_n p_{n-2} + α_n p_{n-1} +  p_n β_n ???
#           Should it be: x p_{n} = γ_n p_{n-1} + α_n p_{n} +  β_n p_{n+1} (dont think so)
#####
recα(::Type{T}, S::OrthogonalPolynomialSpace, n) where T =
    T(resizedata!(S, n).a[n])
recβ(::Type{T}, S::OrthogonalPolynomialSpace, n) where T =
    T(resizedata!(S, n).b[n])
recγ(::Type{T}, S::OrthogonalPolynomialSpace, n) where T =
    T(resizedata!(S, n).b[n-1])

#####
# recA/B/C are given by
#       p_{n+1} = (A_n x + B_n)p_n - C_n p_{n-1}
#####
recA(::Type{T}, S::OrthogonalPolynomialSpace, n) where T =
    1 / recβ(T, S, n+1)
recB(::Type{T}, S::OrthogonalPolynomialSpace, n) where T =
    -recα(T, S, n+1) / recβ(T, S, n+1)
recC(::Type{T}, S::OrthogonalPolynomialSpace, n) where T =
    recγ(T, S, n+1) / recβ(T, S, n+1)


# Returns weights and nodes for N-point quad rule for given weight
function golubwelsch( ω::Fun, N::Integer )
    # Golub--Welsch algorithm. Used here for N<=20.
    β = zeros(N); γ = zeros(N)
    p, β[:], γ[:] = lanczos(ω, N)       # 3-term recurrence
    J = SymTridiagonal(β, γ[1:end-1])     # Jacobi matrix
    D, V = eigen(J)                     # Eigenvalue decomposition
    indx = sortperm(D)                  # Hermite points
    μ = sum(ω)                          # Integral of weight function
    w = μ * V[1, indx].^2               # quad rule weights to output
    x = D[indx]                         # quad rule nodes to output
    # # Enforce symmetry:
    # ii = floor(Int, N/2)+1:N
    # x = x[ii]
    # w = w[ii]
    return x, w
end

golubwelsch(sp::OrthogonalPolynomialSpace, N) = golubwelsch(sp.weight, N)

spacescompatible(A::OrthogonalPolynomialSpace, B::OrthogonalPolynomialSpace) =
    A.weight ≈ B.weight

points(S::OrthogonalPolynomialSpace, n) = golubwelsch(S, n)[1]

# Creates a Vandermonde matrix by evaluating the basis at the grid
function spvandermonde(S::OrthogonalPolynomialSpace, n)
    pts = points(S, n)
    V = Array{Float64}(undef, n, n)
    for k = 1:n
        V[:, k] = Fun(S, [zeros(k-1); 1]).(pts)
    end
    V
end

# Inputs: OP space, f(pts) for desired f
# Output: Coeffs of the function f for its expansion in the OPSpace OPs
function transform(S::OrthogonalPolynomialSpace, vals)
    n = length(vals)
    spvandermonde(S, n) \ vals
end

# Inputs: OP space, coeffs of a function f for its expansion in the OPSpace OPs
# Output: vals = {f(x_j)} where x_j are are the points(S,n)
function itransform(S::OrthogonalPolynomialSpace, cfs)
    n = length(cfs)
    spvandermonde(S, n) * cfs
end


#=====#
# Half Disk

# Obtain quad rule for the weight W^{a,b}(x,y) = x^a * (1-x^2-y^2)^b
function halfdiskquadrule(N, a, b)
    # Return the weights and nodes to use for the even and odd components
    # of a function, i.e. for the half disk Ω:
    #   int_Ω W^{a,b}(x,y) f(x,y) dydx ≈ Σ_j [weⱼ*fe(xⱼ,yⱼ) + woⱼ*fo(xⱼ,yⱼ)]

    # NOTE: the odd part of the quad rule will equal 0 for polynomials, so can be ignored.

    S = Fun(identity, 0..1)
    T = Fun(identity, -1..1)
    ωt = (1 - T^2)^b
    t, wt = golubwelsch(ωt, N)
    ωse = S^a * (1 - S^2)^(b + 0.5)
    se, wse = golubwelsch(ωse, N)
    ωso = S^a * (1 - S^2)^b
    so, wso = golubwelsch(ωso, N)

    # Return the nodes and weights as single vectors
    xe = zeros(N^2)
    ye = zeros(N^2)
    we = zeros(N^2)
    xo = zeros(N^2)
    yo = zeros(N^2)
    wo = zeros(N^2)
    for i = 1:N
        for k = 1:N
            xe[i + (k - 1)N] = se[k]
            ye[i + (k - 1)N] = t[i] * sqrt(1 - se[k]^2)
            we[i + (k - 1)N] = wse[k] * wt[i]
            xo[i + (k - 1)N] = so[k]
            yo[i + (k - 1)N] = t[i] * sqrt(1 - so[k]^2)
            wo[i + (k - 1)N] = wso[k] * sqrt(1 - so[k]^2) * wt[i]
        end
    end
    xe, ye, we, xo, yo, wo
end

# integral of f(x,y) over the half disk with weight
# W^{a,b}(x,y) = x^a * (1-x^2-y^2)^b
function halfdiskintegral(f, N, a, b)
    xe, ye, we, xo, yo, wo = halfdiskquadrule(N, a, b)
    return sum(we .* (f.(xe, ye) + f.(xe, -ye))) / 2
    # return (sum(we .* (f.(xe, ye) + f.(xe, -ye)) / 2)
    #         + sum(wo .* (f.(xo, yo) - f.(xo, -yo)) / 2))
end

function gethalfdiskOP(H, P, ρ, n, k, a, b)
    H̃ = Fun(H(a, b+k+0.5), [zeros(n-k); 1])
    P̃ = Fun(P(b, b), [zeros(k); 1])
    return (x,y) -> (H̃(x) * ρ(x)^k * P̃(y / ρ(x)))
end

# Return the coefficients of the expansion of a function f(x,y) in the OP basis
# {P^{a,b}_{n,k}(x,y)}
function halfdiskfun2coeffs(f, N, a, b)
    # Coeffs are obtained by inner products, approximated (or exact if low
    # enough degree polynomial) using the quad rule above
    X = Fun(identity, 0..1)
    Y = Fun(identity, -1..1)
    ρ = sqrt(1-X^2)
    H = OrthogonalPolynomialFamily(X, (1-X^2))
    P = OrthogonalPolynomialFamily(1+Y, 1-Y)
    c = zeros(sum(1:N+1))
    j = 1
    for n = 0:N
        for k = 0:n
            Q = gethalfdiskOP(H, P, ρ, n, k, a, b)
            ff = (x,y) -> (f(x,y) * Q(x,y))
            QQ = (x,y) -> (Q(x,y) * Q(x,y))
            c[j] = halfdiskintegral(ff, N+1, a, b) / halfdiskintegral(QQ, N+1, a, b)
            j += 1
        end
    end
    c
end

struct HalfDisk{D,T} <: Domain{SVector{2,T}} end

struct HalfDiskSpace{SV,D,T} <: AbstractProductSpace{SV,D,T}
    spaces::SV
    a::T # Power of the "x" factor in the weight
    b::T # Power of the "(1-x^2-y^2)" factor in the weight
    α::Vector{T} # Diagonal recurrence coefficients
    β::Vector{T} # Off diagonal recurrence coefficients
end # TODO

function HalfDiskSpace(a::Float64, b::Float64)
    sps = (Chebyshev(0..1),Chebyshev(-1..1))
    d = typeof(mapreduce(domain,*,sps))
    HalfDiskSpace{typeof(sps),typeof(d),typeof(a)}((Chebyshev(0..1), Chebyshev(-1..1)), a, b, Vector{typeof(a)}(), Vector{typeof(a)}())
end
HalfDiskSpace() = HalfDiskSpace(0.5, 0.5)

in(x::SVector{2}, d::HalfDisk) = 0 ≤ x[1] ≤ 1 && -sqrt(1-x[1]^2) ≤ x[2] ≤ sqrt(1-x[1]^2)

domain(S::HalfDiskSpace) = mapreduce(domain,*,S.spaces)

spacescompatible(A::HalfDiskSpace, B::HalfDiskSpace) = (A.a == B.a && A.b == B.b)

tensorizer(S::HalfDiskSpace) = ApproxFun.Tensorizer(map(ApproxFun.blocklengths,S.spaces))

# every column is in the same space for a TensorSpace
# TODO: remove
columnspace(S::HalfDiskSpace, _) = S.spaces[1]

# canonicaldomain(S::HalfDiskSpace) = domain(S)
# tocanonical(S::HalfDiskSpace, x) = x

function points(S::HalfDiskSpace, n)
    m = Int(cld(-3+sqrt(1+8n),2)) + 1
    if iseven(m)
        mt = Int(m / 2)
        ms = m + 1
    else
        mt = Int((m + 1) / 2)
        ms = m
    end
    ñ = Int((m+1)m/2)
    pts = Vector{SArray{Tuple{2},Float64,1,2}}(undef, ñ)
    a = S.a; b = S.b
    X̃ = Fun(identity, 0..1)
    Ỹ = Fun(identity, -1..1)
    ωt = (1 - Ỹ^2)^b
    t, wt = golubwelsch(ωt, mt)
    ωs = X̃^a * (1 - X̃^2)^(b + 0.5)
    s, ws = golubwelsch(ωs, ms)
    for i = 1:mt
        for k = 1:ms
            pts[i + (k - 1)mt] = s[k], t[i] * sqrt(1 - s[k]^2)
        end
    end
    pts
end # TODO

# Creates a Vandermonde matrix by evaluating the basis at the grid
function spvandermonde(S::HalfDiskSpace, n)
    pts = points(S, n)
    V = Array{Float64}(undef, n, n)
    for k = 1:n
        V[:, k] = Fun(S, [zeros(k-1); 1]).(pts)
    end
    V
end

# Inputs: OP space, f(pts) for desired f
# Output: Coeffs of the function f for its expansion in the OPSpace OPs
function transform(S::HalfDiskSpace, vals)
    n = length(vals)
    spvandermonde(S, n) \ vals
end

# Inputs: OP space, coeffs of a function f for its expansion in the OPSpace OPs
# Output: vals = {f(x_j)} where x_j are are the points(S,n)
function itransform(S::HalfDiskSpace, cfs)
    n = length(cfs)
    spvandermonde(S, n) * cfs
end

#
# n = 16
# f = (x, y) -> exp(x)
# S = Chebyshev()^2
# S.spaces
# pts = points(S, n)
# m = Int(cld(-3+sqrt(1+8n),2)) + 1; ñ = Int((m+1)m/2)
# T = Float64
# cfs = transform(S, T[f(x...) for x in pts])
# F = Fun(S, cfs)
# x = (0.1, 0.2)
# typeof(F.space)
# @which evaluate(F.coefficients,F.space,ApproxFun.Vec(x...))
# ApproxFun.totensor(F.space,F.coefficients)
# @which ApproxFun.tensorizer(F.space)
# ApproxFun.Tensorizer(map(ApproxFun.blocklengths,F.space.spaces))
# F.space.spaces
# ApproxFun.columnspace(S, 3)
#
# n = 6; a = 0.5; b = -0.5
# S = HalfDiskSpace(a, b)
# pts = points(S, n)
# f = (x, y) -> exp(x + y)
# tensorizer(S)
# columnspace(S, 2)
# T = Float64
# transform(S, T[f(x...) for x in pts])
#
# F = Fun(S, [zeros(1); 1])
# F.(pts)
# x = 0.1, 0.2
# typeof(F.space)
# evaluate(F.coefficients,F.space,ApproxFun.Vec(x...))
# f(x...)
# ff = Fun(f, S)
# ff(x...) - f(x...)


#====#

## Tests

function g(a, b, x0, x1)
    # return the integral of x^a * (1-x^2)^b on the interval [x0, x1]
    return (x1^(a+1) * _₂F₁(-b, (a + 1) / 2, (a + 3) / 2, x1^2) / (a + 1)
            - x0^(a+1) * _₂F₁(-b, (a + 1) / 2, (a + 3) / 2, x0^2) / (a + 1))
end

@testset "Evaluation" begin
    x = Fun()
    P = OrthogonalPolynomialFamily(1+x, 1-x)
    a, b = 0.4, 0.2
    @test P(a,b).weight(0.1) ≈ (1+0.1)^a * (1-0.1)^b
    w = sqrt(sum((1+x)^a*(1-x)^b))
    for n = 0:5
        P₅ = Fun(Jacobi(a,b), [zeros(n); 1])
        P₅ = P₅ * w / sqrt(sum((1+x)^a*(1-x)^b*P₅^2))
        P̃₅ = Fun(P(a,b), [zeros(n); 1])
        @test P̃₅(0.1) ≈ P₅(0.1)
    end
end

@testset "Golub–Welsch" begin
    x = Fun()
    P = OrthogonalPolynomialFamily(1+x, 1-x)
    a, b = 0.4, 0.2
    @test all(golubwelsch(P(a,b), 10) .≈ gaussjacobi(10, b,a))

    x = Fun(0..1)
    H = OrthogonalPolynomialFamily(x, 1-x^2)
    N = 10; t,w = golubwelsch(H(a,b), N) # accurate for degree 2N - 1
    f = Fun(Chebyshev(0..1), randn(2N)) # a random degree 2N-1 polynomial
    @test sum(x^a*(1-x^2)^b * f) ≈ w'f.(t)

    N = 3; a = 0.5; b = 0.5; d = 5
    f = x->(x^d)
    S = Fun(identity, 0..1)
    ω = S^a * (1 - S^2)^(b+0.5)
    x, w = golubwelsch(ω, N)
    @test w'*f.(x) ≈ g(a + d, b + 0.5, 0, 1)
end

@testset "Quad Rule" begin
    N = 4; a = 0.5; b = 0.5
    xe, ye, we, xo, yo, wo  = halfdiskquadrule(N, a, b)
    # Even f
    f = (x,y)-> x + y^2
    @test sum(we .* f.(xe, ye)) ≈ (g(0, b, -1+1e-15, 1-1e-15) * g(a+1, b+0.5, 0, 1)
                                    + g(2, b, -1+1e-15, 1-1e-15) * g(a, b+1.5, 0, 1))
    # Odd f
    f = (x,y)-> x*y^3
    @test sum(wo .* f.(xo, yo)) < 1e-12 # Should be zero
end

@testset "Fun expansion in OP basis" begin
    N = 5; a = 0.5; b = 0.5
    X = Fun(identity, 0..1)
    Y = Fun(identity, -1..1)
    ρ = sqrt(1-X^2)
    H = OrthogonalPolynomialFamily(X, (1-X^2))
    P = OrthogonalPolynomialFamily(1+Y, 1-Y)
    f = (x,y)-> x^2 * y^2 + y^4 * x
    c = halfdiskfun2coeffs(f, N, a, b)
    x = 0.6; y = -0.3
    @test f(x, y) ≈ c'*[gethalfdiskOP(H, P, ρ, n, k, a, b)(x,y) for n = 0:N for k = 0:n] # zero for N >= deg(f)
end

@testset "Transform" begin
    n = 20; a = 0.5; b = -0.5
    X = Fun(identity, -1..1)
    P = OrthogonalPolynomialFamily(1-X, 1+X)
    S = P(b, b)
    pts = points(S, n)
    f = x->exp(x)
    vals = f.(pts)
    cfs = transform(S, vals)
    ff = Fun(S, cfs)
    x = 0.4
    @test ff(x) ≈ f(x)
    @test vals ≈ itransform(S, cfs)
end

@testset "Creating function in an OrthogonalPolynomialSpace" begin
    n = 10; a = 0.5; b = -0.5
    X = Fun(identity, 0..1)
    H = OrthogonalPolynomialFamily(X, (1-X^2))
    S = H(a, b)
    f = x -> exp(x)
    F = Fun(f, S)
    x = 0.1567
    @test F(x) ≈ f(x)
end # NOTE: Fun(f, S) is a λ function, and not a Fun

#====#


a,b
a = b = 0.0
n,k= 2,2; m,j=1,1;
    P1 =gethalfdiskOP(H, P, ρ, n, k, a, b)
    P2 = gethalfdiskOP(H, P, ρ, m, j, a, b)
    halfdiskintegral((x,y) -> P1(x,y) * P2(x,y),
                20, a, b)


X = Fun(identity, 0..1)
H = OrthogonalPolynomialFamily(X, (1-X^2))
sum(Fun(x -> Fun(H(0.,0.), [0.,1.])(x), 0..1))
plot(Fun(x -> Fun(H(0.,0.), [0.,1.])(x), 0..1))

H₂ = OrthogonalPolynomialFamily(1+Y, 1-((Y+1)/2)^2)
plot(Fun(x -> Fun(H₂(0.,0.), [0.,1.])(x)))


H(0.,0.).a[1]-0.5


evaluate([0.,1.], H(0.,0.), 0.5)
ApproxFun.tocanonical(H(0.,0.), 0.5)


Fun(H(0.,0.), [0.,1.])(0.1)

ApproxFun.canonicaldomain(H(0.,0.))

lanczos(H(0.,0.).weight,3)

t, w= golubwelsch(H(0.,0.).weight,3)

f = Fun(Chebyshev(0..1), randn(6))
w'*f.(t) - sum(f)
H(0.,0.).a




H(0.,0.).b

H.spaces

plot(H(0.,0.).weight)



P1(0.1,0.2)
sum(Fun(x -> Fun(H(a,b+0.5),[0,1.])(x), 0..1) * sqrt(1-X^2))



P1(0.1,0.2)
H̃ = Fun(H(a, b+k+0.5), [zeros(n-k); 1])

plot(Fun(x -> Fun(H(a,b), [0.0,0,1])(x), 0..1))

X = Fun(identity, 0..1)
Y = Fun(identity, -1..1)

using Plots
Fun(x -> Fun(H(a,b), [0.,0,0,1])(x)) |> sum

H(a,b).weight |> plot
α = (a,b)
OrthogonalPolynomialSpace(P, prod(P.factors.^α)).weight |> plot

prod(H.factors.^α) |> typeof
prod(H.factors.^(0.5,0.5)) |> typeof


H.spaces


H(a,b)
a,b


plot(Fun(x -> Fun(H(a,b), [0,0.0,0,0,0,1])(x)))


a = b
plot(H(a,b).weight)



Fun(H(a,b), [0,0,0,0,0,1])(0.1)
Fun(H(a,b), [0,0,0,0,0,1])(-0.1)

a,b

(1+Y)^(0.1)

using Plots

Y |> space
roots((1-(Y+1)/2)^2)


P1


a = b = 0;






#=====#


# struct OrthogonalPolynomialDerivative{T} <: Derivative{T}
#     space::OrthogonalPolynomialSpace
# end
#
# domainspace(D::OrthogonalPolynomialDerivative) = D.space
# rangespace(D::OrthogonalPolynomialDerivative) = D.space.family(D.space.α .+ 1)
# bandwidths(D::OrthogonalPolynomialDerivative) = (1,?)
# getindex(D::OrthogonalPolynomialDerivative, k::Int, j::Int) = ?
#
# Derivative(sp::OrthogonalPolynomialSpace, k) = (@assert k==1; OrthogonalPolynomialDerivative(sp))
#
#
#
# D = Derivative(Jacobi(a,b))
# @which D[2,3]
#
# bandwidths(D)
