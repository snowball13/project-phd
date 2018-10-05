using ApproxFun
    import ApproxFun: evaluate, PolynomialSpace, recα, recβ, recγ, recA, recB, recC, domain,
                        domainspace, rangespace, bandinds, lanczos
    import Base: getindex
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

domain(S::OrthogonalPolynomialSpace) =
    domain(S.weight)


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
    spaces::Dict{NTuple{N,R}, OrthogonalPolynomialSpace{<:Any,WW,D,R}}
end

function OrthogonalPolynomialFamily(w::Vararg{Fun{<:Space{D,R}},N}) where {D,R,N}
    all(domain.(w) .== Ref(domain(first(w)))) || throw(ArgumentError("domains incompatible"))
    WW =  typeof(prod(w.^0.5))
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

# Obtain quad rule for the weight W^{a,b}(x,y) = x^a * (1-x^2-y^2)^b
function quadrule(N, a, b)
    # Return the weights and nodes to use for the even and odd components
    # of a function, i.e. for the half disk Ω:
    #   int_Ω W^{a,b}(x,y) f(x,y) dydx ≈ Σ_j [weⱼ*fe(xⱼ,yⱼ) + woⱼ*fo(xⱼ,yⱼ)]

    # NOTE: the odd part of the quad rule will equal 0, so can be ignored.

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
            wo[i + (k - 1)N] = wso[k] * sqrt(1 - so[k]^2) * wt[i] * t[i]
        end
    end
    xe, ye, we, xo, yo, wo
end

# integral of f(x,y) over the half disk with weight
# W^{a,b}(x,y) = x^a * (1-x^2-y^2)^b
function halfdiskintegral(f, N, a, b)
    xe, ye, we, xo, yo, wo  = quadrule(N, a, b)
    return (sum(we .* (f.(xe, ye) + f.(xe, -ye)) / 2)
            + sum(wo .* (f.(xo, yo) - f.(xo, -yo)) / 2))
end

function gethalfdiskOP(H, P, ρ, n, k, a, b)
    H̃ = Fun(H(a, b+k+0.5), [zeros(n-k); 1])
    P̃ = Fun(P(b, b), [zeros(k); 1])
    return (x,y) -> (H̃(x) * ρ(x)^k * P̃(y / ρ(x)))
end

# Return the coefficients of the expansion of a function f(x,y) in the OP basis
# {P^{a,b}_{n,k}(x,y)}
function fun2coeffs(f, N, a, b)
    # Coeffs are obtained by inner products, approximated (or exact if low
    # enough degree polynomial) using the quad rule above
    X = Fun(identity, 0..1)
    Y = Fun(identity, -1..1)
    ρ = sqrt(1-X^2)
    H = OrthogonalPolynomialFamily(X, (1-X^2))
    P = OrthogonalPolynomialFamily(1+Y, 1-Y)
    c = zeros(sum(1:N+1))
    for n = 0:N
        for k = 0:n
            Q = gethalfdiskOP(H, P, ρ, n, k, a, b)
            ff = (x,y) -> (f(x,y) * Q(x,y))
            QQ = (x,y) -> (Q(x,y) * Q(x,y))
            c[n+k+1] = halfdiskintegral(ff, N+1, a, b) / halfdiskintegral(QQ, N+1, a, b)
        end
    end
    c
end


# Testing
N = 5; a = 0.5; b = 0.5
X = Fun(identity, 0..1)
Y = Fun(identity, -1..1)
ρ = sqrt(1-X^2)
H = OrthogonalPolynomialFamily(X, (1-X^2))
P = OrthogonalPolynomialFamily(1+Y, 1-Y)
c = zeros(sum(1:N+1))
f = (x,y)-> x + y^2
j = 1
for n = 0:N
    for k = 0:n
        Q = gethalfdiskOP(H, P, ρ, n, k, a, b)
        ff = (x,y) -> (f(x,y) * Q(x,y))
        QQ = (x,y) -> (Q(x,y) * Q(x,y))
        global c[j] = halfdiskintegral(ff, N+1, a, b) / halfdiskintegral(QQ, N+1, a, b)
        global j += 1
    end
end
c

out = 0.0; x = 0.3; y = 0.6
for n = 0:N
    for k = 0:n
        Q = gethalfdiskOP(H, P, ρ, n, k, a, b)
        global out += c[n+k+1] * Q(x,y)
    end
end
out
x = 0.3; y = 0.6
f = (x,y)-> x + y^2
f(x, y)



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




## Tests

## Jacobi example
x = Fun()
P = OrthogonalPolynomialFamily(1+x, 1-x)
a, b = 0.4, 0.2
@test P(a,b).weight(0.1) ≈ (1+0.1)^a * (1-0.1)^b
for n = 0:5
    P₅ = Fun(Jacobi(a,b), [zeros(n); 1])
    P₅ = P₅ * sqrt(sum((1+x)^a*(1-x)^b))/sqrt(sum((1+x)^a*(1-x)^b*P₅^2))
    P̃₅ = Fun(P(a,b), [zeros(n); 1])
    @test P̃₅(0.1) ≈ P₅(0.1)
end

x = Fun(0..1)
H = OrthogonalPolynomialFamily(x, 1-x^2)

#====#

# Test golubwelsch()
function g(a, b, x0, x1)
    # return the integral of x^a * (1-x^2)^b on the interval [x0, x1]
    return (x1^(a+1) * _₂F₁(-b, (a + 1) / 2, (a + 3) / 2, x1^2) / (a + 1)
            - x0^(a+1) * _₂F₁(-b, (a + 1) / 2, (a + 3) / 2, x0^2) / (a + 1))
end
N = 3; a = 0.5; b = 0.5; d = 5
f = x->(x^d)
S = Fun(identity, 0..1)
ω = S^a * (1 - S^2)^(b+0.5)
x, w = golubwelsch(ω, N)
feval = 0.0
for j = 1:N
    global feval += w[j] * f(x[j])
end
@test feval ≈ g(a + d, b + 0.5, 0, 1)



# Test quadrule
N = 4; a = 0.5; b = 0.5
xe, ye, we, xo, yo, wo  = quadrule(N, a, b)
# Even f
f = (x,y)-> x + y^2
@test sum(we .* f.(xe, ye)) ≈ (g(0, b, -1+1e-15, 1-1e-15) * g(a+1, b+0.5, 0, 1)
                                + g(2, b, -1+1e-15, 1-1e-15) * g(a, b+1.5, 0, 1))
# Odd f
f = (x,y)-> x*y^3
@test sum(wo .* f.(xo, yo)) < 1e-12 # Should be zero
