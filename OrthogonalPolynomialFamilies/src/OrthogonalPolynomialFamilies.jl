module OrthogonalPolynomialFamilies

using ApproxFun
    import ApproxFun: evaluate, PolynomialSpace, recα, recβ, recγ, recA, recB, recC, domain,
                        domainspace, rangespace, bandwidths, prectype, canonicaldomain, tocanonical,
                        spacescompatible, points, transform, itransform, AbstractProductSpace, tensorizer,
                        columnspace
    import Base: getindex, in
using StaticArrays
using FastGaussQuadrature
using LinearAlgebra
# using SingularIntegralEquations
using Test

export OrthogonalPolynomialFamily, HalfDisk, HalfDiskSpace


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
pointswithweights(S::OrthogonalPolynomialSpace, n) = golubwelsch(S, n)

# Inputs: OP space, f(pts) for desired f
# Output: Coeffs of the function f for its expansion in the OPSpace OPs
function transform(S::OrthogonalPolynomialSpace, vals)
    n = length(vals)
    pts, w = pointswithweights(S, n)
    # Vandermonde matrix transposed, including weights and normalisations
    Ṽ = Array{Float64}(undef, n, n)
    for k = 0:n-1
        pk = Fun(S, [zeros(k); 1])
        nrm = sum([pk(pts[j])^2 * w[j] for j = 1:n])
        Ṽ[k+1, :] = pk.(pts) .* w / nrm
    end
    Ṽ * vals
end

# Inputs: OP space, coeffs of a function f for its expansion in the OPSpace OPs
# Output: vals = {f(x_j)} where x_j are are the points(S,n)
function itransform(S::OrthogonalPolynomialSpace, cfs)
    n = length(cfs)
    pts, w = pointswithweights(S, n)
    # Vandermonde matrix
    V = Array{Float64}(undef, n, n)
    for k = 0:n-1
        pk = Fun(S, [zeros(k); 1])
        V[:, k+1] = pk.(pts)
    end
    V * cfs
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

struct HalfDisk{T} <: Domain{SVector{2,T}} end

struct HalfDiskSpace{T, FA, F} <: Space{HalfDisk{T}, T}
    a::T # Power of the "x" factor in the weight
    b::T # Power of the "(1-x^2-y^2)" factor in the weight
    H::FA # OPFamily in [0,1]
    P::FA # OPFamily in [-1,1]
    ρ::F # Fun of sqrt(1-X^2) in [0,1]
    opnorms::Vector{T}
end # TODO

function HalfDiskSpace(a::T, b::T) where T
    X = Fun(identity, 0..1)
    Y = Fun(identity, -1..1)
    H = OrthogonalPolynomialFamily(X, 1-X^2)
    P = OrthogonalPolynomialFamily(1+Y, 1-Y)
    ρ = sqrt(1 - X^2)
    HalfDiskSpace{typeof(a), typeof(H), typeof(ρ)}(a, b, H, P, ρ, Vector{T}())
end
HalfDiskSpace() = HalfDiskSpace(0.5, 0.5)

in(x::SVector{2}, D::HalfDisk) = 0 ≤ x[1] ≤ 1 && -sqrt(1-x[1]^2) ≤ x[2] ≤ sqrt(1-x[1]^2)

spacescompatible(A::HalfDiskSpace, B::HalfDiskSpace) = (A.a == B.a && A.b == B.b)

# NOTE n refers to the max degree of the OPs we use (i.e. the degree of f)
function pointswithweights(S::HalfDiskSpace, n)
    pts = Vector{SArray{Tuple{2},Float64,1,2}}(undef, 2(n^2))
    x, y, w = halfdiskquadrule(n, S.a, S.b)[1:3]
    for j = 1:n^2
        pts[j] = x[j], y[j]
        pts[n^2 + j] = x[j], -y[j]
    end
    pts, w
end
points(S::HalfDiskSpace, n) = pointswithweights(S, n)[1]

function halfdisknorm(f::Fun, S::HalfDiskSpace, pts, w)
    n = Int(length(pts)/2)
    sum(([f(pt) for pt in pts[1:n]].^2 + [f(pt) for pt in pts[n+1:end]].^2) .* w) / 2
end

# Inputs: OP space, f(pts) for desired f
# Output: Coeffs of the function f for its expansion in the OPSpace OPs
function transform(S::HalfDiskSpace, vals)
    N = Int(sqrt(length(vals) / 2)) # degree + 1
    N2 = N^2
    m = Int((N+1)N/2)

    pts, w = pointswithweights(S, N)

    # We store the norms of the OPs
    m̃ = length(S.opnorms)
    if m̃ < m
        resize!(S.opnorms, m)
        for k = m̃+1:m
            Pnk = Fun(S, [zeros(k-1); 1])
            S.opnorms[k] = halfdisknorm(Pnk, S, pts, w)
        end
    end

    # Vandermonde matrices transposed, for each set of pts (x, ±y)
    VTp = Array{Float64}(undef, m, N2)
    VTm = copy(VTp)
    for k = 1:m
        Pnk = Fun(S, [zeros(k-1); 1])
        VTp[k, :] = Pnk.(pts[1:N2])
        VTm[k, :] = Pnk.(pts[N2+1:end])
    end
    W = Diagonal(w)
    U = Diagonal(1 ./ S.opnorms[1:m])
    U * (VTp * W * vals[1:N2] + VTm * W * vals[N2+1:end]) / 2
end

# Inputs: OP space, coeffs of a function f for its expansion in the OPSpace OPs
# Output: vals = {f(x_j)} where x_j are are the points(S,n)
function itransform(S::HalfDiskSpace, cfs)
    m = length(cfs)
    N = Int(round(0.5 * (-1 + sqrt(1 + 8m))))
    N2 = N^2
    pts = points(S, N)
    V = Array{Float64}(undef, 2N2, m)
    for k = 1:m
        Pnk = Fun(S, [zeros(k-1); 1])
        V[:, k] = Pnk.(pts)
    end
    V * cfs
end # TODO

function evaluate(cfs::AbstractVector, S::HalfDiskSpace, z)
    #=
    TODO: Implement clenshaw for the half disk?

    This is just a quick and dirty sum.
    =#
    m = length(cfs)
    ret = 0.0
    n = 0; k = 0
    for j = 1:m
        # Pnk = Fun(S, [zeros(j-1); 1])
        Pnk = gethalfdiskOP(S.H, S.P, S.ρ, n, k, S.a, S.b)
        ret += cfs[j] * Pnk(z...)
        k += 1
        if k > n
            n += 1
            k = 0
        end
    end
    ret
end # TODO

end # module
