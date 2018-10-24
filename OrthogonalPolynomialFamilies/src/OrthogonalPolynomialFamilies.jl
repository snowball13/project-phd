module OrthogonalPolynomialFamilies

using ApproxFun
    import ApproxFun: evaluate, PolynomialSpace, recα, recβ, recγ, recA, recB, recC, domain,
                        domainspace, rangespace, bandwidths, prectype, canonicaldomain, tocanonical,
                        spacescompatible, points, transform, itransform, AbstractProductSpace, tensorizer,
                        columnspace, checkpoints, plan_transform
    import Base: getindex, in, *
using StaticArrays
using FastGaussQuadrature
using LinearAlgebra
# using SingularIntegralEquations
using Test

export OrthogonalPolynomialFamily, HalfDisk, HalfDiskSpace, HalfDiskFamily


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

# R should be Float64
abstract type DiskSpaceFamily{R} end

struct HalfDisk{R} <: Domain{SVector{2,R}} end

HalfDisk() = HalfDisk{Float64}()

checkpoints(::HalfDisk) = [ SVector(0.1,0.23), SVector(0.3,0.12)]

struct HalfDiskSpace{DF, R, FA, F} <: Space{HalfDisk{R}, R}
    family::DF # Pointer back to the family
    a::R # Power of the "x" factor in the weight
    b::R # Power of the "(1-x^2-y^2)" factor in the weight
    H::FA # OPFamily in [0,1]
    P::FA # OPFamily in [-1,1]
    ρ::F # Fun of sqrt(1-X^2) in [0,1]
    opnorms::Vector{R}
end

function HalfDiskSpace(fam::DiskSpaceFamily{R}, a::R, b::R) where R
    X = Fun(identity, 0..1)
    Y = Fun(identity, -1..1)
    H = OrthogonalPolynomialFamily(X, 1-X^2)
    P = OrthogonalPolynomialFamily(1+Y, 1-Y)
    ρ = sqrt(1 - X^2)
    HalfDiskSpace{typeof(fam), typeof(a), typeof(H), typeof(ρ)}(fam, a, b, H, P, ρ, Vector{R}())
end
HalfDiskSpace() = HalfDiskSpace(0.5, 0.5)

in(x::SVector{2}, D::HalfDisk) = 0 ≤ x[1] ≤ 1 && -sqrt(1-x[1]^2) ≤ x[2] ≤ sqrt(1-x[1]^2)

spacescompatible(A::HalfDiskSpace, B::HalfDiskSpace) = (A.a == B.a && A.b == B.b)

# NOTE we output ≈n points (x,y), plus the ≈n points corresponding to (x,-y)
function pointswithweights(S::HalfDiskSpace, n)
    # Return the weights and nodes to use for the even
    # of a function, i.e. for the half disk Ω:
    #   int_Ω W^{a,b}(x,y) f(x,y) dydx ≈ Σ_j weⱼ*fe(xⱼ,yⱼ)
    # NOTE: the odd part of the quad rule will equal 0 for polynomials,
    #       so can be ignored.
    N = Int(ceil(sqrt(n)))
    t, wt = golubwelsch((S.P)(S.b, S.b).weight, N)
    s, ws = golubwelsch((S.H)(S.a, S.b + 0.5).weight, N)
    pts = Vector{SArray{Tuple{2},Float64,1,2}}(undef, 2(N^2))
    w = zeros(N^2)
    for i = 1:N
        for k = 1:N
            x, y = s[k], t[i] * sqrt(1 - s[k]^2)
            pts[i + (k - 1)N] = x, y
            pts[N^2 + i + (k - 1)N] = x, -y
            w[i + (k - 1)N] = ws[k] * wt[i]
        end
    end
    pts, w
end
points(S::HalfDiskSpace, n) = pointswithweights(S, n)[1]

domain(::HalfDiskSpace) = HalfDisk()

function halfdisknorm(f, pts, w)
    n = Int(length(pts) / 2)
    sum(([f(pt...) for pt in pts[1:n]].^2 + [f(pt...) for pt in pts[n+1:end]].^2) .* w) / 2
end

struct HalfDiskTransformPlan
    U::Diagonal{Float64,Vector{Float64}}
    VTp::Matrix{Float64}
    W::Diagonal{Float64,Vector{Float64}}
    VTm::Matrix{Float64}
end

function HalfDiskTransformPlan(S, vals)
    n = Int(length(vals) / 2)
    N = Int(sqrt(n)) - 1
    m = Int((N+1)*(N+2) / 2)
    pts, w = pointswithweights(S, n)

    # We store the norms of the OPs
    m̃ = length(S.opnorms)
    if m̃ < m
        resize!(S.opnorms, m)
        for k = m̃+1:m
            Pnk = Fun(S, [zeros(k-1); 1])
            S.opnorms[k] = halfdisknorm(Pnk, pts, w)
        end
    end

    # Vandermonde matrices transposed, for each set of pts (x, ±y)
    VTp = Array{Float64}(undef, m, n)
    VTm = copy(VTp)
    for k = 1:m
        Pnk = Fun(S, [zeros(k-1); 1])
        VTp[k, :] = Pnk.(pts[1:n])
        VTm[k, :] = Pnk.(pts[n+1:end])
    end
    W = Diagonal(w)
    U = Diagonal(1 ./ S.opnorms[1:m])

    HalfDiskTransformPlan(U, VTp, W, VTm)
end

transform(S::HalfDiskSpace, vals) = plan_transform(S, vals) * vals

# Inputs: OP space, f(pts) for desired f
# Output: Coeffs of the function f for its expansion in the OPSpace OPs
function *(P::HalfDiskTransformPlan, vals)
    n = Int(length(vals) / 2)
    P.U * (P.VTp * P.W * vals[1:n] + P.VTm * P.W * vals[n+1:end]) / 2
end

plan_transform(S::HalfDiskSpace, vals) = HalfDiskTransformPlan(S, vals)

# Inputs: OP space, coeffs of a function f for its expansion in the OPSpace OPs
# Output: vals = {f(x_j)} where x_j are are the points(S,n)
function itransform(S::HalfDiskSpace, cfs)
    m = length(cfs)
    N = Int(round(0.5 * (-1 + sqrt(1 + 8m))))
    n = N^2
    pts = points(S, n)
    V = Array{Float64}(undef, 2n, m)
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

# R should be Float64
struct HalfDiskFamily{R} <: DiskSpaceFamily{R}
    spaces::Dict{NTuple{2,R}, HalfDiskSpace}
end

function HalfDiskFamily()
    WW = Fun
    R = Float64
    spaces = Dict{NTuple{2,R}, HalfDiskSpace}()
    HalfDiskFamily{R}(spaces)
end

function (D::HalfDiskFamily{R})(a::R, b::R) where R
    haskey(D.spaces,(a,b)) && return D.spaces[(a,b)]
    D.spaces[(a,b)] = HalfDiskSpace(D, a, b)
end


end # module
