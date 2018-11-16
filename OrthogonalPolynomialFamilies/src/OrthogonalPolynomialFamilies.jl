module OrthogonalPolynomialFamilies

using ApproxFun
    import ApproxFun: evaluate, PolynomialSpace, recα, recβ, recγ, recA, recB, recC, domain,
                        domainspace, rangespace, bandwidths, prectype, canonicaldomain, tocanonical,
                        spacescompatible, points, transform, itransform, AbstractProductSpace, tensorizer,
                        columnspace, checkpoints, plan_transform, clenshaw
    import Base: getindex, in, *
using StaticArrays
using FastGaussQuadrature
using LinearAlgebra
using SparseArrays
using BlockBandedMatrices
using BlockArrays
# using SingularIntegralEquations
using Test

export OrthogonalPolynomialFamily, HalfDisk, HalfDiskSpace, HalfDiskFamily


# Finds the OPs and recurrence for weight w, having already found N₀ OPs
function lanczos!(w, P, β, γ, N₀=0)

    # x * P[n](x) == (γ[n] * P[n+1](x) + β[n] * P[n](x) + γ[n-1] * P[n-1](x))

    N = length(β)
    # x = Fun(identity, space(w)) # NOTE: space(w) does not "work" sometimes
    x = Fun(identity, domain(w))

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


OrthogonalPolynomialSpace(fam::SpaceFamily{D,R}, w::Fun) where {D,R,N} =
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
    if length(α) == 1
        haskey(P.spaces,α) && return P.spaces[α]
        P.spaces[α] = OrthogonalPolynomialSpace(P, (P.factors.^α)[1])
    else
        haskey(P.spaces,α) && return P.spaces[α]
        P.spaces[α] = OrthogonalPolynomialSpace(P, prod(P.factors.^α))
    end
end

#####
# recα/β/γ are given by
#       x p_{n-1} = γ_n p_{n-2} + α_n p_{n-1} +  p_n β_n
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

inner(S::OrthogonalPolynomialSpace, f::Fun, g::Fun, pts, w) =
    sum(f.(pts) .* g.(pts) .* w)

function differentiateop(S::OrthogonalPolynomialSpace, n)
    if n == 0
        return Fun(S, [0])
    end
    f = Fun(S, [0,1])
    dom = domain(S)
    X = Fun(identity, dom)
    T = Float64
    p0 = Fun(S, [1])
    dp1 = (f(Float64(dom.right)) - f(Float64(dom.left))) / (Float64(dom.right) - Float64(dom.left)) # p1 is linear
    if n == 1
        return Fun(dp1, dom)
    end
    k = 1
    p1 = (recB(T, S, k-1) + recA(T, S, k-1) * X) * p0
    dp2 = ((recB(T, S, k) + recA(T, S, k) * X) * dp1
           + recA(T, S, k) * p1)
    pm1 = copy(p0); p0 = copy(p1); dp0 = copy(dp1); dp1 = copy(dp2)
    for k = 2:n-1
        p1 = (recB(T, S, k-1) + recA(T, S, k-1) * X) * p0 - recC(T, S, k-1) * pm1
        dp2 = ((recB(T, S, k) + recA(T, S, k) * X) * dp1
               - recC(T, S, k) * dp0
               + recA(T, S, k) * p1)
        pm1 = copy(p0); p0 = copy(p1); dp0 = copy(dp1); dp1 = copy(dp2)
    end
    dp1
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
    opnorms::Vector{R} # squared norms
    A::Vector{SparseMatrixCSC{R}}
    B::Vector{SparseMatrixCSC{R}}
    C::Vector{SparseMatrixCSC{R}}
    DT::Vector{SparseMatrixCSC{R}}
end

function HalfDiskSpace(fam::DiskSpaceFamily{R}, a::R, b::R) where R
    X = Fun(identity, 0..1)
    Y = Fun(identity, -1..1)
    H = OrthogonalPolynomialFamily(X, 1-X^2)
    P = OrthogonalPolynomialFamily(1+Y, 1-Y)
    ρ = sqrt(1 - X^2)
    HalfDiskSpace{typeof(fam), typeof(a), typeof(H), typeof(ρ)}(fam, a, b, H, P, ρ, Vector{R}(), Vector{SparseMatrixCSC{R}}(), Vector{SparseMatrixCSC{R}}(), Vector{SparseMatrixCSC{R}}(), Vector{SparseMatrixCSC{R}}())
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

function inner(S::HalfDiskSpace, f, g, pts, w)
    n = Int(length(pts) / 2)
    sum(([f(pt...) * g(pt...) for pt in pts[1:n]]
            + [f(pt...) * g(pt...) for pt in pts[n+1:end]]) .* w) / 2
end

# NOTE these are squared norms
function getopnorms(S::HalfDiskSpace, m)
    m̃ = length(S.opnorms)
    n = m̃ == 0 ? -1 : -1 + Int(round(sqrt(1 + 2(m̃ - 1))))
    k = m̃ - Int((n + 1)n / 2)
    resize!(S.opnorms, m)
    P = S.P(S.b, S.b)
    p = Fun(P, [1])
    ptsp, wp = pointswithweights(P, 1)
    normP = inner(P, p, p, ptsp, wp)
    for j = m̃+1:m
        if k > n
            n += 1
            k = 0
        end
        # Pnk = Fun(S, [zeros(j-1); 1])
        # S.opnorms[j] = inner(S, Pnk, Pnk, pts, w)
        H = S.H(S.a, S.b + k + 0.5)
        h = Fun(H, [zeros(n-k); 1])
        ptsh, wh = pointswithweights(H, Int(ceil(n-k+0.5)))
        S.opnorms[j] = inner(H, h, h, ptsh, wh) * normP
        k += 1
    end
    S
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
    if length(S.opnorms) < m
        getopnorms(S, m)
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

# TODO: Store these coeffs?
function recα(S::HalfDiskSpace, n, k, j)
    T = Float64 # TODO
    H = (S.H)(S.a, S.b + k + 0.5)
    if j == 1
        return recγ(T, H, n-k+1)
    elseif j == 2
        return recα(T, H, n-k+1)
    end
    error("Invalid entry to function")
end

# TODO
function recβ(S::HalfDiskSpace, n, k, j)
    T = Float64
    P = (S.P)(S.b, S.b)
    H1 = (S.H)(S.a, S.b + k - 0.5)
    H2 = (S.H)(S.a, S.b + k + 0.5)
    H3 = (S.H)(S.a, S.b + k + 1.5)

    # We store the norms of the 2D OPs
    m = Int((n+2)*(n+3) / 2)
    if length(S.opnorms) < m
        # pts, w = pointswithweights(S, (n + 1.5)^2)
        getopnorms(S, m)
    end

    # Doing this first ensures the 1D ops are calculated
    resizedata!(H1, n-k+3)
    resizedata!(H2, n-k+1)
    resizedata!(H3, n-k+1)

    ptsp, wp = pointswithweights(P, Int(ceil(k+1.5)))
    h2 = Fun(H2, [zeros(n-k); 1])
    if isodd(j)
        ptsh, wh = pointswithweights(H2, Int(ceil(n-k+1.5)))
        p = Fun(P, [zeros(k-1); 1])
        δ = recγ(T, P, k+1) * inner(P, p, p, ptsp, wp)
    else
        ptsh, wh = pointswithweights(H3, Int(ceil(n-k+1.5)))
        p = Fun(P, [zeros(k+1); 1])
        δ = recβ(T, P, k+1) * inner(P, p, p, ptsp, wp)
    end
    if j == 1
        h1 = Fun(H1, [zeros(n-k); 1])
        return inner(H2, h2, h1, ptsh, wh) * δ / S.opnorms[Int((n-1)n / 2) + k]
    elseif j == 2
        h3 = Fun(H3, [zeros(n-k-2); 1])
        return inner(H3, h2, h3, ptsh, wh) * δ / S.opnorms[Int((n-1)n / 2) + k + 2]
    elseif j == 3
        h1 = Fun(H1, [zeros(n-k+1); 1])
        return inner(H2, h2, h1, ptsh, wh) * δ / S.opnorms[Int((n+1)n / 2) + k]
    elseif j == 4
        h3 = Fun(H3, [zeros(n-k-1); 1])
        return inner(H3, h2, h3, ptsh, wh) * δ / S.opnorms[Int((n+1)n / 2) + k + 2]
    elseif j == 5
        h1 = Fun(H1, [zeros(n-k+2); 1])
        return inner(H2, h2, h1, ptsh, wh) * δ / S.opnorms[Int((n+1)*(n+2) / 2) + k]
    elseif j == 6
        h3 = Fun(H3, [zeros(n-k); 1])
        return inner(H3, h2, h3, ptsh, wh) * δ / S.opnorms[Int((n+1)*(n+2) / 2) + k + 2]
    end
    error("Invalid entry to function")
end

function getAs!(S::HalfDiskSpace, N, N₀)
    m = N₀
    if m == 0
        S.A[1] = [recα(S, 1, 0, 1) 0; 0 recβ(S, 0, 0, 6)]
        m += 1
    end
    for n = N+1:-1:m
        v1 = [recα(S, n+1, k, 1) for k = 0:n]
        v2 = [recβ(S, n, k, 6) for k = 0:n-1]
        v3 = [recβ(S, n, k, 5) for k = 1:n]
        S.A[n+1] = [Diagonal(v1) zeros(n+1);
                    Tridiagonal(v3, zeros(n+1), v2) [zeros(n); recβ(S, n, n, 6)]]
    end
end

function getDTs!(S::HalfDiskSpace, N, N₀)
    for n = N+1:-1:N₀
        vα = [1 / recα(S, n+1, k, 1) for k = 0:n]
        m = iseven(n) ? Int(n/2) + 1 : Int((n+1)/2) + 1
        vβ = zeros(m)
        vβ[end] = 1 / recβ(S, n, n, 6)
        if iseven(n)
            for k = 1:m-1
                vβ[end-k] = - (vβ[end-k+1]
                                * recβ(S, n, n - 2(k - 1), 5)
                                / recβ(S, n, n - 2k, 6))
            end
        else
            for k = 1:m-2
                vβ[end-k] = - (vβ[end-k+1]
                                * recβ(S, n, n - 2(k - 1), 5)
                                / recβ(S, n, n - 2k, 6))
            end
            vβ[1] = - (vβ[2]
                        * recβ(S, n, 1, 5)
                        / recα(S, n+1, 0, 1))
        end
        ij = [i for i=1:n+1]
        ind1 = [ij; [n+2 for i=1:m]]
        ind2 = [ij; iseven(n) ? [n+2k for k=1:m] : [1; [n-1+2k for k=2:m]]]
        S.DT[n+1] = sparse(ind1, ind2, [vα; vβ])
        # S.DT[n+1] = sparse(pinv(Array(S.A[n+1])))
    end
end
function getBs!(S::HalfDiskSpace, N, N₀)
    m = N₀
    if N₀ == 0
        S.B[1] = sparse([1, 2], [1, 1], [recα(S, 0, 0, 2), 0])
        m += 1
    end
    for n = N+1:-1:m
        v1 = [recα(S, n, k, 2) for k = 0:n]
        v2 = [recβ(S, n, k, 4) for k = 0:n-1]
        v3 = [recβ(S, n, k, 3) for k = 1:n]
        S.B[n+1] = sparse([Diagonal(v1); Tridiagonal(v3, zeros(n+1), v2)])
    end
end
function getCs!(S::HalfDiskSpace, N, N₀)
    m = N₀
    if N₀ == 0
        # C_0 does not exist
        m += 1
    end
    if m == 1
        S.C[2] = sparse([1, 4], [1, 1], [recα(S, 1, 0, 1), recβ(S, 1, 1, 1)])
        m += 1
    end
    for n = N+1:-1:m
        v1 = [recα(S, n, k, 1) for k = 0:n-1]
        v2 = [recβ(S, n, k, 2) for k = 0:n-2]
        v3 = [recβ(S, n, k, 1) for k = 1:n-1]
        S.C[n+1] = sparse([Diagonal(v1);
                           zeros(1,n);
                           Tridiagonal(v3, zeros(n), v2);
                           [zeros(1, n-1) recβ(S, n, n, 1)]])
    end
end

function resizedata!(S::HalfDiskSpace, N)
    # N is the max degree of the OPs
    N₀ = length(S.B)
    N ≤ N₀ - 2 && return S
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

function jacobix(S, N)
    resizedata!(S, N)
    rows = cols = 1:N+1
    l, u = 1, 1
    λ, μ = 0, 0
    J = BandedBlockBandedMatrix(0.0I, (rows, cols), (l, u), (λ, μ))
    J[1, 1] = S.B[1][1, 1]
    view(J, Block(1, 2)) .= S.A[1][1, :]'
    for n = 2:N
        view(J, Block(n, n-1)) .= S.C[n][1:Int(end/2), :]
        view(J, Block(n, n)) .= S.B[n][1:Int(end/2), :]
        view(J, Block(n, n+1)) .= S.A[n][1:Int(end/2), :]
    end
    view(J, Block(N+1, N)) .= S.C[N+1][1:Int(end/2), :]
    view(J, Block(N+1, N+1)) .= S.B[N+1][1:Int(end/2), :]
    J
end

function jacobiy(S, N)
    resizedata!(S, N)
    rows = cols = 1:N+1
    l, u = 1, 1
    λ, μ = 1, 1
    J = BandedBlockBandedMatrix(0.0I, (rows, cols), (l, u), (λ, μ))
    J[1, 1] = S.B[1][2, 1]
    view(J, Block(1, 2)) .= S.A[1][2, :]'
    for n = 2:N
        view(J, Block(n, n-1)) .= S.C[n][Int(end/2)+1:end, :]
        view(J, Block(n, n)) .= S.B[n][Int(end/2)+1:end, :]
        view(J, Block(n, n+1)) .= S.A[n][Int(end/2)+1:end, :]
    end
    view(J, Block(N+1, N)) .= S.C[N+1][Int(end/2)+1:end, :]
    view(J, Block(N+1, N+1)) .= S.B[N+1][Int(end/2)+1:end, :]
    J
end

function clenshawG(n, z)
    sp = sparse(I, n+1, n+1)
    [z[1] * sp; z[2] * sp]
end
function clenshaw(cfs::AbstractVector, S::HalfDiskSpace, z)
    # TODO: Implement clenshaw for the half disk
    m̃ = length(cfs)
    N = -1 + Int(round(sqrt(1+2(m̃-1))))
    resizedata!(S, N+1)
    m = Int((N+1)*(N+2)/2)
    if m̃ < m
        resize!(cfs, m)
        cfs[m̃+1:end] .= 0.0
    end
    P0 = 1.0
    if N == 0
        return cfs[1] * P0
    end
    ρx = S.ρ(z[1])
    P1 = [Fun(S.H(S.a, S.b+0.5), [0, 1])(z[1]);
          ρx * Fun(S.P(S.b, S.b), [0, 1])(z[2]/ρx)]
    if N == 1
        return cfs[1] * P0 + dot(view(cfs, 2:3), P1)
    end
    inds = m-N:m
    γ2 = view(cfs, inds)'
    inds = (m-2N):(m-N-1)
    γ1 = view(cfs, inds)' - γ2 * S.DT[N] * (S.B[N] - clenshawG(N-1, z))
    for n = N-2:-1:1
        ind = sum(1:n)
        γ = (view(cfs, ind+1:ind+n+1)'
             - γ1 * S.DT[n+1] * (S.B[n+1] - clenshawG(n, z))
             - γ2 * S.DT[n+2] * S.C[n+2])
        γ2 = copy(γ1)
        γ1 = copy(γ)
    end
    cfs[1] * P0 + γ1 * P1 - (P0 * γ2 * S.DT[2] * S.C[2])[1]
end
evaluate(cfs::AbstractVector, S::HalfDiskSpace, z) = clenshaw(cfs, S, z)

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

(D::HalfDiskFamily{R})(a, b) where R = D(convert(R,a), convert(R,b))

# Partial derivatives:
# Seems that: dx takes (a,b)->(a+1,b+1) with nonzero for m=n-1,n-2
#             dy takes (a,b)->(a,b+1) with nonzero for m=n-1
#                   OR takes (a,b)->(a+1,b+1) with nonzero for m=n-1,n-2
function evalderivativex(S::HalfDiskSpace, n, k, x, y)
    H = Fun(S.H(S.a, S.b+k+0.5), [zeros(n-k); 1])
    P = Fun(S.P(S.b, S.b), [zeros(k); 1])
    ρ = S.ρ(x); h = H(x); p = P(y/ρ)
    ρ^(k - 3) * (differentiateop(S.H(S.a, S.b+k+0.5), n-k)(x) * ρ^3 * p
                 - x * k * h * ρ * p
                 + x * y * h * differentiateop(S.P(S.b, S.b), k)(y/ρ))
end
function evalderivativey(S::HalfDiskSpace, n, k, x, y)
    H = Fun(S.H(S.a, S.b+k+0.5), [zeros(n-k); 1])
    H(x) * S.ρ(x)^(k-1) * differentiateop(S.P(S.b, S.b), k)(y/S.ρ(x))
end
function getpartialoperatorx(S::HalfDiskSpace, N)
    if N == 2
        A = BandedBlockBandedMatrix(Zeros{Float64}(sum(1:N),sum(1:(N+1))), (1:N, 1:N+1), (-1,2), (0, 2))
    else
        A = BandedBlockBandedMatrix(Zeros{Float64}(sum(1:N),sum(1:(N+1))), (1:N, 1:N+1), (-1,2), (0, N-1))
    end
    n, k = 1, 0
    cfs = pad(Fun((x,y) -> evalderivativex(S, n, k, x, y),
                differentiatespacex(S)).coefficients, sum(1:n))
    view(A, Block(n, n+1))[1, 1] = cfs[1]
    for n = 2:N, k = 0:n
        cfs = pad(Fun((x,y) -> evalderivativex(S, n, k, x, y), differentiatespacex(S)).coefficients, sum(1:n))
        cfs = PseudoBlockArray(cfs, 1:n)
        if k == 0
            inds1 = 1
            inds2 = 1
        elseif k == 1
            inds1 = 2
            inds2 = 2
        elseif k == n - 1
            inds1 = n-2
            inds2 = n-2:2:n
        elseif k == n
            inds1 = n-1
            inds2 = n-1
        else
            inds1 = k-1:2:k+1
            inds2 = k-1:2:k+1
        end
        if !(n == 2 && k == 1)
            view(A, Block(n-1, n+1))[inds1, k+1] = cfs[Block(n-1)][inds1]
        end
        view(A, Block(n, n+1))[inds2, k+1] = cfs[Block(n)][inds2]
    end
    A
end
function getpartialoperatory(S::HalfDiskSpace, N)
    A = BandedBlockBandedMatrix(Zeros{Float64}(sum(1:N),sum(1:(N+1))), (1:N, 1:N+1), (-1,1), (-1,1))
    for n = 1:N, k = 1:n
        cfs = pad(Fun((x,y) -> evalderivativey(S, n, k, x, y), differentiatespacey(S)).coefficients, sum(1:n))
        cfs = PseudoBlockArray(cfs, 1:n)
        view(A, Block(n, n+1))[k, k+1] = cfs[Block(n)][k]
    end
    A
end
function differentiatex(S::HalfDiskSpace, cfs::AbstractVector)
    m̃ = length(cfs)
    N = -1 + Int(round(sqrt(1+2(m̃-1))))
    m = Int((N+1)*(N+2)/2)
    if m̃ < m
        resize!(cfs, m)
        cfs[m̃+1:end] .= 0.0
    end
    getpartialoperatorx(S, N) * cfs
end
function differentiatey(S::HalfDiskSpace, cfs::AbstractVector)
    m̃ = length(cfs)
    N = -1 + Int(round(sqrt(1+2(m̃-1))))
    m = Int((N+1)*(N+2)/2)
    if m̃ < m
        cfs2 = zeros(m)
        cfs2[1:m̃] = cfs
        getpartialoperatory(S, N) * cfs2
    else
        getpartialoperatory(S, N) * cfs
    end
end
differentiatespacex(S::HalfDiskSpace) =
    (S.family)(S.a+1, S.b+1)
differentiatespacey(S::HalfDiskSpace) =
    (S.family)(S.a, S.b+1)
differentiatex(f::Fun, S::HalfDiskSpace) =
    Fun(differentiatespacex(S), differentiatex(S, f.coefficients))
differentiatey(f::Fun, S::HalfDiskSpace) =
    Fun(differentiatespacey(S), differentiatey(S, f.coefficients))


function evalweightedderivativex(S::HalfDiskSpace, n, k, x, y)

end
function evalweightedderivativey(S::HalfDiskSpace, n, k, x, y)
    j = Int((n+1)*n/2 + k)
    x^(S.a) * (1-x^2-y^2)^(S.b-1) * (-2 * S.b * y * Fun(S, [zeros(j); 1])(x,y)
                                        + evalderivativey(S, n, k, x, y) * (1-x^2-y^2))
end
function getweightedpartialoperatory(S::HalfDiskSpace, N)

end
function getweightedpartialoperatory(S::HalfDiskSpace, N)
    W = BandedBlockBandedMatrix(Zeros{Float64}(sum(1:(N+2)),sum(1:(N+1))), (1:N+2, 1:N+1), (1,-1), (1,-1))
    for n = 0:N, k = 0:n
        p = (x,y) -> (evalweightedderivativey(S, n, k, x, y) * x^(-S.a) * (1-x^2-y^2)^(1-b))
        cfs = pad(Fun(p, (S.family)(S.a, S.b-1)).coefficients, sum(1:(n+2)))
        cfs = PseudoBlockArray(cfs, 1:(n+2))
        view(W, Block(n+2, n+1))[k+2, k+1] = cfs[Block(n+2)][k+2]
    end
    W
end
function gettransformoperator(S, N)
    if Int(S.a) == 1 && Int(S.b) == 0
        T = BandedBlockBandedMatrix(Zeros{Float64}(sum(1:(N+2)),sum(1:(N+1))), (1:N+2, 1:N+1), (1,0), (0,0))
        for n = 0:N, k = 0:n
            j = Int(n*(n+1)/2 + k)
            p = (x,y) -> (x * Fun(S, [zeros(j); 1])(x,y))
            Fun(p, (S.family)(0.0, 0.0)).coefficients
            cfs = pad(Fun(p, (S.family)(0.0, 0.0)).coefficients, sum(1:(n+2)))
            cfs = PseudoBlockArray(cfs, 1:(n+2))
            view(T, Block(n+1, n+1))[k+1, k+1] = cfs[Block(n+1)][k+1]
            view(T, Block(n+2, n+1))[k+1, k+1] = cfs[Block(n+2)][k+1]
        end
        T
    elseif Int(S.a) == 0 && Int(S.b) == 1
        T = BandedBlockBandedMatrix(Zeros{Float64}(sum(1:(N+1)),sum(1:(N+1))), (1:N+1, 1:N+1), (0,1), (0,0))
        for n = 0:N
            for k = 0:n-1
                j = Int(n*(n+1)/2 + k)
                cfs = pad(Fun(Fun(S, [zeros(j); 1]), (S.family)(1.0, 1.0)).coefficients, sum(1:(n+1)))
                cfs = PseudoBlockArray(cfs, 1:(n+1))
                view(T, Block(n, n+1))[k+1, k+1] = cfs[Block(n)][k+1]
                view(T, Block(n+1, n+1))[k+1, k+1] = cfs[Block(n+1)][k+1]
            end
            view(T, Block(n+1, n+1))[n+1, n+1] = 1.0
        end
        T
    else
        error("Invalid HalfDiskSpace")
    end
end

end # module
