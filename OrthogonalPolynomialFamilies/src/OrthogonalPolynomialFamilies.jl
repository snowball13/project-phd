module OrthogonalPolynomialFamilies

using ApproxFun
import ApproxFun: evaluate, domain,
                    domainspace, rangespace, bandwidths, prectype, canonicaldomain, tocanonical,
                    spacescompatible, points, transform, itransform, AbstractProductSpace,
                    checkpoints, plan_transform, clenshaw
import ApproxFunOrthogonalPolynomials: PolynomialSpace, recα, recβ, recγ, recA, recB, recC
import ApproxFunBase: tensorizer, columnspace
import Base: in, *
using StaticArrays
using FastGaussQuadrature
using LinearAlgebra
using SparseArrays
using BlockBandedMatrices
using BlockArrays
# using SingularIntegralEquations
using Test

export OrthogonalPolynomialFamily, HalfDisk, HalfDiskSpace, HalfDiskFamily


abstract type SpaceFamily{D,R} end

struct OrthogonalPolynomialSpace{FA,WW,F,D,B,R,N} <: PolynomialSpace{D,R}
    family::FA # Pointer back to the family
    weight::WW # The full product weight
    params::NTuple{N,B} # The powers of the weight factors (i.e. key to this
                        # space in the dict of the family) (could be BigFloats)
    ops::Vector{F}  # Cache the ops we get for free from lanczos
    a::Vector{R} # Diagonal recurrence coefficients
    b::Vector{R} # Off diagonal recurrence coefficients
    opnorm::Vector{R} # The norm of the OPs (all OPs of an OPSpace have the same norm).
                        # NOTE this is the value of the norm squared
    opptseval::Vector{Vector{R}}
    derivopptseval::Vector{Vector{R}}
end


# Finds the OPs and recurrence for weight w, having already found N₀ OPs
function lanczos!(w, P, β, γ; N₀=0)

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
        # @show "lanczos", N, k
        v = x*P[2] - γ[k-1]*P[1]
        β[k] = sum(P[2]*w*v)
        v = v - β[k]*P[2]
        γ[k] = sqrt(sum(w*v*v))
        P[1] = P[2]
        P[2] = v/γ[k]
    end

    P, β, γ
end

# Finds the OPs and recurrence for weight w
function lanczos(w, N)
    # x * P[n](x) == (γ[n] * P[n+1](x) + β[n] * P[n](x) + γ[n-1] * P[n-1](x))
    P = Array{Fun}(undef, N + 1)
    β = Array{eltype(w)}(undef, N)
    γ = Array{eltype(w)}(undef, N)
    lanczos!(w, P, β, γ)
end

domain(S::OrthogonalPolynomialSpace) = domain(S.weight)
canonicaldomain(S::OrthogonalPolynomialSpace) = domain(S)
tocanonical(S::OrthogonalPolynomialSpace, x) = x

OrthogonalPolynomialSpace(fam::SpaceFamily{D,R}, w::Fun, α::NTuple{N,B}) where {D,R,B,N} =
    OrthogonalPolynomialSpace{typeof(fam),typeof(w),Fun,D,B,R,N}(
        fam, w, α, Vector{Fun}(), Vector{R}(), Vector{R}(), Vector{R}(),
        Vector{Vector{R}}(), Vector{Vector{R}}())

function resizedata!(S::OrthogonalPolynomialSpace, n)
    N₀ = length(S.a)
    n ≤ N₀ && return S
    resize!(S.a, n)
    resize!(S.b, n)
    resize!(S.ops, 2)
    lanczos!(S.weight, S.ops, S.a, S.b, N₀=N₀)
    S
end

# R is range-type, which should be Float64. B is the r-type of the weight Funs,
# which could be BigFloats
struct OrthogonalPolynomialFamily{OPS,FF,D,R,B,N} <: SpaceFamily{D,R}
    factors::FF
    spaces::Dict{NTuple{N,B}, OPS}
end
function OrthogonalPolynomialFamily(w::Vararg{Fun{<:Space{D,B}},N}) where {D,B,N}
    all(domain.(w) .== Ref(domain(first(w)))) || throw(ArgumentError("domains incompatible"))
    R = Float64 # TODO - is there a way to not hardcode this? (see below)
    spaces = Dict{NTuple{N,R}, OrthogonalPolynomialSpace}()
    OrthogonalPolynomialFamily{OrthogonalPolynomialSpace,typeof(w),D,R,B,N}(w, spaces)
end
function OrthogonalPolynomialFamily(::Type{R}, w::Vararg{Fun{<:Space{D,B}},N}) where {D,R,B,N}
    all(domain.(w) .== Ref(domain(first(w)))) || throw(ArgumentError("domains incompatible"))
    spaces = Dict{NTuple{N,R}, OrthogonalPolynomialSpace}()
    OrthogonalPolynomialFamily{OrthogonalPolynomialSpace,typeof(w),D,R,B,N}(w, spaces)
end

function (P::OrthogonalPolynomialFamily{<:Any,<:Any,<:Any,R,B,N})(α::Vararg{B,N}) where {R,B,N}
    haskey(P.spaces,α) && return P.spaces[α]
    # NOTE: This ensures we dont hang on calculating the weight for large
    # parameter values
    maxp = 100
    if any(α .> maxp)
        β = []
        f = []
        for j = 1:length(α)
            r = α[j] % maxp
            t = Int(div(α[j], maxp))
            β = append!(β, [tt == 0 ? r : maxp for tt=0:t])
            f = append!(f, [P.factors[j] for tt=0:t])
        end
        W = length(α) == 1 ? (f.^β)[1] : prod(f.^β)
    else
        W = length(α) == 1 ? (P.factors.^α)[1] : prod(P.factors.^α)
    end
    P.spaces[α] = OrthogonalPolynomialSpace(P, W, α)
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
function golubwelsch(S::OrthogonalPolynomialSpace{<:Any,<:Any,<:Any,<:Any,<:Any,T,<:Any}, N::Integer) where T
    # Golub--Welsch algorithm. Used here for N<=20.
    resizedata!(S, N)       # 3-term recurrence
    J = SymTridiagonal(S.a[1:N], S.b[1:N-1])   # Jacobi matrix
    D, V = eigen(J)                     # Eigenvalue decomposition
    indx = sortperm(D)                  # Hermite points
    μ = T(sum(S.weight))                # Integral of weight function
    w = μ * V[1, indx].^2               # quad rule weights to output
    x = D[indx]                         # quad rule nodes to output
    # # Enforce symmetry:
    # ii = floor(Int, N/2)+1:N
    # x = x[ii]
    # w = w[ii]
    return x, w
end
points(S::OrthogonalPolynomialSpace, n) = golubwelsch(S, n)[1]
pointswithweights(S::OrthogonalPolynomialSpace, n) = golubwelsch(S, n)

spacescompatible(A::OrthogonalPolynomialSpace, B::OrthogonalPolynomialSpace) =
    A.weight ≈ B.weight

# Inputs: OP space, f(pts) for desired f
# Output: Coeffs of the function f for its expansion in the OPSpace OPs
function transform(S::OrthogonalPolynomialSpace, vals::Vector{T}) where T
    n = length(vals)
    pts, w = pointswithweights(S, n)
    getopptseval(S, n-1, pts)
    cfs = zeros(n)
    for k = 0:n-1
        cfs[k+1] = inner2(S, opevalatpts(S, k+1, pts), vals, w) / getopnorm(S)
    end
    cfs
    # # Vandermonde matrix transposed, including weights and normalisations
    # Ṽ = Array{T}(undef, n, n)
    # for k = 0:n-1
    #     pk = Fun(S, [zeros(k); 1])
    #     nrm = sum([pk(pts[j])^2 * w[j] for j = 1:n])
    #     Ṽ[k+1, :] = pk.(pts) .* w / nrm
    # end
    # Ṽ * vals
end

# Inputs: OP space, coeffs of a function f for its expansion in the OPSpace OPs
# Output: vals = {f(x_j)} where x_j are are the points(S,n)
function itransform(S::OrthogonalPolynomialSpace, cfs::Vector{T}) where T
    n = length(cfs)
    pts, w = pointswithweights(S, n)
    vals = zeros(n)
    getopptseval(S, n-1, pts)
    for k = 1:n
        vals[k] = sum([cfs[j] * opevalatpts(S, j, pts)[k] for j = 1:n])
    end
    vals
    # # Vandermonde matrix
    # V = Array{T}(undef, n, n)
    # for k = 0:n-1
    #     pk = Fun(S, [zeros(k); 1])
    #     V[:, k+1] = pk.(pts)
    # end
    # V * cfs
end

inner(S::OrthogonalPolynomialSpace, f::Fun, g::Fun, pts, w) =
    sum(f.(pts) .* g.(pts) .* w)
inner2(S::OrthogonalPolynomialSpace, f, g, w) = sum(f .* g .* w)

function differentiateop(S::OrthogonalPolynomialSpace{<:Any,<:Any,<:Any,<:Any,<:Any,T,<:Any}, n) where T
    if n == 0
        return Fun(S, [0])
    end
    f = Fun(S, [0,1])
    dom = domain(S)
    X = Fun(identity, dom)
    p0 = Fun(S, [1])
    dp1 = (f(T(dom.right)) - f(T(dom.left))) / (T(dom.right) - T(dom.left)) # p1 is linear
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

# Method to gather and evaluate the ops of space S at the transform pts given
function getopptseval(S::OrthogonalPolynomialSpace, N, pts)
    resetopptseval(S)
    for n = 0:N
        opevalatpts(S, n+1, pts)
    end
    S.opptseval
end
function opevalatpts(S::OrthogonalPolynomialSpace{<:Any,<:Any,<:Any,<:Any,<:Any,T,<:Any}, j, pts) where T
    # Here, j refers to the index (i.e. deg(p) - 1)
    N = length(S.opptseval) - 1
    if N ≥ j - 1
        return S.opptseval[j]
    end

    # We iterate up from the last obtained pts eval
    if  N != j - 2
        error("Invalid index")
    end

    resizedata!(S, j)
    resize!(S.opptseval, j)
    S.opptseval[j] = Vector{T}(undef, length(pts))

    # p_{n+1} = (A_n x + B_n)p_n - C_n p_{n-1}
    n = j - 1
    if n == 0
        S.opptseval[j][:] .= 1.0
    elseif n == 1
        A = recA(T, S, n-1)
        B = recB(T, S, n-1)
        P1 = opevalatpts(S, j-1, pts)
        for r = 1:length(pts)
            S.opptseval[j][r] = (A * pts[r] + B) * P1[r]
        end
    else
        A = recA(T, S, n-1)
        B = recB(T, S, n-1)
        C = recC(T, S, n-1)
        P1 = opevalatpts(S, j-1, pts)
        P2 = opevalatpts(S, j-2, pts)
        for r = 1:length(pts)
            S.opptseval[j][r] = (A * pts[r] + B) * P1[r] - C * P2[r]
        end
    end
    S.opptseval[j]
end
resetopptseval(S::OrthogonalPolynomialSpace) = resize!(S.opptseval, 0)

# Method to gather and evaluate the op derivatives of space S at the transform pts given
# NOTE: getopptseval() has to be called first
function getderivopptseval(S::OrthogonalPolynomialSpace, N, pts)
    resetderivopptseval(S)
    for n = 0:N
        derivopevalatpts(S, n+1, pts)
    end
    S.derivopptseval
end
function derivopevalatpts(S::OrthogonalPolynomialSpace{<:Any,<:Any,<:Any,<:Any,<:Any,T,<:Any}, j, pts) where T
    # Here, j refers to the index (i.e. deg(p) - 1)
    N = length(S.derivopptseval) - 1
    if N ≥ j - 1
        return S.derivopptseval[j]
    end

    # We iterate up from the last obtained pts eval
    if  N != j - 2
        error("Invalid index")
    end

    resizedata!(S, j)
    resize!(S.derivopptseval, j)
    S.derivopptseval[j] = Vector{T}(undef, length(pts))

    # p_{n+1} = (A_n x + B_n)p_n - C_n p_{n-1}
    # => dxp_{n+1} = A_n p_n + (A_n x + B_n)dxp_n - C_n dxp_{n-1}
    n = j - 1
    if n == 0 # Scalar
        S.derivopptseval[j][:] .= 0.0 # Scalar
    elseif n == 1 # Linear
        A = recA(T, S, n-1)
        B = recB(T, S, n-1)
        P1 = opevalatpts(S, n, pts)
        dP1 = derivopevalatpts(S, n, pts)
        for r = 1:length(pts)
            S.derivopptseval[j][r] = A * P1[r] + (A * pts[r] + B) * dP1[r]
        end
    else
        A = recA(T, S, n-1)
        B = recB(T, S, n-1)
        C = recC(T, S, n-1)
        P1 = opevalatpts(S, n, pts)
        dP1 = derivopevalatpts(S, n, pts)
        dP2 = derivopevalatpts(S, n-1, pts)
        for r = 1:length(pts)
            S.derivopptseval[j][r] = A * P1[r] + (A * pts[r] + B) * dP1[r] - C * dP2[r]
        end
    end
    S.derivopptseval[j]
end
resetderivopptseval(S::OrthogonalPolynomialSpace) = resize!(S.derivopptseval, 0)

function getopnorm(S::OrthogonalPolynomialSpace)
    # NOTE this is the squared norms
    if length(S.opnorm) == 0
        resize!(S.opnorm, 1)
        pts, w = pointswithweights(S, 1)
        p = [1] # p = opevalatpts(S, 1, pts)
        S.opnorm[1] = inner2(S, p, p, w)
    end
    S.opnorm[1]
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
    H = OrthogonalPolynomialFamily(S, 1-S^2)
    P = OrthogonalPolynomialFamily(1-T^2)
    t, wt = golubwelsch(P(b), N)
    se, wse = golubwelsch(H(a, b+0.5), N)
    so, wso = golubwelsch(H(a, b), N)

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
abstract type DiskSpaceFamily{B,R} end

struct HalfDisk{B,R} <: Domain{SVector{2,R}} end

HalfDisk() = HalfDisk{BigFloat, Float64}()

checkpoints(::HalfDisk) = [ SVector(0.1,0.23), SVector(0.3,0.12)]

struct HalfDiskSpace{DF, B, R} <: Space{HalfDisk{B,R}, R}
    family::DF # Pointer back to the family
    a::B # Power of the "x" factor in the weight
    b::B # Power of the "(1-x^2-y^2)" factor in the weight
    opnorms::Vector{R} # squared norms
    opptseval::Vector{Vector{R}} # Store the ops evaluated at the transform pts
    xderivopptseval::Vector{Vector{R}} # Store the x deriv of the ops evaluated
                                    # at the transform pts
    yderivopptseval::Vector{Vector{R}} # Store the y deriv of the ops evaluated
                                    # at the transform pts
    A::Vector{SparseMatrixCSC{R}}
    B::Vector{SparseMatrixCSC{R}}
    C::Vector{SparseMatrixCSC{R}}
    DT::Vector{SparseMatrixCSC{R}}
end

function HalfDiskSpace(fam::DiskSpaceFamily{B,R}, a::B, b::B) where {B,R}
    HalfDiskSpace{typeof(fam), B, R}(
        fam, a, b, Vector{R}(), Vector{Vector{R}}(),
        Vector{Vector{R}}(), Vector{Vector{R}}(), Vector{SparseMatrixCSC{R}}(),
        Vector{SparseMatrixCSC{R}}(), Vector{SparseMatrixCSC{R}}(),
        Vector{SparseMatrixCSC{R}}())
end

in(x::SVector{2}, D::HalfDisk) = 0 ≤ x[1] ≤ 1 && -sqrt(1-x[1]^2) ≤ x[2] ≤ sqrt(1-x[1]^2)

spacescompatible(A::HalfDiskSpace, B::HalfDiskSpace) = (A.a == B.a && A.b == B.b)

weight(S::HalfDiskSpace{<:Any, <:Any, T}, x, y) where T = x^T(S.a) * (1 - x^2 - y^2)^T(S.b)
weight(S::HalfDiskSpace, z) = weight(S, z[1], z[2])

# NOTE we output ≈n points (x,y), plus the ≈n points corresponding to (x,-y)
function pointswithweights(S::HalfDiskSpace{<:Any, <:Any, T}, n) where T
    # Return the weights and nodes to use for the even
    # of a function, i.e. for the half disk Ω:
    #   int_Ω W^{a,b}(x,y) f(x,y) dydx ≈ Σ_j weⱼ*fe(xⱼ,yⱼ)
    # NOTE: the odd part of the quad rule will equal 0 for polynomials,
    #       so can be ignored.
    N = Int(ceil(sqrt(n)))
    @show "begin pointswithweights()", n, N
    t, wt = pointswithweights((S.family.P)(S.b, S.b), N)
    s, ws = pointswithweights((S.family.H)(S.a, S.b + 0.5), N)
    pts = Vector{SArray{Tuple{2},T,1,2}}(undef, 2(N^2))
    w = zeros(N^2)
    for i = 1:N
        for k = 1:N
            x, y = s[k], t[i] * sqrt(1 - s[k]^2)
            pts[i + (k - 1)N] = x, y
            pts[N^2 + i + (k - 1)N] = x, -y
            w[i + (k - 1)N] = ws[k] * wt[i]
        end
    end
    @show "end pointswithweights()"
    # # Reset the calculated OP pt evals, and return the pts and wghts
    # resetopptseval(S)
    pts, w
end
points(S::HalfDiskSpace, n) = pointswithweights(S, n)[1]

domain(::HalfDiskSpace) = HalfDisk()

function inner(S::HalfDiskSpace, f, g, pts, w)
    n = Int(length(pts) / 2)
    sum(([f(pt...) * g(pt...) for pt in pts[1:n]]
            + [f(pt...) * g(pt...) for pt in pts[n+1:end]]) .* w) / 2
end
function inner2(S::HalfDiskSpace, fpts, gpts, w)
    n = length(w)
    sum(([fpts[pt] * gpts[pt] for pt = 1:n]
            + [fpts[pt] * gpts[pt] for pt = n+1:2n]) .* w) / 2
end

# TODO only store...
function getopnorms(S::HalfDiskSpace{<:Any, <:Any, T}, k) where T
    # NOTE these are squared norms
    m = length(S.opnorms)
    if k + 1 > m
        resize!(S.opnorms, k+1)
        P = S.family.P(S.b, S.b)
        getopnorm(P)
        # TODO: This is hardcoded, need to find a better way
        # Doing H(S.a, S.b + k + 0.5) for large k leads to H.weight
        # reaching maximum number of coefficients when calculating due to
        # BigFloats
        X = Fun(T(0)..1)
        W1 = X^T(S.a) * (1-X^2)^(T(S.b)+0.5)
        W = W1 * (1-X^2)^m
        for j = m+1:k+1
            # H = S.family.H(S.a, S.b + k + 0.5)
            # S.opnorms[j] = getopnorm(H) * P.opnorm[1]
            S.opnorms[j] = sum(W) * P.opnorm[1]
            W = W * (1-X^2)
        end
    end
    S
end

struct HalfDiskTransformPlan{T}
    w::Vector{T}
    pts::Vector{SArray{Tuple{2},T,1,2}}
    S::HalfDiskSpace{<:Any, <:Any, T}
end

function HalfDiskTransformPlan(S::HalfDiskSpace{<:Any, <:Any, T}, vals) where T
    m = Int(length(vals) / 2)
    pts, w = pointswithweights(S, m)
    HalfDiskTransformPlan{T}(w, pts, S)
end

transform(S::HalfDiskSpace, vals) = plan_transform(S, vals) * vals

# Inputs: OP space, f(pts) for desired f
# Output: Coeffs of the function f for its expansion in the OPSpace OPs
function *(P::HalfDiskTransformPlan{T}, vals) where T
    @show "Begin HDTP mult"
    m2 = Int(length(vals) / 2)
    N = Int(sqrt(m2)) - 1
    m1 = Int((N+1)*(N+2) / 2)
    @show m1, m2

    ret = zeros(m1)
    resizedata!(P.S, N)
    getopnorms(P.S, N) # We store the norms of the OPs
    for i = 1:m2
        if i % 100 == 0
            @show m2, i
        end
        pt = [P.pts[i]]
        getopptseval(P.S, N, pt)
        for j = 1:m1
            ret[j] += opevalatpts(P.S, j, pt)[1] * P.w[i] * vals[i]
        end
        pt = [P.pts[i+m2]]
        getopptseval(P.S, N, pt)
        for j = 1:m1
            ret[j] += opevalatpts(P.S, j, pt)[1] * P.w[i] * vals[i+m2]
        end
    end
    resetopptseval(P.S)
    j = 1
    for n = 0:N, k = 0:n
        ret[j] /= (2 * P.S.opnorms[k+1])
        j += 1
    end
    @show "End HDTP mult"
    ret
end

plan_transform(S::HalfDiskSpace, vals) = HalfDiskTransformPlan(S, vals)

# Inputs: OP space, coeffs of a function f for its expansion in the OPSpace OPs
# Output: vals = {f(x_j)} where x_j are are the points(S,n)
function itransform(S::HalfDiskSpace{<:Any, <:Any, T}, cfs) where T
    m = length(cfs)
    pts = points(S, m)
    N = getnk(m)[1]
    npts = length(pts)
    V = Array{Float64}(undef, npts, m)
    for i = 1:npts
        pt = [pts[i]]
        getopptseval(S, N, pt)
        for k = 1:m
            V[i, k] = S.opptseval[k][1]
        end
    end
    V * cfs
end # TODO

# TODO: Store these coeffs?
function recα(S::HalfDiskSpace{<:Any, <:Any, T}, n, k, j) where T
    H = (S.family.H)(S.a, S.b + k + 0.5)
    if j == 1
        recγ(T, H, n-k+1)
    elseif j == 2
        recα(T, H, n-k+1)
    else
        error("Invalid entry to function")
    end
end

function recβ(S::HalfDiskSpace{<:Any, <:Any, T}, n, k, j) where T
    # We get the norms of the 2D OPs
    getopnorms(S, k+1)

    H1 = (S.family.H)(S.a, S.b + k - 0.5)
    H2 = (S.family.H)(S.a, S.b + k + 0.5)
    H3 = (S.family.H)(S.a, S.b + k + 1.5)
    P = (S.family.P)(S.b, S.b)
    getopnorm(P)

    if isodd(j)
        pts, w = pointswithweights(H2, Int(ceil(n-k+1.5)))
        δ = recγ(T, P, k+1) * P.opnorm[1]
    else
        pts, w = pointswithweights(H3, Int(ceil(n-k+1.5)))
        δ = recβ(T, P, k+1) * P.opnorm[1]
    end
    getopptseval(H2, n-k+1, pts)

    if j == 1
        getopptseval(H1, n-k+1, pts)
        (inner2(H2, opevalatpts(H2, n-k+1, pts), opevalatpts(H1, n-k+1, pts), w)
            * δ / S.opnorms[k])
    elseif j == 2
        getopptseval(H3, n-k-1, pts)
        (inner2(H3, opevalatpts(H2, n-k+1, pts), opevalatpts(H3, n-k-1, pts), w)
            * δ / S.opnorms[k+2])
    elseif j == 3
        getopptseval(H1, n-k+2, pts)
        (inner2(H2, opevalatpts(H2, n-k+1, pts), opevalatpts(H1, n-k+2, pts), w)
            * δ / S.opnorms[k])
    elseif j == 4
        getopptseval(H3, n-k, pts)
        (inner2(H3, opevalatpts(H2, n-k+1, pts), opevalatpts(H3, n-k, pts), w)
            * δ / S.opnorms[k+2])
    elseif j == 5
        getopptseval(H1, n-k+3, pts)
        (inner2(H2, opevalatpts(H2, n-k+1, pts), opevalatpts(H1, n-k+3, pts), w)
            * δ / S.opnorms[k])
    elseif j == 6
        getopptseval(H3, n-k+1, pts)
        (inner2(H3, opevalatpts(H2, n-k+1, pts), opevalatpts(H3, n-k+1, pts), w)
            * δ / S.opnorms[k+2])
    else
        error("Invalid entry to function")
    end
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
    @show "resizedata!", "done getBs!"
    getCs!(S, N, N₀)
    @show "resizedata!", "done getCs!"
    getAs!(S, N, N₀)
    @show "resizedata!", "done getAs!"
    getDTs!(S, N, N₀)
    @show "resizedata!", "done getDTs!"
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
    # Transposed operator, so acts directly on coeffs vec
    resizedata!(S, N)
    rows = cols = 1:N+1
    l, u = 1, 1
    λ, μ = 1, 1
    J = BandedBlockBandedMatrix(0.0I, (rows, cols), (l, u), (λ, μ))
    n = 1
    J[1, 1] = S.B[1][2, 1]
    view(J, Block(n, n+1)) .= S.C[n+1][Int(end/2)+1:end, :]'
    for n = 2:N
        view(J, Block(n, n-1)) .= S.A[n-1][Int(end/2)+1:end, :]'
        view(J, Block(n, n)) .= S.B[n][Int(end/2)+1:end, :]'
        view(J, Block(n, n+1)) .= S.C[n+1][Int(end/2)+1:end, :]'
    end
    view(J, Block(N+1, N)) .= S.A[N][Int(end/2)+1:end, :]'
    view(J, Block(N+1, N+1)) .= S.B[N+1][Int(end/2)+1:end, :]'
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
    inds2 = m-N:m
    inds1 = (m-2N):(m-N-1)
    γ2 = view(cfs, inds2)'
    γ1 = view(cfs, inds1)' - γ2 * S.DT[N] * (S.B[N] - clenshawG(N-1, z))
    for n = N-2:-1:0
        ind = sum(1:n)
        γ = (view(cfs, ind+1:ind+n+1)'
             - γ1 * S.DT[n+1] * (S.B[n+1] - clenshawG(n, z))
             - γ2 * S.DT[n+2] * S.C[n+2])
        γ2 = copy(γ1)
        γ1 = copy(γ)
    end
    (γ1 * P0)[1]
end
evaluate(cfs::AbstractVector, S::HalfDiskSpace, z) = clenshaw(cfs, S, z)

# R should be Float64, B BigFloat
struct HalfDiskFamily{B,R,FA,F} <: DiskSpaceFamily{B,R}
    spaces::Dict{NTuple{2,B}, HalfDiskSpace}
    H::FA # OPFamily in [0,1]
    P::FA # OPFamily in [-1,1]
    ρ::F # Fun of sqrt(1-X^2) in [0,1]
end

function HalfDiskFamily(::Type{B},::Type{R}) where {B,R}
    # setprecision(850) do
    #     X = Fun(identity, B(0)..1)
    #     Y = Fun(identity, B(-1)..1)
    #     H = OrthogonalPolynomialFamily(R, X, 1-X^2)
    #     P = OrthogonalPolynomialFamily(R, 1+Y, 1-Y)
    #     ρ = sqrt(1 - X^2)
    #     spaces = Dict{NTuple{2,B}, HalfDiskSpace}()
    #     HalfDiskFamily{B,R,typeof(H),typeof(ρ)}(spaces, H, P, ρ)
    # end
    X = Fun(identity, B(0)..1)
    Y = Fun(identity, B(-1)..1)
    H = OrthogonalPolynomialFamily(R, X, 1-X^2)
    P = OrthogonalPolynomialFamily(R, 1+Y, 1-Y)
    ρ = sqrt(1 - X^2)
    spaces = Dict{NTuple{2,B}, HalfDiskSpace}()
    HalfDiskFamily{B,R,typeof(H),typeof(ρ)}(spaces, H, P, ρ)
end
HalfDiskFamily() = HalfDiskFamily(BigFloat, Float64)

function (D::HalfDiskFamily{B,R,<:Any,<:Any})(a::B, b::B) where {B,R}
    haskey(D.spaces,(a,b)) && return D.spaces[(a,b)]
    D.spaces[(a,b)] = HalfDiskSpace(D, a, b)
end
function (D::HalfDiskFamily{B,R,<:Any,<:Any})(a, b) where {B,R}
    # setprecision(850) do
    #     D(B(a), B(b))
    # end
    D(B(a), B(b))
end

# Partial derivatives:
# Seems that: dx takes (a,b)->(a+1,b+1) with nonzero for m=n-1,n-2
#             dy takes (a,b)->(a,b+1) with nonzero for m=n-1
#                   OR takes (a,b)->(a+1,b+1) with nonzero for m=n-1,n-2

differentiatespacex(S::HalfDiskSpace) =
    (S.family)(S.a+1, S.b+1)
differentiatespacey(S::HalfDiskSpace) =
    (S.family)(S.a, S.b+1)
differentiateweightedspacex(S::HalfDiskSpace) =
    (S.family)(S.a-1, S.b-1)
differentiateweightedspacey(S::HalfDiskSpace) =
    (S.family)(S.a, S.b-1)
differentiatex(f::Fun, S::HalfDiskSpace) =
    Fun(differentiatespacex(S), differentiatex(S, f.coefficients))
differentiatey(f::Fun, S::HalfDiskSpace) =
    Fun(differentiatespacey(S), differentiatey(S, f.coefficients))
function differentiatex(S::HalfDiskSpace, cfs::AbstractVector)
    m̃ = length(cfs)
    N = -1 + Int(round(sqrt(1+2(m̃-1))))
    m = Int((N+1)*(N+2)/2)
    if m̃ < m
        resize!(cfs, m)
        cfs[m̃+1:end] .= 0.0
    end
    partialoperatorx(S, N) * cfs
end
function differentiatey(S::HalfDiskSpace, cfs::AbstractVector)
    m̃ = length(cfs)
    N = -1 + Int(round(sqrt(1+2(m̃-1))))
    m = Int((N+1)*(N+2)/2)
    if m̃ < m
        cfs2 = zeros(m)
        cfs2[1:m̃] = cfs
        partialoperatory(S, N) * cfs2
    else
        partialoperatory(S, N) * cfs
    end
end

function getpartialoperatorxval(S::HalfDiskSpace{<:Any, <:Any, T},
                                    ptsp, wp, ptsh, wh, n, k, m, j) where T
    # NOTE: ρ(x) is explicitly assumed to be sqrt(1-x^2), as calling ρ(x) is too
    # expensive
    # We should have already called getopptseval etc
    Sx = differentiatespacex(S)
    P = S.family.P(S.b, S.b)
    Px = Sx.family.P(Sx.b, Sx.b)
    H = S.family.H(S.a, S.b+k+0.5)
    Hx = S.family.H(Sx.a, Sx.b+j+0.5)
    valp = inner2(Px, opevalatpts(P, k+1, ptsp), opevalatpts(Px, j+1, ptsp), wp)
    valh = inner2(H, (ptsh.^2) .* opevalatpts(H, n-k+1, ptsh),
                    (-ptsh.^2 .+ 1).^(0.5k + 0.5j) .* opevalatpts(Hx, m-j+1, ptsh), wh)
    val = valp * inner2(H, ptsh .* derivopevalatpts(H, n-k+1, ptsh),
                        (-ptsh.^2 .+ 1).^(0.5k + 0.5j + 1) .* opevalatpts(Hx, m-j+1, ptsh), wh)
    val -= k * valh * valp
    val += valh * inner2(Px, ptsp .* derivopevalatpts(P, k+1, ptsp),
                            opevalatpts(Px, j+1, ptsp), wp)
    val /= Sx.opnorms[j+1]
    val
end
function partialoperatorx(S::HalfDiskSpace{<:Any, <:Any, T}, N) where T
    # Takes the space P^{a,b} -> P^{a+1,b+1}
    Sx = differentiatespacex(S)
    P = S.family.P(S.b, S.b)
    Px = Sx.family.P(Sx.b, Sx.b)
    ptsp, wp = pointswithweights(Px, N+2) # TODO
    getopptseval(P, N, ptsp)
    getderivopptseval(P, N, ptsp)
    getopptseval(Px, N, ptsp)
    H = S.family.H(S.a, S.b+0.5)
    ptsh, wh = pointswithweights(H, 2N+2)
    getopnorms(Sx, N-1)

    A = BandedBlockBandedMatrix(
        Zeros{T}(sum(1:N),sum(1:(N+1))), (1:N, 1:N+1), (-1,2), (0, 2))

    # Get pt evals for the H OPs
    for k = 0:N
        @show "dxoperator", N, k
        H = S.family.H(S.a, S.b+k+0.5)
        getopptseval(H, N-k, ptsh)
        getderivopptseval(H, N-k, ptsh)
        Hx = Sx.family.H(Sx.a, Sx.b+k+0.5)
        getopptseval(Hx, N-k+1, ptsh)
    end

    n, k = 1, 0
    m, j = n-1, k
    val = getpartialoperatorxval(S, ptsp, wp, ptsh, wh, n, k, m, j)
    view(A, Block(m+1, n+1))[1, 1] = val
    for n = 2:N, k = 0:n
        if k < 2
            inds1 = (n == 2 && k == 1 ? [] : [k])
            inds2 = [k]
        elseif k == n - 1
            inds1 = [k-2]
            inds2 = [k-2, k]
        elseif k == n
            inds1 = [k-2]
            inds2 = [k-2]
        else
            inds1 = [k-2, k]
            inds2 = [k-2, k]
        end
        m = n-2
        for j in inds1
            val = getpartialoperatorxval(S, ptsp, wp, ptsh, wh, n, k, m, j)
            view(A, Block(m+1, n+1))[j+1, k+1] = val
        end
        m = n-1
        for j in inds2
            val = getpartialoperatorxval(S, ptsp, wp, ptsh, wh, n, k, m, j)
            view(A, Block(m+1, n+1))[j+1, k+1] = val
        end
    end
    A
end
function partialoperatory(S::HalfDiskSpace{<:Any, <:Any, T}, N) where T
    # Takes the space P^{a,b} -> P^{a,b+1}
    A = BandedBlockBandedMatrix(
        Zeros{T}(sum(1:N),sum(1:(N+1))), (1:N, 1:N+1), (-1,1), (-1,1))
    Sy = differentiatespacey(S)
    P = S.family.P(S.b, S.b)
    Py = Sy.family.P(Sy.b, Sy.b)
    pts, w = pointswithweights(Py, N)
    getopptseval(P, N, pts)
    getderivopptseval(P, N, pts)
    getopptseval(Py, N, pts)
    getopnorms(Sy, N-1)
    for k = 1:N
        val = (getopnorm(Sy.family.H(S.a, S.b+k+0.5))
                * inner2(Py, derivopevalatpts(P, k+1, pts), opevalatpts(Py, k, pts), w)
                / Sy.opnorms[k])
        for i = k:N
            view(A, Block(i, i+1))[k, k+1] = val
        end
    end
    A
end
function getweightedpartialoperatorxval(S::HalfDiskSpace{<:Any, <:Any, T},
                                    ptsp, wp, ptsh, wh, n, k, m, j) where T
    # NOTE: ρ(x) is explicitly assumed to be sqrt(1-x^2), as calling ρ(x) is too
    # expensive
    # We should have already called getopptseval etc
    Sx = differentiateweightedspacex(S)
    P = S.family.P(S.b, S.b)
    Px = Sx.family.P(Sx.b, Sx.b)
    H = S.family.H(S.a, S.b+k+0.5)
    Hx = S.family.H(Sx.a, Sx.b+j+0.5)
    valp = inner2(Px, (-ptsp.^2 .+ 1) .* opevalatpts(P, k+1, ptsp),
                    opevalatpts(Px, j+1, ptsp), wp)
    valh = inner2(H, (ptsh.^2) .* opevalatpts(H, n-k+1, ptsh),
                    (-ptsh.^2 .+ 1).^(0.5k + 0.5j) .* opevalatpts(Hx, m-j+1, ptsh), wh)
    val = S.a * valp * inner2(H, (-ptsh.^2 .+ 1).^(0.5k + 0.5j + 1) .* opevalatpts(H, n-k+1, ptsh),
                                opevalatpts(Hx, m-j+1, ptsh), wh)
    val += - 2S.b * valh * inner2(Px, opevalatpts(P, k+1, ptsp),
                                    opevalatpts(Px, j+1, ptsp), wp)
    val += valp * inner2(H, ptsh .* derivopevalatpts(H, n-k+1, ptsh),
                            (-ptsh.^2 .+ 1).^(0.5k + 0.5j + 1) .* opevalatpts(Hx, m-j+1, ptsh), wh)
    val -= k * valh * valp
    val += valh * inner2(Px, ptsp .* derivopevalatpts(P, k+1, ptsp),
                            (-ptsp.^2 .+ 1) .* opevalatpts(Px, j+1, ptsp), wp)
    val /= Sx.opnorms[j+1]
    val
end
function weightedpartialoperatorx(S::HalfDiskSpace{<:Any, <:Any, T}, N) where T
    # Takes weighted space ∂/∂x(W^{a,b}) -> W^{a-1,b-1}
    W = BandedBlockBandedMatrix(
        Zeros{T}(sum(1:(N+3)),sum(1:(N+1))), (1:N+3, 1:N+1), (2,-1), (2,0))
    Sx = differentiateweightedspacex(S)
    P = S.family.P(S.b, S.b)
    Px = Sx.family.P(Sx.b, Sx.b)
    ptsp, wp = pointswithweights(Px, N+3)
    getopptseval(P, N, ptsp)
    getopptseval(Px, N+1, ptsp)
    getderivopptseval(P, N, ptsp)
    ptsh, wh = pointswithweights(S.family.H(S.a-1, S.b-0.5), 2N+2)
    getopnorms(Sx, N+2)

    # Get pt evals for the H OPs
    for k = 0:N
        H = S.family.H(S.a, S.b+k+0.5)
        getopptseval(H, N-k, ptsh)
        getderivopptseval(H, N-k, ptsh)
        for j = k:2:k+2
            Hx = Sx.family.H(Sx.a, Sx.b+j+0.5)
            getopptseval(Hx, N-k+1, ptsh)
        end
    end
    for n = 0:N, k = 0:n
        for m = n+1:n+2, j = k:2:min(m,k+2)
            val = getweightedpartialoperatorxval(S, ptsp, wp, ptsh,
                                                    wh, n, k, m, j)
            view(W, Block(m+1, n+1))[j+1, k+1] = val
        end
    end
    W
end
function weightedpartialoperatory(S::HalfDiskSpace{<:Any, <:Any, T}, N) where T
    # Takes weighted space ∂/∂y(W^{a,b}) -> W^{a,b-1}
    W = BandedBlockBandedMatrix(
        Zeros{T}(sum(1:(N+2)),sum(1:(N+1))), (1:N+2, 1:N+1), (1,-1), (1,-1))
    Sy = differentiateweightedspacey(S)
    P = S.family.P(S.b, S.b)
    Py = Sy.family.P(Sy.b, Sy.b)
    ptsp, wp = pointswithweights(Py, N+2)
    getopptseval(P, N, ptsp)
    getopptseval(Py, N+1, ptsp)
    getderivopptseval(P, N, ptsp)
    getopnorms(Sy, N+1)

    n, m = N, N+1
    for k = 0:N
        j = k + 1
        val = - 2S.b * inner2(P, ptsp .* opevalatpts(P, k+1, ptsp),
                                opevalatpts(Py, j+1, ptsp), wp)
        val += inner2(P, (-ptsp.^2 .+ 1) .* derivopevalatpts(P, k+1, ptsp),
                        opevalatpts(Py, j+1, ptsp), wp)
        val *= getopnorm(Sy.family.H(S.a, S.b+k+0.5))
        val /= Sy.opnorms[j+1]
        for i = k:N
            view(W, Block(i+2, i+1))[k+2, k+1] = val
        end
    end
    W
end

function transformparamsoperator(S::HalfDiskSpace{<:Any, <:Any, T},
            St::HalfDiskSpace{<:Any, <:Any, T}, N; weighted=false) where T
    # Cases we can handle:
    # Case 1: Takes the space P^{a,b} -> P^{a+1,b}
    if weighted == false && St.a == S.a+1 && St.b == S.b
        # Outputs the relevant sum(1:N+1) × sum(1:N+1) matrix operator
        C = BandedBlockBandedMatrix(Zeros{T}(sum(1:(N+1)),sum(1:(N+1))),
                                    (1:N+1, 1:N+1), (0,1), (0,0))
        P = S.family.P(S.b, S.b)
        ptsh, wh = pointswithweights(St.family.H(St.a, St.b + 0.5), N+1)
        rho2ptsh = -ptsh.^2 .+ 1
        getopnorms(St, N)

        # Get pt evals for H OPs
        for k = 0:N
            H = S.family.H(S.a, S.b+k+0.5)
            Ht = St.family.H(St.a, St.b+k+0.5)
            getopptseval(H, N-k, ptsh)
            getopptseval(Ht, N-k, ptsh)
        end

        n, k = 0, 0; m = n
        view(C, Block(m+1, n+1))[k+1, k+1] = sum(wh) * getopnorm(P) / St.opnorms[k+1]
        for n=1:N, k=0:n
            H = S.family.H(S.a, S.b+k+0.5)
            Ht = St.family.H(St.a, St.b+k+0.5)
            M = (k == n ? [n] : [n-1, n])
            for m in M
                val = inner2(H, opevalatpts(H, n-k+1, ptsh),
                                rho2ptsh.^k .* opevalatpts(Ht, m-k+1, ptsh), wh)
                val *= getopnorm(P)
                view(C, Block(m+1, n+1))[k+1, k+1] = val / St.opnorms[k+1]
            end
        end
        C
    # Case 2: Takes the space P^{a,b} -> P^{a,b+1}
    # and Case 3: Takes the space P^{a,b} -> P^{a+1,b+1}
    elseif ((weighted == false && St.a == S.a && St.b == S.b+1)
            || (weighted == false && St.a == S.a+1 && St.b == S.b+1))
        # Outputs the relevant sum(1:N+1) × sum(1:N+1) matrix operator
        C = BandedBlockBandedMatrix(Zeros{T}(sum(1:(N+1)),sum(1:(N+1))),
                                    (1:N+1, 1:N+1), (0,3), (0,2))
        P = S.family.P(S.b, S.b)
        Pt = S.family.P(St.b, St.b)
        ptsp, wp = pointswithweights(Pt, N+2)
        ptsh, wh = pointswithweights(St.family.H(St.a, St.b + 0.5), N+1)
        rho2ptsh = -ptsh.^2 .+ 1
        # Get pt evals for 1D OPs
        getopptseval(P, N, ptsp)
        getopptseval(Pt, N, ptsp)
        getopnorms(St, N)
        for k = 0:N
            H = S.family.H(S.a, S.b+k+0.5)
            Ht = St.family.H(St.a, St.b+k+0.5)
            getopptseval(H, N-k, ptsh)
            getopptseval(Ht, N-k, ptsh)
        end
        for n=0:N, k=0:n
            H = S.family.H(S.a, S.b+k+0.5)
            for m = n-3:n, j = k-2:2:k
                if m ≥ 0 && 0 ≤ j ≤ m
                    Ht = St.family.H(St.a, St.b+j+0.5)
                    val = inner2(H, opevalatpts(H, n-k+1, ptsh),
                                    rho2ptsh.^Int(0.5*(k+j)) .* opevalatpts(Ht, m-j+1, ptsh), wh)
                    val *= inner2(P, opevalatpts(P, k+1, ptsp), opevalatpts(Pt, j+1, ptsp), wp)
                    view(C, Block(m+1, n+1))[j+1, k+1] = val / St.opnorms[j+1]
                end
            end
        end
        C
    # Case 4: Takes the space W^{a,b} -> W^{a-1,b}
    elseif weighted == true && St.a == S.a-1 && St.b == S.b
        # Outputs the relevant sum(1:N+2) × sum(1:N+1) matrix operator
        C = BandedBlockBandedMatrix(Zeros{T}(sum(1:(N+2)),sum(1:(N+1))),
                                    (1:N+2, 1:N+1), (1,0), (0,0))
        P = S.family.P(S.b, S.b)
        ptsh, wh = pointswithweights(S.family.H(S.a, S.b + 0.5), N+1)
        getopnorms(St, N+1)

        # Get pt evals for H OPs
        for k = 0:N
            H = S.family.H(S.a, S.b+k+0.5)
            Ht = St.family.H(St.a, St.b+k+0.5)
            getopptseval(H, N-k, ptsh)
            getopptseval(Ht, N-k+1, ptsh)
        end

        for n=0:N, k=0:n
            H = S.family.H(S.a, S.b+k+0.5)
            Ht = St.family.H(St.a, St.b+k+0.5)
            for m = n:n+1
                val = inner2(H, opevalatpts(H, n-k+1, ptsh),
                                (-ptsh.^2 .+ 1).^k .* opevalatpts(Ht, m-k+1, ptsh), wh)
                val *= getopnorm(P)
                view(C, Block(m+1, n+1))[k+1, k+1] = val / St.opnorms[k+1]
            end
        end
        C
    # Case 5: Takes the space W^{a,b} -> W^{a,b-1}
    # and Case 6: Takes the space W^{a,b} -> W^{a-1,b-1}
    elseif ((weighted == true && St.a == S.a && St.b == S.b-1)
            || (weighted == true && St.a == S.a-1 && St.b == S.b-1))
        # Outputs the relevant sum(1:N+4) × sum(1:N+1) matrix operator
        C = BandedBlockBandedMatrix(Zeros{T}(sum(1:(N+4)),sum(1:(N+1))),
                                    (1:N+4, 1:N+1), (3,0), (2,0))
        P = S.family.P(S.b, S.b)
        Pt = S.family.P(St.b, St.b)
        ptsp, wp = pointswithweights(P, N+2)
        ptsh, wh = pointswithweights(S.family.H(S.a, S.b + 0.5), N+1)
        rho2ptsh = -ptsh.^2 .+ 1
        getopnorms(St, N+3)

        # Get pt evals for P and H OPs
        getopptseval(P, N, ptsp)
        getopptseval(Pt, N+3, ptsp)
        for k = 0:N
            H = S.family.H(S.a, S.b+k+0.5)
            getopptseval(H, N-k, ptsh)
        end
        for j = 0:N+3
            Ht = St.family.H(St.a, St.b+j+0.5)
            getopptseval(Ht, N+3-j, ptsh)
        end

        for n=0:N, k=0:n
            H = S.family.H(S.a, S.b+k+0.5)
            for m = n:n+3, j = k:2:min(k+2, m)
                Ht = St.family.H(St.a, St.b+j+0.5)
                val = inner2(H, opevalatpts(H, n-k+1, ptsh),
                                rho2ptsh.^Int(0.5(k+j)) .* opevalatpts(Ht, m-j+1, ptsh), wh)
                val *= inner2(P, opevalatpts(P, k+1, ptsp), opevalatpts(Pt, j+1, ptsp), wp)
                view(C, Block(m+1, n+1))[j+1, k+1] = val / St.opnorms[j+1]
            end
        end
        C
    else
        error("Invalid HalfDiskSpace")
    end
end

function laplaceoperator(S::HalfDiskSpace{<:Any, <:Any, T},
            St::HalfDiskSpace{<:Any, <:Any, T}, N;
            weighted=false, square=true) where T
    # Outputs the sum(1:N+1) × sum(1:N+1) matrix operator if square=true
    # TODO: Is it more efficient to allocate blocks manually?
    D = S.family
    if (weighted == true && Int(S.a) == 1 && Int(S.b) == 1
            && Int(St.a) == 1 && Int(St.b) == 1)
        A = partialoperatorx(D(S.a-1,S.b-1), N+2)
        @show "laplaceoperator", "1 of 6 done"
        B = weightedpartialoperatorx(D(S.a,S.b), N)
        @show "laplaceoperator", "2 of 6 done"
        C = transformparamsoperator(D(S.a-1,S.b), D(S.a,S.b), N+1)
        @show "laplaceoperator", "3 of 6 done"
        E = partialoperatory(D(S.a-1,S.b-1), N+2)
        @show "laplaceoperator", "4 of 6 done"
        F = transformparamsoperator(D(S.a,S.b-1), D(S.a-1,S.b-1), N+1, weighted=true)
        @show "laplaceoperator", "5 of 6 done"
        G = weightedpartialoperatory(D(S.a,S.b), N)
        @show "laplaceoperator", "6 of 6 done"
        L = A * B + C * E * F * G
        if square
            m = sum(1:(N+1))
            Δ = BandedBlockBandedMatrix(L[1:m, 1:m], (1:N+1, 1:N+1), (1,1), (2,2))
        else
            L
        end
    elseif (weighted == true && Int(S.a) == 2 && Int(S.b) == 2
            && Int(St.a) == 0 && Int(St.b) == 0)
        A = weightedpartialoperatorx(D(S.a-1,S.b-1), N+2)
        @show "laplaceoperator", "1 of 6 done"
        B = weightedpartialoperatorx(D(S.a,S.b), N)
        @show "laplaceoperator", "2 of 6 done"
        C = transformparamsoperator(D(S.a-1,S.b-2), D(S.a-2,S.b-2), N+3, weighted=true)
        @show "laplaceoperator", "3 of 6 done"
        E = weightedpartialoperatory(D(S.a-1,S.b-1), N+2)
        @show "laplaceoperator", "4 of 6 done"
        F = transformparamsoperator(D(S.a,S.b-1), D(S.a-1,S.b-1), N+1, weighted=true)
        @show "laplaceoperator", "5 of 6 done"
        G = weightedpartialoperatory(D(S.a,S.b), N)
        @show "laplaceoperator", "6 of 6 done"
        L = A * B + C * E * F * G
        if square
            m = sum(1:(N+1))
            Δ = BandedBlockBandedMatrix(L[1:m, 1:m], (1:N+1, 1:N+1), (4,-2), (4,0))
        else
            L
        end
    elseif (weighted == false && Int(S.a) == 0 && Int(S.b) == 0
            && Int(St.a) == 2 && Int(St.b) == 2)
        A = partialoperatorx(D(S.a+1,S.b+1), N+1)
        @show "laplaceoperator", "1 of 6 done"
        B = partialoperatorx(D(S.a,S.b), N+2)
        @show "laplaceoperator", "2 of 6 done"
        C = transformparamsoperator(D(S.a+1,S.b+2), D(S.a+2,S.b+2), N)
        @show "laplaceoperator", "3 of 6 done"
        E = partialoperatory(D(S.a+1,S.b+1), N+1)
        @show "laplaceoperator", "4 of 6 done"
        F = transformparamsoperator(D(S.a,S.b+1), D(S.a+1,S.b+1), N+1)
        @show "laplaceoperator", "5 of 6 done"
        G = partialoperatory(D(S.a,S.b), N+2)
        @show "laplaceoperator", "6 of 6 done"
        L = BandedBlockBandedMatrix(sparse(sparse(A * B) + C * E * F * G), (1:N+1, 1:N+3), (-2,4), (0,4))
        if square
            m = sum(1:(N+1))
            Δ = BandedBlockBandedMatrix(L[1:m, 1:m], (1:N+1, 1:N+1), (-2,4), (0,4))
        else
            L
        end
    else
        error("Invalid HalfDiskSpace for Laplacian operator")
    end
end

function biharmonicoperator(S::HalfDiskSpace{<:Any, <:Any, T}, N; square=true) where T
    D = S.family
    if Int(S.a) == 2 && Int(S.b) == 2
        B = (laplaceoperator(D(S.a-2, S.b-2), S, N+2; square=false)
                * laplaceoperator(S, D(S.a-2, S.b-2), N; weighted=true, square=false))
        if square
            m = sum(1:(N+1))
            Δ = BandedBlockBandedMatrix(B[1:m, 1:m], (1:N+1, 1:N+1), (2,2), (4,4))
        else
            B
        end
    else
        error("Invalid HalfDiskSpace for Laplacian operator")
    end
end


# Resize the coeffs vector to be of the expected/standard length for a degree N
# expansion (so we can apply operators).
function resizecoeffs!(f::Fun, N)
    cfs = f.coefficients
    m̃ = length(cfs)
    m = Int((N+1)*(N+2)/2)
    if m̃ < m
        resize!(cfs, m)
        cfs[m̃+1:end] .= 0.0
    elseif m̃ > m
        for j = m+1:m̃
            if cfs[j] > 1e-15
                error("Trying to decrease degree of f")
            end
        end
        resize!(cfs, m)
    end
    cfs
end

#===#
# Help working out indices for a vector of coeffs, and corresponding n,k values
getopindex(n, k) = sum(1:n)+k+1
function getnk(j)
    i = j - 1
    n = 0
    m = 0
    while true
        if (m+1)*(m+2) > 2i
            n = m
            break
        end
        m += 1
    end
    k = i - (n+1)n/2
    Int(n), Int(k)
end
#===#

# Operator Clenshaw
function operatorclenshawG(S::HalfDiskSpace{<:Any, <:Any, T}, n, Jx, Jy, zeromat) where T
    G = Matrix{SparseMatrixCSC{T}}(undef, 2(n+1), n+1)
    for i = 1:n+1
        for j = 1:n+1
            if i == j
                G[i,j] = Jx
                G[i+n+1,j] = Jy
            else
                G[i,j] = zeromat
                G[i+n+1,j] = zeromat
            end
        end
    end
    G
end
function operatorclenshawvector(S::HalfDiskSpace{<:Any, <:Any, T}, v, id) where T
    s = size(v)[1]
    B = Array{SparseMatrixCSC{T}}(undef, (1, s))
    for i = 1:s
        B[1,i] = id * v[i]
    end
    B
end
function operatorclenshawmatrixDT(S::HalfDiskSpace{<:Any, <:Any, T}, A, id) where T
    B = Array{SparseMatrixCSC{T}}(undef, size(A))
    for ij = 1:length(A)
        B[ij] = id * A[ij]
    end
    B
end
function operatorclenshawmatrixBmG(S::HalfDiskSpace{<:Any, <:Any, T}, A, id, Jx, Jy) where T
    ii, jj = size(A)
    B = Array{SparseMatrixCSC{T}}(undef, (ii, jj))
    for i = 1:jj, j = 1:jj
        if i == j
            B[i, j] = (id * A[i, j]) - Jx
            B[i+jj, j] = (id * A[i+jj, j]) - Jy
        else
            B[i,j] = id * A[i, j]
            B[i+jj, j] = id * A[i+jj, j]
        end
    end
    B
end
function operatorclenshaw(cfs, S::HalfDiskSpace, N)
    # Outputs the operator NxN-blocked matrix operator corresponding to the
    # function f given by its coefficients of its expansion in the space S
    m̃ = length(cfs)
    M = getnk(m̃)[1] # Degree of f
    m = getopindex(M, M)
    if m̃ < getopindex(M, M)
        # Pad cfs to correct size
        resize!(cfs, m)
        cfs[m̃+1:end] .= 0.0
    end
    resizedata!(S, M)
    Jx = sparse(jacobix(S, M))
    Jy = sparse(jacobiy(S, M))

    @show "Operator Clenshaw"
    P0 = 1.0
    if M == 0
        return cfs[1] * id * P0
    end
    id = sparse(I, size(Jx))
    if M == 1
        P0 * (operatorclenshawvector(S, cfs[1], id)[1] - (operatorclenshawmatrixDT(S, S.DT[1], id) * operatorclenshawmatrixBmG(S, S.B[1], id, Jx, Jy))[1])
    end
    n = M; @show "Operator Clenshaw", N, n
    inds2 = m-M:m
    γ2 = operatorclenshawvector(S, view(cfs, inds2), id)
    n = M - 1; @show "Operator Clenshaw", M, n
    inds1 = (m-2M):(m-M-1)
    γ1 = (operatorclenshawvector(S, view(cfs, inds1), id)
        - γ2 * operatorclenshawmatrixDT(S, S.DT[M], id) * operatorclenshawmatrixBmG(S, S.B[M], id, Jx, Jy))
    for n = M-2:-1:0
        @show "Operator Clenshaw", N, n
        ind = sum(1:n)
        γ = (operatorclenshawvector(S, view(cfs, ind+1:ind+n+1), id)
             - γ1 * operatorclenshawmatrixDT(S, S.DT[n+1], id) * operatorclenshawmatrixBmG(S, S.B[n+1], id, Jx, Jy)
             - γ2 * operatorclenshawmatrixDT(S, S.DT[n+2] * S.C[n+2], id))
        γ2 = copy(γ1)
        γ1 = copy(γ)
    end
    # Resize the resulting operator to be sum(1:N+1)xsum(1:N+1)
    if N > M
        nn = getopindex(N, N)
        ret = spzeros(nn, nn)
        view(ret, 1:m, 1:m) .= (γ1 * P0)[1]
    else
        ret = (γ1 * P0)[1]
    end
    ret
end
operatorclenshaw(f::Fun, S::HalfDiskSpace) = operatorclenshaw(f.coefficients, S, getnk(ncoefficients(f))[1])
operatorclenshaw(f::Fun, S::HalfDiskSpace, N) = operatorclenshaw(f.coefficients, S, N)
operatorclenshaw(f::Fun) = operatorclenshaw(f, f.space)
operatorclenshaw(f::Fun, N) = operatorclenshaw(f, f.space, N)


# Method to gather and evaluate the ops of space S at the transform pts given
function getopptseval(S::HalfDiskSpace, N, pts)
    resetopptseval(S)
    jj = [getopindex(n, 0) for n=0:N]
    for j in jj
        opevalatpts(S, j, pts)
    end
    S.opptseval
end
function opevalatpts(S::HalfDiskSpace{<:Any, <:Any, T}, j, pts) where T
    len = length(S.opptseval)
    if len ≥ j
        return S.opptseval[j]
    end

    # We iterate up from the last obtained pts eval
    N = len == 0 ? -1 : getnk(len)[1]
    n = getnk(j)[1]
    if  N != n - 1 || (len == 0 && j > 1)
        error("Invalid index")
    end

    jj = getopindex(n, 0)
    resizedata!(S, n)
    resize!(S.opptseval, getopindex(n, n))
    for k = 0:n
        S.opptseval[jj+k] = Vector{T}(undef, length(pts))
    end

    if n == 0
        S.opptseval[1][:] .= 1.0
    elseif n == 1
        nm1 = getopindex(n-1, 0)
        for r = 1:length(pts)
            P1 = [opevalatpts(S, nm1+i, pts)[r] for i = 0:n-1]
            P = - S.DT[n] * (S.B[n] - clenshawG(n-1, pts[r])) * P1
            for k = 0:n
                S.opptseval[jj+k][r] = P[k+1]
            end
        end
    else
        nm1 = getopindex(n-1, 0)
        nm2 = getopindex(n-2, 0)
        for r = 1:length(pts)
            P1 = [opevalatpts(S, nm1+i, pts)[r] for i = 0:n-1]
            P2 = [opevalatpts(S, nm2+i, pts)[r] for i = 0:n-2]
            P = (- S.DT[n] * (S.B[n] - clenshawG(n-1, pts[r])) * P1
                 - S.DT[n] * S.C[n] * P2)
            for k = 0:n
                S.opptseval[jj+k][r] = P[k+1]
            end
        end
    end
    S.opptseval[j]
end
resetopptseval(S::HalfDiskSpace) = resize!(S.opptseval, 0)

# Methods to gather and evaluate the derivatives of the ops of space S at the
# transform pts given
function clenshawGtildex(n, z)
    sp = sparse(I, n+1, n+1)
    [sp; 0.0 * sp]
end
function getxderivopptseval(S::HalfDiskSpace, N, pts)
    resetxderivopptseval(S)
    jj = [getopindex(n, 0) for n=0:N]
    for j in jj
        xderivopevalatpts(S, j, pts)
    end
    S.xderivopptseval
end
resetxderivopptseval(S::HalfDiskSpace) = resize!(S.xderivopptseval, 0)
function xderivopevalatpts(S::HalfDiskSpace{<:Any, <:Any, T}, j, pts) where T
    len = length(S.xderivopptseval)
    if len ≥ j
        return S.xderivopptseval[j]
    end

    # We iterate up from the last obtained pts eval
    N = len == 0 ? -1 : getnk(len)[1]
    n = getnk(j)[1]
    if  N != n - 1 || (len == 0 && j > 1)
        error("Invalid index")
    end

    jj = getopindex(n, 0)
    resizedata!(S, n)
    resize!(S.xderivopptseval, getopindex(n, n))
    for k = 0:n
        S.xderivopptseval[jj+k] = Vector{T}(undef, length(pts))
    end

    if n == 0
        S.xderivopptseval[1][:] .= 0.0
    elseif n == 1
        nm1 = getopindex(n-1, 0)
        for r = 1:length(pts)
            P1 = [opevalatpts(S, nm1+i, pts)[r] for i = 0:n-1]
            dxP = S.DT[n] * clenshawGtildex(n-1, pts[r]) * P1
            for k = 0:n
                S.xderivopptseval[jj+k][r] = dxP[k+1]
            end
        end
    else
        nm1 = getopindex(n-1, 0)
        nm2 = getopindex(n-2, 0)
        for r = 1:length(pts)
            dxP1 = [xderivopevalatpts(S, nm1+i, pts)[r] for i = 0:n-1]
            dxP2 = [xderivopevalatpts(S, nm2+i, pts)[r] for i = 0:n-2]
            P1 = [opevalatpts(S, nm1+i, pts)[r] for i = 0:n-1]
            dxP = (- S.DT[n] * (S.B[n] - clenshawG(n-1, pts[r])) * dxP1
                   - S.DT[n] * S.C[n] * dxP2
                   + S.DT[n] * clenshawGtildex(n-1, pts[r]) * P1)
            for k = 0:n
                S.xderivopptseval[jj+k][r] = dxP[k+1]
            end
        end
    end
    S.xderivopptseval[j]
end
function clenshawGtildey(n, z)
    sp = sparse(I, n+1, n+1)
    [0.0 * sp; sp]
end
function getyderivopptseval(S::HalfDiskSpace, N, pts)
    resetyderivopptseval(S)
    jj = [getopindex(n, 0) for n=0:N]
    for j in jj
        yderivopevalatpts(S, j, pts)
    end
    S.yderivopptseval
end
resetyderivopptseval(S::HalfDiskSpace) = resize!(S.yderivopptseval, 0)
function yderivopevalatpts(S::HalfDiskSpace{<:Any, <:Any, T}, j, pts) where T
    len = length(S.yderivopptseval)
    if len ≥ j
        return S.yderivopptseval[j]
    end

    # We iterate up from the last obtained pts eval
    N = len == 0 ? -1 : getnk(len)[1]
    n = getnk(j)[1]
    if  N != n - 1 || (len == 0 && j > 1)
        error("Invalid index")
    end

    jj = getopindex(n, 0)
    resizedata!(S, n)
    resize!(S.yderivopptseval, getopindex(n, n))
    for k = 0:n
        S.yderivopptseval[jj+k] = Vector{T}(undef, length(pts))
    end

    if n == 0
        S.yderivopptseval[1][:] .= 0.0
    elseif n == 1
        nm1 = getopindex(n-1, 0)
        for r = 1:length(pts)
            P1 = [opevalatpts(S, nm1+i, pts)[r] for i = 0:n-1]
            dyP = S.DT[n] * clenshawGtildey(n-1, pts[r]) * P1
            for k = 0:n
                S.yderivopptseval[jj+k][r] = dyP[k+1]
            end
        end
    else
        nm1 = getopindex(n-1, 0)
        nm2 = getopindex(n-2, 0)
        for r = 1:length(pts)
            dyP1 = [yderivopevalatpts(S, nm1+i, pts)[r] for i = 0:n-1]
            dyP2 = [yderivopevalatpts(S, nm2+i, pts)[r] for i = 0:n-2]
            P1 = [opevalatpts(S, nm1+i, pts)[r] for i = 0:n-1]
            dyP = (- S.DT[n] * (S.B[n] - clenshawG(n-1, pts[r])) * dyP1
                   - S.DT[n] * S.C[n] * dyP2
                   + S.DT[n] * clenshawGtildey(n-1, pts[r]) * P1)
            for k = 0:n
                S.yderivopptseval[jj+k][r] = dyP[k+1]
            end
        end
    end
    S.yderivopptseval[j]
end

weightderivativex(S::HalfDiskSpace, x, y) =
    x^(S.a-1) * (1 - x^2 - y^2)^(S.b-1) * (1 - (2*S.b + 1)*x^2 - y^2)
weightderivativex(S::HalfDiskSpace, z) = weightderivativex(S, z[1], z[2])
weightderivativey(S::HalfDiskSpace, x, y) =
    - 2S.b * x^S.a * y * (1 - x^2 - y^2)^(S.b-1)
weightderivativey(S::HalfDiskSpace, z) = weightderivativey(S, z[1], z[2])



end # module
