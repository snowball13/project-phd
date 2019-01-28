module OrthogonalPolynomialFamilies

using ApproxFun
    import ApproxFun: evaluate, PolynomialSpace, recα, recβ, recγ, recA, recB, recC, domain,
                        domainspace, rangespace, bandwidths, prectype, canonicaldomain, tocanonical,
                        spacescompatible, points, transform, itransform, AbstractProductSpace, tensorizer,
                        columnspace, checkpoints, plan_transform, clenshaw
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
function lanczos2!(S, pts, w, F, N₀=0)
    # F is the added factor as a Fun
    P, β, γ = S.ops, S.a, S.b

    # x * P[n](x) == (γ[n] * P[n+1](x) + β[n] * P[n](x) + γ[n-1] * P[n-1](x))

    # We use our own quadrature rule here, as given by pts and w

    N = length(β)
    X = Fun(identity, domain(S.weight))

    if N₀ <= 0
        N₀ = 1
        f1 = Fun(1 / sqrt(sum(S.weight)), space(X))
        P[1] = f1
        v = X * P[1]
        β[1] = inner(S, v * F, P[1], pts, w)
        v = v - β[1] * P[1]
        γ[1] = sqrt(inner(S, v * F, v, pts, w))
        P[2] = v / γ[1]
    end

    for k = N₀+1:N
        v = X * P[k] - γ[k-1] * P[k-1]
        β[k] = inner(S, v * F, P[k], pts, w)
        v = v - β[k] * P[k]
        γ[k] = sqrt(inner(S, v * F, v, pts, w))
        P[k+1] = v / γ[k]
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

# Decides which lanczos! to call depending on if we would run into
# ill-conditioned errors etc., and then calls it
function lanczosdecider(S, N₀)
    # TODO: Make this cleaner...
    ϕ = Vector{Float64}(undef, length(S.params))
    ψ = []
    for i = 1:length(S.params)
        α = S.params[i]
        if floor(α) == α || α < 1
            ϕ[i] = α
        else # We have non-integer parameter
            ϕ[i] = α - floor(α)
            append!(ψ, i)
        end
    end
    if length(ψ) == 0
        lanczos!(S.weight, S.ops, S.a, S.b, N₀)
    elseif length(ψ) == 1
        m = 100 # TODO
        pts, w = pointswithweights((S.family)(ϕ...), m)
        fctr = ((S.family).factors[ψ[1]])^(S.params[ψ[1]]-ϕ[ψ[1]])
        lanczos2!(S, pts, w, fctr, N₀)
    else
        m = 100 # TODO
        pts, w = pointswithweights((S.family)(ϕ...), m)
        pwrs = [S.params[i]-ϕ[i] for i in ψ]
        fctrs = [(S.family).factors[i] for i in ψ]
        fctr = prod(fctrs.^pwrs)
        lanczos2!(S, pts, w, fctr, N₀)
    end
end


abstract type SpaceFamily{D,R} end

struct OrthogonalPolynomialSpace{F,WW,D,R,N,FN} <: PolynomialSpace{D,R}
    family::F # Pointer back to the family
    weight::WW # The full product weight
    params::NTuple{N,R} # The powers of the weight factors (i.e. key to this
                        # space in the dict of the family)
    a::Vector{R} # Diagonal recurrence coefficients
    b::Vector{R} # Off diagonal recurrence coefficients
    ops::Vector{FN} # Cache the Funs given back from lanczos
    opptseval::Vector{Vector{R}}
end

domain(S::OrthogonalPolynomialSpace) = domain(S.weight)
canonicaldomain(S::OrthogonalPolynomialSpace) = domain(S)
tocanonical(S::OrthogonalPolynomialSpace, x) = x


OrthogonalPolynomialSpace(fam::SpaceFamily{D,R}, w::Fun, α::NTuple{N,R}) where {D,R,N} =
    OrthogonalPolynomialSpace{typeof(fam),typeof(w),D,R,N,Fun}(
        fam, w, α, Vector{R}(), Vector{R}(), Vector{Fun}(), Vector{Vector{R}}())

function resizedata!(S::OrthogonalPolynomialSpace, n)
    N₀ = length(S.a) - 1
    n ≤ N₀ + 1 && return S
    resize!(S.a, n)
    resize!(S.b, n)
    resize!(S.ops, n + 1)
    S.ops[:], S.a[:], S.b[:] = lanczosdecider(S, N₀)
    S
end

# R is range-type, which should be Float64.
struct OrthogonalPolynomialFamily{FF,WW,D,R,N} <: SpaceFamily{D,R}
    factors::FF
    spaces::Dict{NTuple{N,R}, OrthogonalPolynomialSpace}
end

function OrthogonalPolynomialFamily(w::Vararg{Fun{<:Space{D,R}},N}) where {D,R,N}
    all(domain.(w) .== Ref(domain(first(w)))) || throw(ArgumentError("domains incompatible"))
    WW = Fun
    spaces = Dict{NTuple{N,R}, OrthogonalPolynomialSpace{OrthogonalPolynomialFamily{typeof(w),WW,D,R,N},WW,D,R}}()
    OrthogonalPolynomialFamily{typeof(w),WW,D,R,N}(w, spaces)
end

function (P::OrthogonalPolynomialFamily{<:Any,<:Any,<:Any,R,N})(α::Vararg{R,N}) where {R,N}
    if length(α) == 1
        haskey(P.spaces,α) && return P.spaces[α]
        P.spaces[α] = OrthogonalPolynomialSpace(P, (P.factors.^α)[1], α)
    else
        haskey(P.spaces,α) && return P.spaces[α]
        P.spaces[α] = OrthogonalPolynomialSpace(P, prod(P.factors.^α), α)
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
function golubwelsch(S::OrthogonalPolynomialSpace, N::Integer)
    # Golub--Welsch algorithm. Used here for N<=20.
    resizedata!(S, N)       # 3-term recurrence
    J = SymTridiagonal(S.a[1:N], S.b[1:N-1])   # Jacobi matrix
    D, V = eigen(J)                     # Eigenvalue decomposition
    indx = sortperm(D)                  # Hermite points
    μ = sum(S.weight)                          # Integral of weight function
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
    Float64(sum(f.(BigFloat.(pts)) .* g.(BigFloat.(pts)) .* BigFloat.(w)))

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


# Method to gather and evaluate the ops of space S at the transform pts given
function getopptseval(S::OrthogonalPolynomialSpace, N, pts)
    resetopptseval(S)
    for n = 0:N
        opevalatpts(S, n+1, pts)
    end
    S.opptseval
end
function opevalatpts(S::OrthogonalPolynomialSpace, j, pts)
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
    S.opptseval[j] = Vector{Float64}(undef, length(pts))

    # p_{n+1} = (A_n x + B_n)p_n - C_n p_{n-1}
    T = Float64
    n = j - 1
    if n == 0
        S.opptseval[j][:] .= 1.0
    elseif n == 1
        A = recA(T, S, n)
        B = recB(T, S, n)
        P1 = opevalatpts(S, j-1, pts)
        for r = 1:length(pts)
            S.opptseval[j][r] = (A * pts[r] + B) * P1[r]
        end
    else
        A = recA(T, S, n)
        B = recB(T, S, n)
        C = recC(T, S, n)
        P1 = opevalatpts(S, j-1, pts)
        P2 = opevalatpts(S, j-2, pts)
        for r = 1:length(pts)
            S.opptseval[j][r] = (A * pts[r] + B) * P1[r] - C * P2[r]
        end
    end
    S.opptseval[j]
end
resetopptseval(S::OrthogonalPolynomialSpace) = resize!(S.opptseval, 0)


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

function HalfDiskSpace(fam::DiskSpaceFamily{R}, a::R, b::R) where R
    X = Fun(identity, 0..1)
    Y = Fun(identity, -1..1)
    H = OrthogonalPolynomialFamily(X, 1-X^2)
    P = OrthogonalPolynomialFamily(1+Y, 1-Y)
    ρ = sqrt(1 - X^2)
    HalfDiskSpace{typeof(fam), typeof(a), typeof(H), typeof(ρ)}(
        fam, a, b, H, P, ρ, Vector{R}(), Vector{Vector{R}}(),
        Vector{Vector{R}}(), Vector{Vector{R}}(), Vector{SparseMatrixCSC{R}}(),
        Vector{SparseMatrixCSC{R}}(), Vector{SparseMatrixCSC{R}}(),
        Vector{SparseMatrixCSC{R}}())
end
HalfDiskSpace() = HalfDiskSpace(1.0, 1.0)

in(x::SVector{2}, D::HalfDisk) = 0 ≤ x[1] ≤ 1 && -sqrt(1-x[1]^2) ≤ x[2] ≤ sqrt(1-x[1]^2)

spacescompatible(A::HalfDiskSpace, B::HalfDiskSpace) = (A.a == B.a && A.b == B.b)

weight(S::HalfDiskSpace, x, y) = x^S.a * (1 - x^2 - y^2)^S.b
weight(S::HalfDiskSpace, z) = weight(S, z[1], z[2])

# NOTE we output ≈n points (x,y), plus the ≈n points corresponding to (x,-y)
function pointswithweights(S::HalfDiskSpace, n)
    # Return the weights and nodes to use for the even
    # of a function, i.e. for the half disk Ω:
    #   int_Ω W^{a,b}(x,y) f(x,y) dydx ≈ Σ_j weⱼ*fe(xⱼ,yⱼ)
    # NOTE: the odd part of the quad rule will equal 0 for polynomials,
    #       so can be ignored.
    N = Int(ceil(sqrt(n)))
    t, wt = pointswithweights((S.P)(S.b, S.b), N)
    s, ws = pointswithweights((S.H)(S.a, S.b + 0.5), N)
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

# NOTE these are squared norms
function getopnorms(S::HalfDiskSpace, m)
    m̃ = length(S.opnorms)
    if m > m̃
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

function partialoperatorx(S::HalfDiskSpace, N)
    # Takes the space P^{a,b} -> P^{a+1,b+1}
    Sx = differentiatespacex(S)
    pts, w = pointswithweights(Sx, N^2)
    getopnorms(Sx, sum(1:N+1))
    getopptseval(Sx, N+1, pts)
    getopptseval(S, N+1, pts)
    getxderivopptseval(S, N+1, pts)
    if N == 2
        A = BandedBlockBandedMatrix(
            Zeros{Float64}(sum(1:N),sum(1:(N+1))), (1:N, 1:N+1), (-1,2), (0, 2))
    else
        A = BandedBlockBandedMatrix(
            Zeros{Float64}(sum(1:N),sum(1:(N+1))), (1:N, 1:N+1), (-1,2), (0, N-1))
    end
    n, k = 1, 0
    j = getopindex(n, k)
    jj = getopindex(n-1, k)
    view(A, Block(n, n+1))[1, 1] = inner2(xderivopevalatpts(S, j, pts),
                            opevalatpts(Sx, jj, pts), w) / Sx.opnorms[jj]
    for n = 2:N, k = 0:n
        j = getopindex(n, k)
        if k == 0
            inds1 = [0]
            inds2 = [0]
        elseif k == 1
            inds1 = [1]
            inds2 = [1]
        elseif k == n - 1
            inds1 = [n-3]
            inds2 = [n-3, n-1]
        elseif k == n
            inds1 = [n-2]
            inds2 = [n-2]
        else
            inds1 = [k-2, k]
            inds2 = [k-2, k]
        end
        if !(n == 2 && k == 1)
            for i in inds1
                jj = getopindex(n-2, i)
                view(A, Block(n-1, n+1))[i+1, k+1] = inner2(xderivopevalatpts(S, j, pts),
                            opevalatpts(Sx, jj, pts), w) / Sx.opnorms[jj]
            end
        end
        for i in inds2
            jj = getopindex(n-1, i)
            view(A, Block(n, n+1))[i+1, k+1] = inner2(xderivopevalatpts(S, j, pts),
                            opevalatpts(Sx, jj, pts), w) / Sx.opnorms[jj]
        end
    end
    A
end
function partialoperatory(S::HalfDiskSpace, N)
    # Takes the space P^{a,b} -> P^{a,b+1}
    A = BandedBlockBandedMatrix(
        Zeros{Float64}(sum(1:N),sum(1:(N+1))), (1:N, 1:N+1), (-1,1), (-1,1))
    Sy = differentiatespacey(S)
    m = N^2 + 1 # TODO
    pts, w = pointswithweights(Sy, m)
    getopptseval(S, N+1, pts)
    getyderivopptseval(S, N+1, pts)
    getopptseval(Sy, N+1, pts)
    getopnorms(Sy, sum(1:N+1))
    for k = 1:N
        j = getopindex(N, k)
        jj = getopindex(N-1, k-1)
        val = (inner2(yderivopevalatpts(S, j, pts), opevalatpts(Sy, jj, pts), w)
                / Sy.opnorms[jj])
        for i = k:N
            view(A, Block(i, i+1))[k, k+1] = val
        end
    end
    A
end
function weightedpartialoperatorx(S::HalfDiskSpace, N)
    # Takes weighted space ∂/∂y(W^{a,b}) -> W^{a-1,b-1}
    W = BandedBlockBandedMatrix(
        Zeros{Float64}(sum(1:(N+3)),sum(1:(N+1))), (1:N+3, 1:N+1), (2,-1), (2,0))
    Sx = differentiateweightedspacex(S)
    m = Int(ceil(N + 0.5S.a + S.b + 5)^2) # TODO
    pts, w = pointswithweights((S.family)(0,0), m) # Weight of 1 for the inner product
    getopptseval(S, N+1, pts)
    getxderivopptseval(S, N+1, pts)
    getopptseval(Sx, N+3, pts)
    getopnorms(Sx, sum(1:N+3))
    for n = 0:N
        for k = 0:n
            j = getopindex(n, k)
            dpxpts = [(weightderivativex(S, pts[i]) * opevalatpts(S, j, pts)[i]
                       + weight(S, pts[i]) * xderivopevalatpts(S, j, pts)[i])
                       for i = 1:length(pts)]
            inds = [k, k+2]
            for i in inds
                if k < n || i == k
                    jj = getopindex(n+1, i)
                    view(W, Block(n+2, n+1))[i+1, k+1] = inner2(dpxpts, opevalatpts(Sx, jj, pts), w) / Sx.opnorms[jj]
                end
                jj = getopindex(n+2, i)
                view(W, Block(n+3, n+1))[i+1, k+1] = inner2(dpxpts, opevalatpts(Sx, jj, pts), w) / Sx.opnorms[jj]
            end
        end
    end
    W
end
function weightedpartialoperatory(S::HalfDiskSpace, N)
    # Takes weighted space ∂/∂y(W^{a,b}) -> W^{a,b-1}
    W = BandedBlockBandedMatrix(
        Zeros{Float64}(sum(1:(N+2)),sum(1:(N+1))), (1:N+2, 1:N+1), (1,-1), (1,-1))
    Sy = differentiateweightedspacey(S)
    m = Int(ceil(0.5 * (2N + S.a + 2S.b))^2) # TODO
    pts, w = pointswithweights((S.family)(0,0), m) # Weight of 1 for the inner product
    getopptseval(S, N+1, pts)
    getyderivopptseval(S, N+1, pts)
    getopptseval(Sy, N+1, pts)
    getopnorms(Sy, sum(1:N+2))
    for k = 0:N
        j = getopindex(N, k)
        jj = getopindex(N+1, k+1)
        dpypts = [(weightderivativey(S, pts[i]) * opevalatpts(S, j, pts)[i]
                   + weight(S, pts[i]) * yderivopevalatpts(S, j, pts)[i])
                   for i = 1:length(pts)]
        val = inner2(dpypts, opevalatpts(Sy, jj, pts), w) / Sy.opnorms[jj]
        for i = k:N
            view(W, Block(i+2, i+1))[k+2, k+1] = val
        end
    end
    W
end

function transformoperator(S::HalfDiskSpace, N)
    if Int(S.a) == 1 && Int(S.b) == 0
        # Outputs the relevant sum(1:N+2) × sum(1:N+1) matrix operator
        # Takes the space xP^{1,0} -> P^{0,0}
        St = (S.family)(0.0, 0.0)
        T = BandedBlockBandedMatrix(
            Zeros{Float64}(sum(1:(N+2)),sum(1:(N+1))), (1:N+2, 1:N+1), (1,0), (0,0))
        getopnorms(St, sum(1:N+2))
        pts, w = pointswithweights(St, (N+2)^2)
        getopptseval(St, N+2, pts)
        getopptseval(S, N+2, pts)
        for n=0:N, k=0:n
            j = getopindex(n, k)
            p = [opevalatpts(S, j, pts)[i] * pts[i][1] for i = 1:length(pts)]
            for m = n:n+1
                jj = getopindex(m, k)
                view(T, Block(m+1, n+1))[k+1, k+1] = inner2(
                                p, opevalatpts(St, jj, pts), w) / St.opnorms[jj]
            end
        end
        T
    elseif Int(S.a) == 0 && Int(S.b) == 1
        # Outputs the relevant sum(1:N+1) × sum(1:N+1) matrix operator
        # Takes the space P^{0,1} -> P^{1,1}
        St = (S.family)(1.0, 1.0)
        T = BandedBlockBandedMatrix(
            Zeros{Float64}(sum(1:(N+1)),sum(1:(N+1))), (1:N+1, 1:N+1), (0,1), (0,0))
        getopnorms(St, sum(1:N+2))
        pts, w = pointswithweights(St, (N+2)^2)
        getopptseval(St, N+2, pts)
        getopptseval(S, N+2, pts)
        for n = 0:N
            for k = 0:n-1
                j = getopindex(n, k)
                for m = n-1:n
                    jj = getopindex(m, k)
                    view(T, Block(m+1, n+1))[k+1, k+1] = inner2(opevalatpts(S, j, pts), opevalatpts(St, jj, pts), w) / St.opnorms[jj]
                end
            end
            view(T, Block(n+1, n+1))[n+1, n+1] = 1.0
        end
        T
    elseif Int(S.a) == 0 && Int(S.b) == 2
        # Outputs the relevant sum(1:N+1) × sum(1:N+1) matrix operator
        # Takes the space P^{0,2} -> P^{2,2}
        St = (S.family)(2.0, 2.0)
        T = BandedBlockBandedMatrix(
            Zeros{Float64}(sum(1:(N+1)),sum(1:(N+1))), (1:N+1, 1:N+1), (0,2), (0,0))
        getopnorms(St, sum(1:N+2))
        pts, w = pointswithweights(St, (N+2)^2)
        getopptseval(St, N+2, pts)
        getopptseval(S, N+2, pts)
        n = 0; view(T, Block(n+1, n+1))[n+1, n+1] = 1.0
        for n = 1:N
            for k = 0:n-2
                j = getopindex(n, k)
                for m = n-2:n
                    jj = getopindex(m, k)
                    view(T, Block(m+1, n+1))[k+1, k+1] = inner2(opevalatpts(S, j, pts), opevalatpts(St, jj, pts), w) / St.opnorms[jj]
                end
            end
            k = n-1
            j = getopindex(n, k)
            for m = n-1:n
                jj = getopindex(m, k)
                view(T, Block(m+1, n+1))[k+1, k+1] = inner2(opevalatpts(S, j, pts), opevalatpts(St, jj, pts), w) / St.opnorms[jj]
            end
            view(T, Block(n+1, n+1))[n+1, n+1] = 1.0
        end
        T
    else
        error("Invalid HalfDiskSpace")
    end
end

function increaseparamsoperator(S, N)
    # Takes the space P^{a,b} -> P^{a+1,b+1}
    # Outputs the sum(1:N+1) × sum(1:N+1) matrix operator
    St = (S.family)(S.a+1, S.b+1)
    m = 2N^2 - 1 # TODO: check how many points are required for all points() calls
    pts, w = pointswithweights(St, m)
    getopptseval(St, N-2, pts)
    getopptseval(S, N+1, pts)
    T = BandedBlockBandedMatrix(
            Zeros{Float64}(sum(1:(N+1)),sum(1:(N+1))), (1:N+1, 1:N+1), (0,3), (0,2))
    for n = 0:N, k = 0:n
        j = getopindex(n, k)
        for nn = max(0, n-3):n
            for kk = (k < 2 ? k : k-2):2:k
                kk > nn && continue
                jj = getopindex(nn, kk)
                val = (inner2(opevalatpts(S, j, pts), opevalatpts(St, jj, pts), w)
                        / St.opnorms[jj])
                view(T, Block(nn+1, n+1))[kk+1, k+1] = val
            end
        end
    end
    T
end

function convertweightedtononoperator(S, N)
    # NOTE: Only works for W11 -> P00
    # Outputs the sum(1:N+4) × sum(1:N+1) matrix operator
    St = (S.family)(S.a-1, S.b-1)
    Sinner = (S.family)(2S.a-1, 2S.b-1)
    m = 2(N+3)N - 1 # TODO: check how many points are required for all points() calls
    pts, w = pointswithweights(Sinner, m)
    getopptseval(St, N+4, pts)
    getopptseval(S, N+1, pts)
    W = BandedBlockBandedMatrix(
            Zeros{Float64}(sum(1:(N+4)),sum(1:(N+1))), (1:N+4, 1:N+1), (3,0), (2,0))
    for n = 0:N, k = 0:n
        j = getopindex(n, k)
        for nn = n:n+3
            for kk = k:2:min(k+2, nn)
                jj = getopindex(nn,kk)
                val = (inner2(opevalatpts(S, j, pts), opevalatpts(St, jj, pts), w)
                        / St.opnorms[jj])
                view(W, Block(nn+1, n+1))[kk+1, k+1] = val
            end
        end
    end
    W
end
function convertweightedtononoperatorsquare(S, N)
    # Outputs the sum(1:N+1) × sum(1:N+1) matrix operator
    C = BandedBlockBandedMatrix(
        Zeros{Float64}(sum(1:(N+1)),sum(1:(N+1))), (1:N+1, 1:N+1), (3,0), (2,0))
    W = convertweightedtononoperator(S, N+3)
    i = 1
    view(C, Block(i, i)) .= view(W, Block(i, i))
    N == 0 && return C
    i = 2
    view(C, Block(i, i-1)) .= view(W, Block(i, i-1))
    view(C, Block(i, i)) .= view(W, Block(i, i))
    N == 1 && return C
    i = 3
    view(C, Block(i, i-2)) .= view(W, Block(i, i-2))
    view(C, Block(i, i-1)) .= view(W, Block(i, i-1))
    view(C, Block(i, i)) .= view(W, Block(i, i))
    N == 2 && return C
    for i = 4:N+1
        view(C, Block(i, i-3)) .= view(W, Block(i, i-3))
        view(C, Block(i, i-2)) .= view(W, Block(i, i-2))
        view(C, Block(i, i-1)) .= view(W, Block(i, i-1))
        view(C, Block(i, i)) .= view(W, Block(i, i))
    end
    C
end

function laplace(D::HalfDiskFamily, N)
    A = partialoperatorx(D(0.0,0.0), N+2)
    B = weightedpartialoperatorx(D(1.0,1.0), N)
    C = transformoperator(D(0.0,1.0), N+1)
    E = partialoperatory(D(0.0,0.0), N+2)
    F = transformoperator(D(1.0,0.0), N+1)
    G = weightedpartialoperatory(D(1.0,1.0), N)
    A * B + C * E * F * G
end

function laplacesquare(D::HalfDiskFamily, N)
    # sparse([laplace(D, N) zeros(sum(1:N+2), sum(1:N+2)-sum(1:N+1))])

    # Outputs the sum(1:N+1) × sum(1:N+1) matrix operator
    Δ = BandedBlockBandedMatrix(
        Zeros{Float64}(sum(1:(N+1)),sum(1:(N+1))), (1:N+1, 1:N+1), (1,1), (2,5))
    L = laplace(D, N+1)
    i = 1
    view(Δ, Block(i, i)) .= view(L, Block(i, i))
    view(Δ, Block(i, i+1)) .= view(L, Block(i, i+1))
    for i = 2:N
        view(Δ, Block(i, i-1)) .= view(L, Block(i, i-1))
        view(Δ, Block(i, i)) .= view(L, Block(i, i))
        view(Δ, Block(i, i+1)) .= view(L, Block(i, i+1))
    end
    i = N+1
    view(Δ, Block(i, i-1)) .= view(L, Block(i, i-1))
    view(Δ, Block(i, i)) .= view(L, Block(i, i))
    Δ
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

# Operator Clenshaw
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
    Jy = OrthogonalPolynomialFamilies.jacobiy(S, N)' # TODO: Need to transpose in the method!
    if m̃ < m
        resize!(cfs, m)
        cfs[m̃+1:end] .= 0.0
    end
    P0 = 1.0
    if N == 0
        return cfs[1] * id * P0
    end
    if N == 1
        P0 * (converttooperatorclenshawmatrix(cfs[1], Jx)[1] - (converttooperatorclenshawmatrix(S.DT[1], Jx) * (converttooperatorclenshawmatrix(S.B[1], Jx) - operatorclenshawG(0, Jx, Jy)))[1])
    end
    inds2 = m-N:m
    γ2 = converttooperatorclenshawmatrix(view(cfs, inds2), Jx)'
    inds1 = (m-2N):(m-N-1)
    γ1 = (converttooperatorclenshawmatrix(view(cfs, inds1), Jx)'
        - γ2 * converttooperatorclenshawmatrix(S.DT[N], Jx) * (converttooperatorclenshawmatrix(S.B[N], Jx)
                                                                - operatorclenshawG(N-1, Jx, Jy)))
    for n = N-2:-1:0
        ind = sum(1:n)
        γ = (converttooperatorclenshawmatrix(view(cfs, ind+1:ind+n+1), Jx)'
             - γ1 * converttooperatorclenshawmatrix(S.DT[n+1], Jx) * (converttooperatorclenshawmatrix(S.B[n+1], Jx) - operatorclenshawG(n, Jx, Jy))
             - γ2 * converttooperatorclenshawmatrix(S.DT[n+2] * S.C[n+2], Jx))
        γ2 = copy(γ1)
        γ1 = copy(γ)
    end
    (γ1 * P0)[1]
end
operatorclenshaw(f::Fun, S::HalfDiskSpace) = operatorclenshaw(f.coefficients, S)


# Method to gather and evaluate the ops of space S at the transform pts given
function getopptseval(S::HalfDiskSpace, N, pts)
    resetopptseval(S)
    jj = [getopindex(n, 0) for n=0:N]
    for j in jj
        opevalatpts(S, j, pts)
    end
    S.opptseval
end
function opevalatpts(S::HalfDiskSpace, j, pts)
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
        S.opptseval[jj+k] = Vector{Float64}(undef, length(pts))
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
function xderivopevalatpts(S::HalfDiskSpace, j, pts)
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
        S.xderivopptseval[jj+k] = Vector{Float64}(undef, length(pts))
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
function yderivopevalatpts(S::HalfDiskSpace, j, pts)
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
        S.yderivopptseval[jj+k] = Vector{Float64}(undef, length(pts))
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
function inner2(fpts, gpts, w)
    n = length(w)
    sum(([fpts[pt] * gpts[pt] for pt = 1:n]
            + [fpts[pt] * gpts[pt] for pt = n+1:2n]) .* w) / 2
end


end # module
