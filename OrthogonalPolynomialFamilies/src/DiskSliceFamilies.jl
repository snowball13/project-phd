# module DiskSliceFamilies

# using ApproxFun
#     import ApproxFun: evaluate, PolynomialSpace, recα, recβ, recγ, recA, recB, recC, domain,
#                         domainspace, rangespace, bandwidths, prectype, canonicaldomain, tocanonical,
#                         spacescompatible, points, transform, itransform, AbstractProductSpace, tensorizer,
#                         columnspace, checkpoints, plan_transform, clenshaw
#     import Base: in, *
# using OrthogonalPolynomialFamilies
# using StaticArrays
# using FastGaussQuadrature
# using LinearAlgebra
# using SparseArrays
# using BlockBandedMatrices
# using BlockArrays
# # using SingularIntegralEquations
# using Test

export DiskSliceFamily, DiskSliceSpace

# R should be Float64, B should be BigFloat
abstract type DiskFamily{B,R,N} end
struct DiskSlice{B,T} <: Domain{SVector{2,T}} end
DiskSlice() = DiskSlice{BigFloat, Float64}()
checkpoints(::DiskSlice) = [SVector(0.1,0.23), SVector(0.3,0.12)]

struct DiskSliceSpace{DF, B, T, N} <: Space{DiskSlice{B,T}, T}
    family::DF # Pointer back to the family
    params::NTuple{N,B} # Parameters
    opnorms::Vector{T} # squared norms
    opptseval::Vector{Vector{T}} # Store the ops evaluated at the transform pts
    xderivopptseval::Vector{Vector{T}} # Store the x deriv of the ops evaluated
                                    # at the transform pts
    yderivopptseval::Vector{Vector{T}} # Store the y deriv of the ops evaluated
                                    # at the transform pts
    A::Vector{SparseMatrixCSC{T}}
    B::Vector{SparseMatrixCSC{T}}
    C::Vector{SparseMatrixCSC{T}}
    DT::Vector{SparseMatrixCSC{T}}
end

function DiskSliceSpace(fam::DiskFamily{B,T,N}, params::NTuple{N,B}) where {B,T,N}
    DiskSliceSpace{typeof(fam), B, T, N}(
        fam, params, Vector{T}(), Vector{Vector{T}}(),
        Vector{Vector{T}}(), Vector{Vector{T}}(), Vector{SparseMatrixCSC{T}}(),
        Vector{SparseMatrixCSC{T}}(), Vector{SparseMatrixCSC{T}}(),
        Vector{SparseMatrixCSC{T}}())
end

# TODO
in(x::SVector{2}, D::DiskSlice) = D.α ≤ x[1] ≤ D.β && D.γ*D.ρ(x[1]) ≤ x[2] ≤ D.δ*D.ρ(x[1])

spacescompatible(A::DiskSliceSpace, B::DiskSliceSpace) = (A.params == B.params)

domain(::DiskSliceSpace) = DiskSlice()

# R should be Float64, B BigFloat
struct DiskSliceFamily{B,T,N,FAR,FAP,F,I} <: DiskFamily{B,T,N}
    spaces::Dict{NTuple{N,B}, DiskSliceSpace}
    α::T
    β::T
    γ::T
    δ::T
    R::FAR # OPFamily in (α,β)
    P::FAP # OPFamily in (γ,δ)
    ρ::F # Fun of sqrt(1-X^2) in (α,β)
    nparams::I
end

function (D::DiskSliceFamily{B,T,N,<:Any,<:Any,<:Any,<:Any})(params::NTuple{N,B}) where {B,T,N}
    haskey(D.spaces,params) && return D.spaces[params]
    D.spaces[params] = DiskSliceSpace(D, params)
end
(D::DiskSliceFamily{B,T,N,<:Any,<:Any,<:Any,<:Any})(params::Vararg{B,N}) where {B,T,N} =
    D(params)
(D::DiskSliceFamily{B,T,N,<:Any,<:Any,<:Any,<:Any})(params::Vararg{T,N}) where {B,T,N} =
    D(B.(params))

function DiskSliceFamily(::Type{B},::Type{T}, α::T, β::T, γ::T, δ::T) where {B,T}
    nparams = 3 # Default
    X = Fun(identity, B(α)..β)
    Y = Fun(identity, B(γ)..δ)
    ρ = sqrt(1 - X^2) # TODO: Change to anon function
    if isinteger(β) && Int(β) == 1 # 2-param family
        nparams -= 1
        R = OrthogonalPolynomialFamily(T, X-α, ρ)
    else
        R = OrthogonalPolynomialFamily(T, β-X, X-α, ρ)
    end
    P = OrthogonalPolynomialFamily(T, δ-Y, Y-γ)
    spaces = Dict{NTuple{nparams,B}, DiskSliceSpace}()
    DiskSliceFamily{B,T,nparams,typeof(R),typeof(P),typeof(ρ),Int}(spaces, α, β, γ, δ, R, P, ρ, nparams)
end
# Useful quick constructors
DiskSliceFamily(α::T, β::T, γ::T, δ::T) where T = DiskSliceFamily(BigFloat, T, α, β, γ, δ)
DiskSliceFamily(α::T, β::T) where T = DiskSliceFamily(BigFloat, T, α, β, -1.0, 1.0)
DiskSliceFamily(α::T) where T = DiskSliceFamily(BigFloat, T, α, 1.0, -1.0, 1.0)
DiskSliceFamily() = DiskSliceFamily(BigFloat, Float64, 0.0, 1.0, -1.0, 1.0)

#===#
# Retrieve spaces methods
function getRspace(S::DiskSliceSpace, k::Int)
    if S.family.nparams == 3
        (S.family.R)(S.params[1], S.params[2], 2S.params[3] + 2k + 1)
    else
        (S.family.R)(S.params[1], 2S.params[2] + 2k + 1)
    end
end
function getRspace(S::DiskSliceSpace)
    if S.family.nparams == 3
        (S.family.R)(S.params[1], S.params[2], 2S.params[3])
    else
        (S.family.R)(S.params[1], 2S.params[2])
    end
end
getPspace(S::DiskSliceSpace) = (S.family.P)(S.params[end], S.params[end])

#===#
# Weight eval functions
weight(S::DiskSliceSpace{<:Any,<:Any,T,<:Any}, x, y) where T =
    T(getRspace(S).weight(x) * getPspace(S).weight(y/S.family.ρ(x)))
weight(S::DiskSliceSpace, z) = weight(S, z[1], z[2])


#===#
# points() and methods for pt evals and norm vals

# NOTE we output ≈n points (x,y), plus the ≈n points corresponding to (x,-y)
function pointswithweights(S::DiskSliceSpace{<:Any, <:Any, T, <:Any}, n) where T
    # Return the weights and nodes to use for the even part of a function,
    # i.e. for the disk-slice Ω:
    #   int_Ω W^{a,b}(x,y) f(x,y) dydx ≈ Σ_j weⱼ*fe(xⱼ,yⱼ)
    # NOTE: the odd part of the quad rule will equal 0 for polynomials,
    #       so can be ignored.
    N = Int(ceil(sqrt(n))) # ≈ n
    @show "begin pointswithweights()", n, N
    t, wt = pointswithweights(getPspace(S), N)
    s, ws = pointswithweights(getRspace(S, 0), N)
    pts = Vector{SArray{Tuple{2},T,1,2}}(undef, 2(N^2)) # stores both (x,y) and (x,-y)
    w = zeros(N^2) # weights
    for i = 1:N
        for k = 1:N
            x, y = s[k], t[i] * S.family.ρ(s[k]) # t[i] * sqrt(1 - s[k]^2)
            pts[i + (k - 1)N] = x, y
            pts[N^2 + i + (k - 1)N] = x, -y
            w[i + (k - 1)N] = ws[k] * wt[i]
        end
    end
    @show "end pointswithweights()"
    pts, w
end
points(S::DiskSliceSpace, n) = pointswithweights(S, n)[1]

function inner(S::DiskSliceSpace, fpts, gpts, w)
    n = length(w)
    sum(([fpts[pt] * gpts[pt] for pt = 1:n]
            + [fpts[pt] * gpts[pt] for pt = n+1:2n]) .* w) / 2
end

function getopnorms(S::DiskSliceSpace{<:Any, <:Any, T, <:Any}, k) where T
    # NOTE these are squared norms
    m = length(S.opnorms)
    if k + 1 > m
        resize!(S.opnorms, k+1)
        P = getPspace(S)
        getopnorm(P)
        for j = m+1:k+1
            S.opnorms[j] = getopnorm(getRspace(S, j-1)) * P.opnorm[1]
        end
    end
    S
end

# Method to gather and evaluate the ops of space S at the transform pts given
function getopptseval(S::DiskSliceSpace, N, pts)
    resetopptseval(S)
    jj = [getopindex(n, 0) for n=0:N]
    for j in jj
        opevalatpts(S, j, pts)
    end
    S.opptseval
end
function opevalatpts(S::DiskSliceSpace{<:Any, <:Any, T, <:Any}, j, pts) where T
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
            P = - S.DT[n] * (S.B[n] - clenshawG(S, n-1, pts[r])) * P1
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
            P = (- S.DT[n] * (S.B[n] - clenshawG(S, n-1, pts[r])) * P1
                 - S.DT[n] * S.C[n] * P2)
            for k = 0:n
                S.opptseval[jj+k][r] = P[k+1]
            end
        end
    end
    S.opptseval[j]
end
function resetopptseval(S::DiskSliceSpace)
    resize!(S.opptseval, 0)
    S
end

#===#
# transform and itransform

struct DiskSliceTransformPlan{T}
    w::Vector{T}
    pts::Vector{SArray{Tuple{2},T,1,2}}
    S::DiskSliceSpace{<:Any, <:Any, T, <:Any}
end

function DiskSliceTransformPlan(S::DiskSliceSpace{<:Any, <:Any, T, <:Any}, vals) where T
    m = Int(length(vals) / 2)
    pts, w = pointswithweights(S, m)
    DiskSliceTransformPlan{T}(w, pts, S)
end
plan_transform(S::DiskSliceSpace, vals) = DiskSliceTransformPlan(S, vals)
transform(S::DiskSliceSpace, vals) = plan_transform(S, vals) * vals

# Inputs: OP space, f(pts) for desired f
# Output: Coeffs of the function f for its expansion in the DiskSliceSpace OPs
function *(DSTP::DiskSliceTransformPlan, vals)
    @show "Begin DSTP mult"
    m2 = Int(length(vals) / 2)
    N = Int(sqrt(m2)) - 1
    m1 = Int((N+1)*(N+2) / 2)
    @show m1, m2

    ret = zeros(m1)
    resizedata!(DSTP.S, N)
    getopnorms(DSTP.S, N) # We store the norms of the OPs
    for i = 1:m2
        if i % 100 == 0
            @show m2, i
        end
        pt = [DSTP.pts[i]]
        getopptseval(DSTP.S, N, pt)
        for j = 1:m1
            ret[j] += opevalatpts(DSTP.S, j, pt)[1] * DSTP.w[i] * vals[i]
        end
        pt = [DSTP.pts[i+m2]]
        getopptseval(DSTP.S, N, pt)
        for j = 1:m1
            ret[j] += opevalatpts(DSTP.S, j, pt)[1] * DSTP.w[i] * vals[i+m2]
        end
    end
    resetopptseval(DSTP.S)
    j = 1
    for n = 0:N, k = 0:n
        ret[j] /= (2 * DSTP.S.opnorms[k+1])
        j += 1
    end
    @show "End DSTP mult"
    ret
end

# Inputs: OP space, coeffs of a function f for its expansion in the DiskSliceSpace OPs
# Output: vals = {f(x_j)} where x_j are are the points(S,n)
function itransform(S::DiskSliceSpace{<:Any, <:Any, T, <:Any}, cfs) where T
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
end

#===#
# Jacobi operator entries

function recα(S::DiskSliceSpace{<:Any, <:Any, T, <:Any}, n, k, j) where T
    R = getRspace(S, k)
    if j == 1
        recγ(T, R, n-k+1)
    elseif j == 2
        recα(T, R, n-k+1)
    else
        error("Invalid entry to function")
    end
end

function recβ(S::DiskSliceSpace{<:Any, <:Any, T, <:Any}, n, k, j) where T
    # We get the norms of the 2D OPs
    getopnorms(S, k+1)

    R1 = getRspace(S, k-1)
    R2 = getRspace(S, k)
    R3 = getRspace(S, k+1)
    P = getPspace(S)
    getopnorm(P)

    if isodd(j)
        pts, w = pointswithweights(R2, Int(ceil(n-k+1.5)))
        δ = recγ(T, P, k+1) * P.opnorm[1]
    else
        pts, w = pointswithweights(R3, Int(ceil(n-k+1.5)))
        δ = recβ(T, P, k+1) * P.opnorm[1]
    end
    getopptseval(R2, n-k+1, pts)

    if j == 1
        getopptseval(R1, n-k+1, pts)
        (inner2(R2, opevalatpts(R2, n-k+1, pts), opevalatpts(R1, n-k+1, pts), w)
            * δ / S.opnorms[k])
    elseif j == 2
        getopptseval(R3, n-k-1, pts)
        (inner2(R3, opevalatpts(R2, n-k+1, pts), opevalatpts(R3, n-k-1, pts), w)
            * δ / S.opnorms[k+2])
    elseif j == 3
        getopptseval(R1, n-k+2, pts)
        (inner2(R2, opevalatpts(R2, n-k+1, pts), opevalatpts(R1, n-k+2, pts), w)
            * δ / S.opnorms[k])
    elseif j == 4
        getopptseval(R3, n-k, pts)
        (inner2(R3, opevalatpts(R2, n-k+1, pts), opevalatpts(R3, n-k, pts), w)
            * δ / S.opnorms[k+2])
    elseif j == 5
        getopptseval(R1, n-k+3, pts)
        (inner2(R2, opevalatpts(R2, n-k+1, pts), opevalatpts(R1, n-k+3, pts), w)
            * δ / S.opnorms[k])
    elseif j == 6
        getopptseval(R3, n-k+1, pts)
        (inner2(R3, opevalatpts(R2, n-k+1, pts), opevalatpts(R3, n-k+1, pts), w)
            * δ / S.opnorms[k+2])
    else
        error("Invalid entry to function")
    end
end

function getAs!(S::DiskSliceSpace, N, N₀)
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

function getDTs!(S::DiskSliceSpace, N, N₀)
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
function getBs!(S::DiskSliceSpace, N, N₀)
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
function getCs!(S::DiskSliceSpace, N, N₀)
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

function resizedata!(S::DiskSliceSpace, N)
    # N is the max degree of the OPs
    N₀ = length(S.B)
    N ≤ N₀ - 2 && return S
    resize!(S.A, N + 2)
    resize!(S.B, N + 2)
    resize!(S.C, N + 2)
    resize!(S.DT, N + 2)
    getBs!(S, N, N₀)
    @show "done Bs"
    getCs!(S, N, N₀)
    @show "done Cs"
    getAs!(S, N, N₀)
    @show "done As"
    getDTs!(S, N, N₀)
    @show "done DTs"
    S
end

function jacobix(S::DiskSliceSpace, N)
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

function jacobiy(S::DiskSliceSpace, N)
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

#====#
# Evaluation (clenshaw)

function clenshawG(::DiskSliceSpace, n, z)
    sp = sparse(I, n+1, n+1)
    [z[1] * sp; z[2] * sp]
end
function clenshaw(cfs::AbstractVector, S::DiskSliceSpace, z)
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
    γ1 = view(cfs, inds1)' - γ2 * S.DT[N] * (S.B[N] - clenshawG(S, N-1, z))
    for n = N-2:-1:0
        ind = sum(1:n)
        γ = (view(cfs, ind+1:ind+n+1)'
             - γ1 * S.DT[n+1] * (S.B[n+1] - clenshawG(S, n, z))
             - γ2 * S.DT[n+2] * S.C[n+2])
        γ2 = copy(γ1)
        γ1 = copy(γ)
    end
    (γ1 * P0)[1]
end
evaluate(cfs::AbstractVector, S::DiskSliceSpace, z) = clenshaw(cfs, S, z)

# Operator Clenshaw
function operatorclenshawG(S::DiskSliceSpace{<:Any, <:Any, T, <:Any}, n, Jx, Jy, zeromat) where T
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
function operatorclenshawvector(S::DiskSliceSpace{<:Any, <:Any, T, <:Any}, v, id) where T
    s = size(v)[1]
    B = Array{SparseMatrixCSC{T}}(undef, (1, s))
    for i = 1:s
        B[1,i] = id * v[i]
    end
    B
end
function operatorclenshawmatrixDT(S::DiskSliceSpace{<:Any, <:Any, T, <:Any}, A, id) where T
    B = Array{SparseMatrixCSC{T}}(undef, size(A))
    for ij = 1:length(A)
        B[ij] = id * A[ij]
    end
    B
end
function operatorclenshawmatrixBmG(S::DiskSliceSpace{<:Any, <:Any, T, <:Any}, A, id, Jx, Jy) where T
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
function operatorclenshaw(cfs, S::DiskSliceSpace, N)
    # Outputs the operator NxN-blocked matrix operator corresponding to the
    # function f given by its coefficients of its expansion in the space S
    m̃ = length(cfs)
    M = getnk(m̃)[1] # Degree of f

    # Pad cfs to correct size
    m = getopindex(M, M)
    if N < M
        error("Size requested has lower degree than function for the operator")
    end
    if m̃ < m
        resize!(cfs, m)
        cfs[m̃+1:end] .= 0.0
    end

    resizedata!(S, M)
    Jx = sparse(jacobix(S, N))
    Jy = sparse(jacobiy(S, N))
    id = sparse(I, size(Jx))

    @show "Operator Clenshaw"
    P0 = 1.0
    if M == 0
        ret = cfs[1] * id * P0
    elseif M == 1
        ret = P0 * (operatorclenshawvector(S, cfs[1], id)[1] - (operatorclenshawmatrixDT(S, S.DT[1], id) * operatorclenshawmatrixBmG(S, S.B[1], id, Jx, Jy))[1])
    else
        n = M; @show "Operator Clenshaw", N, M, n
        ind = sum(1:n)
        γ2 = operatorclenshawvector(S, view(cfs, ind+1:ind+n+1), id)
        n = M - 1; @show "Operator Clenshaw", M, n
        ind = sum(1:n)
        γ1 = (operatorclenshawvector(S, view(cfs, ind+1:ind+n+1), id)
            - γ2 * operatorclenshawmatrixDT(S, S.DT[n+1], id) * operatorclenshawmatrixBmG(S, S.B[n+1], id, Jx, Jy))
        for n = M-2:-1:0
            @show "Operator Clenshaw", M, n
            ind = sum(1:n)
            γ = (operatorclenshawvector(S, view(cfs, ind+1:ind+n+1), id)
                 - γ1 * operatorclenshawmatrixDT(S, S.DT[n+1], id) * operatorclenshawmatrixBmG(S, S.B[n+1], id, Jx, Jy)
                 - γ2 * operatorclenshawmatrixDT(S, S.DT[n+2] * S.C[n+2], id))
            γ2 = copy(γ1)
            γ1 = copy(γ)
        end
        ret = (γ1 * P0)[1]
    end
    ret
end
operatorclenshaw(f::Fun, S::DiskSliceSpace) = operatorclenshaw(f.coefficients, S, getnk(ncoefficients(f))[1])
operatorclenshaw(f::Fun, S::DiskSliceSpace, N) = operatorclenshaw(f.coefficients, S, N)

#====#
# Methods to gather and evaluate the derivatives of the ops of space S at the
# transform pts given

resetxderivopptseval(S::DiskSliceSpace) = resize!(S.xderivopptseval, 0)
function clenshawGtildex(S::DiskSliceSpace, n, z)
    sp = sparse(I, n+1, n+1)
    [sp; 0.0 * sp]
end
function getxderivopptseval(S::DiskSliceSpace, N, pts)
    resetxderivopptseval(S)
    jj = [getopindex(n, 0) for n=0:N]
    for j in jj
        xderivopevalatpts(S, j, pts)
    end
    S.xderivopptseval
end
function xderivopevalatpts(S::DiskSliceSpace{<:Any, <:Any, T, <:Any}, j, pts) where T
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
            dxP = S.DT[n] * clenshawGtildex(S, n-1, pts[r]) * P1
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
            dxP = (- S.DT[n] * (S.B[n] - clenshawG(S, n-1, pts[r])) * dxP1
                   - S.DT[n] * S.C[n] * dxP2
                   + S.DT[n] * clenshawGtildex(S, n-1, pts[r]) * P1)
            for k = 0:n
                S.xderivopptseval[jj+k][r] = dxP[k+1]
            end
        end
    end
    S.xderivopptseval[j]
end

resetyderivopptseval(S::DiskSliceSpace) = resize!(S.yderivopptseval, 0)
function clenshawGtildey(S::DiskSliceSpace, n, z)
    sp = sparse(I, n+1, n+1)
    [0.0 * sp; sp]
end
function getyderivopptseval(S::DiskSliceSpace, N, pts)
    resetyderivopptseval(S)
    jj = [getopindex(n, 0) for n=0:N]
    for j in jj
        yderivopevalatpts(S, j, pts)
    end
    S.yderivopptseval
end
function yderivopevalatpts(S::DiskSliceSpace{<:Any, <:Any, T, <:Any}, j, pts) where T
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
            dyP = S.DT[n] * clenshawGtildey(S, n-1, pts[r]) * P1
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
            dyP = (- S.DT[n] * (S.B[n] - clenshawG(S, n-1, pts[r])) * dyP1
                   - S.DT[n] * S.C[n] * dyP2
                   + S.DT[n] * clenshawGtildey(S, n-1, pts[r]) * P1)
            for k = 0:n
                S.yderivopptseval[jj+k][r] = dyP[k+1]
            end
        end
    end
    S.yderivopptseval[j]
end

#====#
# Differential operator matrices

differentiatespacex(S::DiskSliceSpace) = (S.family)(S.params .+ 1)
function differentiatespacey(S::DiskSliceSpace)
    len = length(S.params)
    p = ntuple(i -> i == len ? 1 : 0, len)
    (S.family)(S.params .+ p)
end
differentiateweightedspacex(S::DiskSliceSpace) = (S.family)(S.params .- 1)
function differentiateweightedspacey(S::DiskSliceSpace)
    len = length(S.params)
    p = ntuple(i -> i == len ? 1 : 0, len)
    (S.family)(S.params .- p)
end

differentiatex(f::Fun, S::DiskSliceSpace) =
    Fun(differentiatespacex(S), differentiatex(S, f.coefficients))
differentiatey(f::Fun, S::DiskSliceSpace) =
    Fun(differentiatespacey(S), differentiatey(S, f.coefficients))
function differentiatex(S::DiskSliceSpace, cfs::AbstractVector)
    m̃ = length(cfs)
    N = -1 + Int(round(sqrt(1+2(m̃-1))))
    m = Int((N+1)*(N+2)/2)
    if m̃ < m
        resize!(cfs, m)
        cfs[m̃+1:end] .= 0.0
    end
    partialoperatorx(S, N) * cfs
end
function differentiatey(S::DiskSliceSpace, cfs::AbstractVector)
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

function getpartialoperatorxval(S::DiskSliceSpace{<:Any, <:Any, T, <:Any},
                                    ptsp, wp, ptsr, rhoptsr, dxrhoptsr, wr, n, k, m, j) where T
    # We should have already called getopptseval etc
    # ptsr, wr = pointswithweights(getRspace(Sx, -1), 2N+4)
    Sx = differentiatespacex(S)
    P = getPspace(S)
    Px = getPspace(Sx)
    R = getRspace(S, k)
    Rx = getRspace(Sx, j)
    valp = inner2(Px, opevalatpts(P, k+1, ptsp), opevalatpts(Px, j+1, ptsp), wp)
    valr = inner2(Rx, opevalatpts(R, n-k+1, ptsr),
                    rhoptsr.^(k+j+1) .* dxrhoptsr .* opevalatpts(Rx, m-j+1, ptsr), wr)
    val = valp * inner2(Rx, derivopevalatpts(R, n-k+1, ptsr),
                        rhoptsr.^(k+j+2) .* opevalatpts(Rx, m-j+1, ptsr), wr)
    val += k * valr * valp
    val -= valr * inner2(Px, derivopevalatpts(P, k+1, ptsp),
                            ptsp .* opevalatpts(Px, j+1, ptsp), wp)
    val /= Sx.opnorms[j+1]
    val
end
function partialoperatorx(S::DiskSliceSpace{<:Any, <:Any, T, <:Any}, N;
                            transposed=false) where T
    # Takes the space P^{a,b,c} -> P^{a+1,b+1,c+1}
    Sx = differentiatespacex(S)
    P = getPspace(S)
    Px = getPspace(Sx)
    ptsp, wp = pointswithweights(Px, N+2) # TODO
    getopptseval(P, N, ptsp)
    getderivopptseval(P, N, ptsp)
    getopptseval(Px, N, ptsp)
    ptsr, wr = pointswithweights(getRspace(Sx, -1), 2N+4)
    getopnorms(Sx, N-1)

    # ρ.(ptsr) and dρ/dx.(ptsr)
    rhoptsr = T.(S.family.ρ.(ptsr))
    dxrhoptsr = T.(differentiate(S.family.ρ).(ptsr))

    band = S.family.nparams
    if transposed
        A = BandedBlockBandedMatrix(
            Zeros{T}(sum(1:(N+1)), sum(1:N)), (1:N+1, 1:N), (band, -1), (2, 0))
    else
        A = BandedBlockBandedMatrix(
            Zeros{T}(sum(1:N), sum(1:(N+1))), (1:N, 1:N+1), (-1, band), (0, 2))
    end

    # Get pt evals for the R OPs
    for k = 0:N
        @show "dxoperator", N, k
        R = getRspace(S, k)
        getopptseval(R, N-k, ptsr)
        getderivopptseval(R, N-k, ptsr)
        Rx = getRspace(Sx, k)
        getopptseval(Rx, N-k+1, ptsr)
    end

    n, k = 1, 0
    m, j = n-1, k
    val = getpartialoperatorxval(S, ptsp, wp, ptsr, rhoptsr, dxrhoptsr, wr, n, k, m, j)
    view(A, Block(m+1, n+1))[1, 1] = val
    for n = 2:N, k = 0:n
        for m = max(0,n-band):(n-1)
            for j = (k-2):2:min(k,m)
                if j < 0
                    continue
                end
                val = getpartialoperatorxval(S, ptsp, wp, ptsr, rhoptsr,
                                                dxrhoptsr, wr, n, k, m, j)
                if transposed
                    view(A, Block(n+1, m+1))[k+1, j+1] = val
                else
                    view(A, Block(m+1, n+1))[j+1, k+1] = val
                end
            end
        end
    end
    A
end
function partialoperatory(S::DiskSliceSpace{<:Any, <:Any, T, <:Any}, N;
                            transposed=false) where T
    # Takes the space H^{a,b,c} -> H^{a,b,c+1}
    if transposed
        A = BandedBlockBandedMatrix(
            Zeros{T}(sum(1:(N+1)),sum(1:N)), (1:N+1, 1:N), (1,-1), (1,-1))
    else
        A = BandedBlockBandedMatrix(
            Zeros{T}(sum(1:N),sum(1:(N+1))), (1:N, 1:N+1), (-1,1), (-1,1))
    end
    Sy = differentiatespacey(S)
    P = getPspace(S)
    Py = getPspace(Sy)
    pts, w = pointswithweights(Py, N)
    getopptseval(P, N, pts)
    getderivopptseval(P, N, pts)
    getopptseval(Py, N, pts)
    getopnorms(Sy, N-1)
    for k = 1:N
        val = (getopnorm(getRspace(S, k))
                * inner2(Py, derivopevalatpts(P, k+1, pts), opevalatpts(Py, k, pts), w)
                / Sy.opnorms[k])
        for i = k:N
            if transposed
                view(A, Block(i+1, i))[k+1, k] = val
            else
                view(A, Block(i, i+1))[k, k+1] = val
            end
        end
    end
    A
end
function getweightedpartialoperatorxval(S::DiskSliceSpace{<:Any, <:Any, T, <:Any},
                ptsp, wp1, wp, ptsr, rhoptsr, dxrhoptsr, wr010, wr100, wr, n, k, m, j) where T
    # We should have already called getopptseval etc
    # ptsr, wr = pointswithweights(getRspace(Sx, 0), 2N+4)
    Sx = differentiateweightedspacex(S)
    P = getPspace(S)
    Px = getPspace(Sx)
    R = getRspace(S, k)
    Rx = getRspace(Sx, j)

    valp = inner2(Px, wp1 .* opevalatpts(P, k+1, ptsp), opevalatpts(Px, j+1, ptsp), wp)
    valr = inner2(Rx, opevalatpts(R, n-k+1, ptsr) .* wr100 .* wr010,
                    rhoptsr.^(k+j+1) .* dxrhoptsr .* opevalatpts(Rx, m-j+1, ptsr), wr)

    if S.family.nparams == 3
        A1 = - (inner2(Rx, opevalatpts(R, n-k+1, ptsr),
                        wr010 .* rhoptsr.^(k+j+2) .* opevalatpts(Rx, m-j+1, ptsr), wr)
                * S.params[1] * valp)
    else
        A1 = 0.0
    end
    B1 = (inner2(Rx, opevalatpts(R, n-k+1, ptsr),
                    wr100 .* rhoptsr.^(k+j+2) .* opevalatpts(Rx, m-j+1, ptsr), wr)
            * S.params[end-1] * valp)
    # C1 = valr * 2*S.params[end] * valp
    # D1 = - (valr * 2*S.params[end]
    #         * inner2(Px, opevalatpts(P, k+1, ptsp), ptsp .* opevalatpts(Px, j+1, ptsp), wp))
    C1 = (2 * S.params[end]
            * valr
            * inner2(Px, opevalatpts(P, k+1, ptsp), opevalatpts(Px, j+1, ptsp), wp))
    D1 = 0.0
    A2 = (inner2(Rx, derivopevalatpts(R, n-k+1, ptsr) .* wr100 .* wr010,
                rhoptsr.^(k+j+2) .* opevalatpts(Rx, m-j+1, ptsr), wr)
            * valp)
    B2 = valr * k * valp
    C2 = - valr * inner2(Px, ptsp .* derivopevalatpts(P, k+1, ptsp),
                            wp1 .* opevalatpts(Px, j+1, ptsp), wp)

    val = A1 + B1 + C1 + D1 + A2 + B2 + C2
    val / Sx.opnorms[j+1]
end
function weightedpartialoperatorx(S::DiskSliceSpace{<:Any, <:Any, T, <:Any}, N;
                                    transposed=false) where T
    # Takes weighted space ∂/∂x(W^{a,b,c}) -> W^{a-1,b-1,c-1}
    band = S.family.nparams
    if transposed
        W = BandedBlockBandedMatrix(
            Zeros{T}(sum(1:(N+1)),sum(1:(N+1+band))), (1:N+1, 1:N+1+band), (-1, band), (0, 2))
    else
        W = BandedBlockBandedMatrix(
            Zeros{T}(sum(1:(N+1+band)),sum(1:(N+1))), (1:N+1+band, 1:N+1), (band, -1), (2, 0))
    end
    Sx = differentiateweightedspacex(S)
    P = getPspace(S)
    Px = getPspace(Sx)
    ptsp, wp = pointswithweights(Px, N+3)
    getopptseval(P, N, ptsp)
    getopptseval(Px, N+1, ptsp)
    getderivopptseval(P, N, ptsp)
    # ptsr, wr = pointswithweights(getRspace(Sx), 2N+4) # TODO
    ptsr, wr = pointswithweights(getRspace(Sx, 0), 2N+4)
    getopnorms(Sx, N+2)

    # ρ.(ptsr) and dρ/dx.(ptsr)
    rhoptsr = T.(S.family.ρ.(ptsr))
    dxrhoptsr = T.(differentiate(S.family.ρ).(ptsr))
    # w_P^{(1)}.(pts)
    wp1 = (-ptsp.^2 .+ 1) # TODO - dont hardcode!
    # w_R^{(1,0,0)}, w_R^{(0,1,0)}
    wr100 = S.family.nparams == 3 ? (-ptsr .+ S.family.β) : ones(length(ptsr))
    wr010 = (ptsr .- S.family.α)

    # Get pt evals for the R OPs
    for k = 0:N
        R = getRspace(S, k)
        getopptseval(R, N-k, ptsr)
        getderivopptseval(R, N-k, ptsr)
        for j = k:2:k+2
            Rx = getRspace(Sx, j)
            getopptseval(Rx, N-k+1, ptsr)
        end
    end
    for n = 0:N, k = 0:n
        for m = n+1:n+S.family.nparams, j = k:2:min(m,k+2)
            val = getweightedpartialoperatorxval(S, ptsp, wp1, wp, ptsr,
                            rhoptsr, dxrhoptsr, wr010, wr100, wr, n, k, m, j)
            if transposed
                view(W, Block(n+1, m+1))[k+1, j+1] = val
            else
                view(W, Block(m+1, n+1))[j+1, k+1] = val
            end
        end
    end
    W
end
function weightedpartialoperatory(S::DiskSliceSpace{<:Any, <:Any, T, <:Any}, N;
                                    transposed=false) where T
    # Takes weighted space ∂/∂y(W^{a,b,c}) -> W^{a,b,c-1}
    if transposed
        W = BandedBlockBandedMatrix(
            Zeros{T}(sum(1:(N+1)),sum(1:(N+2))), (1:N+1, 1:N+2), (-1,1), (-1,1))
    else
        W = BandedBlockBandedMatrix(
            Zeros{T}(sum(1:(N+2)),sum(1:(N+1))), (1:N+2, 1:N+1), (1,-1), (1,-1))
    end
    Sy = differentiateweightedspacey(S)
    P = getPspace(S)
    Py = getPspace(Sy)
    ptsp, wp = pointswithweights(Py, N+2)
    getopptseval(P, N, ptsp)
    getopptseval(Py, N+1, ptsp)
    getderivopptseval(P, N, ptsp)
    getopnorms(Sy, N+1)
    params = S.params .* 0 .+ ntuple(i -> i == S.family.nparams ? 1 : 0, S.family.nparams)
    wp1 = T.(getPspace(S.family(params)).weight.(ptsp)) # (-ptsp.^2 .+ 1)
    n, m = N, N+1
    for k = 0:N
        j = k + 1
        val = (getopnorm(getRspace(S, k))
                * inner2(P, (wp1 .* derivopevalatpts(P, k+1, ptsp)
                                -2*S.params[end]*ptsp .* opevalatpts(P, k+1, ptsp)),
                        opevalatpts(Py, j+1, ptsp), wp)
                / Sy.opnorms[j+1])
        for i = k:N
            if transposed
                view(W, Block(i+1, i+2))[k+1, k+2] = val
            else
                view(W, Block(i+2, i+1))[k+2, k+1] = val
            end
        end
    end
    W
end

#====#
# Parameter tranformation operators

function transformparamsoperator(S::DiskSliceSpace{<:Any, <:Any, T, <:Any},
            St::DiskSliceSpace{<:Any, <:Any, T, <:Any}, N;
            weighted=false, transposed=false) where T
    # Cases we can handle:
    if weighted == false
        λmult = S.family.nparams - 1
        λ = Int(St.params[1] - S.params[1])
        μ = Int(St.params[end] - S.params[end])
        band = 2μ + λmult*λ

        # Case 1: Takes the space H^{a,b,c} -> H^{a+1,b+1,c}
        if λ == 1 && μ == 0
            # Outputs the relevant sum(1:N+1) × sum(1:N+1) matrix operator
            if transposed
                C = BandedBlockBandedMatrix(Zeros{T}(sum(1:(N+1)),sum(1:(N+1))),
                                            (1:N+1, 1:N+1), (band,0), (0,0))
            else
                C = BandedBlockBandedMatrix(Zeros{T}(sum(1:(N+1)),sum(1:(N+1))),
                                            (1:N+1, 1:N+1), (0,band), (0,0))
            end
            P = getPspace(S)
            ptsr, wr = pointswithweights(getRspace(St, 0), N+1)
            rhoptsr = S.family.ρ.(ptsr)
            getopnorms(St, N)

            # Get pt evals for H OPs
            for k = 0:N
                R = getRspace(S, k)
                Rt = getRspace(St, k)
                getopptseval(R, N-k, ptsr)
                getopptseval(Rt, N-k, ptsr)
            end

            n, k = 0, 0; m = n
            view(C, Block(m+1, n+1))[k+1, k+1] = sum(wr) * getopnorm(P) / St.opnorms[k+1]
            for n=1:N, k=0:n
                R = getRspace(S, k)
                Rt = getRspace(St, k)
                for m = n-band:n
                    if k ≤ m
                        val = inner2(R, opevalatpts(R, n-k+1, ptsr),
                                        rhoptsr.^(2k) .* opevalatpts(Rt, m-k+1, ptsr), wr)
                        val *= getopnorm(P)
                        if transposed
                            view(C, Block(n+1, m+1))[k+1, k+1] = val / St.opnorms[k+1]
                        else
                            view(C, Block(m+1, n+1))[k+1, k+1] = val / St.opnorms[k+1]
                        end
                    end
                end
            end
            C
        # Case 2: Takes the space H^{a,b,c} -> H^{a,b,c+1}
        # and Case 3: Takes the space H^{a,b,c} -> H^{a+1,b+1,c+1}
        elseif (λ == 0 && μ == 1) || (λ == 1 && μ == 1)
            # Outputs the relevant sum(1:N+1) × sum(1:N+1) matrix operator
            if transposed
                C = BandedBlockBandedMatrix(Zeros{T}(sum(1:(N+1)),sum(1:(N+1))),
                                            (1:N+1, 1:N+1), (band,0), (2,0))
            else
                C = BandedBlockBandedMatrix(Zeros{T}(sum(1:(N+1)),sum(1:(N+1))),
                                            (1:N+1, 1:N+1), (0,band), (0,2))
            end
            P = getPspace(S)
            Pt = getPspace(St)
            ptsp, wp = pointswithweights(Pt, N+2)
            ptsr, wr = pointswithweights(getRspace(St, 0), N+1)
            rhoptsr = S.family.ρ.(ptsr)
            # Get pt evals for 1D OPs
            getopptseval(P, N, ptsp)
            getopptseval(Pt, N, ptsp)
            getopnorms(St, N)
            for k = 0:N
                R = getRspace(S, k)
                Rt = getRspace(St, k)
                getopptseval(R, N-k, ptsr)
                getopptseval(Rt, N-k, ptsr)
            end
            for n=0:N, k=0:n
                R = getRspace(S, k)
                for m = n-band:n, j = k-2:2:k
                    if m ≥ 0 && 0 ≤ j ≤ m
                        Rt = getRspace(St, j)
                        val = inner2(R, opevalatpts(R, n-k+1, ptsr),
                                        rhoptsr.^(k+j) .* opevalatpts(Rt, m-j+1, ptsr), wr)
                        val *= inner2(P, opevalatpts(P, k+1, ptsp), opevalatpts(Pt, j+1, ptsp), wp)
                        if transposed
                            view(C, Block(n+1, m+1))[k+1, j+1] = val / St.opnorms[j+1]
                        else
                            view(C, Block(m+1, n+1))[j+1, k+1] = val / St.opnorms[j+1]
                        end
                    end
                end
            end
            C
        else
            error("Invalid DiskSliceSpace")
        end
    elseif weighted == true
        λmult = S.family.nparams - 1
        λ = Int(S.params[1] - St.params[1])
        μ = Int(S.params[end] - St.params[end])
        band = 2μ + λmult*λ

        # Case 4: Takes the space W^{a,b,c} -> W^{a-1,b-1,c}
        if λ == 1 && μ == 0
            # Outputs the relevant sum(1:N+1+band) × sum(1:N+1) matrix operator
            if transposed
                C = BandedBlockBandedMatrix(Zeros{T}(sum(1:(N+1)),sum(1:(N+1+band))),
                                            (1:N+1, 1:N+1+band), (0,band), (0,0))
            else
                C = BandedBlockBandedMatrix(Zeros{T}(sum(1:(N+1+band)),sum(1:(N+1))),
                                            (1:N+1+band, 1:N+1), (band,0), (0,0))
            end
            P = getPspace(S)
            ptsr, wr = pointswithweights(getRspace(S, 0), N+1)
            getopnorms(St, N+1)
            rhoptsr = S.family.ρ.(ptsr)

            # Get pt evals for R OPs
            for k = 0:N
                R = getRspace(S, k)
                Rt = getRspace(St, k)
                getopptseval(R, N-k, ptsr)
                getopptseval(Rt, N-k+1, ptsr)
            end

            for n=0:N, k=0:n
                R = getRspace(S, k)
                Rt = getRspace(St, k)
                for m = n:n+band
                    val = inner2(R, opevalatpts(R, n-k+1, ptsr),
                                    rhoptsr.^(2k) .* opevalatpts(Rt, m-k+1, ptsr), wr)
                    val *= getopnorm(P)
                    if transposed
                        view(C, Block(n+1, m+1))[k+1, k+1] = val / St.opnorms[k+1]
                    else
                        view(C, Block(m+1, n+1))[k+1, k+1] = val / St.opnorms[k+1]
                    end
                end
            end
            C
        # Case 5: Takes the space W^{a,b,c} -> W^{a,b,c-1}
        # and Case 6: Takes the space W^{a,b,c} -> W^{a-1,b-1,c-1}
        elseif (λ == 0 && μ == 1) || (λ == 1 && μ == 1)
            # Outputs the relevant sum(1:N+1+band) × sum(1:N+1) matrix operator
            if transposed
                C = BandedBlockBandedMatrix(Zeros{T}(sum(1:(N+1)),sum(1:(N+1+band))),
                                            (1:N+1, 1:N+1+band), (0,band), (0,2))
            else
                C = BandedBlockBandedMatrix(Zeros{T}(sum(1:(N+1+band)),sum(1:(N+1))),
                                            (1:N+1+band, 1:N+1), (band,0), (2,0))
            end
            P = getPspace(S)
            Pt = getPspace(St)
            ptsp, wp = pointswithweights(P, N+2)
            ptsr, wr = pointswithweights(getRspace(S, 0), N+2)
            rhoptsr = S.family.ρ.(ptsr)
            getopnorms(St, N+band)

            # Get pt evals for P and H OPs
            getopptseval(P, N, ptsp)
            getopptseval(Pt, N+band, ptsp)
            for k = 0:N
                R = getRspace(S, k)
                getopptseval(R, N-k, ptsr)
            end
            for j = 0:N+band
                Rt = getRspace(St, j)
                getopptseval(Rt, N+band-j, ptsr)
            end

            for n=0:N, k=0:n
                R = getRspace(S, k)
                for m = n:n+band, j = k:2:min(k+2, m)
                    Rt = getRspace(St, j)
                    val = inner2(R, opevalatpts(R, n-k+1, ptsr),
                                    rhoptsr.^(k+j) .* opevalatpts(Rt, m-j+1, ptsr), wr)
                    val *= inner2(P, opevalatpts(P, k+1, ptsp), opevalatpts(Pt, j+1, ptsp), wp)
                    if transposed
                        view(C, Block(n+1, m+1))[k+1, j+1] = val / St.opnorms[j+1]
                    else
                        view(C, Block(m+1, n+1))[j+1, k+1] = val / St.opnorms[j+1]
                    end
                end
            end
            C
        else
            error("Invalid DiskSliceSpace")
        end
    end
end

#====#
# Laplacian and biharmonic operator matrices

function laplaceoperator(S::DiskSliceSpace{<:Any, <:Any, T, <:Any},
            St::DiskSliceSpace{<:Any, <:Any, T, <:Any}, N;
            weighted=false, square=true) where T
    # Outputs the sum(1:N+1) × sum(1:N+1) matrix operator if square=true
    D = S.family
    if (weighted == true && S.params == ntuple(x->1, D.nparams)
            && St.params == ntuple(x->1, D.nparams))

        A = partialoperatorx(differentiateweightedspacex(S), N+D.nparams)
        @show "laplaceoperator", "1 of 6 done"
        B = weightedpartialoperatorx(S, N)
        @show "laplaceoperator", "2 of 6 done"
        C = transformparamsoperator(differentiatespacey(D(S.params .- 1)), S, N+D.nparams-1)
        @show "laplaceoperator", "3 of 6 done"
        E = partialoperatory(D(S.params .- 1), N+D.nparams)
        @show "laplaceoperator", "4 of 6 done"
        F = transformparamsoperator(differentiateweightedspacey(S), D(S.params .- 1), N+1, weighted=true)
        @show "laplaceoperator", "5 of 6 done"
        G = weightedpartialoperatory(S, N)
        @show "laplaceoperator", "6 of 6 done"
        L = A * B + C * E * F * G
        if square
            m = sum(1:(N+1))
            Δ = BandedBlockBandedMatrix(L[1:m, 1:m], (1:N+1, 1:N+1), (L.l,L.u), (L.λ,L.μ))
        else
            L
        end
    elseif (weighted == true && S.params == ntuple(x->2, D.nparams)
            && St.params == ntuple(x->0, D.nparams))
        A = weightedpartialoperatorx(differentiateweightedspacex(S), N+D.nparams)
        @show "laplaceoperator", "1 of 6 done"
        B = weightedpartialoperatorx(S, N)
        @show "laplaceoperator", "2 of 6 done"
        C = transformparamsoperator(differentiateweightedspacey(D(S.params .- 1)), D(S.params .- 2), N+D.nparams+1, weighted=true)
        @show "laplaceoperator", "3 of 6 done"
        E = weightedpartialoperatory(D(S.params .- 1), N+D.nparams)
        @show "laplaceoperator", "4 of 6 done"
        F = transformparamsoperator(differentiateweightedspacey(S), D(S.params .- 1), N+1, weighted=true)
        @show "laplaceoperator", "5 of 6 done"
        G = weightedpartialoperatory(S, N)
        @show "laplaceoperator", "6 of 6 done"
        L = A * B + C * E * F * G
        if square
            m = sum(1:(N+1))
            Δ = BandedBlockBandedMatrix(L[1:m, 1:m], (1:N+1, 1:N+1), (L.l, L.u), (L.λ, L.μ))
        else
            L
        end
    elseif (weighted == false && S.params == ntuple(x->0, D.nparams)
            && St.params == ntuple(x->2, D.nparams))
        A = partialoperatorx(differentiatespacex(S), N+1)
        @show "laplaceoperator", "1 of 6 done"
        B = partialoperatorx(S, N+2)
        @show "laplaceoperator", "2 of 6 done"
        C = transformparamsoperator(differentiatespacey(D(S.params .+ 1)), D(S.params .+ 2), N)
        @show "laplaceoperator", "3 of 6 done"
        E = partialoperatory(D(S.params .+ 1), N+1)
        @show "laplaceoperator", "4 of 6 done"
        F = transformparamsoperator(differentiatespacey(S), D(S.params .+ 1), N+1)
        @show "laplaceoperator", "5 of 6 done"
        G = partialoperatory(S, N+2)
        @show "laplaceoperator", "6 of 6 done"
        AA = A * B
        BB = C * E * F * G
        L = BandedBlockBandedMatrix(sparse(AA) + sparse(BB), (1:N+1, 1:N+3),
                                    (max(AA.l,BB.l),max(AA.u,BB.u)), (max(AA.λ,BB.λ),max(AA.μ,BB.μ)))
        if square
            m = sum(1:(N+1))
            Δ = BandedBlockBandedMatrix(L[1:m, 1:m], (1:N+1, 1:N+1), (L.l,L.u), (L.λ,L.μ))
        else
            L
        end
    else
        error("Invalid DiskSliceSpace for Laplacian operator")
    end
end

function biharmonicoperator(S::DiskSliceSpace{<:Any, <:Any, T, <:Any}, N; square=true) where T
    D = S.family
    if S.params == ntuple(x->2, D.nparams)
        A = laplaceoperator(D(S.params .- 2), S, N+2(D.nparams-1); square=false)
        B = laplaceoperator(S, D(S.params .- 2), N; weighted=true, square=false)
        C = A * B
        if square
            m = sum(1:(N+1))
            Δ2 = BandedBlockBandedMatrix(C[1:m, 1:m], (1:N+1, 1:N+1), (C.l, C.u), (C.λ, C.μ))
        else
            C
        end
    else
        error("Invalid HalfDiskSpace for Laplacian operator")
    end
end

# end # module
