
export DiskSliceFamily, DiskSliceSpace

# R should be Float64, B should be BigFloat for best accuracy
abstract type DiskFamily{B,R,N} end


# TODO: Should I make this:
#   struct DiskSlice{B,T} <: Domain{SVector{2,B}}
#       checkpoints::Vector{SArray{Tuple{2},T,1,2}}
#   end
#   domain(S::DiskSliceSpace{DF,B,T,N}) where {DF,B,T,N} = DiskSlice(B,T,S.checkpoints)
#   DiskSlice(::Type{B}, ::Type{T}, checkpoints) where {B,T} = DiskSlice{B,T}(checkpoints)
#   checkpoints(S::DiskSlice) = S.checkpoints

# TODO
struct DiskSlice{B,T} <: Domain{SVector{2,B}} end
DiskSlice(::Type{B}, ::Type{T}) where {B,T} = DiskSlice{B,T}()
checkpoints(::DiskSlice) = [SVector(0.23,-0.78), SVector(0.76,0.56)]
in(x::SVector{2}, D::DiskSlice) = D.α ≤ x[1] ≤ D.β && D.γ*D.ρ(x[1]) ≤ x[2] ≤ D.δ*D.ρ(x[1])

struct DiskSliceSpace{DF, B, T, N} <: Space{DiskSlice{B,T}, T}
    family::DF # Pointer back to the family
    params::NTuple{N,B} # Parameters
    opnorms::Vector{B} # squared norms
    opptseval::Vector{Vector{B}} # Store the ops evaluated at the transform pts
    xderivopptseval::Vector{Vector{B}} # Store the x deriv of the ops evaluated
                                    # at the transform pts
    yderivopptseval::Vector{Vector{B}} # Store the y deriv of the ops evaluated
                                    # at the transform pts
    A::Vector{SparseMatrixCSC{B}} # A, B, C, DT store the clenshaw matrices (for function evaluation)
    B::Vector{SparseMatrixCSC{B}}
    C::Vector{SparseMatrixCSC{B}}
    DT::Vector{SparseMatrixCSC{B}}
end

function DiskSliceSpace(fam::DiskFamily{B,T,N}, params::NTuple{N,B}) where {B,T,N}
    DiskSliceSpace{typeof(fam), B, T, N}(
        fam, params, Vector{B}(), Vector{Vector{B}}(),
        Vector{Vector{B}}(), Vector{Vector{B}}(), Vector{SparseMatrixCSC{B}}(),
        Vector{SparseMatrixCSC{B}}(), Vector{SparseMatrixCSC{B}}(),
        Vector{SparseMatrixCSC{B}}())
end

# TODO
spacescompatible(A::DiskSliceSpace, B::DiskSliceSpace) = (A.params == B.params)
domain(::DiskSliceSpace{DF,B,T,N}) where {DF,B,T,N} = DiskSlice(B,T)

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

# Creates and returns the space desired
function (D::DiskSliceFamily{B,T,N,<:Any,<:Any,<:Any,<:Any})(params::NTuple{N,B}) where {B,T,N}
    haskey(D.spaces,params) && return D.spaces[params]
    D.spaces[params] = DiskSliceSpace(D, params)
end
(D::DiskSliceFamily{B,T,N,<:Any,<:Any,<:Any,<:Any})(params::Vararg{T,N}) where {B,T,N} =
    D(B.(params))

# Constructor for the Family
function DiskSliceFamily(::Type{B},::Type{T}, α::T, β::T, γ::T, δ::T) where {B,T}
    nparams = 3 # Default
    X = Fun(identity, B(α)..β)
    Y = Fun(identity, B(γ)..δ)
    ρ = sqrt(1 - X^2)
    ρ2 = 1 - X^2 # NOTE we use ρ^2 here to help computationally
    if isinteger(β) && Int(β) == 1 # 2-param family
        nparams -= 1
        R = OrthogonalPolynomialFamily(T, X-α, ρ2)
    else
        R = OrthogonalPolynomialFamily(T, β-X, X-α, ρ2)
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
# Retrieve 1D OP spaces methods
function getRspace(S::DiskSliceSpace, k::Int)
    if S.family.nparams == 3
        (S.family.R)(S.params[1], S.params[2], (2S.params[3] + 2k + 1) / 2)
    else
        (S.family.R)(S.params[1], (2S.params[2] + 2k + 1)/2)
    end
end
function getRspace(S::DiskSliceSpace)
    if S.family.nparams == 3
        (S.family.R)(S.params[1], S.params[2], S.params[3])
    else
        (S.family.R)(S.params[1], S.params[2])
    end
end
getPspace(S::DiskSliceSpace) = (S.family.P)(S.params[end], S.params[end])

#===#
# Weight eval functions for the 2D space
function weight(S::DiskSliceSpace, x, y)
    # T(getRspace(S).weight(x) * getPspace(S).weight(y/S.family.ρ(x)))
    if length(S.params) == 2
        (x - S.family.α)^S.params[1] * (1 - x^2 - y^2)^S.params[2]
    else # length(S.params) == 3
        a, b, c = S.params
        (S.family.β - x)^a * (x - S.family.α)^b * (1 - x^2 - y^2)^c
    end
end
weight(S::DiskSliceSpace, z) = weight(S, z[1], z[2])
function weight(::Type{T}, S::DiskSliceSpace, x, y) where T
    # T(getRspace(S).weight(x) * getPspace(S).weight(y/S.family.ρ(x)))
    if length(S.params) == 2
        T((x - S.family.α)^S.params[1] * (1 - x^2 - y^2)^S.params[2])
    else # length(S.params) == 3
        a, b, c = S.params
        T((S.family.β - x)^a * (x - S.family.α)^b * (1 - x^2 - y^2)^c)
    end
end
weight(::Type{T}, S::DiskSliceSpace, z) where T = weight(T, S, z[1], z[2])



#===#
# points() and methods for pt evals and norm vals

# NOTE we output ≈n points (x,y), plus the ≈n points corresponding to (x,-y)
function pointswithweights(S::DiskSliceSpace{<:Any, B, T, <:Any}, n) where {B,T}
    # Return the weights and nodes to use for the even part of a function,
    # i.e. for the disk-slice Ω:
    #   int_Ω W^{a,b}(x,y) f(x,y) dydx ≈ Σ_j weⱼ*fe(xⱼ,yⱼ)
    # NOTE: the odd part of the quad rule will equal 0 for polynomials,
    #       so can be ignored.
    N = 2 * Int(ceil(sqrt(n))) - 1 # degree we approximate up to with M quadrature pts
    M1 = M2 = Int((N + 1) / 2)
    M = M1 * M2 # ≈ n
    @show "begin pointswithweights()", n, N, M
    t, wt = pointswithweights(B, getPspace(S), M2)
    # Need to maunally call the method to get R coeffs here
    m = isodd(M1) ? Int((M1 + 1) / 2) : Int((M1 + 2) / 2); m -= Int(S.params[end])
    getreccoeffsR!(S, m; maxk=0)
    s, ws = pointswithweights(B, getRspace(S, 0), M1)
    pts = Vector{SArray{Tuple{2},B,1,2}}(undef, 2M) # stores both (x,y) and (x,-y)
    w = zeros(B, M) # weights
    for i = 1:M2
        for k = 1:M1
            x, y = s[k], t[i] * S.family.ρ(s[k])
            pts[i + (k - 1)M1] = x, y
            pts[M + i + (k - 1)M1] = x, -y
            w[i + (k - 1)M1] = ws[k] * wt[i]
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

function getopnorms(S::DiskSliceSpace{<:Any, B, T, <:Any}, k) where {B,T}
    # NOTE these are squared norms
    m = length(S.opnorms)
    if k + 1 > m
        resize!(S.opnorms, k+1)
        P = getPspace(S)
        getopnorm(P)

        # for j = m+1:k+1
        #     S.opnorms[j] = getopnorm(getRspace(S, j-1)) * P.opnorm[1]
        # end

        # We get the weights for the R integral
        ptsr, wr = pointswithweights(B, getRspace(S, 0), k+1)
        rhoptsr = S.family.ρ.(ptsr)
        for j = m:k
            R = getRspace(S, j)
            if !isnormset(R)
                setopnorm(R, sum(rhoptsr.^(2j) .* wr))
            end
            S.opnorms[j+1] = R.opnorm[1] * P.opnorm[1]
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
function opevalatpts(S::DiskSliceSpace{<:Any, B, T, <:Any}, j, pts) where {B,T}
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
        S.opptseval[jj+k] = Vector{B}(undef, length(pts))
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
    S::DiskSliceSpace{<:Any, T, <:Any, <:Any}
end

function DiskSliceTransformPlan(S::DiskSliceSpace{<:Any, B, T, <:Any}, vals) where {B,T}
    m = Int(length(vals) / 2)
    pts, w = pointswithweights(S, m)
    DiskSliceTransformPlan{B}(w, pts, S)
end
plan_transform(S::DiskSliceSpace, vals) = DiskSliceTransformPlan(S, vals)
transform(S::DiskSliceSpace, vals) = plan_transform(S, vals) * vals

# Inputs: OP space, f(pts) for desired f
# Output: Coeffs of the function f for its expansion in the DiskSliceSpace OPs
function *(DSTP::DiskSliceTransformPlan{T}, vals::Vector{T}) where T
    @show "Begin DSTP mult"
    m2 = Int(length(vals) / 2)
    N = Int(sqrt(m2)) - 1
    m1 = Int((N+1)*(N+2) / 2)
    @show m1, m2

    # ret = zeros(T, m1)
    ret = zeros(T, m1)
    resizedata!(DSTP.S, N)
    getopnorms(DSTP.S, N) # We store the norms of the OPs
    for i = 1:m2
        if i % 100 == 0
            @show m2, i
        end
        pt = [DSTP.pts[i]]
        getopptseval(DSTP.S, N, pt)
        for j = 1:m1
            # ret[j] += T(opevalatpts(DSTP.S, j, pt)[1] * DSTP.w[i] * vals[i])
            ret[j] += opevalatpts(DSTP.S, j, pt)[1] * DSTP.w[i] * vals[i]
        end
        pt = [DSTP.pts[i+m2]]
        getopptseval(DSTP.S, N, pt)
        for j = 1:m1
            # ret[j] += T(opevalatpts(DSTP.S, j, pt)[1] * DSTP.w[i] * vals[i+m2])
            ret[j] += opevalatpts(DSTP.S, j, pt)[1] * DSTP.w[i] * vals[i+m2]
        end
    end
    resetopptseval(DSTP.S)
    j = 1
    for n = 0:N, k = 0:n
        # ret[j] /= T(2 * DSTP.S.opnorms[k+1])
        ret[j] /= 2 * DSTP.S.opnorms[k+1]
        j += 1
    end
    @show "End DSTP mult"
    ret
end

# Inputs: OP space, coeffs of a function f for its expansion in the DiskSliceSpace OPs
# Output: vals = {f(x_j)} where x_j are are the points(S,n)
function itransform(S::DiskSliceSpace, cfs::Vector{T}) where T
    @show "begin itransform"
    m = length(cfs)
    pts = points(S, m)
    N = getnk(m)[1]
    npts = length(pts)
    V = Array{T}(undef, npts, m)
    @show m, npts, N
    for i = 1:npts
        pt = [pts[i]]
        getopptseval(S, N, pt)
        for k = 1:m
            V[i, k] = T(S.opptseval[k][1])
        end
    end
    @show "end itransform"
    V * cfs
end

#===#
# Jacobi operator/Clenshaw matrices entries

function recα(::Type{T}, S::DiskSliceSpace, n, k, j) where T
    R = getRspace(S, k)
    if j == 1
        recγ(T, R, n-k+1)
    elseif j == 2
        recα(T, R, n-k+1)
    else
        error("Invalid entry to function")
    end
end

function recβ(::Type{T}, S::DiskSliceSpace, n, k, j) where T
    # We get the norms of the 2D OPs
    getopnorms(S, k+1)

    R1 = getRspace(S, k-1)
    R2 = getRspace(S, k)
    R3 = getRspace(S, k+1)
    P = getPspace(S)
    getopnorm(P)

    if isodd(j)
        pts, w = pointswithweights(T, R2, Int(ceil(n-k+1.5)))
        δ = recγ(T, P, k+1) * P.opnorm[1]
    else
        pts, w = pointswithweights(T, R3, Int(ceil(n-k+1.5)))
        δ = recβ(T, P, k+1) * P.opnorm[1]
    end
    getopptseval(R2, n-k+1, pts)

    if j == 1
        getopptseval(R1, n-k+1, pts)
        T(inner2(R2, opevalatpts(R2, n-k+1, pts), opevalatpts(R1, n-k+1, pts), w)
            * δ / S.opnorms[k])
    elseif j == 2
        getopptseval(R3, n-k-1, pts)
        T(inner2(R3, opevalatpts(R2, n-k+1, pts), opevalatpts(R3, n-k-1, pts), w)
            * δ / S.opnorms[k+2])
    elseif j == 3
        getopptseval(R1, n-k+2, pts)
        T(inner2(R2, opevalatpts(R2, n-k+1, pts), opevalatpts(R1, n-k+2, pts), w)
            * δ / S.opnorms[k])
    elseif j == 4
        getopptseval(R3, n-k, pts)
        T(inner2(R3, opevalatpts(R2, n-k+1, pts), opevalatpts(R3, n-k, pts), w)
            * δ / S.opnorms[k+2])
    elseif j == 5
        getopptseval(R1, n-k+3, pts)
        T(inner2(R2, opevalatpts(R2, n-k+1, pts), opevalatpts(R1, n-k+3, pts), w)
            * δ / S.opnorms[k])
    elseif j == 6
        getopptseval(R3, n-k+1, pts)
        T(inner2(R3, opevalatpts(R2, n-k+1, pts), opevalatpts(R3, n-k+1, pts), w)
            * δ / S.opnorms[k+2])
    else
        error("Invalid entry to function")
    end
end

function getAs!(S::DiskSliceSpace{<:Any, T, <:Any, <:Any}, N, N₀) where T
    m = N₀
    if m == 0
        S.A[1] = [recα(T, S, 1, 0, 1) 0; 0 recβ(T, S, 0, 0, 6)]
        m += 1
    end
    for n = N+1:-1:m
        v1 = [recα(T, S, n+1, k, 1) for k = 0:n]
        v2 = [recβ(T, S, n, k, 6) for k = 0:n-1]
        v3 = [recβ(T, S, n, k, 5) for k = 1:n]
        S.A[n+1] = [Diagonal(v1) zeros(T, n+1);
                    Tridiagonal(v3, zeros(T, n+1), v2) [zeros(T, n); recβ(T, S, n, n, 6)]]
    end
end

function getDTs!(S::DiskSliceSpace{<:Any, T, <:Any, <:Any}, N, N₀) where T
    for n = N+1:-1:N₀
        vα = [1 / recα(T, S, n+1, k, 1) for k = 0:n]
        m = iseven(n) ? Int(n/2) + 1 : Int((n+1)/2) + 1
        vβ = zeros(T, m)
        vβ[end] = 1 / recβ(T, S, n, n, 6)
        if iseven(n)
            for k = 1:m-1
                vβ[end-k] = - (vβ[end-k+1]
                                * recβ(T, S, n, n - 2(k - 1), 5)
                                / recβ(T, S, n, n - 2k, 6))
            end
        else
            for k = 1:m-2
                vβ[end-k] = - (vβ[end-k+1]
                                * recβ(T, S, n, n - 2(k - 1), 5)
                                / recβ(T, S, n, n - 2k, 6))
            end
            vβ[1] = - (vβ[2]
                        * recβ(T, S, n, 1, 5)
                        / recα(T, S, n+1, 0, 1))
        end
        ij = [i for i=1:n+1]
        ind1 = [ij; [n+2 for i=1:m]]
        ind2 = [ij; iseven(n) ? [n+2k for k=1:m] : [1; [n-1+2k for k=2:m]]]
        S.DT[n+1] = sparse(ind1, ind2, [vα; vβ])
        # S.DT[n+1] = sparse(pinv(Array(S.A[n+1])))
    end
end
function getBs!(S::DiskSliceSpace{<:Any, T, <:Any, <:Any}, N, N₀) where T
    m = N₀
    if N₀ == 0
        S.B[1] = sparse([1, 2], [1, 1], [recα(T, S, 0, 0, 2), 0])
        m += 1
    end
    for n = N+1:-1:m
        # if n > 200
        #     @show "getsBs!", N, n
        # end
        @show "getsBs!", N, n
        v1 = [recα(T, S, n, k, 2) for k = 0:n]
        v2 = [recβ(T, S, n, k, 4) for k = 0:n-1]
        v3 = [recβ(T, S, n, k, 3) for k = 1:n]
        S.B[n+1] = sparse([Diagonal(v1); Tridiagonal(v3, zeros(T, n+1), v2)])
    end
end
function getCs!(S::DiskSliceSpace{<:Any, T, <:Any, <:Any}, N, N₀) where T
    m = N₀
    if N₀ == 0
        # C_0 does not exist
        m += 1
    end
    if m == 1
        S.C[2] = sparse([1, 4], [1, 1], [recα(T, S, 1, 0, 1), recβ(T, S, 1, 1, 1)])
        m += 1
    end
    for n = N+1:-1:m
        v1 = [recα(T, S, n, k, 1) for k = 0:n-1]
        v2 = [recβ(T, S, n, k, 2) for k = 0:n-2]
        v3 = [recβ(T, S, n, k, 1) for k = 1:n-1]
        S.C[n+1] = sparse([Diagonal(v1);
                           zeros(T, 1, n);
                           Tridiagonal(v3, zeros(T, n), v2);
                           [zeros(T, 1, n-1) recβ(T, S, n, n, 1)]])
    end
end

# Method to calculate and store the Clenshaw matrices
function resizedata!(S::DiskSliceSpace, N)
    # N is the max degree of the OPs
    N₀ = length(S.B)
    N ≤ N₀ - 2 && return S
    @show "begin resizedata! for DiskSliceSpace"

    # First, we need to call this to get the 1D OP rec coeffs
    resizedataonedimops!(S, N+3)

    resize!(S.B, N + 2)
    getBs!(S, N, N₀)
    @show "done Bs"
    resize!(S.C, N + 2)
    getCs!(S, N, N₀)
    @show "done Cs"
    resize!(S.A, N + 2)
    getAs!(S, N, N₀)
    @show "done As"
    resize!(S.DT, N + 2)
    getDTs!(S, N, N₀)
    @show "done DTs"
    S
end

# Method to calculate the rec coeffs for the 1D R OPs (the non classical ones)
# using a recursive algorithm invloving use of the Christoffel-Darboux formula
function getreccoeffsR!(S::DiskSliceSpace{<:Any, B, T, <:Any}, N; maxk=-1) where {B,T}
    c = Int(S.params[end])
    # Set maxkval - allows us to, if set, stop the iteration (note that the N
    # value needs to be adjusted accordingly before input)
    if maxk < 0
        maxkval = N + c
    else
        maxkval = maxk + c
    end
    R00 = getRspace(S, -c) # R00 = R(S.a, S.b, B(0.5))
    M = 2 * (N + c)

    # See if we need to proceed (is length(R(0,0,N+c+0.5).α) > 0)
    length(getRspace(S, N).a) > 0 && return S

    # resizedata!() on R(0,0,0.5) up to deg M to initialise
    resizedata!(R00, M+1)

    # Loop over k value in R(0,0,k+0.5) to recursively build the rec coeffs for
    # each OPSpace, so that length(R(0,0,N+c+0.5).α) == 1
    for k = 0:(maxkval - 1)
        if k % 100 == 0
            @show "getreccoeffsR!", k
        end
        # interim coeffs
        R00 = getRspace(S, k - c) # R00 = R(S.a, S.b, B(0.5)+k)
        chivec = zeros(B, M+1)
        pt = B(1)
        n = 0
        chivec[n+1] = (pt - R00.a[n+1]) / R00.b[n+1]
        for n = 1:M
            chivec[n+1] = (pt - R00.a[n+1] - R00.b[n] / chivec[n]) / R00.b[n+1]
        end
        R10a, R10b = zeros(B, M), zeros(B, M)
        n = 0
        R10b[n+1] = (R00.b[n+1]
                        * sqrt(pt - R00.a[n+2] - R00.b[n+1] / chivec[n+1])
                        / sqrt(pt - R00.a[n+1]))
        R10a[n+1] = B(R00.a[n+1]) - (R00.b[n+1] / chivec[n+1])
        for n = 1:M-1
            R10b[n+1] = (R00.b[n+1]
                            * sqrt(pt - R00.a[n+2] - R00.b[n+1] / chivec[n+1])
                            / sqrt(pt - R00.a[n+1] - R00.b[n] / chivec[n]))
            R10a[n+1] = R00.a[n+1] + (R00.b[n] / chivec[n]) - (R00.b[n+1] / chivec[n+1])
        end
        # wanted coeffs
        chivec = zeros(B, M)
        pt = -B(1)
        n = 0
        chivec[n+1] = (pt - R10a[n+1]) / R10b[n+1]
        for n = 1:M-1
            chivec[n+1] = (pt - R10a[n+1] - R10b[n] / chivec[n]) / R10b[n+1]
        end
        R11 = getRspace(S, k - c + 1) # R11 = R(S.a, S.b, B(0.5)+k+1)
        n₀ = length(R11.a); resize!(R11.a, M-1); resize!(R11.b, M-1)
        if n₀ == 0
            n = n₀
            R11.b[n+1] = (R10b[n+1]
                            * sqrt(abs(pt - R10a[n+2] - R10b[n+1] / chivec[n+1]))
                            / sqrt(abs(pt - R10a[n+1])))
            R11.a[n+1] = R10a[n+1] - (R10b[n+1] / chivec[n+1])
            n₀ += 1
        end
        for n = n₀:M-2
            R11.b[n+1] = (R10b[n+1]
                            * sqrt(abs(pt - R10a[n+2] - R10b[n+1] / chivec[n+1]))
                            / sqrt(abs(pt - R10a[n+1] - R10b[n] / chivec[n])))
            R11.a[n+1] = R10a[n+1] + (R10b[n] / chivec[n]) - (R10b[n+1] / chivec[n+1])
        end
        M = M - 2
    end
    S
end

# Method to retrieve the rec coeffs (resizedata!) for the 1D OP families
# associated with the DSSpace's DiskSliceFamily
function resizedataonedimops!(S::DiskSliceSpace, N)
    # N = max degree desired
    length(getPspace(S).a) ≥ N && length(getRspace(S, N).a) > 0 && return S

    # Use our algorithm to generate the rec coeffs recursively for R's
    # We assume that we have R(a,b,0.5).ops up to the previous required degree
    # (to build on) and thus subsequent OP rec coeffs for R(a,b,k+0.5) for
    # k = 1:N
    @show "resizedata! (one dim ops) for DSSpace", Float64.(S.params)
    getreccoeffsR!(S, N)
    resizedata!(getPspace(S), N)
    S
end

# Methods to return the Jacobi operator matrices for mult by x, y
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
# transform pts given. Used for quicker computation of the operator matrices

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
function xderivopevalatpts(S::DiskSliceSpace{<:Any, T, <:Any, <:Any}, j, pts) where T
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
function yderivopevalatpts(S::DiskSliceSpace{<:Any, T, <:Any, <:Any}, j, pts) where T
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
# Methods to return the spaces corresponding to applying the derivative operators

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

# Convienience mathods to return the ApproxFun.Fun() or the coeffs in the
# relevent space for differentiation
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


#====#
# Differential operator matrices

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
    # T(val)
    val
end
function partialoperatorx(S::DiskSliceSpace{<:Any, B, T, <:Any}, N;
                            transposed=false) where {B,T}
    # Takes the space P^{a,b,c} -> P^{a+1,b+1,c+1}
    Sx = differentiatespacex(S)
    # resizedataonedimops! for both DSSpaces
    resizedataonedimops!(S, N)
    resizedataonedimops!(Sx, N+4)

    P = getPspace(S)
    Px = getPspace(Sx)
    ptsp, wp = pointswithweights(B, Px, N+2) # TODO
    getopptseval(P, N, ptsp)
    getderivopptseval(P, N, ptsp)
    getopptseval(Px, N, ptsp)
    ptsr, wr = pointswithweights(B, getRspace(Sx, -1), N+4)
    getopnorms(Sx, N-1)

    # Evaluate ρ.(ptsr) dρ/dx.(ptsr) at the R inner product points
    rhoptsr = S.family.ρ.(ptsr)
    dxrhoptsr = differentiate(S.family.ρ).(ptsr)

    band = S.family.nparams
    if transposed
        A = BandedBlockBandedMatrix(
            Zeros{B}(sum(1:(N+1)), sum(1:N)), (1:N+1, 1:N), (band, -1), (2, 0))
    else
        A = BandedBlockBandedMatrix(
            Zeros{B}(sum(1:N), sum(1:(N+1))), (1:N, 1:N+1), (-1, band), (0, 2))
    end

    n, k = 1, 0
    m, j = n-1, k
    R = getRspace(S, k)
    getopptseval(R, n-k, ptsr)
    getderivopptseval(R, n-k, ptsr)
    Rx = getRspace(Sx, j)
    getopptseval(Rx, m-j, ptsr)
    val = getpartialoperatorxval(S, ptsp, wp, ptsr, rhoptsr, dxrhoptsr, wr, n, k, m, j)
    view(A, Block(m+1, n+1))[1, 1] = val
    resetopptseval(R)
    resetderivopptseval(R)
    resetopptseval(Rx)
    for k = 0:N
        if k % 20 == 0
            @show "dx", k
        end
        R = getRspace(S, k)
        getopptseval(R, N-k, ptsr)
        getderivopptseval(R, N-k, ptsr)
        for j = (k-2):2:k
            if j < 0 || j > N-1
                continue
            end
            Rx = getRspace(Sx, j)
            getopptseval(Rx, N-1-j, ptsr)
            for n = max(k,2):N
                for m = max(0, n-band):(n-1)
                    if j > m
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
            resetopptseval(Rx)
        end
        resetopptseval(R)
        resetderivopptseval(R)
    end
    A
end
function partialoperatory(S::DiskSliceSpace{<:Any, B, T, <:Any}, N;
                            transposed=false) where {B,T}
    # Takes the space H^{a,b,c} -> H^{a,b,c+1}
    if transposed
        A = BandedBlockBandedMatrix(
            Zeros{B}(sum(1:(N+1)),sum(1:N)), (1:N+1, 1:N), (1,-1), (1,-1))
    else
        A = BandedBlockBandedMatrix(
            Zeros{B}(sum(1:N),sum(1:(N+1))), (1:N, 1:N+1), (-1,1), (-1,1))
    end
    Sy = differentiatespacey(S)
    P = getPspace(S)
    Py = getPspace(Sy)
    pts, w = pointswithweights(B, Py, N)
    getopptseval(P, N, pts)
    getderivopptseval(P, N, pts)
    getopptseval(Py, N, pts)
    getopnorms(Sy, N-1)
    getopnorms(S, N)
    for k = 1:N
        if k % 100 == 0
            @show "dy", k
        end
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
function getweightedpartialoperatorxval(S::DiskSliceSpace{<:Any, B, T, <:Any},
                ptsp, wp1, wp, ptsr, rhoptsr, dxrhoptsr, wr010, wr100, wr, n, k, m, j) where {B,T}
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
function weightedpartialoperatorx(S::DiskSliceSpace{<:Any, B, T, <:Any}, N;
                                    transposed=false) where {B,T}
    # Takes weighted space ∂/∂x(W^{a,b,c}) -> W^{a-1,b-1,c-1}
    band = S.family.nparams
    if transposed
        W = BandedBlockBandedMatrix(
            Zeros{B}(sum(1:(N+1)),sum(1:(N+1+band))), (1:N+1, 1:N+1+band), (-1, band), (0, 2))
    else
        W = BandedBlockBandedMatrix(
            Zeros{B}(sum(1:(N+1+band)),sum(1:(N+1))), (1:N+1+band, 1:N+1), (band, -1), (2, 0))
    end
    Sx = differentiateweightedspacex(S)
    P = getPspace(S)
    Px = getPspace(Sx)
    ptsp, wp = pointswithweights(B, Px, N+3)
    getopptseval(P, N, ptsp)
    getopptseval(Px, N+1, ptsp)
    getderivopptseval(P, N, ptsp)
    ptsr, wr = pointswithweights(B, getRspace(Sx, 0), N+4)
    getopnorms(Sx, N+2)

    # ρ.(ptsr) and dρ/dx.(ptsr)
    rhoptsr = S.family.ρ.(ptsr)
    dxrhoptsr = differentiate(S.family.ρ).(ptsr)
    # w_P^{(1)}.(pts)
    wp1 = (-ptsp.^2 .+ 1) # TODO - dont hardcode!
    # w_R^{(1,0,0)}, w_R^{(0,1,0)}
    wr100 = S.family.nparams == 3 ? (-ptsr .+ S.family.β) : ones(length(ptsr))
    wr010 = (ptsr .- S.family.α)

    # Get pt evals for the R OPs
    for k = 0:N
        if k % 20 == 0
            @show "wghtd dx", k
        end
        R = getRspace(S, k)
        getopptseval(R, N-k, ptsr)
        getderivopptseval(R, N-k, ptsr)
        for j = k:2:(k+2)
            if j > N+S.family.nparams
                continue
            end
            Rx = getRspace(Sx, j)
            getopptseval(Rx, N+S.family.nparams-j, ptsr)
            for n = k:N
                for m = n+1:n+S.family.nparams
                    if j > m
                        continue
                    end
                    val = getweightedpartialoperatorxval(S, ptsp, wp1, wp, ptsr,
                                    rhoptsr, dxrhoptsr, wr010, wr100, wr, n, k, m, j)
                    if transposed
                        view(W, Block(n+1, m+1))[k+1, j+1] = val
                    else
                        view(W, Block(m+1, n+1))[j+1, k+1] = val
                    end
                end
            end
            resetopptseval(Rx)
        end
        resetopptseval(R)
        resetderivopptseval(R)
    end
    W
end
function weightedpartialoperatory(S::DiskSliceSpace{<:Any, B, T, <:Any}, N;
                                    transposed=false) where {B,T}
    # Takes weighted space ∂/∂y(W^{a,b,c}) -> W^{a,b,c-1}
    if transposed
        W = BandedBlockBandedMatrix(
            Zeros{B}(sum(1:(N+1)),sum(1:(N+2))), (1:N+1, 1:N+2), (-1,1), (-1,1))
    else
        W = BandedBlockBandedMatrix(
            Zeros{B}(sum(1:(N+2)),sum(1:(N+1))), (1:N+2, 1:N+1), (1,-1), (1,-1))
    end
    Sy = differentiateweightedspacey(S)
    P = getPspace(S)
    Py = getPspace(Sy)
    ptsp, wp = pointswithweights(B, Py, N+2)
    getopptseval(P, N, ptsp)
    getopptseval(Py, N+1, ptsp)
    getderivopptseval(P, N, ptsp)
    getopnorms(Sy, N+1)
    getopnorms(S, N)
    params = S.params .* 0 .+ ntuple(i -> i == S.family.nparams ? 1 : 0, S.family.nparams)
    wp1 = getweightfun(getPspace(S.family(params))).(ptsp) # (-ptsp.^2 .+ 1)
    n, m = N, N+1
    for k = 0:N
        if k % 100 == 0
            @show "wghtd dy", k
        end
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

function transformparamsoperator(S::DiskSliceSpace{<:Any, B, T, <:Any},
            St::DiskSliceSpace{<:Any, B, T, <:Any}, N;
            weighted=false, transposed=false) where {B,T}
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
                C = BandedBlockBandedMatrix(Zeros{B}(sum(1:(N+1)),sum(1:(N+1))),
                                            (1:N+1, 1:N+1), (band,0), (0,0))
            else
                C = BandedBlockBandedMatrix(Zeros{B}(sum(1:(N+1)),sum(1:(N+1))),
                                            (1:N+1, 1:N+1), (0,band), (0,0))
            end
            P = getPspace(S)
            ptsr, wr = pointswithweights(B, getRspace(St, 0), N+1)
            rhoptsr = S.family.ρ.(ptsr)
            getopnorms(St, N)

            n, k = 0, 0; m = n
            view(C, Block(m+1, n+1))[k+1, k+1] = sum(wr) * getopnorm(P) / St.opnorms[k+1]
            for k = 0:N
                if k % 100 == 0
                    @show "trnsfrm (a,b,c)->(a+1,b+1,c)", k
                end
                R = getRspace(S, k)
                Rt = getRspace(St, k)
                getopptseval(R, N-k, ptsr)
                getopptseval(Rt, N-k, ptsr)
                for n = max(1, k):N
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
                resetopptseval(R)
                resetopptseval(Rt)
            end
            C
        # Case 2: Takes the space H^{a,b,c} -> H^{a,b,c+1}
        # and Case 3: Takes the space H^{a,b,c} -> H^{a+1,b+1,c+1}
        elseif (λ == 0 && μ == 1) || (λ == 1 && μ == 1)
            # Outputs the relevant sum(1:N+1) × sum(1:N+1) matrix operator
            if transposed
                C = BandedBlockBandedMatrix(Zeros{B}(sum(1:(N+1)),sum(1:(N+1))),
                                            (1:N+1, 1:N+1), (band,0), (2,0))
            else
                C = BandedBlockBandedMatrix(Zeros{B}(sum(1:(N+1)),sum(1:(N+1))),
                                            (1:N+1, 1:N+1), (0,band), (0,2))
            end
            P = getPspace(S)
            Pt = getPspace(St)
            ptsp, wp = pointswithweights(B, Pt, N+2)
            ptsr, wr = pointswithweights(B, getRspace(St, 0), N+1)
            rhoptsr = S.family.ρ.(ptsr)
            getopptseval(P, N, ptsp)
            getopptseval(Pt, N, ptsp)
            getopnorms(St, N)

            for k = 0:N
                if k % 100 == 0
                    @show "trnsfrm (a,b,c)->(a+1,b+1,c+1)", k
                end
                R = getRspace(S, k)
                getopptseval(R, N-k, ptsr)
                for j = k-2:2:k
                    if j < 0
                        continue
                    end
                    Rt = getRspace(St, j)
                    getopptseval(Rt, N-j, ptsr)
                    for n = k:N
                        for m = n-band:n
                            if m < 0 || m < j
                                continue
                            end
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
                    resetopptseval(Rt)
                end
                resetopptseval(R)
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
                C = BandedBlockBandedMatrix(Zeros{B}(sum(1:(N+1)),sum(1:(N+1+band))),
                                            (1:N+1, 1:N+1+band), (0,band), (0,0))
            else
                C = BandedBlockBandedMatrix(Zeros{B}(sum(1:(N+1+band)),sum(1:(N+1))),
                                            (1:N+1+band, 1:N+1), (band,0), (0,0))
            end
            P = getPspace(S)
            ptsr, wr = pointswithweights(B, getRspace(S, 0), N+1)
            getopnorms(St, N+1)
            rhoptsr = S.family.ρ.(ptsr)

            for k = 0:N
                if k % 20 == 0
                    @show "wtrnsfrm (a,b,c)->(a-1,b-1,c)", k
                end
                R = getRspace(S, k)
                Rt = getRspace(St, k)
                getopptseval(R, N-k, ptsr)
                getopptseval(Rt, N+band-k, ptsr)
                for n = k:N, m = n:n+band
                    val = inner2(R, opevalatpts(R, n-k+1, ptsr),
                                    rhoptsr.^(2k) .* opevalatpts(Rt, m-k+1, ptsr), wr)
                    val *= getopnorm(P)
                    if transposed
                        view(C, Block(n+1, m+1))[k+1, k+1] = val / St.opnorms[k+1]
                    else
                        view(C, Block(m+1, n+1))[k+1, k+1] = val / St.opnorms[k+1]
                    end
                end
                resetopptseval(R)
                resetopptseval(Rt)
            end
            C
        # Case 5: Takes the space W^{a,b,c} -> W^{a,b,c-1}
        # and Case 6: Takes the space W^{a,b,c} -> W^{a-1,b-1,c-1}
        elseif (λ == 0 && μ == 1) || (λ == 1 && μ == 1)
            # Outputs the relevant sum(1:N+1+band) × sum(1:N+1) matrix operator
            if transposed
                C = BandedBlockBandedMatrix(Zeros{B}(sum(1:(N+1)),sum(1:(N+1+band))),
                                            (1:N+1, 1:N+1+band), (0,band), (0,2))
            else
                C = BandedBlockBandedMatrix(Zeros{B}(sum(1:(N+1+band)),sum(1:(N+1))),
                                            (1:N+1+band, 1:N+1), (band,0), (2,0))
            end
            P = getPspace(S)
            Pt = getPspace(St)
            ptsp, wp = pointswithweights(B, P, N+2)
            ptsr, wr = pointswithweights(B, getRspace(S, 0), N+2)
            rhoptsr = S.family.ρ.(ptsr)
            getopnorms(St, N+band)

            # Get pt evals for P OPs
            getopptseval(P, N, ptsp)
            getopptseval(Pt, N+band, ptsp)

            for k = 0:N
                if k % 20 == 0
                    @show "wtrnsfrm (a,b,c)->(a-1,b-1,c-1)", k
                end
                R = getRspace(S, k)
                getopptseval(R, N-k, ptsr)
                for j = k:2:(k+2)
                    if j > N+band
                        continue
                    end
                    Rt = getRspace(St, j)
                    getopptseval(Rt, N+band-j, ptsr)
                    for n = k:N
                        for m = n:n+band
                            if m < j
                                continue
                            end
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
                    resetopptseval(Rt)
                end
                resetopptseval(R)
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
        AAl, AAu = A.l + B.l, A.u + B.u
        BBl, BBu = C.l + E.l + F.l + G.l, C.u + E.u + F.u + G.u
        AAλ, AAμ = A.λ + B.λ, A.μ + B.μ
        BBλ, BBμ = C.λ + E.λ + F.λ + G.λ, C.μ + E.μ + F.μ + G.μ
        AA = sparse(A) * sparse(B)
        BB = sparse(C) * sparse(E) * sparse(F) * sparse(G)
        L = BandedBlockBandedMatrix(AA + BB, (1:nblocks(A)[1], 1:nblocks(B)[2]),
                                    (max(AAl,BBl),max(AAu,BBu)), (max(AAλ,BBλ),max(AAμ,BBμ)))
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
        # NOTE: Multiplying the BlockBandedMatrices fails with an error, so
        #       convert to sparse
        AAl, AAu = A.l + B.l, A.u + B.u
        BBl, BBu = C.l + E.l + F.l + G.l, C.u + E.u + F.u + G.u
        AAλ, AAμ = A.λ + B.λ, A.μ + B.μ
        BBλ, BBμ = C.λ + E.λ + F.λ + G.λ, C.μ + E.μ + F.μ + G.μ
        AA = sparse(A) * sparse(B)
        BB = sparse(C) * sparse(E) * sparse(F) * sparse(G)
        L = BandedBlockBandedMatrix(AA + BB, (1:nblocks(A)[1], 1:nblocks(B)[2]),
                                    (max(AAl,BBl),max(AAu,BBu)), (max(AAλ,BBλ),max(AAμ,BBμ)))
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
        # NOTE: Multiplying the BlockBandedMatrices fails with an error, so
        #       convert to sparse
        AAl, AAu = A.l + B.l, A.u + B.u
        BBl, BBu = C.l + E.l + F.l + G.l, C.u + E.u + F.u + G.u
        AAλ, AAμ = A.λ + B.λ, A.μ + B.μ
        BBλ, BBμ = C.λ + E.λ + F.λ + G.λ, C.μ + E.μ + F.μ + G.μ
        AA = sparse(A) * sparse(B)
        BB = sparse(C) * sparse(E) * sparse(F) * sparse(G)
        L = BandedBlockBandedMatrix(AA + BB, (1:N+1, 1:N+3),
                                    (max(AAl,BBl),max(AAu,BBu)), (max(AAλ,BBλ),max(AAμ,BBμ)))
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
