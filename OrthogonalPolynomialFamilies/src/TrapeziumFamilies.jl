# module TrapeziumFamilies


export TrapeziumFamily, TrapeziumSpace

# R should be Float64, B should be BigFloat
abstract type DiskFamily{B,R,N} end
struct Trapezium{B,T} <: Domain{SVector{2,T}} end
Trapezium() = Trapezium{BigFloat, Float64}()
checkpoints(::Trapezium) = [SVector(0.1,0.23), SVector(0.3,0.12)]

struct TrapeziumSpace{DF, B, T, N} <: Space{Trapezium{B,T}, T}
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

function TrapeziumSpace(fam::DiskFamily{B,T,N}, params::NTuple{N,B}) where {B,T,N}
    TrapeziumSpace{typeof(fam), B, T, N}(
        fam, params, Vector{T}(), Vector{Vector{T}}(),
        Vector{Vector{T}}(), Vector{Vector{T}}(), Vector{SparseMatrixCSC{T}}(),
        Vector{SparseMatrixCSC{T}}(), Vector{SparseMatrixCSC{T}}(),
        Vector{SparseMatrixCSC{T}}())
end

# TODO
in(x::SVector{2}, D::Trapezium) = D.α ≤ x[1] ≤ D.β && D.γ*D.ρ(x[1]) ≤ x[2] ≤ D.δ*D.ρ(x[1])

spacescompatible(A::TrapeziumSpace, B::TrapeziumSpace) = (A.params == B.params)

domain(::TrapeziumSpace) = Trapezium()

# R should be Float64, B BigFloat
struct TrapeziumFamily{B,T,N,FAR,FAP,F,I} <: DiskFamily{B,T,N}
    spaces::Dict{NTuple{N,B}, TrapeziumSpace}
    α::T
    β::T
    γ::T
    δ::T
    R::FAR # OPFamily in (α,β)
    P::FAP # OPFamily in (γ,δ)
    ρ::F # Fun of 1 - ξ*X in (α,β)
    slope::T # ξ in the function ρ
    nparams::I
end

function (D::TrapeziumFamily{B,T,N,<:Any,<:Any,<:Any,<:Any})(params::NTuple{N,B}) where {B,T,N}
    haskey(D.spaces,params) && return D.spaces[params]
    D.spaces[params] = TrapeziumSpace(D, params)
end
(D::TrapeziumFamily{B,T,N,<:Any,<:Any,<:Any,<:Any})(params::Vararg{B,N}) where {B,T,N} =
    D(params)
(D::TrapeziumFamily{B,T,N,<:Any,<:Any,<:Any,<:Any})(params::Vararg{T,N}) where {B,T,N} =
    D(B.(params))

function TrapeziumFamily(::Type{B},::Type{T}, α::T, β::T, γ::T, δ::T, ξ::T) where {B,T}
    nparams = 4 # Default
    X = Fun(identity, B(α)..β)
    Y = Fun(identity, B(γ)..δ)
    ρ = 1 - ξ*X # TODO: Change to anon function? # NOTE ξ = 0.5 is default
    R = OrthogonalPolynomialFamily(T, β-X, X-α, ρ)
    P = OrthogonalPolynomialFamily(T, δ-Y, Y-γ)
    spaces = Dict{NTuple{nparams,B}, TrapeziumSpace}()
    TrapeziumFamily{B,T,nparams,typeof(R),typeof(P),typeof(ρ),Int}(spaces, α, β, γ, δ, R, P, ρ, nparams)
end
# Useful quick constructors
TrapeziumFamily(ξ::T) where T = TrapeziumFamily(BigFloat, T, 0.0, 1.0, 0.0, 1.0, ξ)
TrapeziumFamily() = TrapeziumFamily(0.5)

#===#
# Retrieve spaces methods
function getRspace(S::TrapeziumSpace, k::Int)
    (S.family.R)(S.params[1], S.params[2], S.params[3] + S.params[4] + 2k + 1)
end
function getRspace(S::TrapeziumSpace)
    (S.family.R)(S.params[1], S.params[2], S.params[3] + S.params[4])
end
getPspace(S::TrapeziumSpace) = (S.family.P)(S.params[end-1], S.params[end])

#===#
# Weight eval functions
weight(S::TrapeziumSpace{<:Any,<:Any,T,<:Any}, x, y) where T =
    T(getRspace(S).weight(x) * getPspace(S).weight(y/S.family.ρ(x)))
weight(S::TrapeziumSpace, z) = weight(S, z[1], z[2])


#===#
# points() and methods for pt evals and norm vals

# NOTE we output ≈n points (x,y)
function pointswithweights(S::TrapeziumSpace{<:Any, <:Any, T, <:Any}, n) where T
    # Return the weights and nodes to use for the integral of a function,
    # i.e. for the trapezium Ω:
    #   int_Ω W^{a,b}(x,y) f(x,y) dydx ≈ Σ_j wⱼ*f(xⱼ,yⱼ)
    N = Int(ceil(sqrt(n))) # ≈ n
    @show "begin pointswithweights()", n, N
    t, wt = pointswithweights(getPspace(S), N)
    s, ws = pointswithweights(getRspace(S, 0), N)
    pts = Vector{SArray{Tuple{2},T,1,2}}(undef, N^2)
    w = zeros(N^2) # weights
    ρs = S.family.ρ.(s)
    for i = 1:N
        for k = 1:N
            pts[i + (k - 1)N] = s[k], t[i] * ρs[k]
            w[i + (k - 1)N] = ws[k] * wt[i]
        end
    end
    @show "end pointswithweights()"
    pts, w
end
points(S::TrapeziumSpace, n) = pointswithweights(S, n)[1]

inner(S::TrapeziumSpace, fpts, gpts, w) = sum(fpts .* gpts .* w) / 2

function getopnorms(S::TrapeziumSpace{<:Any, <:Any, T, <:Any}, k) where T
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
function getopptseval(S::TrapeziumSpace, N, pts)
    resetopptseval(S)
    jj = [getopindex(n, 0) for n=0:N]
    for j in jj
        opevalatpts(S, j, pts)
    end
    S.opptseval
end
function opevalatpts(S::TrapeziumSpace{<:Any, <:Any, T, <:Any}, j, pts) where T
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
function resetopptseval(S::TrapeziumSpace)
    resize!(S.opptseval, 0)
    S
end

#===#
# transform and itransform

struct TrapeziumTransformPlan{T}
    w::Vector{T}
    pts::Vector{SArray{Tuple{2},T,1,2}}
    S::TrapeziumSpace{<:Any, <:Any, T, <:Any}
end

function TrapeziumTransformPlan(S::TrapeziumSpace{<:Any, <:Any, T, <:Any}, vals) where T
    m = Int(length(vals))
    pts, w = pointswithweights(S, m)
    TrapeziumTransformPlan{T}(w, pts, S)
end
plan_transform(S::TrapeziumSpace, vals) = TrapeziumTransformPlan(S, vals)
transform(S::TrapeziumSpace, vals) = plan_transform(S, vals) * vals

# Inputs: OP space, f(pts) for desired f
# Output: Coeffs of the function f for its expansion in the TrapeziumSpace OPs
function *(DSTP::TrapeziumTransformPlan, vals)
    @show "Begin DSTP mult"
    m2 = Int(length(vals))
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
    end
    resetopptseval(DSTP.S)
    j = 1
    for n = 0:N, k = 0:n
        ret[j] /= DSTP.S.opnorms[k+1]
        j += 1
    end
    @show "End DSTP mult"
    ret
end

# Inputs: OP space, coeffs of a function f for its expansion in the TrapeziumSpace OPs
# Output: vals = {f(x_j)} where x_j are are the points(S,n)
function itransform(S::TrapeziumSpace{<:Any, <:Any, T, <:Any}, cfs) where T
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

function recα(S::TrapeziumSpace{<:Any, <:Any, T, <:Any}, n, k, j) where T
    R = getRspace(S, k)
    if j == 1
        recγ(T, R, n-k+1)
    elseif j == 2
        recα(T, R, n-k+1)
    else
        error("Invalid entry to function")
    end
end
function recβ(S::TrapeziumSpace{<:Any, <:Any, T, <:Any}, n, k, j) where T
    # j ∈ {1,...,9}

    # We get the norms of the 2D OPs
    getopnorms(S, k+1)

    R1 = getRspace(S, k-1)
    R2 = getRspace(S, k)
    R3 = getRspace(S, k+1)
    P = getPspace(S)
    getopnorm(P)

    if (j - 1) % 3 == 0
        m = n - 1 + Int((j - 1) / 3)
        δ = recγ(T, P, k+1) * P.opnorm[1]
        pts, w = pointswithweights(R2, n-k+2)
        getopptseval(R2, n-k+1, pts)
        getopptseval(R1, m+1-k+1, pts)
        rinner = inner2(R2, opevalatpts(R2, n-k+1, pts), opevalatpts(R1, m+1-k+1, pts), w)
        ret = rinner * δ / S.opnorms[k]
    elseif (j - 2) % 3 == 0
        # We can calculate this explicitly, due to the linear nature of ρ for
        # the Trapezium case
        m = n - 1 + Int((j - 2) / 3)
        γ = recα(T, P, k+1)
        if m == n + 1
            rinner = - S.family.slope * recβ(T, R2, n+1)
        elseif m == n
            rinner = 1 - S.family.slope * recα(T, R2, n+1)
        else # m == n-1
            rinner = - S.family.slope * recγ(T, R2, n+1)
        end
        ret = rinner * γ
    else
        m = n - 1 + Int((j - 3) / 3)
        δ = recβ(T, P, k+1) * P.opnorm[1]
        pts, w = pointswithweights(R3, n-k+2)
        getopptseval(R2, n-k+1, pts)
        getopptseval(R3, m-1-k+1, pts)
        rinner = inner2(R3, opevalatpts(R2, n-k+1, pts), opevalatpts(R1, m-1-k+1, pts), w)
        ret = rinner * δ / S.opnorms[k+2]
    end
    ret
end

function getAs!(S::TrapeziumSpace, N, N₀)
    m = N₀
    if m == 0
        S.A[1] = [recα(S, 1, 0, 1) 0; recβ(S, 0, 0, 9) recβ(S, 0, 0, 9)]
        m += 1
    end
    for n = N+1:-1:m
        vx = [recα(S, n+1, k, 1) for k = 0:n]
        vy1 = [recβ(S, n, k, 7) for k = 1:n]
        vy2 = [recβ(S, n, k, 8) for k = 0:n]
        vy3 = [recβ(S, n, k, 9) for k = 0:n-1]
        S.A[n+1] = [Diagonal(v1) zeros(n+1);
                    Tridiagonal(vy1, vy2, vy3) [zeros(n); recβ(S, n, n, 9)]]
    end
end
function getDTs!(S::TrapeziumSpace, N, N₀)
    # Instead of calling recβ() in here, access S.A[n] entries
    # Assume getAs!() has already been called
    m = N₀
    if m == 0
        η1 = 1 / recβ(S, n, 0, 9)
        S.DT[1] = [(1 / recα(S, n+1, 0, 1)) 0;
                   (- η1 * recβ(S, n, 0, 8) / recα(S, n+1, 0, 1)) η1]
        m += 1
    end
    for n = N+1:-1:m
        vα = [1 / recα(S, n+1, k, 1) for k = 0:n]
        vβ = zeros(1, 2n + 2)
        vβ[1, end] = 1 / recβ(S, n, n, 9)
        vβ[1, end - 1] = - vβ[end] * recβ(S, n, n, 8) / recβ(S, n, n-1, 9)
        for j = 0:n-1
            ind = 2n - j
            vβ[1, ind] = - (vβ[1, ind + 2] * recβ(S, n, n-j, 7)
                         + vβ[1, ind + 1] * recβ(S, n, n-j-1, 8)) / recβ(S, n, n-j-1, 8)
        end
        vβ[1, 1] = - (vβ[1, n + 4] * recβ(S, n, 1, 7)
                   + vβ[1, n + 3] * recβ(S, n, 0, 8)) / recα(S, n+1, 0, 1)
        S.DT[n+1] = sparse([Diagonal(vα) zeros(n+1, n+1); vβ])
        # S.DT[n+1] = sparse(pinv(Array(S.A[n+1])))
    end
end
function getBs!(S::TrapeziumSpace, N, N₀)
    m = N₀
    if N₀ == 0
        S.B[1] = sparse([1, 2], [1, 1], [recα(S, 0, 0, 2), recβ(S, 0, 0, 5)])
        m += 1
    end
    for n = N+1:-1:m
        vx = [recα(S, n, k, 2) for k = 0:n]
        vy1 = [recβ(S, n, k, 4) for k = 1:n]
        vy2 = [recβ(S, n, k, 5) for k = 0:n]
        vy3 = [recβ(S, n, k, 6) for k = 0:n-1]
        S.B[n+1] = sparse([Diagonal(v1); Tridiagonal(vy1, vy2, vy3)])
    end
end
function getCs!(S::TrapeziumSpace, N, N₀)
    m = N₀
    if N₀ == 0
        # C_0 does not exist
        m += 1
    end
    if m == 1
        S.C[2] = sparse([1, 3, 4], [1, 1, 1],
                        [recα(S, 1, 0, 1), recβ(S, 1, 0, 2), recβ(S, 1, 1, 1)])
        m += 1
    end
    for n = N+1:-1:m
       vx = [recα(S, n, k, 1) for k = 0:n-1]
       vy1 = [recβ(S, n, k, 1) for k = 1:n]
       vy2 = [recβ(S, n, k, 2) for k = 0:n]
       vy3 = [recβ(S, n, k, 3) for k = 0:n-1]
       S.C[n+1] = sparse([Diagonal(v1);
                          zeros(1, n);
                          Tridiagonal(vy1, vy2, vy3);
                          [zeros(1, n-1) recβ(S, n, n, 1)]])
    end
end
function resizedata!(S::TrapeziumSpace, N)
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

function jacobix(S::TrapeziumSpace, N)
    # Jx^T = Jx
    resizedata!(S, N)
    rows = cols = 1:N+1
    l, u = 1, 1
    λ, μ = 0, 0
    J = BandedBlockBandedMatrix(0.0I, (rows, cols), (l, u), (λ, μ))
    n = 1
    view(J, Block(n, n)) .= S.B[n][1, :]
    view(J, Block(n, n+1)) .= S.A[n][1, :]'
    for n = 2:N
        view(J, Block(n, n-1)) .= S.C[n][1:Int(end/2), :]
        view(J, Block(n, n)) .= S.B[n][1:Int(end/2), :]
        view(J, Block(n, n+1)) .= S.A[n][1:Int(end/2), :]
    end
    view(J, Block(N+1, N)) .= S.C[N+1][1:Int(end/2), :]
    view(J, Block(N+1, N+1)) .= S.B[N+1][1:Int(end/2), :]
    J
end

function jacobiy(S::TrapeziumSpace, N)
    # Transposed operator, so acts directly on coeffs vec
    resizedata!(S, N)
    rows = cols = 1:N+1
    l, u = 1, 1
    λ, μ = 1, 1
    J = BandedBlockBandedMatrix(0.0I, (rows, cols), (l, u), (λ, μ))
    n = 1
    view(J, Block(n, n)) .= S.B[n][Int(end/2)+1:end, :]'
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

function clenshawG(::TrapeziumSpace, n, z)
    sp = sparse(I, n+1, n+1)
    [z[1] * sp; z[2] * sp]
end
function clenshaw(cfs::AbstractVector, S::TrapeziumSpace, z)
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
evaluate(cfs::AbstractVector, S::TrapeziumSpace, z) = clenshaw(cfs, S, z)

# Operator Clenshaw
function operatorclenshawG(S::TrapeziumSpace{<:Any, <:Any, T, <:Any}, n, Jx, Jy, zeromat) where T
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
function operatorclenshawvector(S::TrapeziumSpace{<:Any, <:Any, T, <:Any}, v, id) where T
    s = size(v)[1]
    B = Array{SparseMatrixCSC{T}}(undef, (1, s))
    for i = 1:s
        B[1,i] = id * v[i]
    end
    B
end
function operatorclenshawmatrixDT(S::TrapeziumSpace{<:Any, <:Any, T, <:Any}, A, id) where T
    B = Array{SparseMatrixCSC{T}}(undef, size(A))
    for ij = 1:length(A)
        B[ij] = id * A[ij]
    end
    B
end
function operatorclenshawmatrixBmG(S::TrapeziumSpace{<:Any, <:Any, T, <:Any}, A, id, Jx, Jy) where T
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
function operatorclenshaw(cfs, S::TrapeziumSpace, N)
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
operatorclenshaw(f::Fun, S::TrapeziumSpace) = operatorclenshaw(f.coefficients, S, getnk(ncoefficients(f))[1])
operatorclenshaw(f::Fun, S::TrapeziumSpace, N) = operatorclenshaw(f.coefficients, S, N)

#====#
# Methods to gather and evaluate the derivatives of the ops of space S at the
# transform pts given

resetxderivopptseval(S::TrapeziumSpace) = resize!(S.xderivopptseval, 0)
function clenshawGtildex(S::TrapeziumSpace, n, z)
    sp = sparse(I, n+1, n+1)
    [sp; 0.0 * sp]
end
function getxderivopptseval(S::TrapeziumSpace, N, pts)
    resetxderivopptseval(S)
    jj = [getopindex(n, 0) for n=0:N]
    for j in jj
        xderivopevalatpts(S, j, pts)
    end
    S.xderivopptseval
end
function xderivopevalatpts(S::TrapeziumSpace{<:Any, <:Any, T, <:Any}, j, pts) where T
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

resetyderivopptseval(S::TrapeziumSpace) = resize!(S.yderivopptseval, 0)
function clenshawGtildey(S::TrapeziumSpace, n, z)
    sp = sparse(I, n+1, n+1)
    [0.0 * sp; sp]
end
function getyderivopptseval(S::TrapeziumSpace, N, pts)
    resetyderivopptseval(S)
    jj = [getopindex(n, 0) for n=0:N]
    for j in jj
        yderivopevalatpts(S, j, pts)
    end
    S.yderivopptseval
end
function yderivopevalatpts(S::TrapeziumSpace{<:Any, <:Any, T, <:Any}, j, pts) where T
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

differentiatespacex(S::TrapeziumSpace) =
    (S.family)(S.params[1] + 1, S.params[2] + 1, S.params[3], S.params[4] + 1)
differentiatespacey(S::TrapeziumSpace) =
    (S.family)(S.params[1], S.params[2], S.params[3] + 1, S.params[4] + 1)
differentiateweightedspacex(S::TrapeziumSpace) =
    (S.family)(S.params[1] - 1, S.params[2] - 1, S.params[3], S.params[4] - 1)
differentiateweightedspacey(S::TrapeziumSpace) =
    (S.family)(S.params[1], S.params[2], S.params[3] - 1, S.params[4] - 1)

differentiatex(f::Fun, S::TrapeziumSpace) =
    Fun(differentiatespacex(S), differentiatex(S, f.coefficients))
differentiatey(f::Fun, S::TrapeziumSpace) =
    Fun(differentiatespacey(S), differentiatey(S, f.coefficients))
function differentiatex(S::TrapeziumSpace, cfs::AbstractVector)
    m̃ = length(cfs)
    N = -1 + Int(round(sqrt(1+2(m̃-1))))
    m = Int((N+1)*(N+2)/2)
    if m̃ < m
        resize!(cfs, m)
        cfs[m̃+1:end] .= 0.0
    end
    partialoperatorx(S, N) * cfs
end
function differentiatey(S::TrapeziumSpace, cfs::AbstractVector)
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

function getpartialoperatorxval(S::TrapeziumSpace{<:Any, <:Any, T, <:Any},
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
function partialoperatorx(S::TrapeziumSpace{<:Any, <:Any, T, <:Any}, N) where T
    # Takes the space H^{a,b,c,d} -> H^{a+1,b+1,c,d+1}
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

    bandn = 2
    bandk = 1
    A = BandedBlockBandedMatrix(
        Zeros{T}(sum(1:N), sum(1:(N+1))), (1:N, 1:N+1), (-1, bandn), (0, bandk))

    # Get pt evals for the R OPs
    for k = 0:N
        @show "dxoperator", N, k
        R = getRspace(S, k)
        getopptseval(R, N-k, ptsr)
        getderivopptseval(R, N-k, ptsr)
        Rx = getRspace(Sx, k)
        getopptseval(Rx, N-k+1, ptsr)
    end

    for n = 1:N, k = 0:n
        for m = max(0,n-bandn):(n-1)
            for j = (k-bandk):min(k,m)
                if j < 0
                    continue
                end
                val = getpartialoperatorxval(S, ptsp, wp, ptsr, rhoptsr,
                                                dxrhoptsr, wr, n, k, m, j)
                view(A, Block(m+1, n+1))[j+1, k+1] = val
            end
        end
    end
    A
end
function partialoperatory(S::TrapeziumSpace{<:Any, <:Any, T, <:Any}, N) where T
    # Takes the space H^{a,b,c,d} -> H^{a,b,c+1,d+1}
    A = BandedBlockBandedMatrix(
        Zeros{T}(sum(1:N),sum(1:(N+1))), (1:N, 1:N+1), (-1,1), (-1,1))
    Sy = differentiatespacey(S)
    P = getPspace(S)
    Py = getPspace(Sy)
    pts, w = pointswithweights(Py, N)
    getopptseval(P, N, pts)
    getderivopptseval(P, N, pts)
    getopptseval(Py, N, pts)
    for k = 1:N
        val = inner2(Py, derivopevalatpts(P, k+1, pts), opevalatpts(Py, k, pts), w)
        val /= getopnorm(P)
        for i = k:N
            view(A, Block(i, i+1))[k, k+1] = val
        end
    end
    A
end
function getweightedpartialoperatorxval(S::TrapeziumSpace{<:Any, <:Any, T, <:Any},
                ptsp, wp10, wp, ptsr, rhoptsr, dxrhoptsr, wr010, wr100, wr, n, k, m, j) where T
    # We should have already called getopptseval etc
    # ptsr, wr = pointswithweights(getRspace(Sx, 0), 2N+4)
    Sx = differentiateweightedspacex(S)
    P = getPspace(S)
    Px = getPspace(Sx)
    R = getRspace(S, k)
    Rx = getRspace(Sx, j)

    valp = inner2(Px, wp10 .* opevalatpts(P, k+1, ptsp), opevalatpts(Px, j+1, ptsp), wp)
    valr = inner2(Rx, opevalatpts(R, n-k+1, ptsr) .* wr100 .* wr010,
                    rhoptsr.^(k+j+1) .* dxrhoptsr .* opevalatpts(Rx, m-j+1, ptsr), wr)

    A = - (S.params[1]
            * valp
            * inner2(Rx, opevalatpts(R, n-k+1, ptsr) .* wr010 .* rhoptsr.^(k+j+2),
                        opevalatpts(Rx, m-j+1, ptsr), wr))
    A += (S.params[2]
            * valp
            * inner2(Rx, opevalatpts(R, n-k+1, ptsr) .* wr100 .* rhoptsr.^(k+j+2),
                        opevalatpts(Rx, m-j+1, ptsr), wr))
    A += (S.params[end]
            * valr
            * inner2(Px, opevalatpts(P, k+1, ptsp), opevalatpts(Px, j+1, ptsp), wp))
    B = valp * inner2(Rx, derivopevalatpts(R, n-k+1, ptsr) .* wr100 .* wr010,
                        rhoptsr.^(k+j+2) .* opevalatpts(Rx, m-j+1, ptsr), wr)
    B += k * valp * valr
    B -= valr * inner2(Px, ptsp .* derivopevalatpts(P, k+1, ptsp),
                        wp10 .* opevalatpts(Px, j+1, ptsp), wp)

    val = A + B
    val / Sx.opnorms[j+1]
end
function weightedpartialoperatorx(S::TrapeziumSpace{<:Any, <:Any, T, <:Any}, N) where T
    # Takes weighted space ∂/∂x(W^{a,b,c,d}) -> W^{a-1,b-1,c,d-1}
    bandn = 2
    bandk = 1
    W = BandedBlockBandedMatrix(Zeros{T}(sum(1:N+1+bandn), sum(1:(N+1))),
                                (1:N+1+bandn, 1:N+1), (bandn, -1), (bandk, 0))
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
    zeroparams = S.params .* 0
    # w_P^{(1)}.(pts)
    wp10 = T.(getPspace(S.family(zeroparams + (0,0,1,0))).weight.(ptsp))
    # w_R^{(1,0,0)}, w_R^{(0,1,0)}
    wr100 = T.(getRspace(S.family(zeroparams + (1,0,0,0))).weight.(ptsr))
    wr010 = T.(getRspace(S.family(zeroparams + (0,1,0,0))).weight.(ptsr))

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
        for m = n+1:n+bandn, j = k:min(m,k+bandk)
            val = getweightedpartialoperatorxval(S, ptsp, wp10, wp, ptsr,
                            rhoptsr, dxrhoptsr, wr010, wr100, wr, n, k, m, j)
            view(W, Block(m+1, n+1))[j+1, k+1] = val
        end
    end
    W
end
function weightedpartialoperatory(S::TrapeziumSpace{<:Any, <:Any, T, <:Any}, N) where T
    # Takes weighted space ∂/∂y(W^{a,b,c}) -> W^{a,b,c-1,d-1}
    W = BandedBlockBandedMatrix(
        Zeros{T}(sum(1:(N+2)),sum(1:(N+1))), (1:N+2, 1:N+1), (1,-1), (1,-1))
    Sy = differentiateweightedspacey(S)
    P = getPspace(S)
    Py = getPspace(Sy)
    ptsp, wp = pointswithweights(Py, N+2)
    getopptseval(P, N, ptsp)
    getopptseval(Py, N+1, ptsp)
    getderivopptseval(P, N, ptsp)
    getopnorms(Sy, N+1)
    zeroparams = S.params .* 0
    wp10 = T.(getPspace(S.family(zeroparams + (0,0,1,0))).weight.(ptsp))
    wp01 = T.(getPspace(S.family(zeroparams + (0,0,0,1))).weight.(ptsp))
    wp11 = T.(getPspace(S.family(zeroparams + (0,0,1,1))).weight.(ptsp))
    n, m = N, N+1
    for k = 0:N
        j = k + 1
        val = inner(P, (wp11 .* derivopevalatpts(P, k+1, ptsp)
                            + (S.params[end-1] * wp10 - S.params[end] * wp01)
                                .* opevalatpts(P, k+1, ptsp)),
                       opevalatpts(Py, j+1, ptsp), wp)
        val *= getopnorm(getRspace(S, k))
        for i = k:N
            view(W, Block(i+2, i+1))[k+2, k+1] = val / Sy.opnorms[j+1]
        end
    end
    W
end

#====#
# Parameter tranformation operators

function transformparamsoperator(S::TrapeziumSpace{<:Any, <:Any, T, <:Any},
            St::TrapeziumSpace{<:Any, <:Any, T, <:Any}, N; weighted=false) where T
    # Cases we can handle:
    if weighted == false

        ã, b̃, c̃, d̃ = Int.(St.params .- S.params)
        if !((ã == 0 || ã == 1) && (b̃ == 0 || b̃ == 1)
                && (c̃ == 0 || c̃ == 1) && (d̃ == 0 || d̃ == 1))
            error("Invalid TrapeziumSpace")
        end
        bandn = ã + b̃ + c̃ + d̃
        bandk = d̃ + c̃

        C = BandedBlockBandedMatrix(Zeros{T}(sum(1:(N+1)), sum(1:(N+1))),
                                    (1:N+1, 1:N+1), (0, bandn), (0, bandk))

        P = getPspace(S)
        Pt = getPspace(St)
        if bandk > 0
            ptsp, wp = pointswithweights(Pt, N+2)
            getopptseval(P, N, ptsp)
            getopptseval(Pt, N, ptsp)
        end
        ptsr, wr = pointswithweights(getRspace(St, 0), N+1)
        # Get pt evals for R OPs
        for k = 0:N
            R = getRspace(S, k)
            Rt = getRspace(St, k)
            getopptseval(R, N-k, ptsr)
            getopptseval(Rt, N-k, ptsr)
        end
        rhoptsr = S.family.ρ.(ptsr)
        getopnorms(St, N)

        for n = 0:N, k = 0:n
            for m = max(0, n - bandn):n, j = max(0, k - bandk):min(k, m)
                # The P inner product
                if bandk == 0
                    valp = getopnorm(P)
                else
                    valp = inner2(Pt, opevalatpts(P, k+1, ptsp),
                                    opevalatpts(Pt, j+1, ptsp), wp)
                end
                # The R inner product
                valr = inner2(Rt, opevalatpts(R, n-k+1, ptsr) * rhoptsr.^(k+j),
                                opevalatpts(Rt, m-j+1, ptsr), wr)
                view(C, Block(m+1, n+1))[j+1, k+1] = valp * valr / St.opnorms[j+1]
            end
        end
    elseif weighted == true
        ã, b̃, c̃, d̃ = Int.(S.params .- St.params)
        if !((ã == 0 || ã == 1) && (b̃ == 0 || b̃ == 1)
                && (c̃ == 0 || c̃ == 1) && (d̃ == 0 || d̃ == 1))
            error("Invalid TrapeziumSpace")
        end
        bandn = ã + b̃ + c̃ + d̃
        bandk = d̃ + c̃

        C = BandedBlockBandedMatrix(Zeros{T}(sum(1:(N+1+bandn)), sum(1:(N+1))),
                                    (1:N+1+bandn, 1:N+1), (bandn, 0), (bandk, 0))

        P = getPspace(S)
        Pt = getPspace(St)
        if bandk > 0
            ptsp, wp = pointswithweights(P, N+2)
            getopptseval(P, N, ptsp)
            getopptseval(Pt, N, ptsp)
        end
        ptsr, wr = pointswithweights(getRspace(S, 0), N+1)
        # Get pt evals for R OPs
        for k = 0:N
            R = getRspace(S, k)
            Rt = getRspace(St, k)
            getopptseval(R, N-k, ptsr)
            getopptseval(Rt, N-k, ptsr)
        end
        rhoptsr = S.family.ρ.(ptsr)
        getopnorms(St, N)

        for n = 0:N, k = 0:n
            for m = n:(n + bandn), j = k:min(k + bandk, m)
                # The P inner product
                if bandk == 0
                    valp = getopnorm(P)
                else
                    valp = inner2(Pt, opevalatpts(P, k+1, ptsp),
                                    opevalatpts(Pt, j+1, ptsp), wp)
                end
                # The R inner product
                valr = inner2(Rt, opevalatpts(R, n-k+1, ptsr) * rhoptsr.^(k+j),
                                opevalatpts(Rt, m-j+1, ptsr), wr)
                view(C, Block(m+1, n+1))[j+1, k+1] = valp * valr / St.opnorms[j+1]
            end
        end
    end
end

#====#
# Laplacian and biharmonic operator matrices

function laplaceoperator(S::TrapeziumSpace{<:Any, <:Any, T, <:Any},
            St::TrapeziumSpace{<:Any, <:Any, T, <:Any}, N;
            weighted=false, square=true) where T
    # Outputs the sum(1:N+1) × sum(1:N+1) matrix operator if square=true
    D = S.family
    if (weighted == true && S.params == ntuple(x->1, D.nparams)
            && St.params == ntuple(x->1, D.nparams))
        A = transformparamsoperator(D(S.params - (0,0,1,0)), S, N+4)
        @show "laplaceoperator", "1 of 8 done"
        B = partialoperatorx(differentiateweightedspacex(S), N+4)
        @show "laplaceoperator", "2 of 8 done"
        C = transformparamsoperator(D(S.params - (1,1,0,1)), D(S.params .- 1)),
                                    N+3, weighted=true)
        @show "laplaceoperator", "3 of 8 done"
        E = weightedpartialoperatorx(S, N)
        @show "laplaceoperator", "4 of 8 done"
        F = transformparamsoperator(differentiatespacey(D(S.params .- 1)), S, N+4)
        @show "laplaceoperator", "5 of 8 done"
        G = partialoperatory(D(S.params .- 1), N+4)
        @show "laplaceoperator", "6 of 8 done"
        H = transformparamsoperator(differentiateweightedspacey(S), D(S.params .- 1),
                                    N+2, weighted=true)
        @show "laplaceoperator", "7 of 8 done"
        K = weightedpartialoperatory(S, N)
        @show "laplaceoperator", "8 of 8 done"
        L = A * B * C * E + F * G * H * K
        if square
            m = sum(1:(N+1))
            Δ = BandedBlockBandedMatrix(L[1:m, 1:m], (1:N+1, 1:N+1), (L.l,L.u), (L.λ,L.μ))
        else
            L
        end
    else
        # TODO Other cases
        error("Invalid TrapeziumSpace for Laplacian operator")
    end
end

function biharmonicoperator(S::TrapeziumSpace{<:Any, <:Any, T, <:Any}, N; square=true) where T
    # TODO
    D = S.family
    if S.params == ntuple(x->2, D.nparams)
    else
        error("Invalid TrapeziumSpace for Laplacian operator")
    end
end

# end # module
