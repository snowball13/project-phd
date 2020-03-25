# Spherical/Polar Caps

#=
NOTE

    OPs: Q^{(a,b)}_{n,k,i}(x,y,z) := R^{(a,b+k+0.5)}_{n-k}(z) * ρ(z)^k * Y_{k,i}(θ)

for n ∈ ℕ₀, k = 0,...,n, i ∈ {0,1}

where Y_{k,0}(θ) := cos(kθ), Y_{k,1}(θ) := sin(kθ) for k > 0
      Y_{0,0}(θ) := 1 (for k = 0, and there is no Y_{0,1})

and x = cosθ sinϕ, y = sinθ sinϕ, z = cosϕ; ρ(z) = sqrt(1-z^2) = sinϕ

The ordering of the OP vector is not done by polynomial degree, but by Fourier
mode (not by n but by k)

i.e. for a given max degree N:

    ℚ^{(a,b)}_N := [ℚ^{(a,b)}_{N,0};...;ℚ^{(a,b)}_{N,N}]
    ℚ^{(a,b)}_{N,k} := [Q^{(a,b)}_{k,k,0};Q^{(a,b)}_{k,k,1}...;Q^{(a,b)}_{N,k,0};Q^{(a,b)}_{N,k,1}] ∈ ℝ^{2(N-k+1)}
                                                            for k = 1,...,N
    ℚ^{(a,b)}_{N,0} := [Q^{(a,b)}_{0,0,0};Q^{(a,b)}_{1,0,0};Q^{(a,b)}_{1,0,1}...;Q^{(a,b)}_{N,0,0};Q^{(a,b)}_{N,0,1}] ∈ ℝ^{2N+1}
                                                            for k = 0

=#

export SphericalCapFamily, SphericalCapSpace

# T should be Float64, B should be BigFloat

# TODO Sort out checkpoints, in, and domain - should
abstract type SphericalFamily{B,T,N} end
struct SphericalCap{B,T} <: Domain{SVector{2,T}} end
SphericalCap() = SphericalCap{BigFloat, Float64}()
function checkpoints(::SphericalCap)
    # Return 2 3D points that will lie on the domain TODO
    y, z = -0.3, 0.8; x = sqrt(1 - z^2 - y^2); p1 = SVector(x, y, z)
    y, z = 0.4, 0.8; x = sqrt(1 - z^2 - y^2); p2 = SVector(x, y, z)
    [p1, p2]
end
in(x::SVector{3}, D::SphericalCap) = D.α ≤ x[3] ≤ D.β && sqrt(x[1]^2 + x[2]^2) == D.ρ(x[3])

struct SphericalCapSpace{DF, B, T, N} <: Space{SphericalCap{B,T}, T}
    family::DF # Pointer back to the family
    params::NTuple{N,B} # Parameters
    opnorms::Vector{B} # squared norms
    opptseval::Vector{Vector{B}} # Store the ops evaluated at the transform pts
    A::Vector{SparseMatrixCSC{B}}
    B::Vector{SparseMatrixCSC{B}}
    C::Vector{SparseMatrixCSC{B}}
    DT::Vector{SparseMatrixCSC{B}}
end

function SphericalCapSpace(fam::SphericalFamily{B,T,N}, params::NTuple{N,B}) where {B,T,N}
    SphericalCapSpace{typeof(fam), B, T, N}(
        fam, params, Vector{B}(), Vector{Vector{B}}(),
        Vector{SparseMatrixCSC{B}}(), Vector{SparseMatrixCSC{B}}(),
        Vector{SparseMatrixCSC{B}}(), Vector{SparseMatrixCSC{B}}())
end

spacescompatible(A::SphericalCapSpace, B::SphericalCapSpace) = (A.params == B.params)

domain(::SphericalCapSpace{<:Any, B, T, <:Any}) where {B,T} = SphericalCap{B,T}()

struct SphericalCapFamily{B,T,N,FA,F,I} <: SphericalFamily{B,T,N}
    spaces::Dict{NTuple{N,B}, SphericalCapSpace}
    α::T
    β::T
    R::FA # 1D OP family for the semiclassical ops
    ρ::F # Fun of sqrt(1-X^2) in (α,β)
    nparams::I
end

function (D::SphericalCapFamily{B,T,N,<:Any,<:Any})(params::NTuple{N,B}) where {B,T,N}
    haskey(D.spaces,params) && return D.spaces[params]
    D.spaces[params] = SphericalCapSpace(D, params)
end
(D::SphericalCapFamily{B,T,N,<:Any,<:Any})(params::Vararg{T,N}) where {B,T,N} =
    D(B.(params))

function SphericalCapFamily(::Type{B}, ::Type{T}, α::T) where {B,T}
    β = 1.0
    X = Fun(identity, B(α)..β)
    ρ = sqrt(1 - X^2)
    ρ2 = 1 - X^2 # NOTE we use ρ^2 here to help computationally
    # NOTE also that we only have a 2-param family for the spherical/polar cap
    nparams = 2
    R = OrthogonalPolynomialFamily(T, X-α, ρ2)
    spaces = Dict{NTuple{nparams,B}, SphericalCapSpace}()
    SphericalCapFamily{B,T,nparams,typeof(R),typeof(ρ),Int}(spaces, α, β, R, ρ, nparams)
end
# Useful quick constructors
SphericalCapFamily(α::T) where T = SphericalCapFamily(BigFloat, T, α)
SphericalCapFamily() = SphericalCapFamily(BigFloat, Float64, 0.0) # Hemisphere


#=======#


#===#
# Methods to handle the R OPs (the 1D OP family that is a part of the
# SphericalFamily)

# Retrieve 1D OP spaces methods
getRspace(S::SphericalCapSpace, k::Int) =
    (S.family.R)(S.params[1], (2S.params[2] + 2k + 1)/2)
getRspace(S::SphericalCapSpace) =
    (S.family.R)(S.params[1], S.params[2])

# Method to calculate the rec coeffs for the 1D R OPs (the non classical ones)
# using a recursive algorithm invloving use of the Christoffel-Darboux formula
function getreccoeffsR!(S::SphericalCapSpace{<:Any, B, T, <:Any}, N; maxk=-1) where {B,T}
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
        R00 = getRspace(S, k - c) # R00 = R(S.a, B(0.5)+k)
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
        R11 = getRspace(S, k - c + 1) # R11 = R(S.a, B(0.5)+k+1)
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
function resizedataonedimops!(S::SphericalCapSpace, N)
    # N = max degree desired
    length(getRspace(S, N).a) > 0 && return S

    # Use our algorithm to generate the rec coeffs recursively for R's
    # We assume that we have R(a,b,0.5).ops up to the previous required degree
    # (to build on) and thus subsequent OP rec coeffs for R(a,b,k+0.5) for
    # k = 1:N
    @show "resizedata! (one dim ops) for SCSpace", Float64.(S.params), N
    getreccoeffsR!(S, N)
    S
end

# Method to get the normalising constants for the 1D R OPs
function getopnorms(S::SphericalCapSpace{<:Any, B, T, <:Any}, k) where {B,T}
    m = length(S.opnorms)
    if k + 1 > m
        resize!(S.opnorms, k+1)
        # We get the weights for the R integral
        ptsr, wr = pointswithweights(B, getRspace(S, 0), k+1)
        rhoptsr = S.family.ρ.(ptsr)
        for j = m:k
            R = getRspace(S, j)
            if !isnormset(R)
                setopnorm(R, sum(rhoptsr.^(2j) .* wr))
            end
            S.opnorms[j+1] = R.opnorm[1] # * π NOTE we do not add the pi
                                         # factor, as this cancels out whenever
                                         # we need to use the S norms directly
        end
    end
    S
end


#===#
# Recurrence coeffs retrieval methods

# α are the mult by x coeffs, β the mult by y coeffs, γ the mult by z coeffs
function recα(::Type{T}, S::SphericalCapSpace, n, k, j) where T
    getopnorms(S, k+1)
    R1 = getRspace(S, k-1)
    R2 = getRspace(S, k)
    R3 = getRspace(S, k+1)

    # TODO check that the number of pts we are getting here is what is required!
    if isodd(j)
        pts, w = pointswithweights(T, R2, Int(ceil(n-k+1.5)))
    else
        pts, w = pointswithweights(T, R3, Int(ceil(n-k+1.5)))
    end
    getopptseval(R2, n-k+1, pts)

    if j == 1
        getopptseval(R1, n-k+1, pts)
        ret = (inner2(R2, opevalatpts(R2, n-k+1, pts), opevalatpts(R1, n-k+1, pts), w)
                / getopnorm(R1))
    elseif j == 2
        getopptseval(R3, n-k-1, pts)
        ret = (inner2(R3, opevalatpts(R2, n-k+1, pts), opevalatpts(R3, n-k-1, pts), w)
                / getopnorm(R3))
    elseif j == 3
        getopptseval(R1, n-k+2, pts)
        ret = (inner2(R2, opevalatpts(R2, n-k+1, pts), opevalatpts(R1, n-k+2, pts), w)
                / getopnorm(R1))
    elseif j == 4
        getopptseval(R3, n-k, pts)
        ret = (inner2(R3, opevalatpts(R2, n-k+1, pts), opevalatpts(R3, n-k, pts), w)
                / getopnorm(R3))
    elseif j == 5
        getopptseval(R1, n-k+3, pts)
        ret = (inner2(R2, opevalatpts(R2, n-k+1, pts), opevalatpts(R1, n-k+3, pts), w)
                / getopnorm(R1))
    elseif j == 6
        getopptseval(R3, n-k+1, pts)
        ret = (inner2(R3, opevalatpts(R2, n-k+1, pts), opevalatpts(R3, n-k+1, pts), w)
                / getopnorm(R3))
    else
        error("Invalid entry to function")
    end

    if k == 0 || (k == 1 && isodd(j))
        getdegzeropteval(T, S) * T(ret)
    else
        T(ret) / 2
    end
end
function recβ(::Type{T}, S::SphericalCapSpace, n, k, i, j) where T
    # NOTE the extra input i repesents the index in Q_{n,k,i}
    ret = recα(T, S, n, k, j)
    if (i == 0 && isodd(j)) || (i == 1 && iseven(j))
        -ret
    else
        ret
    end
end
function recγ(::Type{T}, S::SphericalCapSpace, n, k, j) where T
    R = getRspace(S, k)
    if j == 1
        recγ(T, R, n-k+1)
    elseif j == 2
        recα(T, R, n-k+1)
    elseif j == 3
        recβ(T, R, n-k+1) # = recγ(T, S, n+1, k, 1)
    else
        error("Invalid entry to function")
    end
end

#===#
# Indexing retrieval methods

function getopindex(S::SphericalCapSpace, n, k, i; bydegree=true, N=0)
    if k > n
        error("Invalid k input to getopindex")
    elseif i > 1 || i < 0
        error("Invalid i input to getopindex")
    elseif k == 0 && i == 1
        error("Invalid inputs to getopindex - i must be zero if k is zero")
    end
    if bydegree
        if k == 0
            ret = n^2 + 1
        else
            ret = n^2 + 2k + i
        end
    else # by Fourier mode k
        # N must be set
        if k == 0
            ret = n + 1
        else
            ret = N + 1 + 2 * sum([N-j+1 for j=1:k-1]) + 2 * (n - k) + i + 1
        end
    end
    ret
end
function getnki(S::SphericalCapSpace, ind; bydegree=true)
    if bydegree
        n = 0
        while true
            if (n+1)^2 ≥ ind
                break
            end
            n += 1
        end
        r = ind - n^2
        if r == 1
            n, 0, 0
        elseif iseven(r)
            n, Int(r/2), 0
        else
            n, Int((r-1)/2), 1
        end
    else
        error("Can only do this for bydegree=true")
    end
end
function getNforcoeffsvec(S::SphericalCapSpace, cfs::AbstractArray)
    # m = length(cfs)
    # N = (-3 + sqrt(9 - 4 * (2 - 2m))) / 2
    # if !isinteger(N)
    #     error("length of coeffs vec does not match an N value")
    # end
    # Int(N)
    # TODO
end


#===#
# points()

# NOTE we output ≈n points (x,y,z), plus the ≈n points corresponding to (-x,-y,z)
function pointswithweights(S::SphericalCapSpace{<:Any, B, T, <:Any}, n;
                            nofactor=false) where {B,T}
    # Return the weights and nodes to use for the even part of a function,
    # i.e. for the spherical cap Ω:
    #   int_Ω w_R^{a,2b}(x) f(x,y,z) dσ(x,y)dz ≈ 0.5 * Σ_j wⱼ*(f(xⱼ,yⱼ,zⱼ) + f(-xⱼ,-yⱼ,zⱼ))
    # NOTE: the odd part of the quad rule will equal 0 for polynomials,
    #       so can be ignored.

    # When nofactor is true, then the weights are not multiplyed by 2π

    N = 2 * Int(ceil(sqrt(n))) - 1 # degree we approximate up to with M quadrature pts
    M1 = Int(ceil((N + 1) / 2))
    M2 = M1 # TODO what should M2 be????
    M = M1 * M2 # ≈ n
    @show "begin pointswithweights()", n, N, M

    # Get the 1D quadrature pts and weights
    # Need to maunally call the method to get R coeffs here
    m = isodd(M1) ? Int((M1 + 1) / 2) : Int((M1 + 2) / 2); m -= Int(S.params[end])
    getreccoeffsR!(S, m; maxk=0)
    t, wt = pointswithweights(B, getRspace(S, 0), M1) # Quad for w_R^{(a,2b+1)}
    s = [(cospi(B(2 * (it - 1)) / M2), sinpi(B(2 * (it - 1)) / M2)) for it=1:M2]
    ws = ones(B, M2) / M2  # Quad for circumference of unit circle
    if !nofactor
        ws *= 2 * B(π)
    end

    # output both pts and the weights that make up the 3D quad rule
    pts = Vector{SArray{Tuple{3},B,1,3}}(undef, 2M) # stores both (x,y,z) and (-x,-y,z)
    w = zeros(B, M)
    for j1 = 1:M1
        z = t[j1]
        rhoz = S.family.ρ(z)
        for j2 = 1:M2
            x, y = rhoz * s[j2][1], rhoz * s[j2][2]
            pts[j2 + (j1 - 1)M1] = x, y, z
            pts[M + j2 + (j1 - 1)M1] = -x, -y, z
            w[j2 + (j1 - 1)M1] = wt[j1] * ws[j2]
        end
    end

    @show "end pointswithweights()"
    pts, w
end
points(S::SphericalCapSpace, n) = pointswithweights(S, n)[1]

#===#
# Point evaluation (opevalatpts)
# NOTE for now, we store these by degree order n.

# Returns the constant that is Q^{a,b}_{0,0,0} ( = Y_{0,0}, so that the Y_ki's
# are normalised)
function getdegzeropteval(::Type{T}, S::SphericalCapSpace) where T
    sqrt(T(2)) / 2
end
getdegzeropteval(S::SphericalCapSpace{<:Any, T, <:Any, <:Any}) where T =
    getdegzeropteval(T, S)
# Methods to gather and evaluate the ops of space S at the transform pts given
function getopptseval(S::SphericalCapSpace, N, pts)
    resetopptseval(S)
    jj = [getopindex(S, n, 0, 0) for n = 0:N]
    for j in jj
        opevalatpts(S, j, pts)
    end
    S.opptseval
end
function resetopptseval(S::SphericalCapSpace)
    resize!(S.opptseval, 0)
    S
end
function opevalatpts(S::SphericalCapSpace{<:Any, B, T, <:Any}, j, pts) where {B,T}
    # NOTE This function should be used only from getopptseval().
    #      The idea here is that we have all OPs up to degree N already
    #      evaluated at pts, and we simply iterate once to calculate the pts
    #      evals for the deg N+1 OPs.
    # The input j refers to the index of a deg N+1 OP, that can be used to
    # return it.

    len = length(S.opptseval)
    if len ≥ j
        return S.opptseval[j]
    end

    # We iterate up from the last obtained pts eval
    N = len == 0 ? -1 : getnki(S, len)[1]
    n = getnki(S, j)[1]
    if  N != n - 1 || (len == 0 && j > 1)
        error("Invalid index")
    end

    if n == 0
        resize!(S.opptseval, 1)
        S.opptseval[1] = Vector{B}(undef, length(pts))
        S.opptseval[1][:] .= getdegzeropteval(B, S)
    else
        jj = getopindex(S, n, 0, 0)
        resizedata!(S, n)
        resize!(S.opptseval, getopindex(S, n, n, 1))
        for k = 0:2n
            S.opptseval[jj+k] = Vector{B}(undef, length(pts))
        end
        if n == 1
            nm1 = getopindex(S, n-1, 0, 0)
            for r = 1:length(pts)
                P1 = [opevalatpts(S, nm1+it, pts)[r] for it = 0:2(n-1)]
                P = - S.DT[n] * (S.B[n] - clenshawG(S, n-1, pts[r])) * P1
                for k = 0:2n
                    S.opptseval[jj+k][r] = P[k+1]
                end
            end
        else
            nm1 = getopindex(S, n-1, 0, 0)
            nm2 = getopindex(S, n-2, 0, 0)
            for r = 1:length(pts)
                P1 = [opevalatpts(S, nm1+it, pts)[r] for it = 0:2(n-1)]
                P2 = [opevalatpts(S, nm2+it, pts)[r] for it = 0:2(n-2)]
                P = (- S.DT[n] * (S.B[n] - clenshawG(S, n-1, pts[r])) * P1
                     - S.DT[n] * S.C[n] * P2)
                for k = 0:2n
                    S.opptseval[jj+k][r] = P[k+1]
                end
            end
        end
    end
    S.opptseval[j]
end
# These funcs returns the S.opptseval for the OP n,k
# We assume that we have called getopptseval(S, N, pts)
function getptsevalforop(S::SphericalCapSpace, ind)
    if length(S.opptseval) < ind
        error("Invalid OP requested in getptsevalforop - getopptseval(S,N,pts) may not have been correctly called")
    else
        S.opptseval[ind]
    end
end
getptsevalforop(S::SphericalCapSpace, n, k, i) =
    getptsevalforop(S, getopindex(S, n, k, i))



#===#
# transform and itransform

struct SphericalCapTransformPlan{T}
    Vp::Array{T}
    Vm::Array{T}
end

function SphericalCapTransformPlan(S::SphericalCapSpace{<:Any, B, T, <:Any}, vals) where {B,T}
    @show "Begin SphericalCapTransformPlan"

    # NOTE N here is the degree of the function f that we are finding the
    #      coefficients for. We should have m2 = (N+1)^2 vals (pts).
    #      m1 is the number of OPs we require, that is all OPs up to and
    #      including deg N, i.e. length of ℚ^{(a,b)}_N, which is (N+1)^2.

    m2 = Int(length(vals) / 2)
    N = m2 < 2 ? Int(sqrt(m2)) - 1 : Int(sqrt(m2)) - 2 # TODO sqrt(m2)-1 doesnt work here when it gives N as odd...
    m1 = (N+1)^2
    @show N, m1, m2

    # nofactor=true means we dont have the 2pi factor in the weights
    pts, w = pointswithweights(S, m2; nofactor=true)
    getopnorms(S, N)

    # calculate the Vandermonde matrix
    Vp = zeros(B, m1, m2); Vm = zeros(B, m1, m2)
    p = Vector{SArray{Tuple{3},B,1,3}}(undef, 2)
    for j = 1:m2
        p[1] = pts[j]; p[2] = pts[j+m2]
        getopptseval(S, N, p)
        indv = 1
        for k = 0:N
            # indv = getopindex(S, k, k, 0; bydegree=false, N=N)
            for n = k:N
                inde = getopindex(S, n, k, 0)
                for i = 0:min(1, k) # This catches the k == 0 case
                    # NOTE use indexing of opptseval to order rows of V by Fourier mode k (not degree)
                    Vp[indv, j] = getptsevalforop(S, inde+i)[1] * w[j] / S.opnorms[k+1]
                    Vm[indv, j] = getptsevalforop(S, inde+i)[2] * w[j] / S.opnorms[k+1]
                    indv += 1
                end
            end
        end
        resetopptseval(S)
    end

    SCTP = SphericalCapTransformPlan{B}(Vp, Vm)
    @show "End SphericalCapTransformPlan"
    SCTP
end
plan_transform(S::SphericalCapSpace, vals) = SphericalCapTransformPlan(S, vals)
transform(S::SphericalCapSpace, vals) = plan_transform(S, vals) * vals

# Inputs: OP space, f(pts) for desired f
# Output: Coeffs of the func f for its expansion in the SphericalCapSpace OPs
# Coeffs output are ordered by Fourier mode k
function *(SCTP::SphericalCapTransformPlan{B}, vals::AbstractVector{T}) where {B,T}
    @show "Begin SCTP mult"
    m = Int(length(vals) / 2)
    ret = SCTP.Vp * vals[1:m] + SCTP.Vm * vals[m+1:end]
    @show "End SCTP mult"
    ret
end

# Inputs: OP space, coeffs of a function f for its expansion in the SphericalCapSpace OPs
# Output: vals = {f(x_j)} where x_j are are the points(S,n)
function itransform(S::SphericalCapSpace, cfs::AbstractVector{T}) where T
    @show "begin itransform"
    ncfs = length(cfs)
    N = Int(sqrt(ncfs)) - 1
    pts = points(S, (N+2)^2)
    npts = length(pts)
    ret = zeros(T, npts)
    m = Int(npts / 2)
    for j = 1:m
        getopptseval(S, N, (pts[j], pts[j + m]))
        indc = 1
        for k = 0:N
            # indc = getopindex(S, k, k, 0; bydegree=false, N=N)
            for n = k:N
                inde = getopindex(S, n, k, 0)
                for i = 0:min(1, k) # This catches the k == 0 case
                    # NOTE use indexing of opptseval to order rows of V by Fourier mode k (not degree)
                    ret[j] += getptsevalforop(S, inde+i)[1] * cfs[indc]
                    ret[j + m] += getptsevalforop(S, inde+i)[2] * cfs[indc]
                    indc += 1
                end
            end
        end
        resetopptseval(S)
    end
    @show "end itransform"
    ret
end


#===#
# Function evaluation (clenshaw)

#=
NOTE
The Clenshaw matrices are stored by degree (and not by Fourier mode k).
This makes the Clenshaw algorithm much easier.
We will just need to reorder/take into account the fact that the coeffs are
stored by Fourier mode (and not degree) in the calculations.

OR since constructing/storing these takes a looong time, we do the clenshaw alg
when needed *not* using the clenshaw matrices.
=#

function getBs!(S::SphericalCapSpace{<:Any, T, <:Any, <:Any}, N, N₀) where T
    m = N₀
    resize!(S.B, N + 1)
    if m == 0
        S.B[1] = sparse([0; 0; recγ(T, S, 0, 0, 2)])
        m += 1
    end
    if m == 1
        n = 1
        S.B[n+1] = sparse(zeros(T, 3*(2n+1), 2n+1))

        k = 0; i = 0
        S.B[n+1][1, 2] = recα(T, S, n, k, 4)
        S.B[n+1][2n+1+1, 3] = recβ(T, S, n, k, i, 4)
        S.B[n+1][2*(2n+1)+1, 1] = recγ(T, S, n, k, 2)

        k = 1
        c2 = recγ(T, S, n, k, 2)
        i = 0
        S.B[n+1][2k+i, 1] = recα(T, S, n, k, 3)
        S.B[n+1][2*(2n+1)+2k+i, 2k+i] = c2
        i = 1
        S.B[n+1][2n+1+2k+i, 1] = recβ(T, S, n, k, i, 3)
        S.B[n+1][2*(2n+1)+2k+i, 2k+i] = c2
        m += 1
    end
    for n = N:-1:m
        S.B[n+1] = sparse(zeros(T, 3*(2n+1), 2n+1))

        k = 0; i = 0
        S.B[n+1][1, 2] = recα(T, S, n, k, 4)
        S.B[n+1][2n+1+1, 3] = recβ(T, S, n, k, i, 4)
        S.B[n+1][2*(2n+1)+1, 1] = recγ(T, S, n, k, 2)

        k = 1
        a3, a4 = recα(T, S, n, k, 3), recα(T, S, n, k, 4)
        c2 = recγ(T, S, n, k, 2)
        i = 0
        S.B[n+1][2k+i, 1] = a3
        S.B[n+1][2k+i, 2(k+1)+i] = a4
        S.B[n+1][2n+1+2k+i, 2(k+1)+i+1] = recβ(T, S, n, k, i, 4)
        S.B[n+1][2*(2n+1)+2k+i, 2k+i] = c2
        i = 1
        S.B[n+1][2k+i, 2(k+1)+i] = a4
        S.B[n+1][2n+1+2k+i, 1] = recβ(T, S, n, k, i, 3)
        S.B[n+1][2n+1+2k+i, 2(k+1)+i-1] = recβ(T, S, n, k, i, 4)
        S.B[n+1][2*(2n+1)+2k+i, 2k+i] = c2

        for k = 2:n-1
            a3, a4 = recα(T, S, n, k, 3), recα(T, S, n, k, 4)
            c2 = recγ(T, S, n, k, 2)
            i = 0
            S.B[n+1][2k+i, 2(k-1)+i] = a3
            S.B[n+1][2k+i, 2(k+1)+i] = a4
            S.B[n+1][2n+1+2k+i, 2(k-1)+i+1] = recβ(T, S, n, k, i, 3)
            S.B[n+1][2n+1+2k+i, 2(k+1)+i+1] = recβ(T, S, n, k, i, 4)
            S.B[n+1][2*(2n+1)+2k+i, 2k+i] = c2
            i = 1
            S.B[n+1][2k+i, 2(k-1)+i] = a3
            S.B[n+1][2k+i, 2(k+1)+i] = a4
            S.B[n+1][2n+1+2k+i, 2(k-1)+i-1] = recβ(T, S, n, k, i, 3)
            S.B[n+1][2n+1+2k+i, 2(k+1)+i-1] = recβ(T, S, n, k, i, 4)
            S.B[n+1][2*(2n+1)+2k+i, 2k+i] = c2
        end

        k = n
        a3 = recα(T, S, n, k, 3)
        c2 = recγ(T, S, n, k, 2)
        i = 0
        S.B[n+1][2k+i, 2(k-1)+i] = a3
        S.B[n+1][2n+1+2k+i, 2(k-1)+i+1] = recβ(T, S, n, k, i, 3)
        S.B[n+1][2*(2n+1)+2k+i, 2k+i] = c2
        i = 1
        S.B[n+1][2k+i, 2(k-1)+i] = a3
        S.B[n+1][2n+1+2k+i, 2(k-1)+i-1] = recβ(T, S, n, k, i, 3)
        S.B[n+1][2*(2n+1)+2k+i, 2k+i] = c2
    end
    S
end
function getCs!(S::SphericalCapSpace{<:Any, T, <:Any, <:Any}, N, N₀) where T
    m = N₀
    resize!(S.C, N + 1)
    if N₀ == 0
        # C_0 does not exist
        m += 1
    end
    if m == 1
        n = 1
        S.C[n+1] = sparse(zeros(T, 3*(2n+1), 2n-1))
        k = 0; i = 0
        S.C[n+1][2, 1] = recα(T, S, n, k+1, 1)
        S.C[n+1][2n+1+3, 1] = recβ(T, S, n, k+1, i+1, 1)
        S.C[n+1][2*(2n+1)+1, 1] = recγ(T, S, n, k, 1)
        m += 1
    end
    for n = N:-1:m
        # We iterate over columns, not rows, for the C matrices

        S.C[n+1] = sparse(zeros(T, 3*(2n+1), 2n-1))

        k = 0; i = 0
        S.C[n+1][2, 1] = recα(T, S, n, k+1, 1)
        S.C[n+1][2n+1+3, 1] = recβ(T, S, n, k+1, i+1, 1)
        S.C[n+1][2*(2n+1)+1, 1] = recγ(T, S, n, k, 1)

        k = 1
        a1, a2 = recα(T, S, n, k+1, 1), recα(T, S, n, k-1, 2)
        c1 = recγ(T, S, n, k, 1)
        i = 0
        S.C[n+1][1, 2k+i] = a2
        S.C[n+1][2(k+1)+i, 2k+i] = a1
        S.C[n+1][2n+1+2(k+1)+i+1, 2k+i] = recβ(T, S, n, k+1, i+1, 1)
        S.C[n+1][2*(2n+1)+2k+i, 2k+i] = c1
        i = 1
        S.C[n+1][2(k+1)+i, 2k+i] = a1
        S.C[n+1][2n+1+1, 2k+i] = recβ(T, S, n, k-1, i-1, 2)
        S.C[n+1][2n+1+2(k+1)+i-1, 2k+i] = recβ(T, S, n, k+1, i-1, 1)
        S.C[n+1][2*(2n+1)+2k+i, 2k+i] = c1

        for k = 2:n-1
            a1, a2 = recα(T, S, n, k+1, 1), recα(T, S, n, k-1, 2)
            c1 = recγ(T, S, n, k, 1)
            i = 0
            S.C[n+1][2(k-1)+i, 2k+i] = a2
            S.C[n+1][2(k+1)+i, 2k+i] = a1
            S.C[n+1][2n+1+2(k-1)+i+1, 2k+i] = recβ(T, S, n, k-1, i+1, 2)
            S.C[n+1][2n+1+2(k+1)+i+1, 2k+i] = recβ(T, S, n, k+1, i+1, 1)
            S.C[n+1][2*(2n+1)+2k+i, 2k+i] = c1
            i = 1
            S.C[n+1][2(k-1)+i, 2k+i] = a2
            S.C[n+1][2(k+1)+i, 2k+i] = a1
            S.C[n+1][2n+1+2(k-1)+i-1, 2k+i] = recβ(T, S, n, k-1, i-1, 2)
            S.C[n+1][2n+1+2(k+1)+i-1, 2k+i] = recβ(T, S, n, k+1, i-1, 1)
            S.C[n+1][2*(2n+1)+2k+i, 2k+i] = c1
        end
    end
    S
end
function getDTs!(S::SphericalCapSpace{<:Any, T, <:Any, <:Any}, N, N₀) where T
    m = N₀
    resize!(S.DT, N + 1)
    if m == 0
        S.DT[1] = sparse([0 0 (1 / recγ(T, S, 0, 0, 3));
                          (1 / recα(T, S, 0, 0, 6)) 0 0;
                          0 (1 / recβ(T, S, 0, 0, 0, 6)) 0])
        m += 1
    end
    if m == 1
        n = 1
        S.DT[n+1] = sparse(zeros(T, 2n+3, 3*(2n+1)))
        k = 0; S.DT[n+1][k+1, 2*(2n+1)+k+1] = 1 / recγ(T, S, n, k, 3)
        for k = 1:n, i = 0:1
            S.DT[n+1][2k+i, 2*(2n+1)+2k+i] = 1 / recγ(T, S, n, k, 3)
        end
        η1 = 1 / recα(T, S, n, n, 6)
        η3 = 1 / recβ(T, S, n, n, 0, 6)
        η2 = - η1 * recα(T, S, n, n, 5) / recγ(T, S, n, n-1, 3)
        S.DT[n+1][2n+2, 2n] = η1
        S.DT[n+1][2n+2, 3*(2n+1)-2] = η2
        S.DT[n+1][2n+3, (2n+1)+2] = η3
        m += 1
    end
    for n = N:-1:m
        S.DT[n+1] = sparse(zeros(T, 2n+3, 3*(2n+1)))
        k = 0; S.DT[n+1][k+1, 2*(2n+1)+k+1] = 1 / recγ(T, S, n, k, 3)
        for k = 1:n, i = 0:1
            S.DT[n+1][2k+i, 2*(2n+1)+2k+i] = 1 / recγ(T, S, n, k, 3)
        end
        η1 = 1 / recα(T, S, n, n, 6)
        η2 = - η1 * recα(T, S, n, n, 5) / recγ(T, S, n, n-1, 3)
        for j = 0:1
            S.DT[n+1][2n+2+j, 2n+j] = η1
            S.DT[n+1][2n+2+j, 3*(2n+1)-3+j] = η2
        end
    end
    S
end
function getAs!(S::SphericalCapSpace{<:Any, T, <:Any, <:Any}, N, N₀) where T
    m = N₀
    resize!(S.A, N + 1)
    if N₀ == 0
        S.A[1] = sparse([0 recα(T, S, 0, 0, 6) 0;
                         0 0 recβ(T, S, 0, 0, 0, 6);
                         recγ(T, S, 0, 0, 3) 0 0])
        m += 1
    end
    for n = N:-1:m
        # We iterate over rows, not cols, for the A matrices

        S.A[n+1] = sparse(zeros(T, 3*(2n+1), 2n+3))

        k = 0; i = 0
        S.A[n+1][1, 2] = recα(T, S, n, k, 6)
        S.A[n+1][2n+1+1, 3] = recβ(T, S, n, k, i, 6)
        S.A[n+1][2*(2n+1)+1, 1] = recγ(T, S, n, k, 3)

        k = 1
        a5, a6 = recα(T, S, n, k, 5), recα(T, S, n, k, 6)
        c3 = recγ(T, S, n, k, 3)
        i = 0
        S.A[n+1][2k+i, 1] = a5
        S.A[n+1][2k+i, 2(k+1)+i] = a6
        S.A[n+1][2n+1+2k+i, 2(k+1)+i+1] = recβ(T, S, n, k, i, 6)
        S.A[n+1][2*(2n+1)+2k+i, 2k+i] = c3
        i = 1
        S.A[n+1][2k+i, 2(k+1)+i] = a6
        S.A[n+1][2n+1+2k+i, 1] = recβ(T, S, n, k, i, 5)
        S.A[n+1][2n+1+2k+i, 2(k+1)+i-1] = recβ(T, S, n, k, i, 6)
        S.A[n+1][2*(2n+1)+2k+i, 2k+i] = c3

        for k = 2:n
            a5, a6 = recα(T, S, n, k, 5), recα(T, S, n, k, 6)
            c3 = recγ(T, S, n, k, 3)
            i = 0
            S.A[n+1][2k+i, 2(k-1)+i] = a5
            S.A[n+1][2k+i, 2(k+1)+i] = a6
            S.A[n+1][2n+1+2k+i, 2(k-1)+i+1] = recβ(T, S, n, k, i, 5)
            S.A[n+1][2n+1+2k+i, 2(k+1)+i+1] = recβ(T, S, n, k, i, 6)
            S.A[n+1][2*(2n+1)+2k+i, 2k+i] = c3
            i = 1
            S.A[n+1][2k+i, 2(k-1)+i] = a5
            S.A[n+1][2k+i, 2(k+1)+i] = a6
            S.A[n+1][2n+1+2k+i, 2(k-1)+i-1] = recβ(T, S, n, k, i, 5)
            S.A[n+1][2n+1+2k+i, 2(k+1)+i-1] = recβ(T, S, n, k, i, 6)
            S.A[n+1][2*(2n+1)+2k+i, 2k+i] = c3
        end
    end
    S
end

function resizedata!(S::SphericalCapSpace, N)
    # N is the max degree of the OPs
    N₀ = length(S.B)
    N ≤ N₀ - 2 && return S
    @show "begin resizedata! for SphericalCapSpace", N

    # First, we need to call this to get the 1D OP rec coeffs
    resizedataonedimops!(S, N+4)

    # getAs!(S, N+1, N₀)
    # @show "done As"
    getBs!(S, N+1, N₀)
    @show "done Bs"
    getCs!(S, N+1, N₀)
    @show "done Cs"
    getDTs!(S, N+1, N₀)
    @show "done DTs"
    S
end

function clenshawG(::SphericalCapSpace, n, z)
    sp = sparse(I, 2n+1, 2n+1)
    [z[1] * sp; z[2] * sp; z[3] * sp]
end
function clenshaw(cfs::AbstractVector{T}, S::SphericalCapSpace, z) where T
    # TODO: Implement clenshaw for the spherical cap, without using the clenshaw
    #       mats (i.e. dont build the mats, just implement the arithmetic)

    # NOTE for now, we simply implement with the clenshaw mats as required

    # Convert the cfs vector from ordered by Fourier mode k, to by degree n
    f = convertcoeffsvecorder(S, cfs)

    m = length(f)
    N = Int(sqrt(m)) - 1
    resizedata!(S, N+1)
    f = PseudoBlockArray(f, [2n+1 for n=0:N])

    P0 = getdegzeropteval(T, S)
    if N == 0
        return f[1] * P0
    end
    γ2 = view(f, Block(N+1))'
    γ1 = view(f, Block(N))' - γ2 * S.DT[N] * (S.B[N] - clenshawG(S, N-1, z))
    for n = N-2:-1:0
        γ = (view(f, Block(n+1))'
             - γ1 * S.DT[n+1] * (S.B[n+1] - clenshawG(S, n, z))
             - γ2 * S.DT[n+2] * S.C[n+2])
        γ2 = copy(γ1)
        γ1 = copy(γ)
    end
    (γ1 * P0)[1]
end
evaluate(cfs::AbstractVector, S::SphericalCapSpace, z) = clenshaw(cfs, S, z)


#===#
# Resizing/Reordering coeffs vectors

# Method to convert coeffs vec from ordered by Fourier mode k to by degree n
function convertcoeffsvecorder(S::SphericalCapSpace, cfs::AbstractVector; todegree=true)
    f = copy(cfs)
    m = length(cfs)
    N = Int(sqrt(m)) - 1
    if (N+1)^2 != m
        error("coeffs vec incorrect length")
    end
    if todegree
        for k = 0:N
            indc = getopindex(S, k, k, 0; bydegree=false, N=N)
            for n = k:N
                for i = 0:min(1, k) # This catches the k == 0 case
                    f[getopindex(S, n, k, i)] = cfs[indc]
                    indc += 1
                end
            end
        end
    else
        # if todegree is false (or not true) then we convert to ordering by
        # Fourier mode k
        for k = 0:N
            indf = getopindex(S, k, k, 0; bydegree=false, N=N)
            for n = k:N
                for i = 0:min(1, k) # This catches the k == 0 case
                    f[indf] = cfs[getopindex(S, n, k, i)]
                    indf += 1
                end
            end
        end
    end
    f
end

# Resize the coeffs vector to be of the expected/standard length for a degree N
# expansion (so we can apply operators).
function resizecoeffs2!(f::Fun, N; bydegree=true)
    # NOTE if bydegree is true, then the order of the coeffs should already
    #      be ordered by degree
    if bydegree
        cfs = f.coefficients
        m̃ = length(cfs)
        m = Int((N+1)*(N+2)/2)
        if m̃ < m
            resize!(cfs, m)
            cfs[m̃+1:end] .= 0.0
        elseif m̃ > m
            for j = m+1:m̃
                if cfs[j] > 1e-16
                    error("Trying to decrease degree of f")
                end
            end
            resize!(cfs, m)
        end
        cfs
    else
        m̃ = length(f.coefficients)
        m = Int((N+1)*(N+2)/2)
        if m̃ > m
            error("Trying to decrease degree of f")
        end
        cfs = convertcoeffsvecorder(f.coefficients; todegree=false)
        resize!(cfs, m)
        cfs[m̃+1:end] .= 0.0
        f.coefficients[:] = convertcoeffsvecorder(cfs; todegree=true)[:]
        f.coefficients
    end
end


#===#
# Methods to return the spaces corresponding to applying the derivative operators
# TODO Check these are in fact correct. Check weighted (i.e. multiplyed by
#      w_R^{(a',0)}(z)) spaces too.

function differentiatespacephi(S::SphericalCapSpace; weighted=false, kind::Int=1)
    if kind == 1 # ρ(z)*∂/∂ϕ operator
        if weighted
            S.family(S.params[1]-1, S.params[2])
        else
            S.family(S.params[1]+1, S.params[2])
        end
    elseif kind == 2 # (1/ρ(z))*∂/∂ϕ operator
        S
    else
        error("invalid parameter")
    end
end
# NOTE this is just to be explicit
function differentiatespacetheta2(S::SphericalCapSpace)
    # ∂²/∂θ² operator
    S
end


#===#
# Differential operator matrices

function getpartialphival(S::SphericalCapSpace, ptsr, rhoptsr2, rhodxrhoptsr,
                            wr, n::Int, m::Int, k::Int)
    Sp = differentiatespacephi(S)
    R = getRspace(S, k); Rp = getRspace(Sp, k)
    ret = k * inner2(Rp, getptsevalforop(R, n-k),
                        getptsevalforop(Rp, m-k) .* rhodxrhoptsr, wr)
    ret += inner2(Rp, getderivptsevalforop(R, n-k),
                    getptsevalforop(Rp, m-k) .* rhoptsr2, wr)
    - ret / getopnorm(Rp)
end
function getweightedpartialphival(S::SphericalCapSpace, ptsr, rhoptsr2,
                                    rhodxrhoptsr, wr10, wr, n::Int, m::Int, k::Int)
    Sp = differentiatespacephi(S; weighted=true)
    R = getRspace(S, k); Rp = getRspace(Sp, k)
    ret = inner2(Rp, getptsevalforop(R, n-k),
                (k * rhodxrhoptsr .* wr10 + S.params[1] * rhoptsr2) .* getptsevalforop(Rp, m-k),
                wr)
    ret += inner2(Rp, getderivptsevalforop(R, n-k),
                    getptsevalforop(Rp, m-k) .* rhoptsr2 .* wr10, wr)
    - ret / getopnorm(Rp)
end
function diffoperatorphi(S::SphericalCapSpace{<:Any, B, T, <:Any}, N::Int;
                            weighted=false) where {B,T}
    # ρ(z)*∂/∂ϕ operator
    # Takes the space:
    #   Q^{a,b} -> Q^{a+1,b} or
    #   w_R^{(a,0)} Q^{a,b} -> w_R^{(a-1,0)} Q^{a-1,b}
    Sp = differentiatespacephi(S; weighted=weighted)
    # resizedataonedimops! for both SCSpaces
    resizedataonedimops!(S, N)
    resizedataonedimops!(Sp, N)
    # Get pts and weights, and set the R norms
    ptsr, wr = pointswithweights(B, getRspace(Sp, 0), N+4) # TODO how many pts needed?
    getopnorms(Sp, N)

    # Evaluate ρ.(ptsr) dρ/dx.(ptsr) at the R inner product points
    rhoptsr2 = S.family.ρ.(ptsr).^2
    rhodxrhoptsr = S.family.ρ.(ptsr) .* differentiate(S.family.ρ).(ptsr)

    if weighted
        band1 = 1
        band2 = 2
        wr10 = (ptsr .- S.family.α)
    else
        band1 = 2
        band2 = 1
    end
    # TODO sort out the blocks and stuff of this, and the allocation of val
    A = BandedBlockBandedMatrix(Zeros{B}(sum(1:(N+1+band2)), sum(1:(N+1))),
                                (N+1+band2:-1:1, N+1:-1:1), (band2, band1), (0, 0))
    for k = 0:N
        if k % 20 == 0
            @show "dϕ", weighted, k
        end
        R = getRspace(S, k); Rp = getRspace(Sp, k)
        getopptseval(R, N-k, ptsr); getopptseval(Rp, N-k, ptsr)
        getderivopptseval(R, N-k, ptsr)
        for n = k:N, m = max(0,n-band1):n+band2
            if weighted
                val = getweightedpartialphival(S, ptsr, rhoptsr, dxrhoptsr, wr10, wr, n, m, k)
            else
                val = getpartialphival(S, ptsr, rhoptsr, dxrhoptsr, wr, n, m, k)
            end
            view(A, Block(k+1, k+1))[m-k+1, n-k+1] = val
        end
        resetopptseval(R); resetopptseval(Rx)
        resetderivopptseval(R)
    end
    A
end
function diffoperatortheta2(S::SphericalCapSpace{<:Any, B, T, <:Any}, N::Int;
                            weighted=false) where {B,T}
    # ∂²/∂θ² operator
    # Takes the space:
    #   Q^{a,b} -> Q^{a,b} or
    #   w_R^{(a,0)} Q^{a,b} -> w_R^{(a,0)} Q^{a,b}
    A = BandedBlockBandedMatrix(Zeros{B}(sum(1:(N+1)), sum(1:(N+1))), (N+1:-1:1, N+1:-1:1), (0, 0), (0, 0))
    for k = 0:N
        view(A, Block(k+1, k+1)) .= Diagonal(ones(N-k+1)) * k^2
    end
    A
end


#===#
# Parameter Transform/Conversion operator matrices

function transformparamsoperator(S::SphericalCapSpace{<:Any, B, T, <:Any},
            St::SphericalCapSpace{<:Any, B, T, <:Any}, N;
            weighted=false) where {B,T}
    # TODO sort out the blocks and stuff of this, and the allocation of val
    # TODO do the other parameter conversions
    if weighted
        band1 = 0
        band2 = Int(S.params[1] - St.params[1])
        ptsr, wr = pointswithweights(B, getRspace(S, 0), N+1) # TODO check npts
    else
        band1 = Int(St.params[1] - S.params[1])
        band2 = 0
        ptsr, wr = pointswithweights(B, getRspace(St, 0), N+1) # TODO check npts
    end
    C = BandedBlockBandedMatrix(Zeros{B}(sum(1:(N+1+band2)),sum(1:(N+1))),
                                (1:N+1+band2, 1:N+1), (band,0), (0,0))
    rhoptsr = S.family.ρ.(ptsr)
    getopnorms(St, N)
    for k = 0:N
        if k % 100 == 0
            @show "trnsfrm (a,b)->(a±1,b)", weighted, k
        end
        R = getRspace(S, k); Rt = getRspace(St, k)
        getopptseval(R, N-k, ptsr); getopptseval(Rt, N-k, ptsr)
        for n = k:N, m = (n-band1):(n+band2)
            if k ≤ m
                val = inner2(R, opevalatpts(R, n-k+1, ptsr),
                                rhoptsr.^(2k) .* opevalatpts(Rt, m-k+1, ptsr), wr)
                view(C, Block(k+1, k+1))[m-k+1, n-k+1] = val / Rt.opnorm[1]
            end
        end
        resetopptseval(R); resetopptseval(Rt)
    end
    C
end


#===#
# Spherical Laplacian operator matrix

# TODO this is currently the operator for ρ²Δ
function laplaceoperator(S::SphericalCapSpace, St::SphericalCapSpace, N;
                            weighted=true, square=false)
    # Outputs the sum(1:N+1) × sum(1:N+1) matrix operator if square=true
    D = S.family
    # if ????
    A = diffoperatorphi(S, N)
    @show "laplaceoperator", "1 of 5 done"
    B = diffoperatorphi(S, N; weighted=true)
    @show "laplaceoperator", "2 of 5 done"
    C = transformparamsoperator(differentiatespacephi(S; weighted=true), S, N)
    @show "laplaceoperator", "3 of 5 done"
    E = transformparamsoperator(S, differentiatespacephi(S; weighted=true), N; weighted=true)
    @show "laplaceoperator", "4 of 5 done"
    F = diffoperatortheta2(S, N)
    @show "laplaceoperator", "5 of 5 done"
    # AAl, AAu = A.l + B.l, A.u + B.u
    # BBl, BBu = C.l + E.l, C.u + E.u + F.u
    # AAλ, AAμ = A.λ + B.λ, A.μ + B.μ
    # BBλ, BBμ = C.λ + E.λ , C.μ + E.μ + F.μ
    # AA = sparse(A) * sparse(B)
    # BB = sparse(C) * sparse(E) * sparse(F)
    # L = BandedBlockBandedMatrix(AA + BB, (1:nblocks(A)[1], 1:nblocks(B)[2]),
    #                             (max(AAl,BBl),max(AAu,BBu)), (max(AAλ,BBλ),max(AAμ,BBμ)))
    # if square
    #     m = sum(1:(N+1))
    #     Δ = BandedBlockBandedMatrix(L[1:m, 1:m], (1:N+1, 1:N+1), (L.l,L.u), (L.λ,L.μ))
    # else
    #     L
    # end
    A, B, C, E, F
end


#===#
# inner product

function inner(::Type{T}, S::SphericalCapSpace, fpts, gpts, w) where T
    m = length(w)
    T(sum((fpts[1:m] .* gpts[1:m] + fpts[m+1:end] .* gpts[m+1:end]) .* w) / 2)
end
inner(S::SphericalCapSpace, fpts, gpts, w::AbstractVector{T}) where T =
    inner(T, S, fpts, gpts, w)



#==========#
# Jacobi operators methods

#=
NOTE
These As Bs and Cs are only needed to be constructed for the Jacobi operators.
=#
function getjacobiAs(S::SphericalCapSpace{<:Any, B, T, <:Any}, N) where {B,T}
    A = Vector{SparseMatrixCSC{T}}()
    resize!(A, N+1)
    A[N+1] = [recα(T, S, N, N, 3) recα(T, S, N, N, 1);
              recβ(T, S, N, N, 3) recβ(T, S, N, N, 1);
              0 0]
    for k = N-1:-1:0
        v1 = [recα(T, S, n, k, 5) for n = N-1:-1:k]
        v2 = [recα(T, S, n, k, 3) for n = N:-1:k]
        v3 = [recα(T, S, n, k, 1) for n = N:-1:k+1]
        Ax = [Tridiagonal(v1, v2, v3) [zeros(T, N-k); recα(T, S, k, k, 1)]]
        v1 = [recβ(T, S, n, k, 5) for n = N-1:-1:k]
        v2 = [recβ(T, S, n, k, 3) for n = N:-1:k]
        v3 = [recβ(T, S, n, k, 1) for n = N:-1:k+1]
        Ay = [Tridiagonal(v1, v2, v3) [zeros(T, N-k); recβ(T, S, k, k, 1)]]
        Az = zeros(T, N-k+1, N-k+2)
        A[k+1] = [Ax; Ay; Az]
    end
    A
end
function getjacobiBs(S::SphericalCapSpace{<:Any, B, T, <:Any}, N) where {B,T}
    # B = Vector{SparseMatrixCSC{T}}()
    # resize!(B, N+1)
    # B[N+1] = [zeros(T, 2); recγ(T, S, N, N, 2)]
    # for k = N-1:-1:0
    #     Bx = zeros(T, N-k+1, N-k+1)
    #     By = copy(Bx)
    #     v1 = [recγ(T, S, n, k, 3) for n = N-1:-1:k]
    #     v2 = [recγ(T, S, n, k, 2) for n = N:-1:k]
    #     v3 = [recγ(T, S, n, k, 1) for n = N:-1:k+1]
    #     Bz = Tridiagonal(v1, v2, v3)
    #     B[k+1] = [Ax; Ay; Az]
    # end
    # B
end
function getjacobiCs(S::SphericalCapSpace{<:Any, B, T, <:Any}, N) where {B,T}
    C = Vector{SparseMatrixCSC{T}}()
    resize!(C, N+1)
    C[N+1] = [recα(T, S, N, N, 4); recα(T, S, N-1, N, 6);
              recβ(T, S, N, N, 4); recβ(T, S, N-1, N, 6);
              0; 0]
    for k = N-1:-1:0
        v1 = [recα(T, S, n, k, 6) for n = N-1:-1:k+1]
        v2 = [recα(T, S, n, k, 4) for n = N:-1:k+1]
        v3 = [recα(T, S, n, k, 2) for n = N:-1:k+2]
        Cx = [Tridiagonal(v1, v2, v3);
              zeros(T, 1, N-k-1) recα(T, S, k, k, 6)]
        v1 = [recβ(T, S, n, k, 6) for n = N-1:-1:k+1]
        v2 = [recβ(T, S, n, k, 4) for n = N:-1:k+1]
        v3 = [recβ(T, S, n, k, 2) for n = N:-1:k+2]
        Cy = [Tridiagonal(v1, v2, v3);
              zeros(T, 1, N-k-1) recβ(T, S, k, k, 6)]
        Cz = zeros(T, N-k+1, N-k)
        C[k+1] = [Cx; Cy; Cz]
    end
    C
end

# Methods to return the Jacobi operator matrices for mult by x, y, z
function jacobix(S::SphericalCapSpace, N)
    # Transposed operator, so acts directly on coeffs vec
    getclenshawmats!(S, N)
end
function jacobiy(S::SphericalCapSpace, N)
    # Transposed operator, so acts directly on coeffs vec
    getclenshawmats!(S, N)
end
function jacobiz(S::SphericalCapSpace, N)
    # Transposed operator, so acts directly on coeffs vec
    getclenshawmats!(S, N)
end
