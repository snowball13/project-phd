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
    ℚ^{(a,b)}_{N,0} := [Q^{(a,b)}_{0,0,0};Q^{(a,b)}_{1,0,0};...;Q^{(a,b)}_{N,0,0}] ∈ ℝ^{N+1}
                                                            for k = 0

=#

# TODO there should only be one parameter, not two (as b is not used)

export SphericalCapFamily, SphericalCapSpace

# T should be Float64, B should be BigFloat

# TODO Sort out checkpoints, in, and domain - should
abstract type SphericalFamily{B,T,N} end
struct SphericalCap{B,T} <: Domain{SVector{2,B}} end
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

    # Stores the α recurrence coeffs (for mult by x). No need to store β's, γ's
    # here too.
    reccoeffsa::Dict{NTuple{2,Int}, Vector{B}}

    # These store the "Clenshaw" matrices. Each subvector contains the x,y,z
    # submatrices
    A::Vector{Vector{BandedBlockBandedMatrix{B}}}
    B::Vector{Vector{BandedBlockBandedMatrix{B}}}
    C::Vector{Vector{BandedBlockBandedMatrix{B}}}
    DT::Vector{Vector{BandedBlockBandedMatrix{B}}}
end

struct SphericalCapTangentSpace{DF, B, T, N} <: Space{SphericalCap{B,T}, T}
    family::DF # Pointer back to the family
    params::NTuple{N,B} # Parameters
end

function SphericalCapSpace(fam::SphericalFamily{B,T,N}, params::NTuple{N,B}) where {B,T,N}
    SphericalCapSpace{typeof(fam), B, T, N}(
        fam, params, Vector{B}(), Vector{Vector{B}}(), Dict{NTuple{2,Int}, Vector{B}}(),
        Vector{Vector{BandedBlockBandedMatrix{B}}}(), Vector{Vector{BandedBlockBandedMatrix{B}}}(),
        Vector{Vector{BandedBlockBandedMatrix{B}}}(), Vector{Vector{BandedBlockBandedMatrix{B}}}())
end

spacescompatible(A::SphericalCapSpace, B::SphericalCapSpace) = (A.params == B.params)

domain(::SphericalCapSpace{<:Any, B, T, <:Any}) where {B,T} = SphericalCap{B,T}()

struct SphericalCapFamily{B,T,N,FA,F,I} <: SphericalFamily{B,T,N}
    spaces::Dict{NTuple{N,B}, SphericalCapSpace}
    tangentspace::Vector{SphericalCapTangentSpace}
    α::B
    β::B
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

function SphericalCapFamily(::Type{B}, ::Type{T}, α::B) where {B,T}
    β = B(1)
    X = Fun(identity, α..β)
    ρ = sqrt(1 - X^2)
    ρ2 = 1 - X^2 # NOTE we use ρ^2 here to help computationally
    # NOTE also that we only have a 2-param family for the spherical/polar cap
    nparams = 2
    R = OrthogonalPolynomialFamily(T, X-α, ρ2)
    spaces = Dict{NTuple{nparams,B}, SphericalCapSpace}()
    tspace = Vector{SphericalCapTangentSpace}()
    SphericalCapFamily{B,T,nparams,typeof(R),typeof(ρ),Int}(spaces, tspace, α, β, R, ρ, nparams)
end
# Useful quick constructors
SphericalCapFamily(α::T) where T = SphericalCapFamily(BigFloat, T, BigFloat(α * 1000) / 1000)
SphericalCapFamily() = SphericalCapFamily(BigFloat, Float64, BigFloat(0)) # Hemisphere



#=======#


#===#
# Weight eval functions for the 3D space
function weight(S::SphericalCapSpace, x, y, z)
    (z - S.family.α)^S.params[1]
    # if length(S.params) == 1
    #     (x - S.family.α)^S.params[1]
    # else # length(S.params) == 2
    #     (S.family.β - x)^S.params[1] * (x - S.family.α)^S.params[2]
    # end
end
weight(S::SphericalCapSpace, z) = weight(S, z[1], z[2], z[3])
function weight(::Type{T}, S::SphericalCapSpace, x, y, z) where T
    T((z - S.family.α)^S.params[1])
    # if length(S.params) == 1
    #     T((x - S.family.α)^S.params[1])
    # else # length(S.params) == 2
    #     T((S.family.β - x)^S.params[1] * (x - S.family.α)^S.params[2])
    # end
end
weight(::Type{T}, S::SphericalCapSpace, z) where T = weight(T, S, z[1], z[2], z[3])


#===#
# Methods to handle the R OPs (the 1D OP family that is a part of the
# SphericalFamily)

# Retrieve 1D OP spaces methods
getRspace(S::SphericalCapSpace, k::Int) =
    (S.family.R)(S.params[1], S.params[2] + k) # was (S.family.R)(S.params[1], (2S.params[2] + 2k + 1)/2)
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
    R00 = getRspace(S, -c) # R00 = R(S.a, S.b, 0)
    M = 2 * (N + c)

    # See if we need to proceed (is length(R(0,0,N+c).α) > 0)
    length(getRspace(S, N).a) > 0 && return S

    # resizedata!() on R(0,0,0) up to deg M to initialise
    resizedata!(R00, M+1)

    # Loop over k value in R(0,0,k) to recursively build the rec coeffs for
    # each OPSpace, so that length(R(0,0,N+c).α) == 1
    for k = 0:(maxkval - 1)
        if k % 100 == 0
            @show "getreccoeffsR!", k
        end
        # interim coeffs
        R00 = getRspace(S, k - c) # R00 = R(S.a, k)
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
        R11 = getRspace(S, k - c + 1) # R11 = R(S.a, k+1)
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

function getrecα(::Type{T}, S::SphericalCapSpace, n::Int, k::Int, jj::Int) where T
    # This function returns the jjth α rec coeff for mult by x for the (n,k) pair

    resizedataonedimops!(S, n+3)

    getopnorms(S, k+1)
    R1 = getRspace(S, k-1)
    R2 = getRspace(S, k)
    R3 = getRspace(S, k+1)

    failval = T(0)
    if isodd(jj)
        k == 0 && return failval
        pts, w = pointswithweights(T, R2, Int(ceil(n-k+1.5)))
    else
        (n == k && jj != 6) && return failval
        (n == 0 && jj == 2) && return failval
        (n == k + 1 && jj == 2) && return failval
        pts, w = pointswithweights(T, R3, Int(ceil(n-k+1.5)))
    end
    getopptseval(R2, n-k+1, pts)

    if jj == 1
        getopptseval(R1, n-k+1, pts)
        ret = (inner2(R2, opevalatpts(R2, n-k+1, pts), opevalatpts(R1, n-k+1, pts), w)
                / getopnorm(R1))
    elseif jj == 2
        getopptseval(R3, n-k-1, pts)
        ret = (inner2(R3, opevalatpts(R2, n-k+1, pts), opevalatpts(R3, n-k-1, pts), w)
                / getopnorm(R3))
    elseif jj == 3
        getopptseval(R1, n-k+2, pts)
        ret = (inner2(R2, opevalatpts(R2, n-k+1, pts), opevalatpts(R1, n-k+2, pts), w)
                / getopnorm(R1))
    elseif jj == 4
        getopptseval(R3, n-k, pts)
        ret = (inner2(R3, opevalatpts(R2, n-k+1, pts), opevalatpts(R3, n-k, pts), w)
                / getopnorm(R3))
    elseif jj == 5
        getopptseval(R1, n-k+3, pts)
        ret = (inner2(R2, opevalatpts(R2, n-k+1, pts), opevalatpts(R1, n-k+3, pts), w)
                / getopnorm(R1))
    elseif jj == 6
        getopptseval(R3, n-k+1, pts)
        ret = (inner2(R3, opevalatpts(R2, n-k+1, pts), opevalatpts(R3, n-k+1, pts), w)
                / getopnorm(R3))
    else
        error("Invalid entry to function")
    end

    if k == 0 || (k == 1 && isodd(jj))
        ret *= getdegzeropteval(T, S)
    else
        ret /= 2
    end
    T(ret)
end
function recα(::Type{T}, S::SphericalCapSpace, n::Int, k::Int, j::Int) where T
    # Check if we already have this stored
    haskey(S.reccoeffsa, (n, k)) && return S.reccoeffsa[(n, k)][j]

    # If we are over a certain degree, dont bother storing (for disk-space reasons)
    n > 150 && return getrecα(T, S, n, k, j)

    # Else, we calculate each (valid) coeff j=1:6 for the (n,k) pair
    S.reccoeffsa[(n, k)] = zeros(T, 6)
    for jj = 1:6
        S.reccoeffsa[(n,k)][jj] = getrecα(T, S, n, k, jj)
    end
    S.reccoeffsa[(n,k)][j]
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

# NOTE we output M≈n points (x,y,z), plus the M≈n points corresponding to (-x,-y,z)
function pointswithweights(S::SphericalCapSpace{<:Any, B, T, <:Any}, n;
                            nofactor=false) where {B,T}
    # Return the weights and nodes to use for the even part of a function,
    # i.e. for the spherical cap Ω:
    #   int_Ω w_R^{a,2b}(x) f(x,y,z) dσ(x,y)dz ≈ 0.5 * Σ_j wⱼ*(f(xⱼ,yⱼ,zⱼ) + f(-xⱼ,-yⱼ,zⱼ))
    # NOTE: the "odd" part of the quad rule will equal 0 for polynomials,
    #       so can be ignored.

    # When nofactor is true, then the weights are not multiplyed by 2π

    if n < 1
        error("At least 1 point needs to be asked for in pointswithweights().")
    end

    # Degree of polynomial f(x,y,z) is N
    N = Int(ceil(-1.5 + 0.5 * sqrt(9 - 4 * (2 - 2n)))) # degree we approximate up to with M quadrature pts
    M1 = Int(ceil((N+1) / 2)) # Quad rule for z interval exact for degree polynomial up to 2M1 - 1 (= N)
    M2 = N + 1 # Quad rule for circle exact for polynomial up to degree M2 - 1 (= N)
    M = M1 * M2 # Quad rule on Ω is exact for polynomials of degree N s.t. we have M points
    @show "begin pointswithweights()", n, M, N

    # Get the 1D quadrature pts and weights
    # Need to maunally call the method to get R coeffs here
    m = isodd(M1) ? Int((M1 + 1) / 2) : Int((M1 + 2) / 2); m -= Int(S.params[end])
    getreccoeffsR!(S, m; maxk=0)
    t, wt = pointswithweights(B, getRspace(S, 0), M1) # Quad for w_R^{(a,2b)}
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
            pts[j2 + (j1 - 1)M2] = x, y, z
            pts[M + j2 + (j1 - 1)M2] = -x, -y, z
            w[j2 + (j1 - 1)M2] = wt[j1] * ws[j2]
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
                x, y, z = pts[r]
                P1 = [opevalatpts(S, nm1+it, pts)[r] for it = 0:2(n-1)]
                P = - clenshawDTBmG(S, n-1, P1, x, y, z; clenshawalg=false)
                for k = 0:2n
                    S.opptseval[jj+k][r] = P[k+1]
                end
            end
        else
            nm1 = getopindex(S, n-1, 0, 0)
            nm2 = getopindex(S, n-2, 0, 0)
            for r = 1:length(pts)
                x, y, z = pts[r]
                P1 = [opevalatpts(S, nm1+it, pts)[r] for it = 0:2(n-1)]
                P2 = [opevalatpts(S, nm2+it, pts)[r] for it = 0:2(n-2)]
                P = - (clenshawDTBmG(S, n-1, P1, x, y, z; clenshawalg=false)
                        + clenshawDTC(S, n-1, P2; clenshawalg=false))
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
    #      coefficients for.
    #      We should have M vals such that the quadrature rule is exact to
    #      calculate integrals of f * Q_{n,k,i} for n=0:N, which will have a max
    #      degree of 2N (i.e. N is constrained by the number of vals we have -
    #      with M pts, we have a quad rule exact for a poly of 2N).
    #      nops is the number of OPs we require (that is, all OPs up to and
    #      including deg N, i.e. length of ℚ^{(a,b)}_N) which is (N+1)^2.
    # nofactor=true means we dont have the 2pi factor in the weights

    npts = Int(length(vals) / 2) # = M

    # Divide by 2 as the quad rule is for ∫_Ω f(x,y,z)*Q_{n,k,i}(x,y,z) dσ(x,y) dz for n=0,...,N where deg(f)=N
    N = Int(floor(ceil(-1.5 + 0.5 * sqrt(9 - 4 * (2 - 2npts))) / 2))
    nops = (N+1)^2
    @show N, npts, nops

    resizedata!(S, N)
    getopnorms(S, N)
    pts, w = pointswithweights(S, npts; nofactor=true)

    # calculate the Vandermonde matrix
    Vp = zeros(B, nops, npts); Vm = zeros(B, nops, npts)
    p = Vector{SArray{Tuple{3},B,1,3}}(undef, 2)
    for j = 1:npts
        if j % 100 == 0
            @show j, npts
        end
        p[1] = pts[j]; p[2] = pts[j+npts]
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
    N = Int(sqrt(ncfs)) - 1 # We have (N+1)^2 OPs (number of OPs deg ≤ N)
    npts = (2N+1) * (N+1) # = (2N+1)(2N+2)/2
    @show npts, N
    pts = points(S, npts)
    ret = zeros(T, 2npts)
    for j = 1:npts
        getopptseval(S, N, (pts[j], pts[j+npts]))
        indc = 1
        for k = 0:N
            # indc = getopindex(S, k, k, 0; bydegree=false, N=N)
            for n = k:N
                inde = getopindex(S, n, k, 0)
                for i = 0:min(1, k) # This catches the k == 0 case
                    # NOTE use indexing of opptseval to order rows of V by Fourier mode k (not degree)
                    ret[j] += getptsevalforop(S, inde+i)[1] * cfs[indc]
                    ret[j + npts] += getptsevalforop(S, inde+i)[2] * cfs[indc]
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

function getclenshawsubblockx(S::SphericalCapSpace{<:Any, T, <:Any, <:Any},
                                n::Int; subblock::String="A") where T
    @assert subblock in ("A", "B", "C") "Invalid subblock given"
    @assert n ≥ 0 "Invalid n - should be non-negative integer"
    if subblock == "A"
        u, l = 6, 5
        maxk = n
        bandn = 1
    elseif subblock == "B"
        u, l = 4, 3
        maxk = n
        bandn = 0
    else
        u, l = 2, 1
        maxk = n - 1
        bandn = -1
        n == 0 && error("n needs to be > 0 when Clenshaw mat C requested")
    end
    mat = BandedBlockBandedMatrix(Zeros{T}(2n+1, 2(n+bandn)+1),
                                    [1; [2 for k=1:n]], [1; [2 for k=1:(n+bandn)]],
                                    (1, 1), (0, 0))
    id = [1 0; 0 1]
    # Handle the "extra bits" and special cases here
    if subblock == "A"
        if n == 0
            view(mat, Block(1, 2)) .= [recα(T, S, n, 0, u) 0]
            return mat
        else
            k = n
            a = recα(T, S, n, k, u)
            view(mat, Block(k+1, k+2)) .= a * id
        end
    elseif subblock == "C"
        if n == 1
            view(mat, Block(2, 1)) .= [recα(T, S, n, 1, l); 0]
            return mat
        else
            k = n
            a = recα(T, S, n, k, l)
            view(mat, Block(k+1, k)) .= a * id
        end
        n == 2 && return mat
    end
    n == 0 && subblock == "B" && return mat
    # Now iterate over k
    k = 1
    al, au = recα(T, S, n, k, l), recα(T, S, n, k-1, u)
    view(mat, Block(k, k+1)) .= [au 0]
    view(mat, Block(k+1, k)) .= [al; 0]
    for k = 2:maxk
        al, au = recα(T, S, n, k, l), recα(T, S, n, k-1, u)
        view(mat, Block(k, k+1)) .= au * id
        view(mat, Block(k+1, k)) .= al * id
    end
    mat
end
function getclenshawsubblocky(S::SphericalCapSpace{<:Any, T, <:Any, <:Any},
                                n::Int; subblock::String="A") where T
    @assert subblock in ("A", "B", "C") "Invalid subblock given"
    @assert n ≥ 0 "Invalid n - should be non-negative integer"
    if subblock == "A"
        u, l = 6, 5
        maxk = n
        bandn = 1
    elseif subblock == "B"
        u, l = 4, 3
        maxk = n
        bandn = 0
    else
        u, l = 2, 1
        maxk = n - 1
        bandn = -1
        n == 0 && error("n needs to be > 0 when Clenshaw mat C requested")
    end
    mat = BandedBlockBandedMatrix(Zeros{T}(2n+1, 2(n+bandn)+1),
                                    [1; [2 for k=1:n]], [1; [2 for k=1:(n+bandn)]],
                                    (1, 1), (1, 1))
    # Handle the "extra bits" and special cases here
    if subblock == "A"
        if n == 0
            view(mat, Block(1, 2)) .= [0 recβ(T, S, n, 0, 0, u)]
            return mat
        else
            k = n
            view(mat, Block(k+1, k+2)) .= [0 recβ(T, S, n, k, 0, u);
                                           recβ(T, S, n, k, 1, u) 0]
        end
    elseif subblock == "C"
        if n == 1
            view(mat, Block(2, 1)) .= [0; recβ(T, S, n, 1, 1, l)]
            return mat
        else
            k = n
            view(mat, Block(k+1, k)) .= [0 recβ(T, S, n, k, 0, l);
                                         recβ(T, S, n, k, 1, l) 0]
        end
    end
    n == 0 && subblock == "B" && return mat
    # Now iterate over k
    k = 1
    view(mat, Block(k, k+1)) .= [0 recβ(T, S, n, k-1, 0, u)]
    view(mat, Block(k+1, k)) .= [0; recβ(T, S, n, k, 1, l)]
    for k = 2:maxk
        view(mat, Block(k, k+1)) .= [0 recβ(T, S, n, k-1, 0, u);
                                       recβ(T, S, n, k-1, 1, u) 0]
        view(mat, Block(k+1, k)) .= [0 recβ(T, S, n, k, 0, l);
                                       recβ(T, S, n, k, 1, l) 0]
    end
    mat
end
function getclenshawsubblockz(S::SphericalCapSpace{<:Any, T, <:Any, <:Any},
                                n::Int; subblock::String="A") where T
    @assert subblock in ("A", "B", "C") "Invalid subblock given"
    if subblock == "A"
        bandn = 1
        j = 3
        maxk = n
    elseif subblock == "B"
        bandn = 0
        j = 2
        maxk = n
    else
        bandn = -1
        j = 1
        maxk = n - 1
    end
    mat = BandedBlockBandedMatrix(Zeros{T}(2n+1, 2(n+bandn)+1),
                                    [1; [2 for k=1:n]], [1; [2 for k=1:(n+bandn)]],
                                    (0, 0), (0, 0))
    k = 0
    view(mat, Block(k+1, k+1)) .= [recγ(T, S, n, k, j)]
    id = [1 0; 0 1]
    for k = 1:maxk
        c = recγ(T, S, n, k, j)
        view(mat, Block(k+1, k+1)) .= c * id
    end
    mat
end

function getBs!(S::SphericalCapSpace{<:Any, T, <:Any, <:Any}, N, N₀) where T
    m = N₀
    resize!(S.B, N + 1)
    subblock = "B"
    for n = N:-1:m
        S.B[n+1] = Vector{BandedBlockBandedMatrix{T}}(undef, 3)
        resize!(S.B[n+1], 3)
        S.B[n+1][1] = getclenshawsubblockx(S, n; subblock=subblock)
        S.B[n+1][2] = getclenshawsubblocky(S, n; subblock=subblock)
        S.B[n+1][3] = getclenshawsubblockz(S, n; subblock=subblock)
    end
    S
end
function getCs!(S::SphericalCapSpace{<:Any, T, <:Any, <:Any}, N, N₀) where T
    m = N₀
    resize!(S.C, N + 1)
    subblock = "C"
    if N₀ == 0
        m += 1 # C_0 does not exist
    end
    for n = N:-1:m
        S.C[n+1] = Vector{BandedBlockBandedMatrix{T}}(undef, 3)
        resize!(S.C[n+1], 3)
        S.C[n+1][1] = getclenshawsubblockx(S, n; subblock=subblock)
        S.C[n+1][2] = getclenshawsubblocky(S, n; subblock=subblock)
        S.C[n+1][3] = getclenshawsubblockz(S, n; subblock=subblock)
    end
    S
end
function getAs!(S::SphericalCapSpace{<:Any, T, <:Any, <:Any}, N, N₀) where T
    m = N₀
    resize!(S.A, N + 1)
    subblock = "A"
    for n = N:-1:m
        S.A[n+1] = Vector{BandedBlockBandedMatrix{T}}(undef, 3)
        resize!(S.A[n+1], 3)
        S.A[n+1][1] = getclenshawsubblockx(S, n; subblock=subblock)
        S.A[n+1][2] = getclenshawsubblocky(S, n; subblock=subblock)
        S.A[n+1][3] = getclenshawsubblockz(S, n; subblock=subblock)
    end
    S
end
function getDTs!(S::SphericalCapSpace{<:Any, T, <:Any, <:Any}, N, N₀) where T
    # Need to store these as BandedBlockBandedMatrices for each subblock
    # corresponding to x,y,z.
    # i.e. We store [DT_{x,n}, DT_{y,n}, DT_{z,n}] where
    #    I = DTn*An = DT_{x,n}*A_{x,n} + DT_{y,n}*A_{y,n} + DT_{z,n}*A_{z,n}

    m = N₀
    resize!(S.DT, N + 1)
    if m == 0
        n = 0
        S.DT[n+1] = Vector{BandedBlockBandedMatrix{T}}(undef, 3)
        resize!(S.DT[n+1], 3)
        S.DT[n+1][1] = BandedBlockBandedMatrix(Zeros{T}(2n+3, 2n+1),
                                    [1; [2 for k=1:(n+1)]], [1; [2 for k=1:n]],
                                    (1, -1), (0, 0))
        S.DT[n+1][2] = BandedBlockBandedMatrix(Zeros{T}(2n+3, 2n+1),
                                    [1; [2 for k=1:(n+1)]], [1; [2 for k=1:n]],
                                    (1, -1), (1, -1))
        S.DT[n+1][3] = BandedBlockBandedMatrix(Zeros{T}(2n+3, 2n+1),
                                    [1; [2 for k=1:(n+1)]], [1; [2 for k=1:n]],
                                    (0, 0), (0, 0))
        S.DT[n+1][1][2, 1] = 1 / recα(T, S, 0, 0, 6)
        S.DT[n+1][2][3, 1] = 1 / recβ(T, S, 0, 0, 0, 6)
        S.DT[n+1][3][1, 1] = 1 / recγ(T, S, 0, 0, 3)
        m += 1
    end
    for n = N:-1:m
        S.DT[n+1] = Vector{BandedBlockBandedMatrix{T}}(undef, 3)
        resize!(S.DT[n+1], 3)
        # Define
        S.DT[n+1][1] = BandedBlockBandedMatrix(Zeros{T}(2n+3, 2n+1),
                                    [1; [2 for k=1:(n+1)]], [1; [2 for k=1:n]],
                                    (1, -1), (0, 0))
        Dx = S.DT[n+1][1]
        S.DT[n+1][2] = BandedBlockBandedMatrix(Zeros{T}(2n+3, 2n+1),
                                    [1; [2 for k=1:(n+1)]], [1; [2 for k=1:n]],
                                    (1, -1), (-1, 1))
        Dy = S.DT[n+1][2]
        S.DT[n+1][3] = BandedBlockBandedMatrix(Zeros{T}(2n+3, 2n+1),
                                    [1; [2 for k=1:(n+1)]], [1; [2 for k=1:n]],
                                    (2, 0), (0, 0))
        Dz = S.DT[n+1][3]

        # Put in nonzeros
        # z
        k = 0; Dz[k+1, k+1] = 1 / recγ(T, S, n, k, 3)
        for k = 1:n, i = 0:1
            view(Dz, Block(k+1, k+1))[i+1, i+1] = 1 / recγ(T, S, n, k, 3)
        end
        η00 = 1 / recβ(T, S, n, n, 1, 6)
        η10 = 1 / recα(T, S, n, n, 6)
        η01 = - η00 * recβ(T, S, n, n, 1, 5) / recγ(T, S, n, n-1, 3)
        view(Dx, Block(n+2, n+1))[2, 2] = η10
        view(Dy, Block(n+2, n+1))[1, 2] = η00
        view(Dz, Block(n+2, n))[1, 1] = η01
        if n > 1
            η11 = - η10 * recβ(T, S, n, n, 0, 5) / recγ(T, S, n, n-1, 3)
            view(Dz, Block(n+2, n))[2, 2] = η11
        end
    end
    S

    # for n = N:-1:m
    #     resize!(S.DT[n+1], 3)
    #     S.DT[n+1][1] = BandedBlockBandedMatrix(Zeros{T}(2n+3, 2n+1),
    #                                 ([1; [2 for k=1:(n+1)]], [1; [2 for k=1:n]]),
    #                                 (1, -1), (0, 0))
    #     S.DT[n+1][3] = BandedBlockBandedMatrix(Zeros{T}(2n+3, 2n+1),
    #                                 ([1; [2 for k=1:(n+1)]], [1; [2 for k=1:n]]),
    #                                 (2, 0), (0, 0))
    # end
    # S
    #
    #
    # m = N₀
    # resize!(S.DT, N + 1)
    # if m == 0
    #     S.DT[1] = sparse([0 0 (1 / recγ(T, S, 0, 0, 3));
    #                       (1 / recα(T, S, 0, 0, 6)) 0 0;
    #                       0 (1 / recβ(T, S, 0, 0, 0, 6)) 0])
    #     m += 1
    # end
    # if m == 1
    #     n = 1
    #     S.DT[n+1] = sparse(zeros(T, 2n+3, 3*(2n+1)))
    #     k = 0; S.DT[n+1][k+1, 2*(2n+1)+k+1] = 1 / recγ(T, S, n, k, 3)
    #     for k = 1:n, i = 0:1
    #         S.DT[n+1][2k+i, 2*(2n+1)+2k+i] = 1 / recγ(T, S, n, k, 3)
    #     end
    #     η1 = 1 / recα(T, S, n, n, 6)
    #     η3 = 1 / recβ(T, S, n, n, 0, 6)
    #     η2 = - η1 * recα(T, S, n, n, 5) / recγ(T, S, n, n-1, 3)
    #     S.DT[n+1][2n+2, 2n] = η1
    #     S.DT[n+1][2n+2, 3*(2n+1)-2] = η2
    #     S.DT[n+1][2n+3, (2n+1)+2] = η3
    #     m += 1
    # end
    # for n = N:-1:m
    #     S.DT[n+1] = sparse(zeros(T, 2n+3, 3*(2n+1)))
    #     k = 0; S.DT[n+1][k+1, 2*(2n+1)+k+1] = 1 / recγ(T, S, n, k, 3)
    #     for k = 1:n, i = 0:1
    #         S.DT[n+1][2k+i, 2*(2n+1)+2k+i] = 1 / recγ(T, S, n, k, 3)
    #     end
    #     η1 = 1 / recα(T, S, n, n, 6)
    #     η2 = - η1 * recα(T, S, n, n, 5) / recγ(T, S, n, n-1, 3)
    #     for j = 0:1
    #         S.DT[n+1][2n+2+j, 2n+j] = η1
    #         S.DT[n+1][2n+2+j, 3*(2n+1)-3+j] = η2
    #     end
    # end
    # S
end

function resizedata!(S::SphericalCapSpace, N; jacobimatcall=false)
    if jacobimatcall
        # N is the max degree of the OPs
        N₀ = length(S.B)
        N ≤ N₀ - 2 && return S
        @show "begin resizedata! for SphericalCapSpace", N

        # First, we need to call this to get the 1D OP rec coeffs
        resizedataonedimops!(S, N+4)

        getAs!(S, N+1, N₀)
        @show "done As"
        getBs!(S, N+1, N₀)
        @show "done Bs"
        getCs!(S, N+1, N₀)
        @show "done Cs"
        # getDTs!(S, N+1, N₀)
        # @show "done DTs"
        S
    else
        # TODO This is now obsolete. Need to remove this storing of clenshaw
        # mats as only used for constructing jacobi operators, which should be
        # changed to be done in the function call itself.
        S
    end
end

# function clenshawDTBmG(S::SphericalCapSpace, n::Int, pt)
#     id = BandedBlockBandedMatrix(I, ([1; [2 for k=1:n]], [1; [2 for k=1:n]]),
#                                     (0, 0), (0, 0))
#     ret = sum([S.DT[n+1][i] * (S.B[n+1][i] - pt[i] * id) for i=1:3])
#     ret
# end
# function clenshawDTC(S::SphericalCapSpace, n::Int)
#     ret = sum([S.DT[n+1][i] * S.C[n+1][i] for i=1:3])
#     ret
# end
# function clenshawG(::SphericalCapSpace, n, z)
#     sp = sparse(I, 2n+1, 2n+1)
#     [z[1] * sp; z[2] * sp; z[3] * sp]
# end

function clenshawDTBmG(S::SphericalCapSpace{<:Any, T, <:Any, <:Any}, n::Int,
                        ξ::AbstractArray{R}, X::R, Y::R, Z::R;
                        clenshawalg=true, operator=false) where {T,R}
    # Returns:
    #   ξ * DTn * (Bn - Gn(x,y,z)) if clenshawalg
    #   DTn * (Bn - Gn(x,y,z)) * ξ if !clenshawalg
    # where ξ is appropriatly sized vector

    # If operator then X, Y, Z are Jacobi matrices

    # NOTE R here is either T (scalar) or BandedBlockBandedMatrix{T} (operator)

    # TODO obtain from the indices of the clenshaw mats OR! store these coeffs
    # as vectors, and only do clenshaw like this!

    if n > 0
        η00 = 1 / recβ(T, S, n, n, 1, 6)
        η10 = 1 / recα(T, S, n, n, 6)
        η01 = - η00 * recβ(T, S, n, n, 1, 5) / recγ(T, S, n, n-1, 3)
        η11 = n == 1 ? 0 : - η10 * recα(T, S, n, n, 5) / recγ(T, S, n, n-1, 3)
    end

    if clenshawalg

        ret = Array{R}(undef, (1, 2n+1))

        if n == 0
            # Special case
            if operator
                ret[1,1] = - (ξ[1,1] * (Z - recγ(T, S, 0, 0, 2) * I) / recγ(T, S, 0, 0, 3)
                                + ξ[1,2] * X / recα(T, S, 0, 0, 6)
                                + ξ[1,3] * Y / recβ(T, S, 0, 0, 0, 6))
            else
                ret[1,1] = - (ξ[1,1] * (Z - recγ(T, S, 0, 0, 2)) / recγ(T, S, 0, 0, 3)
                                + ξ[1,2] * X / recα(T, S, 0, 0, 6)
                                + ξ[1,3] * Y / recβ(T, S, 0, 0, 0, 6))
            end
            return ret
        end

        # z
        ind = 1
        if operator
            for k = 0:n, i = 0:min(1,k)
                ret[1,ind] = - ξ[1,ind] * (Z - recγ(T, S, n, k, 2) * I) / recγ(T, S, n, k, 3)
                ind += 1
            end
        else
            for k = 0:n, i = 0:min(1,k)
                ret[1,ind] = - ξ[1,ind] * (Z - recγ(T, S, n, k, 2)) / recγ(T, S, n, k, 3)
                ind += 1
            end
        end
        ind = 2
        k = n - 1
        ind = n == 1 ? 2 : 3
        if operator
            ret[1,end-ind] -= ξ[1,end-1] * (Z - recγ(T, S, n, k, 2) * I) * η01
        else
            ret[1,end-ind] -= ξ[1,end-1] * (Z - recγ(T, S, n, k, 2)) * η01
        end
        # x
        ret[1,end] -= η10 * ξ[1,end] * X # + as already defined in z loop
        # y
        ret[1,end-ind] += recβ(T, S, n, n, 1, 3) * η00 * ξ[1,end-1]
        ret[1,end] -= η00 * ξ[1,end-1] * Y
        if n > 1
            if operator
                ret[1,end-2] -= ξ[1,end] * (Z - recγ(T, S, n, k, 2) * I) * η11
            else
                ret[1,end-2] -= ξ[1,end] * (Z - recγ(T, S, n, k, 2)) * η11
            end
            ret[1,end-2] += recα(T, S, n, n, 3) * η10 * ξ[1,end]
        end

        ret

    else
        @assert length(ξ) == 2n+1 "Invalid ξ vector"
        ret = zeros(R, 2n+3)

        if n == 0
            # Special case
            ret[1] = - ξ[1] * (Z - recγ(T, S, 0, 0, 2)) / recγ(T, S, 0, 0, 3)
            ret[2] = - ξ[1] * X / recα(T, S, 0, 0, 6)
            ret[3] = - ξ[1] * Y / recβ(T, S, 0, 0, 0, 6)
            return ret
        end

        # z
        ind = 1
        for k = 0:n, i = 0:min(1,k)
            ret[ind] = - ξ[ind] * (Z - recγ(T, S, n, k, 2)) / recγ(T, S, n, k, 3)
            ind += 1
        end
        ind = n == 1 ? 2 : 3
        ret[end-1] = - ξ[end-ind] * (Z - recγ(T, S, n, n-1, 2)) * η01
        # y
        ret[end-1] += ξ[end-ind] * recβ(T, S, n, n, 1, 3) * η00 - ξ[end] * Y * η00
        # x
        ret[end] = - ξ[end] * X * η10
        if n > 1
            ret[end] -= ξ[end-2] * (Z - recγ(T, S, n, n-1, 2)) * η11
            ret[end] += ξ[end-2] * recα(T, S, n, n, 3) * η10
        end
        ret
    end
end
function clenshawDTC(S::SphericalCapSpace{<:Any, T, <:Any, <:Any}, n::Int,
                        ξ::AbstractArray{R}; clenshawalg=true) where {T,R}

    # Returns:
    #   ξ * DTn * (Bn - Gn(x,y,z)) if clenshawalg
    #   DTn * (Bn - Gn(x,y,z)) * ξ if !clenshawalg
    # where ξ is appropriatly sized vector
    # NOTE R here is either T (scalar) or BandedBlockBandedMatrix{T} (operator)

    # TODO obtain from the indices of the clenshaw mats OR! store these coeffs
    # as vectors, and only do clenshaw like this!

    @assert n > 0 "Invalid n"

    η00 = 1 / recβ(T, S, n, n, 1, 6)
    η10 = 1 / recα(T, S, n, n, 6)
    η01 = - η00 * recβ(T, S, n, n, 1, 5) / recγ(T, S, n, n-1, 3)
    η11 = n == 1 ? 0 : - η10 * recα(T, S, n, n, 5) / recγ(T, S, n, n-1, 3)

    if clenshawalg
        ret = Array{R}(undef, (1, 2n-1))

        if n == 1
            # Special Case
            k = n - 1
            ret[1] = (ξ[1, 4] * (recβ(T, S, n, n, 1, 1) * η00 + recγ(T, S, n, k, 1) * η01)
                        + ξ[1, 1] * recγ(T, S, n, k, 1) / recγ(T, S, n, k, 3))
            return ret
        end

        # z
        ind = 1
        for k = 0:n-1, i = 0:min(1,k)
            ret[1,ind] = recγ(T, S, n, k, 1) * ξ[1,ind] / recγ(T, S, n, k, 3)
            ind += 1
        end
        ret[1,end-1] += recγ(T, S, n, n-1, 1) * η01 * ξ[1,end-1] # + as already defined
        ret[1,end] += recγ(T, S, n, n-1, 1) * η11 * ξ[1,end]
        # x
        ret[1,end] += recα(T, S, n, n, 1) * η10 * ξ[1,end]
        # y
        ret[1,end-1] += recβ(T, S, n, n, 1, 1) * η00 * ξ[1,end-1]
        ret
    else
        @assert length(ξ) == 2n - 1 "Invalid ξ vector"
        ret = zeros(R, 2n+3)
        if n == 1
            ret[1] = recγ(T, S, n, n-1, 1) / recγ(T, S, n, n-1, 3)
            ret[4] = recβ(T, S, n, n, 1, 1) * η00 + η01 * recγ(T, S, n, n-1, 1)
            return ξ[1] * ret
        end
        # z
        ind = 1
        for k = 0:n-1, i = 0:min(1,k)
            ret[ind] = recγ(T, S, n, k, 1) * ξ[ind] / recγ(T, S, n, k, 3)
            ind += 1
        end
        ret[end-1] = recγ(T, S, n, n-1, 1) * η01 * ξ[end-1]
        ret[end] = recγ(T, S, n, n-1, 1) * η11 * ξ[end]
        # x
        ret[end] += recα(T, S, n, n, 1) * η10 * ξ[end] # + as already defined
        # y
        ret[end-1] += recβ(T, S, n, n, 1, 1) * η00 * ξ[end-1]
        ret
    end
end
function clenshaw(cfs::AbstractVector{T}, S::SphericalCapSpace, x::R, y::R, z::R) where {T,R}
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
    ξ2 = view(f, Block(N+1))'
    ξ1 = view(f, Block(N))' - clenshawDTBmG(S, N-1, ξ2, x, y, z)
    for n = N-2:-1:0
        ξ = (view(f, Block(n+1))'
                - clenshawDTBmG(S, n, ξ1, x, y, z)
                - clenshawDTC(S, n+1, ξ2))
        ξ2 = copy(ξ1)
        ξ1 = copy(ξ)
    end
    (ξ1 * P0)[1]
end
clenshaw(cfs::AbstractVector, S::SphericalCapSpace, z) =
    clenshaw(cfs, S, z[1], z[2], z[3])
evaluate(cfs::AbstractVector, S::SphericalCapSpace, z) = clenshaw(cfs, S, z)
evaluate(cfs::AbstractVector, S::SphericalCapSpace, x, y, z) =
    clenshaw(cfs, S, x, y, z)

#===#
# Operator Clenshaw

# Operator Clenshaw
function operatorclenshawvector(S::SphericalCapSpace{<:Any, T, <:Any, <:Any},
                                v::AbstractArray{R},
                                id::BandedBlockBandedMatrix) where {T,R}
    s = size(v)[1]
    # B = Array{SparseMatrixCSC{T}}(undef, (1, s))
    B = Array{BandedBlockBandedMatrix{R}}(undef, (1, s))
    for i = 1:s
        B[1,i] = id * v[i]
    end
    B
end
# function operatorclenshawmatrixDT(S::SphericalCapSpace{<:Any, T, <:Any, <:Any}, A, id) where T
#     B = Array{SparseMatrixCSC{T}}(undef, size(A))
#     for ij = 1:length(A)
#         B[ij] = id * A[ij]
#     end
#     B
# end
# function operatorclenshawmatrixBmG(S::SphericalCapSpace{<:Any, T, <:Any, <:Any}, A, id, Jx, Jy, Jz) where T
#     ii, jj = size(A)
#     B = Array{SparseMatrixCSC{T}}(undef, (ii, jj))
#     for i = 1:jj, j = 1:jj
#         if i == j
#             B[i, j] = (id * A[i, j]) - Jx
#             B[i+jj, j] = (id * A[i+jj, j]) - Jy
#             B[i+2jj, j] = (id * A[i+2jj, j]) - Jz
#         else
#             B[i,j] = id * A[i, j]
#             B[i+jj, j] = id * A[i+jj, j]
#             B[i+2jj, j] = id * A[i+2jj, j]
#         end
#     end
#     B
# end
function operatorclenshaw(cfs::AbstractVector{T},
                            S::SphericalCapSpace,
                            M::Int,
                            Jx::BandedBlockBandedMatrix,
                            Jy::BandedBlockBandedMatrix,
                            Jz::BandedBlockBandedMatrix;
                            rotationalinvariant::Bool=false) where T
    # Outputs the operator MxM-blocked matrix operator corresponding to the
    # function f given by its coefficients of its expansion in the space S

    # rotationalinvariance flag: true ⟹ function is only dependent on z, and
    # so we can reduce the arithmatic significantly

    # @show "Operator Clenshaw"

    # Convert the cfs vector from ordered by Fourier mode k, to by degree n
    f = convertcoeffsvecorder(S, cfs)

    m = length(f)
    N = Int(sqrt(m)) - 1 # Degree of function
    resizedata!(S, N+1)
    f = PseudoBlockArray(f, [2n+1 for n=0:N])

    # identity matrix, requiredfor converting the coeffs vector into mat coeffs
    id = BandedBlockBandedMatrix(I, [M+1; 2M:-2:1], [M+1; 2M:-2:1], (0, 0), (0, 0))
    P0 = getdegzeropteval(T, S)

    # Special case
    if N == 0
        return f[1] * P0 * id
    end

    # Begin alg
    if rotationalinvariant
        # NOTE we assume now that the vector f only non-zeros at the first
        # entry of each block in the vector
        k = 0
        ξ2 = view(f, Block(N+1))[1] * id
        ξ1 = (ξ2 * (Jz - recγ(T, S, N-1, k, 2) * I) / recγ(T, S, N-1, k, 3)
                + view(f, Block(N))[1] * id)
        for n = N-2:-1:0
            # @show "Operator Clenshaw (rot inv)", M, N, n
            ξ = (ξ1 * (Jz - recγ(T, S, n, k, 2) * I) / recγ(T, S, n, k, 3)
                    - recγ(T, S, n+1, k, 1) * ξ2 / recγ(T, S, n+1, k, 3)
                    + view(f, Block(n+1))[1] * id)
            ξ2 = copy(ξ1)
            ξ1 = copy(ξ)
        end
        P0 * ξ1
    else
        ξ2 = operatorclenshawvector(S, view(f, Block(N+1)), id)
        ξ1 = (- clenshawDTBmG(S, N-1, ξ2, Jx, Jy, Jz; operator=true)
                + operatorclenshawvector(S, view(f, Block(N)), id))
        for n = N-2:-1:0
            @show "Operator Clenshaw", M, N, n
            ξ = (- clenshawDTBmG(S, n, ξ1, Jx, Jy, Jz; operator=true)
                 - clenshawDTC(S, n+1, ξ2)
                 + operatorclenshawvector(S, view(f, Block(n+1)), id))
            ξ2 = copy(ξ1)
            ξ1 = copy(ξ)
        end
        (ξ1 * P0)[1]
    end
end
function operatorclenshaw(cfs::AbstractVector{T},
                            S::SphericalCapSpace,
                            M::Int) where T
    @show "No Jacobi matrices input. Obtaining Jacobi matrices..."
    Jx = jacobix(S, M); @show "Jx done"
    Jy = jacobiy(S, M); @show "Jy done"
    Jz = jacobiz(S, M); @show "Jz done"
    operatorclenshaw(cfs, S, M, Jx, Jy, Jz)
end
operatorclenshaw(f::Fun, S::SphericalCapSpace) =
    operatorclenshaw(f.coefficients, S, getnki(S, ncoefficients(f))[1])
operatorclenshaw(f::Fun, S::SphericalCapSpace, N::Int) =
    operatorclenshaw(f.coefficients, S, N)


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
        indc = 1
        for k = 0:N
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
        indf = 1
        for k = 0:N
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



#===#
# Methods to return the spaces corresponding to applying the derivative operators

function differentiatespacephi(S::SphericalCapSpace; weighted=false, kind::Int=1)
    if kind == 1 # ρ(z)*∂/∂ϕ operator
        if weighted
            (S.family)((S.params[1]-1, S.params[2]))
        else
            (S.family)((S.params[1]+1, S.params[2]))
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

function getblockindex(S::SphericalCapSpace, n, k, i)
    @assert !(i == 1 && k == 0) "Invalid i value"
    if k == 0
        n + 1
    else
        2(n - k) + i + 1
    end
end

# ρ(z)*∂/∂ϕ operator
function getpartialphival(S::SphericalCapSpace, ptsr, rhoptsr2, rhodxrhoptsr,
                            wr, n::Int, m::Int, k::Int)
    Sp = differentiatespacephi(S)
    R = getRspace(S, k); Rp = getRspace(Sp, k)
    dRrho = (k * getptsevalforop(R, n-k) .* rhodxrhoptsr
                + getderivptsevalforop(R, n-k) .* rhoptsr2)
    ret = inner2(Rp, getptsevalforop(Rp, m-k) .* rhoptsr2.^k, dRrho, wr)
    - ret / getopnorm(Rp)
end
function getweightedpartialphival(S::SphericalCapSpace, ptsr, rhoptsr2,
                                    rhodxrhoptsr, wr10, wr, n::Int, m::Int, k::Int)
    Sp = differentiatespacephi(S; weighted=true)
    R = getRspace(S, k); Rp = getRspace(Sp, k)
    dRrho = (getptsevalforop(R, n-k) .* (k * rhodxrhoptsr .* wr10 + S.params[1] * rhoptsr2)
             + getderivptsevalforop(R, n-k) .* rhoptsr2 .* wr10)
    ret = inner2(Rp, getptsevalforop(Rp, m-k) .* rhoptsr2.^k, dRrho, wr)
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
    resizedataonedimops!(S, N+4)
    resizedataonedimops!(Sp, N+4)
    # Get pts and weights, and set the R norms
    # TODO how many pts needed?
    ptsr, wr = pointswithweights(B, getRspace(Sp, 0), N+4) # R^{(a±1, 1)}
    getopnorms(Sp, N)

    # Evaluate ρ.(ptsr) dρ/dx.(ptsr) at the R inner product points
    rhoptsr2 = S.family.ρ.(ptsr).^2
    rhodxrhoptsr = S.family.ρ.(ptsr) .* differentiate(S.family.ρ).(ptsr)

    # TODO sort out the blocks and stuff of this, and the allocation of val
    if weighted
        band1 = 1
        band2 = 2
        wr10 = (ptsr .- S.family.α)
    else
        band1 = 2
        band2 = 1
    end
    A = BandedBlockBandedMatrix(Zeros{B}((N+band2+1)^2, (N+1)^2),
                                [N+band2+1; 2(N+band2):-2:1], [N+1; 2N:-2:1],
                                (0, 0), (2band2, 2band1))
    for k = 0:N
        if k % 20 == 0
            @show "dϕ", weighted, k
        end
        R = getRspace(S, k); Rp = getRspace(Sp, k)
        getopptseval(R, N-k, ptsr); getopptseval(Rp, N+band2-k, ptsr)
        getderivopptseval(R, N-k, ptsr)
        for n = k:N, m = max(0, n-band1):n+band2 # min(n+band2, N)
            if k ≤ m
                if weighted
                    val = getweightedpartialphival(S, ptsr, rhoptsr2, rhodxrhoptsr, wr10, wr, n, m, k)
                else
                    val = getpartialphival(S, ptsr, rhoptsr2, rhodxrhoptsr, wr, n, m, k)
                end
                for i = 0:min(1,k)
                    view(A, Block(k+1, k+1))[getblockindex(S, m, k, i), getblockindex(S, n, k, i)] = val
                end
            end
        end
        resetopptseval(R); resetopptseval(Rp)
        resetderivopptseval(R)
    end
    A
end

# ∂/∂θ operator
function getpartialthetaval(S::SphericalCapSpace, k::Int, i::Int)
    (-1)^(i+1) * k
end
function diffoperatortheta(S::SphericalCapSpace{<:Any, B, T, <:Any}, N::Int;
                            weighted=false) where {B,T}
    # ∂/∂θ operator
    # Takes the space:
    #   Q^{a,b} -> Q^{a,b} or
    #   w_R^{(a,0)} Q^{a,b} -> w_R^{(a,0)} Q^{a,b}
    # NOTE the weighted keyword here doesnt affect the resulting operator
    A = BandedBlockBandedMatrix(Zeros{B}((N+1)^2, (N+1)^2),
                                [N+1; 2N:-2:1], [N+1; 2N:-2:1],
                                (0, 0), (1, 1))
    for k = 1:N, n = k:N
        ind = getblockindex(S, n, k, 0)
        view(A, Block(k+1, k+1))[ind+1, ind] = getpartialthetaval(S, k, 0)
        view(A, Block(k+1, k+1))[ind, ind+1] = getpartialthetaval(S, k, 1)
    end
    A
end
# ∂²/∂θ² operator
function diffoperatortheta2(S::SphericalCapSpace{<:Any, B, T, <:Any}, N::Int;
                            weighted=false) where {B,T}
    # ∂²/∂θ² operator
    # Takes the space:
    #   Q^{a,b} -> Q^{a,b} or
    #   w_R^{(a,0)} Q^{a,b} -> w_R^{(a,0)} Q^{a,b}
    # NOTE the weighted keyword here doesnt affect the resulting operator
    A = BandedBlockBandedMatrix(Zeros{B}((N+1)^2, (N+1)^2),
                                [N+1; 2N:-2:1], [N+1; 2N:-2:1],
                                (0, 0), (0, 0))
    for k = 1:N
        view(A, Block(k+1, k+1)) .= Diagonal(ones(2(N-k+1)) * (-k^2))
    end
    A
end


#===#
# Parameter Transform/Conversion operator matrices

function gettransformparamsval(S::SphericalCapSpace, St::SphericalCapSpace,
                                rhoptsr2, wr, n::Int, m::Int, k::Int)
    R = getRspace(S, k); Rt = getRspace(St, k)
    ret = inner2(R, getptsevalforop(R, n-k),
                    rhoptsr2.^(k) .* getptsevalforop(Rt, m-k), wr)
    ret / getopnorm(Rt)
end
function transformparamsoperator(S::SphericalCapSpace{<:Any, B, T, <:Any},
            St::SphericalCapSpace{<:Any, B, T, <:Any}, N;
            weighted=false) where {B,T}

    # The St space is the target space. Applying this operator to coeffs in the
    # (weighted) S space will result in coeffs in the (weighted) St space.

    resizedataonedimops!(S, N+1)
    resizedataonedimops!(St, N+1)

    if weighted
        band1 = 0
        band2 = Int(S.params[1] - St.params[1])
        ptsr, wr = pointswithweights(B, getRspace(S, 0), N+1) # TODO check npts
    else
        band1 = Int(St.params[1] - S.params[1])
        band2 = 0
        ptsr, wr = pointswithweights(B, getRspace(St, 0), N+1) # TODO check npts
    end
    C = BandedBlockBandedMatrix(Zeros{B}((N+band2+1)^2, (N+1)^2),
                                [N+band2+1; 2(N+band2):-2:1], [N+1; 2N:-2:1],
                                (0, 0), (2band2, 2band1))
    rhoptsr2 = S.family.ρ.(ptsr).^2
    getopnorms(St, N+band2+1)
    for k = 0:N
        if k % 100 == 0
            @show "trnsfrm (a,b)->(a±1,b)", weighted, k
        end
        R = getRspace(S, k); Rt = getRspace(St, k)
        getopptseval(R, N-k, ptsr); getopptseval(Rt, N-k+band2, ptsr)
        for n = k:N, m = max(0, n-band1):n+band2 # min(n+band2, N)
            if k ≤ m
                val = gettransformparamsval(S, St, rhoptsr2, wr, n, m, k)
                for i = 0:min(1,k)
                    view(C, Block(k+1, k+1))[getblockindex(St, m, k, i), getblockindex(S, n, k, i)] = val
                end
            end
        end
        resetopptseval(R); resetopptseval(Rt)
    end
    C
end
function convertweightedtononweightedoperator(S::SphericalCapSpace, N; Nout=-1)
    # Converts coeffs in W^{a} to coeffs in Q^{a}
    # If Nout is set, then we increase the resulting dimension so that the
    # operator outputs coeffs of degree Nout (probably to be used for Helholtz
    # type equations, to match dimension of rhoLaplacian, in which case we should
    # have Nout=N+3)
    S0 = S.family(0.0, 0.0)
    T = transformparamsoperator(S0, S, N+1; weighted=false)
    Tw = transformparamsoperator(S, S0, N; weighted=true)
    if Nout > 0 # ie is set
        if Nout <= N
            error("invalid Nout value given - must be bigger than N")
        else
            increasedegreeoperator(S, N+1, Nout) * T * Tw
        end
    else
        T * Tw
    end
end
# The operator for ρ^2
function rho2operator(S::SphericalCapSpace{<:Any, B, T, <:Any},
                        St::SphericalCapSpace{<:Any, B, T, <:Any}, N::Int;
                        weightedin=true, weightedout=false, square=false) where {B,T}
    # This operator can also convert the param spaces, by inputing valid entry
    # and tangent spaces and weihgtedin/out keywords
    (!weightedin && weightedout) && error("invalid weighted keywords")

    band1 = band2 = 2
    P = BandedBlockBandedMatrix(Zeros{B}((N+band2+1)^2, (N+1)^2),
                                [N+band2+1; 2(N+band2):-2:1], [N+1; 2N:-2:1],
                                (0, 0), (2band2, 2band1))

    resizedataonedimops!(S, N+band2)

    # Get the operator for mult by ρ²
    ptsr, wr = pointswithweights(B, getRspace(S, 0), N+3)
    rhoptsr2 = S.family.ρ.(ptsr).^2
    getopnorms(St, N+band2+1)
    for k = 0:N
        R = getRspace(S, k)
        getopptseval(R, N-k+band2, ptsr)
        for n = k:N, m = max(0, n-band1):n+band2 # min(n+band2, N)
            if k ≤ m
                val = inner2(R, getptsevalforop(R, n-k), getptsevalforop(R, m-k) .* rhoptsr2.^(k+1), wr)
                for i = 0:min(1,k)
                    view(P, Block(k+1, k+1))[getblockindex(S, m, k, i), getblockindex(S, n, k, i)] = val / getopnorm(R)
                end
            end
        end
        resetopptseval(R)
    end
    if weightedin && !weightedout
        S0 = S.family(S.params .* 0)
        tw = transformparamsoperator(S, S0, N+2; weighted=weightedin)
        t = transformparamsoperator(S0, St, N+3; weighted=weightedout)
        t * tw * P
    else
        t = transformparamsoperator(S, St, N+2; weighted=weightedout)
        t * P
    end

    # rho2 = operatorclenshaw(Fun((x,y,z)->(1-z^2), S, 10), S, N+1)
    # if weightedin && !weightedout
    #     S0 = S.family(S.params .* 0)
    #     Tw = transformparamsoperator(S, S0, N; weighted=weightedin)
    #     T = transformparamsoperator(S0, St, N+1; weighted=weightedout)
    #     band2 = band1 = 1
    #     C = BandedBlockBandedMatrix(rho2 * sparse(T * Tw),
    #                                 ([N+band2+1; 2(N+band2):-2:1], [N+1; 2N:-2:1]),
    #                                 (0, 0), (2band2, 2band1))
    # else
    #     T = transformparamsoperator(S, St, N; weighted=weightedout)
    #     band2 = 0; band1 = 1
    #     BandedBlockBandedMatrix(rho2 * sparse(T),
    #                                 ([N+band2+1; 2(N+band2):-2:1], [N+1; 2N:-2:1]),
    #                                 (0, 0), (2band2, 2band1))
    # end
end


#===#
# "Coeffs degree increaser" operator matrix

function increasedegreeoperator(S::SphericalCapSpace{<:Any, B, T, <:Any}, N, Nto;
                                weighted=false) where {B,T}
    # This operator acts on the coeffs vector of a Fun in the space S to just
    # reorganise the coeffs so that the length is increased from deg N to
    # deg Nto, with all coeffs representing the extra degrees being zero.

    @assert Nto > N "Degree is not being increased (Nto shuld be > N)"

    C = BandedBlockBandedMatrix(Zeros{B}((Nto+1)^2, (N+1)^2),
                                [Nto+1; 2(Nto):-2:1], [N+1; 2N:-2:1],
                                (0, 0), (0, 0))
    @show "incr deg", N, Nto, weighted
    for k = 0:N, n = k:N
        for i = 0:min(1,k)
            view(C, Block(k+1, k+1))[getblockindex(S, n, k, i), getblockindex(S, n, k, i)] = 1.0
        end
    end
    C
end


#===#
# Spherical Laplacian operator matrix and Biharmonic matrix

function getlaplacianval(S::SphericalCapSpace{<:Any, B, T, <:Any},
                            St::SphericalCapSpace{<:Any, B, T, <:Any},
                            ptsr::Vector{B}, wr::Vector{B}, rhoptsr2::Vector{B},
                            w10ptsr::Vector{B}, n::Int, m::Int, k::Int;
                            weighted::Bool=false, specialcase::Bool=false) where {B,T}
    # specialcase == true if S.a == 1 == St.a and S is a weighted space

    a = Int(S.params[1])
    rhoptsr2k = rhoptsr2.^k
    rhoptsr2kp2 = rhoptsr2k .* rhoptsr2
    if specialcase
        if !weighted
            error("specialcase true only if weighted true")
        end
        R = getRspace(S, k)
        ret = - inner2(R, getptsevalforop(R, n-k) .* getptsevalforop(R, m-k) .* rhoptsr2k,
                        (2(k+1) * ptsr .+ k*(k+1) * w10ptsr),
                        wr)
        ret -= inner2(R, w10ptsr .* getderivptsevalforop(R, n-k),
                        getderivptsevalforop(R, m-k) .* rhoptsr2kp2,
                        wr)
        ret / getopnorm(R)
    elseif weighted
        ã = a - Int(St.params[1])
        R = getRspace(S, k); Rt = getRspace(St, k)
        ret = inner2(Rt, getptsevalforop(R, n-k) .* getptsevalforop(Rt, m-k),
                        (-k * (k+1) * w10ptsr.^ã
                            - 2 * a * (k+1) * ptsr .* w10ptsr.^(ã-1)
                            + a * (a-1) * rhoptsr2 .* w10ptsr.^(ã-2)) .* rhoptsr2k,
                        wr)
        ret += inner2(Rt, a * getderivptsevalforop(R, n-k) .* w10ptsr.^(ã-1),
                        getptsevalforop(Rt, m-k) .* rhoptsr2kp2,
                        wr)
        ret -= inner2(Rt, getderivptsevalforop(R, n-k) .* w10ptsr.^ã,
                        getderivptsevalforop(Rt, m-k) .* rhoptsr2kp2,
                        wr)
        ret / getopnorm(Rt)
    else
        ã = Int(St.params[1]) - a
        R = getRspace(S, k); Rt = getRspace(St, k)
        ret = - k * (k+1) * inner2(R, getptsevalforop(R, n-k) .* rhoptsr2k,
                                    getptsevalforop(Rt, m-k) .* w10ptsr.^ã,
                                    wr)
        ret -= (a + ã) * inner2(R, getderivptsevalforop(R, n-k) .* rhoptsr2kp2,
                                    getptsevalforop(Rt, m-k) .* w10ptsr.^(ã-1),
                                    wr)
        ret -= inner2(R, getderivptsevalforop(R, n-k) .* rhoptsr2kp2,
                        getderivptsevalforop(Rt, m-k) .* w10ptsr.^ã,
                        wr)
        ret / getopnorm(Rt)
    end
end
# The operator for spherical laplacian Δ_s
function laplacianoperator(S::SphericalCapSpace{<:Any, B, T, <:Any},
                            St::SphericalCapSpace{<:Any, B, T, <:Any}, N::Int;
                            weighted=true, square=false) where {B,T}
    # (1/ρ^2)*∂/∂θ + (1/ρ)*∂/∂ϕ(ρ*∂/∂ϕ) = Δ_s operator
    # Takes the space:
    #   w_R^{(1,0)} Q^{1} -> Q^{1}  OR
    #   w_R^{(a,0)} Q^{a} -> w_R^{(a-ã,0)} Q^{a-ã}, where ã≥2, a≥ã  OR
    #   Q^{a} -> Q^{a+ã}, where ã≥2.

    # TODO how many pts needed?

    @assert (Int(S.params[1]) ≥ 0 && Int(St.params[1]) ≥ 0) "Invalid SphericalCapSpace"

    # Determine which case we have, and set the appropriate objects
    specialcase = false
    if Int(S.params[1]) == 1 && weighted && Int(St.params[1]) == 1 # Special case
        band1, band2 = 1, 1
        Rpts = getRspace(S, 0)
        specialcase = true
    elseif weighted && Int(S.params[1]) - Int(St.params[1]) == 2 # Could be ≥ 2
        ã = Int(S.params[1]) - Int(St.params[1])
        band1, band2 = 0, ã
        Rpts = getRspace(St, 0)
    elseif !weighted && Int(St.params[1]) - Int(S.params[1]) == 2 # Could be ≥ 2
        ã = Int(St.params[1]) - Int(S.params[1])
        band1, band2 = ã, 0
        Rpts = getRspace(S, 0)
    else
        error("Invalid SphericalCapSpace")
    end

    # Create matrix
    bandm = square ? 0 : band2
    A = BandedBlockBandedMatrix(Zeros{B}((N+bandm+1)^2, (N+1)^2),
                                [N+bandm+1; 2(N+bandm):-2:1], [N+1; 2N:-2:1],
                                (0, 0), (2band2, 2band1))

    # Gather data
    resizedataonedimops!(S, N+5)
    resizedataonedimops!(St, N+5)
    getopnorms(St, N+band2)
    ptsr, wr = pointswithweights(B, Rpts, 2N+4)

    # Evaluate ρ.(ptsr).^2 at the R inner product points
    rhoptsr2 = S.family.ρ.(ptsr).^2
    w10ptsr = ptsr .- S.family.α

    # Begin assigning entries
    for k = 0:N
        if k % 20 == 0
            @show "Δ", weighted, k
        end
        R = getRspace(S, k)
        getopptseval(R, N-k+band2, ptsr)
        getderivopptseval(R, N-k+band2, ptsr)
        if !specialcase
            Rt = getRspace(St, k)
            getopptseval(Rt, N-k+band2, ptsr)
            getderivopptseval(Rt, N-k+band2, ptsr)
        end
        for n = k:N, m = max(0, n-band1):min(n+band2, N+bandm)
            if k ≤ m
                val = getlaplacianval(S, St, ptsr, wr, rhoptsr2, w10ptsr, n, m, k;
                                        weighted=weighted, specialcase=specialcase)
                for i = 0:min(1,k)
                    view(A, Block(k+1, k+1))[getblockindex(St, m, k, i), getblockindex(S, n, k, i)] = val
                end
            end
        end
        resetopptseval(R); resetderivopptseval(R)
        if !specialcase
            resetopptseval(Rt); resetderivopptseval(Rt)
        end
    end
    A
end

function biharmonicoperator(S2::SphericalCapSpace{<:Any, B, T, <:Any}, N::Int;
                            weighted::Bool=true, square::Bool=false) where {B,T}
    @assert (Int(S2.params[1]) == 2 && weighted) "Invalid SphericalCapSpace" # We could have any number over 1
    S0 = S2.family(S2.params .* 0)
    Δ = laplacianoperator(S0, S2, N+2; weighted=false)
    Δw = laplacianoperator(S2, S0, N; weighted=true)
    bihar = Δ * Δw
    if square
        squareoperator(S2, bihar, N)
    else
        bihar
    end
end

function squareoperator(S::SphericalCapSpace{<:Any, B, T, <:Any},
                        A::BandedBlockBandedMatrix, N::Int) where {B,T}
    """ this is to square the given operator, by reassigning the non-zero
        entries correctly for given N value
    """
    @assert (N < blocksize(A)[1] - 1 && N == blocksize(A)[2] - 1) "Invalid size of A or incorrect N val"
    C = BandedBlockBandedMatrix(Zeros{B}((N+1)^2, (N+1)^2),
                                [N+1; 2N:-2:1], [N+1; 2N:-2:1],
                                (A.l, A.u), (A.λ, A.μ))
    for k = 1:N
        inds = 1:2(N-k+1)
        view(C, Block(k+1, k+1)) .= view(A, Block(k+1, k+1))[inds, inds]
    end
    k = 0
    view(C, Block(k+1, k+1)) .= view(A, Block(k+1, k+1))[1:N+1, 1:N+1]
    C
end

# The operator for ρ²Δ # NOTE OBSOLETE
function rho2laplacianoperator(S::SphericalCapSpace, St::SphericalCapSpace, N::Int;
                                weighted=true, square=false)
    # Outputs the sum(1:N+1) × sum(1:N+1) matrix operator if square=true
    # TODO fix the square=true cases
    D = S.family
    paramin = Int(S.params[1]); paramout = Int(St.params[1])
    if weighted
        if paramin == paramout
            A = diffoperatorphi(differentiatespacephi(S; weighted=true), N+2)
            @show "laplaceoperator", "1 of 6 done"
            B = diffoperatorphi(S, N; weighted=true)
            @show "laplaceoperator", "2 of 6 done"
            C = transformparamsoperator(differentiatespacephi(S; weighted=true), S, N+1)
            @show "laplaceoperator", "3 of 6 done"
            E = transformparamsoperator(S, differentiatespacephi(S; weighted=true), N; weighted=true)
            @show "laplaceoperator", "4 of 6 done"
            F = diffoperatortheta2(S, N)
            @show "laplaceoperator", "5 of 6 done"
            G = increasedegreeoperator(S, N+1, N+3)
            @show "laplaceoperator", "6 of 6 done"
            L = A * B + G * C * E * F
            if square
                m = (N+1)^2
                Δ = BandedBlockBandedMatrix(L[1:m, 1:m], [N+1; 2N:-2:1], [N+1; 2N:-2:1],
                                            (L.l, L.u), (L.λ, L.μ))
            else
                L
            end
        elseif  paramin - 2 == paramout
            A = diffoperatorphi(differentiatespacephi(S; weighted=true), N+2; weighted=true)
            B = diffoperatorphi(S, N; weighted=true)
            C = transformparamsoperator(S, St, N; weighted=true)
            E = diffoperatortheta2(S, N)
            F = increasedegreeoperator(S, N+2, N+4)
            L = A * B + F * C * E
            if square
                m = (N+1)^2
                Δ = BandedBlockBandedMatrix(L[1:m, 1:m], [N+1; 2N:-2:1], [N+1; 2N:-2:1],
                                            (L.l, L.u), (L.λ, L.μ))
            else
                L
            end
        else
            error("Invalid spaces for weighted keyword")
        end
    else
        if paramin + 2 == paramout
            A = diffoperatorphi(differentiatespacephi(S; weighted=false), N+1; weighted=false)
            B = diffoperatorphi(S, N; weighted=false)
            C = transformparamsoperator(S, St, N; weighted=false)
            E = diffoperatortheta2(S, N)
            F = increasedegreeoperator(S, N, N+2)
            L = A * B + F * C * E
            if square
                m = (N+1)^2
                Δ = BandedBlockBandedMatrix(L[1:m, 1:m], [N+1; 2N:-2:1], [N+1; 2N:-2:1],
                                            (L.l, L.u), (L.λ, L.μ))
            else
                L
            end
        else
            error("Invalid spaces for weighted keyword")
        end
    end
end

#===#
# Grad operator matrix

#= NOTE: The tangent space basis we use is
        Φ̲_{n,k,i} = ϕ̲ * Q^{0}_{n,k,i}
        Ψ̲_{n,k,i} = θ̲ * Q^{0}_{n,k,abs(i-1)}
    Let u̲ be a function that is a sum of gradients and perp gradients (hence
    lies in the tangent space). Then we can expand ρu̲ in this basis, i.e.
        ρu̲ = Σ u_{n,k,i}\^Φ * Φ̲_{n,k,i} + u_{n,k,i}\^Ψ Ψ̲_{n,k,i}
    where {u\^Φ, u\^Φ} are coefficients. The coefficients vector is ordered
    as follows:
    i.e. for a given max degree N:

        𝕋_N := [𝕋_{N,0};...;𝕋_{N,N}]
        𝕋_{N,k} := [Φ̲_{k,k,0};Ψ̲_{k,k,0};Φ̲_{k,k,1};Ψ̲_{k,k,1}...;Φ̲_{N,k,0};Ψ̲_{N,k,0};Φ̲_{N,k,1};Ψ̲_{N,k,1}] ∈ ℝ^{4(N-k+1)}
                                                                for k = 1,...,N
        𝕋_{N,0} := [Φ̲_{0,0,0};Ψ̲_{0,0,0};Φ̲_{1,0,0};Ψ̲_{1,0,0};...;Φ̲_{N,0,0};Ψ̲_{N,0,0}] ∈ ℝ^{2(N+1)}
                                                                for k = 0

    Note also that
        ρ²∇.u̲ = ρ∇.(ρu̲) - ρu̲.∇ρ = Σ u\^Φ ρ ∂/∂ϕ (Q^{0}_{n,k,i}) + u\^Ψ ∂/∂θ (Q^{0}_{n,k,abs(i-1)})
                                = Σ ũ Q^{1}_{n,k,i}
    where {ũ} are coefficients.

    We output these coefficients {ũ}, for expansion in the ℚ^{1} basis.
=#

function getblockindextangent(S::SphericalCapSpace, n::Int, k::Int, i::Int, j::Int)
    # j refers to either 0 (Φ) or 1 (Ψ)
    if k == 0
        2 * (n + 1) - 1 + j
    else
        4 * (n - k) + 2i + j + 1
    end
end
# Grad operator (ρ∇)
function getrhogradval(S::SphericalCapSpace{<:Any, B, T, <:Any}, ptsr, rhoptsr2,
                        rhodxrhoptsr, wr10, wr, n, m, k, i, j;
                        weighted=true) where {B,T}
    if j == 0
        if weighted
            ret = getweightedpartialphival(S, ptsr, rhoptsr2, rhodxrhoptsr, wr10,
                                            wr, n, m, k)
        else
            ret = getpartialphival(S, ptsr, rhoptsr2, rhodxrhoptsr, wr, n, m, k)
        end
    elseif j == 1
        St = differentiatespacephi(S; weighted=weighted)
        if weighted
            R = getRspace(S, k); Rt = getRspace(St, k)
            ret = inner2(Rt, getptsevalforop(R, n-k) .* rhoptsr2.^k,
                            getptsevalforop(Rt, m-k) .* wr10, wr)
            ret /= getopnorm(Rt)
        else
            if k == 0
                return B(0)
            else
                ret = gettransformparamsval(S, St, rhoptsr2, wr, n, m, k)
            end
        end
        ret *= k * (-1)^(i+1)
    else
        error("invalid param j")
    end
    ret
end
function rhogradoperator(S::SphericalCapSpace{<:Any, B, T, <:Any}, N;
                            weighted=true) where {B,T}
    # Operator acts on coeffs in the space 𝕎^{1}/ℚ^{1}, and results in
    # coefficients {u\^Φ, u\^Φ}, for expansion in the 𝕋^{0}/𝕋^{2} basis
    @assert Int(S.params[1]) == 1 "Invalid SCSpace"
    St = differentiatespacephi(S; weighted=weighted)
    resizedataonedimops!(S, N+1)
    resizedataonedimops!(St, N+1)

    Rt = getRspace(St, 0)
    ptsr, wr = pointswithweights(B, Rt, N+4)
    rhoptsr2 = S.family.ρ.(ptsr).^2
    rhodxrhoptsr = S.family.ρ.(ptsr) .* differentiate(S.family.ρ).(ptsr)
    wr10 = ptsr .- S.family.α # Only used when weighted=true
    if weighted
        band1 = 1
        band2 = 2
    else
        band1 = 2
        band2 = 1
    end
    A = BandedBlockBandedMatrix(Zeros{B}(2 * (N+band2+1)^2, (N+1)^2),
                                2 * [N+band2+1; 2(N+band2):-2:1], [N+1; 2N:-2:1],
                                (0, 0), (2N + 4band2, 2band1))
    for k = 0:N
        if k % 100 == 0
            @show "rhograd", k
        end
        R1 = getRspace(S, k); Rt = getRspace(St, k)
        getopptseval(R1, N-k, ptsr); getopptseval(Rt, N-k+band2, ptsr)
        getderivopptseval(R1, N-k, ptsr)
        for n = k:N, m = max(0, n-band1):n+band2 # min(n+band2, N)
            if k ≤ m
                for i = 0:min(1,k), j = 0:1
                    (j == 1 && (m < n - (band1 - 1) || m > n + (band2 - 1) || k == 0)
                            && continue)
                    val = getrhogradval(S, ptsr, rhoptsr2, rhodxrhoptsr, wr10,
                                        wr, n, m, k, i, j; weighted=weighted)
                    ind1 = getblockindextangent(St, m, k, abs(i-j), j)
                    ind2 = getblockindex(S, n, k, i)
                    view(A, Block(k+1, k+1))[ind1, ind2] = val
                end
            end
        end
        resetopptseval(R1); resetopptseval(Rt); resetderivopptseval(R1)
    end
    A
end


#===#
# Mult by ∇ρ operator matrix
function gradrhooperator(S::SphericalCapSpace{<:Any, B, T, <:Any}, N) where {B,T}
    # Operator acts on coeffs in the non-weighted space ℚ^{1}, and results in
    # coefficients {u\^Φ, u\^Φ}, for expansion in the 𝕋^{2} basis
    @assert Int(S.params[1]) == 1 "Invalid SCSpace"
    S2 = differentiatespacephi(S; weighted=false)

    band1 = 2
    band2 = 1
    A = BandedBlockBandedMatrix(Zeros{B}(2 * (N+band2+1)^2, (N+1)^2),
                                2 * [N+band2+1; 2(N+band2):-2:1], [N+1; 2N:-2:1],
                                (0, 0), (2N + 4band2 + 1, 2band1))

    resizedataonedimops!(S, N+1)
    resizedataonedimops!(S2, N+1)
    getopnorms(S2, N+band2)
    R2 = getRspace(S2, 0)
    ptsr, wr = pointswithweights(B, R2, N+4)
    rhoptsr2 = S.family.ρ.(ptsr).^2
    j = 0 # No θ̲̂ components
    for k = 0:N
        if k % 100 == 0
            @show "gradrho", k
        end
        R1 = getRspace(S, k); R2 = getRspace(S2, k)
        getopptseval(R1, N-k, ptsr); getopptseval(R2, N-k+band2, ptsr)
        for n = k:N, m = max(0, n-band1):n+band2 # min(n+band2, N)
            if k ≤ m
                for i = 0:min(1,k)
                    # TODO check indexing here
                    val = inner2(R2, getptsevalforop(R1, n-k) .* ptsr,
                                     getptsevalforop(R2, m-k) .* rhoptsr2.^k,
                                     wr) / getopnorm(R2)
                    view(A, Block(k+1, k+1))[getblockindextangent(S2, m, k, i, j), getblockindex(S, n, k, i)] = val
                end
            end
        end
        resetopptseval(R1); resetopptseval(R2)
    end
    A
end


#===#
# Uselful for SWE
function unitvecphioperator(S::SphericalCapSpace{<:Any, B, T, <:Any}, N) where {B,T}
    # Assume S is S0
    @assert Int(S.params[1]) == 0 "Invalid SCSpace"
    V = BandedBlockBandedMatrix(Zeros{B}(2(N+1)^2, (N+1)^2),
                                2 * [N+1; 2N:-2:1], [N+1; 2N:-2:1],
                                (0, 0), (2N - 1, 0))
    j = 0
    for k = 0:N, n = k:N
        for i = 0:min(1,k)
            view(V, Block(k+1, k+1))[getblockindextangent(S, n, k, i, j),
                                     getblockindex(S, n, k, i)] = B(1)
        end
    end
    V
end



#===#
# inner product

function inner(::Type{T}, S::SphericalCapSpace, fpts, gpts, w) where T
    m = length(w)
    T(sum((fpts[1:m] .* gpts[1:m] + fpts[m+1:end] .* gpts[m+1:end]) .* w) / 2)
end
inner(S::SphericalCapSpace, fpts, gpts, w::AbstractVector{T}) where T =
    inner(T, S, fpts, gpts, w)


#===#
# Jacobi operators methods for mult by x, y, z

function jacobix(S::SphericalCapSpace{<:Any, B, T, <:Any}, N) where {B,T}
    # Transposed to act on coeffs vec
    resizedata!(S, N; jacobimatcall=true)
    J = BandedBlockBandedMatrix(Zeros{B}((N+1)^2, (N+1)^2),
                                1:2:2N+1, 1:2:2N+1, (1, 1), (2, 2))
    j = 1 # x
    n = 0
    view(J, Block(n+1, n+1))[:,:] = S.B[n+1][j]'
    for n = 1:N
        view(J, Block(n, n+1))[:,:] = S.C[n+1][j]'
        view(J, Block(n+1, n+1))[:,:] = S.B[n+1][j]'
        view(J, Block(n+1, n))[:,:] = S.A[n][j]'
    end
    J
    bandn = bandk = 1; bandi = 0
    Jout = BandedBlockBandedMatrix(Zeros{B}((N+1)^2, (N+1)^2),
                                    [N+1; 2N:-2:1], [N+1; 2N:-2:1],
                                    (bandk, bandk),
                                    (max(4, N-1), max(4, N-1)))
    for row = 1:(N+1)^2
        n, k, i = getnki(S, row)
        ind = getopindex(S, n, k, i; bydegree=false, N=N)
        Jout[ind, :] = convertcoeffsvecorder(S, J[row, :]; todegree=false)
    end
    Jout
end
function jacobiy(S::SphericalCapSpace{<:Any, B, T, <:Any}, N) where {B,T}
    resizedata!(S, N; jacobimatcall=true)
    J = BandedBlockBandedMatrix(Zeros{B}((N+1)^2, (N+1)^2),
                                1:2:2N+1, 1:2:2N+1, (1, 1), (3, 3))
    j = 2 # y
    n = 0
    view(J, Block(n+1, n+1))[:,:] = S.B[n+1][j]'
    for n = 1:N
        view(J, Block(n, n+1))[:,:] = S.C[n+1][j]'
        view(J, Block(n+1, n+1))[:,:] = S.B[n+1][j]'
        view(J, Block(n+1, n))[:,:] = S.A[n][j]'
    end
    J
    bandn = bandk = bandi = 1
    Jout = BandedBlockBandedMatrix(Zeros{B}((N+1)^2, (N+1)^2),
                                    [N+1; 2N:-2:1], [N+1; 2N:-2:1],
                                    (bandk, bandk),
                                    (max(5, N), max(5, N)))
    for row = 1:(N+1)^2
        n, k, i = getnki(S, row)
        ind = getopindex(S, n, k, i; bydegree=false, N=N)
        Jout[ind, :] = convertcoeffsvecorder(S, J[row, :]; todegree=false)
    end
    Jout
end
function jacobiz(S::SphericalCapSpace{<:Any, B, T, <:Any}, N::Int) where {B,T}
    # NOTE in this method, we construct the operator directly (as its simple)
    # so should be quick
    resizedataonedimops!(S, N)
    bandn = 1; bandk = bandi = 0
    J = BandedBlockBandedMatrix(Zeros{B}((N+1)^2, (N+1)^2),
                                [N+1; 2N:-2:1], [N+1; 2N:-2:1],
                                (bandk, bandk),
                                (2bandn+2bandk+bandi, 2bandn+2bandk+bandi))
    # Assign each block
    for k = 1:N-1
        vu = zeros(B, 2(N-k)); vd = zeros(2(N-k+1))
        n = k
        c2 = recγ(B, S, n, k, 2)
        c3 = recγ(B, S, n, k, 3)
        vd[1:2] = [c2; c2]
        vu[1:2] = [c3; c3]
        ind = 3
        for n = k+1:N-1
            # TODO Not sure how to do this without loop...
            c2 = recγ(B, S, n, k, 2)
            c3 = recγ(B, S, n, k, 3)
            vd[ind:ind+1] = [c2; c2]
            vu[ind:ind+1] = [c3; c3]
            ind += 2
        end
        n = N
        c2 = recγ(B, S, n, k, 2)
        vd[ind:ind+1] = [c2; c2]
        view(J, Block(k+1, k+1)) .= diagm(2 => vu, 0 => vd, -2 => vu)
    end
    # k = 0,N are special cases
    k = 0
    vu = [recγ(B, S, n, k, 3) for n = k:N-1]
    vd = [recγ(B, S, n, k, 2) for n = k:N]
    view(J, Block(k+1, k+1)) .= Tridiagonal(vu, vd, vu)
    k = N; n = k
    c2 = recγ(B, S, n, k, 2)
    view(J, Block(k+1, k+1)) .= [c2 0; 0 c2]
    J
end

# function jacobix(S::SphericalCapSpace{<:Any, B, T, <:Any}, N) where {B,T}
#     # Transposed operator, so acts directly on coeffs vec
#     resizedata!(S, N)
#     getAs!(S, N+1, length(S.A))
#     J = BandedBlockBandedMatrix(Zeros{B}((N+1)^2, (N+1)^2),
#                                 (1:2:2N+1, 1:2:2N+1), (1, 1), (2, 2))
#     # Assign by column
#     n = 0
#     inds = 1:2n+1
#     view(J, Block(n+1, n+1)) .= S.B[n+1][inds, :]'
#     view(J, Block(n+2, n+1)) .= S.A[n+1][inds, :]'
#     for n = 1:N-1
#         inds = 1:2n+1
#         view(J, Block(n, n+1)) .= S.C[n+1][inds, :]'
#         view(J, Block(n+1, n+1)) .= S.B[n+1][inds, :]'
#         view(J, Block(n+2, n+1)) .= S.A[n+1][inds, :]'
#     end
#     n = N
#     inds = 1:2n+1
#     view(J, Block(n, n+1)) .= S.C[n+1][inds, :]'
#     view(J, Block(n+1, n+1)) .= S.B[n+1][inds, :]'
#     J
#     # The following converts the operator from bydegree to byfouriermode ordering
#     # TODO improve this
#     Jout = spzeros(B, (N+1)^2, (N+1)^2)
#     for row = 1:(N+1)^2
#         n, k, i = getnki(S, row)
#         ind = getopindex(S, n, k, i; bydegree=false, N=N)
#         Jout[ind, :] = convertcoeffsvecorder(S, J[row, :]; todegree=false)
#     end
#     bandn = bandk = 1; bandi = 0
#     BandedBlockBandedMatrix(Jout, ([N+1; 2N:-2:1], [N+1; 2N:-2:1]),
#                             (bandk, bandk),
#                             (2bandn+2bandk+bandi, 2bandn+2bandk+bandi))
# end
# function jacobiy(S::SphericalCapSpace{<:Any, B, T, <:Any}, N) where {B,T}
#     # Transposed operator, so acts directly on coeffs vec
#     resizedata!(S, N)
#     J = BandedBlockBandedMatrix(Zeros{B}((N+1)^2, (N+1)^2),
#                                 (1:2:2N+1, 1:2:2N+1), (1, 1), (3, 3))
#     # Assign by column
#     n = 0
#     inds = 2n+2:2(2n+1)
#     view(J, Block(n+1, n+1)) .= S.B[n+1][inds, :]'
#     view(J, Block(n+2, n+1)) .= S.A[n+1][inds, :]'
#     for n = 1:N-1
#         inds = 2n+2:2(2n+1)
#         view(J, Block(n, n+1)) .= S.C[n+1][inds, :]'
#         view(J, Block(n+1, n+1)) .= S.B[n+1][inds, :]'
#         view(J, Block(n+2, n+1)) .= S.A[n+1][inds, :]'
#     end
#     n = N
#     inds = 2n+2:2(2n+1)
#     view(J, Block(n, n+1)) .= S.C[n+1][inds, :]'
#     view(J, Block(n+1, n+1)) .= S.B[n+1][inds, :]'
#     J
#     # The following converts the operator from bydegree to byfouriermade ordering
#     # TODO improve this
#     Jout = spzeros(B, (N+1)^2, (N+1)^2)
#     for row = 1:(N+1)^2
#         n, k, i = getnki(S, row)
#         ind = getopindex(S, n, k, i; bydegree=false, N=N)
#         Jout[ind, :] = convertcoeffsvecorder(S, J[row, :]; todegree=false)
#     end
#     bandn = bandk = bandi = 1
#     BandedBlockBandedMatrix(Jout, ([N+1; 2N:-2:1], [N+1; 2N:-2:1]),
#                             (bandk, bandk),
#                             (2bandn+2bandk+bandi, 2bandn+2bandk+bandi))
# end
# function jacobiz(S::SphericalCapSpace{<:Any, B, T, <:Any}, N) where {B,T}
#     # Transposed operator, so acts directly on coeffs vec
#     resizedata!(S, N)
#     getAs!(S, N+1, length(S.A))
#     J = BandedBlockBandedMatrix(Zeros{B}((N+1)^2, (N+1)^2),
#                                 (1:2:2N+1, 1:2:2N+1), (1, 1), (0, 0))
#     # Assign by column
#     n = 0
#     inds = 2(2n+1)+1:3(2n+1)
#     view(J, Block(n+1, n+1)) .= S.B[n+1][inds, :]'
#     view(J, Block(n+2, n+1)) .= S.A[n+1][inds, :]'
#     for n = 1:N-1
#         inds = 2(2n+1)+1:3(2n+1)
#         view(J, Block(n, n+1)) .= S.C[n+1][inds, :]'
#         view(J, Block(n+1, n+1)) .= S.B[n+1][inds, :]'
#         view(J, Block(n+2, n+1)) .= S.A[n+1][inds, :]'
#     end
#     n = N
#     inds = 2(2n+1)+1:3(2n+1)
#     view(J, Block(n, n+1)) .= S.C[n+1][inds, :]'
#     view(J, Block(n+1, n+1)) .= S.B[n+1][inds, :]'
#     J
#     # # The following converts the operator from bydegree to byfouriermade ordering
#     # # TODO improve this
#     # Jout = spzeros(B, (N+1)^2, (N+1)^2)
#     # for row = 1:(N+1)^2
#     #     n, k, i = getnki(S, row)
#     #     ind = getopindex(S, n, k, i; bydegree=false, N=N)
#     #     Jout[ind, :] = convertcoeffsvecorder(S, J[row, :]; todegree=false)
#     # end
#     # bandn = 1; bandk = bandi = 0
#     # BandedBlockBandedMatrix(Jout, ([N+1; 2N:-2:1], [N+1; 2N:-2:1]),
#     #                         (bandk, bandk),
#     #                         (2bandn+2bandk+bandi, 2bandn+2bandk+bandi))
# end


#===#
# Resizing coeffs vectors

function resizecoeffs!(S::SphericalCapSpace, f::Fun, N::Int)
    ncfs = length(f.coefficients)
    m = (N+1)^2
    cfs = convertcoeffsvecorder(S, f.coefficients) # creates new cfs vec
    if ncfs > m && any(i -> abs(i) > 1e-16, cfs[m+1:end])
        error("Trying to decrease degree of f")
    end
    cfs = convertcoeffsvecorder(S, f.coefficients) # creates new cfs vec
    resize!(cfs, m)
    if ncfs < m
        cfs[ncfs+1:end] .= 0.0
    end
    cfs = convertcoeffsvecorder(S, cfs; todegree=false)
    resize!(f.coefficients, m); f.coefficients[:] = cfs[:]
end
