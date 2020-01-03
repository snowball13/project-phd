# Spherical/Polar Caps

export SphericalCapFamily, SphericalCapSpace

# R should be Float64, B should be BigFloat
abstract type SphericalFamily{B,R,N} end
struct SphericalCap{B,T} <: Domain{SVector{2,T}} end
SphericalCap() = SphericalCap{BigFloat, Float64}()
checkpoints(::SphericalCap) = [SVector(0.1, 0.23), SVector(0.3, 0.12)]

struct SphericalCapSpace{DF, B, T, N} <: Space{SphericalCap{B,T}, T}
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

function SphericalCapSpace(fam::SphericalFamily{B,T,N}, params::NTuple{N,B}) where {B,T,N}
    SphericalCapSpace{typeof(fam), B, T, N}(
        fam, params, Vector{T}(), Vector{Vector{T}}(),
        Vector{Vector{T}}(), Vector{Vector{T}}(), Vector{SparseMatrixCSC{T}}(),
        Vector{SparseMatrixCSC{T}}(), Vector{SparseMatrixCSC{T}}(),
        Vector{SparseMatrixCSC{T}}())
end

# TODO !!!!!!!
in(x::SVector{2}, D::SphericalCap) = D.α ≤ x[1] ≤ D.β && D.γ*D.ρ(x[1]) ≤ x[2] ≤ D.δ*D.ρ(x[1])

spacescompatible(A::SphericalCapSpace, B::SphericalCapSpace) = (A.params == B.params)

domain(::SphericalCapSpace) = SphericalCap()

# R should be Float64, B BigFloat
struct SphericalCapFamily{B,T,N,FA,I} <: SphericalFamily{B,T,N}
    spaces::Dict{NTuple{N,B}, SphericalCapSpace}
    α::T
    H::FA # DiskSliceFamily
    nparams::I
end

function (D::SphericalCapFamily{B,T,N,<:Any,<:Any})(params::NTuple{N,B}) where {B,T,N}
    haskey(D.spaces,params) && return D.spaces[params]
    D.spaces[params] = SphericalCapSpace(D, params)
end
(D::SphericalCapFamily{B,T,N,<:Any,<:Any})(params::Vararg{B,N}) where {B,T,N} =
    D(params)
(D::SphericalCapFamily{B,T,N,<:Any,<:Any})(params::Vararg{T,N}) where {B,T,N} =
    D(B.(params))

function SphericalCapFamily(::Type{B}, ::Type{T}, α::T) where {B,T}
    N = 2
    H = DiskSliceFamily(α)
    N = H.nparams
    spaces = Dict{NTuple{N,B}, SphericalCapSpace}()
    SphericalCapFamily{B,T,N,typeof(H),Int}(spaces, α, H, N)
end
# Useful quick constructors
SphericalCapFamily(α::T) where T = SphericalCapFamily(BigFloat, T, α)
SphericalCapFamily() = SphericalCapFamily(BigFloat, Float64, 0.0) # Hemisphere


#=======#

function getAs!(S::SphericalCapSpace, N, N₀)
    for n = N+1:-1:N₀
        A1 = getHspace(S).A[n+1]
        A2 = getHspace(S, S.a, S.b + 1).A[n]
        Ax = [A1[1:n+1, :] zeros(n+1);
              zeros(n, n+2) A2[1:n, :]]
        Ay = [A1[n+2:end, :] zeros(n+1);
              zeros(n, n+2) A2[n+1:end, :]]
        Az = 
        S.A[n+1] = [Ax; Ay; Az]
    end
end


# Jacobi matrices
function jacobix(S::SphericalCapSpace, N)

end
