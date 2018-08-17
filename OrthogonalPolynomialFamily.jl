using ApproxFun


abstract type SpaceFamily{D,R} end

struct OrthogonalPolynomialSpace{WW,D,R} <: Space{D,R}
    weight::WW
    a::Vector{R}  # Diagonal recurrence coefficients
    b::Vector{R}  # Off diagonal recurrence coefficients
end

OrthogonalPolynomialSpace(w::Fun{<:Space{D,R}}) where {D,R} =
    OrthogonalPolynomialSpace{typeof(w),D,R}(w, Vector{R}(), Vector{R}())



# R is range-type, which should be Float64.
struct OrthogonalPolynomialFamily{FF,WW,D,R,N} <: SpaceFamily{D,R}
    factors::FF
    spaces::Dict{NTuple{N,R}, OrthogonalPolynomialSpace{WW,D,R}}
end

function OrthogonalPolynomialFamily(w::Vararg{Fun{<:Space{D,R}},N}) where {D,R,N}
    all(domain.(w) .== Ref(domain(first(w)))) || throw(ArgumentError("domains incompatible"))
    WW =  typeof(prod(w.^0.5))
    spaces = Dict{NTuple{N,R}, OrthogonalPolynomialSpace{WW,D,R}}()
    OrthogonalPolynomialFamily{typeof(w),WW,D,R,N}(w, spaces)
end

function (P::OrthogonalPolynomialFamily{<:Any,<:Any,<:Any,R,N})(α::Vararg{R,N}) where {R,N}
    haskey(P.spaces,α) && return P.spaces[α]
    P.spaces[α] = OrthogonalPolynomialSpace(prod(P.factors.^α))
end


## Tests

using Test

## Jacobi example
x = Fun()
P = OrthogonalPolynomialFamily(1+x,1-x)
@test P(0.4,0.2).weight(0.1) ≈ (1+0.1)^0.4 * (1-0.1)^0.2
