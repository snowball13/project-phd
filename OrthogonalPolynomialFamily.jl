using ApproxFun
    import ApproxFun: evaluate, PolynomialSpace, recα, recβ, recγ, recA, recB, recC

abstract type SpaceFamily{D,R} end


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



struct OrthogonalPolynomialSpace{WW,D,R} <: PolynomialSpace{D,R}
    weight::WW
    a::Vector{R}  # Diagonal recurrence coefficients
    b::Vector{R}  # Off diagonal recurrence coefficients
end

OrthogonalPolynomialSpace(w::Fun{<:Space{D,R}}) where {D,R} =
    OrthogonalPolynomialSpace{typeof(w),D,R}(w, Vector{R}(), Vector{R}())

function resizedata!(S::OrthogonalPolynomialSpace, n)
    n ≤ length(S.a) && return S
    # TODO: populate a and b

end

#####
# recα/β/γ are given by
#       x p_{n-1} =γ_n p_{n-2} + α_n p_{n-1} +  p_n β_n
#####

recα(::Type{T}, S::OrthogonalPolynomialSpace, n) where T =
    T(resizedata!(S, n).a[n])
recβ(::Type{T}, S::OrthogonalPolynomialSpace, n) where T =
    T(resizedata!(S, n).b[n])
recγ(::Type{T}, S::OrthogonalPolynomialSpace, n) where T =
    T(resizedata!(S, n).b[n-1])


recA(::Type{T}, S::OrthogonalPolynomialSpace, n) where T =
    recα(T, S, n)/recβ(T,S,b) # ?
recB(::Type{T}, S::OrthogonalPolynomialSpace, n) where T =

recC(::Type{T}, S::OrthogonalPolynomialSpace, n) where T =


## Tests

using Test

## Jacobi example
x = Fun()
P = OrthogonalPolynomialFamily(1+x,1-x)
@test P(0.4,0.2).weight(0.1) ≈ (1+0.1)^0.4 * (1-0.1)^0.2

P₅ = Fun(Jacobi(0.4,0.2), [zeros(5); 1])
P₅ = P₅/sqrt(sum((1+x)^0.4*(1-x)^0.2*P₅^2))
P̃₅ = Fun(P(0.4,0.2), [zeros(5); 1])

# doesn't yet work
@test P̃₅(0.1) ≈ P₅(0.1)


f = Fun(Jacobi(0.1,0.2), [1.,2.,3.])

f(0.1)
@which clenshaw(f.space, f.coefficients, 0.1)

Chebyshev() |> typeof |> supertype

recα(Float64, Chebyshev(), 1)
@which recβ(Float64, Chebyshev(), 2)

domain(Chebyshev())
