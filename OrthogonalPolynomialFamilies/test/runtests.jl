using ApproxFun, OrthogonalPolynomialFamilies, FastGaussQuadrature, SingularIntegralEquations, Test
import OrthogonalPolynomialFamilies: golubwelsch, lanczos, halfdiskquadrule, gethalfdiskOP,
                                        jacobix, jacobiy, evalderivativex, evalderivativey,
                                        differentiatex, differentiatey



function g(a, b, x0, x1)
    # return the integral of x^a * (1-x^2)^b on the interval [x0, x1]
    return (x1^(a+1) * _₂F₁(-b, (a + 1) / 2, (a + 3) / 2, x1^2) / (a + 1)
            - x0^(a+1) * _₂F₁(-b, (a + 1) / 2, (a + 3) / 2, x0^2) / (a + 1))
end

@testset "Evaluation" begin
    x = Fun()
    P = OrthogonalPolynomialFamily(1+x, 1-x)
    for (a, b) in ((0.4, 0.2), (0.5,0.5), (0.0,0.0))
        @test P(a,b).weight(0.1) ≈ (1+0.1)^a * (1-0.1)^b
        w = sqrt(sum((1+x)^a*(1-x)^b))
        for n = 0:5
            P₅ = Fun(Jacobi(a,b), [zeros(n); 1])
            P₅ = P₅ * w / sqrt(sum((1+x)^a*(1-x)^b*P₅^2))
            P̃₅ = Fun(P(a,b), [zeros(n); 1])
            @test P̃₅(0.1) ≈ P₅(0.1)
        end
    end
end

@testset "Golub–Welsch" begin
    x = Fun()
    P = OrthogonalPolynomialFamily(1+x, 1-x)
    a, b = 0.4, 0.2
    @test all(golubwelsch(P(a,b), 10) .≈ gaussjacobi(10, b,a))

    x = Fun(0..1)
    H = OrthogonalPolynomialFamily(x, 1-x^2)
    N = 10; t,w = golubwelsch(H(a,b), N) # accurate for degree 2N - 1
    f = Fun(Chebyshev(0..1), randn(2N)) # a random degree 2N-1 polynomial
    @test sum(x^a*(1-x^2)^b * f) ≈ w'f.(t)

    N = 3; a = 0.5; b = 0.5; d = 5
    f = x->(x^d)
    S = Fun(identity, 0..1)
    ω = S^a * (1 - S^2)^(b+0.5)
    x, w = golubwelsch(ω, N)
    @test w'*f.(x) ≈ g(a + d, b + 0.5, 0, 1)
end

@testset "Quad Rule" begin
    N = 4; a = 0.5; b = 0.5
    xe, ye, we, xo, yo, wo  = halfdiskquadrule(N, a, b)
    # Even f
    f = (x,y)-> x + y^2
    @test sum(we .* f.(xe, ye)) ≈ (g(0, b, -1+1e-15, 1-1e-15) * g(a+1, b+0.5, 0, 1)
                                    + g(2, b, -1+1e-15, 1-1e-15) * g(a, b+1.5, 0, 1))
    # Odd f
    f = (x,y)-> x*y^3
    @test sum(wo .* f.(xo, yo)) < 1e-12 # Should be zero
end

# @testset "Fun expansion in OP basis" begin
#     N = 5; a = 0.5; b = 0.5
#     X = Fun(identity, 0..1)
#     Y = Fun(identity, -1..1)
#     ρ = sqrt(1-X^2)
#     H = OrthogonalPolynomialFamily(X, (1-X^2))
#     P = OrthogonalPolynomialFamily(1+Y, 1-Y)
#     f = (x,y)-> x^2 * y^2 + y^4 * x
#     c = halfdiskfun2coeffs(f, N, a, b)
#     x = 0.6; y = -0.3
#     @test f(x, y) ≈ c'*[gethalfdiskOP(H, P, ρ, n, k, a, b)(x,y) for n = 0:N for k = 0:n] # zero for N >= deg(f)
# end

@testset "Transform" begin
    n = 20; a = 0.5; b = -0.5

    X = Fun(identity, -1..1)
    P = OrthogonalPolynomialFamily(1-X, 1+X)
    S = P(b, b)
    f = x->exp(x)
    pts = points(S, n)
    vals = f.(pts)
    cfs = transform(S, vals)
    F = Fun(S, cfs)
    x = 0.4
    @test F(x) ≈ f(x)
    @test vals ≈ itransform(S, cfs)

    X = Fun(identity, 0..1)
    H = OrthogonalPolynomialFamily(X, 1-X^2)
    S = P(a, b)
    f = x->sin(3x)
    pts = points(S, n)
    vals = f.(pts)
    cfs = transform(S, vals)
    F = Fun(S, cfs)
    x = 0.1
    @test F(x) ≈ f(x)
    @test itransform(S, cfs) ≈ vals
end

@testset "Creating function in an OrthogonalPolynomialSpace" begin
    n = 10; a = 0.5; b = -0.5
    X = Fun(identity, 0..1)
    H = OrthogonalPolynomialFamily(X, (1-X^2))
    S = H(a, b)
    f = x -> exp(x)
    F = Fun(f, S)
    x = 0.1567
    @test F(x) ≈ f(x)
end # NOTE: Fun(f, S) is a λ function, and not a Fun

@testset "Evaluation for HalfDiskSpace (transform())" begin
    n = 10; a = 0.5; b = 1.5
    f = (x,y) -> y*x^2 + x
    F = HalfDiskFamily()
    S = F(a, b)
    pts = points(S, n)
    vals = [f(pt...) for pt in pts]
    cfs = transform(S, vals)
    x, y = 0.4, -0.7
    z = [x; y]
    @test x^2 + y^2 < 1
    F = Fun(S, cfs)
    @test F(z) ≈ f(z...)
    @test itransform(S, cfs) ≈ vals
    F = Fun(f, S, 10)
    @test F(z) ≈ f(z...)
    F = Fun(f, S)
    @test F(z) ≈ f(z...)
end

@testset "Jacobi matrices" begin
    n = 10; a = 0.5; b = -0.5
    D = HalfDiskFamily()
    S = D(a, b)
    z = [0.1; 0.2]
    f = (x,y)->x*y + x
    pts = points(S, n)
    vals = [f(pt...) for pt in pts]
    cfs = transform(S, vals)
    m = length(cfs)
    N = Int(ceil(0.5 * (-1 + sqrt(1 + 8m)))) - 1
    Jx = jacobix(S, N+1)
    Jy = jacobiy(S, N+1)
    @test evaluate(Jx'[1:m, 1:m] * cfs, S, z) ≈ z[1] * f(z...)
    @test evaluate(Jy'[1:m, 1:m] * cfs, S, z) ≈ z[2] * f(z...)
end

@testset "Evaluate partial derivative of HalfDiskSpace OP" begin
    a, b = 0.5, -0.5
    x, y = 0.5, 0.3; x^2 + y^2 < 1
    D = HalfDiskFamily(); S = D(a, b)
    h = 0.0001
    for n=0:5, k=0:n
        f = Fun(S, [zeros(sum(0:n)+k); 1])
        @test evalderivativex(S, n, k, x, y) ≈ (f(x+h,y) - f(x,y))/h atol=100h
        @test evalderivativey(S, n, k, x, y) ≈ (f(x,y+h) - f(x,y))/h atol=100h
    end
end

@testset "Evaluate partial derivative of (random) function" begin
    a, b = 0.5, 0.5
    x, y = 0.5, 0.3; x^2 + y^2 < 1
    D = HalfDiskFamily(); S = D(a, b)
    N = 4
    f = Fun(S, randn(sum(1:N+1)))
    h = 1e-5; @test (f(x+h,y)-f(x,y))/h ≈ differentiatex(f, f.space)(x,y) atol=100h
    h = 1e-5; @test (f(x,y+h)-f(x,y))/h ≈ differentiatey(f, f.space)(x,y) atol=100h
end




D = HalfDiskFamily()
a = b = 0.0
S = D(a,b)
n = 1; k=1;
f = Fun(S, [zeros(sum(0:n)+k); 1])
H̃ = Fun(S.H(a, b+k+0.5), [zeros(n-k); 1])

Fun(S.P(0.,0.),[1.0])(0.1)
x = Fun();
    P = OrthogonalPolynomialFamily(1+x, 1-x)
    Fun(P(0.,0.),[1.0])(0.1)

P₅ = Fun(Jacobi(b,b), [zeros(k); 1])
    P₅ = P₅ * sqrt(2) / norm(P₅)
P̃ = Fun(S.P(b, b), [zeros(k); 1])

x,y = 0.1, 0.2
H̃(x) * S.ρ(x)^k * P̃(y/S.ρ(x))
f(x,y)

H̃(x) * S.ρ(x)^k * P₅(y/S.ρ(x))

x,y = 0.1,0.2
f(x,y)
H̃(x) * S.ρ(x)^(k) * P₅(y/S.ρ(x))


function gethalfdiskOP_y(H, P, ρ, n, k, a, b, x, y)
    H̃ = Fun(H(a, b+k+0.5), [zeros(n-k); 1])
    X = Fun()
    w = sqrt(sum((1-X^2)^b))
    P₅ = Fun(Jacobi(b,b), [zeros(k); 1])
    P₅ = P₅ * w / sqrt(sum((1-X^2)^b*P₅^2))
    H̃(x) * ρ(x)^(k-1) * P₅'(y/ρ(x))
end
gethalfdiskOP_y(S, n, k, x, y) =  gethalfdiskOP_y(S.H, S.P, S.ρ, n, k, S.a, S.b, x, y)
a,b
for n=0:5, k=0:n
    h = 0.0001
    f = Fun(D(a,b), [zeros(sum(0:n)+k); 1])
    @test gethalfdiskOP_y(D(a,b), n, k, x, y) ≈ (f(x,y+h) - f(x,y))/h atol=100h
end

using Plots

using BlockBandedMatrices, BlockArrays
N = 5
    A = zeros(sum(0:(N+1))+1, sum(0:(N+1))+1)
for n=0:N,k=0:n
    j = sum(0:n)+k+1
    cfs = Fun((x,y) -> gethalfdiskOP_y(D(a,b), n, k, x, y), D(0.5,1.5)).coefficients
    cfs = chop(cfs, 1000eps())
    A[1:length(cfs),j] = cfs
end

N = 5
    A = BandedBlockBandedMatrix(Zeros{Float64}(sum(1:N),sum(1:(N+1))), (1:N, 1:N+1), (-1,1), (-1,1))
    for n=1:N,k=1:n
        j = sum(0:n)+k+1
        cfs = pad(Fun((x,y) -> gethalfdiskOP_y(D(a,b), n, k, x, y), D(0.5,1.5)).coefficients, sum(1:n))
        cfs = PseudoBlockArray(cfs, 1:n)
        view(A,Block(n,n+1))[k,k+1] = cfs[Block(n)][k]
    end

f = Fun(D(0.5,0.5), randn(sum(1:N+1)))

fp = Fun(D(0.5,1.5),A*f.coefficients)

fp(x,y)

(f(x,y+h)-f(x,y))/h

@time A*f.coefficients


n=1; k=1
    j = sum(0:n)+k+1
    cfs = pad(Fun((x,y) -> gethalfdiskOP_y(D(a,b), n, k, x, y), D(0.5,1.5)).coefficients, sum(1:n))
    cfs = PseudoBlockArray(cfs, 1:n)
    view(A,Block(n,n+1))
heatmap(A)

M = PseudoBlockArray(A, 1:N, 1:N)

M[Block(1,2)]



p = plot(ylims=(-1,3), xlims=(0,5), legend=false)
n,k = 2,1
    cfs = abs.(Fun((x,y) -> gethalfdiskOP_y(D(a,b), n, k, x, y), D(0.5,1.5)).coefficients)
    scatter!(1:length(cfs), cfs)




p
ylims!(01,1)
p

gethalfdiskOP_y(S, n, k, x, y)
f = Fun(D(a,b), [zeros(sum(0:n)+k); 1])
h = 0.000001
(f(x,y+h) - f(x,y))/h


####
# Old
####




h

n = k = 0
    a,b = 0.5,0.5
    x, y= 0.1, 0.2
    gethalfdiskOP_y(D(a,b), n, k, x, y)
f = Fun(D(a,b), [zeros(sum(0:n)+k); 1])
    (f(x,y+h) - f(x,y))/h

H, a, b, ρ = S.H, S.a, S.b, S.ρ
    S = D(a,b)
    H̃ = Fun(H(a, b+k+0.5), [zeros(n-k); 1])
    x = Fun()
    w = sqrt(sum((1-x^2)^b))
    P₅ = Fun(Jacobi(b,b), [zeros(k); 1])
    P₅ = P₅ * w / sqrt(sum((1-x^2)^b*P₅^2))
H̃(x) * ρ(x)^(k-1) * P₅'(y/ρ(x))
H̃(x) * ρ(x)^(k) * P₅(y/ρ(x))
f = Fun(S, [zeros(sum(0:n)+k); 1])

H̃ = Fun(H(a, b+k+0.5), [zeros(n-k); 1])
P̃ = Fun(P(b, b), [zeros(k); 1])

P̃(0.4)
P₅ = Fun(Jacobi(b,b), [zeros(k); 1])
            P₅ = P₅ * w / sqrt(sum((1-x^2)^b*P₅^2))
    P₅(0.4)


x = Fun()
    P = OrthogonalPolynomialFamily(1+x, 1-x)
    (a, b) =(0.5,0.5)
    w = sqrt(sum((1-x^2)^b))

        for k = 0:5
            P₅ = Fun(Jacobi(b,b), [zeros(k); 1])
            P₅ = P₅ * w / sqrt(sum((1-x^2)^b*P₅^2))
            P̃₅ = Fun(P(b,b), [zeros(k); 1])

H̃(x) * ρ(x)^(k) * P̃(y/ρ(x))

return (x,y) -> (H̃(x) * ρ(x)^k * P̃(y / ρ(x)))
f(x,y)
0.5,0.5
gethalfdiskOP_y(S, 7,1, x, y)

gethalfdiskOP(H, P, ρ, n, k, a, b)(x,y)



gethalfdiskOP_y(S, 0, 0, x, y)

points( D(1.5,1.5), 20)

chop(Fun((x,y) -> gethalfdiskOP_y(S, 7,1, x, y), D(0.5,1.5)).coefficients, 100eps())
n,k = 2,1

H,b = S.H; b = S.b;
H̃ = Fun(H(a, b+k+0.5), [zeros(n-k); 1])
P₅ = Fun(Jacobi(b,b), [zeros(k); 1])
P₅ = P₅ * sqrt(2) / norm(P₅)
H̃(x) * ρ(x)^(k-1) * P₅'(y/ρ(x))


f = Fun(S, [zeros(sum(0:n)+k); 1])
(f(x,y+h) - f(x,y))/h
gethalfdiskOP_y(D(a,b), n, k, x, y)



S.a


h = 0.00001
(f(x,y+h)-f(x,y))/h




P̃₅ = Fun(P(a,b), [zeros(k); 1])
@test P̃₅(0.1) ≈ P₅(0.1)




S.P(a,b)

x = Fun()
P = S.P
a = b = 0.0
    @test P(a,b).weight(0.1) ≈ (1+0.1)^a * (1-0.1)^b
    w = sqrt(sum((1+x)^a*(1-x)^b))
    n = 0
        P₅ = Fun(Jacobi(a,b), [zeros(n); 1])
        P₅ = P₅ * w / sqrt(sum((1+x)^a*(1-x)^b*P₅^2))
        P̃₅ = Fun(P(a,b), [zeros(n); 1])
        @test P̃₅(0.1) ≈ P₅(0.1)

@which evaluate(P̃₅.coefficients, P̃₅.space, 0.1)

@which P̃₅(0.1)

P̃₅|> space


P₅ = Fun(Jacobi(a,b), [zeros(n); 1])


P = Fun(Jacobi(b, b), [zeros(k); 1])

P̃(0.1)
P(0.1)/norm(P)
P̃(0.1)
x,y = (0.1,0.2)
H̃(x) * S.ρ(x)^k * P̃(y/S.ρ(x))
a = b = 0.0
n = 2
P₅ = Fun(Jacobi(a,b), [zeros(k); 1])
w
P₅ = P₅ / sqrt(sum((1+x)^a*(1-x)^b*P₅^2))
P̃₅ = Fun(S.P(a,b), [zeros(n); 1])
    @test P̃₅(0.1) ≈ P₅(0.1)


end



P(x,y)


function gethalfdiskOP_y(H, P, ρ, n, k, a, b)
    H̃ = Fun(H(a, b+k+0.5), [zeros(n-k); 1])
    P̃ = Fun(P(b, b), [zeros(k); 1])
    return (x,y) -> (H̃(x) * ρ(x)^k * P̃(y / ρ(x)))
end

function Pderivativex(a,b,n,k)

x = Fun()
    P = OrthogonalPolynomialFamily(1+x, 1-x)
    (a, b) =(0.5,0.5)
    w = sqrt(sum((1-x^2)^b))

        for k = 0:5
            P₅ = Fun(Jacobi(b,b), [zeros(k); 1])
            P₅ = P₅ * w / sqrt(sum((1-x^2)^b*P₅^2))
            P̃₅ = Fun(P(a,b), [zeros(k); 1])
            @test P̃₅(0.1) ≈ P₅(0.1)
        end
