using ApproxFun, OrthogonalPolynomialFamilies, FastGaussQuadrature, SingularIntegralEquations, Test
import OrthogonalPolynomialFamilies: golubwelsch, lanczos, halfdiskfun2coeffs, halfdiskquadrule, gethalfdiskOP



function g(a, b, x0, x1)
    # return the integral of x^a * (1-x^2)^b on the interval [x0, x1]
    return (x1^(a+1) * _₂F₁(-b, (a + 1) / 2, (a + 3) / 2, x1^2) / (a + 1)
            - x0^(a+1) * _₂F₁(-b, (a + 1) / 2, (a + 3) / 2, x0^2) / (a + 1))
end

@testset "Evaluation" begin
    x = Fun()
    P = OrthogonalPolynomialFamily(1+x, 1-x)
    a, b = 0.4, 0.2
    @test P(a,b).weight(0.1) ≈ (1+0.1)^a * (1-0.1)^b
    w = sqrt(sum((1+x)^a*(1-x)^b))
    for n = 0:5
        P₅ = Fun(Jacobi(a,b), [zeros(n); 1])
        P₅ = P₅ * w / sqrt(sum((1+x)^a*(1-x)^b*P₅^2))
        P̃₅ = Fun(P(a,b), [zeros(n); 1])
        @test P̃₅(0.1) ≈ P₅(0.1)
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

@testset "Fun expansion in OP basis" begin
    N = 5; a = 0.5; b = 0.5
    X = Fun(identity, 0..1)
    Y = Fun(identity, -1..1)
    ρ = sqrt(1-X^2)
    H = OrthogonalPolynomialFamily(X, (1-X^2))
    P = OrthogonalPolynomialFamily(1+Y, 1-Y)
    f = (x,y)-> x^2 * y^2 + y^4 * x
    c = halfdiskfun2coeffs(f, N, a, b)
    x = 0.6; y = -0.3
    @test f(x, y) ≈ c'*[gethalfdiskOP(H, P, ρ, n, k, a, b)(x,y) for n = 0:N for k = 0:n] # zero for N >= deg(f)
end

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
    n = 4; a = 0.5; b = 1.5
    f = (x,y) -> y*x^2 + x
    S = HalfDiskSpace(a, b)
    pts = points(S, n)
    vals = [f(pt...) for pt in pts]
    cfs = transform(S, vals)
    S
    z = [0.1; 0.2]
    F = Fun(S, cfs)
    F(z) ≈ f(z...)
    itransform(S, cfs) ≈ vals
end
