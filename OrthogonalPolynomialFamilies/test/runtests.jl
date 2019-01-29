using ApproxFun, OrthogonalPolynomialFamilies, FastGaussQuadrature,
        SingularIntegralEquations, Test
import OrthogonalPolynomialFamilies: golubwelsch, lanczos, halfdiskquadrule,
        gethalfdiskOP, jacobix, jacobiy, differentiatex, differentiatey,
        resizecoeffs!, laplace, transformoperator, operatorclenshaw, getnk,
        convertweightedtononoperator, increaseparamsoperator, getopindex



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
    X = Fun(identity, 0..1)
    x, w = golubwelsch(H(a, b+0.5), N)
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

@testset "Evaluate partial derivative of (random) function" begin
    a, b = 0.5, 0.5
    x, y = 0.5, 0.3; x^2 + y^2 < 1
    D = HalfDiskFamily(); S = D(a, b)
    N = 4
    f = Fun(S, randn(sum(1:N+1)))
    h = 1e-5; @test (f(x+h,y)-f(x,y))/h ≈ differentiatex(f, f.space)(x,y) atol=100h
    h = 1e-5; @test (f(x,y+h)-f(x,y))/h ≈ differentiatey(f, f.space)(x,y) atol=100h
end

@testset "Poisson" begin
    # Model Problem: Δ(u*w)(x,y) = f(x,y) in Ω=halfdisk; u(x,y) ≡ 0 on ∂Ω.
    #   where w(x,y) = x*(1-x^2-y^2) is the weight of the D(1.0,1.0) basis.
    a, b = 1.0, 1.0; D = HalfDiskFamily(); S = D(a, b)
    x, y = 0.4, -0.2; z = [x; y] # Test point

    # 1) f(x,y) = -8x => u(x,y) ≡ 1
    N = 1 # degree of f
    c = rand(1)[1]; f = Fun((x,y)->-c*8x, S)
    Δ = laplace(D, N-1)
    u = Fun(S, Δ \ resizecoeffs!(f, N))
    @test u(z) ≈ c # Result u(x,y) where Δ(u*w)(x,y) = f(x,y)

    # 2) f(x,y) = 2 - 12xy - 14x^2 - 2y^2 => u(x,y) = x + y
    U = Fun((x,y)->x+y, S)
    N = 2 # degree of f
    f = Fun((x,y)->(2 - 12x*y - 14x^2 - 2y^2), S)
    Δ = laplace(D, N-1)
    u = Fun(S, Δ \ f.coefficients[1:getopindex(N,N)])
    @test u(z) ≈ U(z)
end

@testset "Conversion operators" begin
    a, b = 1.0, 1.0; D = HalfDiskFamily(); S = D(a, b)
    x, y = 0.457, -0.209; z = [x; y] # Test point
    N = 5
    f1 = Fun(D(0.0, 2.0), rand(sum(1:N+1)))
    T = transformoperator(D(0.0, 2.0), N)
    f2 = Fun(D(2.0, 2.0), T * f1.coefficients)
    @test f2(z) ≈ f1(z)
    f1 = Fun(D(0.0, 1.0), rand(sum(1:N+1)))
    T = transformoperator(D(0.0, 1.0), N)
    f2 = Fun(D(1.0, 1.0), T * f1.coefficients)
    @test f2(z) ≈ f1(z)
    f1 = Fun(D(1.0, 0.0), rand(sum(1:N+1)))
    T = transformoperator(D(1.0, 0.0), N)
    f2 = Fun(D(0.0, 0.0), T * f1.coefficients)
    @test f2(z) ≈ x*f1(z)
end

@testset "Operator Clenshaw" begin
    a, b = 1.0, 1.0; D = HalfDiskFamily(); S = D(a, b)
    x, y = 0.4, -0.2; z = [x; y] # Test point
    N = 14
    oper = Fun((x,y)->cos(x)*y^2, S, 200)
    A = operatorclenshaw(oper, S)
    f = Fun((x,y)->x*y, S)
    resizecoeffs!(f, N)
    foper = Fun(S, A*f.coefficients)
    @test foper(z) ≈ f(z) * oper(z)
end

@testset "Weighted to non-weighted space conversion" begin
    # NOTE: Operator takes W11*P11 -> P00 only
    tol = 1e-8
    a, b = 1.0, 1.0; D = HalfDiskFamily(); S = D(a, b)
    x, y = 0.4, -0.2; z = [x; y] # Test point
    weight(S) = z -> (z[1]^S.a * (1-z[1]^2-z[2]^2)^S.b)
    St = (S.family)(S.a-1, S.b-1)
    maxop = 100
    N = getnk(maxop)[1] + 3
    W = convertweightedtononoperator(S, N)
    for j = 1:maxop
        p = Fun(S, [zeros(j-1); 1])
        resizecoeffs!(p, N)
        q = Fun(St, W * p.coefficients)
        res = abs(q(z) - p(z)*weight(S)(z))
        @test res < tol
        if res >= tol
            @show getnk(j), j, res
        end
    end
end

@testset "Increment parameters operator" begin
    tol = 1e-8
    a, b = 1.0, 1.0; D = HalfDiskFamily(); S = D(a-1, b-1)
    x, y = 0.4, -0.2; z = [x; y] # Test point
    St = (S.family)(S.a+1, S.b+1)
    maxop = 150
    N = getnk(maxop)[1] + 1
    C = increaseparamsoperator(S, N)
    for j = 1:maxop
        p = Fun(S, [zeros(j-1); 1])
        resizecoeffs!(p, N)
        q = Fun(St, C * p.coefficients)
        res = abs(q(z) - p(z))
        res > tol && @show getnk(j), j, res
        @test res < tol
    end

    a, b = 2.0, 1.0; D = HalfDiskFamily(); S = D(a-1, b-1)
    x, y = 0.4, -0.2; z = [x; y] # Test point
    St = (S.family)(S.a+1, S.b+1)
    maxop = 150
    N = getnk(maxop)[1] + 1
    C = increaseparamsoperator(S, N)
    for j = 1:maxop
        p = Fun(S, [zeros(j-1); 1])
        resizecoeffs!(p, N)
        q = Fun(St, C * p.coefficients)
        res = abs(q(z) - p(z))
        res > tol && @show getnk(j), j, res
        @test res < tol
    end
end

#=====#

a, b = 1.0, 1.0; D = HalfDiskFamily(); S = D(a, b)
N = 5
laplace(D, N)
