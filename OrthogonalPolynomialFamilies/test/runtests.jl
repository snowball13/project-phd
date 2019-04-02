using ApproxFun, OrthogonalPolynomialFamilies, FastGaussQuadrature,
        SingularIntegralEquations, Test, SparseArrays
import OrthogonalPolynomialFamilies: golubwelsch, lanczos, halfdiskquadrule,
        gethalfdiskOP, jacobix, jacobiy, differentiatex, differentiatey,
        resizecoeffs!, laplaceoperator, transformparamsoperator, operatorclenshaw, getnk,
        getopindex, weight



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

@testset "Increment/decrement parameters operators" begin
    tol = 1e-8

    a, b = 1.0, 1.0; D = HalfDiskFamily(); S = D(a-1, b-1)
    x, y = 0.4, -0.2; z = [x; y] # Test point
    St = (S.family)(S.a+1, S.b+1)
    maxop = 150
    N = getnk(maxop)[1] + 1
    C = transformparamsoperator(S, (S.family)(S.a+1, S.b+1), N)
    for j = 1:maxop
        p = Fun(S, [zeros(j-1); 1])
        resizecoeffs!(p, N)
        q = Fun(St, C * p.coefficients)
        res = abs(q(z) - p(z))
        res > tol && @show getnk(j), j, res
        @test res < tol
    end

    tol = 1e-13

    paramslist = [(1,1), (1,0), (0,1)]
    for params in paramslist
        a, b = 2.0, 4.0; D = HalfDiskFamily(); S = D(a, b)
        x, y = 0.4, -0.2; z = [x; y] # Test point
        St = (S.family)(S.a+params[1], S.b+params[2])
        maxop = 150
        N = getnk(maxop)[1] + 1
        C = transformparamsoperator(S, St, N)
        for j = 1:maxop
            p = Fun(S, [zeros(j-1); 1])
            resizecoeffs!(p, N)
            q = Fun(St, C * p.coefficients)
            res = abs(q(z) - p(z))
            res > tol && @show getnk(j), j, res
            @test res < tol
        end
    end
    for params in paramslist
        a, b = 2.0, 4.0; D = HalfDiskFamily(); S = D(a, b)
        x, y = 0.4, -0.2; z = [x; y] # Test point
        St = (S.family)(S.a-params[1], S.b-params[2])
        maxop = 150
        N = getnk(maxop)[1] + 1
        C = transformparamsoperator(S, St, N, weighted=true)
        for j = 1:maxop
            p = Fun(S, [zeros(j-1); 1])
            q = Fun(St, C * pad(p.coefficients, size(C)[2]))
            res = abs(weight(St, z) * q(z) - weight(S, z) * p(z))
            res > tol && @show getnk(j), j, res
            @test res < tol
        end
    end
end

@testset "laplaceoperator" begin
    function dxweight(S::HalfDiskSpace{<:Any, <:Any, T}, x, y) where T
        D = S.family
        T(S.a) * weight(D(S.a-1, S.b), x, y) - 2 * T(S.b) * weight(D(S.a+1,S.b-1), x, y)
    end
    dxweight(S::HalfDiskSpace, z) = dxweight(S, z[1], z[2])
    function dyweight(S::HalfDiskSpace{<:Any, <:Any, T}, x, y) where T
        D = S.family
        - 2 * T(S.b) * y * weight(D(S.a,S.b-1), x, y)
    end
    dyweight(S::HalfDiskSpace, z) = dyweight(S, z[1], z[2])
    function d2xweight(S::HalfDiskSpace{<:Any, <:Any, T}, x, y) where T
        D = S.family
        ret = T(S.a) * ((T(S.a) - 1) * weight(D(S.a-2,S.b), x, y) - 2 * T(S.b) * weight(D(S.a,S.b-1), x, y))
        ret -= 2 * T(S.b) * ((T(S.a) + 1) * weight(D(S.a,S.b-1), x, y) - 2 * (T(S.b) - 1) * weight(D(S.a+2,S.b-2), x, y))
        ret
    end
    d2xweight(S::HalfDiskSpace, z) = d2xweight(S, z[1], z[2])
    function d2yweight(S::HalfDiskSpace{<:Any, <:Any, T}, x, y) where T
        D = S.family
        - 2 * T(S.b) * (weight(D(S.a,S.b-1), x, y) - 2 * T(S.b) * (T(S.b) - 1) * y^2 * weight(D(S.a,S.b-2), x, y))
    end
    d2yweight(S::HalfDiskSpace, z) = d2yweight(S, z[1], z[2])

    x, y = 0.34, -0.29; z = [x; y]
    # W11->P11
    D = HalfDiskFamily()
    a, b = 1.0, 1.0; S = D(a, b)
    N = 15
    L = laplaceoperator(S, S, N, weighted=true)
    for n = 1:10
        u = Fun((x,y)->y*x^(n-1)+y^n, S)
        f = Fun((x,y)->(u(x,y)*d2xweight(S,x,y) + 2*y*(n-1)*x^(n-2)*dxweight(S,x,y) + y*(n-2)*(n-1)*x^(n-3)*weight(S,x,y)
                        + u(x,y)*d2yweight(S,x,y) + 2*(n*y^(n-1) + x^(n-1))*dyweight(S,x,y) + n*(n-1)*y^(n-2)*weight(S,x,y)), S)
        resizecoeffs!(f, N)
        cfs = sparse(L) \ f.coefficients
        @test Fun(S, cfs)(z) ≈ u(z)
    end
    # W22->P00
    # TODO
    # P00->P22
    D = HalfDiskFamily()
    a, b = 0.0, 0.0; S = D(a, b)
    N = 15
    L = laplaceoperator(S, D(a+2, a+2), N, weighted=false)
    n = 1; u = Fun((x,y)->2y, S); resizecoeffs!(u, N); @test abs(Fun(D(a+2, b+2), L * u.coefficients)(z)) < 1e-12
    for n = 2:10
        u = Fun((x,y)->y*x^(n-1)+y^n, S)
        resizecoeffs!(u, N)
        cfs = L * u.coefficients
        @test Fun(D(a+2, b+2), cfs)(z) ≈ y*(n-2)*(n-1)*x^(n-3) + n*(n-1)*y^(n-2)
    end
end

#=====#
