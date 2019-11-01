using ApproxFun, OrthogonalPolynomialFamilies, FastGaussQuadrature,
        SingularIntegralEquations, Test, SparseArrays
import OrthogonalPolynomialFamilies: golubwelsch, halfdiskquadrule,
        gethalfdiskOP, jacobix, jacobiy, differentiatex, differentiatey,
        resizecoeffs!, laplaceoperator, transformparamsoperator,
        operatorclenshaw, getnk, getopindex, weight, resizedata!,
        differentiateweightedspacex, differentiateweightedspacey



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
    n = 10; a = 1.0; b = 2.0
    f = (x,y) -> y*x^2 + x
    D = HalfDiskFamily()
    S = D(a, b)
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
    @test evaluate(Jx[1:m, 1:m] * cfs, S, z) ≈ z[1] * f(z...)
    @test evaluate(Jy[1:m, 1:m] * cfs, S, z) ≈ z[2] * f(z...)
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
    oper = Fun((x,y)->cos(x)*y^2, D(a-1, b-1), 200)
    f = Fun((x,y)->x*y, S)
    resizecoeffs!(f, N)
    A = operatorclenshaw(oper, D(a-1, b-1), N+3)
    T = transformparamsoperator(D(a-1, b-1), S, N+3)
    Tw = transformparamsoperator(S, D(a-1, b-1), N, weighted=true)
    foper = Fun(S, T*A*Tw*f.coefficients)
    @test foper(z) ≈ weight(S, z) * f(z) * oper(z)

    a, b = 1.0, 1.0; D = HalfDiskFamily(); S = D(a, b)
    x, y = 0.4, -0.2; z = [x; y] # Test point
    N = 14
    oper = Fun((x,y)->cos(x)*y^2, S, 200)
    f = Fun((x,y)->x*y, S)
    resizecoeffs!(f, N)
    A = operatorclenshaw(oper, S, N)
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
        - 2 * T(S.b) * (weight(D(S.a,S.b-1), x, y) + 4 * T(S.b) * (T(S.b) - 1) * y^2 * weight(D(S.a,S.b-2), x, y))
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
# DiskSlice

@testset "Evaluation for DiskSliceSpace (transform())" begin
    n = 10; a = 1.0; b = 2.0; c = 1.0
    f = (x,y) -> y*x^2 + x
    α, β = 0.4, 0.8
    D = DiskSliceFamily(α, β)
    S = D(a, b, c)
    pts = points(S, n)
    vals = [f(pt...) for pt in pts]
    cfs = transform(S, vals)
    x, y = 0.5673, -0.2786; z = [x; y]
    @test (x^2 + y^2 < 1 && D.α ≤ z[1] ≤ D.β && D.γ*D.ρ(z[1]) ≤ z[2] ≤ D.δ*D.ρ(z[1]))
    F = Fun(S, cfs)
    @test F(z) ≈ f(z...)
    @test itransform(S, cfs) ≈ vals
    F = Fun(f, S, 10)
    @test F(z) ≈ f(z...)
    F = Fun(f, S)
    @test F(z) ≈ f(z...)

    n = 10; a = 1.0; b = 2.0; c = 1.0
    f = (x,y) -> y*x^2 + x
    α, β = 0.4, 0.8
    D = DiskSliceFamily(α)
    S = D(b, c)
    pts = points(S, n)
    vals = [f(pt...) for pt in pts]
    cfs = transform(S, vals)
    x, y = 0.5673, -0.2786; z = [x; y]
    @test (x^2 + y^2 < 1 && D.α ≤ z[1] ≤ D.β && D.γ*D.ρ(z[1]) ≤ z[2] ≤ D.δ*D.ρ(z[1]))
    F = Fun(S, cfs)
    @test F(z) ≈ f(z...)
    @test itransform(S, cfs) ≈ vals
    F = Fun(f, S, 10)
    @test F(z) ≈ f(z...)
    F = Fun(f, S)
    @test F(z) ≈ f(z...)
end

@testset "Jacobi matrices" begin
    n = 10; a, b, c = 1.0, 1.0, 2.0
    α, β = 0.2, 0.9
    D = DiskSliceFamily(α, β)
    S = D(a, b, c)
    x, y = α + 0.1673, -0.2786; z = [x; y]
    @test (x^2 + y^2 < 1 && D.α ≤ z[1] ≤ D.β && D.γ*D.ρ(z[1]) ≤ z[2] ≤ D.δ*D.ρ(z[1]))
    f = (x,y)->x*y + x
    pts = points(S, n)
    vals = [f(pt...) for pt in pts]
    cfs = transform(S, vals)
    m = length(cfs)
    N = Int(ceil(0.5 * (-1 + sqrt(1 + 8m)))) - 1
    Jx = jacobix(S, N+1)
    Jy = jacobiy(S, N+1)
    @test evaluate(Jx[1:m, 1:m] * cfs, S, z) ≈ z[1] * f(z...)
    @test evaluate(Jy[1:m, 1:m] * cfs, S, z) ≈ z[2] * f(z...)

    n = 10; a, b, c = 1.0, 1.0, 2.0
    α, β = 0.2, 0.9
    D = DiskSliceFamily(α)
    S = D(b, c)
    x, y = α + 0.1673, -0.2786; z = [x; y]
    @test (x^2 + y^2 < 1 && D.α ≤ z[1] ≤ D.β && D.γ*D.ρ(z[1]) ≤ z[2] ≤ D.δ*D.ρ(z[1]))
    f = (x,y)->x*y + x
    pts = points(S, n)
    vals = [f(pt...) for pt in pts]
    cfs = transform(S, vals)
    m = length(cfs)
    N = Int(ceil(0.5 * (-1 + sqrt(1 + 8m)))) - 1
    Jx = jacobix(S, N+1)
    Jy = jacobiy(S, N+1)
    @test evaluate(Jx[1:m, 1:m] * cfs, S, z) ≈ z[1] * f(z...)
    @test evaluate(Jy[1:m, 1:m] * cfs, S, z) ≈ z[2] * f(z...)
end

@testset "Operator Clenshaw" begin
    n = 10; a, b, c = 1.0, 1.0, 2.0
    α, β = 0.2, 0.9
    D = DiskSliceFamily(α, β)
    S = D(a, b, c)
    x, y = α + 0.1673, -0.2786; z = [x; y]
    @test (x^2 + y^2 < 1 && D.α ≤ z[1] ≤ D.β && D.γ*D.ρ(z[1]) ≤ z[2] ≤ D.δ*D.ρ(z[1]))
    N = 14
    oper = Fun((x,y)->cos(x)*y^2, S, 200)
    A = operatorclenshaw(oper, S)
    f = Fun((x,y)->x*y, S)
    resizecoeffs!(f, N)
    foper = Fun(S, A*f.coefficients)
    @test foper(z) ≈ f(z) * oper(z)

    n = 10; a, b, c = 1.0, 1.0, 2.0
    α, β = 0.2, 0.9
    D = DiskSliceFamily(α)
    S = D(b, c)
    x, y = α + 0.1673, -0.2786; z = [x; y]
    @test (x^2 + y^2 < 1 && D.α ≤ z[1] ≤ D.β && D.γ*D.ρ(z[1]) ≤ z[2] ≤ D.δ*D.ρ(z[1]))
    N = 14
    oper = Fun((x,y)->cos(x)*y^2, S, 200)
    A = operatorclenshaw(oper, S)
    f = Fun((x,y)->x*y, S)
    resizecoeffs!(f, N)
    foper = Fun(S, A*f.coefficients)
    @test foper(z) ≈ f(z) * oper(z)
end

@testset "Evaluate partial derivative of (random and known) function" begin
    a, b, c = 1.0, 1.0, 2.0
    α, β = 0.2, 0.9
    D = DiskSliceFamily(α, β)
    S = D(a, b, c)
    x, y = α + 0.1673, -0.2786; z = [x; y]
    @test (x^2 + y^2 < 1 && D.α ≤ z[1] ≤ D.β && D.γ*D.ρ(z[1]) ≤ z[2] ≤ D.δ*D.ρ(z[1]))
    N = 4
    f = Fun(S, randn(sum(1:N+1)))
    # NOTE: This test for DX operator nay not be good enough to ensure it is "correct"
    # Its fails for atol=100h
    h = 1e-5; @test (f(x+h,y)-f(x,y))/h ≈ differentiatex(f, f.space)(x,y) atol=1000h
    h = 1e-5; @test (f(x,y+h)-f(x,y))/h ≈ differentiatey(f, f.space)(x,y) atol=100h
    f = Fun((x,y)->y*sin(x), S)
    h = 1e-5; @test y*cos(x) ≈ differentiatex(f, f.space)(x,y) atol=100h
    f = Fun((x,y)->x*sin(y), S)
    h = 1e-5; @test x*cos(y) ≈ differentiatey(f, f.space)(x,y) atol=100h

    a, b, c = 1.0, 1.0, 2.0
    α, β = 0.2, 0.9
    D = DiskSliceFamily(α)
    S = D(b, c)
    x, y = α + 0.1673, -0.2786; z = [x; y]
    @test (x^2 + y^2 < 1 && D.α ≤ z[1] ≤ D.β && D.γ*D.ρ(z[1]) ≤ z[2] ≤ D.δ*D.ρ(z[1]))
    N = 4
    f = Fun(S, randn(sum(1:N+1)))
    # NOTE: This test for DX operator nay not be good enough to ensure it is "correct"
    # Its fails for atol=100h
    h = 1e-5; @test (f(x+h,y)-f(x,y))/h ≈ differentiatex(f, f.space)(x,y) atol=1000h
    h = 1e-5; @test (f(x,y+h)-f(x,y))/h ≈ differentiatey(f, f.space)(x,y) atol=100h
    f = Fun((x,y)->y*sin(x), S)
    h = 1e-5; @test y*cos(x) ≈ differentiatex(f, f.space)(x,y) atol=100h
    f = Fun((x,y)->x*sin(y), S)
    h = 1e-5; @test x*cos(y) ≈ differentiatey(f, f.space)(x,y) atol=100h
end

@testset "Increment/decrement parameters operators" begin
    tol = 1e-13
    x, y = 0.4, -0.2; z = [x; y] # Test point

    paramslist = [(1,1,0), (1,1,1), (0,0,1)]
    for params in paramslist
        α, β = 0.2, 0.8; D = DiskSliceFamily(α, β)
        a, b, c = 1.0, 1.0, 1.0; S = D(a, b, c)
        St = (S.family)(S.params.+params)
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
        α, β = 0.2, 0.8; D = DiskSliceFamily(α, β)
        a, b, c = 1.0, 1.0, 1.0; S = D(a, b, c)
        St = (S.family)(S.params.-params)
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
    paramslist = [(1,0), (1,1), (0,1)]
    for params in paramslist
        α, β = 0.2, 0.8; D = DiskSliceFamily(α)
        a, b, c = 1.0, 1.0, 1.0; S = D(b, c)
        St = (S.family)(S.params.+params)
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
        α, β = 0.2, 0.8; D = DiskSliceFamily(α)
        a, b, c = 1.0, 1.0, 1.0; S = D(b, c)
        St = (S.family)(S.params.-params)
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
    rhoval(x) = sqrt(1-x^2)
    dxrhoval(x) = -x/sqrt(1-x^2)
    d2xrhoval(x) = - (1-x^2)^(-0.5) - x^2 * (1-x^2)^(-1.5)
    function dxweight(S::DiskSliceSpace{<:Any, <:Any, T, <:Any}, x, y) where T
        D = S.family
        if D.nparams == 2
            a, b = S.params
            ret = a * weight(D(a-1, b), x, y)
            ret += 2 * rhoval(x) * dxrhoval(x) * b * weight(D(a, b-1), x, y)
            T(ret)
        else
            a, b, c = S.params
            ret = -a * weight(D(a-1, b, c), x, y)
            ret += b * weight(D(a, b-1, c), x, y)
            ret += 2 * rhoval(x) * dxrhoval(x) * c * weight(D(a, b, c-1), x, y)
            T(ret)
        end
    end
    dxweight(S::DiskSliceSpace, z) = dxweight(S, z[1], z[2])
    function dyweight(S::DiskSliceSpace{<:Any, <:Any, T, <:Any}, x, y) where T
        ret = - 2 * S.params[end] * y * weight(differentiateweightedspacey(S), x, y)
        T(ret)
    end
    dyweight(S::DiskSliceSpace, z) = dyweight(S, z[1], z[2])
    function d2xweight(S::DiskSliceSpace{<:Any, <:Any, T, <:Any}, x, y) where T
        D = S.family
        if D.nparams == 2
            a, b = S.params
            ret1 = a * ((a - 1) * weight(D(a-2, b), x, y)
                                    + 2 * rhoval(x) * dxrhoval(x) * b * weight(D(a-1, b-1), x, y))
            ret2 = (2 * rhoval(x) * dxrhoval(x) * b * (a * weight(D(a-1, b-1), x, y)
                                        + 2 * rhoval(x) * dxrhoval(x) * (b-1) * weight(D(a, b-2), x, y))
                        + 2 * b * (rhoval(x) * d2xrhoval(x) + dxrhoval(x)^2) * weight(D(a, b-1), x, y))
            T(ret1 + ret2)
        else
            a, b, c = S.params
            ret1 = a * ((a-1)*weight(D(a-2,b,c),x,y) - b*weight(D(a-1,b-1,c),x,y) - 2*c*rhoval(x)*dxrhoval(x)*weight(D(a-1,b,c-1),x,y))
            ret2 = b * (-a*weight(D(a-1,b-1,c),x,y) + (b-1)*weight(D(a,b-2,c),x,y) + 2*c*rhoval(x)*dxrhoval(x)*weight(D(a,b-1,c-1),x,y))
            ret3 = 2*c*rhoval(x)*dxrhoval(x) * (-a*weight(D(a-1,b,c-1),x,y) + b*weight(D(a,b-1,c-1),x,y)
                                                        + 2*(c-1)*rhoval(x)*dxrhoval(x)*weight(D(a,b,c-2),x,y))
            ret4 = 2*c*(dxrhoval(x)^2 + rhoval(x)*d2xrhoval(x))*weight(D(a,b,c-1),x,y)
            T(ret1+ret2+ret3+ret4)
        end
    end
    d2xweight(S::DiskSliceSpace, z) = d2xweight(S, z[1], z[2])
    function d2yweight(S::DiskSliceSpace{<:Any, <:Any, T, <:Any}, x, y) where T
        D = S.family
        ret = - 2 * S.params[end] * weight(differentiateweightedspacey(S), x, y)
        ret += 4 * S.params[end] * (S.params[end] - 1) * y^2 * weight(differentiateweightedspacey(differentiateweightedspacey(S)), x, y)
        T(ret)
    end
    d2yweight(S::DiskSliceSpace, z) = d2yweight(S, z[1], z[2])

    a, b, c = 1.0, 1.0, 1.0
    α, β = 0.2, 0.9
    x, y = 0.34, -0.29; z = [x; y]
    N = 15
    # W111->P111
    D = DiskSliceFamily(0.0); S = D(b, c)
    L = laplaceoperator(S, S, N, weighted=true)
    for n = 1:10
        @show n
        u = Fun((x,y)->y*x^(n-1)+y^n, S)
        f = Fun((x,y)->(u(x,y)*d2xweight(S,x,y) + 2*y*(n-1)*x^(n-2)*dxweight(S,x,y) + y*(n-2)*(n-1)*x^(n-3)*weight(S,x,y)
                        + u(x,y)*d2yweight(S,x,y) + 2*(n*y^(n-1) + x^(n-1))*dyweight(S,x,y) + n*(n-1)*y^(n-2)*weight(S,x,y)), S)
        resizecoeffs!(f, N)
        cfs = sparse(L) \ f.coefficients
        @test Fun(S, cfs)(z) ≈ u(z)
    end
    a, b, c = 1.0, 1.0, 1.0
    D = DiskSliceFamily(α, β); S = D(a, b, c)
    L = laplaceoperator(S, S, N, weighted=true)
    for n = 1:10
        @show n
        u = Fun((x,y)->y*x^(n-1)+y^n, S, 200)
        f = Fun((x,y)->(u(x,y)*d2xweight(S,x,y) + 2*y*(n-1)*x^(n-2)*dxweight(S,x,y) + y*(n-2)*(n-1)*x^(n-3)*weight(S,x,y)
                        + u(x,y)*d2yweight(S,x,y) + 2*(n*y^(n-1) + x^(n-1))*dyweight(S,x,y) + n*(n-1)*y^(n-2)*weight(S,x,y)), S, 200)
        resizecoeffs!(f, N)
        cfs = sparse(L) \ f.coefficients
        @test Fun(S, cfs)(z) ≈ u(z)
    end
    # W222->P000
    # TODO
    # P000->P222
    a, b, c = 1.0, 1.0, 1.0
    D = DiskSliceFamily(0.0); S = D(a-1, b-1)
    L = laplaceoperator(S, D(a+1, b+1), N, weighted=false)
    n = 1; u = Fun((x,y)->2y, S); resizecoeffs!(u, N); @test abs(Fun(D(a+1, b+1), L * u.coefficients)(z)) < 1e-12
    for n = 2:10
        @show n
        u = Fun((x,y)->y*x^(n-1)+y^n, S)
        resizecoeffs!(u, N)
        cfs = L * u.coefficients
        @test Fun(D(a+1, b+1), cfs)(z) ≈ y*(n-2)*(n-1)*x^(n-3) + n*(n-1)*y^(n-2)
    end
    a, b, c = 1.0, 1.0, 1.0
    D = DiskSliceFamily(α, β); S = D(a-1, b-1, c-1)
    L = laplaceoperator(S, D(a+1, b+1, c+1), N, weighted=false)
    n = 1; u = Fun((x,y)->2y, S); resizecoeffs!(u, N); @test abs(Fun(D(a+1, b+1, c+1), L * u.coefficients)(z)) < 1e-12
    for n = 2:10
        @show n
        u = Fun((x,y)->y*x^(n-1)+y^n, S, 200)
        resizecoeffs!(u, N)
        cfs = L * u.coefficients
        @test Fun(D(a+1, b+1, c+1), cfs)(z) ≈ y*(n-2)*(n-1)*x^(n-3) + n*(n-1)*y^(n-2)
    end
end


# Trapezium Family/Space testing

@testset "Evaluation for TrapeziumSpace (transform())" begin
    T = TrapeziumFamily()
    a, b, c, d = 2.0, 1.0, 4.0, 1.0
    S = T(a, b, c, d)
    n = 10
    f = (x,y) -> y*x^2 + x
    pts = points(S, n)
    vals = [f(pt...) for pt in pts]
    cfs = transform(S, vals)
    x, y = 0.2, 0.7
    z = [x; y]
    @test T.α ≤ z[1] ≤ T.β && T.γ*T.ρ(z[1]) ≤ z[2] ≤ T.δ*T.ρ(z[1])
    F = Fun(S, cfs)
    F(z)
    f(z...)
    @test F(z) ≈ f(z...)
    @test itransform(S, cfs) ≈ vals
    F = Fun(f, S, 10)
    @test F(z) ≈ f(z...)
    F = Fun(f, S)
    @test F(z) ≈ f(z...)
end

@testset "Jacobi matrices" begin
    n = 10; a, b, c, d = 2.0, 1.0, 4.0, 1.0
    D = TrapeziumFamily(0.3)
    S = D(a, b, c, d)
    x, y = 0.1673, 0.2786; z = [x; y]
    @test (x^2 + y^2 < 1 && D.α ≤ z[1] ≤ D.β && D.γ*D.ρ(z[1]) ≤ z[2] ≤ D.δ*D.ρ(z[1]))
    f = Fun((x,y)->x*y + x, S)
    N = getnk(length(f.coefficients))[1]; resizecoeffs!(f, N+1)
    Jx = jacobix(S, N+1)
    Jy = jacobiy(S, N+1)
    @test evaluate(Jx * f.coefficients, S, z) ≈ z[1] * f(z)
    @test evaluate(Jy * f.coefficients, S, z) ≈ z[2] * f(z)

    a, b, c, d = 2.0, 1.0, 4.0, 1.0
    D = TrapeziumFamily(0.6)
    S = D(a, b, c, d)
    x, y = 0.1673, 0.2786; z = [x; y]
    @test (x^2 + y^2 < 1 && D.α ≤ z[1] ≤ D.β && D.γ*D.ρ(z[1]) ≤ z[2] ≤ D.δ*D.ρ(z[1]))
    N = 30
    Jx = jacobix(S, N+1)
    Jy = jacobiy(S, N+1)
    for n = 0:N, k = 0:n
        m = getopindex(n, k)
        p = Fun(S, [zeros(m-1); 1]); resizecoeffs!(p, n+1)
        len = length(p.coefficients)
        @test evaluate(Jx[1:len, 1:len] * p.coefficients, S, z) ≈ z[1] * p(z)
        @test evaluate(Jy[1:len, 1:len] * p.coefficients, S, z) ≈ z[2] * p(z)
    end
end

@testset "Operator Clenshaw" begin
    a, b, c, d = 2.0, 1.0, 4.0, 1.0
    D = TrapeziumFamily(0.6)
    S = D(a, b, c, d)
    x, y = 0.1673, 0.2786; z = [x; y]
    @test (x^2 + y^2 < 1 && D.α ≤ z[1] ≤ D.β && D.γ*D.ρ(z[1]) ≤ z[2] ≤ D.δ*D.ρ(z[1]))
    N = 14
    oper = Fun((x,y)->cos(x)*y^2, S, 200)
    A = operatorclenshaw(oper, S)
    f = Fun((x,y)->x*y, S); resizecoeffs!(f, N)
    foper = Fun(S, A*f.coefficients)
    @test foper(z) ≈ f(z) * oper(z)
end

@testset "Increment/decrement parameters operators" begin
    tol = 1e-13

    paramslist = [(1,0,0,0), (1,1,0,0), (1,1,1,0), (1,1,1,1),
                  (0,1,0,0), (0,1,1,0), (0,1,1,1), (1,0,1,0),
                  (0,0,1,0), (0,0,1,1), (1,0,1,1), (0,1,0,1),
                  (0,0,0,1), (1,0,0,1), (1,1,0,1)]
    maxcount = length(paramslist)
    for count = 1:maxcount
        params = paramslist[count]
        @show count, maxcount, params
        a, b, c, d = 2.0, 1.0, 4.0, 3.0
        D = TrapeziumFamily(0.6)
        S = D(a, b, c, d)
        x, y = 0.1673, 0.2786; z = [x; y]
        @test (x^2 + y^2 < 1 && D.α ≤ z[1] ≤ D.β && D.γ*D.ρ(z[1]) ≤ z[2] ≤ D.δ*D.ρ(z[1]))
        St = (S.family)(S.params.+params)
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
    for count = 1:maxcount
        params = paramslist[count]
        @show count, maxcount, params
        a, b, c, d = 2.0, 1.0, 4.0, 3.0
        D = TrapeziumFamily(0.6)
        S = D(a, b, c, d)
        x, y = 0.1673, 0.2786; z = [x; y]
        @test (x^2 + y^2 < 1 && D.α ≤ z[1] ≤ D.β && D.γ*D.ρ(z[1]) ≤ z[2] ≤ D.δ*D.ρ(z[1]))
        St = (S.family)(S.params.-params)
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

@testset "Evaluate partial derivative of (random and known) function" begin
    a, b, c, d = 2.0, 1.0, 4.0, 1.0
    D = TrapeziumFamily(0.6)
    S = D(a, b, c, d)
    x, y = 0.1673, 0.2786; z = [x; y]
    @test (x^2 + y^2 < 1 && D.α ≤ z[1] ≤ D.β && D.γ*D.ρ(z[1]) ≤ z[2] ≤ D.δ*D.ρ(z[1]))
    N, N1 = 10, 5
    f = Fun(S, [randn(sum(1:N1)); 0.01*randn(sum(N1+1:N+1))])
    # NOTE: This test for DX/DY operators nay not be good enough to ensure
    # it is "correct"
    h = 1e-6; @test (f(x+h,y)-f(x,y))/h ≈ differentiatex(f, f.space)(x,y) atol=100h
    h = 1e-6; @test (f(x,y+h)-f(x,y))/h ≈ differentiatey(f, f.space)(x,y) atol=100h
    f = Fun((x,y)->y*sin(x), S)
    h = 1e-5; @test y*cos(x) ≈ differentiatex(f, f.space)(x,y) atol=100h
    f = Fun((x,y)->x*sin(y), S)
    h = 1e-5; @test x*cos(y) ≈ differentiatey(f, f.space)(x,y) atol=100h
end

@testset "laplaceoperator" begin
    # NOTE: The derivative weight functions only valid for S.params == (1,1,1,1)
    rhoval(D::TrapeziumFamily, x) = 1 - D.slope * x
    dxrhoval(D::TrapeziumFamily, x) = - D.slope
    d2xrhoval(D::TrapeziumFamily, x) = 0.0
    function dxweight(S::TrapeziumSpace{<:Any, <:Any, T, <:Any}, x, y) where T
        D = S.family
        a, b, c, d = S.params
        ret = (b * weight(D(a, b-1, c, d), x, y)
                - a * weight(D(a-1, b, c, d), x, y)
                + d * D.δ * dxrhoval(D, x) * weight(D(a, b, c, d-1), x, y))
        T(ret)
    end
    dxweight(S::TrapeziumSpace, z) = dxweight(S, z[1], z[2])
    function dyweight(S::TrapeziumSpace{<:Any, <:Any, T, <:Any}, x, y) where T
        D = S.family
        a, b, c, d = S.params
        ret = c * weight(D(a, b, c-1, d), x, y) - d * weight(D(a, b, c, d-1), x, y)
        T(ret)
    end
    dyweight(S::TrapeziumSpace, z) = dyweight(S, z[1], z[2])
    function d2xweight(S::TrapeziumSpace{<:Any, <:Any, T, <:Any}, x, y) where T
        D = S.family
        a, b, c, d = S.params
        ret = (- 2 * a * b * weight(D(a-1, b-1, c, d), x, y)
               + 2 * d * D.δ * dxrhoval(D, x) * (b * weight(D(a, b-1, c, d-1), x, y)
                                                 - a * weight(D(a-1, b, c, d-1), x, y)))

        T(ret)
    end
    d2xweight(S::TrapeziumSpace, z) = d2xweight(S, z[1], z[2])
    function d2yweight(S::TrapeziumSpace{<:Any, <:Any, T, <:Any}, x, y) where T
        D = S.family
        a, b, c, d = S.params
        ret = - 2 * c * d * weight(D(a, b, c-1, d-1), x, y)
        T(ret)
    end
    d2yweight(S::TrapeziumSpace, z) = d2yweight(S, z[1], z[2])

    a, b, c, d = 1.0, 1.0, 1.0, 1.0
    D = TrapeziumFamily(0.6)
    S = D(a, b, c, d)
    x, y = 0.1673, 0.2786; z = [x; y]
    @test (D.α ≤ z[1] ≤ D.β && D.γ*D.ρ(z[1]) ≤ z[2] ≤ D.δ*D.ρ(z[1]))
    N = 15
    # W111->P111
    L = laplaceoperator(S, S, N, weighted=true)
    for n = 1:10
        @show n
        u = Fun((x,y)->y*x^(n-1)+y^n, S)
        f = Fun((x,y)->(u(x,y)*d2xweight(S,x,y) + 2*y*(n-1)*x^(n-2)*dxweight(S,x,y) + y*(n-2)*(n-1)*x^(n-3)*weight(S,x,y)
                        + u(x,y)*d2yweight(S,x,y) + 2*(n*y^(n-1) + x^(n-1))*dyweight(S,x,y) + n*(n-1)*y^(n-2)*weight(S,x,y)), S)
        resizecoeffs!(f, N)
        cfs = sparse(L) \ f.coefficients
        @test Fun(S, cfs)(z) ≈ u(z)
    end
end
