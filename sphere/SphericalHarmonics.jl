module SphericalHarmonics

    using BlockBandedMatrices
    using BlockArrays
    using Base.Test
    using Base.LinAlg

    export Jx, Jy, Jz, sh_eval, func_eval, func_eval_operator, func_eval_jacobi,
            grad_Jx, grad_Jy, grad_Jz, grad_sh, grad_perp_sh, laplacian_sh, tangent_basis_eval,
            get_clenshaw_matrices, tangent_func_eval, func_eval_grad_jacobi

    include("SphericalHarmonicsScalar.jl")
    include("SphericalHarmonicsTangent.jl")

end


using SO, FastTransforms

l = 1; m = 1;
    f = (θ,φ) -> exp(im*m*φ)*(iseven(m) ? cos(l*θ) : sin((l+1)*θ))
    n = 20
    θ = (0.5:n-0.5)/n * π
    φ = (0:2n-2)*2/(2n-1) * π
    F = [f(θ,φ) for θ in θ, φ in φ]
    V = zero(F)
    A_mul_B!(V, FastTransforms.plan_analysis(F), F)
    V |> chopm


l = 5; m = 0;
    f = (θ,φ) -> sh_eval(l, m, sin(θ)*cos(φ), sin(θ)sin(φ), cos(θ))
    n = 20
    θ = (0.5:n-0.5)/n * π
    φ = (0:2n-2)*2/(2n-1) * π
    F = [f(θ,φ) for θ in θ, φ in φ]
    V = zero(F)
    A_mul_B!(V, FastTransforms.plan_analysis(F), F)
    V |> fourier2sph |> chopm



f = (θ,φ) -> sh_eval(0, 0, sin(θ)*cos(φ), sin(θ)sin(φ), cos(θ)) +
                2sh_eval(1, 0, sin(θ)*cos(φ), sin(θ)sin(φ), cos(θ)) +
                3sh_eval(2, 0, sin(θ)*cos(φ), sin(θ)sin(φ), cos(θ)) +
                4sh_eval(1, 1, sin(θ)*cos(φ), sin(θ)sin(φ), cos(θ)) +
                5sh_eval(2, 1, sin(θ)*cos(φ), sin(θ)sin(φ), cos(θ)) +
                6sh_eval(1, -1, sin(θ)*cos(φ), sin(θ)sin(φ), cos(θ)) +
                7sh_eval(2, 2, sin(θ)*cos(φ), sin(θ)sin(φ), cos(θ)) +
                8sh_eval(3, 2, sin(θ)*cos(φ), sin(θ)sin(φ), cos(θ)) +
                9sh_eval(2, -2, sin(θ)*cos(φ), sin(θ)sin(φ), cos(θ)) +
                10sh_eval(3, -2, sin(θ)*cos(φ), sin(θ)sin(φ), cos(θ)) +
                11sh_eval(3, 3, sin(θ)*cos(φ), sin(θ)sin(φ), cos(θ)) +
                12sh_eval(4, 3, sin(θ)*cos(φ), sin(θ)sin(φ), cos(θ)) +
                13sh_eval(3, -3, sin(θ)*cos(φ), sin(θ)sin(φ), cos(θ)) +
                14sh_eval(4, -3, sin(θ)*cos(φ), sin(θ)sin(φ), cos(θ)) +
                15sh_eval(4, 4, sin(θ)*cos(φ), sin(θ)sin(φ), cos(θ)) +
                16sh_eval(5, 4, sin(θ)*cos(φ), sin(θ)sin(φ), cos(θ))
    n = 20
    θ = (0.5:n-0.5)/n * π
    φ = (0:2n-2)*2/(2n-1) * π
    F = [f(θ,φ) for θ in θ, φ in φ]
    V = zero(F)
    A_mul_B!(V, FastTransforms.plan_analysis(F), F)
    V |> fourier2sph |> chopm
    M = size(V,2)
    P = eye(Complex128,M)
    for k = 2:4:M
        P[k:k+1,k:k+1] = [im im;
                      -1 1]/sqrt(2)
      end
      for k = 4:4:M
          P[k:k+1,k:k+1] = [-im im;
                        1 1]/sqrt(2)
        end
        (V |> fourier2sph)          * P |> chopm


function function2sph(f, n)
    n = 20
    θ = (0.5:n-0.5)/n * π
    φ = (0:2n-2)*2/(2n-1) * π
    F = [f(θ,φ) for θ in θ, φ in φ]
    V = zero(F)
    A_mul_B!(V, FastTransforms.plan_analysis(F), F)
    V |> fourier2sph |> chopm
    M = size(V,2)
    P = eye(Complex128,M)
    for k = 2:4:M
        P[k:k+1,k:k+1] = [im im;
                      -1 1]/sqrt(2)
      end
      for k = 4:4:M
          P[k:k+1,k:k+1] = [-im im;
                        1 1]/sqrt(2)
        end
        (V |> fourier2sph)          * P
end



sh_eval
sphevaluate(θ,ϕ,2,2)


l = 1; m = 1;
    f =
    n = 20
    θ = (0.5:n-0.5)/n * π
    φ = (0:2n-2)*2/(2n-1) * π
    F = [f(θ,φ) for θ in θ, φ in φ]
    V = zero(F)
    A_mul_B!(V, FastTransforms.plan_analysis(F), F)
    V |> chopm


V |> |> chopm

function doubleFeval(G, θ, φ)
    ret = 0
    for k = 1:size(G,1), j=1:size(G,2)
       l = k-1
       m = (isodd(j) ? 1 : (-1)) * (j ÷ 2 )
       ret += G[k,j] * exp(im*m*φ)/sqrt(2π) * (iseven(m) ? cos(l*θ) : sin((l+1)*θ))
    end
    ret
end

F = zeros(10,10)
   m = -1; l = 3;
   j = (m ≥ 0 ? 2m+1 : 2(-m))
   k = l+1-abs(m)
   F[k,j] = 1
   G = sph2fourier(F)




doubleFeval(V, 0.1, 0.2)

f(0.1,0.2)



@which A_mul_B!(V, FastTransforms.plan_analysis(F), F)
