module SphericalHarmonics

    export Jx, Jy, Jz, sh_eval, func_eval, func_eval_operator, func_eval_jacobi,
            grad_Jx, grad_Jy, grad_Jz, grad_sh, grad_perp_sh, laplacian_sh, tangent_basis_eval,
            tangent_func_eval, func_eval_grad_jacobi

    include("SphericalHarmonicsScalar.jl")
    include("SphericalHarmonicsTangent.jl")

end
