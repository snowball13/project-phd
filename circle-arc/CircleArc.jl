# Circle arc module
module CircleArc

    using ApproxFun
    using BlockBandedMatrices
    using BlockArrays
    using Base.Test

    export Jx, Jy, innerTh, innerUh, initialise_arc_ops, get_arc_clenshaw_matrices,
            arc_op_eval, arc_op_derivative_eval, arc_func_eval, Q_func_eval,
            func_to_coeffs, func_to_Q_coeffs, arc2Q, Q2arc, arc_derivative_operator,
            arc_derivative_operator_in_Q, get_op_in_Q_basis_coeff_mats

    include("ArcOPs.jl")

end
