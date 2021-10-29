#pragma once

#include <panoc-alm/inner/decl/panoc-stop-crit.hpp>
#include <panoc-alm/util/atomic_stop_signal.hpp>
#include <panoc-alm/util/problem.hpp>
#include <panoc-alm/util/solverstatus.hpp>

#include <stdexcept>

namespace pa::detail {

/// Calculate both ψ(x) and the vector ŷ that can later be used to compute ∇ψ.
/// @f[ \psi(x^k) = f(x^k) + \frac{1}{2}
/// \text{dist}_\Sigma^2\left(g(x^k) + \Sigma^{-1}y,\;D\right) @f]
/// @f[ \hat{y}  @f]
inline real_t calc_ψ_ŷ(const Problem &p, ///< [in]  Problem description
                       crvec x,          ///< [in]  Decision variable @f$ x @f$
                       crvec y, ///< [in]  Lagrange multipliers @f$ y @f$
                       crvec Σ, ///< [in]  Penalty weights @f$ \Sigma @f$
                       rvec ŷ   ///< [out] @f$ \hat{y} @f$
) {
    // g(x)
    p.g(x, ŷ);
    // ζ = g(x) + Σ⁻¹y
    ŷ += Σ.asDiagonal().inverse() * y;
    // d = ζ - Π(ζ, D)
    ŷ = projecting_difference(ŷ, p.D);
    // dᵀŷ, ŷ = Σ d
    real_t dᵀŷ = 0;
    for (unsigned i = 0; i < p.m; ++i) {
        dᵀŷ += ŷ(i) * Σ(i) * ŷ(i); // TODO: vectorize
        ŷ(i) = Σ(i) * ŷ(i);
    }
    // ψ(x) = f(x) + ½ dᵀŷ
    real_t ψ = p.f(x) + 0.5 * dᵀŷ;

    return ψ;
}

/// Calculate both ψ(x) and the vector ŷ that can later be used to compute ∇ψ.
/// @f[ \psi(x^k) = f(x^k) + \frac{1}{2}
/// \text{dist}_\Sigma^2\left(g(x^k) + \Sigma^{-1}y,\;D\right) @f]
/// @f[ \hat{y}  @f]
inline real_t calc_ψ_ŷ(const ProblemFull &p, ///< [in]  Problem description
                       crvec x,          ///< [in]  Decision variable @f$ x @f$
                       crvec y, ///< [in]  Lagrange multipliers @f$ y @f$
                       crvec Σ1, ///< [in]  Penalty weights @f$ \Sigma_1 @f$
                       crvec Σ2, ///< [in]  Penalty weights @f$ \Sigma_2 @f$
                       rvec ŷ1,   ///< [out] @f$ \hat{y}_1 @f$
                       rvec ŷ2   ///< [out] @f$ \hat{y}_2 @f$
) {
    // g1(x)
    p.g1(x, ŷ1);
    // ζ = g1(x) + Σ1⁻¹y
    ŷ1 += Σ1.asDiagonal().inverse() * y;
    // d1 = ζ - Π(ζ, D1)
    ŷ1 = projecting_difference(ŷ1, p.D1);
    // dᵀŷ1, ŷ1 = Σ1 d1
    real_t dᵀŷ1 = 0;
    for (unsigned i = 0; i < p.m1; ++i) {
        dᵀŷ1 += ŷ1(i) * Σ1(i) * ŷ1(i); // TODO: vectorize
        ŷ1(i) = Σ1(i) * ŷ1(i);
    }
    // g2(x)
    p.g2(x, ŷ2);
    // d2 = g2(x) - Π(g2(x), D2)
    ŷ2 = projecting_difference(ŷ2, p.D2);
    // dᵀŷ2, ŷ2 = Σ2 d2
    real_t dᵀŷ2 = 0;
    for (unsigned i = 0; i < p.m2; ++i) {
        dᵀŷ2 += ŷ2(i) * Σ2(i) * ŷ2(i); // TODO: vectorize
        ŷ2(i) = Σ2(i) * ŷ2(i);
    }
    // ψ(x) = f(x) + ½ dᵀŷ
    real_t ψ = p.f(x) + 0.5 * dᵀŷ1 + 0.5 * dᵀŷ2;

    return ψ;
}

/// Calculate ∇ψ(x) using ŷ.
inline void calc_grad_ψ_from_ŷ(const Problem &p, ///< [in]  Problem description
                               crvec x, ///< [in]  Decision variable @f$ x @f$
                               crvec ŷ, ///< [in]  @f$ \hat{y} @f$
                               rvec grad_ψ, ///< [out] @f$ \nabla \psi(x) @f$
                               rvec work_n  ///<       Dimension n
) {
    // ∇ψ = ∇f(x) + ∇g(x) ŷ
    p.grad_f(x, grad_ψ);
    p.grad_g_prod(x, ŷ, work_n);
    grad_ψ += work_n;
}

/// Calculate ∇ψ(x) using ŷ.
inline void calc_grad_ψ_from_ŷ(const ProblemFull &p, ///< [in]  Problem description
                               crvec x, ///< [in]  Decision variable @f$ x @f$
                               crvec ŷ1, ///< [in]  @f$ \hat{y} @f$
                               crvec ŷ2, ///< [in]  @f$ \hat{y} @f$
                               rvec grad_ψ, ///< [out] @f$ \nabla \psi(x) @f$
                               rvec work_n  ///<       Dimension n
) {
    // ∇ψ = ∇f(x) + ∇g1(x) ŷ1 + ∇g2(x) ŷ2
    p.grad_f(x, grad_ψ);
    p.grad_g1_prod(x, ŷ1, work_n);
    grad_ψ += work_n;
    p.grad_g2_prod(x, ŷ2, work_n);
    grad_ψ += work_n;
}

/// Calculate both ψ(x) and its gradient ∇ψ(x).
/// @f[ \psi(x^k) = f(x^k) + \frac{1}{2}
/// \text{dist}_\Sigma^2\left(g(x^k) + \Sigma^{-1}y,\;D\right) @f]
/// @f[ \nabla \psi(x) = \nabla f(x) + \nabla g(x)\ \hat{y}(x) @f]
inline real_t calc_ψ_grad_ψ(const Problem &p, ///< [in]  Problem description
                            crvec x, ///< [in]  Decision variable @f$ x @f$
                            crvec y, ///< [in]  Lagrange multipliers @f$ y @f$
                            crvec Σ, ///< [in]  Penalty weights @f$ \Sigma @f$
                            rvec grad_ψ, ///< [out] @f$ \nabla \psi(x) @f$
                            rvec work_n, ///<       Dimension n
                            rvec work_m  ///<       Dimension m
) {
    // ψ(x) = f(x) + ½ dᵀŷ
    real_t ψ = calc_ψ_ŷ(p, x, y, Σ, work_m);
    // ∇ψ = ∇f(x) + ∇g(x) ŷ
    calc_grad_ψ_from_ŷ(p, x, work_m, grad_ψ, work_n);
    return ψ;
}

/// Calculate both ψ(x) and its gradient ∇ψ(x).
/// @f[ \psi(x^k) = f(x^k) + \frac{1}{2}
/// \text{dist}_\Sigma^2\left(g(x^k) + \Sigma^{-1}y,\;D\right) @f]
/// @f[ \nabla \psi(x) = \nabla f(x) + \nabla g(x)\ \hat{y}(x) @f]
inline real_t calc_ψ_grad_ψ(const ProblemFull &p, ///< [in]  Problem description
                            crvec x,  ///< [in]  Decision variable @f$ x @f$
                            crvec y,  ///< [in]  Lagrange multipliers @f$ y @f$
                            crvec Σ1, ///< [in]  Penalty weights @f$ \Sigma_1 @f$
                            crvec Σ2, ///< [in]  Penalty weights @f$ \Sigma_2 @f$
                            rvec grad_ψ, ///< [out] @f$ \nabla \psi(x) @f$
                            rvec work_n, ///<       Dimension n
                            rvec work_m1,///<       Dimension m1
                            rvec work_m2 ///<       Dimension m2
) {
    // ψ(x) = f(x) + ½ dᵀŷ
    real_t ψ = calc_ψ_ŷ(p, x, y, Σ1, Σ2, work_m1, work_m2);
    // ∇ψ = ∇f(x) + ∇g(x) ŷ
    calc_grad_ψ_from_ŷ(p, x, work_m1, work_m2, grad_ψ, work_n);
    return ψ;
}

/// Calculate the gradient ∇ψ(x).
/// @f[ \nabla \psi(x) = \nabla f(x) + \nabla g(x)\ \hat{y}(x) @f]
inline void calc_grad_ψ(const Problem &p, ///< [in]  Problem description
                        crvec x,          ///< [in]  Decision variable @f$ x @f$
                        crvec y,     ///< [in]  Lagrange multipliers @f$ y @f$
                        crvec Σ,     ///< [in]  Penalty weights @f$ \Sigma @f$
                        rvec grad_ψ, ///< [out] @f$ \nabla \psi(x) @f$
                        rvec work_n, ///<       Dimension n
                        rvec work_m  ///<       Dimension m
) {
    // g(x)
    p.g(x, work_m);
    // ζ = g(x) + Σ⁻¹y
    work_m += (y.array() / Σ.array()).matrix();
    // d = ζ - Π(ζ, D)
    work_m = projecting_difference(work_m, p.D);
    // ŷ = Σ d
    work_m = Σ.asDiagonal() * work_m;

    // ∇ψ = ∇f(x) + ∇g(x) ŷ
    p.grad_f(x, grad_ψ);
    p.grad_g_prod(x, work_m, work_n);
    grad_ψ += work_n;
}

/// Calculate the gradient ∇ψ(x).
/// @f[ \nabla \psi(x) = \nabla f(x) + \nabla g(x)\ \hat{y}(x) @f]
inline void calc_grad_ψ(const ProblemFull &p, ///< [in]  Problem description
                        crvec x,          ///< [in]  Decision variable @f$ x @f$
                        crvec y,     ///< [in]  Lagrange multipliers @f$ y @f$
                        crvec Σ1,    ///< [in]  Penalty weights @f$ \Sigma_1 @f$
                        crvec Σ2,    ///< [in]  Penalty weights @f$ \Sigma_2 @f$
                        rvec grad_ψ, ///< [out] @f$ \nabla \psi(x) @f$
                        rvec work_n, ///<       Dimension n
                        rvec work_m1,///<       Dimension m_1
                        rvec work_m2 ///<       Dimension m_2
) {
    // g1(x)
    p.g1(x, work_m1);
    // ζ = g1(x) + Σ1⁻¹y
    work_m1 += (y.array() / Σ1.array()).matrix();
    // d1 = ζ - Π(ζ, D1)
    work_m1 = projecting_difference(work_m1, p.D1);
    // ŷ1 = Σ1 d1
    work_m1 = Σ1.asDiagonal() * work_m1;

    // g2(x)
    p.g2(x, work_m2);
    // d1 = g2(x) - Π(g2(x), D2)
    work_m2 = projecting_difference(work_m2, p.D2);
    // ŷ2 = Σ2 d2
    work_m2 = Σ2.asDiagonal() * work_m2;

    // ∇ψ = ∇f(x) + ∇g1(x) ŷ1 + ∇g2(x) ŷ2
    p.grad_f(x, grad_ψ);
    p.grad_g1_prod(x, work_m1, work_n);
    grad_ψ += work_n;
    p.grad_g2_prod(x, work_m2, work_n);
    grad_ψ += work_n;
}

/// Calculate the error between ẑ and g(x).
/// @f[ \hat{z}^k = \Pi_D\left(g(x^k) + \Sigma^{-1}y\right) @f]
inline void calc_err_z(const Problem &p, ///< [in]  Problem description
                       crvec x̂,   ///< [in]  Decision variable @f$ \hat{x} @f$
                       crvec y,   ///< [in]  Lagrange multipliers @f$ y @f$
                       crvec Σ,   ///< [in]  Penalty weights @f$ \Sigma @f$
                       rvec err_z ///< [out] @f$ g(\hat{x}) - \hat{z} @f$
) {
    // g(x̂)
    p.g(x̂, err_z);
    // ζ = g(x̂) + Σ⁻¹y
    // ẑ = Π(ζ, D)
    // g(x) - ẑ
    err_z = err_z - project(err_z + Σ.asDiagonal().inverse() * y, p.D);
    // TODO: catastrophic cancellation?
}

/// Calculate the error between ẑ and g(x).
/// @f[ \hat{z}^k = \Pi_D\left(g(x^k) + \Sigma^{-1}y\right) @f]
inline void calc_err_z(const ProblemFull &p, ///< [in]  Problem description
                       crvec x̂,   ///< [in]  Decision variable @f$ \hat{x} @f$
                       crvec y,   ///< [in]  Lagrange multipliers @f$ y @f$
                       crvec Σ1,   ///< [in]  Penalty weights @f$ \Sigma_1 @f$
                       rvec err_z1,///< [out] @f$ g_1(\hat{x}) - \hat{z}_1 @f$
                       rvec err_z2 ///< [out] @f$ g_2(\hat{x}) - \hat{z}_2 @f$
) {
    // g1(x̂)
    p.g1(x̂, err_z1);
    // ζ = g1(x̂) + Σ1⁻¹y
    // ẑ1 = Π(ζ, D)
    // g1(x) - ẑ1
    err_z1 = err_z1 - project(err_z1 + Σ1.asDiagonal().inverse() * y, p.D1);

    // g2(x̂)
    p.g2(x̂, err_z2);
    // ẑ2 = Π(g2(x̂), D2)
    // g2(x) - ẑ2
    err_z2 = err_z2 - project(err_z2, p.D2);
}

/**
 * Projected gradient step
 * @f[ \begin{aligned} 
 * \hat{x}^k &= T_{\gamma^k}\left(x^k\right) \\ 
 * &= \Pi_C\left(x^k - \gamma^k \nabla \psi(x^k)\right) \\ 
 * p^k &= \hat{x}^k - x^k \\ 
 * \end{aligned} @f]
 */
inline auto
projected_gradient_step(const Box &C, ///< [in]  Set to project onto
                        real_t γ,     ///< [in]  Step size
                        crvec x,      ///< [in]  Decision variable @f$ x @f$
                        crvec grad_ψ  ///< [in]  @f$ \nabla \psi(x^k) @f$
) {
    using binary_real_f = real_t (*)(real_t, real_t);
    return (-γ * grad_ψ)
        .binaryExpr(C.lowerbound - x, binary_real_f(std::fmax))
        .binaryExpr(C.upperbound - x, binary_real_f(std::fmin));
}

template <typename ProblemT> 
inline void calc_x̂(const ProblemT &prob, ///< [in]  Problem description
                   real_t γ,            ///< [in]  Step size
                   crvec x,             ///< [in]  Decision variable @f$ x @f$
                   crvec grad_ψ,        ///< [in]  @f$ \nabla \psi(x^k) @f$
                   rvec x̂, ///< [out] @f$ \hat{x}^k = T_{\gamma^k}(x^k) @f$
                   rvec p  ///< [out] @f$ \hat{x}^k - x^k @f$
) {
    p = projected_gradient_step(prob.C, γ, x, grad_ψ);
    x̂ = x + p;
}

/// Compute the ε from the stopping criterion, see @ref PANOCStopCrit.
inline real_t calc_error_stop_crit(
    PANOCStopCrit crit, ///< [in]  What stoppint criterion to use
    crvec pₖ,      ///< [in]  Projected gradient step @f$ \hat x^k - x^k @f$
    real_t γ,      ///< [in]  Step size
    crvec xₖ,      ///< [in]  Current iterate
    crvec grad_̂ψₖ, ///< [in]  Gradient in @f$ \hat x^k @f$
    crvec grad_ψₖ, ///< [in]  Gradient in @f$ x^k @f$
    const Box &C   ///< [in]  Feasible set @f$ C @f$
) {
    switch (crit) {
        case PANOCStopCrit::ApproxKKT: {
            auto err = (1 / γ) * pₖ + (grad_ψₖ - grad_̂ψₖ);
            // These parentheses     ^^^               ^^^     are important to
            // prevent catastrophic cancellation when the step is small
            return vec_util::norm_inf(err);
        }
        case PANOCStopCrit::ApproxKKT2: {
            auto err = (1 / γ) * pₖ + (grad_ψₖ - grad_̂ψₖ);
            // These parentheses     ^^^               ^^^     are important to
            // prevent catastrophic cancellation when the step is small
            return err.norm();
        }
        case PANOCStopCrit::ProjGradNorm: {
            return vec_util::norm_inf(pₖ);
        }
        case PANOCStopCrit::ProjGradNorm2: {
            return pₖ.norm();
        }
        case PANOCStopCrit::ProjGradUnitNorm: {
            return vec_util::norm_inf(
                projected_gradient_step(C, 1, xₖ, grad_ψₖ));
        }
        case PANOCStopCrit::ProjGradUnitNorm2: {
            return projected_gradient_step(C, 1, xₖ, grad_ψₖ).norm();
        }
        case PANOCStopCrit::FPRNorm: {
            return vec_util::norm_inf(pₖ) / γ;
        }
        case PANOCStopCrit::FPRNorm2: {
            return pₖ.norm() / γ;
        }
    }
    throw std::out_of_range("Invalid PANOCStopCrit");
}

/// Increase the estimate of the Lipschitz constant of the objective gradient
/// and decrease the step size until quadratic upper bound or descent lemma is
/// satisfied:
/// @f[ \psi(x + p) \le \psi(x) + \nabla\psi(x)^\top p + \frac{L}{2} \|p\|^2 @f]
/// The projected gradient iterate @f$ \hat x^k @f$ and step @f$ p^k @f$ are
/// updated with the new step size @f$ \gamma^k @f$, and so are other quantities
/// that depend on them, such as @f$ \nabla\psi(x^k)^\top p^k @f$ and
/// @f$ \|p\|^2 @f$. The intermediate vector @f$ \hat y(x^k) @f$ can be used to
/// compute the gradient @f$ \nabla\psi(\hat x^k) @f$ or to update the Lagrange
/// multipliers.
///
/// @return The original step size, before it was reduced by this function.
inline real_t descent_lemma(
    /// [in]  Problem description
    const Problem &problem,
    /// [in]    Tolerance used to ignore rounding errors when the function
    ///         @f$ \psi(x) @f$ is relatively flat or the step size is very
    ///         small, which could cause @f$ \psi(x^k) < \psi(\hat x^k) @f$,
    ///         which is mathematically impossible but could occur in finite
    ///         precision floating point arithmetic.
    real_t rounding_tolerance,
    /// [in]    Maximum allowed Lipschitz constant estimate (prevents infinite
    ///         loop if function or derivatives are discontinuous, and keeps
    ///         step size bounded away from zero).
    real_t L_max,
    /// [in]    Current iterate @f$ x^k @f$
    crvec xₖ,
    /// [in]    Objective function @f$ \psi(x^k) @f$
    real_t ψₖ,
    /// [in]    Gradient of objective @f$ \nabla\psi(x^k) @f$
    crvec grad_ψₖ,
    /// [in]    Lagrange multipliers @f$ y @f$
    crvec y,
    /// [in]    Penalty weights @f$ \Sigma @f$
    crvec Σ,
    /// [out]   Projected gradient iterate @f$ \hat x^k @f$
    rvec x̂ₖ,
    /// [out]   Projected gradient step @f$ p^k @f$
    rvec pₖ,
    /// [out]   Intermediate vector @f$ \hat y(x^k) @f$
    rvec ŷx̂ₖ,
    /// [inout] Objective function @f$ \psi(\hat x^k) @f$
    real_t &ψx̂ₖ,
    /// [inout] Squared norm of the step @f$ \left\| p^k \right\|^2 @f$
    real_t &norm_sq_pₖ,
    /// [inout] Gradient of objective times step @f$ \nabla\psi(x^k)^\top p^k@f$
    real_t &grad_ψₖᵀpₖ,
    /// [inout] Lipschitz constant estimate @f$ L_{\nabla\psi}^k @f$
    real_t &Lₖ,
    /// [inout] Step size @f$ \gamma^k @f$
    real_t &γₖ) {

    real_t old_γₖ = γₖ;
    real_t margin = (1 + std::abs(ψₖ)) * rounding_tolerance;
    while (ψx̂ₖ - ψₖ > grad_ψₖᵀpₖ + 0.5 * Lₖ * norm_sq_pₖ + margin) {
        if (not(Lₖ * 2 <= L_max))
            break;

        Lₖ *= 2;
        γₖ /= 2;

        // Calculate x̂ₖ and pₖ (with new step size)
        calc_x̂(problem, γₖ, xₖ, grad_ψₖ, /* in ⟹ out */ x̂ₖ, pₖ);
        // Calculate ∇ψ(xₖ)ᵀpₖ and ‖pₖ‖²
        grad_ψₖᵀpₖ = grad_ψₖ.dot(pₖ);
        norm_sq_pₖ = pₖ.squaredNorm();

        // Calculate ψ(x̂ₖ) and ŷ(x̂ₖ)
        ψx̂ₖ = calc_ψ_ŷ(problem, x̂ₖ, y, Σ, /* in ⟹ out */ ŷx̂ₖ);
    }
    return old_γₖ;
}

/// Increase the estimate of the Lipschitz constant of the objective gradient
/// and decrease the step size until quadratic upper bound or descent lemma is
/// satisfied:
/// @f[ \psi(x + p) \le \psi(x) + \nabla\psi(x)^\top p + \frac{L}{2} \|p\|^2 @f]
/// The projected gradient iterate @f$ \hat x^k @f$ and step @f$ p^k @f$ are
/// updated with the new step size @f$ \gamma^k @f$, and so are other quantities
/// that depend on them, such as @f$ \nabla\psi(x^k)^\top p^k @f$ and
/// @f$ \|p\|^2 @f$. The intermediate vector @f$ \hat y(x^k) @f$ can be used to
/// compute the gradient @f$ \nabla\psi(\hat x^k) @f$ or to update the Lagrange
/// multipliers.
///
/// @return The original step size, before it was reduced by this function.
inline real_t descent_lemma(
    /// [in]  Problem description
    const ProblemFull &problem,
    /// [in]    Tolerance used to ignore rounding errors when the function
    ///         @f$ \psi(x) @f$ is relatively flat or the step size is very
    ///         small, which could cause @f$ \psi(x^k) < \psi(\hat x^k) @f$,
    ///         which is mathematically impossible but could occur in finite
    ///         precision floating point arithmetic.
    real_t rounding_tolerance,
    /// [in]    Maximum allowed Lipschitz constant estimate (prevents infinite
    ///         loop if function or derivatives are discontinuous, and keeps
    ///         step size bounded away from zero).
    real_t L_max,
    /// [in]    Current iterate @f$ x^k @f$
    crvec xₖ,
    /// [in]    Objective function @f$ \psi(x^k) @f$
    real_t ψₖ,
    /// [in]    Gradient of objective @f$ \nabla\psi(x^k) @f$
    crvec grad_ψₖ,
    /// [in]    Lagrange multipliers @f$ y @f$
    crvec y,
    /// [in]    Penalty weights @f$ \Sigma_1 @f$
    crvec Σ1,
    /// [in]    Penalty weights @f$ \Sigma_2 @f$
    crvec Σ2,
    /// [out]   Projected gradient iterate @f$ \hat x^k @f$
    rvec x̂ₖ,
    /// [out]   Projected gradient step @f$ p^k @f$
    rvec pₖ,
    /// [out]   Intermediate vector @f$ \hat y(x^k)1 @f$
    rvec ŷx̂ₖ1,
    /// [out]   Intermediate vector @f$ \hat y(x^k)2 @f$
    rvec ŷx̂ₖ2,
    /// [inout] Objective function @f$ \psi(\hat x^k) @f$
    real_t &ψx̂ₖ,
    /// [inout] Squared norm of the step @f$ \left\| p^k \right\|^2 @f$
    real_t &norm_sq_pₖ,
    /// [inout] Gradient of objective times step @f$ \nabla\psi(x^k)^\top p^k@f$
    real_t &grad_ψₖᵀpₖ,
    /// [inout] Lipschitz constant estimate @f$ L_{\nabla\psi}^k @f$
    real_t &Lₖ,
    /// [inout] Step size @f$ \gamma^k @f$
    real_t &γₖ) {

    real_t old_γₖ = γₖ;
    real_t margin = (1 + std::abs(ψₖ)) * rounding_tolerance;
    while (ψx̂ₖ - ψₖ > grad_ψₖᵀpₖ + 0.5 * Lₖ * norm_sq_pₖ + margin) {
        if (not(Lₖ * 2 <= L_max))
            break;

        Lₖ *= 2;
        γₖ /= 2;

        // Calculate x̂ₖ and pₖ (with new step size)
        calc_x̂(problem, γₖ, xₖ, grad_ψₖ, /* in ⟹ out */ x̂ₖ, pₖ);
        // Calculate ∇ψ(xₖ)ᵀpₖ and ‖pₖ‖²
        grad_ψₖᵀpₖ = grad_ψₖ.dot(pₖ);
        norm_sq_pₖ = pₖ.squaredNorm();

        // Calculate ψ(x̂ₖ) and ŷ(x̂ₖ)
        ψx̂ₖ = calc_ψ_ŷ(problem, x̂ₖ, y, Σ1, Σ2, /* in ⟹ out */ ŷx̂ₖ1, ŷx̂ₖ2);
    }
    return old_γₖ;
}

/// Check all stop conditions (required tolerance reached, out of time,
/// maximum number of iterations exceeded, interrupted by user,
/// infinite iterate, no progress made)
template <class ParamsT, class DurationT>
inline SolverStatus check_all_stop_conditions(
    /// [in]    Parameters including `max_iter`, `max_time` and `max_no_progress`
    const ParamsT &params,
    /// [in]    Time elapsed since the start of the algorithm
    DurationT time_elapsed,
    /// [in]    The current iteration number
    unsigned iteration,
    /// [in]    A stop signal for the user to interrupt the algorithm
    const AtomicStopSignal &stop_signal,
    /// [in]    Desired primal tolerance
    real_t ε,
    /// [in]    Tolerance of the current iterate
    real_t εₖ,
    /// [in]    The number of successive iterations no progress was made
    unsigned no_progress) {

    bool out_of_time     = time_elapsed > params.max_time;
    bool out_of_iter     = iteration == params.max_iter;
    bool interrupted     = stop_signal.stop_requested();
    bool not_finite      = not std::isfinite(εₖ);
    bool conv            = εₖ <= ε;
    bool max_no_progress = no_progress > params.max_no_progress;
    return conv              ? SolverStatus::Converged
           : out_of_time     ? SolverStatus::MaxTime
           : out_of_iter     ? SolverStatus::MaxIter
           : not_finite      ? SolverStatus::NotFinite
           : max_no_progress ? SolverStatus::NoProgress
           : interrupted     ? SolverStatus::Interrupted
                             : SolverStatus::Unknown;
}

/// Compute the Hessian matrix of the augmented Lagrangian function
/// @f[ \nabla^2_{xx} L_\Sigma(x, y) =
///     \Big. \nabla_{xx}^2 L(x, y) \Big|_{\big(x,\, \hat y(x, y)\big)}
///   + \sum_{i\in\mathcal{I}} \Sigma_i\,\nabla g_i(x) \nabla g_i(x)^\top @f]
inline void calc_augmented_lagrangian_hessian(
    /// [in]  Problem description
    const Problem &problem,
    /// [in]    Current iterate @f$ x^k @f$
    crvec xₖ,
    /// [in]   Intermediate vector @f$ \hat y(x^k) @f$
    crvec ŷxₖ,
    /// [in]    Lagrange multipliers @f$ y @f$
    crvec y,
    /// [in]    Penalty weights @f$ \Sigma @f$
    crvec Σ,
    /// [out]   The constraint values @f$ g(x^k) @f$
    rvec g,
    /// [out]   Hessian matrix @f$ H(x, y) @f$
    mat &H,
    ///         Dimension n
    rvec work_n) {

    // Compute the Hessian of the Lagrangian
    problem.hess_L(xₖ, ŷxₖ, H);
    // Compute the Hessian of the augmented Lagrangian
    problem.g(xₖ, g);
    for (vec::Index i = 0; i < problem.m; ++i) {
        real_t ζ = g(i) + y(i) / Σ(i);
        bool inactive =
            problem.D.lowerbound(i) < ζ && ζ < problem.D.upperbound(i);
        if (not inactive) {
            problem.grad_gi(xₖ, i, work_n);
            H += work_n * Σ(i) * work_n.transpose();
        }
    }
}

/// Compute the Hessian matrix of the augmented Lagrangian function multiplied
/// by the given vector, using finite differences.
/// @f[ \nabla^2_{xx} L_\Sigma(x, y)\, v \approx
///     \frac{\nabla_x L_\Sigma(x+hv, y) - \nabla_x L_\Sigma(x, y)}{h} @f]
inline void calc_augmented_lagrangian_hessian_prod_fd(
    /// [in]    Problem description
    const Problem &problem,
    /// [in]    Current iterate @f$ x^k @f$
    crvec xₖ,
    /// [in]    Lagrange multipliers @f$ y @f$
    crvec y,
    /// [in]    Penalty weights @f$ \Sigma @f$
    crvec Σ,
    /// [in]    Gradient @f$ \nabla \psi(x^k) @f$
    crvec grad_ψ,
    /// [in]    Vector to multiply with the Hessian
    crvec v,
    /// [out]   Hessian-vector product
    rvec Hv,
    ///         Dimension n
    rvec work_n1,
    ///         Dimension n
    rvec work_n2,
    ///         Dimension m
    rvec work_m) {

    real_t cbrt_ε = std::cbrt(std::numeric_limits<real_t>::epsilon());
    real_t h      = cbrt_ε * (1 + xₖ.norm());
    rvec xₖh      = work_n1;
    xₖh           = xₖ + h * v;
    calc_grad_ψ(problem, xₖh, y, Σ, Hv, work_n2, work_m);
    Hv -= grad_ψ;
    Hv /= h;
}

/// Estimate the Lipschitz constant of the gradient @f$ \nabla \psi @f$ using
/// finite differences.
inline real_t initial_lipschitz_estimate(
    /// [in]    Problem description
    const Problem &problem,
    /// [in]    Current iterate @f$ x^k @f$
    crvec xₖ,
    /// [in]    Lagrange multipliers @f$ y @f$
    crvec y,
    /// [in]    Penalty weights @f$ \Sigma @f$
    crvec Σ,
    /// [in]    Finite difference step size relative to xₖ
    real_t ε,
    /// [in]    Minimum absolute finite difference step size
    real_t δ,
    /// [in]    Minimum allowed Lipschitz estimate.
    real_t L_min,
    /// [in]    Maximum allowed Lipschitz estimate.
    real_t L_max,
    /// [out]   @f$ \psi(x^k) @f$
    real_t &ψ,
    /// [out]   Gradient @f$ \nabla \psi(x^k) @f$
    rvec grad_ψ,
    ///         Dimension n
    rvec work_n1,
    ///         Dimension n
    rvec work_n2,
    ///         Dimension n
    rvec work_n3,
    ///         Dimension m
    rvec work_m) {

    auto h        = (xₖ * ε).cwiseAbs().cwiseMax(δ);
    work_n1       = xₖ + h;
    real_t norm_h = h.norm();
    // Calculate ∇ψ(x₀ + h)
    calc_grad_ψ(problem, work_n1, y, Σ, /* in ⟹ out */ work_n2, work_n3,
                work_m);
    // Calculate ψ(xₖ), ∇ψ(x₀)
    ψ = calc_ψ_grad_ψ(problem, xₖ, y, Σ, /* in ⟹ out */ grad_ψ, work_n1,
                      work_m);

    // Estimate Lipschitz constant using finite differences
    real_t L = (work_n2 - grad_ψ).norm() / norm_h;
    return std::clamp(L, L_min, L_max);
}

/// Estimate the Lipschitz constant of the gradient @f$ \nabla \psi @f$ using
/// finite differences.
inline real_t initial_lipschitz_estimate(
    /// [in]    Problem description
    const Problem &problem,
    /// [in]    Current iterate @f$ x^k @f$
    crvec xₖ,
    /// [in]    Lagrange multipliers @f$ y @f$
    crvec y,
    /// [in]    Penalty weights @f$ \Sigma @f$
    crvec Σ,
    /// [in]    Finite difference step size relative to xₖ
    real_t ε,
    /// [in]    Minimum absolute finite difference step size
    real_t δ,
    /// [in]    Minimum allowed Lipschitz estimate.
    real_t L_min,
    /// [in]    Maximum allowed Lipschitz estimate.
    real_t L_max,
    /// [out]   Gradient @f$ \nabla \psi(x^k) @f$
    rvec grad_ψ,
    ///         Dimension n
    rvec work_n1,
    ///         Dimension n
    rvec work_n2,
    ///         Dimension n
    rvec work_n3,
    ///         Dimension m
    rvec work_m) {

    auto h        = (xₖ * ε).cwiseAbs().cwiseMax(δ);
    work_n1       = xₖ + h;
    real_t norm_h = h.norm();
    // Calculate ∇ψ(x₀ + h)
    calc_grad_ψ(problem, work_n1, y, Σ, /* in ⟹ out */ work_n2, work_n3,
                work_m);
    // Calculate ∇ψ(x₀)
    calc_grad_ψ(problem, xₖ, y, Σ, /* in ⟹ out */ grad_ψ, work_n1, work_m);

    // Estimate Lipschitz constant using finite differences
    real_t L = (work_n2 - grad_ψ).norm() / norm_h;
    return std::clamp(L, L_min, L_max);
}

inline real_t initial_lipschitz_estimate(
    /// [in]    Problem description
    const ProblemFull &problem,
    /// [in]    Current iterate @f$ x^k @f$
    crvec xₖ,
    /// [in]    Lagrange multipliers @f$ y @f$
    crvec y,
    /// [in]    Penalty weights @f$ \Sigma @f$
    crvec Σ1,
    /// [in]    Penalty weights @f$ \Sigma @f$
    crvec Σ2,
    /// [in]    Finite difference step size relative to xₖ
    real_t ε,
    /// [in]    Minimum absolute finite difference step size
    real_t δ,
    /// [out]   @f$ \psi(x^k) @f$
    real_t &ψ,
    /// [out]   Gradient @f$ \nabla \psi(x^k) @f$
    rvec grad_ψ,
    ///         Dimension n
    rvec work_n1,
    ///         Dimension n
    rvec work_n2,
    ///         Dimension n
    rvec work_n3,
    ///         Dimension m1
    rvec work_m1,
    ///         Dimension m2
    rvec work_m2) {

    auto h        = (xₖ * ε).cwiseAbs().cwiseMax(δ);
    work_n1       = xₖ + h;
    real_t norm_h = h.norm();
    // Calculate ∇ψ(x₀ + h)
    calc_grad_ψ(problem, work_n1, y, Σ1, Σ2, /* in ⟹ out */ work_n2, work_n3,
                work_m1, work_m2);
    // Calculate ψ(xₖ), ∇ψ(x₀)
    ψ = calc_ψ_grad_ψ(problem, xₖ, y, Σ1, Σ2, /* in ⟹ out */ grad_ψ, work_n1,
                      work_m1, work_m2);

    // Estimate Lipschitz constant
    real_t L = (work_n2 - grad_ψ).norm() / norm_h;
    if (L < std::numeric_limits<real_t>::epsilon())
        L = std::numeric_limits<real_t>::epsilon();
    return L;
}

} // namespace pa::detail