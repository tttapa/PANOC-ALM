#pragma once

#include <chrono>
#include <panoc-alm/inner/decl/panoc.hpp>
#include <panoc-alm/inner/detail/panoc-helpers.hpp>
#include <panoc-alm/inner/directions/decl/panoc-direction-update.hpp>

#include <cassert>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>

namespace pa {

using std::chrono::duration_cast;
using std::chrono::microseconds;

template <class DirectionProviderT>
std::string PANOCSolver<DirectionProviderT>::get_name() const {
    return "PANOCSolver<" + direction_provider.get_name() + ">";
}

template <class DirectionProviderT>
typename PANOCSolver<DirectionProviderT>::Stats
PANOCSolver<DirectionProviderT>::operator()(
    /// [in]    Problem description
    const Problem &problem,
    /// [in]    Constraint weights @f$ \Sigma @f$
    crvec Σ,
    /// [in]    Tolerance @f$ \varepsilon @f$
    real_t ε,
    /// [in]    Overwrite @p x, @p y and @p err_z even if not converged
    bool always_overwrite_results,
    /// [inout] Decision variable @f$ x @f$
    rvec x,
    /// [inout] Lagrange multipliers @f$ y @f$
    rvec y,
    /// [out]   Slack variable error @f$ g(x) - z @f$
    rvec err_z,
    /// [in]    Time remaining
    std::chrono::microseconds time_remaining) {
    
    std::chrono::microseconds delta_time = std::chrono::microseconds(0);
    if (time_remaining > std::chrono::microseconds(0) && params.max_time > time_remaining) {
        delta_time = params.max_time - time_remaining;
    }

    auto start_time = std::chrono::steady_clock::now();
    Stats s;

    const auto n = problem.n;
    const auto m = problem.m;

    // Allocate vectors, init L-BFGS -------------------------------------------

    // TODO: the L-BFGS objects and vectors allocate on each iteration of ALM,
    //       and there are more vectors than strictly necessary.

    vec xₖ = x,   // Value of x at the beginning of the iteration
        x̂ₖ(n),    // Value of x after a projected gradient step
        xₖ₊₁(n),  // xₖ for next iteration
        x̂ₖ₊₁(n),  // x̂ₖ for next iteration
        ŷx̂ₖ(m),   // ŷ(x̂ₖ) = Σ (g(x̂ₖ) - ẑₖ)
        ŷx̂ₖ₊₁(m), // ŷ(x̂ₖ) for next iteration
        pₖ(n),    // Projected gradient step pₖ = x̂ₖ - xₖ
        pₖ₊₁(n), // Projected gradient step pₖ₊₁ = x̂ₖ₊₁ - xₖ₊₁
        qₖ(n),   // Newton step Hₖ pₖ
        grad_ψₖ(n),   // ∇ψ(xₖ)
        grad_̂ψₖ(n),   // ∇ψ(x̂ₖ)
        grad_ψₖ₊₁(n); // ∇ψ(xₖ₊₁)

    vec work_n(n), work_m(m);

    // Keep track of how many successive iterations didn't update the iterate
    unsigned no_progress = 0;

    // Helper functions --------------------------------------------------------

    // Wrappers for helper functions that automatically pass along any arguments
    // that are constant within PANOC (for readability in the main algorithm)
    auto calc_ψ_ŷ = [&problem, &y, &Σ](crvec x, rvec ŷ) {
        return detail::calc_ψ_ŷ(problem, x, y, Σ, ŷ);
    };
    auto calc_ψ_grad_ψ = [&problem, &y, &Σ, &work_n, &work_m](crvec x,
                                                              rvec grad_ψ) {
        return detail::calc_ψ_grad_ψ(problem, x, y, Σ, grad_ψ, work_n, work_m);
    };
    auto calc_grad_ψ_from_ŷ = [&problem, &work_n](crvec x, crvec ŷ,
                                                  rvec grad_ψ) {
        detail::calc_grad_ψ_from_ŷ(problem, x, ŷ, grad_ψ, work_n);
    };
    auto calc_x̂ = [&problem](real_t γ, crvec x, crvec grad_ψ, rvec x̂, rvec p) {
        detail::calc_x̂(problem, γ, x, grad_ψ, x̂, p);
    };
    auto calc_err_z = [&problem, &y, &Σ](crvec x̂, rvec err_z) {
        detail::calc_err_z(problem, x̂, y, Σ, err_z);
    };
    auto descent_lemma = [this, &problem, &y,
                          &Σ](crvec xₖ, real_t ψₖ, crvec grad_ψₖ, rvec x̂ₖ,
                              rvec pₖ, rvec ŷx̂ₖ, real_t &ψx̂ₖ, real_t &pₖᵀpₖ,
                              real_t &grad_ψₖᵀpₖ, real_t &Lₖ, real_t &γₖ) {
        return detail::descent_lemma(
            problem, params.quadratic_upperbound_tolerance_factor, params.L_max,
            xₖ, ψₖ, grad_ψₖ, y, Σ, x̂ₖ, pₖ, ŷx̂ₖ, ψx̂ₖ, pₖᵀpₖ, grad_ψₖᵀpₖ, Lₖ, γₖ);
    };
    auto print_progress = [&](unsigned k, real_t ψₖ, crvec grad_ψₖ,
                              real_t pₖᵀpₖ, real_t γₖ, real_t εₖ) {
        std::cout << "[PANOC] " << std::setw(6) << k
                  << ": ψ = " << std::setw(13) << ψₖ
                  << ", ‖∇ψ‖ = " << std::setw(13) << grad_ψₖ.norm()
                  << ", ‖p‖ = " << std::setw(13) << std::sqrt(pₖᵀpₖ)
                  << ", γ = " << std::setw(13) << γₖ
                  << ", εₖ = " << std::setw(13) << εₖ << "\r\n";
    };

    // Estimate Lipschitz constant ---------------------------------------------

    real_t ψₖ, Lₖ;
    // Finite difference approximation of ∇²ψ in starting point
    if (params.Lipschitz.L₀ <= 0) {
        Lₖ = detail::initial_lipschitz_estimate(
            problem, xₖ, y, Σ, params.Lipschitz.ε, params.Lipschitz.δ,
            params.L_min, params.L_max,
            /* in ⟹ out */ ψₖ, grad_ψₖ, x̂ₖ, grad_̂ψₖ, work_n, work_m);
    }
    // Initial Lipschitz constant provided by the user
    else {
        Lₖ = params.Lipschitz.L₀;
        // Calculate ψ(xₖ), ∇ψ(x₀)
        ψₖ = calc_ψ_grad_ψ(xₖ, /* in ⟹ out */ grad_ψₖ);
    }
    if (not std::isfinite(Lₖ)) {
        s.status = SolverStatus::NotFinite;
        return s;
    }
    real_t γₖ = params.Lipschitz.Lγ_factor / Lₖ;
    real_t τ  = NaN;

    // First projected gradient step -------------------------------------------

    // Calculate x̂₀, p₀ (projected gradient step)
    calc_x̂(γₖ, xₖ, grad_ψₖ, /* in ⟹ out */ x̂ₖ, pₖ);
    // Calculate ψ(x̂ₖ) and ŷ(x̂ₖ)
    real_t ψx̂ₖ        = calc_ψ_ŷ(x̂ₖ, /* in ⟹ out */ ŷx̂ₖ);
    real_t grad_ψₖᵀpₖ = grad_ψₖ.dot(pₖ);
    real_t pₖᵀpₖ      = pₖ.squaredNorm();
    // Compute forward-backward envelope
    real_t φₖ = ψₖ + 1 / (2 * γₖ) * pₖᵀpₖ + grad_ψₖᵀpₖ;

    // Main PANOC loop
    // =========================================================================
    for (unsigned k = 0; k <= params.max_iter; ++k) {

        // Quadratic upper bound -----------------------------------------------
        if (k == 0 || params.update_lipschitz_in_linesearch == false) {
            // Decrease step size until quadratic upper bound is satisfied
            real_t old_γₖ =
                descent_lemma(xₖ, ψₖ, grad_ψₖ,
                              /* in ⟹ out */ x̂ₖ, pₖ, ŷx̂ₖ,
                              /* inout */ ψx̂ₖ, pₖᵀpₖ, grad_ψₖᵀpₖ, Lₖ, γₖ);
            if (k > 0 && γₖ != old_γₖ) // Flush L-BFGS if γ changed
                direction_provider.changed_γ(γₖ, old_γₖ);
            else if (k == 0) // Initialize L-BFGS
                direction_provider.initialize(xₖ, x̂ₖ, pₖ, grad_ψₖ);
            if (γₖ != old_γₖ)
                φₖ = ψₖ + 1 / (2 * γₖ) * pₖᵀpₖ + grad_ψₖᵀpₖ;
        }
        // Calculate ∇ψ(x̂ₖ)
        calc_grad_ψ_from_ŷ(x̂ₖ, ŷx̂ₖ, /* in ⟹ out */ grad_̂ψₖ);

        // Check stop condition ------------------------------------------------
        real_t εₖ = detail::calc_error_stop_crit(params.stop_crit, pₖ, γₖ, xₖ,
                                                 grad_̂ψₖ, grad_ψₖ, problem.C);

        // Print progress
        if (params.print_interval != 0 && k % params.print_interval == 0)
            print_progress(k, ψₖ, grad_ψₖ, pₖᵀpₖ, γₖ, εₖ);
        if (progress_cb)
            progress_cb({k, xₖ, pₖ, pₖᵀpₖ, x̂ₖ, φₖ, ψₖ, grad_ψₖ, ψx̂ₖ, grad_̂ψₖ,
                         Lₖ, γₖ, τ, εₖ, Σ, y, problem, params});

        auto time_elapsed = std::chrono::steady_clock::now() - start_time;
        auto stop_status  = detail::check_all_stop_conditions(
            params, time_elapsed + delta_time, k, stop_signal, ε, εₖ, no_progress);
        if (stop_status != SolverStatus::Unknown) {
            // TODO: We could cache g(x) and ẑ, but would that faster?
            //       It saves 1 evaluation of g per ALM iteration, but requires
            //       many extra stores in the inner loops of PANOC.
            // TODO: move the computation of ẑ and g(x) to ALM?
            if (stop_status == SolverStatus::Converged ||
                stop_status == SolverStatus::Interrupted ||
                stop_status == SolverStatus::MaxTime ||
                always_overwrite_results) {
                calc_err_z(x̂ₖ, /* in ⟹ out */ err_z);
                x = std::move(x̂ₖ);
                y = std::move(ŷx̂ₖ);
            }
            s.iterations   = k;
            s.ε            = εₖ;
            s.elapsed_time = duration_cast<microseconds>(time_elapsed);
            s.status       = stop_status;
            return s;
        }

        // Calculate quasi-Newton step -----------------------------------------
        real_t step_size =
            params.lbfgs_stepsize == LBFGSStepSize::BasedOnGradientStepSize
                ? 1
                : -1;
        if (k > 0)
            direction_provider.apply(xₖ, x̂ₖ, pₖ, step_size,
                                     /* in ⟹ out */ qₖ);

        // Line search initialization ------------------------------------------
        τ                  = 1;
        real_t σₖγₖ⁻¹pₖᵀpₖ = (1 - γₖ * Lₖ) * pₖᵀpₖ / (2 * γₖ);
        real_t φₖ₊₁, ψₖ₊₁, ψx̂ₖ₊₁, grad_ψₖ₊₁ᵀpₖ₊₁, pₖ₊₁ᵀpₖ₊₁;
        real_t Lₖ₊₁, γₖ₊₁;
        real_t ls_cond;
        // TODO: make separate parameter
        real_t margin =
            (1 + std::abs(φₖ)) * params.quadratic_upperbound_tolerance_factor;

        // Make sure quasi-Newton step is valid
        if (k == 0) {
            τ = 0; // Always use prox step on first iteration
        } else if (not qₖ.allFinite()) {
            τ = 0;
            ++s.lbfgs_failures;
            direction_provider.reset(); // Is there anything else we can do?
        }

        // Line search loop ----------------------------------------------------
        do {
            Lₖ₊₁ = Lₖ;
            γₖ₊₁ = γₖ;

            // Calculate xₖ₊₁
            if (τ / 2 < params.τ_min) { // line search failed
                xₖ₊₁.swap(x̂ₖ);          // → safe prox step
                ψₖ₊₁ = ψx̂ₖ;
                grad_ψₖ₊₁.swap(grad_̂ψₖ);
            } else {        // line search didn't fail (yet)
                if (τ == 1) // → faster quasi-Newton step
                    xₖ₊₁ = xₖ + qₖ;
                else
                    xₖ₊₁ = xₖ + (1 - τ) * pₖ + τ * qₖ;
                // Calculate ψ(xₖ₊₁), ∇ψ(xₖ₊₁)
                ψₖ₊₁ = calc_ψ_grad_ψ(xₖ₊₁, /* in ⟹ out */ grad_ψₖ₊₁);
            }

            // Calculate x̂ₖ₊₁, pₖ₊₁ (projected gradient step in xₖ₊₁)
            calc_x̂(γₖ₊₁, xₖ₊₁, grad_ψₖ₊₁, /* in ⟹ out */ x̂ₖ₊₁, pₖ₊₁);
            // Calculate ψ(x̂ₖ₊₁) and ŷ(x̂ₖ₊₁)
            ψx̂ₖ₊₁ = calc_ψ_ŷ(x̂ₖ₊₁, /* in ⟹ out */ ŷx̂ₖ₊₁);

            // Quadratic upper bound -------------------------------------------
            grad_ψₖ₊₁ᵀpₖ₊₁ = grad_ψₖ₊₁.dot(pₖ₊₁);
            pₖ₊₁ᵀpₖ₊₁      = pₖ₊₁.squaredNorm();
            real_t pₖ₊₁ᵀpₖ₊₁_ₖ = pₖ₊₁ᵀpₖ₊₁; // prox step with step size γₖ

            if (params.update_lipschitz_in_linesearch == true) {
                // Decrease step size until quadratic upper bound is satisfied
                (void)descent_lemma(xₖ₊₁, ψₖ₊₁, grad_ψₖ₊₁,
                                    /* in ⟹ out */ x̂ₖ₊₁, pₖ₊₁, ŷx̂ₖ₊₁,
                                    /* inout */ ψx̂ₖ₊₁, pₖ₊₁ᵀpₖ₊₁,
                                    grad_ψₖ₊₁ᵀpₖ₊₁, Lₖ₊₁, γₖ₊₁);
            }

            // Compute forward-backward envelope
            φₖ₊₁ = ψₖ₊₁ + 1 / (2 * γₖ₊₁) * pₖ₊₁ᵀpₖ₊₁ + grad_ψₖ₊₁ᵀpₖ₊₁;
            // Compute line search condition
            ls_cond = φₖ₊₁ - (φₖ - σₖγₖ⁻¹pₖᵀpₖ);
            if (params.alternative_linesearch_cond)
                ls_cond -= (0.5 / γₖ₊₁ - 0.5 / γₖ) * pₖ₊₁ᵀpₖ₊₁_ₖ;

            τ /= 2;
        } while (ls_cond > margin && τ >= params.τ_min);

        // If τ < τ_min the line search failed and we accepted the prox step
        if (τ < params.τ_min && k != 0) {
            ++s.linesearch_failures;
            τ = 0;
        }
        if (k != 0) {
            s.count_τ += 1;
            s.sum_τ += τ * 2;
            s.τ_1_accepted += τ * 2 == 1;
        }

        // Update L-BFGS -------------------------------------------------------
        if (γₖ != γₖ₊₁) // Flush L-BFGS if γ changed
            direction_provider.changed_γ(γₖ₊₁, γₖ);

        s.lbfgs_rejected += not direction_provider.update(
            xₖ, xₖ₊₁, pₖ, pₖ₊₁, grad_ψₖ₊₁, problem.C, γₖ₊₁);

        // Check if we made any progress
        if (no_progress > 0 || k % params.max_no_progress == 0)
            no_progress = xₖ == xₖ₊₁ ? no_progress + 1 : 0;

        // Advance step --------------------------------------------------------
        Lₖ = Lₖ₊₁;
        γₖ = γₖ₊₁;

        ψₖ  = ψₖ₊₁;
        ψx̂ₖ = ψx̂ₖ₊₁;
        φₖ  = φₖ₊₁;

        xₖ.swap(xₖ₊₁);
        x̂ₖ.swap(x̂ₖ₊₁);
        ŷx̂ₖ.swap(ŷx̂ₖ₊₁);
        pₖ.swap(pₖ₊₁);
        grad_ψₖ.swap(grad_ψₖ₊₁);
        grad_ψₖᵀpₖ = grad_ψₖ₊₁ᵀpₖ₊₁;
        pₖᵀpₖ      = pₖ₊₁ᵀpₖ₊₁;
    }
    throw std::logic_error("[PANOC] loop error");
}

template <class DirectionProviderT>
std::string PANOCSolverFull<DirectionProviderT>::get_name() const {
    return "PANOCSolverFull<" + direction_provider.get_name() + ">";
}

template <class DirectionProviderT>
typename PANOCSolverFull<DirectionProviderT>::Stats
PANOCSolverFull<DirectionProviderT>::operator()(
    /// [in]    Problem description
    const ProblemFull &problem,
    /// [in]    Constraint weights @f$ \Sigma @f$
    crvec Σ1,
    /// [in]    Constraint weights @f$ \Sigma @f$
    crvec Σ2,
    /// [in]    Tolerance @f$ \varepsilon @f$
    real_t ε,
    /// [in]    Overwrite @p x, @p y and @p err_z even if not converged
    bool always_overwrite_results,
    /// [inout] Decision variable @f$ x @f$
    rvec x,
    /// [inout] Lagrange multipliers @f$ y @f$
    rvec y,
    /// [out]   Slack variable error @f$ g(x) - z @f$
    rvec err_z1,
    /// [out]   Slack variable error @f$ g(x) - z @f$
    rvec err_z2,
    /// [in]    Time remaining
    std::chrono::microseconds time_remaining) {
    
    std::chrono::microseconds delta_time = std::chrono::microseconds(0);
    if (time_remaining > std::chrono::microseconds(0) && params.max_time > time_remaining) {
        delta_time = params.max_time - time_remaining;
    }

    auto start_time = std::chrono::steady_clock::now();
    Stats s;

    const auto n = problem.n;
    const auto m1 = problem.m1;
    const auto m2 = problem.m2;

    // Allocate vectors, init L-BFGS -------------------------------------------

    // TODO: the L-BFGS objects and vectors allocate on each iteration of ALM,
    //       and there are more vectors than strictly necessary.

    vec xₖ = x,   // Value of x at the beginning of the iteration
        x̂ₖ(n),    // Value of x after a projected gradient step
        xₖ₊₁(n),  // xₖ for next iteration
        x̂ₖ₊₁(n),  // x̂ₖ for next iteration
        ŷx̂ₖ1(m1),   // ŷ(x̂ₖ) = Σ (g1(x̂ₖ) - ẑₖ)
        ŷx̂ₖ2(m2),   // ŷ(x̂ₖ) = Σ (g2(x̂ₖ) - ẑₖ)
        ŷx̂ₖ₊₁1(m1), // ŷ(x̂ₖ) for next iteration
        ŷx̂ₖ₊₁2(m2), // ŷ(x̂ₖ) for next iteration
        pₖ(n),    // Projected gradient step pₖ = x̂ₖ - xₖ
        pₖ₊₁(n), // Projected gradient step pₖ₊₁ = x̂ₖ₊₁ - xₖ₊₁
        qₖ(n),   // Newton step Hₖ pₖ
        grad_ψₖ(n),   // ∇ψ(xₖ)
        grad_̂ψₖ(n),   // ∇ψ(x̂ₖ)
        grad_ψₖ₊₁(n); // ∇ψ(xₖ₊₁)

    vec work_n(n), work_m1(m1), work_m2(m2);

    // Keep track of how many successive iterations didn't update the iterate
    unsigned no_progress = 0;

    // Helper functions --------------------------------------------------------

    // Wrappers for helper functions that automatically pass along any arguments
    // that are constant within PANOC (for readability in the main algorithm)
    auto calc_ψ_ŷ = [&problem, &y, &Σ1, &Σ2](crvec x, rvec ŷ1, rvec ŷ2) {
        return detail::calc_ψ_ŷ(problem, x, y, Σ1, Σ2, ŷ1, ŷ2);
    };
    auto calc_ψ_grad_ψ = [&problem, &y, &Σ1, &Σ2, &work_n, &work_m1, &work_m2](
                                                        crvec x, rvec grad_ψ) {
        return detail::calc_ψ_grad_ψ(problem, x, y, Σ1, Σ2, grad_ψ, work_n, work_m1, work_m2);
    };
    auto calc_grad_ψ_from_ŷ = [&problem, &work_n](crvec x, crvec ŷ1,
                                                  crvec ŷ2, rvec grad_ψ) {
        detail::calc_grad_ψ_from_ŷ(problem, x, ŷ1, ŷ2, grad_ψ, work_n);
    };
    auto calc_x̂ = [&problem](real_t γ, crvec x, crvec grad_ψ, rvec x̂, rvec p) {
        detail::calc_x̂(problem, γ, x, grad_ψ, x̂, p);
    };
    auto calc_err_z = [&problem, &y, &Σ1](crvec x̂, rvec err_z1, rvec err_z2) {
        detail::calc_err_z(problem, x̂, y, Σ1, err_z1, err_z2);
    };
    auto descent_lemma = [this, &problem, &y,
                          &Σ1, &Σ2](crvec xₖ, real_t ψₖ, crvec grad_ψₖ, rvec x̂ₖ,
                              rvec pₖ, rvec ŷx̂ₖ1, rvec ŷx̂ₖ2, real_t &ψx̂ₖ, real_t &pₖᵀpₖ,
                              real_t &grad_ψₖᵀpₖ, real_t &Lₖ, real_t &γₖ) {
        return detail::descent_lemma(
            problem, params.quadratic_upperbound_tolerance_factor, params.L_max,
            xₖ, ψₖ, grad_ψₖ, y, Σ1, Σ2, x̂ₖ, pₖ, ŷx̂ₖ1, ŷx̂ₖ2, ψx̂ₖ, pₖᵀpₖ, grad_ψₖᵀpₖ, Lₖ, γₖ);
    };
    auto print_progress = [&](unsigned k, real_t ψₖ, crvec grad_ψₖ,
                              real_t pₖᵀpₖ, real_t γₖ, real_t εₖ) {
        std::cout << "[PANOC] " << std::setw(6) << k
                  << ": ψ = " << std::setw(13) << ψₖ
                  << ", ‖∇ψ‖ = " << std::setw(13) << grad_ψₖ.norm()
                  << ", ‖p‖ = " << std::setw(13) << std::sqrt(pₖᵀpₖ)
                  << ", γ = " << std::setw(13) << γₖ
                  << ", εₖ = " << std::setw(13) << εₖ << "\r\n";
    };

    // Estimate Lipschitz constant ---------------------------------------------

    real_t ψₖ, Lₖ;
    // Finite difference approximation of ∇²ψ in starting point
    if (params.Lipschitz.L₀ <= 0) {
        Lₖ = detail::initial_lipschitz_estimate(
            problem, xₖ, y, Σ1, Σ2, params.Lipschitz.ε, params.Lipschitz.δ,
            /* in ⟹ out */ ψₖ, grad_ψₖ, x̂ₖ, grad_̂ψₖ, work_n, work_m1, work_m2);
    }
    // Initial Lipschitz constant provided by the user
    else {
        Lₖ = params.Lipschitz.L₀;
        // Calculate ψ(xₖ), ∇ψ(x₀)
        ψₖ = calc_ψ_grad_ψ(xₖ, /* in ⟹ out */ grad_ψₖ);
    }
    if (not std::isfinite(Lₖ)) {
        s.status = SolverStatus::NotFinite;
        return s;
    }
    real_t γₖ = params.Lipschitz.Lγ_factor / Lₖ;
    real_t τ  = NaN;

    // First projected gradient step -------------------------------------------

    // Calculate x̂₀, p₀ (projected gradient step)
    calc_x̂(γₖ, xₖ, grad_ψₖ, /* in ⟹ out */ x̂ₖ, pₖ);
    // Calculate ψ(x̂ₖ) and ŷ(x̂ₖ)
    real_t ψx̂ₖ        = calc_ψ_ŷ(x̂ₖ, /* in ⟹ out */ ŷx̂ₖ1, ŷx̂ₖ2);
    real_t grad_ψₖᵀpₖ = grad_ψₖ.dot(pₖ);
    real_t pₖᵀpₖ      = pₖ.squaredNorm();
    // Compute forward-backward envelope
    real_t φₖ = ψₖ + 1 / (2 * γₖ) * pₖᵀpₖ + grad_ψₖᵀpₖ;

    // Main PANOC loop
    // =========================================================================
    for (unsigned k = 0; k <= params.max_iter; ++k) {

        // Quadratic upper bound -----------------------------------------------
        if (k == 0 || params.update_lipschitz_in_linesearch == false) {
            // Decrease step size until quadratic upper bound is satisfied
            real_t old_γₖ =
                descent_lemma(xₖ, ψₖ, grad_ψₖ,
                              /* in ⟹ out */ x̂ₖ, pₖ, ŷx̂ₖ1, ŷx̂ₖ2,
                              /* inout */ ψx̂ₖ, pₖᵀpₖ, grad_ψₖᵀpₖ, Lₖ, γₖ);
            if (k > 0 && γₖ != old_γₖ) // Flush L-BFGS if γ changed
                direction_provider.changed_γ(γₖ, old_γₖ);
            else if (k == 0) // Initialize L-BFGS
                direction_provider.initialize(xₖ, x̂ₖ, pₖ, grad_ψₖ);
            if (γₖ != old_γₖ)
                φₖ = ψₖ + 1 / (2 * γₖ) * pₖᵀpₖ + grad_ψₖᵀpₖ;
        }
        // Calculate ∇ψ(x̂ₖ)
        calc_grad_ψ_from_ŷ(x̂ₖ, ŷx̂ₖ1, ŷx̂ₖ2, /* in ⟹ out */ grad_̂ψₖ);

        // Check stop condition ------------------------------------------------
        real_t εₖ = detail::calc_error_stop_crit(params.stop_crit, pₖ, γₖ, xₖ,
                                                 grad_̂ψₖ, grad_ψₖ, problem.C);

        // Print progress
        if (params.print_interval != 0 && k % params.print_interval == 0)
            print_progress(k, ψₖ, grad_ψₖ, pₖᵀpₖ, γₖ, εₖ);
        if (progress_cb)
            progress_cb({k, xₖ, pₖ, pₖᵀpₖ, x̂ₖ, φₖ, ψₖ, grad_ψₖ, ψx̂ₖ, grad_̂ψₖ,
                         Lₖ, γₖ, τ, εₖ, Σ1, Σ2, y, problem, params}); //TODO

        auto time_elapsed = std::chrono::steady_clock::now() - start_time;
        auto stop_status  = detail::check_all_stop_conditions(
            params, time_elapsed + delta_time, k, stop_signal, ε, εₖ, no_progress);
        if (stop_status != SolverStatus::Unknown) {
            // TODO: We could cache g(x) and ẑ, but would that faster?
            //       It saves 1 evaluation of g per ALM iteration, but requires
            //       many extra stores in the inner loops of PANOC.
            // TODO: move the computation of ẑ and g(x) to ALM?
            if (stop_status == SolverStatus::Converged ||
                stop_status == SolverStatus::Interrupted ||
                stop_status == SolverStatus::MaxTime ||
                always_overwrite_results) {
                calc_err_z(x̂ₖ, /* in ⟹ out */ err_z1, err_z2);
                x = std::move(x̂ₖ);
                y = std::move(ŷx̂ₖ1);
            }
            s.iterations   = k;
            s.ε            = εₖ;
            s.elapsed_time = duration_cast<microseconds>(time_elapsed);
            s.status       = stop_status;
            return s;
        }

        // Calculate quasi-Newton step -----------------------------------------
        real_t step_size =
            params.lbfgs_stepsize == LBFGSStepSize::BasedOnGradientStepSize
                ? 1
                : -1;
        if (k > 0)
            direction_provider.apply(xₖ, x̂ₖ, pₖ, step_size,
                                     /* in ⟹ out */ qₖ);

        // Line search initialization ------------------------------------------
        τ                  = 1;
        real_t σₖγₖ⁻¹pₖᵀpₖ = (1 - γₖ * Lₖ) * pₖᵀpₖ / (2 * γₖ);
        real_t φₖ₊₁, ψₖ₊₁, ψx̂ₖ₊₁, grad_ψₖ₊₁ᵀpₖ₊₁, pₖ₊₁ᵀpₖ₊₁;
        real_t Lₖ₊₁, γₖ₊₁;
        real_t ls_cond;
        // TODO: make separate parameter
        real_t margin =
            (1 + std::abs(φₖ)) * params.quadratic_upperbound_tolerance_factor;

        // Make sure quasi-Newton step is valid
        if (k == 0) {
            τ = 0; // Always use prox step on first iteration
        } else if (not qₖ.allFinite()) {
            τ = 0;
            ++s.lbfgs_failures;
            direction_provider.reset(); // Is there anything else we can do?
        }

        // Line search loop ----------------------------------------------------
        do {
            Lₖ₊₁ = Lₖ;
            γₖ₊₁ = γₖ;

            // Calculate xₖ₊₁
            if (τ / 2 < params.τ_min) { // line search failed
                xₖ₊₁.swap(x̂ₖ);          // → safe prox step
                ψₖ₊₁ = ψx̂ₖ;
                grad_ψₖ₊₁.swap(grad_̂ψₖ);
            } else {        // line search didn't fail (yet)
                if (τ == 1) // → faster quasi-Newton step
                    xₖ₊₁ = xₖ + qₖ;
                else
                    xₖ₊₁ = xₖ + (1 - τ) * pₖ + τ * qₖ;
                // Calculate ψ(xₖ₊₁), ∇ψ(xₖ₊₁)
                ψₖ₊₁ = calc_ψ_grad_ψ(xₖ₊₁, /* in ⟹ out */ grad_ψₖ₊₁);
            }

            // Calculate x̂ₖ₊₁, pₖ₊₁ (projected gradient step in xₖ₊₁)
            calc_x̂(γₖ₊₁, xₖ₊₁, grad_ψₖ₊₁, /* in ⟹ out */ x̂ₖ₊₁, pₖ₊₁);
            // Calculate ψ(x̂ₖ₊₁) and ŷ(x̂ₖ₊₁)
            ψx̂ₖ₊₁ = calc_ψ_ŷ(x̂ₖ₊₁, /* in ⟹ out */ ŷx̂ₖ₊₁1, ŷx̂ₖ₊₁2);

            // Quadratic upper bound -------------------------------------------
            grad_ψₖ₊₁ᵀpₖ₊₁ = grad_ψₖ₊₁.dot(pₖ₊₁);
            pₖ₊₁ᵀpₖ₊₁      = pₖ₊₁.squaredNorm();
            real_t pₖ₊₁ᵀpₖ₊₁_ₖ = pₖ₊₁ᵀpₖ₊₁; // prox step with step size γₖ

            if (params.update_lipschitz_in_linesearch == true) {
                // Decrease step size until quadratic upper bound is satisfied
                (void)descent_lemma(xₖ₊₁, ψₖ₊₁, grad_ψₖ₊₁,
                                    /* in ⟹ out */ x̂ₖ₊₁, pₖ₊₁, ŷx̂ₖ₊₁1, ŷx̂ₖ₊₁2,
                                    /* inout */ ψx̂ₖ₊₁, pₖ₊₁ᵀpₖ₊₁,
                                    grad_ψₖ₊₁ᵀpₖ₊₁, Lₖ₊₁, γₖ₊₁);
            }

            // Compute forward-backward envelope
            φₖ₊₁ = ψₖ₊₁ + 1 / (2 * γₖ₊₁) * pₖ₊₁ᵀpₖ₊₁ + grad_ψₖ₊₁ᵀpₖ₊₁;
            // Compute line search condition
            ls_cond = φₖ₊₁ - (φₖ - σₖγₖ⁻¹pₖᵀpₖ);
            if (params.alternative_linesearch_cond)
                ls_cond -= (0.5 / γₖ₊₁ - 0.5 / γₖ) * pₖ₊₁ᵀpₖ₊₁_ₖ;

            τ /= 2;
        } while (ls_cond > margin && τ >= params.τ_min);

        // If τ < τ_min the line search failed and we accepted the prox step
        if (τ < params.τ_min && k != 0) {
            ++s.linesearch_failures;
            τ = 0;
        }
        if (k != 0) {
            s.count_τ += 1;
            s.sum_τ += τ * 2;
            s.τ_1_accepted += τ * 2 == 1;
        }

        // Update L-BFGS -------------------------------------------------------
        if (γₖ != γₖ₊₁) // Flush L-BFGS if γ changed
            direction_provider.changed_γ(γₖ₊₁, γₖ);

        s.lbfgs_rejected += not direction_provider.update(
            xₖ, xₖ₊₁, pₖ, pₖ₊₁, grad_ψₖ₊₁, problem.C, γₖ₊₁);

        // Check if we made any progress
        if (no_progress > 0 || k % params.max_no_progress == 0)
            no_progress = xₖ == xₖ₊₁ ? no_progress + 1 : 0;

        // Advance step --------------------------------------------------------
        Lₖ = Lₖ₊₁;
        γₖ = γₖ₊₁;

        ψₖ  = ψₖ₊₁;
        ψx̂ₖ = ψx̂ₖ₊₁;
        φₖ  = φₖ₊₁;

        xₖ.swap(xₖ₊₁);
        x̂ₖ.swap(x̂ₖ₊₁);
        ŷx̂ₖ1.swap(ŷx̂ₖ₊₁1);
        ŷx̂ₖ2.swap(ŷx̂ₖ₊₁2);
        pₖ.swap(pₖ₊₁);
        grad_ψₖ.swap(grad_ψₖ₊₁);
        grad_ψₖᵀpₖ = grad_ψₖ₊₁ᵀpₖ₊₁;
        pₖᵀpₖ      = pₖ₊₁ᵀpₖ₊₁;
    }
    throw std::logic_error("[PANOC] loop error");
}

} // namespace pa
