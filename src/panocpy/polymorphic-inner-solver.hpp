#pragma once

#include <panoc-alm/inner/decl/structured-panoc-lbfgs.hpp>
#include <panoc-alm/inner/directions/lbfgs.hpp>
#include <panoc-alm/inner/guarded-aa-pga.hpp>
#include <panoc-alm/inner/panoc.hpp>
#include <panoc-alm/inner/pga.hpp>
#include <panoc-alm/util/solverstatus.hpp>

#include <memory>
#include <type_traits>

#include <pybind11/cast.h>
#include <pybind11/pybind11.h>
namespace py = pybind11;

namespace pa {

template <class InnerSolver>
auto InnerSolverCallWrapper() {
    return [](InnerSolver &solver, const pa::Problem &p, pa::crvec Σ,
              pa::real_t ε, pa::vec x,
              pa::vec y) -> std::tuple<pa::vec, pa::vec, pa::vec, py::dict> {
        pa::vec z(p.m);
        auto stats = solver(p, Σ, ε, true, x, y, z, std::chrono::microseconds(0));
        return std::make_tuple(std::move(x), std::move(y), std::move(z),
                               stats.ptr->to_dict());
    };
}

template <class InnerSolver>
auto InnerSolverFullCallWrapper() {
    return [](InnerSolver &solver, const pa::ProblemFull &p, pa::crvec Σ1,
              pa::crvec Σ2, pa::real_t ε, pa::vec x, pa::vec y)
               -> std::tuple<pa::vec, pa::vec, pa::vec, pa::vec, py::dict> {
        pa::vec z1(p.m1);
        pa::vec z2(p.m2);
        (void)solver(p, Σ1, Σ2, ε, true, x, y, z1, z2);
        return std::make_tuple(std::move(x), std::move(y), std::move(z1),
                               std::move(z2), py::dict{});
    };
}

class PolymorphicInnerSolverStatsAccumulatorBase
    : public std::enable_shared_from_this<
          PolymorphicInnerSolverStatsAccumulatorBase> {
  public:
    virtual ~PolymorphicInnerSolverStatsAccumulatorBase() = default;
    virtual py::dict to_dict() const                      = 0;
    virtual void accumulate(const class PolymorphicInnerSolverStatsBase &) = 0;
};

class PolymorphicInnerSolverStatsBase
    : public std::enable_shared_from_this<PolymorphicInnerSolverStatsBase> {
  public:
    virtual ~PolymorphicInnerSolverStatsBase() = default;
    virtual py::dict to_dict() const           = 0;
    virtual std::shared_ptr<PolymorphicInnerSolverStatsAccumulatorBase>
    accumulator() const = 0;
};

class PolymorphicInnerSolverBase
    : public std::enable_shared_from_this<PolymorphicInnerSolverBase> {
  public:
    struct Stats {
        std::shared_ptr<PolymorphicInnerSolverStatsBase> ptr;
        SolverStatus status;
        real_t ε;
        unsigned iterations;

        static Stats from_dict(py::dict d) {
            using PolyStats    = pa::PolymorphicInnerSolverStatsBase;
            using PolyAccStats = pa::PolymorphicInnerSolverStatsAccumulatorBase;
            using InnerStats   = pa::PolymorphicInnerSolverBase::Stats;
            struct AccStats : PolyAccStats {
                AccStats(py::dict dict) : dict(std::move(dict)) {}
                py::dict dict;
                py::dict to_dict() const override { return dict; }
                void accumulate(const PolyStats &s) override {
                    if (this->dict.contains("accumulate"))
                        this->dict["accumulate"](this->dict, s.to_dict());
                    else
                        throw py::key_error("Stats accumulator does not define "
                                            "an accumulate function");
                }
            };
            struct Stats : PolyStats {
                Stats(py::dict dict) : dict(std::move(dict)) {}
                py::dict dict;
                py::dict to_dict() const override { return dict; }
                std::shared_ptr<PolyAccStats> accumulator() const override {
                    if (this->dict.contains("accumulator"))
                        return {
                            std::make_shared<AccStats>(
                                dict["accumulator"].cast<py::dict>()),
                        };
                    else
                        throw py::key_error(
                            "Stats do not define an accumulator");
                }
            };
            bool ok = d.contains("status") && d.contains("ε") &&
                      d.contains("iterations");
            if (not ok)
                throw py::key_error(
                    "Stats should contain status, ε and iterations");
            return {
                std::static_pointer_cast<PolyStats>(std::make_shared<Stats>(d)),
                d["status"].cast<decltype(InnerStats::status)>(),
                d["ε"].cast<decltype(InnerStats::ε)>(),
                d["iterations"].cast<decltype(InnerStats::iterations)>(),
            };
        }
    };

    virtual ~PolymorphicInnerSolverBase() = default;
    virtual Stats operator()(
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
        std::chrono::microseconds time_remaining) = 0;
    virtual void stop()                   = 0;
    virtual std::string get_name() const  = 0;
    virtual py::object get_params() const = 0;
};

struct PolymorphicInnerSolverWrapper {
    using Stats = PolymorphicInnerSolverBase::Stats;
    std::shared_ptr<PolymorphicInnerSolverBase> solver;
    PolymorphicInnerSolverWrapper(
        std::shared_ptr<PolymorphicInnerSolverBase> &&solver)
        : solver(std::move(solver)) {}

    Stats operator()(const Problem &problem, crvec Σ, real_t ε,
                     bool always_overwrite_results, rvec x, rvec y,
                     rvec err_z, std::chrono::microseconds time_remaining) {
        return solver->operator()(problem, Σ, ε, always_overwrite_results, x, y,
                                  err_z, time_remaining);
    }
    void stop() { solver->stop(); }
    std::string get_name() const { return solver->get_name(); }
    py::object get_params() const { return solver->get_params(); }
};

template <class InnerSolverStats>
struct InnerStatsAccumulator;

template <>
struct InnerStatsAccumulator<PolymorphicInnerSolverWrapper::Stats> {
    std::shared_ptr<PolymorphicInnerSolverStatsAccumulatorBase> ptr;
    py::dict to_dict() const { return ptr->to_dict(); }
};

inline InnerStatsAccumulator<PolymorphicInnerSolverWrapper::Stats> &
operator+=(InnerStatsAccumulator<PolymorphicInnerSolverWrapper::Stats> &acc,
           const PolymorphicInnerSolverWrapper::Stats &s) {
    assert(s.ptr);
    if (not acc.ptr)
        acc.ptr = s.ptr->accumulator();
    acc.ptr->accumulate(*s.ptr);
    return acc;
}

class PolymorphicInnerSolverTrampoline : public PolymorphicInnerSolverBase {
  public:
    Stats operator()(const Problem &problem, crvec Σ, real_t ε,
                     bool always_overwrite_results, rvec x, rvec y,
                     rvec err_z, std::chrono::microseconds time_remaining) override {
        py::dict stats;
        std::tie(x, y, err_z, stats) =
            call(problem, Σ, ε, always_overwrite_results, x, y);
        return Stats::from_dict(stats);
    }
    virtual std::tuple<pa::vec, pa::vec, pa::vec, py::dict>
    call(const pa::Problem &problem, pa::crvec Σ, pa::real_t ε,
         bool always_overwrite_results, pa::vec x, pa::vec y) {
        using ret = std::tuple<pa::vec, pa::vec, pa::vec, py::dict>;
        PYBIND11_OVERRIDE_PURE_NAME(ret, PolymorphicInnerSolverBase, "__call__",
                                    call, problem, Σ, ε,
                                    always_overwrite_results, x, y);
    }
    std::string get_name() const override {
        PYBIND11_OVERRIDE_PURE(std::string, PolymorphicInnerSolverBase,
                               get_name, );
    }
    py::object get_params() const override {
        PYBIND11_OVERRIDE_PURE(py::object, PolymorphicInnerSolverBase,
                               get_params, );
    }
    void stop() override {
        PYBIND11_OVERRIDE_PURE(void, PolymorphicInnerSolverBase, stop, );
    }
};

inline py::dict stats_to_dict(const PANOCStats &s) {
    using py::operator""_a;
    return py::dict{
        "status"_a              = s.status,
        "ε"_a                   = s.ε,
        "elapsed_time"_a        = s.elapsed_time,
        "iterations"_a          = s.iterations,
        "linesearch_failures"_a = s.linesearch_failures,
        "lbfgs_failures"_a      = s.lbfgs_failures,
        "lbfgs_rejected"_a      = s.lbfgs_rejected,
        "τ_1_accepted"_a        = s.τ_1_accepted,
        "count_τ"_a             = s.count_τ,
        "sum_τ"_a               = s.sum_τ,
    };
}

inline py::dict stats_to_dict(const InnerStatsAccumulator<PANOCStats> &s) {
    using py::operator""_a;
    return py::dict{
        "elapsed_time"_a        = s.elapsed_time,
        "iterations"_a          = s.iterations,
        "linesearch_failures"_a = s.linesearch_failures,
        "lbfgs_failures"_a      = s.lbfgs_failures,
        "lbfgs_rejected"_a      = s.lbfgs_rejected,
        "τ_1_accepted"_a        = s.τ_1_accepted,
        "count_τ"_a             = s.count_τ,
        "sum_τ"_a               = s.sum_τ,
    };
}

inline py::dict stats_to_dict(const StructuredPANOCLBFGSSolver::Stats &s) {
    using py::operator""_a;
    return py::dict{
        "status"_a              = s.status,
        "ε"_a                   = s.ε,
        "elapsed_time"_a        = s.elapsed_time,
        "iterations"_a          = s.iterations,
        "linesearch_failures"_a = s.linesearch_failures,
        "lbfgs_failures"_a      = s.lbfgs_failures,
        "lbfgs_rejected"_a      = s.lbfgs_rejected,
        "τ_1_accepted"_a        = s.τ_1_accepted,
        "count_τ"_a             = s.count_τ,
        "sum_τ"_a               = s.sum_τ,
    };
}

inline py::dict stats_to_dict(const PGASolver::Stats &s) {
    using py::operator""_a;
    return py::dict{
        "status"_a       = s.status,
        "ε"_a            = s.ε,
        "elapsed_time"_a = s.elapsed_time,
        "iterations"_a   = s.iterations,
    };
}

inline py::dict stats_to_dict(const GAAPGASolver::Stats &s) {
    using py::operator""_a;
    return py::dict{
        "status"_a                     = s.status,
        "ε"_a                          = s.ε,
        "elapsed_time"_a               = s.elapsed_time,
        "iterations"_a                 = s.iterations,
        "accelerated_steps_accepted"_a = s.accelerated_steps_accepted,
    };
}

inline py::dict stats_to_dict(
    const InnerStatsAccumulator<StructuredPANOCLBFGSSolver::Stats> &s) {
    using py::operator""_a;
    return py::dict{
        "elapsed_time"_a        = s.elapsed_time,
        "iterations"_a          = s.iterations,
        "linesearch_failures"_a = s.linesearch_failures,
        "lbfgs_failures"_a      = s.lbfgs_failures,
        "lbfgs_rejected"_a      = s.lbfgs_rejected,
        "τ_1_accepted"_a        = s.τ_1_accepted,
        "count_τ"_a             = s.count_τ,
        "sum_τ"_a               = s.sum_τ,
    };
}

inline py::dict
stats_to_dict(const InnerStatsAccumulator<PGASolver::Stats> &s) {
    using py::operator""_a;
    return py::dict{
        "elapsed_time"_a = s.elapsed_time,
        "iterations"_a   = s.iterations,
    };
}

inline py::dict
stats_to_dict(const InnerStatsAccumulator<GAAPGASolver::Stats> &s) {
    using py::operator""_a;
    return py::dict{
        "elapsed_time"_a               = s.elapsed_time,
        "iterations"_a                 = s.iterations,
        "accelerated_steps_accepted"_a = s.accelerated_steps_accepted,
    };
}

template <class InnerSolver>
class PolymorphicInnerSolver : public PolymorphicInnerSolverBase {
  public:
    PolymorphicInnerSolver(InnerSolver &&innersolver)
        : innersolver(std::forward<InnerSolver>(innersolver)) {}
    PolymorphicInnerSolver(const InnerSolver &innersolver)
        : innersolver(innersolver) {}
    template <class... Args>
    PolymorphicInnerSolver(Args... args)
        : innersolver(InnerSolver{std::forward<Args>(args)...}) {}

    struct WrappedStatsAccumulator
        : PolymorphicInnerSolverStatsAccumulatorBase {
        InnerStatsAccumulator<typename InnerSolver::Stats> acc;
        void
        accumulate(const PolymorphicInnerSolverStatsBase &bstats) override {
            auto &stats = dynamic_cast<const WrappedStats &>(bstats).stats;
            acc += stats;
        }
        py::dict to_dict() const override { return stats_to_dict(acc); }
    };
    struct WrappedStats : PolymorphicInnerSolverStatsBase {
        using Stats = typename InnerSolver::Stats;
        WrappedStats(const Stats &stats) : stats(stats) {}
        Stats stats;
        std::shared_ptr<PolymorphicInnerSolverStatsAccumulatorBase>
        accumulator() const override {
            return std::static_pointer_cast<
                PolymorphicInnerSolverStatsAccumulatorBase>(
                std::make_shared<WrappedStatsAccumulator>());
        }
        py::dict to_dict() const override { return stats_to_dict(stats); }
    };

    Stats operator()(
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
        std::chrono::microseconds time_remaining) override {
        auto stats =
            innersolver(problem, Σ, ε, always_overwrite_results, x, y, err_z, time_remaining);
        return {
            std::static_pointer_cast<PolymorphicInnerSolverStatsBase>(
                std::make_shared<WrappedStats>(stats)),
            stats.status,
            stats.ε,
            stats.iterations,
        };
    }
    void stop() override { innersolver.stop(); }
    std::string get_name() const override { return innersolver.get_name(); }
    py::object get_params() const override {
        return py::cast(innersolver.get_params());
    }

    void set_progress_callback(
        std::function<void(const typename InnerSolver::ProgressInfo &)> cb) {
        this->innersolver.set_progress_callback(std::move(cb));
    }

    InnerSolver innersolver;
};

// template <class InnerSolver>
// class PolymorphicInnerSolverFull : public PolymorphicInnerSolverBase {
//   public:
//     PolymorphicInnerSolverFull(InnerSolver &&innersolver)
//         : innersolver(std::forward<InnerSolver>(innersolver)) {}
//     PolymorphicInnerSolverFull(const InnerSolver &innersolver)
//         : innersolver(innersolver) {}
//     template <class... Args>
//     PolymorphicInnerSolverFull(Args... args)
//         : innersolver(InnerSolver{std::forward<Args>(args)...}) {}

//     struct WrappedStatsAccumulator
//         : PolymorphicInnerSolverStatsAccumulatorBase {
//         InnerStatsAccumulator<InnerSolver> acc;
//         void
//         accumulate(const PolymorphicInnerSolverStatsBase &bstats) override {
//             auto &stats = dynamic_cast<const WrappedStats &>(bstats).stats;
//             acc += stats;
//         }
//         py::dict to_dict() const override {
//             return stats_to_dict<InnerSolver>(acc);
//         }
//     };
//     struct WrappedStats : PolymorphicInnerSolverStatsBase {
//         using Stats = typename InnerSolver::Stats;
//         WrappedStats(const Stats &stats) : stats(stats) {}
//         Stats stats;
//         std::shared_ptr<PolymorphicInnerSolverStatsAccumulatorBase>
//         accumulator() const override {
//             return std::static_pointer_cast<
//                 PolymorphicInnerSolverStatsAccumulatorBase>(
//                 std::make_shared<WrappedStatsAccumulator>());
//         }
//         py::dict to_dict() const override {
//             return stats_to_dict<InnerSolver>(stats);
//         }
//     };

//     Stats operator()(
//         /// [in]    Problem description
//         const ProblemFull &problem,
//         /// [in]    Constraint weights @f$ \Sigma @f$
//         crvec Σ1,
//         /// [in]    Constraint weights @f$ \Sigma @f$
//         crvec Σ2,
//         /// [in]    Tolerance @f$ \varepsilon @f$
//         real_t ε,
//         /// [in]    Overwrite @p x, @p y and @p err_z even if not converged
//         bool always_overwrite_results,
//         /// [inout] Decision variable @f$ x @f$
//         rvec x,
//         /// [inout] Lagrange multipliers @f$ y @f$
//         rvec y,
//         /// [out]   Slack variable error @f$ g(x) - z @f$
//         rvec err_z1,
//         /// [out]   Slack variable error @f$ g(x) - z @f$
//         rvec err_z2) override {
//         auto stats =
//             innersolver(problem, Σ1, Σ1, ε, always_overwrite_results, x, y,
//                         err_z1, err_z2);
//         return {
//             std::static_pointer_cast<PolymorphicInnerSolverStatsBase>(
//                 std::make_shared<WrappedStats>(stats)),
//             stats.status,
//             stats.ε,
//             stats.iterations,
//         };
//     }
//     void stop() override { innersolver.stop(); }
//     std::string get_name() const override { return innersolver.get_name(); }
//     py::object get_params() const override {
//         return py::cast(innersolver.get_params());
//     }

//     void set_progress_callback(
//         std::function<void(const typename InnerSolver::ProgressInfo &)> cb) {
//         this->innersolver.set_progress_callback(std::move(cb));
//     }

//     InnerSolver innersolver;
// };

} // namespace pa

#include "polymorphic-panoc-direction.hpp"
#include <panoc-alm/alm.hpp>

namespace pa {

using PolymorphicPGASolver    = PolymorphicInnerSolver<PGASolver>;
using PolymorphicGAAPGASolver = PolymorphicInnerSolver<GAAPGASolver>;
using PolymorphicPANOCSolver =
    PolymorphicInnerSolver<PANOCSolver<PolymorphicPANOCDirectionBase>>;
using PolymorphicPANOCSolverFull =
    PolymorphicInnerSolver<PANOCSolverFull<PolymorphicPANOCDirectionBase>>;
using PolymorphicStructuredPANOCLBFGSSolver =
    PolymorphicInnerSolver<StructuredPANOCLBFGSSolver>;

using PolymorphicALMSolver = ALMSolver<PolymorphicInnerSolverWrapper>;

inline py::dict stats_to_dict(const PolymorphicALMSolver::Stats &s) {
    using py::operator""_a;
    return py::dict{
        "outer_iterations"_a           = s.outer_iterations,
        "elapsed_time"_a               = s.elapsed_time,
        "initial_penalty_reduced"_a    = s.initial_penalty_reduced,
        "penalty_reduced"_a            = s.penalty_reduced,
        "inner_convergence_failures"_a = s.inner_convergence_failures,
        "ε"_a                          = s.ε,
        "δ"_a                          = s.δ,
        "norm_penalty"_a               = s.norm_penalty,
        "status"_a                     = s.status,
        "inner"_a                      = s.inner.to_dict(),
    };
}

template <class InnerSolver>
inline py::dict
stats_to_dict(typename pa::ALMSolverFull<PANOCSolverFull<>>::Stats &s) {
    using py::operator""_a;
    return py::dict{
        "outer_iterations"_a           = s.outer_iterations,
        "elapsed_time"_a               = s.elapsed_time,
        "initial_penalty_reduced"_a    = s.initial_penalty_reduced,
        "penalty_reduced"_a            = s.penalty_reduced,
        "inner_convergence_failures"_a = s.inner_convergence_failures,
        "ε"_a                          = s.ε,
        "δ₁"_a                         = s.δ₁,
        "δ₂"_a                         = s.δ₂,
        "norm_penalty₁"_a              = s.norm_penalty₁,
        "norm_penalty₂"_a              = s.norm_penalty₂,
        "penalty₂"_a                   = s.penalty₂,
        "status"_a                     = s.status,
        "inner"_a                      = stats_to_dict(s.inner),
    };
}

} // namespace pa