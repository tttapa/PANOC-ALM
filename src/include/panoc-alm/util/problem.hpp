#pragma once

#include "box.hpp"

#include <functional>
#include <memory>
#include <type_traits>

namespace pa {

/**
 * @class Problem
 * @brief   Problem description for minimization problems.
 * 
 * @f[ \begin{aligned}
 *  & \underset{x}{\text{minimize}}
 *  & & f(x) &&&& f : \mathbb{R}^n \rightarrow \mathbb{R} \\
 *  & \text{subject to}
 *  & & \underline{x} \le x \le \overline{x} \\
 *  &&& \underline{z} \le g(x) \le \overline{z} &&&& g :
 *  \mathbb{R}^n \rightarrow \mathbb{R}^m
 * \end{aligned} @f]
 */
struct Problem {
    unsigned int n; ///< Number of decision variables, dimension of x
    unsigned int m; ///< Number of constraints, dimension of g(x) and z
    Box C;          ///< Constraints of the decision variables, @f$ x \in C @f$
    Box D;          ///< Other constraints, @f$ g(x) \in D @f$

    /// Signature of the function that evaluates the cost @f$ f(x) @f$
    /// @param  [in] x
    ///         Decision variable @f$ x \in \mathbb{R}^n @f$
    using f_sig = real_t(crvec x);
    /// Signature of the function that evaluates the gradient of the cost
    /// function @f$ \nabla f(x) @f$
    /// @param  [in] x
    ///         Decision variable @f$ x \in \mathbb{R}^n @f$
    /// @param  [out] grad_fx
    ///         Gradient of cost function @f$ \nabla f(x) \in \mathbb{R}^n @f$
    using grad_f_sig = void(crvec x, rvec grad_fx);
    /// Signature of the function that evaluates the constraints @f$ g(x) @f$
    /// @param  [in] x
    ///         Decision variable @f$ x \in \mathbb{R}^n @f$
    /// @param  [out] gx
    ///         Value of the constraints @f$ g(x) \in \mathbb{R}^m @f$
    using g_sig = void(crvec x, rvec gx);
    /// Signature of the function that evaluates the gradient of the constraints
    /// times a vector
    /// @f$ \nabla g(x)\ y @f$
    /// @param  [in] x
    ///         Decision variable @f$ x \in \mathbb{R}^n @f$
    /// @param  [in] y
    ///         Vector @f$ y \in \mathbb{R}^m @f$ to multiply the gradient by
    /// @param  [out] grad_gxy
    ///         Gradient of the constraints
    ///         @f$ \nabla g(x)\ y \in \mathbb{R}^n @f$
    using grad_g_prod_sig = void(crvec x, crvec y, rvec grad_gxy);
    /// Signature of the function that evaluates the gradient of one specific
    /// constraints
    /// @f$ \nabla g_i(x) @f$
    /// @param  [in] x
    ///         Decision variable @f$ x \in \mathbb{R}^n @f$
    /// @param  [in] i
    ///         Which constraint @f$ 0 \le i \lt m @f$
    /// @param  [out] grad_gi
    ///         Gradient of the constraint
    ///         @f$ \nabla g_i(x) \mathbb{R}^n @f$
    using grad_gi_sig = void(crvec x, unsigned i, rvec grad_gi);
    /// Signature of the function that evaluates the Hessian of the Lagrangian
    /// multiplied by a vector
    /// @f$ \nabla_{xx}^2L(x, y)\ v @f$
    /// @param  [in] x
    ///         Decision variable @f$ x \in \mathbb{R}^n @f$
    /// @param  [in] y
    ///         Lagrange multipliers @f$ y \in \mathbb{R}^m @f$
    /// @param  [in] v
    ///         Vector to multiply by @f$ v \in \mathbb{R}^n @f$
    /// @param  [out] Hv
    ///         Hessian-vector product
    ///         @f$ \nabla_{xx}^2 L(x, y)\ v \in \mathbb{R}^{n} @f$
    using hess_L_prod_sig = void(crvec x, crvec y, crvec v, rvec Hv);
    /// Signature of the function that evaluates the Hessian of the Lagrangian
    /// @f$ \nabla_{xx}^2L(x, y) @f$
    /// @param  [in] x
    ///         Decision variable @f$ x \in \mathbb{R}^n @f$
    /// @param  [in] y
    ///         Lagrange multipliers @f$ y \in \mathbb{R}^m @f$
    /// @param  [out] H
    ///         Hessian @f$ \nabla_{xx}^2 L(x, y) \in \mathbb{R}^{n\times n} @f$
    using hess_L_sig = void(crvec x, crvec y, rmat H);

    /// Cost function @f$ f(x) @f$
    std::function<f_sig> f;
    /// Gradient of the cost function @f$ \nabla f(x) @f$
    std::function<grad_f_sig> grad_f;
    /// Constraint function @f$ g(x) @f$
    std::function<g_sig> g;
    /// Gradient of the constraint function times vector @f$ \nabla g(x)\ y @f$
    std::function<grad_g_prod_sig> grad_g_prod;
    /// Gradient of a specific constraint @f$ \nabla g_i(x) @f$
    std::function<grad_gi_sig> grad_gi;
    /// Hessian of the Lagrangian function times vector
    /// @f$ \nabla_{xx}^2 L(x, y)\ v @f$
    std::function<hess_L_prod_sig> hess_L_prod;
    /// Hessian of the Lagrangian function @f$ \nabla_{xx}^2 L(x, y) @f$
    std::function<hess_L_sig> hess_L;

    Problem() = default;
    Problem(unsigned int n, unsigned int m)
        : n(n), m(m), C{vec::Constant(n, +inf), vec::Constant(n, -inf)},
          D{vec::Constant(m, +inf), vec::Constant(m, -inf)} {}
    Problem(unsigned n, unsigned int m, Box C, Box D, std::function<f_sig> f,
            std::function<grad_f_sig> grad_f, std::function<g_sig> g,
            std::function<grad_g_prod_sig> grad_g_prod,
            std::function<grad_gi_sig> grad_gi,
            std::function<hess_L_prod_sig> hess_L_prod,
            std::function<hess_L_sig> hess_L)
        : n(n), m(m), C(std::move(C)), D(std::move(D)), f(std::move(f)),
          grad_f(std::move(grad_f)), g(std::move(g)),
          grad_g_prod(std::move(grad_g_prod)), grad_gi(std::move(grad_gi)),
          hess_L_prod(std::move(hess_L_prod)), hess_L(std::move(hess_L)) {}
};

class ProblemWithParam : public pa::Problem {
  public:
    using pa::Problem::Problem;
    void set_param(pa::crvec p) { *param = p; }
    void set_param(pa::vec &&p) { *param = std::move(p); }
    pa::vec &get_param() { return *param; }
    const pa::vec &get_param() const { return *param; }
    std::shared_ptr<pa::vec> get_param_ptr() const { return param; }

  private:
    std::shared_ptr<pa::vec> param = std::make_shared<pa::vec>();
};

struct EvalCounter {
    unsigned f           = 0;
    unsigned grad_f      = 0;
    unsigned g           = 0;
    unsigned grad_g_prod = 0;
    unsigned grad_gi     = 0;
    unsigned hess_L_prod = 0;
    unsigned hess_L      = 0;

    void reset() { *this = {}; }
};

inline EvalCounter &operator+=(EvalCounter &a, EvalCounter b) {
    a.f += b.f;
    a.grad_f += b.grad_f;
    a.g += b.g;
    a.grad_g_prod += b.grad_g_prod;
    a.grad_gi += b.grad_gi;
    a.hess_L_prod += b.hess_L_prod;
    a.hess_L += b.hess_L;
    return a;
}

inline EvalCounter operator+(EvalCounter a, EvalCounter b) { return a += b; }

class ProblemWithCounters : public Problem {
  public:
    ProblemWithCounters(Problem &&p) : Problem(std::move(p)) {
        attach_counters(*this);
    }
    ProblemWithCounters(const Problem &p) : Problem(p) {
        attach_counters(*this);
    }

    ProblemWithCounters()                            = delete;
    ProblemWithCounters(const ProblemWithCounters &) = delete;
    ProblemWithCounters(ProblemWithCounters &&)      = delete;
    ProblemWithCounters &operator=(const ProblemWithCounters &) = delete;
    ProblemWithCounters &operator=(ProblemWithCounters &&) = delete;

  public:
    EvalCounter evaluations;

  private:
    static void attach_counters(ProblemWithCounters &);
};

/// Moves the state constraints in the set C to the set D, resulting in an
/// unconstraint inner problem. The new constraints function g becomes the
/// concatenation of the original g function and the identity function. The
/// new set D is the cartesian product of the original D × C.
class ProblemOnlyD : public Problem {
  public:
    ProblemOnlyD(Problem &&p) : original(std::move(p)) { transform(); }
    ProblemOnlyD(const Problem &p) : original(p) { transform(); }

  private:
    Problem original; // TODO: Keeping this copy around is unnecessary.
    vec work;

    void transform();
};

/**
 * @class ProblemFull
 * @brief   Problem description for minimization problems.
 * 
 * @f[ \begin{aligned}
 *  & \underset{x}{\text{minimize}}
 *  & & f(x) &&&& f : \mathbb{R}^n \rightarrow \mathbb{R} \\
 *  & \text{subject to}
 *  & & \underline{x} \le x \le \overline{x} \\
 *  &&& \underline{z} \le g(x) \le \overline{z} &&&& g :
 *  \mathbb{R}^n \rightarrow \mathbb{R}^m
 * \end{aligned} @f]
 */
struct ProblemFull {
    unsigned int n; ///< Number of decision variables, dimension of x
    unsigned int m1; ///< Number of ALM constraints, dimension of g1(x) and z1
    unsigned int m2; ///< Number of quadratic penalty constraints, dimension of g2(x) and z2
    Box C;          ///< Constraints of the decision variables, @f$ x \in C @f$
    Box D1;          ///< ALM constraints, @f$ g1(x) \in D1 @f$
    Box D2;          ///< Quadratic penalty constraints, @f$ g2(x) \in D2 @f$

    /// Signature of the function that evaluates the cost @f$ f(x) @f$
    /// @param  [in] x
    ///         Decision variable @f$ x \in \mathbb{R}^n @f$
    using f_sig = real_t(crvec x);
    /// Signature of the function that evaluates the gradient of the cost
    /// function @f$ \nabla f(x) @f$
    /// @param  [in] x
    ///         Decision variable @f$ x \in \mathbb{R}^n @f$
    /// @param  [out] grad_fx
    ///         Gradient of cost function @f$ \nabla f(x) \in \mathbb{R}^n @f$
    using grad_f_sig = void(crvec x, rvec grad_fx);
    /// Signature of the function that evaluates the ALM constraints @f$ g1(x) @f$
    /// @param  [in] x
    ///         Decision variable @f$ x \in \mathbb{R}^n @f$
    /// @param  [out] g1x
    ///         Value of the constraints @f$ g1(x) \in \mathbb{R}^m @f$
    using g1_sig = void(crvec x, rvec g1x);
    /// Signature of the function that evaluates the gradient of the ALM constraints
    /// times a vector
    /// @f$ \nabla g1(x)\ y @f$
    /// @param  [in] x
    ///         Decision variable @f$ x \in \mathbb{R}^n @f$
    /// @param  [in] y
    ///         Vector @f$ y \in \mathbb{R}^m1 @f$ to multiply the gradient by
    /// @param  [out] grad_g1xy
    ///         Gradient of the constraints
    ///         @f$ \nabla g1(x)\ y \in \mathbb{R}^n @f$
    using grad_g1_prod_sig = void(crvec x, crvec y, rvec grad_g1xy);
    /// Signature of the function that evaluates the gradient of one specific
    /// ALM constraint
    /// @f$ \nabla g1_i(x) @f$
    /// @param  [in] x
    ///         Decision variable @f$ x \in \mathbb{R}^n @f$
    /// @param  [in] i
    ///         Which constraint @f$ 0 \le i \lt m1 @f$
    /// @param  [out] grad_g1i
    ///         Gradient of the constraint
    ///         @f$ \nabla g1_i(x) \mathbb{R}^n @f$
    using grad_g1i_sig = void(crvec x, unsigned i, rvec grad_g1i);
    /// Signature of the function that evaluates the quadratic penalty 
    /// constraints @f$ g2(x) @f$
    /// @param  [in] x
    ///         Decision variable @f$ x \in \mathbb{R}^n @f$
    /// @param  [out] g2x
    ///         Value of the constraints @f$ g2(x) \in \mathbb{R}^m @f$
    using g2_sig = void(crvec x, rvec g2x);
    /// Signature of the function that evaluates the gradient of the quadratic
    /// penalty constraints times a vector
    /// @f$ \nabla g2(x)\ y @f$
    /// @param  [in] x
    ///         Decision variable @f$ x \in \mathbb{R}^n @f$
    /// @param  [in] y
    ///         Vector @f$ y \in \mathbb{R}^m @f$ to multiply the gradient by
    /// @param  [out] grad_g2xy
    ///         Gradient of the constraints
    ///         @f$ \nabla g2(x)\ y \in \mathbb{R}^n @f$
    using grad_g2_prod_sig = void(crvec x, crvec y, rvec grad_g2xy);
    /// Signature of the function that evaluates the gradient of one specific
    /// quadratic penalty constraints
    /// @f$ \nabla g2_i(x) @f$
    /// @param  [in] x
    ///         Decision variable @f$ x \in \mathbb{R}^n @f$
    /// @param  [in] i
    ///         Which constraint @f$ 0 \le i \lt m2 @f$
    /// @param  [out] grad_g2i
    ///         Gradient of the constraint
    ///         @f$ \nabla g2_i(x) \mathbb{R}^n @f$
    using grad_g2i_sig = void(crvec x, unsigned i, rvec grad_g2i);
    /// Signature of the function that evaluates the Hessian of the Lagrangian
    /// multiplied by a vector
    /// @f$ \nabla_{xx}^2L(x, y)\ v @f$
    /// @param  [in] x
    ///         Decision variable @f$ x \in \mathbb{R}^n @f$
    /// @param  [in] y
    ///         Lagrange multipliers @f$ y \in \mathbb{R}^m @f$
    /// @param  [in] v
    ///         Vector to multiply by @f$ v \in \mathbb{R}^n @f$
    /// @param  [out] Hv
    ///         Hessian-vector product
    ///         @f$ \nabla_{xx}^2 L(x, y)\ v \in \mathbb{R}^{n} @f$
    using hess_L_prod_sig = void(crvec x, crvec y, crvec v, rvec Hv);
    /// Signature of the function that evaluates the Hessian of the Lagrangian
    /// @f$ \nabla_{xx}^2L(x, y) @f$
    /// @param  [in] x
    ///         Decision variable @f$ x \in \mathbb{R}^n @f$
    /// @param  [in] y
    ///         Lagrange multipliers @f$ y \in \mathbb{R}^m @f$
    /// @param  [out] H
    ///         Hessian @f$ \nabla_{xx}^2 L(x, y) \in \mathbb{R}^{n\times n} @f$
    using hess_L_sig = void(crvec x, crvec y, rmat H);

    /// Cost function @f$ f(x) @f$
    std::function<f_sig> f;
    /// Gradient of the cost function @f$ \nabla f(x) @f$
    std::function<grad_f_sig> grad_f;
    /// Constraint function @f$ g1(x) @f$
    std::function<g1_sig> g1;
    /// Gradient of the constraint function times vector @f$ \nabla g1(x)\ y @f$
    std::function<grad_g1_prod_sig> grad_g1_prod;
    /// Gradient of a specific constraint @f$ \nabla g1_i(x) @f$
    std::function<grad_g1i_sig> grad_g1i;
    /// Constraint function @f$ g2(x) @f$
    std::function<g2_sig> g2;
    /// Gradient of the constraint function times vector @f$ \nabla g2(x)\ y @f$
    std::function<grad_g2_prod_sig> grad_g2_prod;
    /// Gradient of a specific constraint @f$ \nabla g2_i(x) @f$
    std::function<grad_g2i_sig> grad_g2i;
    /// Hessian of the Lagrangian function times vector
    /// @f$ \nabla_{xx}^2 L(x, y)\ v @f$
    std::function<hess_L_prod_sig> hess_L_prod;
    /// Hessian of the Lagrangian function @f$ \nabla_{xx}^2 L(x, y) @f$
    std::function<hess_L_sig> hess_L;

    ProblemFull() = default;
    ProblemFull(unsigned int n, unsigned int m1, unsigned int m2)
        : n(n), m1(m1), m2(m2), C{vec::Constant(n, +inf), vec::Constant(n, -inf)},
          D1{vec::Constant(m1, +inf), vec::Constant(m1, -inf)},
          D2{vec::Constant(m1, +inf), vec::Constant(m1, -inf)} {}
    ProblemFull(unsigned n, unsigned int m1, unsigned int m2, Box C, Box D1, Box D2,
            std::function<f_sig> f,
            std::function<grad_f_sig> grad_f,
            std::function<g1_sig> g1,
            std::function<grad_g1_prod_sig> grad_g1_prod,
            std::function<grad_g1i_sig> grad_g1i,
            std::function<g2_sig> g2,
            std::function<grad_g2_prod_sig> grad_g2_prod,
            std::function<grad_g2i_sig> grad_g2i,
            std::function<hess_L_prod_sig> hess_L_prod,
            std::function<hess_L_sig> hess_L)
        : n(n), m1(m1), m2(m2), C(std::move(C)), D1(std::move(D1)), D2(std::move(D2)),
          f(std::move(f)), grad_f(std::move(grad_f)), g1(std::move(g1)),
          grad_g1_prod(std::move(grad_g1_prod)), grad_g1i(std::move(grad_g1i)),
          g2(std::move(g2)), grad_g2_prod(std::move(grad_g2_prod)),
          grad_g2i(std::move(grad_g2i)), hess_L_prod(std::move(hess_L_prod)),
          hess_L(std::move(hess_L)) {}
};

class ProblemFullWithParam : public pa::ProblemFull {
  public:
    using pa::ProblemFull::ProblemFull;
    void set_param(pa::crvec p) { *param = p; }
    void set_param(pa::vec &&p) { *param = std::move(p); }
    pa::vec &get_param() { return *param; }
    const pa::vec &get_param() const { return *param; }
    std::shared_ptr<pa::vec> get_param_ptr() const { return param; }

  private:
    std::shared_ptr<pa::vec> param = std::make_shared<pa::vec>();
};

struct EvalCounterFull {
    unsigned f           = 0;
    unsigned grad_f      = 0;
    unsigned g           = 0;
    unsigned grad_g_prod = 0;
    unsigned grad_gi     = 0;
    unsigned hess_L_prod = 0;
    unsigned hess_L      = 0;

    void reset() { *this = {}; }
};

inline EvalCounterFull &operator+=(EvalCounterFull &a, EvalCounterFull b) {
    a.f += b.f;
    a.grad_f += b.grad_f;
    a.g += b.g;
    a.grad_g_prod += b.grad_g_prod;
    a.grad_gi += b.grad_gi;
    a.hess_L_prod += b.hess_L_prod;
    a.hess_L += b.hess_L;
    return a;
}

inline EvalCounterFull operator+(EvalCounterFull a, EvalCounterFull b) { return a += b; }

// class ProblemWithCounters : public Problem {
//   public:
//     ProblemWithCounters(Problem &&p) : Problem(std::move(p)) {
//         attach_counters(*this);
//     }
//     ProblemWithCounters(const Problem &p) : Problem(p) {
//         attach_counters(*this);
//     }

//     ProblemWithCounters()                            = delete;
//     ProblemWithCounters(const ProblemWithCounters &) = delete;
//     ProblemWithCounters(ProblemWithCounters &&)      = delete;
//     ProblemWithCounters &operator=(const ProblemWithCounters &) = delete;
//     ProblemWithCounters &operator=(ProblemWithCounters &&) = delete;

//   public:
//     EvalCounter evaluations;

//   private:
//     static void attach_counters(ProblemWithCounters &);
// };

// /// Moves the state constraints in the set C to the set D, resulting in an
// /// unconstraint inner problem. The new constraints function g becomes the
// /// concatenation of the original g function and the identity function. The
// /// new set D is the cartesian product of the original D × C.
// class ProblemOnlyD : public Problem {
//   public:
//     ProblemOnlyD(Problem &&p) : original(std::move(p)) { transform(); }
//     ProblemOnlyD(const Problem &p) : original(p) { transform(); }

//   private:
//     Problem original; // TODO: Keeping this copy around is unnecessary.
//     vec work;

//     void transform();
// };

} // namespace pa
