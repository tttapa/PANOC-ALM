from typing import Tuple, Union
import casadi as cs
import panocpy as pa
from tempfile import TemporaryDirectory
import os


def generate_casadi_problem(
    f: cs.Function,
    g: cs.Function,
    second_order: bool = False,
    name: str = "PANOC_ALM_problem",
) -> Tuple[cs.CodeGenerator, int, int, int]:
    """Convert the objective and constraint functions into a CasADi code
    generator.

    :param f:            Objective function.
    :param g:            Constraint function.
    :param second_order: Whether to generate functions for evaluating Hessians.
    :param name: Optional string description of the problem (used for filename).

    :return:   * Code generator that generates the functions and derivatives
                 used by the solvers.
               * Dimensions of the decision variables (primal dimension).
               * Number of nonlinear constraints (dual dimension).
               * Number of parameters.
    """

    assert f.n_in() in [1, 2]
    assert f.n_in() == g.n_in()
    assert f.size1_in(0) == g.size1_in(0)
    if f.n_in() == 2:
        assert f.size1_in(1) == g.size1_in(1)
    assert f.n_out() == 1
    assert g.n_out() == 1
    n = f.size1_in(0)
    m = g.size1_out(0)
    p = f.size1_in(1) if f.n_in() == 2 else 0
    xp = (f.sx_in(0), f.sx_in(1)) if f.n_in() == 2 else (f.sx_in(0),)
    xp_names = (f.name_in(0), f.name_in(1)) if f.n_in() == 2 else (f.name_in(0),)
    x = xp[0]
    y = cs.SX.sym("y", m)
    v = cs.SX.sym("v", n)

    L = f(*xp) + cs.dot(y, g(*xp)) if m > 0 else f(*xp)

    cgname = f"{name}.c"
    cg = cs.CodeGenerator(cgname)
    cg.add(
        cs.Function(
            "f",
            [*xp],
            [f(*xp)],
            [*xp_names],
            ["f"],
        )
    )
    cg.add(
        cs.Function(
            "grad_f",
            [*xp],
            [cs.gradient(f(*xp), x)],
            [*xp_names],
            ["grad_f"],
        )
    )
    cg.add(
        cs.Function(
            "g",
            [*xp],
            [g(*xp)],
            [*xp_names],
            ["g"],
        )
    )
    cg.add(
        cs.Function(
            "grad_g",
            [*xp, y],
            [cs.jtimes(g(*xp), x, y, True)],
            [*xp_names, "y"],
            ["grad_g"],
        )
    )
    if second_order:
        cg.add(
            cs.Function(
                "hess_L",
                [*xp, y],
                [cs.hessian(L, x)[0]],
                [*xp_names, "y"],
                ["hess_L"],
            )
        )
        cg.add(
            cs.Function(
                "hess_L_prod",
                [*xp, y, v],
                [cs.gradient(cs.jtimes(L, x, v, False), x)],
                [*xp_names, "y", "v"],
                ["hess_L_prod"],
            )
        )
    return cg, n, m, p

def generate_casadi_problem_full(
    f: cs.Function,
    g1: cs.Function,
    g2: cs.Function,
    second_order: bool = False,
    name: str = "PANOC_ALM_problem",
) -> Tuple[cs.CodeGenerator, int, int, int]:
    """Convert the objective and constraint functions into a CasADi code
    generator.

    :param f:            Objective function.
    :param g1:           ALM constraint function.
    :param g2:           Quadratic penalty constraint function.
    :param second_order: Whether to generate functions for evaluating Hessians.
    :param name: Optional string description of the problem (used for filename).

    :return:   * Code generator that generates the functions and derivatives
                 used by the solvers.
               * Dimensions of the decision variables (primal dimension).
               * Number of nonlinear constraints (dual dimension).
               * Number of parameters.
    """

    assert f.n_in() in [1, 2]
    assert f.n_in() == g1.n_in() == g2.n_in()
    assert f.size1_in(0) == g1.size1_in(0) == g2.size1_in(0)
    if f.n_in() == 2:
        assert f.size1_in(1) == g1.size1_in(1) == g2.size1_in(1)
    assert f.n_out() == 1
    assert g1.n_out() == 1
    assert g2.n_out() == 1
    n = f.size1_in(0)
    m1 = g1.size1_out(0)
    m2 = g2.size1_out(0)
    p = f.size1_in(1) if f.n_in() == 2 else 0
    xp = (f.sx_in(0), f.sx_in(1)) if f.n_in() == 2 else (f.sx_in(0),)
    xp_names = (f.name_in(0), f.name_in(1)) if f.n_in() == 2 else (f.name_in(0),)
    x = xp[0]
    y1 = cs.SX.sym("y1", m1)
    y2 = cs.SX.sym("y2", m2)
    v = cs.SX.sym("v", n)

    L = f(*xp) + cs.dot(y1, g1(*xp)) if m1 > 0 else f(*xp)

    cgname = f"{name}.c"
    cg = cs.CodeGenerator(cgname)
    cg.add(
        cs.Function(
            "f",
            [*xp],
            [f(*xp)],
            [*xp_names],
            ["f"],
        )
    )
    cg.add(
        cs.Function(
            "grad_f",
            [*xp],
            [cs.gradient(f(*xp), x)],
            [*xp_names],
            ["grad_f"],
        )
    )
    cg.add(
        cs.Function(
            "g1",
            [*xp],
            [g1(*xp)],
            [*xp_names],
            ["g1"],
        )
    )
    cg.add(
        cs.Function(
            "grad_g1",
            [*xp, y1],
            [cs.jtimes(g1(*xp), x, y1, True)],
            [*xp_names, "y2"],
            ["grad_g1"],
        )
    )
    cg.add(
        cs.Function(
            "g2",
            [*xp],
            [g2(*xp)],
            [*xp_names],
            ["g2"],
        )
    )
    cg.add(
        cs.Function(
            "grad_g2",
            [*xp, y2],
            [cs.jtimes(g2(*xp), x, y2, True) if m2 > 0 else []],
            [*xp_names, "y2"],
            ["grad_g2"],
        )
    )
    if second_order:
        cg.add(
            cs.Function(
                "hess_L",
                [*xp, y1],
                [cs.hessian(L, x)[0]],
                [*xp_names, "y1"],
                ["hess_L"],
            )
        )
        cg.add(
            cs.Function(
                "hess_L_prod",
                [*xp, y1, v],
                [cs.gradient(cs.jtimes(L, x, v, False), x)],
                [*xp_names, "y1", "v"],
                ["hess_L_prod"],
            )
        )
    return cg, n, m1, m2, p

def compile_and_load_problem(
    cgen: cs.CodeGenerator,
    n: int,
    m: int,
    p: int,
    name: str = "PANOC_ALM_problem",
) -> Union[pa.Problem, pa.ProblemWithParam]:
    """Compile the C-code using the given code-generator and load it as a
    panocpy Problem.

    :param cgen: Code generator to generate C-code for the costs and the
                 constraints with.
    :param n:    Dimensions of the decision variables (primal dimension).
    :param m:    Number of nonlinear constraints (dual dimension).
    :param p:    Number of parameters.
    :param name: Optional string description of the problem (used for filename).

    :return:   * Problem specification that can be passed to the solvers.
    """

    with TemporaryDirectory(prefix="") as tmpdir:
        cfile = cgen.generate(os.path.join(tmpdir, ""))
        sofile = os.path.join(tmpdir, f"{name}.so")
        os.system(f"cc -fPIC -shared -O3 -march=native {cfile} -o {sofile}")
        if p > 0:
            prob = pa.load_casadi_problem_with_param(sofile, n, m)
        else:
            prob = pa.load_casadi_problem(sofile, n, m)
    return prob

def compile_and_load_problem_full(
    cgen: cs.CodeGenerator,
    n: int,
    m1: int,
    m2: int,
    p: int,
    name: str = "PANOC_full_problem",
) -> Union[pa.ProblemFull, pa.ProblemFullWithParam]:
    """Compile the C-code using the given code-generator and load it as a
    panocpy ProblemFull.

    :param cgen: Code generator to generate C-code for the costs and the
                 constraints with.
    :param n:    Dimensions of the decision variables (primal dimension).
    :param m1:   Number of ALM constraints (dual dimension 1).
    :param m2:   Number of quadratic penalty constraints (dual dimension 2).
    :param p:    Number of parameters.
    :param name: Optional string description of the problem (used for filename).

    :return:   * ProblemFull specification that can be passed to the solvers.
    """

    with TemporaryDirectory(prefix="") as tmpdir:
        cfile = cgen.generate(tmpdir)
        sofile = os.path.join(tmpdir, f"{name}.so")
        os.system(f"cc -fPIC -shared -O3 -march=native {cfile} -o {sofile}")
        if p > 0:
            prob = pa.load_casadi_problem_full_with_param(sofile, n, m1, m2)
        else:
            prob = pa.load_casadi_problem_full(sofile, n, m1, m2)
    return prob


def generate_and_compile_casadi_problem(
    f: cs.Function,
    g: cs.Function,
    second_order: bool = False,
    name: str = "PANOC_ALM_problem",
) -> Union[pa.Problem, pa.ProblemWithParam]:
    """Compile the objective and constraint functions into a panocpy Problem.

    :param f:            Objective function.
    :param g:            Constraint function.
    :param second_order: Whether to generate functions for evaluating Hessians.
    :param name: Optional string description of the problem (used for filename).

    :return:   * Problem specification that can be passed to the solvers.
    """
    cgen = generate_casadi_problem(f, g, second_order, name)
    return compile_and_load_problem(*cgen, name)