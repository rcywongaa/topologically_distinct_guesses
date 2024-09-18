import sympy
from sympy import (
    Symbol,
    symbols,
    Matrix,
    Function,
    sqrt,
    sin,
    cos,
    diff,
    init_printing,
    pprint,
    latex,
    print_latex,
)
from math import pi


def VectorSymbol(prefix, rows):
    return Matrix([[Symbol(prefix + str(i))] for i in range(1, rows + 1)])


def VectorFunction(prefix, rows, *variables):
    return Matrix([[Function(prefix + str(i))(*variables)] for i in range(rows)])


def make_R(phi):
    phi1 = phi[0]
    phi2 = phi[1]
    phi3 = phi[2]
    v = Matrix([[0], [-phi3], [phi2]])

    def skew(a):
        a1 = a[0]
        a2 = a[1]
        a3 = a[2]
        return Matrix([[0, -a3, a2], [a3, 0, -a1], [-a2, a1, 0]])

    I = Matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    c = phi1
    R = I + skew(v) + (skew(v) @ skew(v)) * 1 / (1 + c)
    R = Matrix(
        [
            [R[0, 0], R[0, 1], R[0, 2], 0],
            [R[1, 0], R[1, 1], R[1, 2], 0],
            [R[2, 0], R[2, 1], R[2, 2], 0],
            [0, 0, 0, 1],
        ]
    )
    return R


def make_T(x, y, z):
    return Matrix([[1, 0, 0, x], [0, 1, 0, y], [0, 0, 1, z], [0, 0, 0, 1]])


def eq1():
    x_b = VectorSymbol("x_b", 3)
    x_e = VectorSymbol("x_e", 3)
    delta = symbols("Delta")
    # delta = x_e - x_b
    d = symbols("d")
    # d = delta.norm()
    phi = VectorSymbol("phi", 3)
    # phi = delta.normalized()
    n = symbols("n")
    R = make_R(phi)
    T_xb = make_T(*x_b)
    l_1perp = symbols("l_{1\perp}")
    r = symbols("r")
    T_d = make_T(l_1perp, 0, 0)
    # r = sqrt(1-d**2/4)
    S = Matrix([[r, 0, 0, 0], [0, r, 0, 0], [0, 0, r, 0], [0, 0, 0, 1]])
    c = Matrix([[0], [cos(n)], [sin(n)], [1]])

    f = T_xb @ R @ T_d @ S @ c
    pprint(f)
    print_latex(f)


if __name__ == "__main__":
    init_printing()
    eq1()
