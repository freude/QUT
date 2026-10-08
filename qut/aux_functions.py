import numpy as np
import types
import copy
import functools


def detach(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Create copies of the arguments
        args_copy = copy.deepcopy(args)
        kwargs_copy = copy.deepcopy(kwargs)

        # Call the original function with the copies
        result = func(*args_copy, **kwargs_copy)
        return result
    return wrapper


def parse_code(data):

    from qiskit.qasm2 import load as load2
    from qiskit.qasm2 import loads as loads2
    from qiskit.qasm3 import load as load3
    from qiskit.qasm3 import loads as loads3
    from qiskit.circuit.quantumcircuit import QuantumCircuit

    def func(my_func):
        if isinstance(my_func, types.FunctionType):
            return my_func
        else:
            raise Exception

    def circ(my_func):
        if isinstance(my_func, QuantumCircuit):
            return my_func
        else:
            raise Exception

    parsers = [loads3, load3, loads2, load2, func, circ]

    for parser in parsers:
        try:
            return parser(data)
        except Exception:
            continue

    print("All parsing attempts failed.")
    return None


def make_keys(nn):
    ans = []
    nn += 1
    def genbin(n, bs = ''):

        if n-1:
            genbin(n-1, bs + '0')
            genbin(n-1, bs + '1')
        else:
            # print('1' + bs)
            ans.append(bs)

    genbin(nn, bs='')

    return ans


def fro_norm(a, b):

    a = np.array(a)
    b = np.array(b)

    return np.linalg.norm(np.abs(a - b), ord='fro')


def likelihood(aa: np.ndarray, bb: np.ndarray) -> float:

    aa_real = np.real(aa).flatten()
    aa_imag = np.imag(aa).flatten()

    bb_real = np.real(bb).flatten()
    bb_imag = np.imag(bb).flatten()

    ans = 1

    for j in range(len(aa_real)):
        if np.abs(aa_real[j]) > 1e-5:
            ans1 = np.exp(-((aa_real[j] - bb_real[j]) ** 2) / np.abs(aa_real[j]) / 2 / 64)
        else:
            ans1 = 1

        if np.abs(aa_imag[j]) > 1e-5:
            ans2 = np.exp(-((aa_imag[j] - bb_imag[j]) ** 2) / np.abs(aa_imag[j]) / 2 / 64)
        else:
            ans2 = 1

        ans *= (ans1 * ans2)

    return ans


def print_pretty_evaluation_results(title, fr1, fr2, fr3, j1, j2, j3, fnr1, fnr2, fnr3, fpr1, fpr2, fpr3):

    line_lenght = 78

    data1 = [
        ["Pearson’s 𝜒2-test       ",  fr1, j1, fnr1, fpr1],
        ["State tomography        ",  fr2, j2, fnr2, fpr2],
        ["Process tomography      ",  fr3, j3, fnr3, fpr3]
    ]

    # Define header and format template:
    # Left-align Name (10 chars), Right-align Age (5 chars), Left-align City (15 chars)
    row_format1 = "{:<15} {:<15} {:<15} {:<15} {:<15}"
    row_format = "{:<15} {:<15.2f} {:<15.2f} {:<15.2f} {:<15.2f}"

    # print("-" * line_lenght) # Separator line
    print(" " * int(0.5 * (line_lenght - len(title)) - 1), title)
    print("-" * line_lenght) # Separator line
    print(row_format1.format("Protocol           ",
                             "Fraction killed",
                             "      J       ",
                             "   1-alpha    ",
                             "     beta     "))

    print("-" * line_lenght) # Separator line

    for row in data1:
        print(row_format.format(*row))

    print("-" * line_lenght) # Separator line
    print("\n")
