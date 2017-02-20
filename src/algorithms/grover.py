import pyquil.quil as pq
import pyquil.forest as forest
from pyquil.gates import *
import numpy as np
from collections import Counter

"""
        GROVER'S ALGORITHM
        by Andrew Gleeson

        Grover's algorithm is a quantum algorithm which probabilistically finds an element in list given a condition
        function which returns 1 if the element matches the condition and 0 otherwise. Let the list have 2^n elements
        for some n, meaning each element has a unique n-bit ID. Grover's algorithm has two main elements: the oracle
        transformation, and inverting about the mean.

        The oracle is a unitary transformation which has the following properties, given some condition function C(x):

            1. If C(x) = true, ORACLE(x) = -x
            2. If C(x) = false, ORACLE(x) = x

        We can represent this as the identity matrix, where the 1 in the same row as the solution has been flipped to -1.
        For example, if the solution is the first item in a list of 4 items, then:

            ORACLE = [[-1, 0, 0, 0],
                      [0, 1, 0, 0],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1]]

        The next part of the algorithm is the transformation which inverts about the mean, which is also called the
        Grover diffusion operator. Given a list of numbers, this operation calculates the average value of the list,
        and then reflects each value across it. For example, if the average value is m, then:

            INVERT(x) = m + (m - x) = 2m - x.

        Given a uniform superposition state |s>, the diffusion operator can be represented by:

            INVERT = 2|s><s| - I

        Grover's algorithm works by repeatedly applying the oracle and inversion operators on a uniform input state,
        which causes the correct index to be inverted by the oracle, and then inverting about the mean makes that
        index more likely to be measured.

        Interestingly, the probability of measuring the correct index is periodic - it first peaks at sqrt(N)/2,
        where N is the number of qubits.


"""


def grover(num_qubits, sol_index):
    program = pq.Program()

    # In PyQuil, we can define custom gates like this
    program += build_invert_about_mean(num_qubits)
    program += build_oracle(num_qubits, sol_index)

    diffusion_inst = tuple(["INVERT"] + [i for i in range(num_qubits)])
    oracle_inst = tuple(["ORACLE"] + [i for i in range(num_qubits + 1)])

    # the probability of success is greatest at iteration sqrt(N)/2
    num_steps = int(np.round(np.sqrt(num_qubits) / 2))

    # initialize
    for qubit in range(num_qubits):
        program.inst(X(qubit), X(qubit))

    # Set ancillary bit to |1>
    program.inst(X(num_qubits))

    # Create uniform input state
    for i in range(num_qubits + 1):
        program.inst(H(i))

    # Repeatedly apply Grover iteration
    for step in range(num_steps):
        program.inst(oracle_inst)
        program.inst(diffusion_inst)

    # Measure qubits
    for i in range(num_qubits):
        program.measure(i, i)

    return program


def build_invert_about_mean(num_qubits):
    """ Builds the diffusion operator
    """

    dimension = pow(2, num_qubits)

    # State vector representing |00>
    zero = np.matrix([1] + [0 for _ in range(dimension - 1)])

    #
    invert = 2 * np.kron(zero, zero.T) - np.eye(dimension)

    # Hadamard gate on one qubit
    hadamard_1 = np.matrix([[1, 1], [1, -1]]) / np.sqrt(2)

    # Hadamard gate on two qubits
    hadamard = hadamard_1
    for _ in range(num_qubits - 1):
        hadamard = np.kron(hadamard, hadamard_1)

    invert_about_mean_matrix = hadamard.dot(invert).dot(hadamard)

    invert_about_mean = pq.Program()
    invert_about_mean.defgate("INVERT", invert_about_mean_matrix)
    return invert_about_mean


def build_oracle(num_qubits, sol_index):
    """ Builds the oracle gate
    """
    state_size = pow(2, num_qubits + 1)
    diagonal_entries = [1 for _ in range(state_size)]
    # bit hackery for correctness (not sure why but otherwise it breaks)
    reversed_index = int(bin(sol_index)[2:].zfill(num_qubits + 1)[::-1], 2)
    diagonal_entries[reversed_index] = -1

    oracle = pq.Program()
    oracle.defgate("ORACLE", np.diag(diagonal_entries))
    return oracle


# UTIL
def main():
    qvm = forest.Connection()

    num_qubits = 2
    sol_index = 0

    p = grover(num_qubits, sol_index)

    print p

    output = qvm.run(p, [0, 1, 2, 3], 20)

    print most_common(output)


def most_common(lst):
    lst = map(lambda x: tuple(x), lst)
    data = Counter(lst)
    return data.most_common(1)[0][0]


if __name__ == '__main__':
    main()
