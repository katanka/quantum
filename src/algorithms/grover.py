import pyquil.quil as pq
import pyquil.forest as forest
from pyquil.gates import *
import numpy as np
from collections import Counter

"""
        GROVER'S ALGORITHM

            Grover's algorithm is a quantum algorithm which probabilistically finds an element in list given a condition
        function which returns 1 if the element matches the condition and 0 otherwise. Let the list have 2^n
        elements for some n, meaning each element has a unique n-bit ID. Grover's algorithm works be encoding this
        condition function, also known as the oracle, into a unitary transformation which has the following properties:

            1. If x meets the condition, ORACLE(x) = -x
            2. If x does not meet the condition, ORACLE(x) = x

        We can represent this as the identity matrix, where the 1 in the same row as the solution has been flipped to -1.
        

"""

def grover(qvm, num_qubits, sol_index):
    p = pq.Program()

    p.defgate("invert-about-mean", build_invert_about_mean(num_qubits))
    p.defgate("oracle", build_oracle(num_qubits, sol_index))

    diffusion_inst = tuple(["invert-about-mean"] + [i for i in range(num_qubits)])
    oracle_inst = tuple(["oracle"] + [i for i in range(num_qubits + 1)])

    steps = int(np.round(np.sqrt(num_qubits)/2))

    # initialize
    for qubit in range(num_qubits+1):
        p.inst(X(qubit), X(qubit))

    print_program_state("Initial state:", qvm, p, num_qubits)

    # Set ancillary bit to |1>
    p.inst(X(num_qubits))

    print_program_state("Flip ancillary bit:", qvm, p, num_qubits)

    # Create uniform input state
    for i in range(num_qubits + 1):
        p.inst(H(i))

    print_program_state("Hadamard:", qvm, p, num_qubits)

    # Grover's Algorithm
    for step in range(steps):
        p.inst(oracle_inst)
        p.inst(diffusion_inst)
        print_program_state("After step %d:" % (step + 1), qvm, p, num_qubits)


    # Create uniform input state
    for i in range(num_qubits+1):
        p.measure(i, i)

    return p

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
    for _ in range(num_qubits-1):
        hadamard = np.kron(hadamard, hadamard_1)

    return hadamard.dot(invert).dot(hadamard)

def build_oracle(num_qubits, sol_index):
    """ Builds the oracle
    """

    # add 1 to account for ancillary qubit
    dim = pow(2, num_qubits + 1)

    arr = [1 for _ in range(dim)]

    reversed = int(bin(sol_index)[2:].zfill(num_qubits + 1)[::-1], 2)

    arr[reversed] = -1

    return np.diag(arr)


# UTIL

def most_common(lst):
    lst = map(lambda x: tuple(x), lst)
    data = Counter(lst)
    return data.most_common(1)[0][0]

def main():
    qvm = forest.Connection()

    num_qubits = 4
    sol_index = 1

    p = grover(qvm, num_qubits, sol_index)

    print ("Measured: ")

    output = qvm.run(p, [0,1,2,3], 20)

    print (most_common(output))



def print_program_state(title, qvm, p, num_qubits):
    print("\n" + title)
    psi = qvm.wavefunction(p)[0]
    for i in range(len(psi)):
        print "%s: %.3f" % (bin(i)[2:].zfill(num_qubits + 1), pow(np.real(psi[i]), 2))

if __name__ == '__main__':
    main()