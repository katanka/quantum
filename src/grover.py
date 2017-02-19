import pyquil.quil as pq
import pyquil.forest as forest
from pyquil.gates import *
import numpy as np
from collections import Counter

def build_diffusion_operator(num_qubits):
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

def grover(qvm, num_qubits, sol_index):
    p = pq.Program()

    p.defgate("diffusion", build_diffusion_operator(num_qubits))
    p.defgate("oracle", build_oracle(num_qubits, sol_index))

    diffusion_inst = tuple(["diffusion"] + [i for i in range(num_qubits)])
    oracle_inst = tuple(["oracle"] + [i for i in range(num_qubits + 1)])

    steps = int(np.round(np.sqrt(num_qubits)))

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


def most_common(lst):
    lst = map(lambda x: tuple(x), lst)
    data = Counter(lst)
    return data.most_common(1)[0][0]

def main():
    qvm = forest.Connection()

    num_qubits = 4
    sol_index = 2

    p = grover(qvm, num_qubits, sol_index)

    print ("Measured: ")

    output = qvm.run(p, [0,1,2,3], 20)

    print (most_common(output))

    #for line in output:
    #     print(line)


def print_program_state(title, qvm, p, num_qubits):
    print("\n" + title)
    psi = qvm.wavefunction(p)[0]
    for i in range(len(psi)):
        print "%s: %.3f" % (bin(i)[2:].zfill(num_qubits + 1), pow(np.real(psi[i]), 2))

if __name__ == '__main__':
    main()
