import qiskit.quantum_info as qi
from qiskit import *
import subprocess
import itertools
import warnings
import numpy as np
import json
from math import sqrt


warnings.filterwarnings("ignore")


class Tensor:

    def __init__(self, name, indices=[]):
        self.name = name
        self.indices = []

    def __repr__(self):
        return f"{self.name}: {self.indices}"


def complex_to_tuple(lst):
    if isinstance(lst, complex):
        return (lst.real, lst.imag)
    elif isinstance(lst, list):
        return [complex_to_tuple(element) for element in lst]
    else:
        return lst


def omit_complex(lst):
    if isinstance(lst, complex):
        return lst.real
    elif isinstance(lst, list):
        return [omit_complex(element) for element in lst]
    else:
        return lst


def dict_to_complex(d):
    if isinstance(d, dict):
        return complex(d["re"], d["im"])
    elif isinstance(d, list):
        return [dict_to_complex(element) for element in d]
    else:
        return d


# target always last in list
def dump_qc(qc: QuantumCircuit) -> list[dict]:
    return [{"name": gate.operation.name, "ind": [qb.index for qb in gate.qubits],
             "data": complex_to_tuple((qi.Operator(gate.operation).data).tolist())} for gate in qc]


def flatten(lst: list) -> list:
    return list(itertools.chain(*lst))


def assign_indices(dumped_qc):
    index_set = (i for i in range(1, 1000000))
    index = {}
    for gate in dumped_qc:
        new_indices = []
        for ind in gate["ind"]:
            old_index = index.get(ind)
            if old_index is None:
                old_index = index_set.__next__()
            new_index = index_set.__next__()
            new_indices.append(old_index)
            new_indices.append(new_index)
            index[ind] = new_index
        gate["ind"] = new_indices


if __name__ == '__main__':

    qc = QuantumCircuit(4)
    qr = QuantumRegister(1)
    cr = ClassicalRegister(1)

    matrix = [[0, 0, 0, 1],
              [0, 0, 1, 0],
              [1, 0, 0, 0],
              [0, 1, 0, 0]]
    #qc.h(0)
    #qc.x(0)
    #qc.cx(0, 1)
    #qc.cx(0, 2)
    #qc.ccx(0, 2, 1)
    # qc.unitary(matrix, [1, 2])

    U1 = qi.random_unitary(2)
    U2 = qi.random_unitary(2)
    U3 = qi.random_unitary(4)
    U4 = qi.random_unitary(4)

    qc.unitary(U2, [0])
    qc.cx(0, 1)
    qc.unitary(U3, [2, 3])
    qc.unitary(U1, [0, 1])
    qc.unitary(U4, [0, 2])
    print(qc)

    dumped_qc = dump_qc(qc)
    assign_indices(dumped_qc)

    for gate in dumped_qc:
        print(gate)

    print("\n")
    with open('./qc.json', "w") as f:
        json.dump(dumped_qc, f)

    args = [r"D:\Praktikum QC\Code\TensorTools\julia_files\run.bat", r"D:\Praktikum QC\Code\TensorTools\qc.json"]
    data = subprocess.check_output(args).decode("utf-8")
    print(data)


    with open('./qc_out.json', "r") as f:
        data = json.load(f)

    data = dict_to_complex(data)
    for d in data:
        U = np.array(d[0])
        V = np.array(d[1])


    print(np.array(data[1][0]).reshape(4, 4))
    qc2 = QuantumCircuit(4)
    qc2.unitary(np.array(data[1][0]).reshape(4, 4), [0, 1])
    print(qc2)