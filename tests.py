import numpy as np
from TensorNetwork import TensorNetwork
from QiskitAdapter import QiskitAdapter
from qiskit import QuantumCircuit
import qiskit.quantum_info as qi
import unittest


def compare_res(r1, r2, nqb):
    diff = TensorNetwork.get_diff_tensor(r1, r2)
    zero_matrix = np.zeros((2 ** nqb, 2 ** nqb), dtype=int)
    if np.allclose(zero_matrix, diff):
        return None
    else:
        return diff


def process_result(t, t_n):
    t = t.T
    t = t_n.adapter.unpack(t)
    return t


def test_full_contract_single_2qb_gate():
    qc1 = QuantumCircuit(2)
    qc2 = QuantumCircuit(2)

    u = qi.random_unitary(4, seed=1337)
    qc1.unitary(u, [0, 1])
    qc2.cx(0, 1)

    qr1 = TensorNetwork.qiskit_circuit_to_unitary(qc1).data
    qr2 = TensorNetwork.qiskit_circuit_to_unitary(qc2).data

    tn1 = TensorNetwork(2, adapter=QiskitAdapter(qc1))
    tn2 = TensorNetwork(2, adapter=QiskitAdapter(qc2))

    tnr1 = tn1.fully_contract().tensor
    tnr2 = tn2.fully_contract().tensor

    tnr1 = process_result(tnr1, tn1)
    tnr2 = process_result(tnr2, tn2)

    tr1 = compare_res(tnr1, qr1, 2)
    tr2 = compare_res(tnr2, qr2, 2)

    if tr1 is None and tr2 is None:
        return None
    else:
        return tr1, tr2


def test_contraction_bell():

    qc1 = QuantumCircuit(2)
    qc1.h(0)
    qc1.cx(0, 1)

    qr1 = TensorNetwork.qiskit_circuit_to_unitary(qc1).data
    tn1 = TensorNetwork(2, adapter=QiskitAdapter(qc1))
    tnr1 = tn1.fully_contract().tensor
    tnr1 = process_result(tnr1, tn1)
    tr1 = compare_res(tnr1, qr1, 2)
    if tr1 is None:
        return None
    else:
        return tr1


def test_align_gate():
    qc1 = QuantumCircuit(2)
    u = qi.random_unitary(4, seed=1337)
    qc1.unitary(u, [1, 0])
    qr1 = TensorNetwork.qiskit_circuit_to_unitary(qc1).data
    tn1 = TensorNetwork(2, adapter=QiskitAdapter(qc1))
    tnr1 = tn1.get_node_by_name("unitary-0").tensor
    tnr1 = process_result(tnr1, tn1)
    tr1 = compare_res(tnr1, qr1, 2)
    if tr1 is None:
        return None
    else:
        return tr1


def run_all_test():

    tests = [test_full_contract_single_2qb_gate,
             test_contraction_bell,
             test_align_gate]

    for test in tests:
        res = test()
        if res is not None:
            print(f"{test.__name__} failed with \n{res}")
        else:
            print(f"{test.__name__} passed")


if __name__ == '__main__':
    run_all_test()