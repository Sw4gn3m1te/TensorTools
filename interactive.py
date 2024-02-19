import numpy as np

from TensorNetwork import TensorNetwork
from QiskitAdapter import QiskitAdapter
from qiskit import QuantumCircuit, Aer, transpile
import qiskit.quantum_info as qi
import matplotlib.pyplot as mpl

simulator = Aer.get_backend("aer_simulator")

# define quantum circuit here ###############
nqb = 2
qc = QuantumCircuit(nqb)

u = qi.random_unitary(4, seed=420)

# qc.cx(2, 0)
# qc.h(2)
# qc.ccx(0, 2, 1)
# qc.unitary(u, [1, 2])
# qc.cx(1, 0)

qc.h(0)
qc.unitary(np.eye(4), [0, 1])
qc.cx(0, 1)

############################################

qc_reset = qc.copy()

print(qc, "\n")
t_n = TensorNetwork(adapter=QiskitAdapter(qc))
qc = t_n.to_qiskit_circuit()
while True:
    print(qc, "\n")

    op = input("simulate (s) | contract (c) | collapse (f) | enlarge gates (l)|\n"
               "svd-decompose (d) | get tensor (g) | reset (r) | exit(e)\n")
    if op == "e":
        exit(1)

    elif op == "c":
        g1 = input("select first gate for contraction by name\n")
        g2 = input("select second gate for contraction by name\n")

        res = t_n.partial_contract_by_name(g1, g2)
        if res is None:
            continue
        else:
            t_n, new_node_name = res
        qc = t_n.to_qiskit_circuit()

    elif op == "d":
        name = input("select node to perform svd on by name\n")
        t_n = t_n.perform_svd_on_node(name)
        qc = t_n.to_qiskit_circuit()

    elif op == "f":
        t_n, new_node_name = t_n.contract_backwards()
        qc = t_n.to_qiskit_circuit()

    elif op == "s":
        num_shots = input("choose number of shots for simulation\n")
        qc_t = t_n.to_qiskit_circuit()
        qc_t.measure_all()
        qc_t = transpile(qc_t, simulator)
        print(qc_t)
        job = simulator.run(qc_t, shots=num_shots, memory=True)
        data = sorted(job.result().get_memory(qc_t))
        fig, ax = mpl.subplots(1, 1)
        ax.hist(data)
        mpl.tight_layout()
        mpl.show()

    elif op == "g":
        name = input("select node by name\n")
        node = t_n.get_node_by_name(name)
        print(t_n.adapter.unpack(node.tensor.T), "\n")

    elif op == "l":
        t_n = t_n.enlarge_gates_with_id()
        qc = t_n.to_qiskit_circuit()

    elif op == "r":
        qc = qc_reset.copy()
        t_n = TensorNetwork(adapter=QiskitAdapter(qc))
        qc = t_n.to_qiskit_circuit()
        print("resetting...\n")

    else:
        print("no valid option\n")
        continue





