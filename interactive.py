from TensorNetwork import TensorNetwork
from QiskitAdapter import QiskitAdapter
from qiskit import QuantumCircuit, Aer, transpile
import qiskit.quantum_info as qi
import matplotlib.pyplot as mpl

simulator = Aer.get_backend("aer_simulator")

# define quantum circuit here
nqb = 3
qc = QuantumCircuit(nqb)

u = qi.random_unitary(4, seed=420)

qc.cx(2, 0)
qc.h(2)
qc.ccx(0, 2, 1)
#qc.unitary(u, [1, 2])
qc.cx(1, 0)

print(qc, "\n")
t_n = TensorNetwork(nqb, adapter=QiskitAdapter(qc))
qc = t_n.to_qiskit_circuit()
while True:
    print(qc, "\n")

    op = input("simulate (s) | contract (c) | collapse (f) | exit(e)\n")
    if op == "e":
        exit(1)
    if op == "c":
        g1 = input("select first gate for contraction by name\n")
        g2 = input("select second gate for contraction by name\n")

        res = t_n.partial_contract_by_name(g1, g2)
        if res is None:
            continue
        else:
            t_n, new_node_name = res
        qc = t_n.to_qiskit_circuit()

    if op == "f":
        t_n, new_node_name = t_n.contract_backwards()
        qc = t_n.to_qiskit_circuit()

    if op == "s":
        qc_t = t_n.to_qiskit_circuit()
        qc_t.measure_all()
        qc_t = transpile(qc_t, simulator)
        print(qc_t)
        job = simulator.run(qc_t, shots=1000, memory=True)
        data = sorted(job.result().get_memory(qc_t))
        fig, ax = mpl.subplots(1, 1)
        ax.hist(data)
        mpl.tight_layout()
        mpl.show()





