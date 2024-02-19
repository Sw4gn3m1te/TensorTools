from TensorNetwork import TensorNetwork
from QiskitAdapter import QiskitAdapter
from qiskit import QuantumCircuit

nqb = 1

qc = QuantumCircuit(nqb)
qc.h(0)
qc.x(0)

tn = TensorNetwork(adapter=QiskitAdapter(qc))