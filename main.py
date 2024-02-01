from QiskitAdapter import QiskitAdapter
from TensorNetwork import TensorNetwork
import numpy as np
from qiskit import QuantumCircuit
import qiskit.quantum_info as qi

import warnings
warnings.filterwarnings("ignore")
np.set_printoptions(linewidth=np.inf)


nqb = 3
qc = QuantumCircuit(nqb)
qc.unitary(qi.random_unitary(2**nqb), [2, 0, 1])
#qc.ccx(0, 2, 1)
print(qc)
n = 2*nqb
t1 = TensorNetwork.qiskit_circuit_to_unitary(qc).data
t_n = TensorNetwork(nqb, adapter=QiskitAdapter(qc))
u = t_n.get_node_by_name("unitary-0").tensor
t2 = t_n.adapter.unpack(u)
t2 = t_n.adapter.convert_to_qiskit_matrix(t2)

#ccx = t_n.get_node_by_name("ccx-1")


# print("\n", t1, "\n")
# print(t2, "\n")
diff = TensorNetwork.get_diff_tensor(t1, t2)
zero_matrix = np.zeros((2**nqb, 2**nqb), dtype=int)
if np.allclose(zero_matrix, diff):
    print("True")
else:
    print("False")
    print(diff)

qc2 = QuantumCircuit(3)
qc2.unitary(t2, [0, 1, 2])
TensorNetwork.simulate_and_draw_result_comparison(qc, qc2, shots=10000)


