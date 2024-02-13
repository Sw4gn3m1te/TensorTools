from QiskitAdapter import QiskitAdapter
from TensorNetwork import TensorNetwork
import numpy as np
from qiskit import QuantumCircuit
import qiskit.quantum_info as qi
from math import sqrt

import warnings
warnings.filterwarnings("ignore")
np.set_printoptions(linewidth=np.inf)


nqb = 3
qc = QuantumCircuit(nqb)

#qc.cx(0, 1)
#qc.h(1)

u = qi.random_unitary(2**nqb, seed=420)

u1 = qi.random_unitary(2, seed=421)
u2 = qi.random_unitary(2, seed=422)
u3 = qi.random_unitary(4, seed=423)
u4 = qi.random_unitary(4, seed=424)
u5 = qi.random_unitary(8, seed=425)

#uh = np.array([[1/np.sqrt(2), -1/np.sqrt(2)], [1/np.sqrt(2), 1/np.sqrt(2)]])
ut = np.arange(1, 17)
ut = ut.reshape(4, 4)
ut2 = np.arange(17, 33)
ut2 = ut2.reshape(4, 4)
ut3 = np.array(
    [[1, 0, 0, 0],
     [0, 0, 0, 0],
     [0, 0, 0, 0],
     [0, 0, 0, 0]])


qc.cx(2, 0)
qc.h(2)
qc.ccx(0, 2, 1)
qc.cx(1, 0)
#qc.h(1)
#qc.unitary(np.eye(4), [0, 1])
#qc.unitary(np.array([[1, 2], [3, 4]]), 0)
#qc.unitary(ut, [0, 1])
#qc.unitary(ut2, [0, 1])
#qc.cx(0, 1)
#qc.h(1)
#qc.ccx(0, 2, 1)



print(qc)
n = 2*nqb
t1 = TensorNetwork.qiskit_circuit_to_unitary(qc).data
t_n = TensorNetwork(nqb, adapter=QiskitAdapter(qc))
t_n.get_graph()

# for node in t_n.nodes:
#    print(t_n.adapter.unpack(node.tensor))

print("\n")
#t2 = t_n.get_node_by_name("unitary-0").tensor
#
# res = np.tensordot(*[node.tensor.T for node in t_n.nodes], axes=[1, 0]).T
# print(t_n.adapter.unpack(res), "EASPORTS")

#t2 = t_n.fully_contract().tensor
t_n, name = t_n.contract_backwards()
#t_n, name = t_n.partial_contract_by_name("h-1", "ccx-2")
t2 = t_n.get_node_by_name(name).tensor

#t_n, name = t_n.partial_contract_by_name("cx-0", "unitary-1")
#t2 = t_n.get_node_by_name(name).tensor
t2 = t_n.adapter.unpack(t2.T)
#t2 = t_n.adapter.convert_to_qiskit_matrix(t2)



print("\n", t1, "\n")
print(t2, "\n")
diff = TensorNetwork.get_diff_tensor(t1, t2)
zero_matrix = np.zeros((2**nqb, 2**nqb), dtype=int)
if np.allclose(zero_matrix, diff):
    print("True")
else:
    pass
    print("False")
    print(diff)


qc2 = QuantumCircuit(nqb)
qc2.unitary(t2, [_ for _ in range(nqb)])
TensorNetwork.simulate_and_draw_result_comparison(qc, qc2, shots=10000)


