from __future__ import annotations
import qiskit
import tensornetwork as tn
import qiskit.quantum_info as qi
from qiskit import QuantumCircuit
import numpy as np
import copy
from functools import wraps, partial
from qiskit import Aer, transpile
import matplotlib.pyplot as mpl
import math


import warnings
warnings.filterwarnings("ignore")
np.set_printoptions(linewidth=np.inf)


def use_deep_copy(func):

    @wraps(func)
    def wrapper(*args, **kwargs):
        args_copy = [copy.deepcopy(arg) for arg in args]
        kwargs_copy = {k: copy.deepcopy(v) for k, v in kwargs.items()}
        return func(*args_copy, **kwargs_copy)
    return wrapper


class TensorNetwork:

    def __init__(self, num_qubits: int, init_qubits=None):
        self.nodes = []
        self.edges = []
        self.num_qubits = num_qubits
        self.init_states = [tn.Node(TensorNetwork.get_zero_qb(1), name=f"IQb_{i}") for i in range(num_qubits)]
        if not init_qubits:
            self.init_qubits = [istate[0] for istate in self.init_states]
        else:
            self.init_qubits = init_qubits
        self.out_edges = []

    @staticmethod
    def dump_qiskit_circuit(qc):
        return [{"name": f"{gate.operation.name}-{e}", "ind": [qb.index for qb in gate.qubits],
                 "data": qi.Operator(gate.operation).data} for e, gate in enumerate(qc) if gate.operation.name != "measure"]

    @staticmethod
    def fix_shape(mat: np.ndarray):
        return mat.reshape(*(2,)*int(np.log2(mat.size)), order="F")

    @staticmethod
    def fix_shape2(mat: np.ndarray):
        n = int(math.sqrt(mat.size))
        return mat.reshape(n, n)

    @classmethod
    def get_from_qiskit_circuit(cls, qc: QuantumCircuit):
        num_qubits = qc.num_qubits
        dumped_qc = TensorNetwork.dump_qiskit_circuit(qc)
        tensor_network = cls(num_qubits)
        last_elements = tensor_network.init_qubits.copy()
        for gate in dumped_qc:
            node = tn.Node(TensorNetwork.fix_shape(gate["data"]), name=gate["name"])
            for e, i in enumerate(gate["ind"]):
                edge = tn.connect(last_elements[i], node[e])
                last_elements[i] = node[e + len(gate["ind"])]
            tensor_network.nodes.append(node)
        tensor_network.out_edges = last_elements

        return tensor_network

    @staticmethod
    def get_zero_qb(size: int):
        qb = np.zeros(2 ** size, dtype=complex)
        qb[0] = 1
        return qb

    @use_deep_copy
    def fully_contract(self):
        all_nodes = self.init_states + self.nodes
        tn.contractors.optimal(all_nodes, self.out_edges)
        return self

    @use_deep_copy
    def get_tensor_of_backwards_contracted_gates(self):
        for node in self.init_states:
            tn.remove_node(node)
        last = self.nodes[-1]
        # does this work every time ? only when ignore edge order = True ?
        res = tn.contractors.optimal(self.nodes, tn.get_all_dangling(self.nodes), ignore_edge_order=True)
        return res.tensor
        #for node in reversed(self.nodes[:-1]):
        #    last = last @ node
        #self.nodes = [last]
        #return self.nodes[0].tensor

    def set_init_qubits(self, init_qubits):
        self.init_qubits = init_qubits

    @use_deep_copy
    def partial_contract_by_name(self, node_name_1: str, node_name_2: str) -> TensorNetwork:
        # partial_contract_by_name is not commutative in its arguments ?
        node1, node2 = (next((node for node in self.init_states + self.nodes if node.name == node_name), None) for node_name in (node_name_1, node_name_2))
        assert node1 and node2
        res = tn.contract_between(node1, node2, name=f"{node_name_1}+{node_name_2}")
        self.nodes.remove(node1)
        self.nodes.remove(node2)
        self.nodes.append(res)
        return self

    def get_node_by_name(self, name: str):
        return next((obj for obj in self.init_states + self.nodes if obj.name == name), None)

    def calculate(self, input_qubits):
        pass

    @use_deep_copy
    def perform_svd_on_node(self, name: str, max_singular_values=None, max_truncation_err=None) -> TensorNetwork:
        node = self.get_node_by_name(name)
        (u, s, v, truncated) = tn.split_node_full_svd(node, left_edges=node[:len(node.edges) // 2], right_edges=node[len(node.edges) // 2:],
                                                      max_singular_values=max_singular_values, max_truncation_err=max_truncation_err,
                                                      left_name=f"u-{name}", middle_name=f"s-{name}", right_name=f"v-{name}")
        self.nodes.remove(node)
        self.nodes += [u, s, v]
        return self

    def get_graph(self):
        tn.to_graphviz(self.nodes).render()


def simulate_and_draw_result_comparison(qc1, qc2, shots=1000):
    simulator = Aer.get_backend("aer_simulator")
    list(map(QuantumCircuit.measure_all, [qc1, qc2]))
    transpile_with_simulator = partial(transpile, backend=simulator)
    transpiled = list(map(transpile_with_simulator, [qc1, qc2]))
    sim_with_args = partial(simulator.run, shots=shots, memory=True)
    sim_res = [run.result() for run in list(map(sim_with_args, transpiled))]
    data1, data2 = sorted(sim_res[0].get_memory(qc1)), sorted(sim_res[1].get_memory(qc2))
    draw_comparison_diagram(data1, data2)


def draw_comparison_diagram(data1, data2):
    fig, (ax1, ax2) = mpl.subplots(2, 1, sharex=True)
    ax1.hist(data1)
    ax1.set_title("QC 1")
    ax2.hist(data2)
    ax2.set_title("QC 2")
    mpl.tight_layout()
    mpl.show()


def qiskit_circuit_to_unitary(qc):
    simulator = Aer.get_backend('unitary_simulator')
    job = simulator.run(transpile(qc, simulator))
    result = job.result()
    return result.get_unitary()


if __name__ == '__main__':

    U1 = qi.random_unitary(2)
    U2 = qi.random_unitary(2)
    U3 = qi.random_unitary(4)
    U4 = qi.random_unitary(4)

    qc = QuantumCircuit(3)
    qc.h(0)
    qc.cx(0, 1)
    qc.ccx(0, 2, 1)
    #qc.unitary(U2, [0])
    #qc.unitary(U3, [0, 1])
    #qc.unitary(U1, [1])
    #qc.unitary(U4, [0, 2])
    print(qc)
    print(qiskit_circuit_to_unitary(qc))

    tensornetwork = TensorNetwork.get_from_qiskit_circuit(qc)
    t = TensorNetwork.fix_shape2(tensornetwork.get_tensor_of_backwards_contracted_gates())
    print(t)
    qc2 = QuantumCircuit(3)
    qc2.unitary(t, [0, 1, 2])

    simulate_and_draw_result_comparison(qc, qc2, shots=1000)



