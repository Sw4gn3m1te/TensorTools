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

    """
    decorator, if used a new deepcopy is used for all function arguments (can be used if function arguments are mutable)

    """
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
    def get_p_mat(n: int) -> np.array:
        """
        returns a permutation matrix for conversion from/to qiskit

        :param n: number of qubits (size will be 2^n)
        """
        p_mat = np.zeros([2 ** n, 2 ** n])
        for i in range(2 ** n):
            bit = i
            revs_i = 0
            for j in range(n):
                if bit & 0b1:
                    revs_i += 1 << (n - j - 1)
                bit = bit >> 1
            p_mat[i, revs_i] = 1
        return p_mat

    @classmethod
    def get_from_qiskit_circuit(cls, qc: QuantumCircuit):
        """
        constructs TensorNetwork object from qiskit circuit

        """
        num_qubits = qc.num_qubits
        dumped_qc = TensorNetwork.dump_qiskit_circuit(qc)
        tensor_network = cls(num_qubits)
        last_elements = tensor_network.init_qubits
        # reverse because qcs must be analyzed backwards
        for gate in dumped_qc:
            node = tn.Node(TensorNetwork.pack(TensorNetwork.convert_from_qiskit_matrix(gate["data"])), name=gate["name"])
            for e, i in enumerate(gate["ind"]):
                edge = tn.connect(last_elements[i], node[e])
                tensor_network.edges.append(edge)
                last_elements[i] = node[e + len(gate["ind"])]
            tensor_network.nodes.append(node)
        tensor_network.out_edges = last_elements

        for node in tensor_network.init_states:
            tn.remove_node(node)

        return tensor_network

    @staticmethod
    def dump_qiskit_circuit(qc: qiskit.QuantumCircuit) -> list[dict]:

        """
        dumps qiskit circuit object

        :param qc: qiskit circuit
        :return: list(dict('name':str, 'data':np.array, 'ind':list(int)))
        """

        return [{"name": f"{gate.operation.name}-{e}", "ind": sorted([qb.index for qb in gate.qubits]),
                 "data": qi.Operator(gate.operation).data} for e, gate in enumerate(qc) if gate.operation.name != "measure"]

    @staticmethod
    def pack(mat: np.ndarray) -> np.array:
        """

        reshapes (nxn)-matrix into appropriate n-rank tensor
        """
        return mat.reshape(*(2,)*int(np.log2(mat.size)))

    @staticmethod
    def unpack(mat: np.ndarray) -> np.array:
        """

        reshapes n-rank tensor into appropriate (nxn)-matrix
        """
        n = int(math.sqrt(mat.size))
        return mat.reshape(n, n)

    @staticmethod
    def get_zero_qb(size: int):

        qb = np.zeros(2 ** size, dtype=complex)
        qb[0] = 1
        return qb

    @use_deep_copy
    def fully_contract(self, remove_inputs=False) -> tn.AbstractNode:
        """
        contracts the entire network into a single node (currently using auto contractor)

        :param remove_inputs: remove input qubits if connected to the network
        """
        if remove_inputs:
            for node in self.init_states:
                tn.remove_node(node)
        #res = tn.contractors.optimal(self.nodes, tn.get_all_dangling(self.nodes))
        all_nodes = self.edges + self.init_qubits
        res = tn.contractors.auto(self.nodes, tn.get_all_dangling(self.nodes))
        return res

    def set_init_qubits(self, init_qubits):
        self.init_qubits = init_qubits

    @staticmethod
    def convert_from_qiskit_matrix(mat: np.array) -> np.array:
        """
        transforms matrix from qiskit shape to TensorNetwork shape using permutation matrix

        """

        n = int(np.log2(mat.size)/2)
        p_mat = TensorNetwork.get_p_mat(n)
        mat = p_mat @ mat
        return mat

    @staticmethod
    def convert_to_qiskit_matrix(mat: np.array) -> np.array:
        """
        transforms matrix from Tensornetwork shape to qiskit shape using permutation matrix

        """
        n = int(np.log2(mat.size) / 2)
        # p_mat_inv = np.linalg.inv(TensorNetwork.get_p_mat(n))
        p_mat = TensorNetwork.get_p_mat(n)
        mat = p_mat.T @ mat
        return mat

    @use_deep_copy
    # self.out_edges is incorrect after application
    def partial_contract_by_name(self, node_name_1: str, node_name_2: str) -> tuple[TensorNetwork, str]:
        """
        finds the edge connecting two nodes and contracts it

        :param node_name_1:
        :param node_name_2:
        :return: new TensorNetwork, name of new node ('node_name_1+node_name_2')
        """
        node1, node2 = (next((node for node in self.init_states + self.nodes if node.name == node_name), None) for node_name in (node_name_1, node_name_2))
        assert node1 and node2
        new_node_name = f"{node_name_1}+{node_name_2}"
        res = tn.contract_between(node1, node2, name=new_node_name)
        self.nodes.remove(node1)
        self.nodes.remove(node2)
        self.nodes.append(res)
        return self, new_node_name

    def get_node_by_name(self, name: str) -> tn.AbstractNode:
        """
        getter for nodes by name

        :param name:
        """
        return next((obj for obj in self.init_states + self.nodes if obj.name == name), None)

    @use_deep_copy
    def perform_svd_on_node(self, name: str, max_singular_values=None, max_truncation_err=None) -> TensorNetwork:
        """
        performs a svd on a node given by name

        :param name:
        :param max_singular_values:
        :param max_truncation_err:
        :return: TensorNetwork with new split node
        """
        node = self.get_node_by_name(name)
        (u, s, v, truncated) = tn.split_node_full_svd(node, left_edges=node[:len(node.edges) // 2], right_edges=node[len(node.edges) // 2:],
                                                      max_singular_values=max_singular_values, max_truncation_err=max_truncation_err,
                                                      left_name=f"u-{name}", middle_name=f"s-{name}", right_name=f"v-{name}")
        self.nodes.remove(node)
        self.nodes += [u, s, v]
        return self

    def get_graph(self):
        tn.to_graphviz(self.nodes).render()

    @staticmethod
    def simulate_and_draw_result_comparison(qc1: qiskit.QuantumCircuit, qc2: qiskit.QuantumCircuit, shots=1000) -> None:
        """
        Simulates two qiskit.QuantumCircuit objects using aer_simulator and draws a comparison histogram

        :param qc1:
        :param qc2:
        :param shots:
        """
        simulator = Aer.get_backend("aer_simulator")
        list(map(QuantumCircuit.measure_all, [qc1, qc2]))
        transpile_with_simulator = partial(transpile, backend=simulator)
        transpiled = list(map(transpile_with_simulator, [qc1, qc2]))
        sim_with_args = partial(simulator.run, shots=shots, memory=True)
        sim_res = [run.result() for run in list(map(sim_with_args, transpiled))]
        data1, data2 = sorted(sim_res[0].get_memory(qc1)), sorted(sim_res[1].get_memory(qc2))
        TensorNetwork.draw_comparison_diagram(data1, data2)

    @staticmethod
    def draw_comparison_diagram(data1, data2):
        fig, (ax1, ax2) = mpl.subplots(2, 1, sharex=True)
        ax1.hist(data1)
        ax1.set_title("QC 1")
        ax2.hist(data2)
        ax2.set_title("QC 2")
        mpl.tight_layout()
        mpl.show()

    @staticmethod
    def qiskit_circuit_to_unitary(qc: qiskit.QuantumCircuit) -> np.array:
        """
        gets the unitary of a quantum circuit using qiskits unitary simulator

        :param qc:
        :return:
        """
        simulator = Aer.get_backend('unitary_simulator')
        job = simulator.run(transpile(qc, simulator))
        result = job.result()
        return result.get_unitary()

    @staticmethod
    def get_diff_tensor(t1, t2, threshold=1E-10):

        """
        returns a tensor t1-t2, entries are set to 0 if below threshold

        :param t1:
        :param t2:
        :param threshold:
        :return: diff tensor
        """

        diff = t1 - t2
        diff[np.abs(diff) < threshold] = 0
        return diff

    def contract_backwards(self) -> np.array:
        """
        uses TensorNetwork.partial_contract_by_name to contract the entire network starting from the back

        """

        new_name = self.nodes[-1].name
        t = self
        for node in reversed(self.nodes[:-1]):
            t, new_name = t.partial_contract_by_name(new_name, node.name)
        t = t.unpack(t.get_node_by_name(new_name).tensor)
        return t


if __name__ == '__main__':

    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)  # 2 3 0 1

    print(qc)

    t1 = TensorNetwork.qiskit_circuit_to_unitary(qc).data
    dumped_qc = TensorNetwork.dump_qiskit_circuit(qc)

    tensor_network = TensorNetwork.get_from_qiskit_circuit(qc)
    #t2 = tensor_network.convert_to_qiskit_matrix(tensor_network.unpack(tensor_network.fully_contract(remove_inputs=False).tensor))
    t2 = tensor_network.contract_backwards()
    qc2 = QuantumCircuit(2)
    qc2.unitary(t2, [0, 1])

    TensorNetwork.simulate_and_draw_result_comparison(qc, qc2, shots=10000)



    print("\n", t1, "\n")
    print(t2, "\n")
    diff = TensorNetwork.get_diff_tensor(t1, t2)
    zero_matrix = np.zeros((4, 4), dtype=int)
    if np.allclose(zero_matrix, diff):
        print("True")
    else:
        print("False")


    #
    # t2 = tensornetwork.fix_shape2(tensornetwork.get_tensor_of_backwards_contracted_gates())
    # #t2 = tensornetwork.translate_qiskit_data(t2, 2)
    # print(t1)
    # print("\n")
    # print(t2)


    #t2 = t2.fix_shape2(t2.get_node_by_name("cx-0").tensor)
    #print(t2.convert_from_qiskit_matrix(t1))
    #d_edges = list(tn.get_subgraph_dangling(t2.nodes))
    #d_edges = list(reversed(sorted(d_edges, key=lambda e: e.axis1)))
    #d_edges = list(sorted(d_edges, key=lambda e: e.axis1))
    #d_edges = [d_edges[3], d_edges[2], d_edges[1], d_edges[0]]
    #print(d_edges)
    #c1 = tn.contractors.optimal(t2.nodes, d_edges)
    #t2 = t2.fix_shape2(c1.tensor)

    #t2 = TensorNetwork.convert_to_qiskit_matrix(t2)
    #t2 = t2.fully_contract(remove_inputs=True)
    #t2 = TensorNetwork.fix_shape2(t2.tensor)
    #new_name = t2.nodes[-1].name
    #for node in reversed(t2.nodes[:-1]):
    #    t2, new_name = t2.partial_contract_by_name(new_name, node.name)
    #t2 = t2.fix_shape2(t2.get_node_by_name(new_name).tensor)

    #t2 = tensornetwork.partial_contract_by_name("unitary-1", "h-0")
    #t2 = tensornetwork.fix_shape2(t2.get_node_by_name("unitary-1+h-0").tensor)
    #t2 = tensornetwork.fix_shape2(t2.partial_contract_by_name("unitary-2+cx-1", "h-0").get_node_by_name("unitary-2+cx-1+h-0").tensor)
    #qc2 = QuantumCircuit(2)
    #qc2.unitary(t2, [0, 1])

    #TensorNetwork.simulate_and_draw_result_comparison(qc, qc2, shots=10000)
    #
    # print("\n", t1, "\n")
    # print(t2, "\n")
    # diff = TensorNetwork.get_diff_tensor(t1, t2)
    # zero_matrix = np.zeros((4, 4), dtype=int)
    # if np.allclose(zero_matrix, diff):
    #     print("True")
    # else:
    #     print("False")
