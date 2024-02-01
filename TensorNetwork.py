from __future__ import annotations

import functools

import qiskit
import tensornetwork as tn
import qiskit.quantum_info as qi
from qiskit import QuantumCircuit
import numpy as np
import copy
from functools import wraps, partial
from itertools import permutations
from qiskit import Aer, transpile, execute
import matplotlib.pyplot as mpl
import math
from QiskitAdapter import QiskitAdapter

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

    def __init__(self, num_qubits: int, init_qubits=None, adapter: QiskitAdapter = None):
        self.nodes = []
        self.edges = []
        self.num_qubits = num_qubits
        self.init_states = [tn.Node(TensorNetwork.get_zero_qb(1), name=f"IQb_{i}") for i in range(num_qubits)]
        if not init_qubits:
            self.init_qubits = [istate[0] for istate in self.init_states]
        else:
            self.init_qubits = init_qubits
        self.out_edges = []
        self.adapter = adapter
        self.populate_with_data()

    def populate_with_data(self):
        """
        fills TensorNetwork with data by using adapter

        """
        dump = self.adapter.dump()
        self.num_qubits = dump["num_qb"]
        last_elements = self.init_qubits
        for gate in dump["data"]:
            m = gate["mat"]
            inds = gate["inds"]
            m = self.adapter.pack(m)
            m = self.adapter.align(m, inds)
            m = self.adapter.unpack(m)
            m = self.adapter.convert_from_qiskit_matrix(m)
            m = self.adapter.pack(m)
            node = tn.Node(m, name=gate["name"])
            for e, i in enumerate(inds):
                edge = tn.connect(last_elements[i], node[e])
                self.edges.append(edge)
                last_elements[i] = node[e + len(inds)]
            self.nodes.append(node)
        self.out_edges = last_elements

        for node in self.init_states:
            tn.remove_node(node)

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