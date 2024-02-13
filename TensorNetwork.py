from __future__ import annotations

import qiskit
import tensornetwork as tn

from qiskit import QuantumCircuit
import numpy as np
import copy
from typing import Optional
from functools import wraps, partial
from qiskit import Aer, transpile
import matplotlib.pyplot as mpl
from QiskitAdapter import QiskitAdapter

import warnings
warnings.filterwarnings("ignore")
np.set_printoptions(linewidth=np.inf)


colors = {"red": "\u001b[38;5;1m", "orange": "\u001b[38;5;208m", "yellow": "\u001b[38;5;11m", "green": "\u001b[38;5;10m", "blue": "\u001b[38;5;12m", "default": "\u001b[0;39m"}

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
        self.in_edges = []
        self.gate_locations = {}
        self.adapter = adapter
        self.populate_with_data()

    def populate_with_data(self):
        """
        fills TensorNetwork with data by using adapter

        """
        dump = self.adapter.dump()
        self.num_qubits = dump["num_qb"]
        last_elements = self.init_qubits
        last_elements = [None] * self.num_qubits
        for gate in dump["data"]:
            m = gate["mat"]
            inds = gate["inds"]
            m = self.adapter.pack(m)
            m = self.adapter.align(m, inds)
            m = m.T
            m = self.adapter.unpack(m)

            m = self.adapter.pack(m)
            inds = list(sorted(inds))
            self.gate_locations.update({gate["name"]: set(inds)})
            node = tn.Node(m, name=gate["name"])
            for en, ind in enumerate(inds):
                if last_elements[ind] is None:
                    last_elements[ind] = node.get_edge(en + len(inds))
                    self.in_edges.append((ind, node.get_edge(en)))
                    continue

                edge = tn.connect(last_elements[ind], node[en])
                self.edges.append(edge)
                last_elements[ind] = node[en + len(inds)]
            self.nodes.append(node)

        self.in_edges = [item[1] for item in sorted(self.in_edges, key=lambda x: x[0])]
        self.out_edges = last_elements

    @staticmethod
    def get_zero_qb(size: int):

        qb = np.zeros(2 ** size, dtype=complex)
        qb[0] = 1
        return qb

    @use_deep_copy
    def get_fully_contracted_network_as_node(self) -> tn.AbstractNode:
        """
        contracts the entire network into a single node (currently using auto contractor)

        """
        oeo = self.in_edges + self.out_edges
        res = tn.contractors.greedy(self.nodes, output_edge_order=oeo)
        return res

    @use_deep_copy
    # self.out_edges is incorrect after application
    def partial_contract_by_name(self, node_name_1: str, node_name_2: str) -> Optional[tuple[TensorNetwork, str]]:
        """
        finds the edge connecting two nodes and contracts it

        :param node_name_1:
        :param node_name_2:
        :return: new TensorNetwork, name of new node ('node_name_1+node_name_2')
        """
        node1, node2 = (next((node for node in self.init_states + self.nodes if node.name == node_name), None) for node_name in (node_name_1, node_name_2))
        assert node1 and node2
        new_node_name = f"{node_name_1}+{node_name_2}"

        node1_ind = self.nodes.index(node1)
        node2_ind = self.nodes.index(node2)
        if node2_ind < node1_ind:
            node1, node2 = node2, node1

        if abs(node1_ind - node2_ind) != 1:
            print("nodes are not neighbours\n")
            return

        if len(self.gate_locations.get(node1.name).intersection(self.gate_locations.get(node2.name))) == 0:
            print("nodes are not connected\n")
            return

        node1_edges = node1.get_all_edges()
        node2_edges = node2.get_all_edges()
        node1_inds = self.gate_locations.pop(node1.name)
        node2_inds = self.gate_locations.pop(node2.name)

        subgraph_in_edges = [(i, edge) for (i, edge) in zip(sorted(list(node1_inds)), node1_edges[:len(node1_edges) // 2])]
        subgraph_in_edges += [(i, node2_edges[:len(node2_edges) // 2][i]) for i in sorted(list(node2_inds.difference(node1_inds)))]
        subgraph_out_edges = [(i + len(subgraph_in_edges), edge) for (i, edge) in zip(sorted(list(node2_inds)), node2_edges[len(node2_edges) // 2:])]
        subgraph_out_edges += [(i + len(subgraph_in_edges), node1_edges[len(node1_edges) // 2:][i]) for i in sorted(list(node1_inds.difference(node2_inds)))]

        oeo = [e[1] for e in sorted(subgraph_in_edges + subgraph_out_edges, key=lambda x: x[0])]
        new_node = tn.contract_between(node1, node2, name=new_node_name, output_edge_order=oeo)
        i = self.nodes.index(node1)
        self.gate_locations.update({new_node_name: node1_inds.union(node2_inds)})
        self.nodes.remove(node1)
        self.nodes.remove(node2)
        self.nodes.insert(i, new_node)

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

    @use_deep_copy
    def contract_backwards(self) -> tuple[TensorNetwork, str]:
        """
        uses TensorNetwork.partial_contract_by_name to contract the entire network starting from the back

        """

        new_name = self.nodes[-1].name
        t = self
        for node in reversed(self.nodes[:-1]):
            t, new_name = t.partial_contract_by_name(new_name, node.name)
        return t, new_name

    def to_qiskit_circuit(self, color: str = None) -> QuantumCircuit:
        qc = QuantumCircuit(self.num_qubits)
        for node in self.nodes:
            if color in ["red", "orange", "yellow", "green", "blue"]:
                label = color_text(node.name, color)
            else:
                label = node.name
            qc.unitary(self.adapter.unpack(node.tensor.T), self.gate_locations.get(node.name), label=label)
        return qc


def color_text(s: str, color) -> str:
    return colors.get(color) + s + colors.get("default")