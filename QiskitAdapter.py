import numpy as np
import math


class QiskitAdapter:

    def __init__(self, qc):
        self.qc = qc

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

    @staticmethod
    def pack(mat: np.ndarray) -> np.array:
        """

        reshapes (nxn)-matrix into appropriate n-rank tensor
        """
        return mat.reshape(*(2,) * int(np.log2(mat.size)))

    @staticmethod
    def unpack(mat: np.ndarray) -> np.array:
        """

        reshapes n-rank tensor into appropriate (nxn)-matrix
        """
        n = int(math.sqrt(mat.size))
        return mat.reshape(n, n)

    @staticmethod
    def convert_from_qiskit_matrix(mat: np.array) -> np.array:
        """
        transforms matrix from qiskit shape to TensorNetwork shape using permutation matrix

        """

        n = int(np.log2(mat.size) / 2)
        p_mat = QiskitAdapter.get_p_mat(n)
        mat = p_mat @ mat @ p_mat
        return mat

    @staticmethod
    def convert_to_qiskit_matrix(mat: np.array) -> np.array:
        """
        transforms matrix from Tensornetwork shape to qiskit shape using permutation matrix

        """
        n = int(np.log2(mat.size) / 2)
        # p_mat_inv = np.linalg.inv(TensorNetwork.get_p_mat(n))
        p_mat = QiskitAdapter.get_p_mat(n)
        mat = p_mat.T @ mat @ p_mat.T
        return mat

    @staticmethod
    def align(mat: np.array, inds: list):

        """
        transposes mat using inds because qiskits wierd representation

        """
        mask = inds + [e + len(inds) for e in inds]
        t = [(e, i) for e, i in zip([_ for _ in reversed(range(len(mask)))], mask)]
        t = sorted(t, key=lambda x: x[1])
        t = [e[0] for e in reversed(t)]
        mat = mat.transpose(*t)
        return mat

    def dump(self) -> dict:
        """
        dumps datastructure to dict


        :return: data = {"num_qb": int, "data": list[dict["name": str, "inds": list[int], "mat": np.array]]}
        """
        data = [
            {"name": f"{gate.operation.name}-{e}",
             "inds": [qb.index for qb in gate.qubits],
             "mat": gate.operation.to_matrix()
             } for e, gate in enumerate(self.qc) if gate.operation.name != "measure"]

        return {"num_qb": self.qc.num_qubits, "data": data}
