# Tensor Tools

### interactive.py
- setup your qiskit circuit in the 'define quantum circuit here section'
- run the script
- select an option

| option | function      | description                                                           |
|--------|---------------|-----------------------------------------------------------------------|
| s      | simulation    | simulates the quantum circuit using AER with 100 shots                |
| c      | contract      | contracts two nodes by name, the nodes must be connected and adjacent |
| f      | collapse      | fully contracts the circuit into a single node                        |
| l      | enlarge       | enlarges each tensor to cover all qubits by contraction with id gate  |
| d      | svd decompose | decomposes a node using svd, new nodes u,s,d are created              |
| g      | get tensor    | prints the tensor of a node                                           |
| r      | reset         | reset network to initial state                                        |
| e      | exit          | exit script                                                           | 
