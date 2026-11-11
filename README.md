# Implementing AQFT using Adders
This is an implementation of AQFT circuit 1 from https://doi.org/10.1038/s41598-025-21087-2

## Installation
The easiest way to get the python notebook up and run is to install uv and then run the following command within the folder:
```bash
uv run approximate-aqft
```
This will install all dependencies for both the package and the notebook.

To run the tests run the following command:

```bash
uv run pytest -v
```

## Usage

The implemented AQFT circuit 1 is constructed as a gate using the function:

```python
from qiskit import QuantumCircuit, QuantumRegister
from approximate_aqft.aqft import aqft

number_of_qubits = 4
b = 3

x_register = QuantumRegister(number_of_qubits, name="x")
psi_register = QuantumRegister(b + 1, name="psi")
adder_ancilla_register = QuantumRegister(1, name="Adder Ancilla (carry 0)")
pgt_ancilla_register = QuantumRegister(1, name="PGT Ancilla")

aqft_circuit = QuantumCircuit(
    x_register, psi_register, adder_ancilla_register, pgt_ancilla_register
)
aqft_circuit.append(
    aqft(number_of_qubits, b),
    x_register[:]
    + psi_register[:]
    + adder_ancilla_register[:]
    + pgt_ancilla_register[:],
)
```

The `number_of_qubits` and `b` must be positive integers.