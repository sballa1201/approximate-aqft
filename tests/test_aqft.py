import numpy as np
import pytest
from qiskit.quantum_info import Operator, Statevector
from qiskit.synthesis.qft import synth_qft_full

from approximate_qft.aqft import *

# These tolerances are used since there should be no error in the tests below due to the low number of qubits.
ATOL = 1e-10
RTOL = 0


@pytest.mark.parametrize(
    "number_of_qubits, initial_state, expected_state",
    [
        # Standard examples
        (
            3,
            Statevector.from_label("001" + "000" + "0"),
            Statevector.from_label("001" + "001" + "0"),
        ),
        (
            3,
            Statevector.from_label("000" + "001" + "0"),
            Statevector.from_label("000" + "001" + "0"),
        ),
        (
            3,
            Statevector.from_label("100" + "011" + "0"),
            Statevector.from_label("100" + "111" + "0"),
        ),
        (
            3,
            Statevector.from_label("101" + "001" + "0"),
            Statevector.from_label("101" + "110" + "0"),
        ),
        # Example with overflow.
        (
            3,
            Statevector.from_label("111" + "001" + "0"),
            Statevector.from_label("111" + "000" + "0"),
        ),
        (
            2,
            Statevector.from_label("11" + "01" + "0"),
            Statevector.from_label("11" + "00" + "0"),
        ),
    ],
)
def test_curraco_ripple_carry_adder_modulo(
    number_of_qubits, initial_state, expected_state
):

    # ARRANGE
    cra_circuit = curraco_ripple_carry_adder_modulo(number_of_qubits)

    # ACT
    evolved_state = initial_state.evolve(cra_circuit)

    # Assert
    np.testing.assert_allclose(evolved_state, expected_state, rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("number_of_qubits", [1, 2, 3, 4, 5, 6, 7, 8])
def test_state_preparation(number_of_qubits):
    # ARRANGE
    state_preparation_circuit = state_preparation(number_of_qubits)

    # ACT
    initial_state = Statevector.from_label("0" * number_of_qubits)
    evolved_state = initial_state.evolve(state_preparation_circuit)

    # ASSERT
    expected_state = Statevector.from_int(0, 2**number_of_qubits)
    for k in range(1, 2**number_of_qubits):
        expected_state = expected_state + np.exp(
            2 * np.pi * k * 1j / 2**number_of_qubits
        ) * Statevector.from_int(k, 2**number_of_qubits)

    expected_state = expected_state / 2 ** (number_of_qubits / 2)

    np.testing.assert_allclose(evolved_state, expected_state, rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("number_of_qubits", [1, 2, 3, 4, 5, 6, 7, 8])
def test_state_uncompute(number_of_qubits):
    # ARRANGE
    state_uncompututation_circuit = state_uncompute(number_of_qubits)

    # ACT
    initial_state = Statevector.from_int(0, 2**number_of_qubits)
    for k in range(1, 2**number_of_qubits):
        initial_state = initial_state + np.exp(
            2 * np.pi * k * 1j / 2**number_of_qubits
        ) * Statevector.from_int(k, 2**number_of_qubits)

    initial_state = initial_state / 2 ** (number_of_qubits / 2)
    evolved_state = initial_state.evolve(state_uncompututation_circuit)

    # ASSERT
    expected_state = Statevector.from_int(0, 2**number_of_qubits)

    np.testing.assert_allclose(evolved_state, expected_state, rtol=RTOL, atol=ATOL)


# This test is adapted from lab 2.
@pytest.mark.parametrize(
    "num_of_qubits",
    [
        (1),
        (2),
        (3),
        (4),
        (5),
    ],
)
def test_qft(num_of_qubits):

    # Arrange
    reg = QuantumRegister(num_of_qubits)
    qft_gate = QuantumCircuit(reg, name="QFT")
    qft_gate.append(qft(num_of_qubits), reg)

    # Act and Assert
    qiskit_qft_gate = synth_qft_full(num_of_qubits)

    np.testing.assert_allclose(
        Operator(qft_gate).data,
        Operator(qiskit_qft_gate).data,
        atol=ATOL,
        rtol=RTOL,
    )


# This test is adapted from lab 2.
@pytest.mark.parametrize(
    "num_of_qubits",
    [
        (1),
        (2),
        (3),
        (4),
        (5),
    ],
)
def test_iqft(num_of_qubits):

    # Arrange
    reg = QuantumRegister(num_of_qubits)
    iqft_gate = QuantumCircuit(reg, name="IQFT")
    iqft_gate.append(iqft(num_of_qubits), reg)

    # Act and Assert
    qiskit_iqft_gate = synth_qft_full(num_of_qubits, inverse=True)

    np.testing.assert_allclose(
        Operator(iqft_gate).data,
        Operator(qiskit_iqft_gate).data,
        atol=ATOL,
        rtol=RTOL,
    )


@pytest.mark.parametrize(
    "number_of_qubits",
    [
        (1),
        (2),
        (3),
        (4),
        (5),
        (6),
    ],
)
def test_adder_aqft_against_qft(number_of_qubits):

    # ARRANGE
    # aqft implementation
    x_register = QuantumRegister(number_of_qubits, name="x")
    psi_register = QuantumRegister(number_of_qubits + 1, name="psi")
    adder_ancilla_register = QuantumRegister(1, name="Adder Ancilla (carry 0)")
    pgt_ancilla_register = QuantumRegister(1, name="PGT Ancilla")

    aqft_circuit = QuantumCircuit(
        x_register, psi_register, adder_ancilla_register, pgt_ancilla_register
    )
    aqft_circuit.append(
        aqft(number_of_qubits, number_of_qubits),
        x_register[:]
        + psi_register[:]
        + adder_ancilla_register[:]
        + pgt_ancilla_register[:],
    )

    # qiskit implementation
    input_reg = QuantumRegister(number_of_qubits)
    qiskit_qft_circuit = QuantumCircuit(input_reg, name="QFT")
    qiskit_qft_circuit.append(synth_qft_full(number_of_qubits), input_reg)

    # ACT
    input_states = [
        Statevector.from_label(
            "0" * (number_of_qubits + 3) + f"{i:0{number_of_qubits}b}"
        )
        for i in range(2**number_of_qubits)
    ]
    evolved_states = [state.evolve(aqft_circuit) for state in input_states]
    expected_states = [state.evolve(qiskit_qft_circuit) for state in input_states]

    # ASSERT
    for i in range(len(expected_states)):
        np.testing.assert_allclose(
            evolved_states[i], expected_states[i], rtol=RTOL, atol=ATOL
        )


@pytest.mark.parametrize(
    "number_of_qubits, b",
    [(i, j) for i in range(1, 7) for j in range(1, i + 1)],
)
def test_adder_aqft_against_reference_z_gate_aqft(number_of_qubits, b):
    # ARRANGE
    # aqft adder implementation
    x_register = QuantumRegister(number_of_qubits, name="x")
    psi_register = QuantumRegister(b + 1, name="psi")
    adder_ancilla_register = QuantumRegister(1, name="Adder Ancilla (carry 0)")
    pgt_ancilla_register = QuantumRegister(1, name="PGT Ancilla")

    aqft_adder_circuit = QuantumCircuit(
        x_register, psi_register, adder_ancilla_register, pgt_ancilla_register
    )
    aqft_adder_circuit.append(
        aqft(number_of_qubits, b),
        x_register[:]
        + psi_register[:]
        + adder_ancilla_register[:]
        + pgt_ancilla_register[:],
    )

    # aqft z-gate reference implementation
    input_reg = QuantumRegister(number_of_qubits)
    aqft_z_gate_circuit = QuantumCircuit(input_reg)
    aqft_z_gate_circuit.append(
        aqft_z_gate_reference_implementation(number_of_qubits, b), input_reg
    )

    # ACT
    input_states = [
        Statevector.from_label(
            "0" * (number_of_qubits + 3) + f"{i:0{number_of_qubits}b}"
        )
        for i in range(2**number_of_qubits)
    ]
    evolved_states = [state.evolve(aqft_adder_circuit) for state in input_states]
    expected_states = [state.evolve(aqft_z_gate_circuit) for state in input_states]

    # ASSERT
    for i in range(len(expected_states)):
        np.testing.assert_allclose(
            evolved_states[i], expected_states[i], rtol=RTOL, atol=ATOL
        )
