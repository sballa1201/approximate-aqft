import numpy as np
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit import Gate
from qiskit.circuit.library import ZGate


def qft(num_of_qubits: int) -> Gate:
    """Constructs a QFT gate with n qubits. Assumed that n is positive. This code snippet is taken from lab 2.

    Args:
        num_of_qubits: Number of qubits to be used in the gate. Assumed to be a positive integer.

    Returns:
        A QFT gate with n qubits.
    """
    circuit = QuantumCircuit(num_of_qubits, name="QFT")
    for j in range(num_of_qubits // 2):
        circuit.swap(j, num_of_qubits - 1 - j)
    for j in range(num_of_qubits):
        circuit.h(j)
        for ell in range(j + 1, num_of_qubits):
            circuit.cp(np.pi / 2 ** (ell - j), j, ell)

    circuit = circuit.to_gate()
    circuit.label = "QFT"
    return circuit


def iqft(num_of_qubits: int) -> Gate:
    """Constructs an inverse QFT gate with n qubits. Assumed that n is positive. This code snippet is taken from lab 2.

    Args:
        num_of_qubits: Number of qubits to be used in the gate. Assumed to be a positive integer.

    Returns:
        An inverse QFT gate with n qubits.
    """
    circuit = QuantumCircuit(num_of_qubits, name="IQFT")
    for j in reversed(range(num_of_qubits)):
        for ell in reversed(range(j + 1, num_of_qubits)):
            circuit.cp(-np.pi / 2 ** (ell - j), j, ell)
        circuit.h(j)
    for j in reversed(range(num_of_qubits // 2)):
        circuit.swap(j, num_of_qubits - 1 - j)

    circuit = circuit.to_gate()
    circuit.label = "IQFT"
    return circuit


def state_preparation(number_of_qubits: int) -> Gate:
    """Prepares the special state phi in the paper. Note that phi is in fact QFT(|1>).
    The paper uses RUS circuits to prepare this state but to simplify things we simply apply an X gate to the least significant bit (to apply the transformation |0> to |1>) and then a standard QFT.

        Args:
            num_of_qubits: Number of qubits to be used in the gate. Assumed to be a positive integer.

        Returns:
            A circuit that prepares QFT(|1>) from |0>.
    """
    reg = QuantumRegister(number_of_qubits)
    circuit = QuantumCircuit(reg, name="State Preparation")
    circuit.x(0)
    circuit.append(qft(number_of_qubits), reg)

    circuit = circuit.to_gate()
    circuit.label = "State Preparation"
    return circuit


def state_uncompute(number_of_qubits: int) -> Gate:
    """Unprepares the special state phi in the paper. Note that phi is in fact QFT(|1>). So we apply an IQFT and then an x gate on the least significant bit.

    Args:
        num_of_qubits: Number of qubits to be used in the gate. Assumed to be a positive integer.

    Returns:
        A circuit that unprepares QFT(|1>) to |0>.
    """
    reg = QuantumRegister(number_of_qubits)
    circuit = QuantumCircuit(reg, name="State Uncompute")

    circuit.append(iqft(number_of_qubits), reg)
    circuit.x(0)

    circuit = circuit.to_gate()
    circuit.label = "State Uncompute"
    return circuit


def curraco_ripple_carry_adder_modulo(number_of_qubits: int) -> Gate:
    """Quantum Adder implemented from [[3]]
    Args:
        num_of_qubits: Number of qubits to be used in the gate. Assumed to be a positive integer.
    Returns:
        A Quantum Adder on n-qubits
    """
    ancilla_register = QuantumRegister(1, name="c0")
    b_register = QuantumRegister(number_of_qubits, name="b")
    a_register = QuantumRegister(number_of_qubits, name="a")

    circuit = QuantumCircuit(
        ancilla_register, b_register, a_register, name="Curraco Adder"
    )

    # Sum first n-1 bits outputting the final carry into b_{n-1}

    # First MAJ gate
    circuit.append(curraco_maj(), [ancilla_register[0], b_register[0], a_register[0]])

    # All other MAJ gates
    for i in range(1, number_of_qubits - 1):
        circuit.append(curraco_maj(), [a_register[i - 1], b_register[i], a_register[i]])

    # Add the n-2 carry bit into b_{n-1}
    circuit.cx(a_register[-2], b_register[-1])

    # All UMA gates except the last
    for i in range(number_of_qubits - 2, 0, -1):
        circuit.append(curraco_uma(), [a_register[i - 1], b_register[i], a_register[i]])

    # Last UMA gate
    circuit.append(curraco_uma(), [ancilla_register[0], b_register[0], a_register[0]])

    # Final sum on the most significant bit
    circuit.cx(a_register[-1], b_register[-1])

    circuit = circuit.to_gate()
    circuit.label = "Curraco Moduluo 2^n Ripple Carry Adder"
    return circuit


def curraco_maj() -> Gate:
    """3 qubit MAJ gate as defined in [[3]].
    Returns:
        3 qubit MAJ gate
    """
    carry_register = QuantumRegister(1, name="c")
    b_register = QuantumRegister(1, name="b")
    a_register = QuantumRegister(1, name="a")

    circuit = QuantumCircuit(carry_register, b_register, a_register)

    circuit.cx(a_register, b_register)
    circuit.cx(a_register, carry_register)
    circuit.ccx(carry_register, b_register, a_register)
    circuit = circuit.to_gate()
    circuit.label = "MAJ"
    return circuit


def curraco_uma() -> Gate:
    """3 qubit UMA gate as defined in [[3]].
    Returns:
        3 qubit UMA gate
    """
    carry_register = QuantumRegister(1, name="c")
    b_register = QuantumRegister(1, name="b")
    a_register = QuantumRegister(1, name="a")

    circuit = QuantumCircuit(carry_register, b_register, a_register)

    circuit.ccx(carry_register, b_register, a_register)
    circuit.cx(a_register, carry_register)
    circuit.cx(carry_register, b_register)

    circuit = circuit.to_gate()
    circuit.label = "UMA"
    return circuit


def fanout_gate(number_of_qubits: int) -> Gate:
    """A  sequence of CNOT gates, all with that same control qubit, but differing target qubits.
    Args:
        number_of_qubits: 1 control qubit and number_of_qubits - 1 target qubits
    Returns:
        A gate where the first qubit is a control qubit for n-1 CNOTS for the other n-1 target qubits.
    """
    control_register = QuantumRegister(1, name="Control Qubit")
    target_register = QuantumRegister(number_of_qubits - 1, name="Target Qubits")
    circuit = QuantumCircuit(control_register, target_register)

    for i in range(number_of_qubits - 1):
        circuit.cx(control_register[0], target_register[i])

    circuit = circuit.to_gate()
    circuit.label = "Fanout Gate"
    return circuit


def aqft_after_prep(number_of_qubits, b):
    """Reference implementation of AQFT circuit 1 in paper [[1]] using powers of Z-gates
    Args:
            num_of_qubits: Number of qubits to be used in the gate. Assumed to be a positive integer.
            b: Number of Z-gates in each IPGT before truncating
    Returns:
        A reference AQFT gate with n qubits truncating Z-gates of power less than 1/2^b.
    """
    adder_bits = b + 1

    x_register = QuantumRegister(number_of_qubits, name="x")
    phi_register = QuantumRegister(adder_bits, name="phi")
    adder_ancilla_register = QuantumRegister(1, name="Adder Ancilla (carry 0)")
    pgt_ancilla_register = QuantumRegister(1, name="PGT Ancilla")

    circuit = QuantumCircuit(
        x_register, phi_register, adder_ancilla_register, pgt_ancilla_register
    )

    # First Layer

    circuit.h(x_register[0])
    circuit.z(x_register[0])
    for i in range(1, number_of_qubits):
        circuit.s(x_register[i])

    # The first adder
    truncated_bits = min(number_of_qubits, b)
    circuit.append(
        curraco_ripple_carry_adder_modulo(truncated_bits + 1),
        adder_ancilla_register[:]
        + phi_register[0 : truncated_bits + 1]
        + x_register[0:truncated_bits][::-1]
        + pgt_ancilla_register[:],
    )

    # # The rest of the adders and Hadamards
    for i in range(number_of_qubits - 1):
        truncated_bits = min(number_of_qubits - i, b)

        # First fanout on truncated_bits qubits starting at qubit i

        circuit.append(fanout_gate(truncated_bits), x_register[i : i + truncated_bits])

        # Adder on (ancilla bit + x_register[i:i+truncated_bits]) + phi_register
        # Note that the carry ancilla bit and the phi_register will be unchanged and x will become IPGT(x) as required.
        # The pgt_ancilla bit will have a Z gate applied to it but Z(|0>) = |0> so it remains unchanged too.
        circuit.append(
            curraco_ripple_carry_adder_modulo(truncated_bits + 1),
            adder_ancilla_register[:]
            + phi_register[-truncated_bits - 1 :]
            + x_register[i : i + truncated_bits][::-1]
            + pgt_ancilla_register[:],
        )

        # Last fanout on truncated_bits qubits starting at qubit i
        circuit.append(fanout_gate(truncated_bits), x_register[i : i + truncated_bits])

        # Add Hadamard and Z^1/2 gate
        circuit.h(x_register[i + 1])
        circuit.s(x_register[i + 1])

    # Add Z^1/2 gates for all qubits except last (since it is already done from the last loop)
    for i in range(number_of_qubits - 1):
        circuit.s(x_register[i])

    # # Final reverse adder on last b bits
    truncated_bits = min(number_of_qubits, b)
    reversed_truncated_x_reg = [
        x_register[number_of_qubits - 1 - i] for i in range(truncated_bits)
    ]
    circuit.append(
        curraco_ripple_carry_adder_modulo(truncated_bits + 1),
        adder_ancilla_register[:]
        + phi_register[0 : truncated_bits + 1]
        + reversed_truncated_x_reg[::-1]
        + pgt_ancilla_register[:],
    )

    circuit = circuit.to_gate()
    circuit.label = "AQFT Post Prep"
    return circuit


def aqft(number_of_qubits, b):
    """Implementation of the circuit in paper [[1]] including state preparation, bit swapping and uncomputation.
    Args:
            num_of_qubits: Number of qubits to be used in the gate. Assumed to be a positive integer.
            b: Number of Z-gates in each IPGT before truncating
    Returns:
        An AQFT gate with n qubits truncating Z-gates of power less than 1/2^b.
    """

    adder_bits = b + 1

    x_register = QuantumRegister(number_of_qubits, name="x")
    phi_register = QuantumRegister(adder_bits, name="phi")
    adder_ancilla_register = QuantumRegister(1, name="Adder Ancilla (carry 0)")
    pgt_ancilla_register = QuantumRegister(1, name="PGT Ancilla")

    circuit = QuantumCircuit(
        x_register, phi_register, adder_ancilla_register, pgt_ancilla_register
    )

    # Prepare the state phi
    circuit.append(state_preparation(adder_bits), phi_register)

    # Swap the order of qubits in x_register
    # Swap to reverse order of qubits
    for j in range(number_of_qubits // 2):
        circuit.swap(x_register[j], x_register[number_of_qubits - 1 - j])

    # Apply AQFT
    circuit.append(
        aqft_after_prep(number_of_qubits, b),
        x_register[:]
        + phi_register[:]
        + adder_ancilla_register[:]
        + pgt_ancilla_register[:],
    )

    # Uncompute the state phi
    circuit.append(state_uncompute(adder_bits), phi_register)

    return circuit


def aqft_z_gate_reference_implementation(number_of_qubits, b):
    """Reference implementation of the gate in paper [[1]] using powers of Z-gates
    Args:
            num_of_qubits: Number of qubits to be used in the gate. Assumed to be a positive integer.
            b: Number of Z-gates in each IPGT before truncating
    Returns:
        A reference AQFT gate with n qubits truncating Z-gates of power less than 1/2^b.

    """
    x_register = QuantumRegister(number_of_qubits, name="x")

    circuit = QuantumCircuit(x_register)

    for j in range(number_of_qubits // 2):
        circuit.swap(x_register[j], x_register[number_of_qubits - 1 - j])

    # First Layer
    circuit.h(x_register[0])
    circuit.z(x_register[0])
    for i in range(1, number_of_qubits):
        circuit.append(ZGate().power(1 / 2), [x_register[i]])

    # The first IPGT
    truncated_bits = min(number_of_qubits, b)
    for i in range(truncated_bits):
        circuit.append(ZGate().power(-1 / 2 ** (i + 1)), [x_register[i]])

    # # The rest of the IPGTs and hadamards
    for i in range(number_of_qubits - 1):
        truncated_bits = min(number_of_qubits - i, b)

        # First fanout on truncated_bits qubits starting at qubit i
        for j in range(1, truncated_bits):
            circuit.cx(x_register[i], x_register[i + j])

        # IPGT
        for j in range(i, i + truncated_bits):
            circuit.append(ZGate().power(-1 / 2 ** (j - i + 1)), [x_register[j]])

        # Last fanout on truncated_bits qubits starting at qubit i
        for j in range(1, truncated_bits):
            circuit.cx(x_register[i], x_register[i + j])

        # Add Hadamard and Z^1/2 gate
        circuit.h(x_register[i + 1])
        circuit.append(ZGate().power(1 / 2), [x_register[i + 1]])

    # Add Z^1/2 gates for all qubits except last (since it is already done from the last loop)
    for i in range(number_of_qubits - 1):
        circuit.append(ZGate().power(1 / 2), [x_register[i]])

    # Final reverse IPGT last b bits
    truncated_bits = min(number_of_qubits, b)
    for i in range(truncated_bits):
        circuit.append(
            ZGate().power(-1 / 2 ** (i + 1)), [x_register[number_of_qubits - 1 - i]]
        )

    circuit = circuit.to_gate()
    circuit.label = "AQFT No Adder verification"
    return circuit


def aqft_error(number_of_qubits, error):
    return int(np.ceil(np.log2(number_of_qubits / error)))
