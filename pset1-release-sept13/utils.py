https://tutorcs.com
WeChat: cstutorcs
QQ: 749389476
Email: tutorcs@163.com

from webbrowser import get
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit import Aer
from qiskit.visualization import plot_state_city
from qiskit.compiler import transpile
from qiskit.visualization import plot_histogram
from qiskit.quantum_info.operators import Operator
from qiskit.quantum_info import Statevector

import numpy as np
from typing import Callable, List, Tuple
import math
from functools import *
import copy

QuantumClassicalOperator = Callable[[QuantumRegister, ClassicalRegister], QuantumCircuit]
QuantumOperator = Callable[[QuantumRegister], QuantumCircuit]
def append(global_circuit: QuantumCircuit,
             operator: QuantumClassicalOperator,
             quantum_register: List[int],
             classical_register: List[int]) -> QuantumCircuit:
    delegated_qregister = QuantumRegister(len(quantum_register), "quantum register")
    delegated_cregister = ClassicalRegister(len(classical_register), "classical register")
    delegated_operation_circuit = operator(delegated_qregister, delegated_cregister)
    global_circuit.append(delegated_operation_circuit,
                          qargs = [global_circuit.qubits[reg] for reg in quantum_register],
                          cargs = [global_circuit.clbits[reg] for reg in classical_register])
    return global_circuit.decompose(delegated_operation_circuit.name)

def append(global_circuit: QuantumCircuit,
             operator: QuantumOperator,
             quantum_register: List[int]) -> QuantumCircuit:
    delegated_qregister = QuantumRegister(len(quantum_register), "quantum register")
    delegated_operation_circuit = operator(delegated_qregister)
    global_circuit.append(delegated_operation_circuit,
                          qargs = [global_circuit.qubits[reg] for reg in quantum_register],
                          cargs = [])
    return global_circuit.decompose(delegated_operation_circuit.name)

def strip_zeros(f: str) -> str:
    # remove trailing zeros of a float
    return f.rstrip('0').rstrip('.') if '.' in f else f

def get_basis(n_qubit: int) -> List[str]:
    basis = []
    def helper(n: int, arr: List[int], i: int) -> None:
        if i == n:
            basis.append(''.join(arr))
            return
        arr[i] = '0'
        helper(n, arr, i + 1)
        arr[i] = '1'
        helper(n, arr, i + 1)
    helper(n_qubit, ['0']*n_qubit, 0)
    return basis

def beautify(amplitudes: List[complex], n_qubit: int, little_endian: bool = True) -> str:
    # Given a list of amplitudes
    # Return a beautified string in little-endian

    basis = get_basis(n_qubit)
    assert len(amplitudes)==len(basis), \
         f'Length of amplitudes {len(amplitudes)} does not match claimed dimension ({n_qubit})'
    base_amp = [(base, amp) for base, amp in zip(basis, amplitudes)]
    if not little_endian:
        base_amp = [(base[::-1], amp) for base, amp in zip(basis, amplitudes)]
        base_amp = sorted(base_amp, key = lambda x: x[0])
    s = ''
    for base, amp in base_amp:
        if amp.real==0 and amp.imag==0: continue
        s_amp = ''
        if amp.real != 0: s_amp += strip_zeros(f'{abs(amp.real):.2f}')
        neg = True if amp.real < 0 else False
        if amp.imag != 0:
            if neg:
                s_amp = '-' + s_amp
                neg = False
            neg = neg or (s_amp == '' and amp.imag<0)
            if s_amp != '': s_amp += '+' if amp.imag > 0 else '-'
            s_amp += strip_zeros(f'{abs(amp.imag):.2f}')
            s_amp += 'i'
        if amp.real!=0 and amp.imag!=0: s_amp = '(' + s_amp + ')'
        if s != '': s += ' + ' if not neg else ' - '
        elif neg:s += '-'
        s += f'{s_amp} |{base}>'
    return s

def apply_oracle_gate(type: str, input: str) -> str:
    a, b, c = input
    a, b, c = int(a), int(b), int(c)
    assert type in ['OR', 'XOR', 'AND']
    if type == 'OR': c = c ^ (a | b)
    elif type == 'XOR': c = c ^ (a ^ b)
    elif type == 'AND': c = c ^ (a & b)
    return f'{a}{b}{c}'

def test_gates(gate_operators: List[QuantumOperator]) -> None:
    print("Testing gates...")
    basis = get_basis(3)
    gate_types = ['OR', 'XOR', 'AND']
    qr = QuantumRegister(3, name="input")
    qc = QuantumCircuit(qr)
    for gate_type, gate_operator in zip(gate_types, gate_operators):
        correct = True
        for base in basis:
            oracle_output = apply_oracle_gate(type=gate_type, input=base)
            gate = gate_operator(qr)
            qc0 = copy.deepcopy(qc)
            for i, bit in enumerate(base):
                if bit == '1': qc0.x(i)
            qc0.compose(gate, qubits=[0,1,2], inplace=True)
            state = Statevector(qc0)
            for amp, base0 in zip(state, basis):
                if base0[::-1] == oracle_output: correct = correct and amp==1
                else: correct = correct and amp==0
        msg = 'OK' if correct else 'Error'
        print(f'{gate_type} gate: {msg}.')

def apply_oracle_adder(a: str, b: str) -> str:
    # a, b in little-endian
    # return c in little-endian
    c = int(a[::-1],2) + int(b[::-1], 2)
    return f'{c:03b}'[::-1]

def test_two_bit_adder(adder: QuantumCircuit, num_anc: int, has_scratch: bool) -> None:
    if has_scratch:
        print("Testing two-bit adder with scratch...")
    else:
        print("Testing two-bit adder without scratch...")
    basis = get_basis(2)
    qr = QuantumRegister(7+num_anc, name="input")
    cr = ClassicalRegister(7+num_anc, name="measurement outcomes")
    qc = QuantumCircuit(qr, cr)
    error = False
    for a in basis:
        for b in basis:
            c = apply_oracle_adder(a, b)
            qc0 = copy.deepcopy(qc)
            for i, bit in zip(range(0,2), a):
                if bit == '1': qc0.x(i)
            for i, bit in zip(range(2,4), b):
                if bit == '1': qc0.x(i)
            qc0.compose(adder, qubits=[i for i in range(7+num_anc)], inplace=True)
            qc0.measure([i for i in range(7+num_anc)], cr)
            backend = Aer.get_backend('qasm_simulator')
            job_sim = backend.run(transpile(qc0, backend), shots=1024)
            result_sim = job_sim.result()
            measurements = list(result_sim.get_counts(qc0).keys())
            if len(measurements)!=1:
                print(f'Error: obtained non-deterministic result.')
                Error = True
            output = measurements[0][::-1]
            oa, ob, oc, od = output[0:2], output[2:4], output[4:7], output[7:]
            # Note that oa, ob, oc are all expressed as little-endian now
            has_error = True
            if oc!=c:
                print(f'Error (incorrect): A = {a}, B = {b}, C = {oc}.')
            elif oa != a or ob !=b:
                print(f'Error (A or B modified): A = {a} -> {oa}, B = {b} -> {ob}.')
            elif not has_scratch and od != '0' * num_anc:
                print(f'Error (scratch): D = {od}.')
            else: has_error = False
            error = has_error or error
    if not error: print('OK.')
