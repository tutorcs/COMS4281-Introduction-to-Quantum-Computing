https://tutorcs.com
WeChat: cstutorcs
QQ: 749389476
Email: tutorcs@163.com
from webbrowser import get
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit import Aer, execute
from qiskit.visualization import plot_state_city
from qiskit.compiler import transpile
from qiskit.visualization import plot_histogram
from qiskit.quantum_info.operators import Operator
from qiskit.quantum_info import Statevector
from qiskit.extensions import UnitaryGate

import numpy as np
import numpy.fft as fft
from typing import Callable, List, Tuple
import math
from functools import *
import copy
import random
import matplotlib.pyplot as plt

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
