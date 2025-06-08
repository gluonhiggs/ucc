from qiskit.transpiler.basepasses import TransformationPass
from qiskit.dagcircuit import DAGCircuit
from qiskit.quantum_info import Clifford
from qiskit.circuit.library import CXGate, HGate, SGate, XGate, YGate, ZGate, CZGate, SwapGate
from qiskit.transpiler.passes import TemplateOptimization
from qiskit.converters import dag_to_circuit, circuit_to_dag
import numpy as np
from itertools import combinations

class CliffordOptimizationPass(TransformationPass):
    """A custom pass to optimize Clifford circuits based on the paper."""
    
    def __init__(self):
        super().__init__()
        self.templates = self._define_templates()
        # Placeholder for precomputed optimal costs (to be populated offline, e.g., via BFS on Clifford group)
        self.optimal_costs_2q = {}  # {Clifford object: min_cnot_count}
        self.optimal_costs_3q = {}
        self.optimal_costs = {2: self.optimal_costs_2q, 3: self.optimal_costs_3q}
        if not self.optimal_costs_2q or not self.optimal_costs_3q:
            raise ValueError("Optimal cost tables must be precomputed and loaded.")
    def _define_templates(self):
        """Define Clifford-specific templates from the paper's Figure 1 (a-h) and (i) for H pushing."""
        from qiskit import QuantumCircuit
        templates = []

        # (a): CZ q0,q1; CZ q0,q1 = I
        template_a = QuantumCircuit(2)
        template_a.cz(0, 1)
        template_a.cz(0, 1)
        templates.append(template_a)

        # (b): CNOT-H-S sequence
        template_b = QuantumCircuit(2)
        template_b.cx(0, 1)
        template_b.h(0)
        template_b.s(0)
        template_b.s(0)
        template_b.h(0)
        template_b.cx(0, 1)
        templates.append(template_b)

        # (c): H q0; H q0 = I
        template_c = QuantumCircuit(1)
        template_c.h(0)
        template_c.h(0)
        templates.append(template_c)

        # (d): CNOT-Swap optimization
        template_d = QuantumCircuit(2)
        template_d.cx(0, 1)
        template_d.h(0)
        template_d.h(1)
        template_d.cx(0, 1)
        template_d.h(0)
        template_d.h(1)
        template_d.swap(0, 1)
        templates.append(template_d)

        # (e): Three-qubit CNOT sequence
        template_e = QuantumCircuit(3)
        template_e.cx(0, 1)
        template_e.h(1)
        template_e.cx(1, 2)
        template_e.h(1)
        template_e.cx(1, 2)
        template_e.cx(0, 1)
        templates.append(template_e)

        # (f): S-H-S-H-S-H = I
        template_f = QuantumCircuit(1)
        template_f.s(0)
        template_f.h(0)
        template_f.s(0)
        template_f.h(0)
        template_f.s(0)
        template_f.h(0)
        templates.append(template_f)

        # (g): S-S = Z
        template_g = QuantumCircuit(1)
        template_g.s(0)
        template_g.s(0)
        template_g.z(0)
        templates.append(template_g)

        # (h): S-CNOT sequence
        template_h = QuantumCircuit(2)
        template_h.s(0)
        template_h.s(1)
        template_h.cx(0, 1)
        template_h.h(1)
        template_h.s(1)
        template_h.sdg(1)
        template_h.h(1)
        template_h.cx(0, 1)
        template_h.cx(0, 1)
        templates.append(template_h)

        # (i): H pushing rule (H-CNOT -> CNOT-H)
        template_i = QuantumCircuit(2)
        template_i.h(1)
        template_i.cx(0, 1)
        templates.append(template_i)  # Replace with CNOT-H in application

        return templates

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        """Run the optimization pipeline iteratively until convergence."""
        prev_cnot_count = dag.count_ops().get('cx', 0) + 1  # Ensure first iteration runs
        while True:
            current_dag = dag.copy()
            dag = self.push_pauli_and_swap_gates(dag)
            dag = self.apply_template_matching(dag)
            dag = self.symbolic_peephole_optimization(dag)
            dag = self.apply_template_matching(dag, single_qubit_only=True)
            current_cnot_count = dag.count_ops().get('cx', 0)
            if current_cnot_count >= prev_cnot_count:
                return current_dag if current_cnot_count > prev_cnot_count else dag
            prev_cnot_count = current_cnot_count

    def push_pauli_and_swap_gates(self, dag: DAGCircuit) -> DAGCircuit:
        """Push Pauli (X, Y, Z) and Swap gates to the end using commutation rules (Eqs. 2-5)."""
        new_dag = DAGCircuit()
        new_dag.qubits = dag.qubits
        pauli_swap_nodes = []
        compute_nodes = []

        for node in dag.topological_op_nodes():
            if node.op.__class__ in [XGate, YGate, ZGate, SwapGate]:
                pauli_swap_nodes.append(node)
            else:
                compute_nodes.append(node)

        for node in compute_nodes:
            new_dag.apply_operation_back(node.op, node.qargs, node.cargs)
        for node in pauli_swap_nodes:
            new_dag.apply_operation_back(node.op, node.qargs, node.cargs)
        return new_dag
    
    def push_hadamard_phase(self, dag: DAGCircuit) -> DAGCircuit:
        """Push H and S gates through two-qubit gates to the right."""
        new_dag = DAGCircuit()
        new_dag.qubits = dag.qubits
        pending_gates = {}  # qubit: list of [gate, qargs]

        for node in dag.topological_op_nodes():
            qargs = node.qargs
            if len(qargs) == 1 and node.op.__class__ in [HGate, SGate]:
                pending_gates.setdefault(qargs[0], []).append([node.op, qargs])
            elif node.op.__class__ in [CXGate, CZGate]:
                c, t = qargs
                # Apply pending gates on control/target
                for q in [c, t]:
                    if q in pending_gates:
                        for op, qa in pending_gates[q]:
                            new_dag.apply_operation_back(op, qa)
                        del pending_gates[q]
                # Transform through CNOT/CZ
                if node.op.__class__ == CXGate:
                    if c in pending_gates:
                        for i, (op, _) in enumerate(pending_gates[c]):
                            if op.__class__ == HGate:
                                pending_gates[c][i][0] = ZGate()  # H-CNOT -> Z on control
                            elif op.__class__ == SGate:
                                pending_gates[c][i][0] = SGate()  # S unchanged
                    if t in pending_gates:
                        for i, (op, _) in enumerate(pending_gates[t]):
                            if op.__class__ == HGate:
                                pending_gates[t][i][0] = XGate()  # H-CNOT -> X on target
                            elif op.__class__ == SGate:
                                pending_gates[t][i][0] = YGate()  # S-CNOT -> Y
                new_dag.apply_operation_back(node.op, qargs)
            else:
                new_dag.apply_operation_back(node.op, qargs)

        # Apply remaining pending gates
        for q, gates in pending_gates.items():
            for op, qa in gates:
                new_dag.apply_operation_back(op, qa)
        return new_dag
    
    def optimize_swap_stage(self, dag: DAGCircuit) -> DAGCircuit:
        """Resynthesize the swap stage to align with two-qubit gates."""
        swap_dag = DAGCircuit()
        swap_dag.qubits = dag.qubits
        compute_nodes = [n for n in dag.topological_op_nodes() if n.op.__class__ not in [SwapGate]]
        swap_nodes = [n for n in dag.topological_op_nodes() if n.op.__class__ == SwapGate]
        two_qubit_gates = [n for n in compute_nodes if n.op.__class__ in [CXGate, CZGate]]

        # Map swaps to potential merges
        used_twqg = set()
        for swap in swap_nodes:
            qargs = swap.qargs
            merged = False
            for node in two_qubit_gates:
                if node not in used_twqg and set(node.qargs) == set(qargs):
                    swap_dag.apply_operation_back(node.op, qargs)
                    used_twqg.add(node)
                    merged = True
                    break
            if not merged:
                swap_dag.apply_operation_back(SwapGate(), qargs)

        # Add remaining compute nodes
        for node in compute_nodes:
            if node not in used_twqg:
                swap_dag.apply_operation_back(node.op, node.qargs)
        return swap_dag

    def apply_template_matching(self, dag: DAGCircuit, single_qubit_only=False) -> DAGCircuit:
        """Convert CNOT to CZ, apply template matching, and optimize Swaps."""
        from qiskit.circuit.library import CZGate
        # Step 1: Convert CNOT to CZ
        cz_dag = DAGCircuit()
        cz_dag.qubits = dag.qubits
        for node in dag.topological_op_nodes():
            if node.op.__class__ == CXGate:
                control, target = node.qargs
                cz_dag.apply_operation_back(HGate(), [target])
                cz_dag.apply_operation_back(CZGate(), [control, target])
                cz_dag.apply_operation_back(HGate(), [target])
            else:
                cz_dag.apply_operation_back(node.op, node.qargs)

        # Step 2: Apply template matching
        template_pass = TemplateOptimization(self.templates)
        if single_qubit_only:
            single_qubit_templates = [t for t in self.templates if t.num_qubits == 1]
            template_pass = TemplateOptimization(single_qubit_templates)
        opt_dag = template_pass.run(cz_dag)
        opt_dag = self.push_hadamard_phase(opt_dag)  # Added
        return self.optimize_swap_stage(opt_dag)  # Updated

    def symbolic_peephole_optimization(self, dag: DAGCircuit) -> DAGCircuit:
        """Apply symbolic peephole optimization for 2 and 3-qubit subsets."""
        n = len(dag.qubits)
        subsets = list(combinations(range(n), 2)) + list(combinations(range(n), 3))
        np.random.shuffle(subsets)

        for subset in subsets:
            A = set(subset)
            B = set(range(n)) - A
            sub_dag = DAGCircuit()
            sub_dag.qubits = [dag.qubits[i] for i in subset]
            entangling_groups = {}

            # Project circuit and identify entangling gates
            for node in dag.topological_op_nodes():
                if node.op.__class__ == CXGate:
                    control_idx, target_idx = [dag.qubits.index(q) for q in node.qargs]
                    if (control_idx in A and target_idx in B) or (control_idx in B and target_idx in A):
                        control = control_idx if control_idx in A else target_idx
                        entangling_groups.setdefault(control, []).append(node)
                    elif control_idx in A and target_idx in A:
                        sub_dag.apply_operation_back(node.op, node.qargs)
                elif node.qargs and all(dag.qubits.index(q) in A for q in node.qargs):
                    sub_dag.apply_operation_back(node.op, node.qargs)

            if not entangling_groups:
                continue

            # Define SPGs
            k = len(entangling_groups)
            pauli_ops = []
            phase_qubits = []
            for control, nodes in entangling_groups.items():
                target_idx = dag.qubits.index(nodes[0].qargs[1])
                pauli_idx = subset.index(target_idx)
                pauli_op = ['I'] * len(subset)
                pauli_op[pauli_idx] = 'X'
                pauli_ops.append(''.join(pauli_op))
                phase_qubits.append(dag.qubits[control])

            sub_circuit = dag_to_circuit(sub_dag)
            cliff = Clifford(sub_circuit)
            opt_cost, (opt_cliff, opt_phases) = self._optimize_subcircuit(cliff, pauli_ops, len(subset))
            if opt_cost >= len(sub_dag.op_nodes()):
                continue

            # Reconstruct optimized DAG
            new_dag = DAGCircuit()
            new_dag.qubits = dag.qubits
            opt_circuit = opt_cliff.to_circuit()
            opt_dag = circuit_to_dag(opt_circuit)

            for node in dag.topological_op_nodes():
                qargs_indices = [dag.qubits.index(q) for q in node.qargs]
                if node.op.__class__ == CXGate and any(n in sum(entangling_groups.values(), []) for n in [node]):
                    continue
                elif node.qargs and all(q in A for q in qargs_indices):
                    continue
                new_dag.apply_operation_back(node.op, node.qargs)

            qubit_map = {opt_dag.qubits[i]: dag.qubits[subset[i]] for i in range(len(subset))}
            for node in opt_dag.topological_op_nodes():
                mapped_qargs = [qubit_map.get(q, q) for q in node.qargs]
                new_dag.apply_operation_back(node.op, mapped_qargs)

            # Reattach entangling gates with phase correction
            for i, nodes in enumerate(entangling_groups.values()):
                phase_factor = opt_phases[i]
                for node in nodes:
                    new_dag.apply_operation_back(CXGate(), node.qargs)
                    if phase_factor == -1:
                        new_dag.apply_operation_back(ZGate(), [node.qargs[0]])
                    elif phase_factor == 1j:
                        new_dag.apply_operation_back(SGate(), [node.qargs[0]])
                    elif phase_factor == -1j:
                        new_dag.apply_operation_back(SGate().inverse(), [node.qargs[0]])

            dag = new_dag

        return dag

    def _optimize_subcircuit(self, clifford, pauli_ops, n_qubits):
        k = len(pauli_ops)
        if k == 0:
            return 0, clifford
        opt_dict = self.optimal_costs[n_qubits]
        P = [self._pauli_string_to_op(p, n_qubits) for p in pauli_ops]
        R = clifford

        f = [{} for _ in range(k + 1)]
        f[0] = {Clifford(np.eye(2**n_qubits)): (0, [])}  # (cost, phase_factors)
        def cnot_cost(U):
            return opt_dict.get(U, float('inf'))

        for j in range(1, k + 1):
            for U_j in opt_dict.keys():
                Q_j = U_j.adjoint().conjugate(P[j-1]).compose(U_j)
                spg_cost = sum(1 for p in Q_j.to_label() if p != 'I')
                min_cost = float('inf')
                best_phases = []
                for U_jm1, (cost, phases) in f[j-1].items():
                    transition_cost = cnot_cost(U_j.compose(U_jm1.adjoint()))
                    total = cost + transition_cost + spg_cost
                    if total < min_cost:
                        min_cost = total
                        phase_factor = (-1j) ** Q_j.phase  # Simplified phase tracking
                        best_phases = phases + [phase_factor]
                f[j][U_j] = (min_cost, best_phases)

        min_cost = float('inf')
        opt_Uk = None
        opt_phases = []
        for U_k, (cost, phases) in f[k].items():
            total_cost = cnot_cost(U_k) + cost
            if total_cost < min_cost:
                min_cost = total_cost
                opt_Uk = U_k
                opt_phases = phases

        opt_cliff = opt_Uk.compose(R)
        return min_cost, (opt_cliff, opt_phases)

    def _pauli_string_to_op(self, pauli_str, n_qubits):
        """Convert Pauli string to Clifford operator."""
        from qiskit.quantum_info import Pauli
        return Pauli(pauli_str[::-1].zfill(n_qubits))
    

from qiskit.circuit import QuantumCircuit
from qiskit.dagcircuit import DAGCircuit
from qiskit.circuit.library import XGate, YGate, ZGate, SwapGate, HGate, SGate, CXGate, CZGate
from collections import defaultdict

class PauliTracker:
    """Tracks Pauli operators across qubits."""
    def __init__(self, num_qubits):
        self.num_qubits = num_qubits
        # Default to identity (None means no Pauli, i.e., I)
        self.paulis = defaultdict(lambda: None)

    def apply_pauli(self, gate, qubit):
        """Multiply current Pauli with a new Pauli gate on a qubit."""
        current = self.paulis[qubit]
        new_gate = gate.__class__
        if current is None:
            self.paulis[qubit] = new_gate
        elif current == XGate and new_gate == XGate:
            self.paulis[qubit] = None  # X * X = I
        elif current == YGate and new_gate == YGate:
            self.paulis[qubit] = None  # Y * Y = I
        elif current == ZGate and new_gate == ZGate:
            self.paulis[qubit] = None  # Z * Z = I
        elif (current == XGate and new_gate == YGate) or (current == YGate and new_gate == XGate):
            self.paulis[qubit] = ZGate  # X * Y = iZ, Y * X = -iZ (phase ignored)
        elif (current == XGate and new_gate == ZGate) or (current == ZGate and new_gate == XGate):
            self.paulis[qubit] = YGate  # X * Z = -iY, Z * X = iY (phase ignored)
        elif (current == YGate and new_gate == ZGate) or (current == ZGate and new_gate == YGate):
            self.paulis[qubit] = XGate  # Y * Z = iX, Z * Y = -iX (phase ignored)

    def commute_through(self, gate, qargs):
        """Commute the Pauli through a compute gate, updating the Pauli."""
        gate_type = gate.__class__
        if gate_type == HGate:
            q = qargs[0]
            current = self.paulis[q]
            if current == XGate:
                self.paulis[q] = ZGate  # H X = Z H
            elif current == YGate:
                self.paulis[q] = YGate  # H Y = -Y H (phase ignored)
            elif current == ZGate:
                self.paulis[q] = XGate  # H Z = X H
        elif gate_type == SGate:
            q = qargs[0]
            current = self.paulis[q]
            if current == XGate:
                self.paulis[q] = YGate  # S X = Y S
            elif current == YGate:
                self.paulis[q] = XGate  # S Y = -X S (phase ignored)
            elif current == ZGate:
                self.paulis[q] = ZGate  # S Z = Z S
        elif gate_type == CXGate:
            q0, q1 = qargs  # control, target
            p0, p1 = self.paulis[q0], self.paulis[q1]
            self.paulis[q0] = None
            self.paulis[q1] = None
            if p0 == XGate:
                self.apply_pauli(XGate(), q0)
                self.apply_pauli(XGate(), q1)  # CNOT X1 = X1 X2 CNOT
            elif p0 == ZGate:
                self.apply_pauli(ZGate(), q0)  # CNOT Z1 = Z1 CNOT
            if p1 == XGate:
                self.apply_pauli(XGate(), q1)  # CNOT X2 = X2 CNOT
            elif p1 == ZGate:
                self.apply_pauli(ZGate(), q0)
                self.apply_pauli(ZGate(), q1)  # CNOT Z2 = Z1 Z2 CNOT
        elif gate_type == CZGate:
            q0, q1 = qargs
            p0, p1 = self.paulis[q0], self.paulis[q1]
            # CZ commutes with Z on either qubit, swaps X/Y with phase
            if p0 == XGate or p0 == YGate:
                self.paulis[q0] = None
                self.apply_pauli(XGate() if p0 == XGate else YGate(), q1)
            if p1 == XGate or p1 == YGate:
                self.paulis[q1] = None
                self.apply_pauli(XGate() if p1 == XGate else YGate(), q0)

    def to_gates(self):
        """Convert Pauli operator to list of gates."""
        gates = []
        for q, gate_type in self.paulis.items():
            if gate_type is not None:
                gates.append((gate_type(), [q]))
        return gates

def push_pauli_and_swap_gates(self, dag: DAGCircuit) -> DAGCircuit:
    """Push Pauli (X, Y, Z) and Swap gates to the end using commutation rules."""
    new_dag = DAGCircuit()
    new_dag.qubits = dag.qubits
    
    # Initialize Pauli tracker and permutation
    pauli = PauliTracker(len(dag.qubits))
    permutation = list(range(len(dag.qubits)))  # Identity permutation
    
    # Process gates in topological order
    for node in dag.topological_op_nodes():
        gate = node.op
        qargs = [dag.qubits.index(q) for q in node.qargs]
        
        if gate.__class__ in [XGate, YGate, ZGate]:
            # Apply Pauli on the permuted qubit
            q = permutation[qargs[0]]
            pauli.apply_pauli(gate, q)
        elif gate.__class__ == SwapGate:
            # Update permutation by swapping the mapped qubits
            q0, q1 = permutation[qargs[0]], permutation[qargs[1]]
            permutation[qargs[0]], permutation[qargs[1]] = q1, q0
        else:
            # Compute gate: adjust qubits, commute Pauli through, add to DAG
            mapped_qargs = [dag.qubits[permutation[q]] for q in qargs]
            pauli.commute_through(gate, mapped_qargs)
            new_dag.apply_operation_back(gate, mapped_qargs, node.cargs)
    
    # Add Swap gates to implement permutation
    current_perm = list(range(len(dag.qubits)))
    for i in range(len(permutation)):
        if current_perm[i] != permutation[i]:
            j = current_perm.index(permutation[i])
            new_dag.apply_operation_back(SwapGate(), [dag.qubits[i], dag.qubits[j]])
            current_perm[i], current_perm[j] = current_perm[j], current_perm[i]
    
    # Add final Pauli gates
    for gate, qargs in pauli.to_gates():
        new_dag.apply_operation_back(gate, [dag.qubits[qargs[0]]])
    
    return new_dag