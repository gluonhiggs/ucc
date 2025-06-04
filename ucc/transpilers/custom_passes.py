from qiskit.transpiler import TransformationPass
from qiskit.transpiler.passes import Commuting2qGateRouter
from qiskit.transpiler.passes.routing.commuting_2q_gate_routing import SwapStrategy

class ParallelizeCommutingGates(TransformationPass):
    def __init__(self, coupling_map):
        super().__init__()
        if coupling_map.size() < 2:
            self.swap_strategy = None  # No parallelization for < 2 qubits
        else:
            line = list(coupling_map.physical_qubits)
            self.swap_strategy = SwapStrategy.from_line(line)
        self.commuting_router = Commuting2qGateRouter(swap_strategy=self.swap_strategy)

    def run(self, dag):
        if self.swap_strategy is None:
            return dag  # Return unchanged for single-qubit backends
        return self.commuting_router.run(dag)