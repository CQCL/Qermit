from pytket.circuit import OpType
from pytket.passes import AutoRebase

clifford_ops = [OpType.CZ, OpType.H, OpType.Z, OpType.S, OpType.X]
non_clifford_ops = [OpType.Rz]

cpc_rebase_pass = AutoRebase(gateset=set(clifford_ops + non_clifford_ops))
