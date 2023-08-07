from qermit.postselection import PostselectMgr, gen_postselect_mitres
from pytket.circuit import Bit
from collections import Counter
from pytket.backends.backendresult import BackendResult
from pytket.utils.outcomearray import OutcomeArray
from pytket.extensions.qiskit import AerBackend
from qermit import CircuitShots
from pytket import Circuit


def test_postselect_manager():
    
    compute_cbits = [Bit(name='A', index=0), Bit(name='C', index=0)]
    post_select_cbits = [Bit(name='B', index=0), Bit(name='A', index=1)]

    count_mgr = PostselectMgr(
        compute_cbits=compute_cbits,
        postselect_cbits=post_select_cbits,
    )

    counts = {
        (0,0,0,0):100,
        (0,1,0,0):100,
        (0,0,0,1):100,
        (0,1,0,1):100,
        (1,0,0,0):100,
        (1,1,0,0):100,
    }

    result = BackendResult(
        counts=Counter(
            {
                OutcomeArray.from_readouts([key]): val
                for key, val in counts.items()
            }
        ),
        c_bits=[Bit(name='A', index=0), Bit(name='A', index=1), Bit(name='B', index=0), Bit(name='C', index=0)],
    )

    assert count_mgr.post_select_result(result=result).get_counts() == Counter({(0, 0): 100, (0, 1): 100, (1, 0): 100})
    assert count_mgr.merge_result(result=result).get_counts() == Counter({(0, 0): 200, (0, 1): 200, (1, 0): 200})


def test_postselect_mitres():

    backend = AerBackend()
    circuit = Circuit(2).H(0).measure_all()
    cbits = circuit.bits
    postselect_mgr = PostselectMgr(
        compute_cbits=[cbits[1]],
        postselect_cbits=[cbits[0]],
    )
    postselect_mitres = gen_postselect_mitres(
        backend = backend,
        postselect_mgr = postselect_mgr
    )
    result_list = postselect_mitres.run([CircuitShots(Circuit = circuit, Shots = 50)])
    assert list(result_list[0].get_counts().keys()) == [(0,)]