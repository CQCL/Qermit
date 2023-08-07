from collections import Counter
from pytket.backends.backendresult import BackendResult
from pytket.utils.outcomearray import OutcomeArray 

class PostselectMgr:
    
    def __init__(
        self,
        compute_cbits,
        postselect_cbits,
    ):
        
        intersect = set(compute_cbits).intersection(set(postselect_cbits))
        if intersect:
            raise Exception(
                f"{intersect} are post select and compute qubits. " +
                "They cannot be both."
            )

        self.compute_cbits = compute_cbits
        self.postselect_cbits = postselect_cbits

        self.cbits = compute_cbits + postselect_cbits
        
    def get_post_selected_shot(self, shot):
        return tuple([bit for bit, reg in zip(shot, self.cbits) if reg not in self.postselect_cbits])
    
    def is_post_select_shot(self, shot):
        return all(bit==0 for bit, reg in zip(shot, self.cbits) if reg in self.postselect_cbits)

    def dict_to_result(self, result_dict):

        if result_dict == {}:
            return BackendResult()
        
        return BackendResult(
            counts=Counter({
                OutcomeArray.from_readouts([key]): val
                for key, val in result_dict.items()
            }),
            c_bits=self.compute_cbits,
        )
        
    def post_select_result(self, result):

        post_select_dict = {}
        for shot, count in result.get_counts(cbits=self.cbits).items():
            if self.is_post_select_shot(shot):                
                post_select_dict[self.get_post_selected_shot(shot)] = count

        return self.dict_to_result(post_select_dict)

    def merge_result(self, result):
        
        merge_dict = {}
        for shot, count in result.get_counts(cbits=self.cbits).items():
            post_selected_shot = self.get_post_selected_shot(shot)
            merge_dict[post_selected_shot] = merge_dict.get(post_selected_shot, 0) + count

        return self.dict_to_result(merge_dict)
