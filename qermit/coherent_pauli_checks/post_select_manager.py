from collections import Counter

class PostSelectMgr:
    
    def __init__(self, counts, cbits, post_select_cbits):
        
        self.counts = counts
        self.cbits = cbits
        self.post_select_cbits = post_select_cbits
        
    def get_post_selected_shot(self, shot):
        return tuple([bit for bit, reg in zip(shot, self.cbits) if reg not in self.post_select_cbits])
    
    def is_post_select_shot(self, shot):
        return all(bit==0 for bit, reg in zip(shot, self.cbits) if reg in self.post_select_cbits)
        
    def post_select(self):
        
        post_select_dict = {}
        for shot, count in self.counts.items():
            if self.is_post_select_shot(shot):                
                post_select_dict[self.get_post_selected_shot(shot)] = count
                
        return Counter(post_select_dict)
    
    def merge(self):
        
        merge_dict = {}
        for shot, count in self.counts.items():
            post_selected_shot = self.get_post_selected_shot(shot)
            merge_dict[post_selected_shot] = merge_dict.get(post_selected_shot, 0) + count
            
        return Counter(merge_dict)