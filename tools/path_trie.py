from functools import cmp_to_key

class Node:
    def __init__(self, num_actions):
        self.counter = 0
        self.num_actions = num_actions
        self.children = None
    
    def has_children(self):
        return self.children is not None
    
    def get_count(self, ):
        return self.counter
    
    def increase(self, n=1):
        self.counter += n
    
    def __getitem__(self, action):
        if not self.has_children():
            self.children = [None] * self.num_actions
            for i in range(self.num_actions):
                self.children[i] = Node(self.num_actions)
        return self.children[action]

class PathTrie:
    def __init__(self, num_actions):
        self.root = Node(num_actions)
        self.num_actions = num_actions
    
    def add(self, path, add_subpaths=True, min_length=1):
        '''
        Add path to trie (increase its count). Path is stored reversed!
        :param path: sequence of integers
        :param add_subpaths: if True, suffixes of path are added too
        :param min_length: minimum length of subpath to add
        '''
        length = len(path)
        path = reversed(path)
        node = self.root
        for i, a in enumerate(path):
            node = node[a]
            if (i == length-1) or (add_subpaths and i+1 >= min_length):
                node.increase()
        return node.get_count()
    
    def add_all_subpaths(self, path, min_length, max_length):
        '''
        Given whole path, traverse it and add all subpaths of specified length.
        '''
        for end in range(1, len(path)+1):
            start = max(0, end - max_length)
            if end-start < min_length: continue
            self.add(path[start:end], add_subpaths=True, min_length=min_length)
    
    def __getitem__(self, path):
        '''
        Get count of specified path
        '''
        path = reversed(path)
        node = self.root
        for a in path:
            node = node[a]
        return node.get_count()
    
    def items(self, action_map=None, min_count=1, sort=True, null_hyp={}):
        '''
        Get all paths and counts.
        :param action_map: dictionary from action numbers into chars (for better readability)
        :param min_count: return only paths with count min_count or more
        :param sort: sort result (path-count DESC, path-length DESC, path ASC)
        :param null_hyp: use normalization by null hypothesis, {'num_paths':int, 'num_steps':int}
        :return: [(path1, count1), ...]
        '''
        paths = []
        
        def top2path(top):
            if action_map is None:
                return list(reversed(top))
            else:
                return ''.join( map(lambda a: action_map[a], reversed(top)) )
        
        def use_null_hyp(top, cnt):
            if not null_hyp:
                return cnt
            sq_len = len(top)
            steps = null_hyp['num_steps']
            paths = null_hyp['num_paths']
            null_cnt = (steps + paths * (-sq_len+1))  /  (self.num_actions**sq_len)
            return cnt / null_cnt
        
        def collect_subtree_items(node, top):
            if len(top) >= 0 and node.get_count() >= min_count:
                f = use_null_hyp(top, node.get_count())
                paths.append( (top2path(top), f) )
            if node.has_children():
                for a in range(self.num_actions):
                    new_top = top + [a]
                    child = node[a]
                    collect_subtree_items(child, new_top)
        
        # Collect
        collect_subtree_items(self.root, [])
                
        # Sort
        if sort:
            def cmp(a, b): return (a > b) - (a < b)
            def comparator(a, b): return -cmp(a[1], b[1])  or  -cmp(len(a[0]), len(b[0]))  or  cmp(a[0], b[0])
            paths.sort(key=cmp_to_key(comparator))
        
        return paths
    

if __name__ == '__main__':
    '''
    Example usage
    '''
    trie = PathTrie(2)
    path = [1,1,0,0,0,1,1,1,0,0,0,1,1]
    trie.add_all_subpaths(path, min_length=3, max_length=100)
    
    for path, count in trie.items(action_map={0:'L', 1:'s'}, min_count=2):
        print(path, ':', count)





