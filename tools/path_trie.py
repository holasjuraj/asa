from functools import cmp_to_key
import numpy as np

class Node:
    def __init__(self, num_actions):
        self.counter = 0
        self.start_observations = []
        self.end_observations = []
        self.num_actions = num_actions
        self.children = None
    
    def has_children(self):
        return self.children is not None
    
    def get_count(self):
        return self.counter
    
    def get_starts(self):
        return np.array(self.start_observations)
    
    def get_ends(self):
        return np.array(self.end_observations)
    
    def add_count(self, n=1, start_obs=None, end_obs=None):
        '''
        Add to the counter of this path, store observations.
        Observations are stored only if both start and end obs. is provided.
        :param n: amount to increase the counter
        :param start_obs: observation before this path
        :param end_obs: observation after last action of this path
        '''
        self.counter += n
        if start_obs is not None  and  end_obs is not None:
            self.start_observations.append(start_obs)
            self.end_observations.append(end_obs)
    
    def __getitem__(self, action):
        if not self.has_children():
            self.children = [None] * self.num_actions
            for i in range(self.num_actions):
                self.children[i] = Node(self.num_actions)
        return self.children[action]



class PathTrie:
    all_aggregations = ['mean', 'most_freq', 'nearest_mean', 'medoid']
    
    def __init__(self, num_actions):
        self.root = Node(num_actions)
        self.num_actions = num_actions
        self.num_eps = 0   # number of paths/roll-outs/episodes collected within the batch
        self.num_steps = 0 # total number of steps taken in batch
    
    
    def add(self, path, observations, add_subpaths=True, min_length=1):
        '''
        Add path to trie (increase its count). Path is stored reversed (last action is under the trie's root)!
        :param path: sequence of actions (must be integers!)
        :param observations: sequence of observations, long len(path)+1 (incl. obs. after last action)
        :param add_subpaths: if True, suffixes of path are added too
        :param min_length: minimum length of subpath to add
        '''
        length = len(path)
        path = reversed(path)
        end_obs = observations[-1] # observation after last action
        
        node = self.root
        for i, a in enumerate(path):
            node = node[a]
            if (i == length-1) or (add_subpaths and i+1 >= min_length):
                node.add_count(start_obs=observations[length - i - 1], end_obs=end_obs)
        return node.get_count()
    
    
    def add_all_subpaths(self, path, observations, min_length=1, max_length=np.inf):
        '''
        Given whole path, traverse it and add all subpaths of specified length.
        :param path: sequence of actions (must be integers!)
        :param observations: sequence of observations, equally long as path
        :param min_length: minimum length of subpath to add
        :param max_length: maximum length of subpath to add
        '''
        self.num_eps += 1
        self.num_steps += len(path)
        if type(observations) == np.ndarray:
            observations = observations.tolist()
        observations.append(None) # "missing" observation after last action
        
        for end in range(1, len(path)+1):
            start = max(0, end - max_length)
            if end-start < min_length: continue
            self.add(path[start:end], observations[start:end+1],
                     add_subpaths=True, min_length=min_length)
    
        
    def apply_null_hyp(self, top, cnt, null_hyp_opts={}):
        '''
        Normalize count with null hypothesis count.
        :param top: top of the trie - path (sequence of actions) in reverse order
        :param cnt: count (# of occurences) of this path
        :param null_hyp_opts: override trie's counts of:
                              {'num_paths' : number of paths/roll-outs/episodes collected within the batch (int),
                               'num_steps' : total number of steps taken in batch (int)}
        '''
        sq_len = len(top)
        num_steps = null_hyp_opts.get('num_steps', self.num_steps)
        num_eps   = null_hyp_opts.get('num_paths', self.num_eps)
        '''
        Following formula is a variation of: https://math.stackexchange.com/a/220549
        n: length of total string = num_steps
        m: length of searched substring = sq_len
        p_i: probability of letter/action = (1 / self.num_actions) for all i
        Formula = (n + num_eps*m - num_eps*1) * (1 / self.num_actions)^m
        '''
        null_cnt = (num_steps + num_eps * (-sq_len+1))  /  (self.num_actions**sq_len)
        return cnt / null_cnt
    
    
    def aggregate_observations(self, observations, aggregations='all'):
        '''
        Aggregate path-start/path-end observations.
        :param observations: numpy array of observations
        :param aggregations: 'all' or list of aggregations to compute
        :return: {'mean': <mean of observations>,
                  'most_freq': <most frequent element>
                  'nearest_mean': <element that is nearest to the mean>
                  'medoid': <element with shortest distance to all others>}
        '''
        if aggregations == 'all':
            aggregations = PathTrie.all_aggregations
        if not isinstance(aggregations, list):
            aggregations = [aggregations]
        result = {}
        if 'mean' in aggregations:
            result['mean'] = observations.mean(axis=0)
        if 'most_freq' in aggregations:
            (values, counts) = np.unique(observations, axis=0, return_counts=True)
            result['most_freq'] = values[np.argmax(counts)]
        if 'nearest_mean' in aggregations:
            obs_mean = observations.mean(axis=0)
            result['nearest_mean'] = observations[ np.linalg.norm(observations - obs_mean, axis=1).argmin() ]
        if 'medoid' in aggregations:
            count = observations.shape[0]
            dist = np.array([[np.linalg.norm(observations[i] - observations[j]) if j < i else 0
                              for j in range(count)] for i in range(count)]) # lower triangle
            dist = dist + dist.T # full distance matrix
            result['medoid'] = observations[ dist.sum(axis=0).argmin() ]
        return result
    
    
    def items(self, action_map=None, min_count=1, min_f_score=0, sort=True, max_results=np.inf, null_hyp_opts={}, aggregations='all'):
        '''
        Get all paths, their counts and f-scores.
        :param action_map: dictionary from action numbers into chars (for better readability)
        :param min_count: return only paths with count min_count or more
        :param min_f_score: return only paths with f-score min_f_score or more
        :param sort: sort result. Boolean, or list of field indexes to sort along, e.g.[2, 1] : count=1, f-score=2.
                     Default sorting: (f-score DESC, count DESC, path-length DESC, path ASC)
        :param max_results: number of results to return, only applied if sort is used
        :param null_hyp_opts: override trie's counts of:
                              {'num_paths' : number of paths/roll-outs/episodes collected within the batch (int),
                               'num_steps' : total number of steps taken in batch (int)}
        :param aggregations: list of aggregations to compute on start/end observations, or 'all'
        :return: [(path, count, f_score, aggregated_observations_dict, trie_node), ...]
        '''
        paths = []
        # TODO change: return list of dicts, not list of tuples
        # TODO add: some counting of std of paths within a node?
        
        def top_to_path(top):
            '''
            Convert "top of trie" to list/text representation of path
            :param top: top of the trie - path (sequence of actions) in reverse order
            '''
            if action_map is None:
                return list(reversed(top))
            else:
                return ''.join( map(lambda a: action_map[a], reversed(top)) )
        
        def collect_subtree_items(node, top):
            '''
            Recursive method to collect all items (paths and their counts, f-scores) in a subtrie.
            :param node: trie node to start in
            :param top: top of the trie - path up to this node (sequence of actions) in reverse order
            '''
            if len(top) >= 0:
                c = node.get_count()
                f = self.apply_null_hyp(top, c, null_hyp_opts)
                if c >= min_count and f >= min_f_score:
                    paths.append([
                        top_to_path(top),
                        c,
                        f,
                        None, # aggregations will be computed later
                        node] )
            if node.has_children():
                for a in range(self.num_actions):
                    new_top = top + [a]
                    child = node[a]
                    collect_subtree_items(child, new_top)
        
        # Collect
        collect_subtree_items(self.root, [])
                
        # Sort
        if sort:
            def cmp(a, b): return (a > b) - (a < b)  # Java-like compareTo()
            if not isinstance(sort, list): sort = [2, 1]
            def comparator(a, b):
                res = 0
                for i in sort:
                    res = res or -cmp(a[i], b[i])
                return res or  -cmp(len(a[0]), len(b[0]))  or  cmp(a[0], b[0])
            paths.sort(key=cmp_to_key(comparator))
            paths = paths[:min(len(paths), max_results)]
        
        # Compute aggregations for returning paths
        for path in paths:
            node = path[4]
            # start and end observations are expected to be flattened vectors (no scalars or matrices)
            startsEnds = np.concatenate([node.get_starts(), node.get_ends()], axis=1)
            path[3] = self.aggregate_observations(startsEnds, aggregations)
        
        return paths



if __name__ == '__main__':
    '''
    Example usage.
    Good with: def aggregate_observations(self, observations): observations.tolist()
    '''
    trie = PathTrie(2)
    # obs: 1 2 3 4 5 6 7 8 9 0 1 2 3 .
    # acts: s s L L L s s s L L L s L
    path = [1,1,0,0,0,1,1,1,0,0,0,1,0]
    observations = np.atleast_2d(np.arange(1, 14)).T
    trie.add_all_subpaths(path, observations, min_length=3)
    
    for path in trie.items(action_map={0:'L', 1:'s'}, min_count=2):
        print('{:12} {:3} {:5.2f} {}\t{}'.format(*path))





