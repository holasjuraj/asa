from functools import cmp_to_key
import numpy as np


class Node:
    def __init__(self, num_actions):
        self._counter = 0
        self._start_obss = []
        self._end_obss = []
        self._num_actions = num_actions
        self._children = None

    @property
    def has_children(self):
        return self._children is not None

    @property
    def count(self):
        return self._counter

    @property
    def start_observations(self):
        return np.array(self._start_obss)

    @property
    def end_observations(self):
        return np.array(self._end_obss)

    def add_count(self, start_obs, end_obs):
        """
        Add to the counter of this path, store observations.
        Observations are stored only if both start and end obs. is provided.
        :param start_obs: observation before this path
        :param end_obs: observation after last action of this path
        """
        self._counter += 1
        assert start_obs is not None
        assert end_obs is not None
        self._start_obss.append(start_obs)
        self._end_obss.append(end_obs)

    def __getitem__(self, action):
        if not self.has_children:
            self._children = [None] * self._num_actions
            for i in range(self._num_actions):  # type: int
                # noinspection PyTypeChecker
                self._children[i] = Node(self._num_actions)
        return self._children[action]


class PathTrie:
    all_aggregations = ['mean', 'most_freq', 'nearest_mean', 'medoid']

    def __init__(self, num_actions):
        self.root = Node(num_actions)
        self.num_actions = num_actions
        self.num_eps = 0    # number of paths/roll-outs/episodes collected within the batch
        self.num_steps = 0  # total number of steps taken in batch
        self.action_counts = [0] * self.num_actions

    def add(self, path, observations, add_subpaths=True, min_length=1):
        """
        Add path to trie (increase its count). Path is stored reversed (last action is under the trie's root)!
        :param path: sequence of actions (must be integers!)
        :param observations: sequence of observations, long len(path)+1 (incl. obs. after last action)
        :param add_subpaths: if True, suffixes of path are added too
        :param min_length: minimum length of subpath to add
        """
        length = len(path)
        path = reversed(path)
        end_obs = observations[-1]  # observation after last action

        node = self.root
        for i, a in enumerate(path):
            node = node[a]
            if (i == length-1) or (add_subpaths and i+1 >= min_length):
                node.add_count(start_obs=observations[length - i - 1], end_obs=end_obs)

    def add_all_subpaths(self, actions, observations, min_length=1, max_length=np.inf):
        """
        Given whole path, traverse it and add all subpaths of specified length.
        :param actions: sequence of actions (must be integers!)
        :param observations: sequence of observations preceding each action
        :param min_length: minimum length of subpath to add
        :param max_length: maximum length of subpath to add
        """
        # TODO assert actions 1D shape (after conversion to np array)
        self.num_eps += 1
        self.num_steps += len(actions)
        for a in actions:
            self.action_counts[a] += 1
        # TODO {pass the last observation as well | Re: no, rather ignore last action. Done.},
        #   then change actions and observations to numpy arrays
        #   Changes in callers of this function, in add(), get_starts(), get_ends(), Node.add_count(), maybe others
        if type(observations) == np.ndarray:
            observations = observations.tolist()

        for end in range(1, len(actions)):  # last action is ignored, because there is no observation ofter it
            start = max(0, end - max_length)
            if end-start < min_length:
                continue
            self.add(actions[start:end], observations[start:end + 1],
                     add_subpaths=True, min_length=min_length)

    def apply_null_hyp(self, top, cnt, null_hyp_opts=None):
        """
        Normalize count with null hypothesis count.
        :param top: top of the trie - path (sequence of actions) in reverse order
        :param cnt: count (# of occurrences) of this path
        :param null_hyp_opts: override trie's counts of:
                              {'num_paths' : number of paths/roll-outs/episodes collected within the batch (int),
                               'num_steps' : total number of steps taken in batch (int),
                               'action_counts' : how many times each action occurred (list)}
        """
        if not null_hyp_opts:
            null_hyp_opts = {}
        sq_len = len(top)
        num_steps = null_hyp_opts.get('num_steps', self.num_steps)
        num_eps = null_hyp_opts.get('num_paths', self.num_eps)
        action_probs = np.asarray(null_hyp_opts.get('action_counts', self.action_counts))
        action_probs = (action_probs + 1) / np.sum(action_probs)
        sq_prob = float(np.prod([action_probs[a] for a in top]))
        '''
        Following formula is a variation of: https://math.stackexchange.com/a/1740273
        n: length of total string = num_steps
        m: length of searched substring = sq_len
        p_i: probability of letter/action = action_probs[top[i]]
        Formula = (n - num_eps*m + num_eps*1) * prod(p_i)
        '''
        null_cnt = (num_steps + num_eps * (-sq_len+1)) / (self.num_actions**sq_len)
        return cnt / null_cnt

    @staticmethod
    def aggregate_observations(observations, aggregations='all'):
        """
        Aggregate path-start/path-end observations.
        :param observations: numpy array of observations
        :param aggregations: 'all' or list of aggregations to compute
        :return: {'mean': <mean of observations>,
                  'most_freq': <most frequent element>
                  'nearest_mean': <element that is nearest to the mean>
                  'medoid': <element with shortest distance to all others>}
        """
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
            result['nearest_mean'] = observations[np.linalg.norm(observations - obs_mean, axis=1).argmin()]
        if 'medoid' in aggregations:
            count = observations.shape[0]
            dist = np.array([[np.linalg.norm(observations[i] - observations[j]) if j < i else 0
                              for j in range(count)] for i in range(count)])    # lower triangle
            dist = dist + dist.T    # full distance matrix
            result['medoid'] = observations[dist.sum(axis=0).argmin()]
        return result

    # noinspection PyMethodMayBeStatic
    def map_actions_to_text(self, actions, action_map=None):
        """
        Convert sequence of actions into readable form using action_map.
        If action_map is None, then None is returned
        :param actions: list - sequence of actions
        :param action_map: dictionary from action numbers into chars (for better readability)
        """
        if action_map is None:
            return None
        else:
            return ''.join(map(lambda a: action_map[a], actions))

    def item_for_path(self,
                      path,
                      action_map=None,
                      null_hyp_opts=None,
                      aggregations='all'):
        """
        Return trie node for given path, or None if such node does not exist or is empty
        :param path: list of actions (integers)
        :param action_map: dictionary from action numbers into chars (for better readability)
        :param null_hyp_opts: override trie's counts of:
                              {'num_paths' : number of paths/roll-outs/episodes collected within the batch (int),
                               'num_steps' : total number of steps taken in batch (int)}
        :param aggregations: list of aggregations to compute on start/end observations, or 'all'
        :return: {actions, actions_text, count, f_score,
                  start_observations, end_observations, agg_observations, trie_node}
        """
        top = list(reversed(path))
        if not null_hyp_opts:
            null_hyp_opts = {}

        node = self.root
        for action in top:
            if not node.has_children:
                return None
            node = node[action]
        if node.count == 0:
            return None  # empty node

        c = node.count
        f = self.apply_null_hyp(top, c, null_hyp_opts)
        starts_ends = np.concatenate([node.start_observations, node.end_observations], axis=1)
        result = {
                'actions': path,
                'actions_text': self.map_actions_to_text(path, action_map),
                'count': c,
                'f_score': f,
                'start_observations': node.start_observations,
                'end_observations': node.end_observations,
                'agg_observations': self.aggregate_observations(starts_ends, aggregations),
                'trie_node': node
            }
        return result

    def items(self,
              action_map=None,
              min_count=1,
              min_f_score=0,
              sort=True,
              max_results=np.inf,
              null_hyp_opts=None,
              aggregations='all'):
        """
        Get all paths, their counts and f-scores.
        :param action_map: dictionary from action numbers into chars (for better readability)
        :param min_count: return only paths with count min_count or more
        :param min_f_score: return only paths with f-score min_f_score or more
        :param sort: sort result. Boolean, or list of dict keys to sort along, e.g.['f_score', 'count'].
                     Default sorting: (f-score DESC, count DESC, path-length DESC, path ASC)
        :param max_results: number of results to return, only applied if sort is used
        :param null_hyp_opts: override trie's counts of:
                              {'num_paths' : number of paths/roll-outs/episodes collected within the batch (int),
                               'num_steps' : total number of steps taken in batch (int)}
        :param aggregations: list of aggregations to compute on start/end observations, or 'all'
        :return: [{actions, actions_text, count, f_score,
                   start_observations, end_observations, agg_observations, trie_node}, ...]
        """
        if not null_hyp_opts:
            null_hyp_opts = {}
        paths = []
        # TODO? add some counting of std of paths within a node?

        def top_to_actions(top):
            """
            Convert "top of trie" to list of actions
            :param top: top of the trie - path (sequence of actions) in reverse order
            """
            return list(reversed(top))

        def collect_subtree_items(node, top):
            """
            Recursive method to collect all items (paths and their counts, f-scores) in a subtree.
            :param node: trie node to start in
            :param top: top of the trie - path up to this node (sequence of actions) in reverse order
            """
            if len(top) >= 0:
                c = node.count
                f = self.apply_null_hyp(top, c, null_hyp_opts)
                if c >= min_count and f >= min_f_score:
                    actions = top_to_actions(top)
                    paths.append({
                            'actions': actions,
                            'actions_text': self.map_actions_to_text(actions, action_map),
                            'count': c,
                            'f_score': f,
                            'start_observations': node.start_observations,
                            'end_observations': node.end_observations,
                            'agg_observations': None,   # aggregations will be computed later
                            'trie_node': node
                        })
            if node.has_children:
                for a in range(self.num_actions):
                    new_top = top + [a]
                    child = node[a]
                    collect_subtree_items(child, new_top)

        # Collect
        collect_subtree_items(self.root, [])

        # Sort
        if sort:
            def cmp(a, b):
                return (a > b) - (a < b)  # Java-like compareTo()
            if not isinstance(sort, list):
                sort = ['f_score', 'count']

            def comparator(a, b):
                res = 0
                for i in sort:
                    res = res or -cmp(a[i], b[i])
                return res \
                       or -cmp(len(a['actions']), len(b['actions'])) \
                       or cmp(a['actions'], b['actions'])

            paths.sort(key=cmp_to_key(comparator))
            paths = paths[:min(len(paths), max_results)]

        # Compute aggregations for returning paths
        for path in paths:
            n = path['trie_node']
            # start and end observations are expected to be flattened vectors (no scalars or matrices)
            starts_ends = np.concatenate([n.start_observations, n.end_observations], axis=1)
            path['agg_observations'] = self.aggregate_observations(starts_ends, aggregations)

        return paths


if __name__ == '__main__':
    '''
    Example usage.
    Good with: def aggregate_observations(self, observations): observations.tolist()
    '''
    trie = PathTrie(2)
    # obs: 1 2 3  4  5  6  7  8  9  0  1  2  3  .
    # acts: s s  L  L  L  s  s  s  L  L  L  s  L
    pth = [1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0]
    obss = np.atleast_2d(np.arange(1, 14)).T
    trie.add_all_subpaths(pth, obss, min_length=3)

    for p in trie.items(action_map={0: 'L', 1: 's'}, min_count=2):
        print('{actions_text:12} {count:3} {f_score:5.2f} {agg_observations}'.format(**p))

    my_item = trie.item_for_path([0, 0, 1], action_map={0: 'L', 1: 's'})
    print('My item:\n{actions_text:12} {count:3} {f_score:5.2f} {agg_observations}'.format(**my_item))
