import numpy as np
import matplotlib
matplotlib.use('qt5Agg')
import matplotlib.pyplot as plt

from minibot_env import MinibotEnv


class TestBot(MinibotEnv):
    '''
    Trimmed-down version of MiniBot to test _get_raw_move().
    '''
    EPSILON = 0.0001
    def __init__(self):
        self.agent_width = 2.4/np.pi
        self.max_action_distance = 0.2
        self.agent_position = np.array([0., 0.])
        self.agent_ori = 0.

    def move(self, action):
        mv, oc = self._get_raw_move(action)
        self.agent_position += mv
        self.agent_ori += oc



def plot_path(env, path):
    '''
    Plot single path in map.
    '''
    m = env._get_current_map()
    rows, cols = m.shape

    from matplotlib.collections import PatchCollection
    from matplotlib.patches import Rectangle
    plt.figure()
    plt.tight_layout()

    # Plot cells grid
    plt.xlim(-0.5, cols - 0.5)
    plt.ylim(-0.5, rows - 0.5)
    # Grid
    x_grid = np.arange(rows + 1) - 0.5
    y_grid = np.arange(cols + 1) - 0.5
    plt.plot(x_grid, np.stack([y_grid] * x_grid.size), ls='-', c='k', lw=1, alpha=0.8)
    plt.plot(np.stack([x_grid] * y_grid.size), y_grid, ls='-', c='k', lw=1, alpha=0.8)
    # Start, goal, walls and holes
    start = env._rc_to_xy(  np.argwhere(m == 'S').T  , rows)
    goal  = env._rc_to_xy(  np.argwhere(m == 'G').T  , rows)
    holes = env._rc_to_xy(  np.argwhere(m == 'H').T  , rows)
    walls = env._rc_to_xy(  np.argwhere(m == 'W').T  , rows)
    plt.scatter(*start, c='r', marker='o', s=50 )
    plt.scatter(*goal,  c='r', marker='x', s=50 )
    plt.scatter(*holes, c='k', marker='v', s=100)
    plt.gca().add_collection(PatchCollection([Rectangle(xy-0.5, 1, 1) for xy in walls.T], color='navy'))
    
    # Plot path
    # Starting position
    (start_r,), (start_c,) = np.nonzero(m == 'S')
    start_pos = env._rc_to_xy([start_r, start_c])
    # All others
    path = np.c_[start_pos, path]
    plt.plot(path[0], path[1], '-b')
    plt.plot(path[0], path[1], 'xk')

    plt.show()



if __name__ == '__main__':
    # # See TestBot walking
    # bot = TestBot()
    # eps = 100
    # trajectory = np.zeros((2, eps+1))
    # trajectory[:, 0] = bot.agent_position
    # for ep in range(eps):
    #     a = np.array([1., (ep+1)/eps])  # left wheel always full power, right wheel increases power
    #     bot.move(a)
    #     trajectory[:, ep+1] = bot.agent_position

    # plt.plot(trajectory[0], trajectory[1], '-b')
    # plt.plot(trajectory[0], trajectory[1], 'xk')
    # plt.axis('equal')
    # plt.show()

    # See MiniBot collision handling
    test_map = ["G##.",
                ".#..",
                ".#S.",
                "....",
                "##.."
               ]
    steps_s = [
            'RRRRRR.......RRR.l...', # slide left
            'RRRRRR.......RRR.ll..', # slide down
            'RRRRRR...RRR.r...',     # slide left
            'RRRRRR...RRR.rr..'      # slide up
            ]
    for steps in steps_s:
        bot = MinibotEnv()
        bot.all_maps = [test_map]   # dirty way to add new map
        bot.__init__()              #
        bot.reset()
        path = np.zeros((2, len(steps)))
        for i, step in enumerate(steps):
            action = np.array([1, 1])
            if step == 'l':     action = np.array([0, 1])   # left
            elif step == 'L':   action = np.array([-1, 1])  # sharp left
            elif step == 'r':   action = np.array([1, 0])   # right
            elif step == 'R':   action = np.array([1, -1])  # sharp right
            elif step == 'b':   action = np.array([-1, -1]) # back
            bot.step(action)
            path[:, i] = bot.agent_pos

        plot_path(bot, path)
