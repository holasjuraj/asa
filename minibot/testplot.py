import numpy as np
import matplotlib
matplotlib.use('qt5Agg')
import matplotlib.pyplot as plt


class Bot:
    '''
    Little test to visually check correctness of robot`s moves.
    '''
    EPSILON = 0.0001
    def __init__(self):
        self.agent_width = 2.4/np.pi
        self.max_action_distance = 0.2
        self.agent_position = np.array([0., 0.])
        self.agent_ori = 0.

    def _get_move(self, action):
        '''
        Computes
        1) a vector: in which direction should the agent moves !!! in his frame of reference !!!
        2) orientation change
        '''
        al, ar = action * self.max_action_distance  # scale motor power <-1,1> to actual distance <-0.2,0.2>
        w = self.agent_width

        if np.abs(al - ar) < self.EPSILON:
            # al == ar -> Agent moves in straight line
            relative_move_vector = np.array([0., al])
            ori_change = 0
        elif np.abs(al + ar) < self.EPSILON:
            # al == -ar -> Agent rotates in place
            relative_move_vector = np.array([0., 0.])
            ori_change = ar * 2 / w
        else:
            # Agent moves and rotates at the same time
            r = (w * (ar + al)) / (2 * (ar - al))
            alpha = (ar + al) / (2 * r)
            me_pos = np.array([0., 0.])  # agent`s position in his frame of reference
            rot_center = np.array([-r, 0.])
            me_pos = me_pos - rot_center                      # 1) move to rotation center
            me_pos = self._rotation_matrix(alpha) @ me_pos    # 2) rotate
            me_pos = me_pos + rot_center                      # 3) move back
            # Vector me_pos now represents in which direction the agent should move !!! in his frame of reference !!!
            relative_move_vector = me_pos
            ori_change = alpha
        absolute_move_vector = self._rotation_matrix(self.agent_ori) @ relative_move_vector
        return (absolute_move_vector, ori_change)

    def _rotation_matrix(self, alpha):
        sin = np.sin(alpha)
        cos = np.cos(alpha)
        return np.array([[cos, -sin], [sin, cos]])

    def move(self, action):
        mv, oc = self._get_move(action)
        self.agent_position += mv
        self.agent_ori += oc

if __name__ == '__main__':
    bot = Bot()
    eps = 100
    trajectory = np.zeros((2, eps+1))
    trajectory[:, 0] = bot.agent_position
    for ep in range(eps):
        a = np.array([1., (ep+1)/eps])  # left wheel always full power, right wheel increases power
        bot.move(a)
        trajectory[:, ep+1] = bot.agent_position

    plt.plot(trajectory[0], trajectory[1], '-b')
    plt.plot(trajectory[0], trajectory[1], 'xk')
    plt.axis('equal')
    plt.show()
