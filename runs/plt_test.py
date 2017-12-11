#!/usr/bin/env python

import numpy as np
import matplotlib
import time
matplotlib.use('qt5Agg')
import matplotlib.pyplot as plt


wait = 0.0

def ion():
    plt.ion()
    # time.sleep(wait)


def ioff():
    plt.ioff()


def redraw():
    plt.gcf().canvas.draw()
    plt.waitforbuttonpress(timeout=0.001)
    time.sleep(wait)


t = time.time()
ion()
print('ion: {:3f}ms'.format((time.time() - t)*1000))

T = []
for i in range(10):
    end = np.linspace(-2, 2, 10)[i]
    plt.plot(np.arange(-2, end, 0.1) ** 2, '.g')
    t = time.time()
    redraw()
    t = (time.time() - t) * 1000
    T.append(t)
    print('redraw {:2d}: {:3f}ms'.format(i, t))
    
t = time.time()
ioff()
print('iof: {:3f}ms'.format((time.time() - t)*1000))
print('Avg redraw: {:3f}'.format(np.mean(np.asarray(T))))
    
    
#     plt.show()
# plt.savefig('testfig.png')
