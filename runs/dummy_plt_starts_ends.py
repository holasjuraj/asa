#!/usr/bin/env python

import numpy as np
import matplotlib
matplotlib.use('qt5Agg')
import matplotlib.pyplot as plt

# Simple dummy plot of starts and ends of 'sLLLs' sequence, count=465, f-score=3.520, mean-aggregation
s = np.array([[0.41935484,  0.14623656,  0.07311828,  0.09032258,  0.16344086],
              [0.41505376,  0.09462366,  0.02150538,  0.04516129,  0.11612903],
              [0.41290323,  0.07311828,  0.00000000,  0.96989247,  0.97634409],
              [0.41935484,  0.07956989,  0.00215054,  0.98494624,  0.98709677],
              [0.42365591,  0.08387097,  0.00645161,  0.68602151,  0.68602151]])

e = np.array([[0.51494253,  0.10574713,  0.03218391,  0.65977011,  0.66206897],
              [0.51264368,  0.10344828,  0.03218391,  0.99080460,  0.99310345],
              [0.49655172,  0.07816092,  0.00000000,  0.95862069,  0.96781609],
              [0.49885057,  0.08045977,  0.00229885,  0.00459770,  0.00919540],
              [0.51724138,  0.09885057,  0.02068966,  0.02298851,  0.02758621]])

_, ax = plt.subplots(1,2)
ax[0].imshow(s, interpolation='nearest', cmap='gray_r', vmin=0, vmax=1)
ax[0].set_title('starts')
ax[1].imshow(e, interpolation='nearest', cmap='gray_r', vmin=0, vmax=1)
ax[1].set_title('ends')

plt.show()