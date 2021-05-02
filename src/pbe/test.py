# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 21:36:53 2020

@author: Admin
"""

from numba import jit
import numpy as np
import time
import pbe

# %% User functions


# Coagulation function


@jit
def aggregation_kernel_1(p1, p2, t=None, Y=None):
    x1 = p1
    x2 = p2

    beta0 = 1

    result = beta0

    return result


@jit
def aggregation_kernel_2(p1, p2, t=None, Y=None):
    (x1, y1) = p1
    (x2, y2) = p2

    v1 = x1 + y1
    v2 = x2 + y2

    fx1 = x1/v1
    fx2 = x2/v2

    beta0 = 1
    # eff = e11*fx1*fx2 + ((1-fx1)*fx2 + fx1*(1-fx2))*e12 + (1-fx1)*(1-fx2)*e22
    # beta = beta0*eff*(2 + (v1/v2)**(1/3) + (v2/v1)**(1/3))

    result = beta0

    return result


def psdini1(x):

    x0 = 1.
    N0 = 1.

    result = 4*N0/x0*(x/x0)*np.exp(-2*(x/x0))

    return result


def psdini2(x, y):

    x0 = 1.
    y0 = 1.
    N0 = 1.

    result = 4*N0/(x0*y0)*(x/x0)*np.exp(-2*(x/x0) - (y/y0))

    return result

# %% test


time_start = time.time()

Mx = 50
My = 10
xrange = [0.1, 10]
yrange = [1, 100]
xkind = 'geo'
ykind = 'lin'
xlabel = 'x variable (kg)'
ylabel = 'y variable (kg)'

g1 = pbe.GridRectangular(dim=1,
                         varrange=[xrange],
                         numcells=[Mx],
                         kind=[xkind],
                         label=[xlabel])

g2 = pbe.GridRectangular(dim=2,
                         varrange=[xrange, yrange],
                         numcells=[Mx, My],
                         kind=[xkind, ykind],
                         label=[xlabel, ylabel])

g1.show()
g2.show()

t0 = 0
tend = 1
tsteps = 10
times = np.linspace(t0, tend, tsteps)

s1 = pbe.System(grid=g1,
                aggfnc=aggregation_kernel_1,
                inifnc=psdini1,
                times=times,
                comment='My first 1D PBE in Python')

s2 = pbe.System(grid=g2,
                aggfnc=aggregation_kernel_2,
                inifnc=psdini2,
                times=times,
                comment='My first 2D PBE in Python')


s1.solve()
s1.show()

time_end = time.time()
print("Elapsed (after compilation) = %s" % (time_end-time_start))
