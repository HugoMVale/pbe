# -*- coding: utf-8 -*-
"""
Created on Sun Feb  2 18:12:35 2020

@author: hugomvale
"""

import numpy as np
from numba import jit
import sys
from fuzzywuzzy import fuzz
import matplotlib.pyplot as plt

# %% Simple error message function


def error_message(varname, varvalue):
    raise ValueError(
        "Invalid input for variable `{}` = {}".format(varname, varvalue))
    sys.exit()


# %% General grid class

class GridGeneric():
    "Generic grid class."

    mytype = 'Generic (abstract) grid object'
    maxdim = 3

    def __init__(self, dim=1, varrange=[[0, 1]], numcells=[1],
                 kind=['linear'], label=[''], comment=['']):

        self.__check_dim(dim)
        self.__check_varrange(varrange)
        self.__check_numcells(numcells)
        self.__check_kind(kind)
        self.__check_label(label)
        self.comment = comment
        self.make_grid()

    def __check_dim(self, dim):
        "Check if dimension input is valid."

        dim = int(round(dim))
        if 1 <= dim <= self.maxdim:
            self.dim = dim
        else:
            error_message('dim', dim)

    def __check_varrange(self, varrange):
        "Check if variable range input is valid."

        varrange = np.asarray(varrange)
        s = varrange.shape
        cond_1 = s[0] == self.dim
        cond_2 = s[1] == 2
        cond_3 = all((varrange[:, 1]-varrange[:, 0]) > 0)
        if cond_1 and cond_2 and cond_3:
            self.varrange = varrange
        else:
            error_message('varrange', varrange)

    def __check_numcells(self, numcells):
        "Check if number of cells input is valid."

        numcells = np.asarray(numcells)
        cond_1 = len(numcells) == self.dim
        cond_2 = all(numcells >= 1)
        if cond_1 and cond_2:
            self.numcells = numcells
        else:
            error_message('numcells', numcells)

    def __check_kind(self, kind):
        "Check if grid kind input is valid."

        if len(kind) != self.dim:
            error_message(self, kind, 'kind')
        for i in range(len(kind)):
            if fuzz.partial_ratio(kind[i], 'linear') > 90:
                kind[i] = 'linear'
            elif fuzz.partial_ratio(kind[i], 'geometric') > 90:
                kind[i] = 'geometric'
            else:
                error_message('kind', kind)
        self.kind = kind

    def __check_label(self, label):
        "Check if variables label input is valid."

        cond_1 = len(label) == self.dim
        if cond_1:
            self.label = label
        else:
            error_message('label', label)

    def set_varrange(self, varrange):
        "Set variable range and update grid."

        self.__check_varrange(varrange)
        self.make_grid()

    def set_numcells(self, numcells):
        "Set number of cells and update grid."

        self.__check_numcells(numcells)
        self.make_grid()

    def set_kind(self, kind):
        "Set grid kind and update grid."

        self.__check_kind(kind)
        self.make_grid()

    def set_label(self, label):
        "Set variables label."

        self.__check_label(label)


# %% Rectangular grid class

class GridRectangular(GridGeneric):
    "Rectangular grid class."

    mytype = 'Rectangular grid object, 1D or 2D'
    maxdim = 2

    def make_grid(self):
        "Make the grid"

        # Total number of cells in grid
        self.totalcells = np.prod(self.numcells)

        # Slicing along all axes
        self.edges = [None]*self.dim
        for i in range(self.dim):
            if self.kind[i] == 'linear':
                self.edges[i] = np.linspace(self.varrange[i, 0], self.varrange[i, 1],
                                            self.numcells[i]+1)
            elif self.kind[i] == 'geometric':
                if self.varrange[i, 0] > 0:
                    self.edges[i] = np.geomspace(self.varrange[i, 0], self.varrange[i, 1],
                                                 self.numcells[i]+1)
                else:
                    raise ValueError(
                        "Geometric grid requires non-negative `range`.")

        # Map cells
        self.idx = [[0]*self.dim]*self.totalcells

        # . Could not find a simple way to generalize mapping to n dimensions
        icell = 0
        if self.dim == 1:
            for i in range(self.numcells[0]):
                self.idx[icell] = [i]
                icell = icell + 1
        elif self.dim == 2:
            for ii in range(self.numcells[1]):
                for i in range(self.numcells[0]):
                    self.idx[icell] = [i, ii]
                    icell = icell + 1

        # Set cell boundaries
        self.cell_low = np.zeros([self.totalcells, self.dim])
        self.cell_high = np.zeros([self.totalcells, self.dim])

        for i in range(self.totalcells):
            for ii in range(self.dim):
                k = self.idx[i][ii]
                self.cell_low[i, ii] = self.edges[ii][k]
                self.cell_high[i, ii] = self.edges[ii][k+1]

        # Compute other cell properties
        self.cell_width = self.cell_high - self.cell_low
        self.cell_pivot = self.cell_low + 0.5*self.cell_width
        self.cell_area = np.prod(self.cell_width, 1)

    def show(self):
        "Generate plot to visualize grid."

        if self.dim == 1:
            # Plot cell pivots
            plt.plot(self.cell_pivot, np.ones(self.cell_pivot.size), 'bo')
            # Plot x edges
            for value in self.edges[0]:
                plt.axvline(value, color='k')
            # Plot labels
            plt.xlabel(self.label[0])
            # Axes limits
            plt.xlim(self.varrange[0])
            ax = plt.gca()
            ax.axes.yaxis.set_ticklabels([])
            plt.tick_params(axis="y", which="both", left=False, right=False)
            # Set log axes if required
            if self.kind[0] == 'geometric':
                plt.xscale('log')

        elif self.dim == 2:
            # Plot cell pivots
            plt.plot(self.cell_pivot[:, 0], self.cell_pivot[:, 1], 'bo')
            # Plot x edges
            for value in self.edges[0]:
                plt.axvline(value, color='k')
            # Plot y edges
            for value in self.edges[1]:
                plt.axhline(value, color='k')
            # Plot labels
            plt.xlabel(self.label[0])
            plt.ylabel(self.label[1])
            # Axes limits
            plt.xlim(self.varrange[0])
            plt.ylim(self.varrange[1])
            # Set log axes if required
            if self.kind[0] == 'geometric':
                plt.xscale('log')
            if self.kind[1] == 'geometric':
                plt.yscale('log')

        plt.show()

# %% System class


class System():

    mytype = 'PBE system object'

    def __init__(self, grid, aggfnc, inifnc, times=[0, 1], comment=''):

        self.__check_grid(grid)
        self.aggfnc = aggfnc
        self.inifnc = inifnc
        self.__check_times(times)
        self.comment = comment
        self.eval_ic()
        self.eval_agg_array()

    def __check_grid(self, grid):
        "Check if grid input is valid."

        if isinstance(grid, GridGeneric):
            self.grid = grid
        else:
            error_message('grid', grid)

    def __check_times(self, times):
        "Check if time input is valid."

        cond_1 = len(times) >= 2
        cond_2 = times[-1]-times[0] > 0
        if cond_1 and cond_2:
            self.times = times
        else:
            error_message('times', times)

    def eval_ic(self):
        """
        Since the spatial discretization is based on the FV method, we need to
        compute the initial cell average values. This is done by definition,
        i.e. by evaluating the integral of the number density function over
        the cell domain.
        Method:
        * 1D: Simpson's 1/3 rule.
        * 2D: The multiple integral is computed as an iterated integral;
              Simpson's 1/3 rule is applied in both directions.
        """
        self.ic = np.zeros([self.grid.totalcells])

        # The code is easy to read, but not very efficient, because the
        # borders are being evaluated twice.
        try:
            if self.grid.dim == 1:
                self.ic = (self.inifnc(self.grid.cell_low[:, 0]) +
                           4*self.inifnc(self.grid.cell_pivot[:, 0]) +
                           self.inifnc(self.grid.cell_high[:, 0]))/6

            elif self.grid.dim == 2:
                self.ic = (self.inifnc(self.grid.cell_low[:, 0],  self.grid.cell_low[:, 1]) +
                           4*self.inifnc(self.grid.cell_pivot[:, 0], self.grid.cell_low[:, 1]) +
                           self.inifnc(self.grid.cell_high[:, 0], self.grid.cell_low[:, 1]) +
                           4*self.inifnc(self.grid.cell_low[:, 0],  self.grid.cell_pivot[:, 1]) +
                           16*self.inifnc(self.grid.cell_pivot[:, 0], self.grid.cell_pivot[:, 1]) +
                           4*self.inifnc(self.grid.cell_high[:, 0], self.grid.cell_pivot[:, 1]) +
                           self.inifnc(self.grid.cell_low[:, 0],  self.grid.cell_high[:, 1]) +
                           4*self.inifnc(self.grid.cell_pivot[:, 0], self.grid.cell_high[:, 1]) +
                           self.inifnc(self.grid.cell_high[:, 0], self.grid.cell_high[:, 1]))/36
            # Check for unfeasible values
            if any(np.isnan(self.ic)) or any(np.isinf(self.ic)) or any(self.ic < 0):
                error_message('inifnc', self.ic)
        except Exception:
            error_message('inifnc', self.inifnc)

    def eval_agg_array(self):
        """
        Initialize and evaluate the aggregation array.
        Temporary method, ignoring symmetry and non-vectorized.

        idea: V[(j-1)n - j(j-1)/2 + i]

        """
        self.agg_array = np.zeros([self.grid.totalcells, self.grid.totalcells])
        for i in range(0, self.grid.totalcells):
            for ii in range(0, self.grid.totalcells):
                self.agg_array[i, ii] = self.aggfnc(self.grid.cell_pivot[i],
                                                    self.grid.cell_pivot[ii])

    def solve(self):
        """
        Solve the PBE.
        The solution is done in terms of

        """

        # Set integration options
        from scipy.integrate import solve_ivp
        self.atol = 1e-18
        self.rtol = 1e-4

        # Call ode solver
        t_span = [self.times[0], self.times[-1]]
        t_eval = self.times
        method = 'LSODA'

        self.sol = solve_ivp(self.__rate_agg,
                             t_span,
                             self.ic,
                             method,
                             t_eval,
                             dense_output=False,
                             events=None,
                             vectorized=False,
                             args=None
                             )
        # ,
        # **options)

        # Map output to shaped array

        pass

    def __rate_agg(self, t, n):
        """
        Net rate of aggregation = birth - death.

        """
        # Evaluate aggregation array
        self.eval_agg_array()

        # Birth
        birth = 0*n
        for i in range(self.grid.totalcells):
            # birth[i] = self.agg_array[j, k]*n[j]*n[k]
            pass

        # Death
        death = n * (self.agg_array @ n)

        # Net rate
        dndt = birth - death

        return dndt

    def show(self):
        "Plot the solution."

        # Get solution
        n = self.sol.y

        # Plot for 1D
        if self.grid.dim == 1:
            # Plot labels
            plt.xlabel(self.grid.label[0])
            plt.ylabel('number density function, n(x)')
            # Axes limits
            plt.xlim(self.grid.varrange[0])
            # Set log axes if required
            if self.grid.kind[0] == 'geometric':
                plt.xscale('log')
            # Plot n
            for i in range(n.shape[1]):
                nt = n[:, i]
                plt.plot(self.grid.cell_pivot, nt)
