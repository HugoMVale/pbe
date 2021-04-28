# -*- coding: utf-8 -*-
"""
Created on Sun Feb  2 18:12:35 2020

@author: hugomvale
"""

import numpy as np
from numba import jit
import sys
from fuzzywuzzy import fuzz
from scipy.integrate import solve_ivp, odeint

#%% Simple error message function

def error_message(varname,varvalue):
    raise ValueError ("Invalid input for variable `{}` = {}".format(varname,varvalue))
    sys.exit()


#%% General grid class
    
class GridGeneric():  
    "Generic grid class."
    
    mytype = 'Generic (abstract) grid object'
    maxdim = 3
    
    def __init__(self, dim=1, varrange=[[0,1]], numcells=[1], 
                 kind=['linear'], label=[''], comment=['']):
        
        # self.dim = None
        # self.varrange = None
        # self.numcells = None
        # self.kind = None
        # self.label = None        
        # self.comment = None
        
        self.__check_dim(dim)
        self.__check_varrange(varrange)
        self.__check_numcells(numcells)
        self.__check_kind(kind)
        self.__check_label(label)
        self.make_grid()

    def __check_dim(self,dim):
        "Check if dimension input is valid."
        
        dim = int(round(dim))
        if 1 <= dim <= self.maxdim:
            self.dim = dim
        else:
            error_message('dim',dim)
            
    def __check_varrange(self,varrange):
        "Check if variable range input is valid."
        
        varrange = np.asarray(varrange)
        s = varrange.shape
        cond_1 = s[0]==self.dim
        cond_2 = s[1]==2
        cond_3 = all((varrange[:,1]-varrange[:,0])>0)
        if cond_1 and cond_2 and cond_3:
            self.varrange = varrange
        else:
            error_message('varrange',varrange)
    
    def __check_numcells(self,numcells):
        "Check if number of cells input is valid."
        
        numcells = np.array(numcells)
        cond_1 = len(numcells)==self.dim
        cond_2 = all(numcells>=1)
        if cond_1 and cond_2:
            self.numcells = numcells
        else:
            error_message('numcells',numcells)
            
    def __check_kind(self,kind):
        "Check if grid kind input is valid."
        
        if len(kind)!=self.dim:
            error_message(self,kind,'kind')
        for i in range(len(kind)):
            if fuzz.partial_ratio(kind[i],'linear')>90:
                kind[i] = 'linear'
            elif fuzz.partial_ratio(kind[i],'geometric')>90:
                kind[i] = 'geometric'
            else:
               error_message('kind',kind)
        self.kind = kind

    def __check_label(self,label):
        "Check if variables label input is valid."
        
        cond_1 = len(label)==self.dim
        if cond_1:
            self.label = label
        else:
            error_message('label',label)
    
    def set_varrange(self,varrange):
        "Set variable range and update grid."
        
        self.check_varrange(varrange)
        self.make_grid()
        
    def set_numcells(self,numcells):
        "Set number of cells and update grid."
        
        self.__check_numcells(numcells)
        self.make_grid()
        
    def set_kind(self,kind):
        "Set grid kind and update grid."
        
        self.__check_kind(kind)
        self.make_grid()

    def set_label(self,label):
        "Set variables label."
        
        self.__check_label(label)
                   
    def set_comment(self,comment):
        "Set comment."
        
        self.comment = comment
    

        
#%% Rectangular grid class
        
class GridRectangular(GridGeneric):
    "Rectangular grid class."
    
    mytype = 'Rectangular grid object, 1D or 2D'
    maxdim = 2
    
    def make_grid(self):
        "Make the grid"
        
        # Total number of cells in grid
        self.totalcells = np.prod(self.numcells)
        
        # Slicing along all axes
        list_points = [None]*self.dim
        for i in range(self.dim):
            if self.kind[i] =='linear':    
                list_points[i] = np.linspace(self.varrange[i,0],self.varrange[i,1],
                                             self.numcells[i]+1)
            elif self.kind[i] =='geometric':
                if self.varrange[i,0]>0:
                    list_points[i] = np.geomspace(self.varrange[i,0],self.varrange[i,1],
                                                  self.numcells[i]+1)
                else:
                    raise ValueError("Geometric grid requires non-negative `range`.")    
        self.list_points = list_points

        # Map cells
        self.idx = [[0]*self.dim]*self.totalcells
        
        # . Could not find a simple way to generalize mapping to n dimensions        
        icell = 0
        if self.dim==1:
            for i in range(self.numcells[0]):
                self.idx[icell] = [i]
                icell = icell + 1
        elif self.dim==2:
            for ii in range(self.numcells[1]):
                for i in range(self.numcells[0]):    
                    self.idx[icell] = [i,ii]
                    icell = icell + 1
        
        # Set cell boundaries
        self.cell_low  = np.zeros([self.totalcells,self.dim])         
        self.cell_high = np.zeros([self.totalcells,self.dim])
        
        for i in range(self.totalcells):
            for ii in range(self.dim):
                k = self.idx[i][ii]
                self.cell_low[i,ii] = list_points[ii][k]
                self.cell_high[i,ii] = list_points[ii][k+1]
        
        # Compute other cell properties        
        self.cell_width   = self.cell_high - self.cell_low    
        self.cell_center  = self.cell_low + 0.5*self.cell_width
        self.cell_area    = np.prod(self.cell_width,1)
          
#%% System class
        
class system():
    
    mytype = 'PBE system object'
    
    def __init__(self, grid, aggfnc, inifnc, times, comment=''):
    
        self.grid = None
        self.aggfnc = None
        self.inifnc = inifnc
        self.times = None        
        self.comment = comment
        self.__check_grid(grid)
        self.__check_times(times)
        self.eval_ic()
        
    def __check_grid(self,grid):
        "Check if grid input is valid."
        
        if isinstance(grid, GridGeneric):
            self.grid = grid
        else:
            error_message(self,grid,'grid')
    
    def __check_times(self,times):
        "Check if time input is valid."
        
        cond_1 = len(times)>=2
        cond_2 = times[-1]-times[0]>0
        if cond_1 and cond_2:
            self.times = times
        else:
            error_message('times',times)
    
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
        self.ic = np.zeros([self.grid.totalcells,1])
        
        # The code is easy to read, but not very efficient, because the 
        # borders are being evaluated twice.
        try:
            if self.grid.dim==1:
                self.ic = (  self.inifnc(self.grid.cell_low)   +
                           4*self.inifnc(self.grid.cell_center) +
                             self.inifnc(self.grid.cell_high) )/6
            
            elif self.grid.dim==2:
                self.ic = (  self.inifnc(self.grid.cell_low[:,0],  self.grid.cell_low[:,1]) +
                           4*self.inifnc(self.grid.cell_center[:,0],self.grid.cell_low[:,1]) +
                             self.inifnc(self.grid.cell_high[:,0], self.grid.cell_low[:,1]) +
                           4*self.inifnc(self.grid.cell_low[:,0],  self.grid.cell_center[:,1]) +
                          16*self.inifnc(self.grid.cell_center[:,0],self.grid.cell_center[:,1]) +
                           4*self.inifnc(self.grid.cell_high[:,0], self.grid.cell_center[:,1]) +
                             self.inifnc(self.grid.cell_low[:,0],  self.grid.cell_high[:,1]) +
                           4*self.inifnc(self.grid.cell_center[:,0],self.grid.cell_high[:,1]) +
                             self.inifnc(self.grid.cell_high[:,0], self.grid.cell_high[:,1]))/36
            # check for unfeasible values
            if any(np.isnan(self.ic)) or any(np.isinf(self.ic)) or any(self.ic<0):
                error_message('inifnc',self.ic)  
        except:
           error_message('inifnc',self.inifnc)     
   
    def solve(self):
        "Solve the set of PBE"
        
        # Set inital condition
        
        # Set integration options
        
        # Call ode solver
        
        # Map output to shaped array
        
        pass  
        
        