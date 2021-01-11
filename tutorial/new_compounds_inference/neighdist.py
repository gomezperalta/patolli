#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  2 10:29:44 2018

@author: iG
"""

import pandas as pd
import numpy as np
import itertools as it
import time


def positions(pos = dict(),angles=[],abc=[], dist=100):
    
    """
    The aim of this function is to generate a crystal from the unit cell. This crystal must be as large enough to construct a sphere of the crystal 
    atoms with a given cutoff radius.
    Parameters:
        pos: A Python dictionary. In fact, this is a dictionary of dictionaries. It contains the relative positions for each occupied Wyckoff site
            within the unit cell.
        angles: A list with the values of the angles alpha, beta and gamma of the unit cell.
        cell: A list with the values of the vectors a, b and c of the unit cell.
        dist: A float. This defines how large the generated crystal must be along the largest vector abc of the lattice parameter. This value must be
            as large enough to construct a sphere of the crystal atoms with a cutoff radius (25 angstroms, in our work). 
            This parameters is considered in angstroms.
    Returns:
        mot: A numpy array with the positions, in cartesian coordinates, occupied by the crystal atoms. 
        zero: The index in mot corresponding to the center of the crystal.
        n: an integer, which tells how many unit cells are along the largest abc lattice parameter.
        mult: a list, with the multiplicities of each occupied Wyckoff site.
    """
    #The next line gets the multiplicity of each site.
    mult = [np.asarray(list(pos[i].values())).shape[1] for i in range(len(pos))]
    
    pos = np.concatenate([np.asarray(list(pos[item].values())) \
                          for item in range(len(pos))],axis=1)
        
    mot = pos.reshape((pos.shape[1],pos.shape[2]))
    #The next line computes the volume of the unit cell
    volumen=abc[0]*abc[1]*abc[2]*np.sqrt(1-(np.cos(np.deg2rad(angles[0])))**2 - \
               (np.cos(np.deg2rad(angles[1])))**2 -\
               (np.cos(np.deg2rad(angles[2])))**2 + \
               2*np.cos(np.deg2rad(angles[0]))*np.cos(np.deg2rad(angles[1]))*np.cos(np.deg2rad(angles[2])))
    
    #The next line defines the matrix that transforms the crystal system coordinates to cartesian ones.
    matrix=np.array([[abc[0],abc[1]*np.cos(np.deg2rad(angles[2])),abc[2]*np.cos(np.deg2rad(angles[1]))],
                      [0,abc[1]*np.sin(np.deg2rad(angles[2])),abc[2]*(np.cos(np.deg2rad(angles[0]))-np.cos(np.deg2rad(angles[1]))*np.cos(np.deg2rad(angles[2])))/np.sin(np.deg2rad(angles[2]))],
                      [0,0,volumen/(abc[0]*abc[1]*np.sin(np.deg2rad(angles[2])))]])

    mt = np.round(matrix,5)
    
    #The next line obtains how many unit cells lay on the direction of the largest abc lattice parameter.
    n = int(np.ceil((dist+10)/np.min(mt[mt > 0])))

    if n > 30:
        print('Number of unit cell for each half - dimension is ',n,'\n')
        n = 30
    
    #The next line generates all the traslational operations to create the crystal 
    tras = list(it.product(np.arange(-n,n+1),repeat=3))
    tras = np.asarray(tras)
    
    #The next  line defines the center of the generated crystal.
    zero = tras.index((0,0,0))
    
    h,w = mot.shape
    d = tras.shape[0]

    tras = tras.T
    
    mot = mot[:,:,np.newaxis]
    tras = tras[np.newaxis, :, :]
    
    
    mot = np.repeat(mot, d, axis=2)
    tras = np.repeat(tras, h, axis=0)

    mot = tras + mot
    mot = np.swapaxes(mot,1,2)
    mot = mot.astype(float)
    mot = np.matmul(mot,matrix)

    return mot, zero, n, mult

def exponential(x,n = 1, coef = 1):
    return np.exp(-coef*np.power(x,n))

def potential(x, n = 1, coef = 1):
    return np.power(coef*x,-n)

def angcos(x, dist = 5):
    return np.multiply(0.5*(np.cos(np.pi*x/dist) + 1), x <= dist)

def angtanh(x, dist = 5):
    return np.multiply(np.power(np.tahn(1-x/dist),3), x <= dist)

def rij(mult=[1,1,3], p = np.zeros((1,1,1)), zero = 1, dist=100, 
        sites = 4, radii = [1,1,1]):
    
    radii = [item for item in radii if item != 0]
    l = [sum(mult[:i]) for i in range(len(mult)+1)]
    rij = list()

    for i,atrad_i in zip(range(1,len(l)),radii):
        r = p - p[l[i-1],zero,:]
        r = np.linalg.norm(r,axis=2)
    
        for j,atrad_j in zip(range(1,len(l)), radii):
            coef = (atrad_i + atrad_j)**(-2)
            init = l[j-1]
            fin = l[j]
            rj = r[init:fin,:]
            rj = rj[rj <= dist]
            rj = rj[rj != 0]
            rj = np.sum(exponential(x = rj, n =2, coef=coef)*angcos(x=rj,dist=dist))
            rij += [rj]
    
    lon = int((len(rij))**(1/2))
    rij = np.asarray(rij).reshape((lon,lon))

    s = sites

    if lon != s:
        rij = np.concatenate((np.zeros((rij.shape[0],s-lon)),rij),axis=1)
        rij = np.concatenate((np.zeros((s-lon,s)),rij), axis=0)    
        
    return rij
