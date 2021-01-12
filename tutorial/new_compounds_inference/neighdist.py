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
        mot: A numpy array with the positions, in cartesian coordinates, occupied by the crystal atoms. In fact, mot is a 4d-tensor (for each atom,
        there is three coordinates).
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

#The next two functions model the influence of the neighbors on a atom in a Wyckoff site. 
#The first function has a gaussian profile and it is used in patolli. 
#Nevertheless, it could be of interest to use a potential profile.
def exponential(x,n = 1, coef = 1):
    return np.exp(-coef*np.power(x,n))

def potential(x, n = 1, coef = 1):
    return np.power(coef*x,-n)

#The next two functions set a cutoff radius. 
#In patolli, we only used angcos.
def angcos(x, dist = 5):
    return np.multiply(0.5*(np.cos(np.pi*x/dist) + 1), x <= dist)

def angtanh(x, dist = 5):
    return np.multiply(np.power(np.tahn(1-x/dist),3), x <= dist)

def rij(mult=[1,1,3], p = np.zeros((1,1,1)), zero = 1, dist=100, 
        sites = 4, radii = [1,1,1]):
    """
    This function computes the local environment function without the electronegativity difference of the involved atom pair; i.e.
    the central atom and its neighbor.
    Parameters:
        mult: a list with the multiplicites of each Wyckoff site. This list is sorted ascendingly.
        p: An array with the positions of the crystal atoms. This array is computed with the function positions, described above.
        zero: The index of the atoms in the center of the crystal (array p). This value is computed with the function positions.
        dist: The cutoff radius, in angstroms.
        sites: The number of sites to consider in the compound.
        radii: a list of the atomic radii of the average species in each Wyckoff site. The atomic radii are in angstroms. These
            atomic radii correspond to those reported by Rahm, 2016.
    Returns:
        rij: A square matrix, where each element corresponds to the local environment function for a given atom pair ij. This value is dimensionless.
    """
    #The next line takes into account only the radii of the occupied sites.
    radii = [item for item in radii if item != 0]
    #The next line adds progressively the multiplicities. If the multiplicities are [4,4,4,8], the next line returns [0,4,8,12,20].
    #Similarly, if the multiplicities are [1,1,3], l = [0,1,2,5].
    l = [sum(mult[:i]) for i in range(len(mult)+1)]
    
    rij = list()
    
    #With the next for statement, we define the different Wyckoff sites (chemical environments) as centers i. 
    #The local function is calculated for each center i.
    for i,atrad_i in zip(range(1,len(l)),radii):
        #The next two lines compute the distance of all atoms to the center i.
        r = p - p[l[i-1],zero,:]
        r = np.linalg.norm(r,axis=2)
        
        #The next for statement defines the different neighbors j, in terms of Wyckoff sites, to a center i.
        for j,atrad_j in zip(range(1,len(l)), radii):
            #Next line defines the denominator of the exponential (gaussian) function. This denominator is squared.
            coef = (atrad_i + atrad_j)**(-2)
            #With the next two lines, the neighbors located in different Wyckoff sites to j are excluded.
            init = l[j-1]
            fin = l[j]
            rj = r[init:fin,:]
            #With the next line, the atoms within the cutoff radius are considered.
            rj = rj[rj <= dist]
            #The next line avoids repetition of the atom in center as neighbor.
            rj = rj[rj != 0]
            #The next line computes the local function without the electronegativity difference.
            rj = np.sum(exponential(x = rj, n =2, coef=coef)*angcos(x=rj,dist=dist))
            #Appending the local function to the list rij.
            rij += [rj]
    
    #The next two lines reshape the list as an square matrix.
    lon = int((len(rij))**(1/2))
    rij = np.asarray(rij).reshape((lon,lon))

    s = sites
    #The next if statement adds zeros if the compound has less Wyckoff sites than the considered (check the parameters). 
    #For example, if the compound has three Wyckoff sites but this was constrained to be describe with four, it adds zeros to reshape rij matrix.
    if lon != s:
        rij = np.concatenate((np.zeros((rij.shape[0],s-lon)),rij),axis=1)
        rij = np.concatenate((np.zeros((s-lon,s)),rij), axis=0)    
        
    return rij
