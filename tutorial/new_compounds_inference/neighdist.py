#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  2 10:29:44 2018

@author: iG
"""

import Wyckoff_finder as wf
import pandas as pd
import numpy as np
import itertools as it
import time


def positions(pos = dict(),angles=[],abc=[], dist=100):#archivo=1010902, dist = 100):
    
    mult = [np.asarray(list(pos[i].values())).shape[1] for i in range(len(pos))]
    
    pos = np.concatenate([np.asarray(list(pos[item].values())) \
                          for item in range(len(pos))],axis=1)
        
    mot = pos.reshape((pos.shape[1],pos.shape[2]))
    
    volumen=abc[0]*abc[1]*abc[2]*np.sqrt(1-(np.cos(np.deg2rad(angles[0])))**2 - \
               (np.cos(np.deg2rad(angles[1])))**2 -\
               (np.cos(np.deg2rad(angles[2])))**2 + \
               2*np.cos(np.deg2rad(angles[0]))*np.cos(np.deg2rad(angles[1]))*np.cos(np.deg2rad(angles[2])))

    matrix=np.array([[abc[0],abc[1]*np.cos(np.deg2rad(angles[2])),abc[2]*np.cos(np.deg2rad(angles[1]))],
                      [0,abc[1]*np.sin(np.deg2rad(angles[2])),abc[2]*(np.cos(np.deg2rad(angles[0]))-np.cos(np.deg2rad(angles[1]))*np.cos(np.deg2rad(angles[2])))/np.sin(np.deg2rad(angles[2]))],
                      [0,0,volumen/(abc[0]*abc[1]*np.sin(np.deg2rad(angles[2])))]])

    mt = np.round(matrix,5)

    n = int(np.ceil((dist+10)/np.min(mt[mt > 0])))

    if n > 30:
        print('Number of unit cell for each half - dimension is ',n,'\n')
        n = 30

    tras = list(it.product(np.arange(-n,n+1),repeat=3))

    zero = tras.index((0,0,0))
    tras = np.asarray(tras)

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
