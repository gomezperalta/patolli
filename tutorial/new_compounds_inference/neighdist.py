#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  2 10:29:44 2018

@author: ivan
"""

import Wyckoff_finder as wf
import pandas as pd
import numpy as np
import itertools as it
import time


def positions(pos = dict(),angles=[],abc=[], dist=100):#archivo=1010902, dist = 100):
    
    #pos, _, angles, abc = wf.wyckoff_positions(archivo = archivo)
#def positions(archivo=1010902, dist = 100):
    
    #pos, _, angles, abc = wf.wyckoff_positions(archivo = archivo)
    
    mult = [np.asarray(list(pos[i].values())).shape[1] for i in range(len(pos))]
    
    pos = np.concatenate([np.asarray(list(pos[item].values())) \
                          for item in range(len(pos))],axis=1)
        
    mot = pos.reshape((pos.shape[1],pos.shape[2]))
    
    volumen=abc[0]*abc[1]*abc[2]*np.sqrt(1-(np.cos(np.deg2rad(angles[0])))**2 - \
               (np.cos(np.deg2rad(angles[1])))**2 -\
               (np.cos(np.deg2rad(angles[2])))**2 + \
               2*np.cos(np.deg2rad(angles[0]))*np.cos(np.deg2rad(angles[1]))*np.cos(np.deg2rad(angles[2])))

    #La variable matrix convierte las coordenadas relativas a coordenadas absolutas en un sistema cartesiano
    matrix=np.array([[abc[0],abc[1]*np.cos(np.deg2rad(angles[2])),abc[2]*np.cos(np.deg2rad(angles[1]))],
                      [0,abc[1]*np.sin(np.deg2rad(angles[2])),abc[2]*(np.cos(np.deg2rad(angles[0]))-np.cos(np.deg2rad(angles[1]))*np.cos(np.deg2rad(angles[2])))/np.sin(np.deg2rad(angles[2]))],
                      [0,0,volumen/(abc[0]*abc[1]*np.sin(np.deg2rad(angles[2])))]])

    mt = np.round(matrix,5)

    n = int(np.ceil((dist+10)/np.min(mt[mt > 0])))

    if n > 30:
        print('Number of unit cell for each half - dimension is ',n,'\n')
        #print(matrix)
        #print('\n')
        #cells = input('Please introduce a smaller value. If not, n = 30'+'\n')
        
        #if not cells:
        #    n = 30
        #else:
        #    n = int(cells)
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
    '''
    tras = np.stack([tras]*h, axis = 0)
    mot = np.stack([mot]*d, axis = 2)
    '''
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
            #print(coef)
            init = l[j-1]
            fin = l[j]
            rj = r[init:fin,:]
            rj = rj[rj <= dist]
            rj = rj[rj != 0]
            #print(np.sum(exponential(x = rj, n =2, coef=coef)*angcos(x=rj,dist=dist)))
            rj = np.sum(exponential(x = rj, n =2, coef=coef)*angcos(x=rj,dist=dist))#np.sum(exponential(x=rj,n=2, coef=coef))
            rij += [rj]
    
    lon = int((len(rij))**(1/2))
    rij = np.asarray(rij).reshape((lon,lon))

    s = sites

    if lon != s:
        rij = np.concatenate((np.zeros((rij.shape[0],s-lon)),rij),axis=1)
        rij = np.concatenate((np.zeros((s-lon,s)),rij), axis=0)    
        
    return rij

'''
df = pd.read_csv('dbtraval_190618.csv')
X = np.load('Xtraval_190618_raw_non.npy')
S = np.load('Straval_190618.npy')

row=186

p, z, n, m = positions(archivo=df['cif'][row])
radii = np.nan_to_num(X[row,:,1]/S[row,:,0])
try:
    r = rij(mult=m,p=p,zero=z,dist=dist, radii = radii) 
except:
    with open('problematic_samples.txt','a') as f:
        f.write('Check row ' + str(row))
        f.write('\n')
        f.close()
        r=0
print(r)

'''
'''
#Next code was to generate the files for each test collection
for date in ['040618', '180618', '190618']:
        
    for sets in ['test']:
    
        start = time.time()
        
        df = pd.read_csv('db' + sets + '_' + date +'.csv')
        X = np.load('X' + sets + '_' + date +'_raw_non.npy')
        S = np.load('S' + sets + '_' + date +'.npy')
        
        df = df.iloc[:1400,:].reset_index(drop=True)
        X = X[:1400]
        S = S[:1400]
        print('Working with dataframe ', date, sets)
        for row in range(len(df)):
    
            p,z,n,m = positions(archivo = df['cif'][row], dist = dist)

            radii = np.nan_to_num(X[row,:,1]/S[row,:,0])
    
            if row == 0:
                r = rij(mult=m,p=p,zero=z,dist=dist, radii = radii)    
                r = np.expand_dims(r,axis=0)
                
            else:
                try:
                    r_temp = rij(mult=m,p=p,zero=z,dist=dist, radii = radii)    
                    r_temp = np.expand_dims(r_temp,axis=0)
                except:
                    with open('problematic_samples_' + sets + '_' + date +  '.txt','a') as f:
                        f.write('Check row ' + str(row) + ' in database ' + \
                                date + '_' + sets)
                        f.write('\n')
                        f.close()
                    r_temp=np.zeros((1,r.shape[1],r.shape[2]))
                    print('There was a problem with row ' + str(row) + \
                          ' in ' + date + '_' + sets)
                r = np.concatenate((r,r_temp))
           
            if row%100 == 0:
                print(row)

        np.save('Xrij_' + sets + '_' + date, r) 
            
        print(time.time()-start)
'''
'''
#Next code was for true samples of traval - collections. They all are the same.
for date in ['040618']:#, '180618', '190618']:
        
    for sets in ['traval']:
    
        start = time.time()
        
        df = pd.read_csv('db' + sets + '_' + date +'.csv')
        X = np.load('X' + sets + '_' + date +'_raw_non.npy')
        S = np.load('S' + sets + '_' + date +'.npy')
        
        idx = int(len(df)/2)
        
        df = df.iloc[:idx,:].reset_index(drop=True)
        X = X[:idx]
        S = S[:idx]
        print('Working with dataframe ', date, sets)
        for row in range(len(df)):
    
            p,z,n,m = positions(archivo = df['cif'][row], dist = dist)

            radii = np.nan_to_num(X[row,:,1]/S[row,:,0])
    
            if row == 0:
                r = rij(mult=m,p=p,zero=z,dist=dist, radii = radii)    
                r = np.expand_dims(r,axis=0)
                
            else:
                try:
                    r_temp = rij(mult=m,p=p,zero=z,dist=dist, radii = radii)    
                    r_temp = np.expand_dims(r_temp,axis=0)
                except:
                    with open('problematic_samples_' + sets + '_true_' + date +  '.txt','a') as f:
                        f.write('Check row ' + str(row) + ' in database ' + \
                                date + '_' + sets)
                        f.write('\n')
                        f.close()
                    r_temp=np.zeros((1,r.shape[1],r.shape[2]))
                    print('There was a problem with row ' + str(row) + \
                          ' in ' + date + '_' + sets +' for true')
                r = np.concatenate((r,r_temp))
           
            if row%100 == 0:
                print(row)

        np.save('Xrij_' + sets + '_true_' + date, r) 
            
        print(time.time()-start)
'''
'''
#Next code is for false samples. Some collections are depleted while the task
#was completed.
for date in ['040618']:
        
    for sets in ['traval']:
    
        start = time.time()
        
        df = pd.read_csv('db' + sets + '_' + date +'.csv')
        X = np.load('X' + sets + '_' + date +'_raw_non.npy')
        S = np.load('S' + sets + '_' + date +'.npy')
        
        idx = int(len(df)/2)
        
        df = df.iloc[idx:,:].reset_index(drop=True)
        X = X[idx:]
        S = S[idx:]
        
        detenido = 232
        
        df = df.iloc[detenido:,:].reset_index(drop=True)
        X = X[detenido:]
        S = S[detenido:]
        
        print('Working with dataframe ', date, sets)
        for row in range(len(df)):
    
            p,z,n,m = positions(archivo = df['cif'][row], dist = dist)

            radii = np.nan_to_num(X[row,:,1]/S[row,:,0])
    
            if row == 0:
                r = rij(mult=m,p=p,zero=z,dist=dist, radii = radii)    
                r = np.expand_dims(r,axis=0)
                
            else:
                try:
                    r_temp = rij(mult=m,p=p,zero=z,dist=dist, radii = radii)    
                    r_temp = np.expand_dims(r_temp,axis=0)
                except:
                    with open('problematic_samples_' + sets + '_false_' + date +  '.txt','a') as f:
                        f.write('Check row ' + str(row) + ' in database ' + \
                                date + '_' + sets)
                        f.write('\n')
                        f.close()
                    r_temp=np.zeros((1,r.shape[1],r.shape[2]))
                    print('There was a problem with row ' + str(row) + \
                          ' in ' + date + '_' + sets +' for false')
                r = np.concatenate((r,r_temp))
            np.save('Xrij_' + sets + '_false_' + date, r)           
            #if row%100 == 0:
            print(row)
        
        print(time.time()-start)   
'''
