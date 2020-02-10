#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 16:21:23 2019

@author: iG
"""

import pandas as pd
import numpy as np
import itertools

def CrystalMaker(motif = pd.DataFrame(),abc = [1,1,1], 
                 angles=[90,90,90], n=1, saveas='generated_file'):
    
    motif[2]=[float(i) for i in motif[2].values]
    motif[3]=[float(i) for i in motif[3].values]
    motif[4]=[float(i) for i in motif[4].values]
    
    motif.columns = np.arange(len(motif.columns))
    print(motif)
    motif[4]=1*(motif[1].values == 0)+1*(motif[2].values == 0)+1*(motif[3].values == 0)
    
    motif_three=motif[motif[4] == 3].iloc[:,0:4].reset_index(drop=True)
    motif_two=motif[motif[4] == 2].iloc[:,0:4].reset_index(drop=True)
    motif_one=motif[motif[4] == 1].iloc[:,0:4].reset_index(drop=True)
    motif=motif.iloc[:,0:4].reset_index(drop=True)

    volumen=abc[0]*abc[1]*abc[2]*np.sqrt(1-(np.cos(np.deg2rad(angles[0])))**2-(np.cos(np.deg2rad(angles[1])))**2-(np.cos(np.deg2rad(angles[2])))**2+2*np.cos(np.deg2rad(angles[0]))*np.cos(np.deg2rad(angles[1]))*np.cos(np.deg2rad(angles[2])))

    matrix=np.array([[abc[0],abc[1]*np.cos(np.deg2rad(angles[2])),abc[2]*np.cos(np.deg2rad(angles[1]))],
                      [0,abc[1]*np.sin(np.deg2rad(angles[2])),abc[2]*(np.cos(np.deg2rad(angles[0]))-np.cos(np.deg2rad(angles[1]))*np.cos(np.deg2rad(angles[2])))/np.sin(np.deg2rad(angles[2]))],
                      [0,0,volumen/(abc[0]*abc[1]*np.sin(np.deg2rad(angles[2])))]])

    if len(motif_one) != 0:
        
        positions_one=motif_one.replace(0,1).iloc[:,1:].values
        positions_one=pd.DataFrame(positions_one)
        positions_one=positions_one.rename(columns={0:1,1:2,2:3})
        positions_one[0]=motif_one[0]
        motif=motif.append(positions_one, ignore_index=True)
        
    tras0=np.array(list(itertools.product([0,1],repeat=3)))[1:7]
    tras1=np.array(list(itertools.product([0,1],repeat=3)))[1:]
        
    if len(motif_two) != 0:
        for vector in tras0:
            
            positions_two=np.add(motif_two.iloc[:,1:].values,vector)
            data=pd.DataFrame(positions_two)
            data=data.rename(columns={0:1,1:2,2:3})
            data[0]=motif_two[0]
            data=data[[0,1,2,3]]
            data=data[data <= 1].dropna()
            data=data.reset_index(drop=True)
            motif=motif.append(data, ignore_index=True)

    if len(motif_three) != 0:
        for vector in tras1:
            
            positions_three=np.add(motif_three.iloc[:,1:].values,vector)
            positions_three=pd.DataFrame(positions_three)
            data=pd.DataFrame(positions_three)
            data=data.rename(columns={0:1,1:2,2:3})
            data[0]=motif_three[0]
            data=data[[0,1,2,3]]
            data=data[data <= 1].dropna()
            data=data.reset_index(drop=True)
            motif=motif.append(data, ignore_index=True)
    
    if n != 1:
        
        rows=len(motif)
        
        print('Each lattice parameter is increased by '+str(n)+' time(s)'+'\n')

        multiple=[m for m in range(n)]

        traslation=np.array(list(itertools.product(multiple, repeat=3)))[1:]
        
        for vector in traslation:
            data=np.add(motif.iloc[:rows,1:].values, vector)
            data=pd.DataFrame(data)
            data=data.rename(columns={0:1,1:2,2:3})
            data[0]=motif.iloc[:rows,0].values
            data=data[[0,1,2,3]]
            data=data.reset_index(drop=True)
            motif=motif.append(data, ignore_index=True)
                        
        motif=motif.round(4)
        data=data[data <= n].dropna()
        motif=motif.drop_duplicates() 
        motif=motif.reset_index(drop=True)

    positions=np.round(np.matmul(motif.iloc[:,1:].values,matrix),4)
    positions=pd.DataFrame(positions)
    positions=positions.rename(columns={0:'x',1:'y',2:'z'})
    positions['element']=motif[0]
        
    positions['element']=positions['element'].map(lambda x: x.lstrip('+-').rstrip('0123456789'))
    positions['element']=positions['element'].map(lambda x: x.lstrip('0123456789').rstrip('+-'))
    positions['element']=positions['element'].str.replace('\d+','')

    positions=positions[['element','x','y','z']]
    positions=positions.round(4)

    atoms=len(positions)
    
    with open(str(saveas)+'.xyz', 'w') as file:
        file.write(str(atoms)+'\n'+'\n')
        file.write(positions.to_string(header=None, index=None, col_space=0))
        file.close()
    
    
    return positions


