# -*- coding: utf-8 -*-
"""
Created on Fri Mar 16 12:24:33 2018

@author: iG
"""

import keras.layers as layers
import keras.models as models
import keras.utils as kutils
import keras.callbacks as callbacks
import keras.optimizers as optimizer
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support as PRFS

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import itertools
import time
import copy
import os


#os.environ["MKL_THREADING_LAYER"] = "GNU"


np.random.seed(10)

def lab2symm(sgnum='001',label='a'):
    wyck_dic=np.load('WyckoffSG_dict.npy').item()['wycksym']
    return wyck_dic[sgnum].get(label)

def create_dictionary(file='dictionary'):
    
    """
    Parameters:
        file: a txt-file which contains the spacegroups with their symmetry-site
        -occupations that define a structure. This file is transformed into a 
        python dictionary
        
    Return:
        diccio: Python dictionary
    """
    
    start=time.time()
    f=list(filter(None,open(str(file)+'.txt','r').read().split('\n')))

    sg_ikeys=[f.index(sg) for sg in f if 'spacegroup' in sg]+[len(f)]
    sg_keys=[str(int(sg.split(':')[1])).zfill(3) for sg in f if 'spacegroup' in sg]
    
    diccio={}

    for item in range(len(sg_ikeys)-1):
        text=f[sg_ikeys[item]+1:sg_ikeys[item+1]]
        option=[text.index(i) for i in text if 'option' in i]+[len(text)]

        dicc_1={}
        for inneritem in range(len(option)-1):
            innertext=text[option[inneritem]+1:option[inneritem+1]]
                    
            values=[]
            for letra in innertext:
                
                temp_values=[i for j in [s.split(',') for s in [letra.split(':')[1].replace(' ','')]] for i in j]
                values=values+temp_values
            dicc_1[inneritem]=[lab2symm(sgnum=sg_keys[item],label=j) for j in sorted(values)]
        
        diccio[sg_keys[item]]=dicc_1
    print('Crystal structure definition text file already converted into a Python dictionary in', 
          round(time.time()-start,2),' s')    
    return diccio

def ctrl_dictionary(archivo='model_control_file'):

    """
    Parameters: A txt - file which has all the characteristics of the 
                trainings of neural networks. This txt - file is converted into
                a python dictionary for being used in entrenador - function.
    Return: A dictionary for entrenador function
    """
    
    f=list(filter(None,open(str(archivo)+'.txt','r').read().split('\n')))

    sg_ikeys=[f.index(sg) for sg in f if 'NAME' in sg]+[len(f)]
    
    diccio={}
    for item in range(len(sg_ikeys)-1):
        text = f[sg_ikeys[item]:sg_ikeys[item+1]]
        key = [entry.split(':')[0] for entry in text]
        value = [entry.split(':')[1] for entry in text]
        diccio[item] = {k:v for k,v in zip(key,value)}

    return diccio

def create_collection(df=pd.DataFrame(), sites=-1, elements=-1,
                    maxatoms=-1, dictionary='diccionario'):
    """
    Parameters: 
        database: A pickle file which has information about the space group
                    and the occupied sites in the structure. 
                    It must be specified with extension
        dictionary: A txt - file which contains the spacegroups and the symmetry
                    sites which define a given structure.
    Returns:
        final_data: A pandas DataFrame which contains True and False samples.
    """
    
    
    diccionario = create_dictionary(file=str(dictionary))
    
    sitios=max([len(j) for i in diccionario.values() for j in i.values()])
    #print(sitios)
    start=time.time()
    #print('Loading database. This may take some time...')
    #df = pd.read_pickle(database)
    #print('Database loaded. This process took ',np.round(time.time()-start,2))
   
    if sites == -1:
        df = df[df['sitios'] <= sitios].reset_index(drop=True)
    else:
        df = df[df['sitios'] <= sites].reset_index(drop=True)
    
    if maxatoms != -1:
        df = df[df['atoms'] <= maxatoms].reset_index(drop=True)
    '''
    if elements != -1:
        data = df[df['elements'] >= elements].reset_index(drop=True)
    else:
    '''
    data = df
    
       
    trydata=data.loc[data['sgnum'].isin([int(i) for i in \
                     list(diccionario.keys())])].reset_index(drop=True)
    wyck_dic=np.load('WyckoffSG_dict.npy').item()['wycksym']

    trydata['labels'] = [[list(item.keys())[0] for item in \
           list(trydata['WyckOcc'][row].values())] for row in range(len(trydata))]

    trydata['symmetry'] = [[wyck_dic[str(trydata['sgnum'][row]).zfill(3)].get(letra) \
           for letra in trydata['labels'][row]] for row in range(len(trydata))]


    target=[]
    for row in range(len(trydata)):
    
        spacegroup = str(trydata['sgnum'][row]).zfill(3)
        comparador=list(diccionario[spacegroup].values())
    
        if trydata['symmetry'][row] in comparador:
            target.append(True)
        else:
            target.append(False)
        
    target=pd.Series(target,name='target')
    trydata=trydata.join(target)
    print('True data identified')
    
    data_true=trydata[trydata['target'] == True].reset_index(drop=True)
    exclude_cif=trydata[trydata['target'] == True]['cif']
    sitios=list(set(data_true['sitios']))
    data_rem=df.loc[~df['cif'].isin(exclude_cif)].reset_index(drop=True)
    
    final_data=data_true[['cif','formula','WyckOcc','sgnum','sitios',
                          'atoms','elements','target']]
    for item in sitios:
        data_temp=data_rem[data_rem['sitios'] == item].reset_index(drop=True)
        
        cantidad=len(data_true[data_true['sitios'] == item])

        vector=np.random.permutation(np.random.permutation(np.random.permutation(np.arange(len(data_temp)))))[:cantidad]
        data_temp=data_temp.take(vector)
        data_temp['target']=False
        final_data=pd.concat((final_data, data_temp), ignore_index=True)
        
    print('Crystal compounds collection to train the ANNs created in',
          round(time.time()-start,2),' s')
    
    final_data = final_data[['cif','formula','sgnum','WyckOcc','sitios',
                             'elements','atoms','target']]
    if elements != -1:
        false_positive = final_data[final_data['target'] == True][final_data['elements'] < elements].index
        if len(false_positive) != 0:
            numcif_todrop = len(false_positive)
            false_todrop = np.random.choice(final_data[final_data['target'] == False].index,numcif_todrop)
            idx_todrop = np.append(false_positive, false_todrop)
            
            final_data = final_data.drop(final_data.index[idx_todrop])
            final_data = final_data.reset_index(drop=True) 

    final_data.to_csv('compounds_collection.csv', index=None)
        
    return final_data

def raw_features_extractor(df_coleccion=pd.DataFrame(), sites=-1, elements = -1, maxatoms= -1,
                  dictionary='diccionario', datos_elementos=pd.DataFrame(),  
                  site_normalization=False):
    
    """
    Parameters: 
        database: A pickle file which contains information about the spacegroups and
                    the occupied symmetry sites. This must be specified with extension.
        dictionary: A txt - file which contains the symmetry site occupation for each
                    spacegroup where a given structure crystallizes. This must not be
                    specified with extension
        sites: Constriction to choose structures with a maximum amount of sites. 
                By default, there is not a constriction
        features: A csv - file which contains the features to be use for each present 
                element in the sites of the structure.
        include: if this is equal to 'atoms', info about atoms per occupied sites is
                included in the features. If this is 'mult', multiplicity of occupied
                sites is included in the features. If empty, i.e. '', nothing of above 
                mentioned is included
    Returns:
        X: A matrix of samples x sites x features.
        y: A True - False vector
        s: A matrix of samples x multiplicity for each site.
        df: A pandas DataFrame with True/False values.
    """
    
    #df=create_collection(database=database,sites=sites, elements=elements, maxatoms=maxatoms, 
    #                   dictionary=dictionary)
    
    start=time.time()
    
    #datos=pd.read_csv(features)
    datos=datos_elementos.fillna(-1)

    dicc=dict(datos[['Symbol','Z']].values)

    dicc['D']=1
    dicc['Bk']=97
    dicc['Cf']=98
    dicc['Es']=99
    dicc['Fm']=100
    dicc['Md']=101
    dicc['No']=102
    dicc['Lr']=103
    
    max_sitios = max(df_coleccion['sitios'].values)

    df=df_coleccion[df_coleccion['sitios'] <= max_sitios].reset_index(drop=True)
    
    X=np.zeros((len(df),max_sitios,104))
    y=np.zeros((len(df),1))
    mult=np.zeros((len(df),max_sitios))
    wyckmul=np.load('WyckoffSG_dict.npy').item()['wyckmul']
    
    todelete = list() 
    
    for row in range(len(df)):
        
        item=df['WyckOcc'][row]
        sitios=list(item.values()) #Diccionario de elementos con fracciones de ocupación en ese sitio     
        sitocc=np.zeros((len(sitios),104)) #Vector para 104 elementos de la tabla periódica
        spacegroup = str(df['sgnum'][row]).zfill(3)
        
        try:
        
            s=[int(wyckmul[spacegroup][i]) for j in [list(item.keys()) for item in \
               sitios] for i in j]
        
        except:
            print('There exists an error concerning with the space group number of CIF ', df['cif'][row],'\n')
            print('Please check in www.crystallography.net to provide the correct space group number of that CIF',
                  '\n','\n')
            spacegroup=input('Give me the correct spacegroup:'+'\n'+'\n')
            s=[int(wyckmul[spacegroup][i]) for j in [list(item.keys()) for item in \
               list(df['WyckOcc'][row].values())] for i in j]
        
        occs=[]
        for i in range(len(sitios)):

            for j in list(sitios[i].values()):
                
                ocupacion=np.array(list(j.values()))
                llaves=[llave.replace('+','').replace('-','').replace('1',
                        '').replace('2','').replace('3','').replace('4',
                                   '') for llave in np.array(list(j.keys()))]
                llaves=[llave.replace('.','') for llave in llaves]
                llaves=[llave.replace('5','').replace('6','').replace('7',
                        '').replace('8','').replace('9','').replace('0',
                                   '') for llave in llaves]
                vector=np.zeros((1,104))
                occs=[sum(ocupacion)]+occs
                
                try:
                    
                    idx=[dicc[k] for k in llaves]
                    
                except:
                    print('The compound with the cif ', df['cif'][row], ' will be deleted')
                    print('The database will be updated')
                    todelete += [row]
                    
                    
                for k in idx:
                    vector[0][k-1] = ocupacion[idx.index(k)]
                        
                
            sitocc[i]=vector
    
        while sitocc.shape[0] != max_sitios:
            sitocc=np.concatenate((np.zeros((1,104)),sitocc))
            s=[0]+s
        
        X[row,:,:]=sitocc
        y[row]=df['target'][row]
        mult[row]=s
    
    S = np.expand_dims(mult,axis=2)
    features=datos.iloc[:,2:].values
    x=X[:,:,:96]
    
    fracsum = np.expand_dims(np.sum(x,axis=2), axis=2)
    
    if site_normalization == True:
        x = np.nan_to_num(x/fracsum)
        
    x=np.dot(x,features) 
    
    x = np.delete(x, todelete,axis=0)
    y = np.delete(y, todelete,axis=0)
    S = np.delete(S, todelete,axis=0)
    fracsum = np.delete(fracsum, todelete,axis=0)
    df = df.drop(df.index[todelete]).reset_index(drop=True)
    
    print('Atomic radii and electronegativities for each Wyckoff site extracted in',
          round(time.time()-start,2),' s')   
    
    np.save('raw_features', x)
    np.save('output_values', y)
    np.save('multiplicities', S)
    np.save('occupation_fractions', fracsum)
    
    if len(todelete) != 0:
        df.to_csv('compounds_collection.csv', index=None)
    
    return x, y, S, fracsum, df

def inout_creator(df = pd.DataFrame(), features='datosrahm.csv',
                  site_normalization=False):
    
    """
    Parameters: 
        df:  A pandas DataFrame which contains information about the spacegroups and
                    the occupied symmetry sites. This must be specified with extension.
        features: A csv - file which contains the features to be use for each present 
                element in the sites of the structure.
        site_normalization: if the vacancies in a Wyckoff site are not considered. 
                            If False, vacancies are considered
    Returns:
        X: A matrix of samples x sites x features.
        fracsum: A matrix of samples x sites x occupation
        df: A pandas DataFrame with True/False values.
    """
    
    df = df
   
    start=time.time()
    
    datos=pd.read_csv(features)
    datos=datos.fillna(-1)

    dicc=dict(datos[['Symbol','Z']].values)

    dicc['D']=1
    dicc['Bk']=97
    dicc['Cf']=98
    dicc['Es']=99
    dicc['Fm']=100
    dicc['Md']=101
    dicc['No']=102
    dicc['Lr']=103
    
    max_sitios = max(df['sitios'].values)
    
    X=np.zeros((len(df),max_sitios,104))

    mult=np.zeros((len(df),max_sitios))
    wyckmul=np.load('WyckoffSG_dict.npy').item()['wyckmul']
    
    todelete = list()
    
    for row in range(len(df)):
        item=df['WyckOcc'][row]
        sitios=list(item.values()) #Diccionario de elementos con fracciones de ocupación en ese sitio     
        sitocc=np.zeros((len(sitios),104)) #Vector para 104 elementos de la tabla periódica
        spacegroup = str(df['sgnum'][row]).zfill(3)
        
        try:
        
            s=[int(wyckmul[spacegroup][i]) for j in [list(item.keys()) for item in \
               sitios] for i in j]
        
        except:
            print(row)
            print('There exists an error concerning with the space group of CIF ', df['cif'][row],'\n')
            print('Please check in www.crystallography.net to provide the correct space group number of that CIF',
                  '\n','\n')
            spacegroup=input('Give me the correct spacegroup:'+'\n'+'\n')
            s=[int(wyckmul[spacegroup][i]) for j in [list(item.keys()) for item in \
               list(df['WyckOcc'][row].values())] for i in j]
        
        occs=[]
        for i in range(len(sitios)):

            for j in list(sitios[i].values()):
                
                ocupacion=np.array(list(j.values()))
                llaves=[llave.replace('+','').replace('-','').replace('1',
                        '').replace('2','').replace('3','').replace('4',
                                   '') for llave in np.array(list(j.keys()))]
                llaves=[llave.replace('.','') for llave in llaves]
                llaves=[llave.replace('5','').replace('6','').replace('7',
                        '').replace('8','').replace('9','').replace('0',
                                   '') for llave in llaves]
                vector=np.zeros((1,104))
                occs=[sum(ocupacion)]+occs
                
                try:
                    
                    idx=[dicc[k] for k in llaves]
                
                except:
                    print('The compound with the cif ', df['cif'][row], ' will be deleted')
                    print('The database will be updated')
                    todelete += [row]
                    
                for k in idx:
                    vector[0][k-1] = ocupacion[idx.index(k)]
                        
            
            sitocc[i]=vector
        
            
        while sitocc.shape[0] != max_sitios:
            sitocc=np.concatenate((np.zeros((1,104)),sitocc))
            s=[0]+s
        
        X[row,:,:]=sitocc
        mult[row]=s
    
    features=datos.iloc[:,2:].values
    x=X[:,:,:96]
    
    fracsum = np.expand_dims(np.sum(x,axis=2), axis=2)
    
    if site_normalization == True:
        x = np.nan_to_num(x/fracsum)
        
    x=np.dot(x,features)    
    
  
    x = np.delete(x, todelete,axis=0)
    df = df.drop(df.index[todelete]).reset_index(drop=True)
    

    print('inout_creator lasted ',round(time.time()-start,2),' s')    
    return x, fracsum, df

def compute_quotients(X = np.zeros((1,1,2))):
    """
    Returns the atomic radii pair quotients and the atomic radii 
    pair sum - quotients as a numpy array. Thjs is the first part of
    all the features used to train the ANNs
    Parameters:
        X: A numpy array, which is created with the function raw_features_extractor
    Returns:
        X: A numpy array of dimension [samples,1,features]
    """
    
    start=time.time()
    rad = X[:,:,1]

    X = np.reshape(X,(X.shape[0],1,X.shape[1]*X.shape[2]))

    drad = np.asarray([[item[0]/item[1] if item[1] != 0 else 0 for item in list(itertools.combinations(rad[sample],2))] \
                        for sample in range(X.shape[0])])

    dradsum = np.asarray([[item[0]/item[1] if item[1] != 0 else 0 for item in itertools.combinations([ \
                       item[0]+item[1] for item in list(itertools.combinations(rad[sample],2))], 2)] \
                       for sample in range(drad.shape[0])])
    
    drad = np.reshape(drad,(drad.shape[0],1,drad.shape[-1]))
    drads = np.reshape(dradsum,(dradsum.shape[0],1,dradsum.shape[-1]))

    X = np.concatenate((drad,drads), axis=2)
    print('Geometric and packing factors computed in', round(time.time()-start,2),' s')
    np.save('X', X)
    
    return X


def append_local_functions(X = np.zeros((1,1,1)), df = pd.DataFrame(), local_function='fij_2.0_25_diccio'):
    """
    Returns the features with the local functions. In case the local function
    does not exist for a sample in the collection, this is deleted and the
    collection is updated.
    Parameters:
        X: The numpy array created with compute_quotients
        df: The pandas DataFrame created with raw_features_extractor
        local_function: The numpy dictionary having the local function to use.
    Returns:
        X: The numpy array with all neccesary features for the ANNs.
        df: The pandas DataFrame updated.
    """
    start = time.time()
    print('The dictionary ' + local_function + ' will be used for local functions')
    fij = np.load(local_function + '.npy').item()
    
    delrow = list()
    n = np.max(df['sitios'])
    
    f = np.zeros((df.shape[0],n,n))
    
    for row in range(df.shape[0]):
        if df['cif'][row] not in fij.keys():
            delrow += [row]
        else:
            loc = fij[df['cif'][row]]
            s = loc.shape[1]
            f[row,-s:,-s:] = loc
    
    if len(delrow) != 0:
        
        print('The compounds with the next cifs will be deleted since ',
              'their local functions are not currently available')
        print([df['cif'][i] for i in delrow])
        print('The compound collection will be updated')
        
        totake = [i for i in range(df.shape[0]) if i not in delrow]
        df = df.take(totake).reset_index(drop=True)
        X = X[totake]
        f = f[totake]
        df.to_csv('compounds_collection.csv', index=None)
    
    fn = np.zeros((f.shape[0], f.shape[1], f.shape[2] - 1))        
    for item in range(f.shape[0]):
        delec = f[item]
        delec = delec[~np.eye(delec.shape[0], dtype=bool)].reshape(delec.shape[0],-1)
        fn[item] = delec
    
    f = fn
    f = f.reshape((f.shape[0], 1, f.shape[1]*f.shape[2]))
    
    X = np.concatenate((X,f), axis = 2)
    print('Local functions appended to features in ', round(time.time()-start,2),' s')
    return X, df

def split_collection(X = np.zeros((1)), df = pd.DataFrame(), frac = 0.15):
    """
    Splits the complete compounds collection in two sets:
        one for training and cross - validation and another for testing.
    Parameters:
        X: A numpy array with the features of all compounds in the collection to split.
        df: A pandas DataFrame with all the compounds.
        frac: The fraction reserved to create the test - set. If frac equals zero,
        arguments are passed to returns without modifications.
    Returns:
        Xtraval: A numpy array with the features of the compounds in the 
        training and cross - validation sets. This is saved as Xtraval.
        Xtest: A numpy array with the features of the compounds in the 
        test set. This is saved as Xtest.
        dftraval: A panda DataFrame with the compounds in the training and
        cross validation sets. This is saved as dftraval.
        dftest: A panda DataFrame with the compounds in the test set. This is 
        saved as dftest.
    """
    if frac != 0:
        
        tidx = df[df['target'] == True].index
        fidx = df[df['target'] == False].index
    
        ttest = np.random.choice(tidx, size = int(frac*len(tidx)), replace = False)
        ftest = np.random.choice(fidx, size = int(frac*len(fidx)), replace = False)

        ttest = [i for i in ttest]
        ftest = [i for i in ftest]

        ttraval = [i for i in tidx if i not in ttest]
        ftraval = [i for i in fidx if i not in ftest]

        traval = ttraval + ftraval
        test = ttest + ftest

        Xtraval = X[traval]
        Xtest = X[test]

        dftraval = df.take(traval).reset_index(drop=True)
        dftest = df.take(test).reset_index(drop=True)

        np.save('Xtraval', Xtraval)
        np.save('Xtest', Xtest)

        dftraval.to_csv('dbtraval.csv', index=None)
        dftest.to_csv('dbtest.csv', index=None)
    
    else:
        Xtraval = X
        dftraval = df
        Xtest = None
        dftest = None
    
    return Xtraval, Xtest, dftraval, dftest

def plotgraph(readfile='archivo.csv', outfiles='',cost_function='Categorical Cross Entropy'):
    
    plt.rcParams['figure.figsize']=(12,9)
    
    df=pd.read_csv(str(readfile), header=None)
    df=df.rename(columns={0:'cost_train', 1:'acc_train', 2:'cost_test', 3:'acc_test'})

    plt.figure(1)
    plt.title('Accuracy', fontsize=20, fontweight='bold')
    train=plt.scatter(np.arange(1,len(df)+1,1), df['acc_train'].values*100, color='red',
                      marker='x',s=100)
    test=plt.scatter(np.arange(1,len(df)+1,1), df['acc_test'].values*100, color='blue',
                     marker='o',s=100)
    plt.legend([train,test],['Training set', 'Test set'], fontsize=18)
    plt.xlabel('Epochs',fontsize=20, fontweight='bold')
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.ylabel('Accuracy %', fontsize=20, fontweight='bold')
    plt.annotate(str(round(df['acc_train'].values[-1]*100,2))+' %', xy=(len(df),int(df['acc_train'].values[-1]*100)), 
                 xytext=(len(df),int(df['acc_train'].values[-1]*100-1)), fontsize=14, fontweight='bold')
    plt.annotate(str(round(df['acc_test'].values[-1]*100,2))+' %', xy=(len(df),int(df['acc_test'].values[-1]*100)),
                 xytext=(len(df),int(df['acc_test'].values[-1]*100+2)), fontsize=14, fontweight='bold')
    plt.savefig('Accuracy'+'_'+str(outfiles)+'.png')
    
    plt.figure(2)
    plt.title(str(cost_function), fontsize=20, fontweight='bold')
    train=plt.scatter(np.arange(1,len(df)+1,1), df['cost_train'].values, color='red',
                      marker='x',s=100)
    test=plt.scatter(np.arange(1,len(df)+1,1), df['cost_test'].values, color='blue',
                     marker='o',s=100)
    plt.legend([train,test],['Training set', 'Test set'], fontsize=18)
    plt.xlabel('Epochs',fontsize=20, fontweight='bold')
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.ylabel('Cost function', fontsize=20, fontweight='bold')
    plt.annotate(str(round(df['cost_train'].values[-1],4)), xy=(len(df),df['cost_train'].values[-1]), 
                 xytext=(len(df),1.1*df['cost_train'].values[-1]), fontsize=14, fontweight='bold')
    plt.annotate(str(round(df['cost_test'].values[-1],4)), xy=(len(df),df['cost_test'].values[-1]),
                 xytext=(len(df),1.1*df['cost_test'].values[-1]+0.05), fontsize=14, fontweight='bold')
    plt.savefig('Cost_function'+'_'+str(outfiles)+'.png')
    plt.close('all')

    return 

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function is taken from http://scikit-learn.org/stable/auto_examples/
    model_selection/plot_confusion_matrix.html#
    sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py
    
    
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.rcParams['figure.figsize']=(12,9)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        #print("Normalized confusion matrix")
    #else:
        #print('Confusion matrix, without normalization')

    #print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=20, fontweight='bold')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, fontsize=16)
    plt.yticks(tick_marks, classes, fontsize=16)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black",
                 fontsize=18, fontweight='bold')

    plt.ylabel('True label', fontsize=18, fontweight='bold')
    plt.xlabel('Predicted label', fontsize=18, fontweight='bold')
    plt.tight_layout()
    
    return


def modelo(sitios=1, hidden_layers=[1], activation='tanh',features=1, 
           beta_1=0.9, beta_2=0.999,lr=0.001, decay=0, dropout=0):
    
        
    inputs=['in_'+str(i) for i in range(sitios)]
    hidden_output=['hout_'+str(i) for i in range(sitios)]

    for i,h in zip(inputs, hidden_output):
        globals() [i] = layers.Input(shape=(features,))
        vmiddle = layers.Dense(hidden_layers[0], 
                               kernel_initializer='random_uniform')(globals() [i])
        vmiddle = layers.Activation(activation)(vmiddle)
        vmiddle = layers.Dropout(dropout)(vmiddle)
    
        if len(hidden_layers) != 1:
    
            for item in range(1,len(hidden_layers)):
                vmiddle = layers.Dense(hidden_layers[item], 
                                       kernel_initializer='random_uniform')(vmiddle)
                vmiddle = layers.Activation(activation)(vmiddle)
                vmiddle = layers.Dropout(dropout)(vmiddle)
            vmiddle =layers.Dense(1, kernel_initializer='random_uniform')(vmiddle)
            vexit =layers.Activation('sigmoid')(vmiddle)
        
        else:
            vmiddle =layers.Dense(1, kernel_initializer='random_uniform')(vmiddle)
            vexit =layers.Activation('sigmoid')(vmiddle)

    model = models.Model(inputs=[globals() [i] for i in inputs], outputs=vexit)
    model.compile(loss='binary_crossentropy', 
                  optimizer=optimizer.Adam(beta_1=beta_1, beta_2=beta_2, lr=lr, decay=decay,), 
                  metrics=['accuracy'])
    
    return model

def ajuste(model, X, Y,epochs=300,batch_size=16,test_frac=0.30, saveas='modelo_nn', verbose=1):
    '''
    Parametros:
        model: Valor de salida de la funcion modelo
        X: Matriz de ejemplos x rasgos
        Y:Matriz de ejemplos x numero_de_etiquetas
        epochs: epochs
        batch_size: batch_size
        test_frac: Fraccion del conjunto completo a usar como conjunto de prueba.
    Regresa:
        Funcion de perdida y precision(Accuracy) despues de cada epoca.
    '''
    modelCheckpoint=callbacks.ModelCheckpoint(str(saveas)+'.h5', monitor='val_loss', 
                                                    verbose=0, save_best_only=True, mode='auto')
    history = callbacks.History()
    data = model.fit(X,Y,validation_split=test_frac, epochs=epochs,batch_size=batch_size,
                     callbacks=[modelCheckpoint,history],shuffle=True, verbose=verbose)

    kutils.plot_model(model,to_file=str(saveas)+'.png', show_shapes=True, show_layer_names=True)
    
    #np.save(str('pruebann.png').split('.')[0]+'_params',modelo.get_weights())
    
    """ Creacion del archivo csv """
    acc_log = data.history['acc']
    val_acc_log = data.history['val_acc']
    loss_log = data.history['loss']
    val_loss_log = data.history['val_loss']
    acc_log = np.array(acc_log)
    val_acc_log = np.array(val_acc_log)
    loss_log = np.array(loss_log)
    val_loss_log = np.array(val_loss_log)
    mat = np.vstack((loss_log, acc_log, val_loss_log, val_acc_log))
    mat = np.transpose(mat)
    dataframe1 = pd.DataFrame(data=mat)
    dataframe1.to_csv(str(saveas)+'.csv', sep=',', header=False, float_format='%.7f', index=False)
    
    return data, dataframe1, model

def test_models(directorio=''):
    """
    This function tests all models once the training finished.
    Parameters:
        directorio: A string with the name of the directory where the models are.
    Returns:
        a txt - file with the name test_results, which is inside the given directory.
    """
    
    print('The trained models will be tested now')
    start = time.time()
    
    busqueda = "ls " + directorio + "/*.h5 > model_names.txt"

    os.system(busqueda)

    X = np.load(directorio + '/Xtest.npy')
    diccio = np.load(directorio + '/feature_standarisation.npy').item()
    y = pd.read_csv(directorio + '/dbtest.csv')['target'].values

    X = (X - diccio['mean'])/diccio['std']
    x = np.reshape(X,(X.shape[0],X.shape[2]))
    
    with open('model_names.txt','r') as f:
        for line in f:
            modelo = models.load_model(line[:len(line)-1])
            nombre = line.split('/')[1]
            outpred = modelo.predict(x)
            prediction = outpred >= 0.5
            
            cost = -(np.dot(y,np.log10(outpred)) + \
                     np.dot((1-y),np.log10(1-outpred)))/y.shape[0]
            precision,recall,fscore,support = PRFS(y, prediction)
            
            with open(directorio + '/test_results.txt','a') as tr:
                tr.write(nombre + '\n')
                tr.write('cost function: '+str(cost[0])+'\n')
                tr.write('samples: '+str(support)+'\n')
                tr.write('precision: '+str(np.round(precision*100,2))+'\n')
                tr.write('recall: '+str(np.round(recall*100,2))+'\n')
                tr.write('f1-score: '+str(np.round(fscore*100,2))+'\n')
                tr.write('\n')
                tr.close()
    
    print('The test of all trained models lasted ', round(time.time()-start,2),' s')
    os.system('rm model_names.txt')
    
    return

def test_all_false(directorio = str(), database = 'red_cod-db.pkl', 
                   local_function = 'fij_2.0_25_diccio'):
    """
    This function tests all models once the training finished.
    Parameters:
        directorio: A string with the name of the directory where the models are.
        database: a pickle file containing the entire database
        local_function: a numpy dictionary with the local functions to append.
    Returns:
        a txt - file with the name test_with_all_false, which is inside the given directory.
    """
    df = pd.read_pickle(database)
    collection = pd.read_csv(directorio + '/compounds_collection.csv')
    
    cifs = [i for i in collection['cif']]
    maxsites = np.max(collection['sitios'])
    
    df = df[df['sitios'] > 0][df['sitios'] <= maxsites].reset_index(drop=True)
    df = df.loc[~df['cif'].isin(cifs)].reset_index(drop=True)
    
    x, _, df = inout_creator(df=df, site_normalization=False)
    
    x = compute_quotients(X=x)
    x, df = append_local_functions(X = x,df=df)
    
    busqueda = "ls " + directorio + "/*.h5 > model_names.txt"
    os.system(busqueda)
    
    diccio = np.load(directorio + '/feature_standarisation.npy').item()
    
    X = (x - diccio['mean'])/diccio['std']
    x = np.reshape(X,(X.shape[0],X.shape[2]))
    print('Testing with ',df.shape[0],'false samples')
    with open('model_names.txt','r') as f:
        for line in f:
            modelo = models.load_model(line[:len(line)-1])
            nombre = line.split('/')[1]
            
            outpred = modelo.predict(x)
            prediction = outpred >= 0.5
            df['y_pred'] = np.ravel(prediction)
                
            with open(directorio+'/test_with_all_false.txt','a') as tr:
                tr.write(nombre + '\n')
                
                for sitios in range(1, max(df['sitios']) + 1):
                
                    acc = df[df['sitios'] == sitios][df['y_pred'] == False].shape[0]
                    miniset = df[df['sitios'] == sitios].shape[0]
                    percent = round(100*acc/miniset,2)
                
                    
                    tr.write('With '+ str(sitios) + ' sites:' + str(percent) +\
                             '(' + str(miniset) + ' samples)' + '\n')
                tr.close()
    return

def entrenador(X = np.zeros((1,1,1)), df = pd.DataFrame(),
               control_file='model_control_file', dictionary='structure_dictionary',
               verbose=1, directory_tosave='no_title'):

    """
    This is the function to train neural networks.
    
    Parameters: 
        database: A pickle file which contains information about the spacegroups and
                    the occupied symmetry sites. This must be specified with extension.
        sites: Constriction to choose structures with a maximum amount of sites. 
                By default, there is not a constriction.
        dictionary: A txt - file which contains the symmetry site occupation for each
                    spacegroup where a given structure crystallizes. This must not be
                    specified with extension
        features: A csv - file which contains the features to be use for each present 
                element in the sites of the structure.
        control_file: A txt - file which has all the characteristics for the training of
                    each neural network.
        verbose: Verbosity during training. 0 stands for silent mode, 2 is for
                extreme verbosity and 1 is just enough verbosity.
        include: if this is equal to 'atoms', info about atoms per occupied sites is
                included in the features. If this is 'mult', multiplicity of occupied
                sites is included in the features. If empty, i.e. '', nothing of above 
                mentioned is included
    Returns:
        A h5 - model with the trained neural network, plots of accuracy and cost function,
        as well as crude and normalized confusion matrices, a csv - file with data about
        accuracy and cost function for training and test sets for all epochs and a txt - file
        with information about precision - recall - F1 score for each model.
        
    """
    
    start_main=time.time()

    Y = df['target'].values
    class_names=list(set(df['target']))
    
    sitios=X.shape[1]
    features=X.shape[2]
    
    Xor=copy.deepcopy(X)
    X,y = shuffle(X,Y,random_state=0)
    
    x={}
    xor={}
    
    for sitio in range(sitios):
        x[sitio] = X[:,sitio,:]
        xor[sitio] = Xor[:,sitio,:]
    
    directorio = directory_tosave

    os.system('mkdir ' + directorio)
    os.system('mv compounds_collection.csv ' + directorio +'/')
    os.system('mv multiplicities.npy ' + directorio +'/')
    os.system('mv occupation_fractions.npy ' + directorio +'/')
    os.system('mv output_values.npy ' + directorio +'/')
    os.system('mv raw_features.npy ' + directorio +'/')
    os.system('mv X*.npy ' + directorio +'/')
    os.system('mv db*.csv ' + directorio +'/')
    os.system('mv feature_standarisation* ' + directorio +'/')
    
    
    ctrl_diccio = ctrl_dictionary(archivo=control_file)
    print('\n')
    print('*************************************************************'+
          '*************************************************************'+
          '*************************************************************'+
          '*************************************************************')
    print('ANNs TRAINING WILL START NOW.')
    print('\n')
    print('There are ',len(ctrl_diccio.keys()),' ANNs to train')
    
    for item in list(ctrl_diccio):
        print('Training ', item+1,'/',len(ctrl_diccio.keys()))
        diccionary = ctrl_diccio[item]
    
        hidden_layers=[float(x) for x in diccionary['HIDDEN_LAYERS'].split(",")]
        epochs=int(diccionary['EPOCHS'])
        batch_size=int(diccionary['BATCH_SIZE'])
        test_frac=float(diccionary['VAL_FRAC'])
        cost_function=diccionary['COST_FUNCTION']
        learning_rate=float(diccionary['LEARNING_RATE'])
        beta_1=float(diccionary['BETA_1'])
        beta_2=float(diccionary['BETA_2'])
        decay=float(diccionary['DECAY'])
        dropout=float(diccionary['DROPOUT'])
        activation=diccionary['ACTIVATION']
        name=diccionary['NAME']
        
        hidden_layers = np.asarray(hidden_layers)*features
        hidden_layers = [int(x) for x in hidden_layers]
        
        model = modelo(sitios=sitios, hidden_layers=hidden_layers, activation=activation,
                       features=features, beta_1=beta_1, beta_2=beta_2, lr=learning_rate, decay=decay, 
                       dropout=dropout)
        
        start=time.time()
        data, dataframe, model = ajuste(model, X=[x[i] for i in range(sitios)], Y = y, epochs=epochs, 
                                               batch_size=batch_size, test_frac=test_frac, saveas=name,
                                               verbose=verbose)
        
        print('NN training lasted ',np.round(time.time() - start,2),'s')
        print('\n')
        plotgraph(readfile=name+'.csv', outfiles=name, cost_function=cost_function)
    
        img=Image.open('Cost_function_'+ name + '.png')
        img.show()

        #img=Image.open('Cost_function_'+ name + '.png')
        #img.show()

        y_pred = (model.predict([xor[i] for i in range(sitios)]) > 0.5)
        
        precision, recall, fscore, support = PRFS(df['target'],y_pred)
        cnf_matrix=confusion_matrix(df['target'],y_pred)
        np.save(str(name)+'_cnfmat.npy',cnf_matrix)
        precision = np.round(100*precision,2)
        recall = np.round(100*recall,2)
        fscore = np.round(100*fscore,2)
        
        with open('PRFS_'+str(control_file)+'.txt', 'a') as prfs:
            prfs.write(str(name)+'\n')
            prfs.write('classes: '+str(class_names)+'\n')
            prfs.write('samples: '+str(support)+'\n')
            prfs.write('precision: '+str(precision)+'\n')
            prfs.write('recall: '+str(recall)+'\n')
            prfs.write('f1-score: '+str(fscore)+'\n')
            prfs.write('\n')
            prfs.close()
        #'''    
        plt.figure(1)
        plot_confusion_matrix(cnf_matrix, classes=class_names,
                              title='Confusion matrix, without normalization')
        plt.savefig('cnfmat_'+str(name)+'.png')
        
        plt.figure(2)
        plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                          title='Normalized confusion matrix')
        plt.savefig('normcnfmat_'+str(name)+'.png')
        
        plt.close('all')
        #'''
        os.system('mv *' + name + '* ' + directorio)
    os.system('mv PRFS_' + str(control_file) + '.txt ' + directorio)
    os.system('cp ' + control_file + '.txt ' + directorio)
    os.system('cp ' + dictionary + '.txt ' + directorio)
    
    print('Whole process lasted ', np.round(-start_main+time.time(),2),'s')                
    return 

def intverb(s):
    if len(s) == 0 and type(s) == str:
        s = 1
    else:
        s = int(s)
    return s

def geompackfeatures(n):
    
    radios = list()
    for site in range(1,n+1):
        r = 'r_{' + str(site) + '}'
        radios += [r]

    pairrad = list(itertools.combinations(radios,2))

    s=list()
    geomfac=list()

    for item in pairrad:
        item=list(item)
        s += [item[0] + '+' + item[1]]
        geomfac += ['{' + item[0] + '}{' + item[1] + '}']

    pf_prev=list(itertools.combinations(s,2))
    packfac=list()

    for item in pf_prev:
        item=list(item)
        packfac += ['{' + item[0] + '}{' + item[1] + '}']

    subs = list()
    
    for i in range(1,n+1):
        for j in range(1,n+1):
            if i != j:
                subs += ['f_{' + str(i) + str(j) + '}']
                    
    return geomfac, packfac, subs
