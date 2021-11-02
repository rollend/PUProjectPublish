from operator import mul
import matplotlib.pyplot as plt
import numpy as np
from holoviews import opts
from scipy.signal.ltisys import dfreqresp
from scipy.spatial import Voronoi
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import pandas as pd
import ipywidgets as widgets
from ipywidgets import interact, interact_manual

import seaborn as sns
from kneed import KneeLocator
from PatientGraphPheno import *
from config.bedrock_connection import *
#from patientKG import utils_pickle
import utils_pickle
from holoviews.operation.datashader import datashade, bundle_graph
import holoviews as hv
from holoviews import opts
from datetime import datetime
import re
import plotly.graph_objects as go
from pivottablejs import pivot_ui
from IPython.display import display, HTML
from sklearn.feature_selection import VarianceThreshold
from sklearn import preprocessing
sns.set(style="ticks")

def show_SpellHRG_HRG_Table(HRG,Degree,Readmit): 
    Degree_ReAdmitted_HRG = patientKG.utils_pickle.read("Degree_ReAdmitted_HRG")   
    return Degree_ReAdmitted_HRG.loc[(Degree_ReAdmitted_HRG['SpellHRG'] == HRG)
                                     &(((Degree_ReAdmitted_HRG['Sum_Degree']>=Degree[0])&(Degree_ReAdmitted_HRG['Sum_Degree'] <=Degree[1])))
                                    &(((Degree_ReAdmitted_HRG['ReAdmitted in DAYS']>=Readmit[0])&(Degree_ReAdmitted_HRG['ReAdmitted in DAYS'] <=Readmit[1])))]

    #This below block is for Jupyter-Notebook
    """stats = interact(PatientGraphVisuExplore.show_SpellHRG_HRG_Table,
                    HRG=widgets.Dropdown(options=list(Degree_HRG['SpellHRG'].dropna().unique()))
                    ,Degree=widgets.IntRangeSlider(value=[5,100],
        min=0,
        max=max(Degree_HRG['Sum_Degree']),
        step=1,
        description='Degree:',
        disabled=False,
        continuous_update=False,
        orientation='horizontal',
        readout=True,
        readout_format='d')
    ,Readmit=widgets.IntRangeSlider(value=[-1,30],
        min=-1,
        max=30,
        step=1,
        description='ReAdmitted:',
        disabled=False,
        continuous_update=False,
        orientation='horizontal',
        readout=True,
        readout_format='d'))"""

def plot_SpellHRG_Income_Scatter(HRG,Degree,Readmit):
    Degree_ReAdmitted_HRG = utils_pickle.read("Degree_ReAdmitted_HRG")      
    data=Degree_ReAdmitted_HRG.loc[(Degree_ReAdmitted_HRG['SpellHRG'] == HRG)
                                     &(((Degree_ReAdmitted_HRG['Sum_Degree']>=Degree[0])&(Degree_ReAdmitted_HRG['Sum_Degree'] <=Degree[1])))
                                    &(((Degree_ReAdmitted_HRG['ReAdmitted in DAYS']>=Readmit[0])&(Degree_ReAdmitted_HRG['ReAdmitted in DAYS'] <=Readmit[1])))]
    plt.scatter(data['Sum_Degree'], data['INCOME'], edgecolors='r')
    
    """
    stats = interact(PatientGraphVisuExplore.plot_SpellHRG_Income_Scatter,
                    HRG=widgets.Dropdown(options=list(Degree_HRG['SpellHRG'].dropna().unique()))
                    ,Degree=widgets.IntRangeSlider(value=[5,100],
        min=0,
        max=max(Degree_HRG['Sum_Degree']),
        step=1,
        description='Degree:',
        disabled=False,
        continuous_update=False,
        orientation='horizontal',
        readout=True,
        readout_format='d')
    ,Readmit=widgets.IntRangeSlider(value=[-1,30],
        min=-1,
        max=30,
        step=1,
        description='ReAdmitted:',
        disabled=False,
        continuous_update=False,
        orientation='horizontal',
        readout=True,
        readout_format='d'))"""


def plot_SpellHRG_LOS_Scatter(HRG,Degree,Readmit): 
    Degree_ReAdmitted_HRG = utils_pickle.read("Degree_ReAdmitted_HRG")     
    for item in HRG:
        #print(item)
        data=Degree_ReAdmitted_HRG.loc[(Degree_ReAdmitted_HRG['SpellHRG'] == item)
                                         &(((Degree_ReAdmitted_HRG['Sum_Degree']>=Degree[0])&(Degree_ReAdmitted_HRG['Sum_Degree'] <=Degree[1])))
                                        &(((Degree_ReAdmitted_HRG['ReAdmitted in DAYS']>=Readmit[0])&(Degree_ReAdmitted_HRG['ReAdmitted in DAYS'] <=Readmit[1])))]
        plt.scatter(data['Sum_Degree'], data['Total_LOS'], edgecolors='r')
    
    """ 
    stats = interact(PatientGraphVisuExplore.plot_SpellHRG_LOS_Scatter,
                    HRG=widgets.SelectMultiple(
        options=list(Degree_HRG['SpellHRG'].dropna().unique()),
        value=['WJ06E'],
        #rows=10,
        description='HRG',
        disabled=False
    )
                    #widgets.Dropdown(options=list(Degree_HRG['SpellHRG'].dropna().unique()))
                    ,Degree=widgets.IntRangeSlider(value=[5,100],
        min=0,
        max=max(Degree_HRG['Sum_Degree']),
        step=1,
        description='Degree:',
        disabled=False,
        continuous_update=False,
        orientation='horizontal',
        readout=True,
        readout_format='d')
    ,Readmit=widgets.IntRangeSlider(value=[-1,30],
        min=-1,
        max=30,
        step=1,
        description='ReAdmitted:',
        disabled=False,
        continuous_update=False,
        orientation='horizontal',
        readout=True,
        readout_format='d'))"""

def plot_SpellHRG_Pairplot(HRG,Degree,Readmit):    
    df = pd.DataFrame()
    Degree_ReAdmitted_HRG = utils_pickle.read("Degree_ReAdmitted_HRG") 
    for item in HRG:
        #print(item)
        data=Degree_ReAdmitted_HRG.loc[(Degree_ReAdmitted_HRG['SpellHRG'] == item)
                                         &(((Degree_ReAdmitted_HRG['Sum_Degree']>=Degree[0])&(Degree_ReAdmitted_HRG['Sum_Degree'] <=Degree[1])))
                                        &(((Degree_ReAdmitted_HRG['ReAdmitted in DAYS']>=Readmit[0])&(Degree_ReAdmitted_HRG['ReAdmitted in DAYS'] <=Readmit[1])))]
        df = pd.concat([df,data])
    sns.pairplot(df[df.columns.difference(['ACTIVITY_IDENTIFIER','POD_CODE','ReAdmitted in DAYS'])], hue="SpellHRG")
    
    """
    stats = interact(PatientGraphVisuExplore.plot_SpellHRG_Pairplot,
                    HRG=widgets.SelectMultiple(
        options=list(Degree_HRG['SpellHRG'].dropna().unique()),
        value=['WJ06E'],
        #rows=10,
        description='HRG',
        disabled=False
    )
                    #widgets.Dropdown(options=list(Degree_HRG['SpellHRG'].dropna().unique()))
                    ,Degree=widgets.IntRangeSlider(value=[5,100],
        min=0,
        max=max(Degree_HRG['Sum_Degree']),
        step=1,
        description='Degree:',
        disabled=False,
        continuous_update=False,
        orientation='horizontal',
        readout=True,
        readout_format='d')
    ,Readmit=widgets.IntRangeSlider(value=[-1,30],
        min=-1,
        max=30,
        step=1,
        description='ReAdmitted:',
        disabled=False,
        continuous_update=False,
        orientation='horizontal',
        readout=True,
        readout_format='d'))"""

def plot_SpellHRG_HRG_ICD(HRG,ICD,Degree,Readmit,POD):    
    df = pd.DataFrame()
    Degree_ReAdmitted_HRG = utils_pickle.read("Degree_ReAdmitted_HRG") 
    for item in HRG:
        #print(item)
        if ICD == None:
            data=Degree_ReAdmitted_HRG.loc[(Degree_ReAdmitted_HRG['SpellHRG'] == item)&(Degree_ReAdmitted_HRG['POD_CODE'] == POD)
                                             &(((Degree_ReAdmitted_HRG['Sum_Degree']>=Degree[0])&(Degree_ReAdmitted_HRG['Sum_Degree'] <=Degree[1])))
                                            &(((Degree_ReAdmitted_HRG['ReAdmitted in DAYS']>=Readmit[0])&(Degree_ReAdmitted_HRG['ReAdmitted in DAYS'] <=Readmit[1])))]
        else:
            data=Degree_ReAdmitted_HRG.loc[(Degree_ReAdmitted_HRG['SpellPDiag'] == ICD)&(Degree_ReAdmitted_HRG['SpellHRG'] == item)&(Degree_ReAdmitted_HRG['POD_CODE'] == POD)
                                             &(((Degree_ReAdmitted_HRG['Sum_Degree']>=Degree[0])&(Degree_ReAdmitted_HRG['Sum_Degree'] <=Degree[1])))
                                            &(((Degree_ReAdmitted_HRG['ReAdmitted in DAYS']>=Readmit[0])&(Degree_ReAdmitted_HRG['ReAdmitted in DAYS'] <=Readmit[1])))]
        df = pd.concat([df,data])
    features = ['Sum_Degree','Global_Central', 'Total_LOS', 'INCOME']#,'Turnaround_Degree','DIAG_COUNT']
    # Separating out the features
    x = df.loc[:, features].values
    # Separating out the target
    #y = test.loc[:,['target']].values
    # Standardizing the features
    x = StandardScaler().fit_transform(x)
    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(x)
    
    principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2'])
    #Voronoi at least four points, though clusters not less than 4
    kmeans = KMeans(n_clusters=2)
    #pair = ['INCOME','Total_LOS']
    kmeans.fit(principalDf)
    labels = kmeans.predict(principalDf)
    centroids = kmeans.cluster_centers_    
    #print(centroids)
    v = np.vstack([centroids,[0,0]])
    #print(v)
    vor = Voronoi(principalComponents)

    # plot
    regions, vertices = voronoi_finite_polygons_2d(vor)
    fig = plt.figure(figsize=(10, 10))
    colmap = {1: 'g', 2: 'r', 3: 'b', 4:'y'}
    marker = {1:'circle', 2:'diamond', 3:'dot', 4:'triangle'}
    size = {1:2,2:2,3:2,4:2}
    colors = list(map(lambda x: colmap[x+1], labels))
    markers = list(map(lambda x: marker[x+1], labels))
    sizes = list(map(lambda x: size[x+1], labels))
    #print(principalComponents)
    df['principal component 1'] = principalComponents[:,0]
    df['principal component 2'] = principalComponents[:,1]
    df['color'] = colors
    df['marker'] = markers
    df['sizes'] = sizes
    opts.defaults(opts.Points(padding=0.1, size=8, line_color='black'))
    data ={'x':list(df['principal component 1'])
                    ,'y':list(df['principal component 2'])
                    ,'color':list(df['color'])
                    ,'marker':list(df['marker']) 
                   ,'sizes':list(df['sizes'])}
    
    #hv.Points(data, vdims=['color', 'marker', 'sizes']).opts(color='color', marker='marker', size='sizes')

    plt.scatter(df['principal component 1'], df['principal component 2'], color=colors, alpha=0.5, edgecolor='k')
    #for idx, centroid in enumerate(centroids):
        #plt.scatter(*centroid, color=colmap[idx+1])
    df['labels'] = labels
    #print(list(df['labels'].unique()))
    shape_ = {}
    for item in list(df['labels'].unique()):
        shape_.update({item:[(df[df['labels'] ==item].shape[0]),df[df['labels'] == item]['Sum_Degree'].mean()]})
        print('Complex Degree:',df[df['labels'] == item]['Sum_Degree'].mean())
   
    #print(shape_)
    #print(sorted(shape_.items(), key=lambda x: x[1]))
    
    minor_=sorted(shape_.items(), key=lambda x: x[1])[0][0]
    major_=sorted(shape_.items(), key=lambda x: x[1])[1][0]
    
    #sns.pairplot(df[df['labels'] ==1][df.columns.difference(['ACTIVITY_IDENTIFIER','POD_CODE'])], hue="SpellHRG")
    #for label,x,y in zip(df[df['labels'] == minor_]['ACTIVITY_IDENTIFIER'],df[df['labels'] == minor_]['principal component 1'],df[df['labels'] == minor_]['principal component 2']):
    for label,x,y in zip(df['ACTIVITY_IDENTIFIER'],df['principal component 1'],df['principal component 2']):
    
        label = label        
        plt.annotate(label, (x,y),textcoords="offset points",xytext=(0,10),ha='center', size =20)
    
    test=zip(regions, df['color'])
    for item in test:
       
        polygon = vertices[item[0]]
        #print(region,polygon)
        #print(*zip(*polygon))
        plt.fill(*zip(*polygon), alpha=0.4
                 ,color=item[1]
                )
    plt.xlim(vor.min_bound[0]-0.1, vor.max_bound[0]+0.1)
    plt.ylim(vor.min_bound[1]-0.1, vor.max_bound[1]+0.1)
    print('Minor Complex Degree:',df[df['labels'] == minor_]['Sum_Degree'].mean())
    print('Major Complex Degree:',df[df['labels'] == major_]['Sum_Degree'].mean())
    #df.loc[(df['POD_CODE'] == POD)]
    return df[(df['POD_CODE'] == POD)][['ACTIVITY_IDENTIFIER','age','sex','SpellHRG']+features+ ['ReAdmitted in DAYS','POD_CODE','SpellPDiag','SpellSDiag']]#,'ALL_DIAG']]
    """
    codes =list(Degree_ReAdmitted_HRG['SpellHRG'].unique())     

    cardi=['DZ31Z',
    'EC21Z',
    'EC22Z',
    'EY50Z',
    'EY51Z',
    'EY01A',
    'EY01B',
    'EY02A',
    'EY02B',
    'EY11Z',
    'EY12A',
    'EY12B',
    'EY13Z',
    'EY16A',
    'EY16B',
    'EY17A',
    'EY17B']

    init_code = list(set(codes).intersection(cardi))

    stats = interact(plot_SpellHRG_HRG_ICD,
    HRG=widgets.SelectMultiple(
        options=
        init_code,
        #list(Degree_HRG['SpellHRG'].dropna().unique()),
        value=init_code,
        #rows=10,
        description='HRG',
        disabled=False
    ),
                    ICD=widgets.Dropdown(
        options=
        #init_code,
        sorted(list(Degree_HRG['SpellPDiag'].dropna().unique())),value=None 
    )
                    ,POD=widgets.Dropdown(options=list(Degree_HRG['POD_CODE'].dropna().unique()))
                    ,Degree=widgets.IntRangeSlider(value=[5,500],
        min=0,
        max=max(Degree_HRG['Sum_Degree']),
        step=1,
        description='Degree:',
        disabled=False,
        continuous_update=False,
        orientation='horizontal',
        readout=True,
        readout_format='d')
    ,Readmit=widgets.IntRangeSlider(value=[-1,30],
        min=-1,
        max=30,
        step=1,
        description='ReAdmitted:',
        disabled=False,
        continuous_update=False,
        orientation='horizontal',
        readout=True,
        readout_format='d'))"""

def plot_SpellHRG_ICD(ICD,Degree,Age,POD):    
    df = pd.DataFrame()
    Degree_ReAdmitted_HRG = utils_pickle.read("Degree_ReAdmitted_HRG") 
    for item in ICD:
        #print(item)
        data=Degree_ReAdmitted_HRG.loc[(Degree_ReAdmitted_HRG['SpellPDiag'] == item)&(Degree_ReAdmitted_HRG['POD_CODE'] == POD)
                                         &(((Degree_ReAdmitted_HRG['Sum_Degree']>=Degree[0])&(Degree_ReAdmitted_HRG['Sum_Degree'] <=Degree[1])))
                                        &(((Degree_ReAdmitted_HRG['age'].astype(int)>=Age[0])&(Degree_ReAdmitted_HRG['age'].astype(int) <=Age[1])))]
        df = pd.concat([df,data])
    features = ['Sum_Degree','Global_Central', 'Total_LOS', 'INCOME']#,'Turnaround_Degree','DIAG_COUNT']
    # Separating out the features
    x = df.loc[:, features].values
    # Separating out the target
    #y = test.loc[:,['target']].values
    # Standardizing the features
    x = StandardScaler().fit_transform(x)
    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(x)
    
    principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2'])
    #Voronoi at least four points, though clusters not less than 4
    kmeans = KMeans(n_clusters=2)
    #pair = ['INCOME','Total_LOS']
    kmeans.fit(principalDf)
    labels = kmeans.predict(principalDf)
    centroids = kmeans.cluster_centers_    
    #print(centroids)
    v = np.vstack([centroids,[0,0]])
    #print(v)
    vor = Voronoi(principalComponents)

    # plot
    regions, vertices = voronoi_finite_polygons_2d(vor)
    fig = plt.figure(figsize=(10, 10))
    colmap = {1: 'g', 2: 'r', 3: 'b', 4:'y'}
    marker = {1:'circle', 2:'diamond', 3:'dot', 4:'triangle'}
    size = {1:2,2:2,3:2,4:2}
    colors = list(map(lambda x: colmap[x+1], labels))
    markers = list(map(lambda x: marker[x+1], labels))
    sizes = list(map(lambda x: size[x+1], labels))
    #print(principalComponents)
    df['principal component 1'] = principalComponents[:,0]
    df['principal component 2'] = principalComponents[:,1]
    df['color'] = colors
    df['marker'] = markers
    df['sizes'] = sizes
    opts.defaults(opts.Points(padding=0.1, size=8, line_color='black'))
    data ={'x':list(df['principal component 1'])
                    ,'y':list(df['principal component 2'])
                    ,'color':list(df['color'])
                    ,'marker':list(df['marker']) 
                   ,'sizes':list(df['sizes'])}
    
    #hv.Points(data, vdims=['color', 'marker', 'sizes']).opts(color='color', marker='marker', size='sizes')

    plt.scatter(df['principal component 1'], df['principal component 2'], color=colors, alpha=0.5, edgecolor='k')
    #for idx, centroid in enumerate(centroids):
        #plt.scatter(*centroid, color=colmap[idx+1])
    df['labels'] = labels
    #print(list(df['labels'].unique()))
    shape_ = {}
    for item in list(df['labels'].unique()):
        shape_.update({item:[(df[df['labels'] ==item].shape[0]),df[df['labels'] == item]['Sum_Degree'].mean()]})
        print('Complex Degree:',df[df['labels'] == item]['Sum_Degree'].mean())
   
    #print(shape_)
    #print(sorted(shape_.items(), key=lambda x: x[1]))
    
    minor_=sorted(shape_.items(), key=lambda x: x[1])[0][0]
    major_=sorted(shape_.items(), key=lambda x: x[1])[1][0]
    
    #sns.pairplot(df[df['labels'] ==1][df.columns.difference(['ACTIVITY_IDENTIFIER','POD_CODE'])], hue="SpellHRG")
    #for label,x,y in zip(df[df['labels'] == minor_]['ACTIVITY_IDENTIFIER'],df[df['labels'] == minor_]['principal component 1'],df[df['labels'] == minor_]['principal component 2']):
    for label,x,y in zip(df['ACTIVITY_IDENTIFIER'],df['principal component 1'],df['principal component 2']):
    
        label = label        
        plt.annotate(label, (x,y),textcoords="offset points",xytext=(0,10),ha='center', size =20)
    
    test=zip(regions, df['color'])
    for item in test:
       
        polygon = vertices[item[0]]
        #print(region,polygon)
        #print(*zip(*polygon))
        plt.fill(*zip(*polygon), alpha=0.4
                 ,color=item[1]
                )
    plt.xlim(vor.min_bound[0]-0.1, vor.max_bound[0]+0.1)
    plt.ylim(vor.min_bound[1]-0.1, vor.max_bound[1]+0.1)
    print('Minor Complex Degree:',df[df['labels'] == minor_]['Sum_Degree'].mean())
    print('Major Complex Degree:',df[df['labels'] == major_]['Sum_Degree'].mean())
    #df.loc[(df['POD_CODE'] == POD)]
    return df[(df['POD_CODE'] == POD)][['age','sex','SpellHRG']+features+ ['POD_CODE','SpellPDiag','SpellSDiag']]#,'ALL_DIAG']]
    
    #This block is for Jupyter-Notebook script
    """ 
    codes =list(Degree_ReAdmitted_HRG['SpellHRG'].unique())    

    cardi=['DZ31Z',
    'EC21Z',
    'EC22Z',
    'EY50Z',
    'EY51Z',
    'EY01A',
    'EY01B',
    'EY02A',
    'EY02B',
    'EY11Z',
    'EY12A',
    'EY12B',
    'EY13Z',
    'EY16A',
    'EY16B',
    'EY17A',
    'EY17B']
   
    init_code = list(set(codes).intersection(cardi))

    stats = interact(plot_SpellHRG_ICD,                    
    ICD=widgets.SelectMultiple(
        options=
        #init_code,
        list(Degree_HRG['SpellPDiag'].dropna().unique()),
        value=['A415'],
        #rows=10,
        description='ICD',
        disabled=False
    )
                    ,POD=widgets.Dropdown(options=list(Degree_HRG['POD_CODE'].dropna().unique()))
                    ,Degree=widgets.IntRangeSlider(value=[5,500],
        min=0,
        max=max(Degree_HRG['Sum_Degree']),
        step=1,
        description='Degree:',
        disabled=False,
        continuous_update=False,
        orientation='horizontal',
        readout=True,
        readout_format='d')
    ,Age=widgets.IntRangeSlider(value=[-1,30],
        min=-1,
        max=100,
        step=1,
        description='Age:',
        disabled=False,
        continuous_update=False,
        orientation='horizontal',
        readout=True,
        readout_format='d')) """

def plot_SpellHRG_HRG(HRG,Degree,Readmit,POD):    
    df = pd.DataFrame()
    Degree_ReAdmitted_HRG = utils_pickle.read("Degree_ReAdmitted_HRG") 
    for item in HRG:
        #print(item)
        data=Degree_ReAdmitted_HRG.loc[(Degree_ReAdmitted_HRG['SpellHRG'] == item)&(Degree_ReAdmitted_HRG['POD_CODE'] == POD)
                                         &(((Degree_ReAdmitted_HRG['Sum_Degree']>=Degree[0])&(Degree_ReAdmitted_HRG['Sum_Degree'] <=Degree[1])))
                                        &(((Degree_ReAdmitted_HRG['ReAdmitted in DAYS']>=Readmit[0])&(Degree_ReAdmitted_HRG['ReAdmitted in DAYS'] <=Readmit[1])))]
        df = pd.concat([df,data])
    features = ['Sum_Degree','Global_Central', 'Total_LOS', 'INCOME','Turnaround_Degree','DIAG_COUNT']
    # Separating out the features
    x = df.loc[:, features].values
    # Separating out the target
    #y = test.loc[:,['target']].values
    # Standardizing the features
    x = StandardScaler().fit_transform(x)
    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(x)
    
    principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2'])
    #Voronoi at least four points, though clusters not less than 4
    kmeans = KMeans(n_clusters=2)
    #pair = ['INCOME','Total_LOS']
    kmeans.fit(principalDf)
    
    labels = kmeans.predict(principalDf)
    centroids = kmeans.cluster_centers_    
    #print(centroids)
    v = np.vstack([centroids,[0,0]])
    #print(v)
    vor = Voronoi(principalComponents)

    # plot
    regions, vertices = voronoi_finite_polygons_2d(vor)
    fig = plt.figure(figsize=(10, 10))
    colmap = {1: 'g', 2: 'r', 3: 'b', 4:'y'}
    marker = {1:'circle', 2:'diamond', 3:'dot', 4:'triangle'}
    size = {1:2,2:2,3:2,4:2}
    colors = list(map(lambda x: colmap[x+1], labels))
    markers = list(map(lambda x: marker[x+1], labels))
    sizes = list(map(lambda x: size[x+1], labels))
    #print(principalComponents)
    df['principal component 1'] = principalComponents[:,0]
    df['principal component 2'] = principalComponents[:,1]
    df['color'] = colors
    df['marker'] = markers
    df['sizes'] = sizes
    opts.defaults(opts.Points(padding=0.1, size=8, line_color='black'))
    data ={'x':list(df['principal component 1'])
                    ,'y':list(df['principal component 2'])
                    ,'color':list(df['color'])
                    ,'marker':list(df['marker']) 
                   ,'sizes':list(df['sizes'])}
    
    #hv.Points(data, vdims=['color', 'marker', 'sizes']).opts(color='color', marker='marker', size='sizes')

    plt.scatter(df['principal component 1'], df['principal component 2'], color=colors, alpha=0.5, edgecolor='k')
    #for idx, centroid in enumerate(centroids):
        #plt.scatter(*centroid, color=colmap[idx+1])
    df['labels'] = labels
    #print(list(df['labels'].unique()))
    shape_ = {}
    for item in list(df['labels'].unique()):
        shape_.update({item:[(df[df['labels'] ==item].shape[0]),df[df['labels'] == item]['Sum_Degree'].mean()]})
        print('Complex Degree:',df[df['labels'] == item]['Sum_Degree'].mean())
   
    #print(shape_)
    #print(sorted(shape_.items(), key=lambda x: x[1]))
    
    minor_=sorted(shape_.items(), key=lambda x: x[1])[0][0]
    major_=sorted(shape_.items(), key=lambda x: x[1])[1][0]
    
    #sns.pairplot(df[df['labels'] ==1][df.columns.difference(['ACTIVITY_IDENTIFIER','POD_CODE'])], hue="SpellHRG")
    #for label,x,y in zip(df[df['labels'] == minor_]['ACTIVITY_IDENTIFIER'],df[df['labels'] == minor_]['principal component 1'],df[df['labels'] == minor_]['principal component 2']):
    for label,x,y in zip(df['ACTIVITY_IDENTIFIER'],df['principal component 1'],df['principal component 2']):
    
        label = label        
        plt.annotate(label, (x,y),textcoords="offset points",xytext=(0,10),ha='center', size =20)
    
    test=zip(regions, df['color'])
    for item in test:
       
        polygon = vertices[item[0]]
        #print(region,polygon)
        #print(*zip(*polygon))
        plt.fill(*zip(*polygon), alpha=0.4
                 ,color=item[1]
                )
    plt.xlim(vor.min_bound[0]-0.1, vor.max_bound[0]+0.1)
    plt.ylim(vor.min_bound[1]-0.1, vor.max_bound[1]+0.1)
    print('Minor Complex Degree:',df[df['labels'] == minor_]['Sum_Degree'].mean())
    print('Major Complex Degree:',df[df['labels'] == major_]['Sum_Degree'].mean())
    #df.loc[(df['POD_CODE'] == POD)]
    return df[(df['POD_CODE'] == POD)][['ACTIVITY_IDENTIFIER','SpellHRG']+features+ ['ReAdmitted in DAYS','POD_CODE','ALL_DIAG']]
    
    #The below block is for Jupyter-Notebook
    """
    codes =list(Degree_ReAdmitted_HRG['SpellHRG'].unique())     

    cardi=['DZ31Z',
    'EC21Z',
    'EC22Z',
    'EY50Z',
    'EY51Z',
    'EY01A',
    'EY01B',
    'EY02A',
    'EY02B',
    'EY11Z',
    'EY12A',
    'EY12B',
    'EY13Z',
    'EY16A',
    'EY16B',
    'EY17A',
    'EY17B']

    init_code = list(set(codes).intersection(cardi))

    stats = interact(plot_SpellHRG_HRG,
                    
    HRG=widgets.SelectMultiple(
        options=
        #init_code,
        list(Degree_HRG['SpellHRG'].dropna().unique()),
        value=init_code,
        #rows=10,
        description='HRG',
        disabled=False
    )
                    ,POD=widgets.Dropdown(options=list(Degree_HRG['POD_CODE'].dropna().unique()))
                    ,Degree=widgets.IntRangeSlider(value=[5,500],
        min=0,
        max=max(Degree_HRG['Sum_Degree']),
        step=1,
        description='Degree:',
        disabled=False,
        continuous_update=False,
        orientation='horizontal',
        readout=True,
        readout_format='d')
    ,Readmit=widgets.IntRangeSlider(value=[-1,30],
        min=-1,
        max=30,
        step=1,
        description='ReAdmitted:',
        disabled=False,
        continuous_update=False,
        orientation='horizontal',
        readout=True,
        readout_format='d'))"""

def plot_SpellHRG_HRG_Degree(HRG,Degree,Readmit,POD):    
    df = pd.DataFrame()
    Degree_ReAdmitted_HRG = utils_pickle.read("Degree_ReAdmitted_HRG")
    Degree_ReAdmitted_HRG = Degree_ReAdmitted_HRG[Degree_ReAdmitted_HRG['SpellHRG'].notna()] 
    Degree_ReAdmitted_HRG['ReAdmitted in DAYS'] = Degree_ReAdmitted_HRG['ReAdmitted in DAYS'].fillna(-1) 
    for item in HRG:
        #print(item)
        data=Degree_ReAdmitted_HRG.loc[(Degree_ReAdmitted_HRG['SpellHRG'] == item)&(Degree_ReAdmitted_HRG['POD_CODE'] == POD)
                                        &(((Degree_ReAdmitted_HRG['Sum_Degree']>=Degree[0])&(Degree_ReAdmitted_HRG['Sum_Degree'] <=Degree[1])))
                                        &(((Degree_ReAdmitted_HRG['ReAdmitted in DAYS']>=Readmit[0])&(Degree_ReAdmitted_HRG['ReAdmitted in DAYS'] <=Readmit[1])))]
        df = pd.concat([df,data])
    
    features = ['Sum_Degree','Global_Central', 'Total_LOS', 'INCOME','Turnaround_Degree','DIAG_COUNT']  
    principalComponents = sliced_principle_components(df,features,2)
    principalDf = pd.DataFrame(data = principalComponents, columns = ['principal component 1', 'principal component 2'])
    kmax = 10
    best_n = best_eblow_k(principalDf.values.tolist(),kmax = 10) 
    df = plot_vor(df,principalComponents, best_n)
    return df[(df['POD_CODE'] == POD)][['ACTIVITY_IDENTIFIER','SpellHRG']+features+ ['ReAdmitted in DAYS','POD_CODE','ALL_DIAG','labels']]


    
    """
    codes =list(Degree_ReAdmitted_HRG['SpellHRG'].unique())     

    cardi=['DZ31Z',
    'EC21Z',
    'EC22Z',
    'EY50Z',
    'EY51Z',
    'EY01A',
    'EY01B',
    'EY02A',
    'EY02B',
    'EY11Z',
    'EY12A',
    'EY12B',
    'EY13Z',
    'EY16A',
    'EY16B',
    'EY17A',
    'EY17B']

    init_code = list(set(codes).intersection(cardi))

    stats = interact(plot_SpellHRG_HRG_Degree,
                    
    HRG=widgets.SelectMultiple(
        options=
        #init_code,
        list(Degree_HRG['SpellHRG'].dropna().unique()),
        value=init_code,
        #rows=10,
        description='HRG',
        disabled=False
    )
                    ,POD=widgets.Dropdown(options=list(Degree_HRG['POD_CODE'].dropna().unique()))
                    ,Degree=widgets.IntRangeSlider(value=[5,500],
        min=0,
        max=max(Degree_HRG['Sum_Degree']),
        step=1,
        description='Degree:',
        disabled=False,
        continuous_update=False,
        orientation='horizontal',
        readout=True,
        readout_format='d')
    ,Readmit=widgets.IntRangeSlider(value=[-1,30],
        min=-1,
        max=30,
        step=1,
        description='ReAdmitted:',
        disabled=False,
        continuous_update=False,
        orientation='horizontal',
        readout=True,
        readout_format='d'))"""
def plot_SpellHRG_HRG_Degree_PairCompare(HRG,Degree,Readmit,POD):    
    df = pd.DataFrame()
    Degree_ReAdmitted_HRG = utils_pickle.read("../Degree_ReAdmitted_HRG")
    Degree_ReAdmitted_HRG = Degree_ReAdmitted_HRG[Degree_ReAdmitted_HRG['SpellHRG'].notna()] 
    Degree_ReAdmitted_HRG['ReAdmitted in DAYS'] = Degree_ReAdmitted_HRG['ReAdmitted in DAYS'].fillna(-1) 
    for item in HRG:
        #print(item)
        data=Degree_ReAdmitted_HRG.loc[(Degree_ReAdmitted_HRG['SpellHRG'] == item)&(Degree_ReAdmitted_HRG['POD_CODE'] == POD)
                                        &(((Degree_ReAdmitted_HRG['Sum_Degree']>=Degree[0])&(Degree_ReAdmitted_HRG['Sum_Degree'] <=Degree[1])))
                                        &(((Degree_ReAdmitted_HRG['ReAdmitted in DAYS']>=Readmit[0])&(Degree_ReAdmitted_HRG['ReAdmitted in DAYS'] <=Readmit[1])))]
        df = pd.concat([df,data])
    
    features = ['Sum_Degree','Global_Central', 'Total_LOS', 'INCOME','Turnaround_Degree','DIAG_COUNT']  
    principalComponents = sliced_principle_components(df,features,2)
    principalDf = pd.DataFrame(data = principalComponents, columns = ['principal component 1', 'principal component 2'])
    kmax = 10
    best_n = best_eblow_k(principalDf.values.tolist(),kmax = 10) 
    df = plot_vor(df,principalComponents, best_n)    
    features.append('labels')    
    sns.pairplot(df[features], hue="labels", diag_kws={'bw':'1.0'})
    #df[(df['POD_CODE'] == POD)][['ACTIVITY_IDENTIFIER','SpellHRG']+features+ ['ReAdmitted in DAYS','POD_CODE','ALL_DIAG','labels']]
    return df[(df['POD_CODE'] == POD)][['ACTIVITY_IDENTIFIER','SpellHRG']+features+ ['ReAdmitted in DAYS','POD_CODE','ALL_DIAG','labels']]

def multi_table(df, title, addAll=True):
    fig = go.Figure()

    
    fig.add_trace(go.Table(
        header=dict(values=list(df.columns)),
        cells=dict(values=df.transpose().values.tolist()
        )
    ))

    button_all = dict(label='All',
    method='update',
    args=[{'visible':df['labels'].isin(list(df['labels'].unique())),'title':'All','showlegend':True}])

    def create_layout_button(column):
        return dict(label=column, method='update', args=[{'visible':df['labels'].isin([column]),'title':column,'showlegend':True}])
    
    fig.update_layout(updatemenus=[go.layout.Updatemenu(active=0, buttons=([button_all]*addAll)+list(df['labels'].map(lambda column:create_layout_button(column))))],yaxis_type="log")

    fig.show()
    return

def plot_Spell_PU_Degree_PairCompare(Label,selected_list,Degree,Waterlow_Standard,features = ['Sum_Degree','Global_Central', 'Total_LOS', 'Turnaround_Degree']  ):    
    def pop_std(x):
        return x.std(ddof=0)
    all_modelled_events = ['WardStay,LabTest', 'WardStay',
       'WardStay,Waterlow,LabTest,PatientPosition',
       'WardStay,Waterlow,LabTest,Skinasses,TV,PatientPosition',
       'WardStay,Waterlow,LabTest',
       'WardStay,Waterlow,LabTest,Skinasses,PatientPosition',
       'WardStay,Waterlow,LabTest,TV,PatientPosition',
       'WardStay,Waterlow,Skinasses,PatientPosition',
       'WardStay,Waterlow,PatientPosition', 'WardStay,PatientPosition',
       'WardStay,LabTest,PatientPosition', 'WardStay,Waterlow',
       'WardStay,Skinasses', 'WardStay,Skinasses,PatientPosition',
       'WardStay,Waterlow,Skinasses', 'WardStay,LabTest,Skinasses',
       'WardStay,LabTest,Skinasses,PatientPosition',
       'WardStay,Waterlow,Skinasses,TV,PatientPosition',
       'WardStay,LabTest,Skinasses,TV',
       'WardStay,Waterlow,TV,PatientPosition',
       'WardStay,Waterlow,LabTest,Skinasses',
       'WardStay,LabTest,Skinasses,TV,PatientPosition',
       'WardStay,LabTest,TV', 'WardStay,LabTest,TV,PatientPosition',
       'WardStay,Waterlow,LabTest,TV', 'WardStay,TV,PatientPosition',
       'WardStay,Waterlow,TV', 'WardStay,TV', 'WardStay,Skinasses,TV',
       'WardStay,Waterlow,LabTest,Skinasses,TV']
    selected_list=list(selected_list)
    selected_list.append('WardStay')
    modelled_events =[]
    for item in all_modelled_events:
        #print(item.split(','))        
        #print(selected_list)
        if set(item.split(','))==set(selected_list):
            modelled_events.append(item)
    if len(modelled_events)==0:
        print("No Events!")
        return
    df = pd.DataFrame()
    Degree_ReAdmitted_HRG = patientKG.utils_pickle.read("PU_RESULT")
    #Degree_ReAdmitted_HRG = Degree_ReAdmitted_HRG[Degree_ReAdmitted_HRG['SpellHRG'].notna()] 
    #Degree_ReAdmitted_HRG['ReAdmitted in DAYS'] = Degree_ReAdmitted_HRG['ReAdmitted in DAYS'].fillna(-1) 
    # for item in Label:
        #print(item)
    df=Degree_ReAdmitted_HRG.loc[(Degree_ReAdmitted_HRG['Label'].isin(Label))& (Degree_ReAdmitted_HRG['modelled_events'].isin(modelled_events))                                   
                                        &(((Degree_ReAdmitted_HRG['Sum_Degree']>=Degree[0])
                                        &(Degree_ReAdmitted_HRG['Sum_Degree'] <=Degree[1])))
                                        &(Degree_ReAdmitted_HRG['Waterlow_Standard'] == Waterlow_Standard)
                                        #&(((Degree_ReAdmitted_HRG['ReAdmitted in DAYS']>=Readmit[0])&(Degree_ReAdmitted_HRG['ReAdmitted in DAYS'] <=Readmit[1])))
                                        ]
        # df = pd.concat([df,data])
    
    #features = ['Sum_Degree','Global_Central', 'Total_LOS', 'Turnaround_Degree']  
    principalComponents,pca_explained,pca_components = sliced_principle_components(df,features,2)
    principalDf = pd.DataFrame(data = principalComponents, columns = ['principal component 1', 'principal component 2'])
    kmax = 10
    best_n = best_eblow_k(principalDf.values.tolist(),kmax = 10) 
    try:
        df = plot_vor(df,principalComponents, best_n)    
    except:
        df = plot(df,principalComponents, best_n)
    #print(list(features))
    selector = VarianceThreshold()
    x = df[list(features)].values #returns a numpy array
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    df_x = pd.DataFrame(x_scaled)

    selector.fit_transform(df_x)
    #print(selector.variances_)
    feature_list = []
    for item in features:
        feature_list.append(item)
    for i in range(len(pca_components)):
        for item in pca_components[i]:
           result_i=[item*pca_explained[i]]
    features_score = selector.variances_    
    feature_rank = pd.DataFrame(list(zip(features, features_score)), columns=['variable_name','score'])
    #print(feature_rank)
    test3 =pd.DataFrame()
    
    #print(test3)
    for item in df[feature_list].columns:
        test1=df.groupby(['labels'])[item].agg({item:'mean'}).T
        test2=df.groupby(['labels'])[item].agg({item:pop_std}).T
        test4 =pd.DataFrame()
        for item in df['labels'].unique():
            test4[item] = test1[item].round(2).astype(str)+" (\u00B1"+test2[item].round(2).astype(str)+")"        
        test3=test3.append(test4)
    
    test3 = test3.reindex(sorted(test3.columns),axis=1)
    test3['variable_name'] = test3.index
    #print(test3)
    test3 =test3.merge(feature_rank, how='left', on='variable_name')
    #test3 = test3.set_index('variable_name')
    test5=pd.DataFrame(df.groupby(['labels'])['activity_identifier'].agg({'activity_identifier':'count'}).T)
    test3 = test3.sort_values(by='score',ascending=False)
    test3=pd.concat([test5,test3]).set_index('variable_name')
    display(test3)
    all_features = feature_list.copy()  
    if len(feature_list)>5:              
        feature_list = list(test3.sort_values(by='score',ascending=False).index[:5].values)
    feature_list.append('labels')
    lis_ = df[['labels','color']].drop_duplicates()
    palette={y['labels']:str(y['color']) for x,y in lis_.iterrows()}
    sns.pairplot(df[feature_list], hue="labels", diag_kws={'bw':'1.0'},palette=palette)
    #df[(df['POD_CODE'] == POD)][['ACTIVITY_IDENTIFIER','SpellHRG']+features+ ['ReAdmitted in DAYS','POD_CODE','ALL_DIAG','labels']]
    #df[(df['Waterlow_Standard'] == Waterlow_Standard)][['ACTIVITY_IDENTIFIER']+features+ ['labels']]  
    
    return pivot_ui(df[(df['Waterlow_Standard'] == Waterlow_Standard)][['activity_identifier']+all_features+ ['Waterlow_Standard','Careplan','Label','labels']])

def data_load_clean():
    Red004_Conn = Red004()
    PU_RESULT = pd.read_sql_query('SELECT * from [AdvancedAnalytics].[dbo].[Variance_Analysis]',Red004_Conn)
    Red004_Conn.close()
    PU_RESULT = PU_RESULT[~PU_RESULT['Label'].str.contains('Empty')]
    PU_RESULT=PU_RESULT.fillna(0)
    encode_list=[#'Chief Complaint SNOMED Code'
            #,'PRESENTING_COMPLAINT'
            'modelled_events'
            ,'all_codes'
            ,'WARD STAY LOCATION'
            ,'ETHNIC_CATEGORY_CODE'
            ,'PERSON_MARITAL_STATUS_CODE'
            ,'RELIGIOUS_OR_OTHER_BELIEF_SYSTEM_AFFILIATION'
    ,'Waterlow_Standard'
    ,'Careplan'
            ,'WL - Age'
      ,'WL - Broken Type'
      ,'WL - Build/Weight for Height'
      ,'WL - Continence'
      ,'WL - Gender'
      ,'WL - Lack of Appetite'
      ,'WL - Major Surgery / Trauma'
      ,'WL - Medication'
      ,'WL - Mobility'
      ,'WL - Neurological Deficit'
      ,'WL - Recent Weight Loss'
      ,'WL - Skin Type'
      ,'WL - Tissue Malnutrition'
      #,'WL - Waterlow Score'
      ,'WL - Weight Lost'
      ,'Wound Status',
    'Photograph Wound',
    'Wound Width',
    'Wound Depth',
    'Wound Exudate Odour',
    'Dressing Type:',
    'Wound Surrounding Tissue Colour',
    'Wound Cleansing',
    'Dressing Assessment:',
    'Wound Undermining Location',
    'Wound Tunneling Location',
    'Wound Odour',
    'Already Being Cared for in the Community',
    'Wound Exudate Colour',
    'Equipment Induced Pressure Ulcer',
    'Wound Edge',
    'Wound Percent Epithelialised:',
    'Equipment Type',
    'Wound Dressing Activity',
    'Wound Colour',
    'Next Dressing Change',
    'Pressure Ulcer Present On Admission',
    'Wound Length',
    'Wound Percent Tissue Eschar',
    'Pressure Ulcer Datix Number',
    'Pressure Ulcer Datix completed',
    'Consent to Photograph',
    'Wound Percent Granulated',
    'Wound Percent Tissue Slough',
    'Wound Type  - Wound Assessment',
    'Wound Tunneling Depth',
    'Wound Exudate Volume',
    'Wound Undermining Depth',
    'Wound Exudate Type',
    'Wound Surrounding Tissue',
    'Pressure Ulcer/Blister Category'
    ,'Referral Status Tissue Viability'
      ,'Referral - Tissue Viability','PATIENTPOSITION','Label']
    for column in PU_RESULT[PU_RESULT.columns.difference(encode_list)]:
            try:
                PU_RESULT[column] = PU_RESULT[column].replace(' ', np.NaN).replace('----',np.NaN).replace('----',np.NaN, regex=True).replace('[a-zA-Z]',np.NaN,regex=True).astype(float)
            except Exception as e:
                if column == 'C-reactive protein':
                    PU_RESULT[column] = PU_RESULT[column].replace('<1', 0.5).replace('<0.2', 0.5).replace('<0.3', 0.5).replace(' ', np.NaN).replace('[a-zA-Z]',np.NaN,regex=True).astype(float)
                elif column =='Glucose, CSF':
                    PU_RESULT[column] = PU_RESULT[column].replace('<0.1', 0.1).replace('<0.2', 0.5).replace('<0.3', 0.5).replace(' ', np.NaN).replace('[a-zA-Z]',np.NaN,regex=True).astype(float)
                elif e == 'cannot astype a datetimelike from [datetime64[ns]] to [float64]':
                    pass
    PU_RESULT=PU_RESULT.fillna(0)
    utils_pickle.write(PU_RESULT,"PU_RESULT")
    return

def PU_Demo():
    Label=['No-diagnose', 'Diagnosed_PU']
    Variable_selection = ['WL - Waterlow Score','Sum_Degree','Global_Central', 'Total_LOS', 'Turnaround_Degree','Mean cell volume', 'Haematocrit', 'Red blood cell count',
       'Basophil count', 'White blood cell count', 'Mean cell haemoglobin',
       'Neutrophil count', 'Eosinophil count', 'Haemoglobin',
       'Lymphocyte count', 'Platelet count', 'Mean cell haemoglobin conc',
       'Monocyte count', 'Haemoglobin A1c IFCC', 'C-reactive protein',
       'Glucose fasting', 'Glucose random', 'Glucose, CSF',
       'Glucose, dialysis fluid', 'Glucose, fluid', 'Albumin']
    Waterlow_Standard = [
                    'No waterlow', 0,
       "{'rule 1': 'Pass', 'rule 2': 'Pass', 'rule 3': 'Pass', 'rule 4': 'Pass'}",
       "{'rule 1': 'Pass', 'rule 2': 'Pass', 'rule 3': 'Pass', 'rule 4': 'Fail'}",
       "{'rule 1': 'Pass', 'rule 2': 'Fail', 'rule 3': 'Fail', 'rule 4': 'Fail'}",
       "{'rule 1': 'Pass', 'rule 2': 'Fail', 'rule 3': 'Fail', 'rule 4': 'Pass'}"]
    
    events_list =['Waterlow','LabTest','Skinasses','TV','PatientPosition'] 
    stats = interact(plot_Spell_PU_Degree_PairCompare,
    Label=widgets.SelectMultiple(
        options=
        Label,
        value=  Label,
        #rows=10,
        description='Pressure Ulcer',
        disabled=False
    ),selected_list=widgets.SelectMultiple(
        options=
        events_list,
        value=  ['Waterlow','LabTest'],
        #rows=10,
        description='Events',
        disabled=False
    )
                    
                    ,Degree=widgets.IntRangeSlider(value=[5,500],
        min=0,
        max=500,
        step=1,
        description='Degree:',
        disabled=False,
        continuous_update=False,
        orientation='horizontal',
        readout=True,
        readout_format='d')
                    ,Waterlow_Standard=widgets.Dropdown(options=Waterlow_Standard)
                    ,features=widgets.SelectMultiple(
        options=
        Variable_selection,
        value=  ['Sum_Degree','Global_Central', 'Total_LOS', 'Turnaround_Degree','WL - Waterlow Score'] ,
        #rows=10,
        description='Variables',
        disabled=False
    )
    )
    return stats

def sub_timeline_layout(tt, labels=None):
    node = {}
    for node_index, node_feature in tt.nodes(data=True):
        node.update({node_index:node_feature['activity_start_time']})
    
    sorted_dic = sorted(node.items(), key=lambda kv: kv[1])
    pos = {}
    i=0
    x = 0
    y = 0
    list_=[]
    for i in range(len(sorted_dic)):
        if i >0:
            diff = datetime.strptime(sorted_dic[i][1],'%Y.%m.%d %H:%M:%S')-datetime.strptime(sorted_dic[i-1][1],'%Y.%m.%d %H:%M:%S')            
            x = x +(diff.seconds)/18
            y = y 
            pos.update({sorted_dic[i][0]:np.array([x,y])})        
        else:
            pos.update({sorted_dic[0][0]:np.array([0,0])})
    
        if labels is not None:                       
            result = ''.join([i for i in labels[sorted_dic[i][0]] if not i.isdigit()])
            if result == '._start':
                continue
            elif result == '._end':
                continue
            else:
                list_.append(result)
    unique_events = set(list_)
    pos_y = 20
    for item in unique_events:        
        for i in range(len(sorted_dic)):
            event_match = re.match( r'{}'.format(item), labels[sorted_dic[i][0]], re.M|re.I)
            if event_match:
                x= pos[sorted_dic[i][0]][0]
                y = pos_y
                pos.update({sorted_dic[i][0]:np.array([x,y])})
        
        pos_y = pos_y + 30 
            
    
    return pos

def PU_Path_Vis_Demo(item = 123456):    
    # hv.extension('bokeh')
    # defaults = dict(width=1000, height=1000, padding=0.1)
    # Load individual visualization requires patientKG Class     
    graph = utils_pickle.read("GraphCalculationResults/Ward_Stay/KG_{}".format(item))
    # hv.opts.defaults(opts.EdgePaths(**defaults), opts.Graph(**defaults), opts.Nodes(**defaults))
    
    simple_graph=hv.Graph.from_networkx(graph.graph, graph.timeline_layout())
    simple_graph

    days_nodes = {}

    for i in range(0, 16, 1):
        nodes_list = [0,1]
        for k,v in graph.graph.nodes(data=True):
            if k > 1:
                diff = (datetime.strptime(v['activity_start_time'],'%Y.%m.%d %H:%M:%S') -datetime.strptime(graph.graph.nodes[0]['activity_start_time'],'%Y.%m.%d %H:%M:%S')).total_seconds()            
                if diff <= i*3600*24:
                    nodes_list.append(k)                
        
        days_nodes.update({i:nodes_list})      
    return hv.HoloMap({i: hv.Graph.from_networkx(graph.graph.subgraph(days_nodes[i]),  sub_timeline_layout(graph.graph.subgraph(days_nodes[i]),dict((int(v),k) for k,v in graph.node_dic.items())), iterations=i, seed=10) for i in range(0, 16, 1)},
            kdims='Iterations')

if __name__ == '__main__':
#     Degree_HRG = utils_pickle.read("../Degree_HRG")
#     Degree_ReAdmitted_HRG = utils_pickle.read("../Degree_ReAdmitted_HRG")
    
#     HRG = [
# 'EY01A',
# 'EY01B',
# 'EY02A',
# 'EY02B',
# 'EY11Z',
# 'EY12A',
# 'EY12B',
# 'EY13Z',
# 'EY16A',
# 'EY16B',
# 'EY17A',
# 'EY17B']
#     Degree = [0,100]
#     Readmit = [-1,15]
#     POD = "{AandE, NEL, UNBUN}"   
#     plot_SpellHRG_HRG_Degree_PairCompare(HRG,Degree,Readmit,POD)
    # Label=['No-diagnose', 'Diagnosed_PU']
    # modelled_events = ['Waterlow','LabTest']
    # Degree = [0,100]
    # Waterlow_Standard = "No waterlow"#"{'rule 1': 'Pass', 'rule 2': 'Pass', 'rule 3': 'Pass', 'rule 4': 'Pass'}"
    # features = ['Sum_Degree','Global_Central', 'Total_LOS', 'Turnaround_Degree']  
    # plot_Spell_PU_Degree_PairCompare(Label,modelled_events,Degree,Waterlow_Standard,features)
    PU_Path_Vis_Demo()    
    #data_load_clean()

    