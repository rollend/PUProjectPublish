from operator import mul
import sys
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
from ipywidgets import interact, interactive, fixed, interact_manual, Text, interactive_output
from ipywidgets import Button, HBox, VBox,Layout,Label
import panel as pn
import seaborn as sns
from kneed import KneeLocator
from PatientGraphPheno import *
from patientKG.config.bedrock_connection import *
#from patientKG import utils_pickle
import patientKG.utils_pickle
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
import urllib, json
sns.set(style="ticks")
hv.extension('bokeh')
defaults = dict(width=1000, height=1000, padding=0.1)
from patientKG.tests.test_graphs import *
from ipywidgets import TwoByTwoLayout
import itertools
import time
from IPython.display import IFrame
import json, io
from patientKG.priorKnowledge.Hb1AC import *
from patientKG.priorKnowledge.Albumin import *
from patientKG.priorKnowledge.FBC import *
from patientKG.priorKnowledge.Creactive import *
from scipy.stats import chi2_contingency
import scipy.stats as stats
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

def sankey(df):

    labels = ['Total']
    source=[]
    target=[]
    value=[]
    color = ['#57c19c']
    color_map={'g':'green','r':'red','y':'yellow','b':'blue'}
    total = df['activity_identifier'].nunique()
    grouped = pd.DataFrame(df.groupby(['labels','Label','color'])['activity_identifier'].nunique()).reset_index()
    for item in sorted(df['labels'].unique()):
        labels.append(str(item))
        source.append(0)
        target.append(labels.index(str(item)))
        value.append(grouped[grouped['labels']==item]['activity_identifier'].sum())
        color.append(str(color_map[grouped[grouped['labels']==item]['color'].unique()[0]]))
    for item in sorted(df['labels'].unique()):
        for item2 in sorted(df['Label'].unique()):
            try:
                num = int(grouped[(grouped['labels']==item)&(grouped['Label']==item2)]['activity_identifier'])
                labels.append(str(item+"_"+item2))
            except:
                continue
    color.append('black')
    color.append('pink')
    
    for index,row in grouped.iterrows():
        source_label, target_label,value_ = row['labels'], row['Label'],row['activity_identifier']
        source.append(labels.index(str(source_label)))
        target.append(labels.index(str(source_label+"_"+target_label)))
        value.append(value_)
    
    percentage_node = ["{:.2f}".format(total/total*100)+"%"]
    diff = list(set(source)-set([0]))
    i=0
    cn =0 
    while i < len(source):
        if source[i] == 0:
            percentage_node.append("{:.2f}".format(value[i]/total*100)+"%")
            cn+=1        
        i+=1
    
    while cn < len(source):        
        percentage_node.append("{:.2f}".format(value[cn]/value[target.index(source[cn])]*100)+"%")
        cn+=1
    percentage_link = ["{:.2f}".format(total/total*100)+"%", "60%", "70%", "60%", "100%"]

    fig = go.Figure(data=[go.Sankey(
        node = dict(
        pad = 15,
        thickness = 20,
        line = dict(color = "black", width = 0.5),
        label = labels,
        customdata = percentage_node,
        hovertemplate='%{label}: %{value}(%{customdata})<extra></extra>',
        color = color
        ),
        link = dict(
        source = source, # indices correspond to labels, eg A1, A2, A1, B1, ...
        target = target,
        value =  value,
        #customdata = percentage_link,
        #hovertemplate='Link from %{source.label}<br />'+
        #'to %{target.label}<br /> %{value}(%{customdata})'+
        #'<extra></extra>',
    ))])
    #
    return fig.update_layout(title_text="Cluster via Outcome", font_size=10)#labels, source, target, value

def plot_Spell_PU_Degree_PairCompare(Label, Expected_LOS,selected_list,Age,Waterlow_Standard,features = ['Complex_Degree','Global_Central', 'Total_LOS', 'Turnaround_Degree']  ):    
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

    Waterlow_Compliance = list(Waterlow_Standard)

    if len(Waterlow_Compliance)==1 and Waterlow_Compliance[0]!='Rule 1: Use Waterlow' and Waterlow_Compliance[0]!='No Waterlow':
            return "In RBH we only use Waterlow!"
    
    diction={'Rule 1: Use Waterlow':{'rule 1': 'Pass'}, 'Rule 2: 4 Hours Admission':{'rule 2': 'Pass'}, 'Rule 3: AE 4hours':{'rule 3': 'Pass'}, 'Rule 4: Ward Transfer 4hours':{'rule 4': 'Pass'},'No Waterlow':'No Waterlow'}
    
    waterlow_group=[]    

    rule_group={}
    for index, key in enumerate(diction):
        rule_number = index+1
        if key != 'No Waterlow':
            if key in Waterlow_Compliance:
                rule_group.update(diction[key])
            else:
                rule_group.update({'rule {}'.format(rule_number):'Fail'})
        else:
            waterlow_group.append(diction[key])
    waterlow_group.append(str(rule_group))

    df = pd.DataFrame()
    Degree_ReAdmitted_HRG = patientKG.utils_pickle.read("PU_RESULT")
    #Degree_ReAdmitted_HRG = Degree_ReAdmitted_HRG[Degree_ReAdmitted_HRG['SpellHRG'].notna()] 
    #Degree_ReAdmitted_HRG['ReAdmitted in DAYS'] = Degree_ReAdmitted_HRG['ReAdmitted in DAYS'].fillna(-1) 
    # for item in Label:
        #print(item)

    los_dic= {"Expected Long for HRG":"Normal", "Unexpected Long for HRG":"Abnormal","Unexpected short - live discharge":"Not yet", 'Unknown': 'Unknown'}
    LOS_LIST =[]
    for item in Expected_LOS:
        LOS_LIST.append(los_dic[item])

    try:
        df=Degree_ReAdmitted_HRG.loc[(Degree_ReAdmitted_HRG['Label'].isin(Label))&(Degree_ReAdmitted_HRG['Expected_LOS'].isin(LOS_LIST))& (Degree_ReAdmitted_HRG['modelled_events'].isin(modelled_events))                                   
                                        &(((Degree_ReAdmitted_HRG['HPS_AGE_AT_ADMISSION_DATE']>=Age[0])
                                        &(Degree_ReAdmitted_HRG['HPS_AGE_AT_ADMISSION_DATE'] <=Age[1])))
                                        &(Degree_ReAdmitted_HRG['Waterlow_Standard'].isin(waterlow_group))
                                        #&(((Degree_ReAdmitted_HRG['ReAdmitted in DAYS']>=Readmit[0])&(Degree_ReAdmitted_HRG['ReAdmitted in DAYS'] <=Readmit[1])))
                                        ]
        # df = pd.concat([df,data])
    except:
        return "No Sample!"
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
    
    #Section Outcomes to Estimated groups
    total = df['activity_identifier'].nunique()
    outcomes = df.groupby(['labels','Label'])['activity_identifier'].nunique()
    fig = sankey(df)
    fig.show()
    
    
    
    #Section phenotype table with variables
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
        sub_df = df[df[item]!=0]
        test1=sub_df.groupby(['labels'])[item].agg({item:'mean'}).T
        test2=sub_df.groupby(['labels'])[item].agg({item:pop_std}).T
        test4 =pd.DataFrame()
        for item in sub_df['labels'].unique():
            test4[item] = test1[item].round(2).astype(str)+" (\u00B1"+test2[item].round(2).astype(str)+")"        
        test3=test3.append(test4)
    
    test3 = test3.reindex(sorted(test3.columns),axis=1)
    test3['variable_name'] = test3.index
    #print(test3)
    test3 =test3.merge(feature_rank, how='left', on='variable_name')
    #test3 = test3.set_index('variable_name')
    test5=pd.DataFrame(df.groupby(['labels'])['activity_identifier'].agg({'activity_identifier':lambda x: x.nunique()}).T)
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
    
    return df[['activity_identifier']+all_features+ ['Waterlow_Standard','Careplan','Label','labels','Expected_LOS']]


def transform(PU_RESULT):
    PU_RESULT = PU_RESULT.replace(0,np.NaN)
    avg_list = ['WL - Waterlow Score','Mean cell volume', 'Haematocrit', 'Red blood cell count',
       'Basophil count', 'White blood cell count', 'Mean cell haemoglobin',
       'Neutrophil count', 'Eosinophil count', 'Haemoglobin',
       'Lymphocyte count', 'Platelet count', 'Mean cell haemoglobin conc',
       'Monocyte count', 'Haemoglobin A1c IFCC', 'C-reactive protein',
       'Glucose fasting', 'Glucose random', 'Glucose, CSF',
       'Glucose, dialysis fluid', 'Glucose, fluid', 'Albumin']

    concate_list = [
        'WL - Age',
        'WL - Broken Type',
        'WL - Build/Weight for Height',
        'WL - Continence',
        'WL - Gender',
        'WL - Lack of Appetite',
        'WL - Major Surgery / Trauma',
        'WL - Medication',
        'WL - Mobility',
        'WL - Neurological Deficit',
        'WL - Recent Weight Loss',
        'WL - Skin Type',
        'WL - Tissue Malnutrition',
        'WL - Weight Lost',
        'PATIENTPOSITION',
        'Referral Status Tissue Viability',
        'Wound Status',
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
    ]

    max_list = ['modelled_events',
            'local_patient_identifier',
            'all_codes',
            'all_hrg',
            'HPS_ACTIVITY_DATE_TIME',
            'HPS_DISCHARGE_DATE_TIME_HOSPITAL_PROVIDER_SPELL',
            'Complex_Degree',
            'Global_Central',
            'Total_LOS',
            'Turnaround_Degree',
            'Waterlow_Standard',
            'Careplan',
            'HPS_ADMISSION_METHOD_CODE_HOSPITAL_PROVIDER_SPELL',
            'HPS_AGE_AT_ADMISSION_DATE',
            'PERSON_MARITAL_STATUS_CODE_DESC','weight',
        'height','Pressure Ulcer Present On Admission',
        'Label','DT_ATRISK','ward_move','careplan_ontime','numberof_repositioning','carplan_numberof_repositioning','careplan_compliance_degree']

    for item in concate_list:
        PU_RESULT[item] = PU_RESULT.groupby(['activity_identifier'])[item].transform(lambda x: ' '.join(str(x)))
        PU_RESULT = PU_RESULT.drop_duplicates()
    print("Concate Finished")
    for item in avg_list:
        PU_RESULT[item] = PU_RESULT.groupby(['activity_identifier'])[item].transform(np.mean)
        PU_RESULT = PU_RESULT.drop_duplicates()
    print("Avg Finished")
    for item in max_list:
        try:
            PU_RESULT[item] = PU_RESULT.groupby(['activity_identifier'])[item].transform(np.max)
        except:
            PU_RESULT[item] = PU_RESULT[item].astype(str)
            PU_RESULT[item] = PU_RESULT.groupby(['activity_identifier'])[item].transform(np.max)
        PU_RESULT = PU_RESULT.drop_duplicates()
    PU_RESULT = PU_RESULT.drop_duplicates()
    return PU_RESULT

def check_blood_normal(Reference_Range,input_node_fields,PU_RESULT):
    for item in input_node_fields:
        print(item)
        ref_inuse ={}
        for key, value in Reference_Range[item].items():    
            if key == 'Male':
                ref_inuse.update({'Sex is male':value})
            elif key == 'Female':
                ref_inuse.update({'Sex is female':value})
            elif key == 'Unknown':
                ref_inuse.update({'Sex is unknown':value})
            else:
                ref_inuse.update({key:value})
        PU_RESULT[item +'_normal'] = PU_RESULT.apply(lambda row: -1 if (pd.isnull(row[item]))else (1 if float(ref_inuse[row['PERSON_GENDER_CODE_DESC']]['min']) <= row[item]<=float(ref_inuse[row['PERSON_GENDER_CODE_DESC']]['max'])  else 0),axis=1)
    return PU_RESULT
 

def apply_tag(PU_RESULT):
    PU_RESULT['Age_Group'] = PU_RESULT.apply(lambda row: 'Over 60' if row['HPS_AGE_AT_ADMISSION_DATE'] >=60 else 'Under 60',axis=1)
    PU_RESULT['Gender_Group'] = PU_RESULT.apply(lambda row: 'Male' if row['PERSON_GENDER_CODE_DESC'] =='Sex is male' else ('Female' if row['PERSON_GENDER_CODE_DESC'] =='Sex is female' else 'Other'),axis=1)
    PU_RESULT['Risk_Group'] = PU_RESULT.apply(lambda row: 'PU High Risk' if row['WL - Waterlow Score'] >10 else 'PU Low Risk',axis=1)
    PU_RESULT['PU_Group'] = PU_RESULT.apply(lambda row: 'PU Patient' if row['Label'] =='Diagnosed_PU' else 'No PU',axis=1)
    PU_RESULT['Surgery_Group'] = PU_RESULT.apply(lambda row: 'Surgical Patient' if row['Surgery'] =='1' else 'No Surgical',axis=1)
    PU_RESULT['BMI_Group'] = PU_RESULT.apply(lambda row: 'Unknown BMI - Missing value' if (row['height']==0 or row['weight'] ==0)else ('Obese' if (row['weight']/row['height'])*100 >=30  else ('Under Weight' if (row['weight']/row['height'])*100 <18.5 else ('Healthy' if 18.5<=(row['weight']/row['height'])*100<25 else 'Over Weight' ))),axis=1)    
    PU_RESULT['Cohort_Group'] = PU_RESULT[['Age_Group', 'Gender_Group', 'Risk_Group','PU_Group','BMI_Group','Surgery_Group']].agg(','.join, axis=1)
    PU_RESULT['Waterloo Assessment pass'] = PU_RESULT.apply(lambda row: 1 if row['Waterlow_Standard'] ==  "{'rule 1': 'Pass', 'rule 2': 'Pass', 'rule 3': 'Pass', 'rule 4': 'Pass'}" else 0,axis=1)
    PU_RESULT['Waterloo Assessment fail'] = PU_RESULT.apply(lambda row: 1 if row['Waterlow_Standard'] !=  "{'rule 1': 'Pass', 'rule 2': 'Pass', 'rule 3': 'Pass', 'rule 4': 'Pass'}" else 0,axis=1)
    PU_RESULT['Waterloo Assessment on time'] = PU_RESULT.apply(lambda row: 1 if row['Waterlow_Standard'] ==  "{'rule 1': 'Pass', 'rule 2': 'Pass', 'rule 3': 'Pass', 'rule 4': 'Pass'}" else 0,axis=1)
    PU_RESULT['Waterloo Assessment not on time'] = PU_RESULT.apply(lambda row: 1 if row['Waterlow_Standard'] !=  "{'rule 1': 'Pass', 'rule 2': 'Pass', 'rule 3': 'Pass', 'rule 4': 'Pass'}" else 0,axis=1)
    PU_RESULT['PU plan on time'] = PU_RESULT.apply(lambda row: 1 if (row['careplan_ontime'] in  ([1]) )else 0,axis=1)
    PU_RESULT['PU plan not on time'] = PU_RESULT.apply(lambda row: 1 if (row['careplan_ontime'] not in  ([1]) )else 0,axis=1)
    PU_RESULT['Re-positioning on time'] = PU_RESULT.apply(lambda row: 1 if (row['Careplan'] in  (['No careplan', 'No risk',"0,0"]) )else 0,axis=1)
    PU_RESULT['Re-positioning not on time'] = PU_RESULT.apply(lambda row: 1 if (row['Careplan'] not in  (['No careplan', 'No risk',"0,0"]) )else 0,axis=1)
    PU_RESULT['Careplan Compliance'] = PU_RESULT.apply(lambda row: 0 if (float(row['careplan_compliance_degree']) ==0) else (1 if float(row['careplan_compliance_degree'])<0.5 else (2 if 0.5<float(row['careplan_compliance_degree'])<0.8 else 3 )),axis=1)
    PU_RESULT['Repositioning Compliance'] = PU_RESULT.apply(lambda row: 0 if (float(row['careplan_compliance_degree']) ==0) else (1 if float(row['careplan_compliance_degree'])<0.5 else (2 if 0.5<float(row['careplan_compliance_degree'])<0.8 else 3 )),axis=1)



    Reference_Range,input_node_fields = CR_inputs_reference()   
    PU_RESULT = check_blood_normal(Reference_Range,input_node_fields,PU_RESULT)
    Reference_Range,input_node_fields = Hb1AC_inputs_reference()
    PU_RESULT = check_blood_normal(Reference_Range,input_node_fields,PU_RESULT)
    Reference_Range,input_node_fields = Albumin_inputs_reference()
    PU_RESULT = check_blood_normal(Reference_Range,input_node_fields,PU_RESULT)
    Reference_Range,input_node_fields = FBC_inputs_reference()
    PU_RESULT = check_blood_normal(Reference_Range,input_node_fields,PU_RESULT)
    PU_RESULT=PU_RESULT.fillna(0)
    return PU_RESULT

def data_load_clean():
    Red004_Conn = Red004()
    PU_RESULT = pd.read_sql_query('SELECT * from [AdvancedAnalytics].[dbo].[Variance_Analysis]',Red004_Conn)
    HRG_stat = pd.read_sql_query('SELECT [FY] ,[HRG_CODE], [HRG_NAME] ,[ORDINARY_ELECTIVE_LONG_STAY_TRIMPOINT_DAYS] ,[NON_ELECTIVE_LONG_STAY_TRIMPOINT_DAYS] FROM [LOCAL_REFERENCE_DB].[ref].[NATIONAL_TARIFF_APC_OPROC] \
  where FY = \'2020/2021\'',Red004_Conn)
    Red004_Conn.close() 
    
    PU_RESULT = PU_RESULT[~PU_RESULT['Label'].str.contains('Empty')]
    PU_RESULT=PU_RESULT.fillna(0)
    encode_list=[#'Chief Complaint SNOMED Code'
            #,'PRESENTING_COMPLAINT'
            'modelled_events'
            ,'all_codes'
             ,'all_hrg'
            ,'WARD STAY LOCATION'
            ,'ETHNIC_CATEGORY_CODE'
            ,'PERSON_MARITAL_STATUS_CODE'
            ,'PERSON_GENDER_CODE_DESC'
            ,'ETHNIC_CATEGORY_CODE_DESC'
            ,'RELIGIOUS_OR_OTHER_BELIEF_SYSTEM_AFFILIATION'
            ,'PERSON_MARITAL_STATUS_CODE_DESC'
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
                PU_RESULT[column] = PU_RESULT[column].replace(' ', np.NaN).replace(['/'], np.NaN).replace('----',np.NaN).replace('See Lab Comment:',np.NaN).replace('----',np.NaN, regex=True).replace('[a-zA-Z]',np.NaN,regex=True).astype(float)
            except Exception as e:
                if column == 'C-reactive protein':
                    PU_RESULT[column] = PU_RESULT[column].replace('<1', 0.5).replace(['/'], np.NaN).replace('<0.2', 0.5).replace('<0.3', 0.5).replace('<0.6', 0.5).replace(' ', np.NaN).replace('[a-zA-Z]',np.NaN,regex=True).astype(float)
                elif column =='Glucose, CSF':
                    PU_RESULT[column] = PU_RESULT[column].replace('<0.1', 0.1).replace('<0.2', 0.5).replace('<0.3', 0.5).replace(' ', np.NaN).replace('[a-zA-Z]',np.NaN,regex=True).astype(float)
                elif e == 'cannot astype a datetimelike from [datetime64[ns]] to [float64]':
                    pass
    # try:
    #     PU_RESULT['all_hrg'] = PU_RESULT.apply(lambda row: list(set(row['all_hrg'].split(","))) if row['all_hrg'] != 0 else row['all_hrg'],axis=1)
    #     PU_RESULT['all_hrg']=PU_RESULT['all_hrg'].apply(str)
    # except:
    #     pass
    PU_RESULT=PU_RESULT.fillna(0)
    for index,row in PU_RESULT.iterrows():
        #print(row['all_hrg'].strip("[']"))
        try:
            upper_boundary = int(HRG_stat[HRG_stat['HRG_CODE'] ==  row['all_hrg'].strip("[']")]['NON_ELECTIVE_LONG_STAY_TRIMPOINT_DAYS'])*3600*24
            lower_boundary = 2
            condition = 'Abnormal'
            if 2< row['Total_LOS'] <= upper_boundary:
                condition = 'Normal'
            PU_RESULT.at[index,'Expected_LOS'] = condition
        except:
            PU_RESULT.at[index,'Expected_LOS'] = "Unknown"
    print(len(PU_RESULT))
    PU_RESULT = transform(PU_RESULT)
    print("Transform finished.")
    print(len(PU_RESULT))
    utils_pickle.write(PU_RESULT,"PU_RESULT")
    PU_RESULT= apply_tag(PU_RESULT)
    utils_pickle.write(PU_RESULT,"PU_RESULT")
    column_map = {"Sex":'PERSON_GENDER_CODE_DESC', "Ethcity":'ETHNIC_CATEGORY_CODE_DESC'}
    list_dummy_column_map={}
    for item in column_map:
        dummy_column_map, PU_RESULT = get_dummy_list(column_map, PU_RESULT, item)
        list_dummy_column_map.update(dummy_column_map)
    #HRG_TLOS_AVG = pd.DataFrame(PU_RESULT.groupby(['activity_identifier','all_hrg'])['Total_LOS'].mean()).reset_index().groupby(['all_hrg'])['Total_LOS'].mean()
    #HRG_TLOS_STD = pd.DataFrame(PU_RESULT.groupby(['activity_identifier','all_hrg'])['Total_LOS'].mean()).reset_index().groupby(['all_hrg'])['Total_LOS'].std()
    #avg_=pd.DataFrame(pd.DataFrame(PU_RESULT.groupby(['activity_identifier','all_hrg'])['Total_LOS'].mean()).reset_index().groupby(['all_hrg'])['Total_LOS'].mean()).reset_index()
    #std_=pd.DataFrame(pd.DataFrame(PU_RESULT.groupby(['activity_identifier','all_hrg'])['Total_LOS'].mean()).reset_index().groupby(['all_hrg'])['Total_LOS'].std()).reset_index()
    #count_=pd.DataFrame(pd.DataFrame(PU_RESULT.groupby(['activity_identifier','all_hrg'])['Total_LOS'].mean()).reset_index().groupby(['all_hrg'])['Total_LOS'].count()).reset_index()
    #hrg_stat = avg_.merge(std_, on='all_hrg',how='left').merge(count_, on='all_hrg',how='left')

    
    utils_pickle.write(list_dummy_column_map, "PU_RESULT_DUMMY_COLUMNS")
    utils_pickle.write(PU_RESULT,"PU_RESULT")
    return

def PU_Demo():
    Label=['No-diagnose', 'Diagnosed_PU']
    Variable_selection = ['WL - Waterlow Score','Complex_Degree','Global_Central', 'Total_LOS', 'Turnaround_Degree','Mean cell volume', 'Haematocrit', 'Red blood cell count',
       'Basophil count', 'White blood cell count', 'Mean cell haemoglobin',
       'Neutrophil count', 'Eosinophil count', 'Haemoglobin',
       'Lymphocyte count', 'Platelet count', 'Mean cell haemoglobin conc',
       'Monocyte count', 'Haemoglobin A1c IFCC', 'C-reactive protein',
       'Glucose fasting', 'Glucose random', 'Glucose, CSF',
       'Glucose, dialysis fluid', 'Glucose, fluid', 'Albumin']

    Demographic_variable_selection = ['Weight','Sex', 'Age']

    Assessment = ['Waterloo Assessment pass', 'Waterloo Assessment fail', 'Waterloo Assessment on time', 'Waterloo Assessment not on time']

    Prevention = ['PU plan on time','PU plan not on time', 'Re-positioning on time','Re-positioning not on time']

    Patient_Cohort = ['Surgical Patient', 'Medical Patient', 'Ward Outliers','Over 60', 'Over Weight', 'Male', 'Female','PU High Risk', 'PU Patient','NO PU']

    Waterlow_Standard = [
                    'No waterlow', 
       "{'rule 1': 'Pass', 'rule 2': 'Pass', 'rule 3': 'Pass', 'rule 4': 'Pass'}",
       "{'rule 1': 'Pass', 'rule 2': 'Pass', 'rule 3': 'Pass', 'rule 4': 'Fail'}",
       "{'rule 1': 'Pass', 'rule 2': 'Fail', 'rule 3': 'Fail', 'rule 4': 'Fail'}",
       "{'rule 1': 'Pass', 'rule 2': 'Fail', 'rule 3': 'Fail', 'rule 4': 'Pass'}"]
    
    Waterlow_Compliance = ['Rule 1: Use Waterlow', 'Rule 2: 4 Hours Admission', 'Rule 3: AE 4hours', 'Rule 4: Ward Transfer 4hours','No Waterlow']        

    LOS = ["Expected Long for HRG", "Unexpected Long for HRG","Unexpected short - live discharge","Unknown"]

    events_list =['Waterlow','LabTest','Skinasses','TV','PatientPosition'] 
    stats = interact(plot_Spell_PU_Degree_PairCompare,
    Label=widgets.SelectMultiple(
        options=
        Label,
        value=  Label,
        #rows=10,
        description='Pressure Ulcer',
        disabled=False
    ),Expected_LOS=widgets.SelectMultiple(
        options=
        LOS,
        value=  LOS,
        #rows=10,
        description='Expected LOS',
        disabled=False
    ),selected_list=widgets.SelectMultiple(
        options=
        events_list,
        value=  ['Waterlow','LabTest'],
        #rows=10,
        description='Events',
        disabled=False
    )
                    
                    ,Age=widgets.IntRangeSlider(value=[0,120],
        min=0,
        max=120,
        step=1,
        description='Age:',
        disabled=False,
        continuous_update=False,
        orientation='horizontal',
        readout=True,
        readout_format='d')
                    ,Waterlow_Standard=widgets.SelectMultiple(
        options=
        Waterlow_Compliance,
        value=  ['Rule 1: Use Waterlow'] ,
        #rows=10,
        description='WaterlowStandard',
        disabled=False)
                    ,features=widgets.SelectMultiple(
        options=
        Variable_selection,
        value=  ['Complex_Degree','Global_Central', 'Total_LOS', 'Turnaround_Degree','WL - Waterlow Score'] ,
        #rows=10,
        description='Variables',
        disabled=False
    )
                    )
    return stats

def sub_timeline_layout(tt, labels=None):
    node = {}
    for node_index, node_feature in tt.nodes(data=True):
            if node_feature['name'] == 'Spell_Start':
                node.update({node_index:node_feature['activity_start_time']})
            elif node_feature['name'] == 'Spell_End':
                node.update({node_index:node_feature['activity_end_time']})
            else:
                node.update({node_index:node_feature['activity_start_time']})
    # for node_index, node_feature in tt.nodes(data=True):
    #     node.update({node_index:node_feature['activity_start_time']})
    
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

def PU_Path_Vis_Demo_fromRecords(item = 1234567):
      
    # hv.extension('bokeh')
    # defaults = dict(width=1000, height=1000, padding=0.1)
    # Load individual visualization requires patientKG Class     
    graph = patientKG.utils_pickle.read("GraphCalculationResults/Ward_Stay/KG_{}".format(item))
    # hv.opts.defaults(opts.EdgePaths(**defaults), opts.Graph(**defaults), opts.Nodes(**defaults))
    label = dict((int(v),k) for k,v in graph.node_dic.items())
    combined = graph.graph
    # for index, item in combined.nodes(data=True):
    #     print(item['color'])
    # combined._node[6]["WARD STAY LOCATION"]=""
    # combined._node[7]["WARD STAY LOCATION"]=""
    # combined._node[8]["WARD STAY LOCATION"]=""
    # combined._node[9]["WARD STAY LOCATION"]=""   
    attr={}
    for index, node in combined.nodes(data=True):
        if index==0 or index == 1:
            attr.update({index:{'abv': node['name']}})
        else:
            attr.update({index:{'abv':"".join(e[0] for e in node['name'].split())}})
    nx.set_node_attributes(combined, attr) 
    hv.opts.defaults(
        opts.EdgePaths(**defaults), opts.Graph(**defaults), opts.Nodes(**defaults))
    pos = graph.timeline_layout(label)
    optss = dict(node_size='size', edge_line_width=0.5 ,node_color='color', cmap=['#30a2da','yellow','red','green','black'])
    simple_graph=hv.Graph.from_networkx(combined, pos).options(**optss)
    labels = hv.Labels(simple_graph.nodes, ['x', 'y'], 'abv')
    # print(simple_graph.nodes)
    # print(graph.graph.degree)
    #bokeh_server = pn.Row(simple_graph* labels.opts(text_font_size='16pt', text_color='white', bgcolor='gray')).show(port=12345)
    return  pn.Row(simple_graph* labels.opts(text_font_size='16pt', text_color='white', bgcolor='gray'))
    # days_nodes = {}

    # for i in range(0, 16, 1):
    #     nodes_list = [0,1]
    #     for k,v in graph.graph.nodes(data=True):
    #         if k > 1:
    #             diff = (datetime.strptime(v['activity_start_time'],'%Y.%m.%d %H:%M:%S') -datetime.strptime(graph.graph.nodes[0]['activity_start_time'],'%Y.%m.%d %H:%M:%S')).total_seconds()            
    #             if diff <= i*3600*24:
    #                 nodes_list.append(k)                
        
    #     days_nodes.update({i:nodes_list})
    # debug_ = {i: hv.Graph.from_networkx(graph.graph.subgraph(days_nodes[i]),  sub_timeline_layout(graph.graph.subgraph(days_nodes[i]),dict((int(v),k) for k,v in graph.node_dic.items())), iterations=i, seed=10) for i in range(0, 16, 1)}     
    # return hv.HoloMap({i: hv.Graph.from_networkx(graph.graph.subgraph(days_nodes[i]),  sub_timeline_layout(graph.graph.subgraph(days_nodes[i]),dict((int(v),k) for k,v in graph.node_dic.items())), iterations=i, seed=10) for i in range(0, 16, 1)},
    #         kdims='Iterations')

def PU_Path_Vis_Demo_live(item = 1234567):
    #While modelling process is okey, but using Jupyter running as windows service, login detail is'RBBH_MSDOMAIN1\\RBHDBSRED008$', which currently has no access to database
    #Thus result in live querying fail.     
    return test_compose(str(item))

def PU_Path_DEMO():
    try:
        print("\
        WA: Waterlow Assessemnt \n   \
            Node in Red: Not implement. \n \
        Cplb: C-reactive protein level, blood\n \
            Node in Red: test result is out of normal range;\n \
        PP: Patient Position\n \
            Node in red: breach 6 hours repositioning requirement\n \
            Node in yellow: breach 4 hours repositioning requirement\n \
        WS: Ward Stay\n \
            Node in Red: Waterlow assessment not performed within hours after ward transfer\n \
        Fbcb: Full blood count, blood\n \
            Node in Red: test result is out of normal range;")
        stats = interact(PU_Path_Vis_Demo_fromRecords,
        item=widgets.Text(value='1234567',
        placeholder='Type in Spell Number',
        description='Spell:',
        disabled=False))
        return stats
    except:
        return "No Such Spell!"

def generate_cohort_pattern(Patient_Cohort):
    pattern=''
    Union_criteria = [('Male', 'Female'),('PU Patient','NO PU'),('No Surgical','Surgical Patient'),('PU Low Risk', 'PU High Risk')]
    union_BMI = ['Healthy','Under Weight','Over Weight','Obese','Unknown BMI - Missing value']
    tt=[]
    bmi =[]
    for item in Patient_Cohort:
        check = [(x,y) for x, y in Union_criteria if (x  == item or y ==item)]
        if len(check)<1 and item not in union_BMI:
            y = '(?=.*{})'.format(item)
            pattern+=y
        elif item in union_BMI:
            bmi.append(item)
        else:
            tt.append(check)
        

    ttt= [[g[0], len(list(g[1]))] for g in itertools.groupby(tt)]        

    for item in ttt:
        if item[1] > 1:
            pattern+='((?=.*{})|(?=.*{}))'.format(item[0][0][0],item[0][0][1])
        elif item[1] == 1:
            for check_item in Patient_Cohort:
                check = [(x,y) for x, y in Union_criteria if (x  == check_item or y ==check_item)]
                if len(check)==1 and check == item[0]:
                    y = '(?=.*{})'.format(check_item)
                    pattern+=y
    union_pattern=''
    while bmi:
        y = '(?=.*{})|'.format(bmi[0])
        union_pattern+=y
        bmi.pop(0)
    if len(union_pattern)>0:
        union_pattern= "("+union_pattern[:-1]+")"
        pattern+=union_pattern
    return pattern

def get_dummy_list(column_map, df, column):
    column_mapped = [column_map[column]]
    df[column] = df[column_mapped]
    dum_df = pd.get_dummies(df, columns=column_mapped, prefix=["Type_is"] )
    column_diff = list(set(dum_df.columns) - set(df.columns))
    dummy_column_map = {column:column_diff}
    return dummy_column_map, dum_df


def predefined_cohort(Patient_Cohort):
    df = pd.DataFrame()
    PU_RESULT = patientKG.utils_pickle.read("PU_RESULT")
    
    pattern = generate_cohort_pattern(Patient_Cohort)
              
    PU_RESULT_COHORT =PU_RESULT.loc[(PU_RESULT['Cohort_Group'].str.contains(pattern))]

    return PU_RESULT_COHORT   

TEMPLATE = u"""
<!DOCTYPE html>
<html>
    <head>
        <meta charset="UTF-8">
        <title>PivotTable.js</title>

        <!-- external libs from cdnjs -->
        <link rel="stylesheet" type="text/css" href="https://cdnjs.cloudflare.com/ajax/libs/c3/0.4.11/c3.min.css">
        <script type="text/javascript" src="https://cdnjs.cloudflare.com/ajax/libs/d3/3.5.5/d3.min.js"></script>
        <script type="text/javascript" src="https://cdnjs.cloudflare.com/ajax/libs/c3/0.4.11/c3.min.js"></script>
        <script type="text/javascript" src="https://cdnjs.cloudflare.com/ajax/libs/jquery/1.11.2/jquery.min.js"></script>
        <script type="text/javascript" src="https://cdnjs.cloudflare.com/ajax/libs/jqueryui/1.11.4/jquery-ui.min.js"></script>
        <script type="text/javascript" src="https://cdnjs.cloudflare.com/ajax/libs/jquery-csv/0.71/jquery.csv-0.71.min.js"></script>


        <link rel="stylesheet" type="text/css" href="https://cdnjs.cloudflare.com/ajax/libs/pivottable/2.19.0/pivot.min.css">
        <script type="text/javascript" src="https://cdnjs.cloudflare.com/ajax/libs/pivottable/2.19.0/pivot.min.js"></script>
        <script type="text/javascript" src="https://cdnjs.cloudflare.com/ajax/libs/pivottable/2.19.0/d3_renderers.min.js"></script>
        <script type="text/javascript" src="https://cdnjs.cloudflare.com/ajax/libs/pivottable/2.19.0/c3_renderers.min.js"></script>
        <script type="text/javascript" src="https://cdnjs.cloudflare.com/ajax/libs/pivottable/2.19.0/export_renderers.min.js"></script>

        <style>
            body {font-family: Verdana;}
            .node {
              border: solid 1px white;
              font: 10px sans-serif;
              line-height: 12px;
              overflow: hidden;
              position: absolute;
              text-indent: 2px;
            }
            .c3-line, .c3-focused {stroke-width: 3px !important;}
            .c3-bar {stroke: white !important; stroke-width: 1;}
            .c3 text { font-size: 12px; color: grey;}
            .tick line {stroke: white;}
            .c3-axis path {stroke: grey;}
            .c3-circle { opacity: 1 !important; }
            .c3-xgrid-focus {visibility: hidden !important;}
        </style>
    </head>
    <body>
        <script type="text/javascript">
            $(function(){

                $("#output").pivotUI(
                    $.csv.toArrays($("#output").text())
                    , $.extend({
                        renderers: $.extend(
                            $.pivotUtilities.renderers,
                            $.pivotUtilities.c3_renderers,
                            $.pivotUtilities.d3_renderers,
                            $.pivotUtilities.export_renderers
                            ),
                        hiddenAttributes: [""]
                    } 
                    , {
                        onRefresh: function(config) {
                            var config_copy = JSON.parse(JSON.stringify(config));
                            //delete some values which are functions
                            delete config_copy["aggregators"];
                            delete config_copy["renderers"];
                            //delete some bulky default values
                            delete config_copy["rendererOptions"];
                            delete config_copy["localeStrings"];
                            $("#output2").text(JSON.stringify(config_copy, undefined, 2));
                        }
                    }
                    , %(kwargs)s
                    , %(json_kwargs)s)
                ).show();
             });
        </script>
        <div id="output" style="display: none;">%(csv)s</div>

        <textarea id="output2"
        style="float: left; width: 0px; height: 0px; margin: 0px; opacity:0;" readonly>
        </textarea>

        <button onclick="copyTextFunction()">Copy settings</button>
        <script>
        function copyTextFunction() {
                    var copyText = document.getElementById("output2");
                    copyText.select();
                    document.execCommand("copy");
                    }
        </script>

    </body>
</html>
"""

def pivot_cht_ui(df, name="test", url="",
    width="100%", height="500",json_kwargs='',  **kwargs):
  #print(name)
  outfile_path = name + '.html'
  with io.open(outfile_path, 'wt', encoding='utf8') as outfile:
      csv = df.to_csv(encoding='utf8')
      if hasattr(csv, 'decode'):
          csv = csv.decode('utf8')
      outfile.write(TEMPLATE %
            dict(csv=csv, kwargs=json.dumps(kwargs),json_kwargs=json_kwargs))
  
  return IFrame(src=url or outfile_path, width=width, height=height)

def get_pvalue(df,feature_list,categorical_feature_list):
    rows_list = []
    outcome_number  = len(df['Label'].unique())
    for item in df[feature_list].columns:
        if item not in categorical_feature_list:
            dia_list = []
            undiag_list = []
            for label in df['Label'].unique():
                if label == 'Diagnosed_PU':
                    dia_list.append(df[df['Label']==label][item].values)
                else:
                    undiag_list.append(df[df['Label']==label][item].values)
            dd=[]
            ddd = []
            for da_item in list(dia_list[0]):    
                dd.append(da_item)
            for und_item in list(undiag_list[0]):    
                ddd.append(und_item)
            fvalue, pvalue = stats.f_oneway(*[dd,ddd])
            rows_list.append((item,pvalue)) 
        else:
            dict1 = {}
            contigency= pd.crosstab(df[item], df['Label']) 
            c, p, dof, expected = chi2_contingency(contigency)
            dict1.update({item:p})
            rows_list.append((item,p))
    return rows_list

#def plot_func(df, Outcome, Patient_Cohort, DateRange,Demographic, Assessment, Prevention, Blood_Results, Blood_Normalty,Management):
    


def plot_Spell_PU_Degree_PairCompare_v2(Outcome, Patient_Cohort, DateRange,Demographic, Assessment, Prevention, Blood_Results, Blood_Normalty,Management):    
    
    #Corhot Selection
    #print(list(Patient_Cohort))
    if list(Patient_Cohort) == ['All']:
        df= patientKG.utils_pickle.read("PU_RESULT")
        #plot_func(df, Outcome, Patient_Cohort, DateRange,Demographic, Assessment, Prevention, Blood_Results, Blood_Normalty,Management)
    else:
        Patient_Cohort = list(Patient_Cohort)
        try:
            df=predefined_cohort(Patient_Cohort)
            if df.empty:
                return "No Sample!"
            #else:
                #plot_func(df, Outcome, Patient_Cohort, DateRange,Demographic, Assessment, Prevention, Blood_Results, Blood_Normalty,Management)
            # df = pd.concat([df,data])
        except:
            print("No Sample!")
            sys.exit(1) 
            return "No Sample!" 
    def pop_std(x):
        return x.std(ddof=0)
    df['date'] = pd.to_datetime(df['HPS_ACTIVITY_DATE_TIME'])
    mask = (df['date'] > DateRange[0]) & (df['date'] <= DateRange[1])
    df = df.loc[mask]
    
    features = []
    df= df.fillna(0)
    Demographic_map = {"Weight":"weight", 
    "Sex":['Type_is_Sex is female', 'Type_is_Sex is male','Type_is_Sex is unknown', 'Type_is_Unspecified']
    ,"Age":'HPS_AGE_AT_ADMISSION_DATE'
    ,"Ethcity":'ETHNIC_CATEGORY_CODE_DESC'}

    Assessment_map = {"Waterlow Assessment Outcomes":"Waterloo Assessment pass", "Waterloo Assessment fail":"Waterloo Assessment fail", "Waterlow Assessment timeliness":"Waterloo Assessment on time","Waterloo Assessment not on time":"Waterloo Assessment not on time" }

    Prevention_map = {'PU plan initia timeliness':'PU plan on time','PU plan not on time':'PU plan not on time', 'Re-positioning timeliness':'Re-positioning on time','Re-positioning not on time':'Re-positioning not on time','Re-positioning Compliance':'Repositioning Compliance'}

    Management_map={'Ward Move':'ward_move'}

    One_hot_encoding_map= utils_pickle.read("PU_RESULT_DUMMY_COLUMNS")

    for item in Demographic:
        if item not in One_hot_encoding_map.keys():
            features.append(Demographic_map[item])
        else:
            features = features +list(One_hot_encoding_map[item])

    for item in Assessment:        
        if item not in One_hot_encoding_map.keys():
            features.append(Assessment_map[item])
        else:
            features = features +list(One_hot_encoding_map[item])
    
    for item in Prevention:        
        if item not in One_hot_encoding_map.keys():
            features.append(Prevention_map[item])
        else:
            features = features +list(One_hot_encoding_map[item])
    
    for item in Blood_Results:
        if item not in One_hot_encoding_map.keys():
            features.append(item)
        else:
            features = features +list(One_hot_encoding_map[item])
    
    for item in Blood_Normalty:
        if item not in One_hot_encoding_map.keys():
            features.append(item+'_normal')
        else:
            features = features +list(One_hot_encoding_map[item])
    
    for item in Management:
        if item not in One_hot_encoding_map.keys():
            features.append(Management_map[item])
        else:
            features = features +list(One_hot_encoding_map[item])


    #features = ['Sum_Degree','Global_Central', 'Total_LOS', 'Turnaround_Degree']  
    try:
        principalComponents,pca_explained,pca_components = sliced_principle_components(df,features,2)
        principalDf = pd.DataFrame(data = principalComponents, columns = ['principal component 1', 'principal component 2'])
        kmax = 10
        best_n = best_eblow_k(principalDf.values.tolist(),kmax = 10) 
        try:
            df = plot_vor(df,principalComponents, best_n)    
        except:
            df = plot(df,principalComponents, best_n)
        #print(list(features))
        
        #Section Outcomes to Estimated groups
        total = df['activity_identifier'].nunique()
        outcomes = df.groupby(['labels','Label'])['activity_identifier'].nunique()
        #print("Sankey")
        fig = sankey(df)        
        fig.show()   

        #Section phenotype table with variables
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
        categorical_feature_list = []
        for item in Blood_Normalty:
            if item not in One_hot_encoding_map.keys():
                categorical_feature_list.append(item+'_normal')
        for item in Assessment:        
            if item not in One_hot_encoding_map.keys():
                categorical_feature_list.append(Assessment_map[item])        
        
        for item in Prevention:        
            if item not in One_hot_encoding_map.keys():
                categorical_feature_list.append(Prevention_map[item])

        for key,value in One_hot_encoding_map.items():
            for item in value:
                categorical_feature_list.append(item)  
        
        for item in df[feature_list].columns:
            if item not in categorical_feature_list:
                sub_df = df[df[item]!=0]
                #test1=sub_df.groupby(['labels'])[item].agg(item='mean').T
                #test2=sub_df.groupby(['labels'])[item].agg(item=pop_std).T
                test1=sub_df.groupby(['labels'])[item].agg([(item, 'mean')]).T
                test2=sub_df.groupby(['labels'])[item].agg([(item, pop_std)]).T
                #print(test1)
                #print('Test 1 over')
                #print(test2)
                #test1=sub_df.groupby(['labels'])[item].agg({item:'mean'}).T
                #test2=sub_df.groupby(['labels'])[item].agg({item:pop_std}).T
                test4 =pd.DataFrame()
                for item in sub_df['labels'].unique():
                    test4[item] = test1[item].round(2).astype(str)+" (\u00B1"+test2[item].round(2).astype(str)+")"    
            else:
                test1=df.groupby(['labels',item])[item].agg([(item, 'count')]).T
                test2=pd.DataFrame(df.groupby(['labels',item])[item].agg([('count', 'count')]).reset_index())                
                #print(test1)
                #print(test2)

                #test1=df.groupby(['labels',item])[item].agg(item='count').T
                #test2=pd.DataFrame(df.groupby(['labels',item])[item].agg(count='count').reset_index())

                #test1=df.groupby(['labels',item])[item].agg({item:'count'}).T
                #test2=pd.DataFrame(df.groupby(['labels',item])[item].agg({'count':'count'}).reset_index())                
                
                test4 =pd.DataFrame()
                index = pd.Index([item])
                test4['label'] =[item]
                for label in df['labels'].unique():
                    try:
                        zero_count = str(int(test2[(test2['labels']==label) & (test2[item]==0)]['count']))
                    except:
                        zero_count = 0

                    try:
                        one_count = str(int(test2[(test2['labels']==label) & (test2[item]==1)]['count']))
                    except:
                        one_count = 0
                    test4[label] = [str(one_count)+" ("+str(zero_count)+")" ]
                test4=test4.set_index('label')               
            test3=test3.append(test4)
        
        
        rows_list = get_pvalue(df,feature_list,categorical_feature_list) 
        pvalue = pd.DataFrame(rows_list, columns = ['variable_name','p-value (ANOVA&Chi-square)'])
        
        test3 = test3.reindex(sorted(test3.columns),axis=1)
        test3['variable_name'] = test3.index
        #print(test3)
        test3 =test3.merge(feature_rank, how='left', on='variable_name')
        test3 = test3.merge(pvalue, how='left', on='variable_name')
        #test3 = test3.set_index('variable_name')
        test5=pd.DataFrame(df.groupby(['labels'])['activity_identifier'].agg([('activity_identifier', lambda x: x.nunique())]).T)
        #test5=pd.DataFrame(df.groupby(['labels'])['activity_identifier'].agg({'activity_identifier':lambda x: x.nunique()}).T)
        #test5=pd.DataFrame(df.groupby(['labels'])['activity_identifier'].agg(activity_identifier=lambda x: x.nunique()).T)
        test3 = test3.sort_values(by='score',ascending=False)
        #print(test5)
        test3=pd.concat([test5,test3]).set_index('variable_name')
        display(test3)

        all_features = feature_list.copy()  
        if len(feature_list)>5:              
            feature_list = list(test3.sort_values(by='score',ascending=False).index[:5].values)
        feature_list.append('labels')
        lis_ = df[['labels','color']].drop_duplicates()
        palette={y['labels']:str(y['color']) for x,y in lis_.iterrows()}
        #sns.pairplot(df[feature_list], hue="labels", diag_kws={'bw':'1.0'},palette=palette)
        #df[(df['POD_CODE'] == POD)][['ACTIVITY_IDENTIFIER','SpellHRG']+features+ ['ReAdmitted in DAYS','POD_CODE','ALL_DIAG','labels']]
        #df[(df['Waterlow_Standard'] == Waterlow_Standard)][['ACTIVITY_IDENTIFIER']+features+ ['labels']]
        #Known Jupyternotebook issue of nondeterministic output display
        #time.sleep(100)
        features_display= []
        for item in all_features:
            if 'Type_is' not in item:
                features_display.append(item)
        utils_pickle.write(df[['activity_identifier','Sex','Ethcity']+features_display+ ['Waterlow_Standard','Careplan','Label','labels','Expected_LOS','Cohort_Group']],"Cohort_Analysis")
        return pivot_cht_ui(df[['activity_identifier','Sex','Ethcity']+features_display+ ['Waterlow_Standard','Careplan','Label','labels','Expected_LOS']], outfile_path='pivottablejs.html',json_kwargs="""
        {
        "derivedAttributes": {},
        "hiddenAttributes": [
                ""
            ],
            "hiddenFromAggregators": [],
            "hiddenFromDragDrop": [],
            "menuLimit": 500,
            "cols": [],
            "rows": [
                "activity_identifier",
                "Label",
                "labels"
            ],
            "vals": [
                "activity_identifier"
            ],
            "rowOrder": "key_a_to_z",
            "colOrder": "key_a_to_z",
            "exclusions": {},
            "inclusions": {},
            "unusedAttrsVertical": 85,
            "autoSortUnusedAttrs": false,
            "sorters": {},
            "outfile_path": "pivottablejs.html",
            "inclusionsInfo": {},
            "aggregatorName": "Count Unique Values",
            "rendererName": "Table"
            }
        """
        )    
    except:
        print("Something Wrong with Sample Size or Calculation! Display pivot table only!")
        utils_pickle.write(df[['activity_identifier','Sex','Ethcity']+features_display+ ['Waterlow_Standard','Careplan','Label','labels','Expected_LOS','Cohort_Group']],"Cohort_Analysis")
        return pivot_cht_ui(df[['activity_identifier','Sex','Ethcity']+ ['Waterlow_Standard','Careplan','Label','Expected_LOS']], outfile_path='pivottablejs.html',json_kwargs="""
        {
        "derivedAttributes": {},
        "hiddenAttributes": [
                ""
            ],
            "hiddenFromAggregators": [],
            "hiddenFromDragDrop": [],
            "menuLimit": 500,
            "cols": [],
            "rows": [
                "activity_identifier",
                "Label"
                
            ],
            "vals": [
                "activity_identifier"
            ],
            "rowOrder": "key_a_to_z",
            "colOrder": "key_a_to_z",
            "exclusions": {},
            "inclusions": {},
            "unusedAttrsVertical": 85,
            "autoSortUnusedAttrs": false,
            "sorters": {},
            "outfile_path": "pivottablejs.html",
            "inclusionsInfo": {},
            "aggregatorName": "Count Unique Values",
            "rendererName": "Table"
            }
        """
        )                      
    #print(len(df))
    #print(DateRange)
    
def PU_Demo_v2():
    
    Blood_Normalty = ['Red blood cell count'
                            ,'Mean cell haemoglobin'
                            , 'Haemoglobin'
                            , 'Haematocrit'
                            , 'Platelet count'
                            , 'Mean cell volume'
                            , 'Mean cell haemoglobin conc'
                            , 'White blood cell count'
                            , 'Monocyte count'
                            , 'Neutrophil count'
                            , 'Lymphocyte count'
                            , 'Eosinophil count'
                            , 'Basophil count'     
                            ,'Albumin'
                            ,'C-reactive protein'
                            ,'Haemoglobin A1c IFCC']

    Blood_Results = [
       'Glucose fasting', 'Glucose random', 'Glucose, CSF',
       'Glucose, dialysis fluid', 'Glucose, fluid']

    Demographic = [
        'Weight',
        'Sex', 'Age','Ethcity']

    Assessment = ['Waterlow Assessment Outcomes', 'Waterlow Assessment timeliness']

    Prevention = ['PU plan initia timeliness', 'Re-positioning timeliness','Re-positioning Compliance']

    Patient_Cohort = ['All',
        'Surgical Patient'
        ,'No Surgical'
    #,'Medical Patient'
    #,'Ward Outliers'
    ,'Over 60'
    ,'Unknown BMI - Missing value'    
    ,'Under Weight'
    ,'Healthy'
    ,'Over Weight'
    ,'Obese'
    , 'Male'
    , 'Female'
    ,'PU High Risk'
    ,'PU Low Risk'
    ]

    Outcomes= [
    'Pressure Ulcer', 'Length of Stay']

    
    Management = ['Ward Move']

    #Waterlow_Compliance = ['Rule 1: Use Waterlow', 'Rule 2: 4 Hours Admission', 'Rule 3: AE 4hours', 'Rule 4: Ward Transfer 4hours','No Waterlow']        

    #LOS = ["Expected Long for HRG", "Unexpected Long for HRG","Unexpected short - live discharge","Unknown"]

    #events_list =['Waterlow','LabTest','Skinasses','TV','PatientPosition'] 

    start_date = datetime(2018, 6, 12)
    end_date = datetime(2021, 6, 22)
    dates = pd.date_range(start_date, end_date, freq='D')
    options = [(date.strftime(' %d%b%y '), date) for date in dates]
    index = (0, len(options)-1)

    layout = widgets.Layout(width='500px', height='100px')
    style={'description_width': '250px'}
    selection_range_slider = widgets.SelectionRangeSlider(
        options=options,
        index=index,
        description='Dates',
        orientation='horizontal',
        layout={'width': '700px'}, style=style
    )
    
    Outcome=widgets.SelectMultiple(
        options=
        Outcomes,
        value=  ['Pressure Ulcer'],
        #rows=10,
        description='Outcomes',
        disabled=False,layout=layout, style=style)

    Patient_Cohort=widgets.SelectMultiple(
        options=
        Patient_Cohort,
        value=  ['Surgical Patient'],
        #rows=10,
        description='Patient Cohort',
        disabled=False,layout=layout, style=style)

    Demographic=widgets.SelectMultiple(
        options=
        Demographic,
        value=  Demographic,
        #rows=10,
        description='Demographic',
        disabled=False,layout=layout,style=style
    )
    Assessment=widgets.SelectMultiple(
        options=
        Assessment,
        value=  Assessment,
        #rows=10,
        description='Clinical Events – PU Risk Assessment',
        disabled=False,layout=layout,style=style
    )                    
    Prevention =widgets.SelectMultiple(
        options=
        Prevention,
        value=  Prevention ,
        #rows=10,
        description='Clinical Events – PU Prevention',
        disabled=False,layout=layout,style=style)
    Blood_Results =widgets.SelectMultiple(
        options=
        Blood_Results,
        value=  Blood_Results ,
        #rows=10,
        description='Laboratory – Other',
        disabled=False,layout=layout,style=style) 

    Blood_Normalty =widgets.SelectMultiple(
        options=
        Blood_Normalty,
        value=  Blood_Normalty,
        #rows=10,
        description='Laboratory – bloods',
        disabled=False,layout=layout,style=style)
    Management =widgets.SelectMultiple(
        options=
        Management,
        value=  Management,
        #rows=10,
        description='Patient Management',
        disabled=False,layout=layout,style=style)    
    # from ipywidgets import Button, Layout, jslink, IntText, IntSlider
    # TwoByTwoLayout(top_left=Blood_Results,
    #             top_right=Demographic,
    #             bottom_left=Assessment,
    #             bottom_right=Prevention)
    
    #left_box = VBox([Patient_Cohort])
    #right_box = VBox([Demographic,Assessment,Prevention,Blood_Results])
    #ui = HBox([left_box, right_box])
    #stats = interactive_output(plot_Spell_PU_Degree_PairCompare_v2,
     #{'Patient_Cohort':Patient_Cohort,'Demographic':Demographic,'Assessment':Assessment,'Prevention':Prevention,'Blood_Results':Blood_Results})
    stats = interact_manual(plot_Spell_PU_Degree_PairCompare_v2,
    Outcome=Outcome, Patient_Cohort=Patient_Cohort,DateRange=selection_range_slider,Demographic=Demographic,Assessment=Assessment,Prevention=Prevention,Blood_Results=Blood_Results, Blood_Normalty=Blood_Normalty,Management=Management)
    # def create_expanded_button(description, button_style):
    #     return Button(description=description, button_style=button_style, layout=Layout(height='auto', width='auto'))

    # top_left_button = create_expanded_button("Top left", 'info')
    # top_right_button = create_expanded_button("Top right", 'success')
    # bottom_left_button = create_expanded_button("Bottom left", 'danger')
    # bottom_right_button = create_expanded_button("Bottom right", 'warning')

    # top_left_text = IntText(description='Top left', layout=Layout(width='auto', height='auto'))
    # top_right_text = IntText(description='Top right', layout=Layout(width='auto', height='auto'))
    # bottom_left_slider = IntSlider(description='Bottom left', layout=Layout(width='auto', height='auto'))
    # bottom_right_slider = IntSlider(description='Bottom right', layout=Layout(width='auto', height='auto'))
    
    
                    
    #display(ui,stats)
    
    return stats
    # TwoByTwoLayout(top_left=top_left_button,
    #            top_right=top_right_button,
    #            bottom_left=bottom_left_button,
    #            bottom_right=bottom_right_button)
   
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
    # Age = [0,120]
    # Waterlow_Standard = ['Rule 1: Use Waterlow']#"{'rule 1': 'Pass', 'rule 2': 'Pass', 'rule 3': 'Pass', 'rule 4': 'Pass'}"
    # features = ['Complex_Degree','Global_Central', 'Total_LOS', 'Turnaround_Degree','WL - Waterlow Score']
    # plot_Spell_PU_Degree_PairCompare(Label,modelled_events,Age,Waterlow_Standard,features)
    #PU_Path_Vis_Demo()    
    #data_load_clean()

    Blood_Normalty = ['Red blood cell count'
                            ,'Mean cell haemoglobin'
                            , 'Haemoglobin'
                            , 'Haematocrit'
                            , 'Platelet count'
                            , 'Mean cell volume'
                            , 'Mean cell haemoglobin conc'
                            , 'White blood cell count'
                            , 'Monocyte count'
                            , 'Neutrophil count'
                            , 'Lymphocyte count'
                            , 'Eosinophil count'
                            , 'Basophil count'     
                            ,'Albumin'
                            ,'C-reactive protein'
                            ,'Haemoglobin A1c IFCC']

    Blood_Results = [
       'Glucose fasting', 'Glucose random', 'Glucose, CSF',
       'Glucose, dialysis fluid', 'Glucose, fluid']


    Demographic = [
        'Weight',
        'Sex', 'Age','Ethcity']

    Assessment = ['Waterlow Assessment Outcomes', 'Waterlow Assessment timeliness']

    Prevention = ['PU plan initia timeliness', 'Re-positioning timeliness']

    Patient_Cohort = [
        'Surgical Patient',
          'Unknown BMI - Missing value'        
        ]
    Management = ['Ward Move']
    start_date = datetime(2018, 6, 12)
    end_date = datetime(2021, 6, 22)

    DateRange = ['20180612','20210622']
    Outcome=[]
    plot_Spell_PU_Degree_PairCompare_v2(Outcome, Patient_Cohort,DateRange, Demographic, Assessment, Prevention, Blood_Results,Blood_Normalty,Management)

    PU_Path_Vis_Demo_live()

    