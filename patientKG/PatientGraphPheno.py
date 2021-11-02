import matplotlib.pyplot as plt
import numpy as np
from holoviews import opts
from scipy.spatial import Voronoi, voronoi_plot_2d
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import pandas as pd
from kneed import KneeLocator

def voronoi_finite_polygons_2d(vor, radius=None):
    """
    Reconstruct infinite voronoi regions in a 2D diagram to finite
    regions.

    Parameters
    ----------
    vor : Voronoi
        Input diagram
    radius : float, optional
        Distance to 'points at infinity'.

    Returns
    -------
    regions : list of tuples
        Indices of vertices in each revised Voronoi regions.
    vertices : list of tuples
        Coordinates for revised Voronoi vertices. Same as coordinates
        of input vertices, with 'points at infinity' appended to the
        end.

    """

    if vor.points.shape[1] != 2:
        raise ValueError("Requires 2D input")

    new_regions = []
    new_vertices = vor.vertices.tolist()

    center = vor.points.mean(axis=0)
    if radius is None:
        radius = vor.points.ptp().max()*2

    # Construct a map containing all ridges for a given point
    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))

    # Reconstruct infinite regions
    for p1, region in enumerate(vor.point_region):
        vertices = vor.regions[region]

        if all([v >= 0 for v in vertices]):
            # finite region
            new_regions.append(vertices)
            continue

        # reconstruct a non-finite region
        ridges = all_ridges[p1]
        new_region = [v for v in vertices if v >= 0]

        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                # finite ridge: already in the region
                continue

            # Compute the missing endpoint of an infinite ridge

            t = vor.points[p2] - vor.points[p1] # tangent
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])  # normal

            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far_point = vor.vertices[v2] + direction * radius

            new_region.append(len(new_vertices))
            new_vertices.append(far_point.tolist())

        # sort region counterclockwise
        vs = np.asarray([new_vertices[v] for v in new_region])
        c = vs.mean(axis=0)
        angles = np.arctan2(vs[:,1] - c[1], vs[:,0] - c[0])
        new_region = np.array(new_region)[np.argsort(angles)]

        # finish
        new_regions.append(new_region.tolist())

    return new_regions, np.asarray(new_vertices)

def calculate_WSS(points, kmax):    
    sse =[]
    for k in range(1, kmax+1):
        kmeans=KMeans(n_clusters=k).fit(points)
        centroids=kmeans.cluster_centers_
        pred_clusters = kmeans.predict(points)
        curr_sse = 0

        for i in range(len(points)):
            curr_center = centroids[pred_clusters[i]]            
            curr_sse += (points[i][0]-curr_center[0])**2+(points[i][1]-curr_center[1])**2
    
        sse.append(curr_sse)
    return sse

def sliced_principle_components(dataframe, features = ['Sum_Degree','Global_Central', 'Total_LOS','Turnaround_Degree'],max_components=2):    
    # Separating out the features
    x = dataframe.loc[:, features].values
    # Separating out the target
    #y = test.loc[:,['target']].values
    # Standardizing the features
    x = StandardScaler().fit_transform(x)
    pca = PCA(n_components=max_components)
    principalComponents = pca.fit_transform(x)
    #print(pca.explained_variance_ratio_)
    #print(abs(pca.components_))
    return principalComponents,pca.explained_variance_ratio_,abs(pca.components_)

def best_eblow_k(points, kmax=10, vis_elbow = False):
    sse = calculate_WSS(points,kmax)
    x = range(1,len(sse)+1)
    kn=KneeLocator(x, sse,curve='convex',direction ='decreasing')
    if vis_elbow == True:
        plt.xlabel('number of clusters k')
        plt.ylabel('Sum of squared distances')
        plt.plot(x, sse, 'bx-')
        plt.vlines(kn.knee, plt.ylim()[0], plt.ylim()[1], linestyles='dashed')
        plt.show()
    return kn.knee

def silhouette(dataframe,kmax,vis_sil=False):
    sil=[]
    for k in range(2, kmax+1):
        kmean_test = KMeans(n_clusters = k).fit(dataframe)
        labels_test = kmean_test.labels_
        sil.append(silhouette_score(dataframe,labels_test,metric='euclidean'))
    number_of_cluster = sil.index(max(sil)) + 2
    x = range(1,len(sil)+1)
    if vis_sil == True:
        plt.xlabel('number of clusters k')
        plt.ylabel('Silhouette Score')
        plt.plot(x, sil, 'bx-')        
        plt.show()
    return number_of_cluster

def get_cols_marks(best_n,labels):
    colmap = {1: 'g', 2: 'r', 3: 'b', 4:'y',5:'c'}
    marker = {1:'circle', 2:'diamond', 3:'dot', 4:'triangle',5:'x'}
    size = {1:2,2:2,3:2,4:2,5:2}
    colors = list(map(lambda x: colmap[x+1], labels))
    markers = list(map(lambda x: marker[x+1], labels))
    sizes = list(map(lambda x: size[x+1], labels))
    return colors,markers,sizes

def plot_vor(df,principalComponents,best_n):
    principalDf = pd.DataFrame(data = principalComponents, columns = ['principal component 1', 'principal component 2'])
    kmeans = KMeans(n_clusters=best_n)   
    kmeans.fit(principalDf)    
    labels = kmeans.predict(principalDf)
    centroids = kmeans.cluster_centers_ 
    v = np.vstack([centroids,[0,0]])
    vor = Voronoi(principalComponents)
    regions, vertices = voronoi_finite_polygons_2d(vor)
    fig = plt.figure(figsize=(10, 10))
    colors, markers, sizes = get_cols_marks(best_n,labels)
    #print(principalComponents)
    df['principal component 1'] = principalComponents[:,0]
    df['principal component 2'] = principalComponents[:,1]
    df['color'] = colors
    df['marker'] = markers
    df['sizes'] = sizes
    #opts.defaults(opts.Points(padding=0.1, size=8, line_color='black'))
    """ data ={'x':list(df['principal component 1'])
                    ,'y':list(df['principal component 2'])
                    ,'color':list(df['color'])
                    ,'marker':list(df['marker']) 
                ,'sizes':list(df['sizes'])} """
    
    #hv.Points(data, vdims=['color', 'marker', 'sizes']).opts(color='color', marker='marker', size='sizes')
    plt.scatter(df['principal component 1'], df['principal component 2'], color=colors, alpha=0.5, edgecolor='k')
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
    for label,x,y in zip(df['activity_identifier'],df['principal component 1'],df['principal component 2']):
    
        label = label        
        #plt.annotate(label, (x,y),textcoords="offset points",xytext=(0,10),ha='center', size =20)
    
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
    #print('Minor Complex Degree:',df[df['labels'] == minor_]['Sum_Degree'].mean())
    #print('Major Complex Degree:',df[df['labels'] == major_]['Sum_Degree'].mean())
    #df.loc[(df['POD_CODE'] == POD)]
    plt.show()
    return df

def plot(df,principalComponents,best_n):
    principalDf = pd.DataFrame(data = principalComponents, columns = ['principal component 1', 'principal component 2'])
    kmeans = KMeans(n_clusters=best_n)   
    kmeans.fit(principalDf)    
    labels = kmeans.predict(principalDf)
    centroids = kmeans.cluster_centers_ 
    v = np.vstack([centroids,[0,0]])
    vor = Voronoi(principalComponents)
    #voronoi_plot_2d(vor)
    #regions, vertices = voronoi_finite_polygons_2d(vor)
    fig = plt.figure(figsize=(10, 10))
    colors, markers, sizes = get_cols_marks(best_n,labels)
    #print(principalComponents)
    df['principal component 1'] = principalComponents[:,0]
    df['principal component 2'] = principalComponents[:,1]
    df['color'] = colors
    df['marker'] = markers
    df['sizes'] = sizes
    #opts.defaults(opts.Points(padding=0.1, size=8, line_color='black'))
    """ data ={'x':list(df['principal component 1'])
                    ,'y':list(df['principal component 2'])
                    ,'color':list(df['color'])
                    ,'marker':list(df['marker']) 
                ,'sizes':list(df['sizes'])} """
    
    #hv.Points(data, vdims=['color', 'marker', 'sizes']).opts(color='color', marker='marker', size='sizes')
    plt.scatter(df['principal component 1'], df['principal component 2'], color=colors, alpha=0.5, edgecolor='k')
    
    df['labels'] = labels
    for index, row in df[['labels','color']].drop_duplicates().iterrows():
        plt.annotate(row['labels'], df.loc[df['labels']==row['labels'],['principal component 1','principal component 2']].mean(),
        
        size =20,weight='bold',
        color='white',
                 backgroundcolor=row['color'])
    #print(list(df['labels'].unique()))
    shape_ = {}
    for item in list(df['labels'].unique()):
        shape_.update({item:[(df[df['labels'] ==item].shape[0]),df[df['labels'] == item]['Sum_Degree'].mean()]})
        #print('Complex Degree:',df[df['labels'] == item]['Sum_Degree'].mean())

    #print(shape_)
    #print(sorted(shape_.items(), key=lambda x: x[1]))
    
    minor_=sorted(shape_.items(), key=lambda x: x[1])[0][0]
    major_=sorted(shape_.items(), key=lambda x: x[1])[1][0]
    
    #sns.pairplot(df[df['labels'] ==1][df.columns.difference(['ACTIVITY_IDENTIFIER','POD_CODE'])], hue="SpellHRG")
    #for label,x,y in zip(df[df['labels'] == minor_]['ACTIVITY_IDENTIFIER'],df[df['labels'] == minor_]['principal component 1'],df[df['labels'] == minor_]['principal component 2']):
    for label,x,y in zip(df['activity_identifier'],df['principal component 1'],df['principal component 2']):
    
        label = label        
        #plt.annotate(label, (x,y),textcoords="offset points",xytext=(0,10),ha='center', size =20)
    
    


    # test=zip(regions, df['color'])
    # for item in test:
    
    #     polygon = vertices[item[0]]
    #     #print(region,polygon)
    #     #print(*zip(*polygon))
    #     plt.fill(*zip(*polygon), alpha=0.4
    #             ,color=item[1]
    #             )
    plt.xlim(vor.min_bound[0]-0.1, vor.max_bound[0]+0.1)
    plt.ylim(vor.min_bound[1]-0.1, vor.max_bound[1]+0.1)
    #print('Minor Complex Degree:',df[df['labels'] == minor_]['Sum_Degree'].mean())
    #print('Major Complex Degree:',df[df['labels'] == major_]['Sum_Degree'].mean())
    #df.loc[(df['POD_CODE'] == POD)]
    plt.show()
    return df