from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import re
import sys
import pandas as pd
import numpy as np
import logging
log = logging.getLogger(__name__)
#import collections
#import itertools
#import time
from datetime import datetime
import matplotlib.pyplot as plt
from .pipline_clean import *
from .graph_node_decorator import *
##Networkx: The call order of arguments values and name switched between v1.x & v2.x.
import networkx as nx

class PatientKG:
    def __init__(self, df,node_attributes_list,global_attributes,turnaround_reference, cd_spec, loglevel = logging.INFO):        
        self.base = return_clean(df,cd_spec)
        self.cd_spec = cd_spec
        self.event_list = node_attributes_list 
        self.global_attributes = global_attributes
        self.logging = logging.basicConfig(level=loglevel)
        self.logger = logging.getLogger(__name__)        
        self.node_dic = {}
        self.nodeDecorator = NodeDecorator()
        self.node_size_reference =  [20, 40, 80, 160]
        self.cmap =['#30a2da','yellow','red','green','black']
        self.graph = nx.DiGraph()
        self.event_node_index = 2
        self.turnaround_reference = turnaround_reference
        self.ae_node_list = {}
        self.start_node_list = {}
        self.attrs = {}
        self.events_cate_list = self.get_unique_cat_list()
        self.spell_start_time,self.spell_end_time = self.get_start_end_dt()
        self.start_node_name = ''
        self.end_node_name = ''
        
              

    def get_unique_cat_list(self):
        try:            
            return self.base[self.cd_spec['EventCatalog']].unique()
        except:
            return []

    def get_start_end_dt(self):
        try:
            tes = self.base.columns.unique()
            start=pd.to_datetime(self.base[self.cd_spec['SpellStartDT']].unique()).strftime('%Y.%m.%d %H:%M:%S').item()
            end=pd.to_datetime(self.base[self.cd_spec['SpellEndDT']].unique())
            return start,end
        except:
            start=pd.to_datetime(self.base['SPELL_START_DATE_TIME'].unique()).strftime('%Y.%m.%d %H:%M:%S').item()
            end=pd.to_datetime(self.base['SPELL_DISCHARGE_DATE_TIME'].unique())
            return start,end


    def update_global_variable(self):
        attributes = {}
        for item in self.global_attributes:
            attributes.update({item:self.base[item].unique()[0]})
        self.graph.graph.update(attributes)
    
    def check_integration(self):
        events= all(elem in self.event_list for elem in list(self.base.columns)) 
        return events

    def create_start_end_node(self):
        try:
            unique_spell = self.base[self.cd_spec['unique_id']].unique()
            if len(unique_spell) == 1:            
                self.start_node_name= str(unique_spell).strip('[]\'')+'_start'
                self.end_node_name = str(unique_spell).strip('[]\'')+'_end'
                self.node_dic.update({self.start_node_name: 0})
                self.node_dic.update({self.end_node_name: 1})
            else:
                print('Not valide unique identifier!')
        except:
            print('Not valide unique identifier!')
        self.graph.add_node(self.node_dic[self.start_node_name])
        self.graph.add_node(self.node_dic[self.end_node_name])
        if self.spell_end_time.isnull():
            self.spell_end_time = pd.to_datetime(['2100-12-12']).strftime('%Y.%m.%d %H:%M:%S').item() 
        else:
            self.spell_end_time = self.spell_end_time.strftime('%Y.%m.%d %H:%M:%S').item() 
        self.attrs = {self.node_dic[self.start_node_name]:{'activity_start_time':self.spell_start_time,'activity_end_time':self.spell_end_time,'name':'Spell_Start','size':20,'color':3},
        self.node_dic[self.end_node_name]:{'activity_start_time':self.spell_start_time,'activity_end_time':self.spell_end_time,'name':'Spell_End','size':20,'color':4}}
    
        for item in self.event_list:
            self.attrs[self.node_dic[self.start_node_name]].update({item:''})
            self.attrs[self.node_dic[self.end_node_name]].update({item:''})

    def update_node_dictionary_with_events(self, period=None):
        self.event_node_index = 2
        for index, row in self.base.sort_values(by=[self.cd_spec['EventCatalog'],self.cd_spec['EventStartDT']]).iterrows():
            if period == None:
                node_name =str(row[self.cd_spec['EventCatalog']])+str(index)
                self.node_dic.update({node_name:self.event_node_index})
                self.event_node_index = self.event_node_index+1
            else:
                time_difference =(datetime.strptime(row[self.cd_spec['EventStartDT']].strftime('%Y.%m.%d %H:%M:%S'), '%Y.%m.%d %H:%M:%S')-datetime.strptime(self.spell_start_time,'%Y.%m.%d %H:%M:%S')).total_seconds()
                if time_difference < period:       
                    node_name =str(row[self.cd_spec['EventCatalog']])+str(index)
                    self.node_dic.update({node_name:self.event_node_index})
                    self.event_node_index = self.event_node_index+1   
    

    def add_events_to_graph(pgraph, index,row):
        pgraph.graph.add_node(pgraph.node_dic[str(row[pgraph.cd_spec['EventCatalog']])+str(index)])
        try:
            event_time=pd.to_datetime(row[pgraph.cd_spec['EventStartDT']]).strftime('%Y.%m.%d %H:%M:%S')
        except:
            event_time=pd.to_datetime(row[pgraph.cd_spec['EventEndDT']]).strftime('%Y.%m.%d %H:%M:%S')
        turn_around = '' #row['TURNAROUND']
        #range_ = list(self.turnaround_reference.loc[row[self.cd_spec['EventCatalog']]][['min','25%','50%', '75%','max']])            
        size = 20
        color = 0
        #size, color = self.nodeDecorator.return_size_color(turn_around, range_)
        diff = (datetime.strptime(event_time,'%Y.%m.%d %H:%M:%S')-datetime.strptime(pgraph.spell_start_time,'%Y.%m.%d %H:%M:%S')).total_seconds()
        node_name =str(row[pgraph.cd_spec['EventCatalog']])+str(index)
        if diff < 0:           
            pgraph.ae_node_list.update({node_name:row[pgraph.cd_spec['EventStartDT']]})
        else:
            pgraph.start_node_list.update({node_name:row[pgraph.cd_spec['EventStartDT']]})            
                #G.add_edge(node_dic[start_node],node_dic[str(row[self.cd_spec['EventCatalog']])+str(index)])
                #G.add_edge(node_dic[str(row[self.cd_spec['EventCatalog']])+str(index)],node_dic[end_node])
        tt= pd.to_datetime(row[pgraph.cd_spec['EventEndDT']]).strftime('%Y.%m.%d %H:%M:%S')
        pgraph.attrs.update({pgraph.node_dic[str(row[pgraph.cd_spec['EventCatalog']])+str(index)]:{}})
        pgraph.attrs[pgraph.node_dic[str(row[pgraph.cd_spec['EventCatalog']])+str(index)]].update(
                    {'activity_start_time':event_time
                        ,'activity_end_time':tt
                     ,'name':str(row[pgraph.cd_spec['EventCatalog']])
                     ,'turnaround':turn_around
                     ,'size':size
                     ,'color':color})
        for item2 in pgraph.event_list:      
                result= row[item2]
                pgraph.attrs[pgraph.node_dic[str(row[pgraph.cd_spec['EventCatalog']])+str(index)]].update({item2:str(result)})
        

    def decorate_LabTest_nodes(self, index, row):
        try:
            turn_around = row['TURNAROUND']
            #print(self.turnaround_reference.loc[self.turnaround_reference[self.cd_spec['EventCatalog']] == row[self.cd_spec['EventCatalog']]])
            #range_ = list(self.turnaround_reference.loc[self.turnaround_reference[self.cd_spec['EventCatalog']] == row[self.cd_spec['EventCatalog']]][['min_','25%','50%', '75%','max_']])            
            #size = 20
            #color = 0
            size, color = self.nodeDecorator.return_size_color(self, row)
            self.attrs[self.node_dic[str(row[self.cd_spec['EventCatalog']])+str(index)]].update(
                    {
                     'turnaround':turn_around
                     ,'size':size
                     ,'color':color})
        except:
            pass

    

    def standard_compliance(self):
        try:
            self.graph = waterlow_assess_policy.policy_compliance(self.graph)  
        except:
            self.graph.graph.update({'waterlow_standard_compliance': 'No waterlow'})
        
        try:
            self.graph = care_plan_policy.policy_compliance(self.graph)  
        except:
            self.graph.graph.update({'care_plan_compliance': 'No careplan'})
            self.graph.graph.update({'care_plan_ontime': 0})

        return

    def add_nodes_by_period(self, period=None):
        self.node_dic = {}
        self.graph.clear()
        self.create_start_end_node()
        self.update_node_dictionary_with_events(period)
        self.update_global_variable()
        for index, row in self.base.sort_values(by=[self.cd_spec['EventCatalog'],self.cd_spec['EventStartDT']]).iterrows():      
            if period == None: 
                self.add_events_to_graph(index,row)
                self.decorate_LabTest_nodes(index,row)
            else:
                time_difference =(datetime.strptime(row[self.cd_spec['EventStartDT']].strftime('%Y.%m.%d %H:%M:%S'), '%Y.%m.%d %H:%M:%S')- datetime.strptime(self.spell_start_time,'%Y.%m.%d %H:%M:%S')).total_seconds()
                if time_difference < period:
                    self.add_events_to_graph(self,index,row)
                    self.decorate_LabTest_nodes(index,row)
        nx.set_node_attributes(self.graph,self.attrs)
                

    def add_full_edges(self):
        try:
            self.graph.remove_edges_from(list(self.graph.edges()))
        except:
            pass
        for item in self.events_cate_list:    
            edge_list = []        
            for u, outer_d in self.graph.nodes(data=True):           
                name = list(self.node_dic.keys())[list(self.node_dic.values()).index(u)]
                edge = re.match( r'{}'.format(item), name, re.M|re.I)        
                if edge:
                    edge_list.append({'name':u,'activity_end_time':outer_d['activity_end_time']})
                else:
                    continue        
            if len(edge_list)>1:            
                for i in range(len(edge_list)-1):
                    self.graph.add_edge(edge_list[i]['name'], edge_list[i+1]['name'])
            else:
                continue
        self.graph.add_edge(0, 1)
        
        for k in self.graph.nodes():
            if k > 1:
                self.graph.add_edge(self.node_dic[self.start_node_name],k)  
                self.graph.add_edge(k,self.node_dic[self.end_node_name])

    def add_linked_edges(self):
        try:
            self.graph.remove_edges_from(list(self.graph.edges()))
        except:
            pass
        for item in self.events_cate_list:    
            edge_list = []        
            start_event = []
            end_event = []
            special = re.search('[@_!#$%^&*()<>?/\|}{~:]',item)
                
            if(special == None): 
                pass         
            else: 
                item = item.replace("(","\\(")
                item = item.replace(")","\\)")
            #print(item)
            for u, outer_d in self.graph.nodes(data=True):           
                name = list(self.node_dic.keys())[list(self.node_dic.values()).index(u)]
                edge = re.match( r'{}'.format(item), name, re.M|re.I)
                if edge:
                    edge_list.append({'name':u,'activity_end_time':outer_d['activity_end_time']})
                else:
                    continue        
            if len(edge_list)>1:            
                for i in range(len(edge_list)-1):
                    self.graph.add_edge(edge_list[i]['name'], edge_list[i+1]['name'])
            else:
                pass        
            #print(edge_list)
            try:
                max_ = edge_list[0]
                for x in edge_list:
                    if x['activity_end_time'] > max_['activity_end_time']:
                        max_=x
                self.graph.add_edge(max_['name'],self.node_dic[self.end_node_name])
                min_ = edge_list[0]
                for x in edge_list:
                    if x['activity_end_time'] < min_['activity_end_time']:
                        min_=x
                self.graph.add_edge(self.node_dic[self.start_node_name], min_['name'])
            except:
                self.graph.add_edge(0,1)
        self.graph.add_edge(0, 1)

    def timeline_layout(self, labels=None):
        node = {}
        for node_index, node_feature in self.graph.nodes(data=True):
            if node_feature['name'] == 'Spell_Start':
                node.update({node_index:node_feature['activity_start_time']})
            elif node_feature['name'] == 'Spell_End':
                node.update({node_index:node_feature['activity_end_time']})
            else:
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

    def return_clean_labels(self,labels):
        for k,v in labels.items():    
            result = ''.join([i for i in v if not i.isdigit()])    
            labels.update({k:result})
        for k,v in labels.items():
            if v == '._start':
                labels.update({k:'Spell_Start'})
            elif v == '._end':
                labels.update({k:'Spell_End'})
        return labels

    def plot_graph(self, labels = None):
        plt.figure(figsize=(30,30))    
        red_edges=[]
        node_size = []
        pos=self.timeline_layout(dict((int(v),k) for k,v in self.node_dic.items()))
        labels = self.return_clean_labels(dict((int(v),k) for k,v in self.node_dic.items()))
        for n in range(len(self.graph.nodes)):
            if n == 0 or n==1:
                node_size.append(2000)
            else:
                node_size.append(self.graph.nodes[n]['turnaround'])
        ##node_size = 2000
        edge_colours = ['black' if not edge in red_edges else 'red'
                        for edge in self.graph.edges()]
        black_edges = [edge for edge in self.graph.edges() if edge not in red_edges]    
        nx.draw_networkx_nodes(self.graph, pos, cmap=plt.get_cmap('jet'), 
                                node_size = node_size)
        nx.draw_networkx_labels(self.graph, pos,labels=labels,font_size = 40)
        nx.draw_networkx_edges(self.graph, pos, edgelist=red_edges, node_size=node_size, edge_color='r', arrows=True,arrowsize=30, width=2)
        nx.draw_networkx_edges(self.graph, pos, edgelist=black_edges, node_size=node_size,arrows=True,arrowsize=40, width=2)   
    
        plt.show()
    
    def compose_pKG(self, pKG2):
        for index, row in pKG2.base.sort_values(by=[pKG2.cd_spec['EventCatalog'],pKG2.cd_spec['EventStartDT']]).iterrows():
            #print(index, row)
            max_node_index = self.graph.number_of_nodes()
            e_node_index = pKG2.node_dic[str(row[pKG2.cd_spec['EventCatalog']])+str(index)]
            self.graph.add_node(max_node_index)
            try:
                event_time=pd.to_datetime(row[pKG2.cd_spec['EventStartDT']]).strftime('%Y.%m.%d %H:%M:%S')
            except:
                event_time=pd.to_datetime(row[pKG2.cd_spec['EventEndDT']]).strftime('%Y.%m.%d %H:%M:%S')
            turn_around = '' #row['TURNAROUND']
            #range_ = list(self.turnaround_reference.loc[row[self.cd_spec['EventCatalog']]][['min','25%','50%', '75%','max']])            
            size = 20
            color = 0
            size, color = self.nodeDecorator.return_size_color(pKG2,row)
            diff = (datetime.strptime(event_time,'%Y.%m.%d %H:%M:%S')-datetime.strptime(pKG2.spell_start_time,'%Y.%m.%d %H:%M:%S')).total_seconds()
            node_name =str(row[pKG2.cd_spec['EventCatalog']])+str(index)
            if diff < 0:           
                self.ae_node_list.update({node_name:row[pKG2.cd_spec['EventStartDT']]})
            else:
                self.start_node_list.update({node_name:row[pKG2.cd_spec['EventStartDT']]})            
                    #G.add_edge(node_dic[start_node],node_dic[str(row[self.cd_spec['EventCatalog']])+str(index)])
                    #G.add_edge(node_dic[str(row[self.cd_spec['EventCatalog']])+str(index)],node_dic[end_node])
            tt= pd.to_datetime(row[pKG2.cd_spec['EventEndDT']]).strftime('%Y.%m.%d %H:%M:%S')
            self.attrs.update({max_node_index:{}})
            self.attrs[max_node_index].update(
                        {'activity_start_time':event_time
                            ,'activity_end_time':tt
                        ,'name':str(row[pKG2.cd_spec['EventCatalog']])
                        ,'turnaround':turn_around
                        ,'size':size
                        ,'color':color
                        })
            for item2 in pKG2.event_list:      
                    result= row[item2]
                    self.attrs[max_node_index].update({item2:str(result)})            
            self.node_dic.update({node_name:max_node_index})
        self.events_cate_list = np.append(self.events_cate_list, pKG2.base[pKG2.cd_spec['EventCatalog']].unique())
        nx.set_node_attributes(self.graph,self.attrs)
    
    


    

    
    
   

    


    
    
    
    
   
    
