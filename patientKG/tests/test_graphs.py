import pyodbc
import pandas as pd
from patientKG import *
import holoviews as hv
from holoviews import opts
from bokeh.plotting import show
import panel as pn
import networkx as nx
from ..config.bedrock_connection import *
from ..priorKnowledge import labturnaround
from patientKG import utils_pickle
from PU.pu_events import *

hv.extension('bokeh')
defaults = dict(width=1000, height=1000, padding=0.1)

def test_hrg_example():
        Bedrock_Conn = Bedrock() 
        #DF = pd.read_sql_query('Select * from dbo.vw_STRANDED_ORDERS_IN_ALL_PIVOT_FILTED where activity_identifier = \'4532029\'',Bedrock_Conn)
        DF = pd.read_csv('patientKG/tests/MockPatient.csv')
        Bedrock_Conn.close()
        Event_list = [
        'RED BLOOD CELL COUNT'
        ,'MEAN CELL HAEMOGLOBIN'
        ,'HAEMOGLOBIN'
        ,'HAEMATOCRIT'
        ,'PLATELET COUNT'
        ,'MEAN CELL VOLUME'
        ,'MEAN CELL HAEMOGLOBIN CONC'
        ,'WHITE BLOOD CELL COUNT'
        ,'MONOCYTE COUNT'
        ,'NEUTROPHIL COUNT'
        ,'LYMPHOCYTE COUNT'
        ,'EOSINOPHIL COUNT'
        ,'BASOPHIL COUNT'
        ,'SODIUM'
        ,'UREA LEVEL'
        ,'CREATININE'
        ,'POTASSIUM'
        ,'C-REACTIVE PROTEIN']

        Event_list_Dic = {
        'RED BLOOD CELL COUNT':'numeric'
        ,'MEAN CELL HAEMOGLOBIN':'numeric'
        ,'HAEMOGLOBIN':'numeric'
        ,'HAEMATOCRIT':'numeric'
        ,'PLATELET COUNT':'numeric'
        ,'MEAN CELL VOLUME':'numeric'
        ,'MEAN CELL HAEMOGLOBIN CONC':'numeric'
        ,'WHITE BLOOD CELL COUNT':'numeric'
        ,'MONOCYTE COUNT':'numeric'
        ,'NEUTROPHIL COUNT':'numeric'
        ,'LYMPHOCYTE COUNT':'numeric'
        ,'EOSINOPHIL COUNT':'numeric'
        ,'BASOPHIL COUNT':'numeric'
        ,'SODIUM':'numeric'
        ,'UREA LEVEL':'numeric'
        ,'CREATININE':'numeric'
        ,'POTASSIUM':'numeric'
        ,'C-REACTIVE PROTEIN':'numeric'}

        Columns_Data_Spec_Dic = {
            'unique_id':'ACTIVITY_IDENTIFIER'#Used as unique identifier
            ,'LengthofStay':'TOTAL_LOS'
            ,'Age':'HPS_AGE_AT_ADMISSION_DATE' #Used as Global Attributes
            ,'EventCatalog':'ORDER_CATALOG_DESCRIPTION'
            ,'EventStartDT':'ORDER_DATE_TIME'
            ,'EventEndDT':'ORDER_RESULT_PERFORMED_DATE_TIME'
            ,'SpellStartDT':'HPS_START_DATE_TIME_HOSPITAL_PROVIDER_SPELL'
            ,'SpellEndDT':'HPS_DISCHARGE_DATE_TIME_HOSPITAL_PROVIDER_SPELL'
        }

        Columns_Data_Spec = [
            ['cd_mapping','unique_id','ACTIVITY_IDENTIFIER']
            ,['cd_mapping','LengthofStay','TOTAL_LOS']
            ,['cd_mapping','Age','HPS_AGE_AT_ADMISSION_DATE']
            ,['cd_mapping','EventCatalog','ORDER_CATALOG_DESCRIPTION']
            ,['cd_mapping','EventStartDT','ORDER_DATE_TIME']
            ,['cd_mapping','EventEndDT','ORDER_RESULT_PERFORMED_DATE_TIME']
            ,['cd_mapping','SpellStartDT','HPS_START_DATE_TIME_HOSPITAL_PROVIDER_SPELL']
            ,['cd_mapping','SpellEndDT','HPS_DISCHARGE_DATE_TIME_HOSPITAL_PROVIDER_SPELL']
            ,['event_date_type','RED BLOOD CELL COUNT','numeric']
            ,['event_date_type','MEAN CELL HAEMOGLOBIN','numeric']
            ,['event_date_type','HAEMOGLOBIN','numeric']
            ,['event_date_type','HAEMATOCRIT','numeric']
            ,['event_date_type','PLATELET COUNT','numeric']
            ,['event_date_type','MEAN CELL VOLUME','numeric']
            ,['event_date_type','MEAN CELL HAEMOGLOBIN CONC','numeric']
            ,['event_date_type','WHITE BLOOD CELL COUNT','numeric']
            ,['event_date_type','MONOCYTE COUNT','numeric']
            ,['event_date_type','NEUTROPHIL COUNT','numeric']
            ,['event_date_type','LYMPHOCYTE COUNT','numeric']
            ,['event_date_type','EOSINOPHIL COUNT','numeric']
            ,['event_date_type','BASOPHIL COUNT','numeric']
            ,['event_date_type','SODIUM','numeric']
            ,['event_date_type','UREA LEVEL','numeric']
            ,['event_date_type','CREATININE','numeric']
            ,['event_date_type','POTASSIUM','numeric']
            ,['event_date_type','C-REACTIVE PROTEIN','numeric']
            ]
        df = pd.DataFrame(Columns_Data_Spec)
        df.columns = ['Function','key','value']
        print(df[df['Function']=='event_date_type'])
        
        #reference=DF.groupby('ORDER_CATALOG_DESCRIPTION')['TURNAROUND'].describe()
        reference = labturnaround.LabTurnAround().get_reference_from_db()
        item = 123456
        test1=DF[DF['ACTIVITY_IDENTIFIER']==item]
        test = graphs_base.PatientKG(DF[DF['ACTIVITY_IDENTIFIER']==item],Event_list,['TOTAL_LOS','HPS_AGE_AT_ADMISSION_DATE'],reference,Columns_Data_Spec_Dic)
        test.add_nodes_by_period(period=None)
        test.add_full_edges()
        test.add_linked_edges()
        graph1 = test.graph
        hv.opts.defaults(
            opts.EdgePaths(**defaults), opts.Graph(**defaults), opts.Nodes(**defaults))
        simple_graph=hv.Graph.from_networkx(graph1, test.timeline_layout(dict((int(v),k) for k,v in test.node_dic.items())))
        #print(graph1.degree)
        #bokeh_server = pn.Row(simple_graph).show(port=12345)
        #bokeh_server.stop()
        #show(simple_graph)
        return test
    
def test_pu_example_wardstay(item='4194205'):
    Red004_Conn = Red004()
    DF = pd.read_sql_query('SELECT * FROM [AdvancedAnalytics].[dbo].[Patient_Episode_Ward_Stay] where ACTIVITY_IDENTIFIER = '+ item +'  order by ACTIVITY_IDENTIFIER, CE_EPISODE_NUMBER, WARD_STAY_ORDER',Red004_Conn)
    # DF = pd.read_csv('MockPatient.csv')
    Red004_Conn.close()    
    Event_list = ['WARD STAY LOCATION']
    Columns_Data_Spec_Dic = {
        'unique_id':'ACTIVITY_IDENTIFIER'#Used as unique identifier
        ,'LengthofStay':'TOTAL_LOS'
        ,'Age':'HPS_AGE_AT_ADMISSION_DATE'#Used as Global Variable
        ,'EventCatalog':'EVENT_CATALOG'
        ,'EventStartDT':'EVENT_INITIAL_DATE_TIME'
        ,'EventEndDT':'EVENT_RESULT_DATE_TIME'
        ,'SpellStartDT':'SPELL_START_DATE_TIME'
        ,'SpellEndDT':'SPELL_DISCHARGE_DATE_TIME'
        }
    reference = labturnaround.LabTurnAround().get_reference_from_db()
    test = graphs_base.PatientKG(DF[DF['ACTIVITY_IDENTIFIER']==item],Event_list,['TOTAL_LOS','HPS_AGE_AT_ADMISSION_DATE'],reference,Columns_Data_Spec_Dic)
    test.add_nodes_by_period(period=None)
    test.add_full_edges()
    test.add_linked_edges()
    graph1 = test.graph
    hv.opts.defaults(
        opts.EdgePaths(**defaults), opts.Graph(**defaults), opts.Nodes(**defaults))
    simple_graph=hv.Graph.from_networkx(graph1, test.timeline_layout(dict((int(v),k) for k,v in test.node_dic.items())))
    #print(graph1.degree)
    #bokeh_server = pn.Row(simple_graph).show(port=12345)
    #bokeh_server.stop()
    #show(simple_graph)
    return test

def test_pu_example_waterlow(item='4194205'):
    Red004_Conn = Red004()
    DF = pd.read_sql_query('SELECT * FROM [AdvancedAnalytics].[dbo].[Waterlow_Events_Pivot] where ACTIVITY_IDENTIFIER = '+ item +'  order by ACTIVITY_IDENTIFIER, CE_EPISODE_NUMBER',Red004_Conn)
    # DF = pd.read_csv('MockPatient.csv')
    Red004_Conn.close()
    #item = '4194205'
    Event_list = ['WL - Age'
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
      ,'WL - Waterlow Score'
      ,'WL - Weight Lost']
    Columns_Data_Spec_Dic = {
            'unique_id':'ACTIVITY_IDENTIFIER'#Used as unique identifier
            ,'LengthofStay':'TOTAL_LOS'
            ,'Age':'HPS_AGE_AT_ADMISSION_DATE' #Used as Global Attributes
            ,'EventCatalog':'ORDER_CATALOG_DESCRIPTION'
            ,'EventStartDT':'ORDER_DATE_TIME'
            ,'EventEndDT':'ORDER_RESULT_PERFORMED_DATE_TIME'
            ,'SpellStartDT':'HPS_START_DATE_TIME_HOSPITAL_PROVIDER_SPELL'
            ,'SpellEndDT':'HPS_DISCHARGE_DATE_TIME_HOSPITAL_PROVIDER_SPELL'
        }

    reference = labturnaround.LabTurnAround().get_reference_from_db()
    test = graphs_base.PatientKG(DF[DF['ACTIVITY_IDENTIFIER']==item],Event_list,['TOTAL_LOS','HPS_AGE_AT_ADMISSION_DATE'],reference,Columns_Data_Spec_Dic)
    test.add_nodes_by_period(period=None)
    test.add_full_edges()
    test.add_linked_edges()
    graph1 = test.graph
    hv.opts.defaults(
        opts.EdgePaths(**defaults), opts.Graph(**defaults), opts.Nodes(**defaults))
    simple_graph=hv.Graph.from_networkx(graph1, test.timeline_layout(dict((int(v),k) for k,v in test.node_dic.items())))
    print(graph1.degree)
    #bokeh_server = pn.Row(simple_graph).show(port=12345)
    #bokeh_server.stop()
    #show(simple_graph)
    return test

def test_pu_example_labtests(item='4194205'):
    Red004_Conn = Red004()
    DF = pd.read_sql_query('SELECT * FROM [AdvancedAnalytics].[dbo].[Labtests_Events_Pivot] where ACTIVITY_IDENTIFIER = '+ item +'  order by ACTIVITY_IDENTIFIER, CE_EPISODE_NUMBER',Red004_Conn)
    # DF = pd.read_csv('MockPatient.csv')
    Red004_Conn.close()
    #item = '4194205'
    Event_list = ['Glucose fasting',
    'Glucose, dialysis fluid',
    'Albumin',
    'Glucose, fluid',
    'Glucose, CSF',
    'Haematocrit',
    'Red blood cell count',
    'C-reactive protein',
    'Mean cell volume',
    'Glucose random',
    'Mean cell haemoglobin conc',
    'Monocyte count',
    'Haemoglobin A1c IFCC',
    'Basophil count',
    'White blood cell count',
    'Mean cell haemoglobin',
    'Neutrophil count',
    'Eosinophil count',
    'Haemoglobin',
    'Lymphocyte count',
    'Platelet count']
    Columns_Data_Spec_Dic = {
            'unique_id':'ACTIVITY_IDENTIFIER'#Used as unique identifier
            ,'LengthofStay':'TOTAL_LOS'
            ,'Age':'HPS_AGE_AT_ADMISSION_DATE' #Used as Global Attributes
            ,'EventCatalog':'ORDER_CATALOG_DESCRIPTION'
            ,'EventStartDT':'ORDER_DATE_TIME'
            ,'EventEndDT':'ORDER_RESULT_PERFORMED_DATE_TIME'
            ,'SpellStartDT':'HPS_START_DATE_TIME_HOSPITAL_PROVIDER_SPELL'
            ,'SpellEndDT':'HPS_DISCHARGE_DATE_TIME_HOSPITAL_PROVIDER_SPELL'
        }

    reference = labturnaround.LabTurnAround().get_reference_from_db()
    test = graphs_base.PatientKG(DF[DF['ACTIVITY_IDENTIFIER']==item],Event_list,['TOTAL_LOS','HPS_AGE_AT_ADMISSION_DATE'],reference,Columns_Data_Spec_Dic)
    test.add_nodes_by_period(period=None)
    test.add_full_edges()
    test.add_linked_edges()
    # graph1 = test.graph
    # hv.opts.defaults(
    #     opts.EdgePaths(**defaults), opts.Graph(**defaults), opts.Nodes(**defaults))
    # optss = dict(node_size='size', edge_line_width=0.5 ,color_index='color', cmap=['#30a2da','yellow','red','green','black'])
    # simple_graph=hv.Graph.from_networkx(graph1, test.timeline_layout(dict((int(v),k) for k,v in test.node_dic.items()))).options(**optss)
    # print(graph1.degree)
    # bokeh_server = pn.Row(simple_graph).show(port=12345)
    # bokeh_server.stop()
    # show(simple_graph)
    return test

def test_pu_example_skinassess(item='5010499'):
    Red004_Conn = Red004()
    DF = pd.read_sql_query('SELECT * FROM [AdvancedAnalytics].[dbo].[Skinassess_Events_Pivot] where ACTIVITY_IDENTIFIER = '+ item +'  order by ACTIVITY_IDENTIFIER, CE_EPISODE_NUMBER',Red004_Conn)
    # DF = pd.read_csv('MockPatient.csv')
    Red004_Conn.close()
    #item = '4194205'
    Event_list = ['Wound Status',
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
    'Pressure Ulcer/Blister Category']
    Columns_Data_Spec_Dic = {
            'unique_id':'ACTIVITY_IDENTIFIER'#Used as unique identifier
            ,'LengthofStay':'TOTAL_LOS'
            ,'Age':'HPS_AGE_AT_ADMISSION_DATE' #Used as Global Attributes
            ,'EventCatalog':'ORDER_CATALOG_DESCRIPTION'
            ,'EventStartDT':'ORDER_DATE_TIME'
            ,'EventEndDT':'ORDER_RESULT_PERFORMED_DATE_TIME'
            ,'SpellStartDT':'HPS_START_DATE_TIME_HOSPITAL_PROVIDER_SPELL'
            ,'SpellEndDT':'HPS_DISCHARGE_DATE_TIME_HOSPITAL_PROVIDER_SPELL'
        }

    reference = labturnaround.LabTurnAround().get_reference_from_db()
    test = graphs_base.PatientKG(DF[DF['ACTIVITY_IDENTIFIER']==item],Event_list,['TOTAL_LOS','HPS_AGE_AT_ADMISSION_DATE'],reference,Columns_Data_Spec_Dic)
    test.add_nodes_by_period(period=None)
    test.add_full_edges()
    test.add_linked_edges()
    # graph1 = test.graph
    # hv.opts.defaults(
    #     opts.EdgePaths(**defaults), opts.Graph(**defaults), opts.Nodes(**defaults))
    # simple_graph=hv.Graph.from_networkx(graph1, test.timeline_layout(dict((int(v),k) for k,v in test.node_dic.items())))
    # print(graph1.degree)
    # bokeh_server = pn.Row(simple_graph).show(port=12345)
    # bokeh_server.stop()
    # show(simple_graph)
    return test

def test_pu_example_tv(item='5010499'):
    Red004_Conn = Red004()
    DF = pd.read_sql_query('SELECT * FROM [AdvancedAnalytics].[dbo].[TVReferral_Events_Pivot] where ACTIVITY_IDENTIFIER = '+ item +'  order by ACTIVITY_IDENTIFIER, CE_EPISODE_NUMBER',Red004_Conn)
    # DF = pd.read_csv('MockPatient.csv')
    Red004_Conn.close()
    reference = labturnaround.LabTurnAround().get_reference_from_db()
    Event_list_TV, Columns_Data_Spec_Dic_TV = pu_tv()
    test = graphs_base.PatientKG(DF[DF['ACTIVITY_IDENTIFIER']==item],Event_list_TV,['TOTAL_LOS','HPS_AGE_AT_ADMISSION_DATE'],reference,Columns_Data_Spec_Dic_TV)
    test.add_nodes_by_period(period=None)
    test.add_full_edges()
    test.add_linked_edges()
    graph1 = test.graph
    hv.opts.defaults(
        opts.EdgePaths(**defaults), opts.Graph(**defaults), opts.Nodes(**defaults))
    simple_graph=hv.Graph.from_networkx(graph1, test.timeline_layout(dict((int(v),k) for k,v in test.node_dic.items())))
    print(graph1.degree)
    bokeh_server = pn.Row(simple_graph).show(port=12345)
    bokeh_server.stop()
    show(simple_graph)
    return test
    
def test_pu_example_pp(item='5010499'):
    Red004_Conn = Red004()
    DF = pd.read_sql_query('SELECT * FROM [AdvancedAnalytics].[dbo].[PatientPosition_Events_Pivot] where ACTIVITY_IDENTIFIER = '+ item +'  order by ACTIVITY_IDENTIFIER, CE_EPISODE_NUMBER',Red004_Conn)
    # DF = pd.read_csv('MockPatient.csv')
    Red004_Conn.close()
    reference = labturnaround.LabTurnAround().get_reference_from_db()
    Event_list_TV, Columns_Data_Spec_Dic_TV = pu_pp()
    test = graphs_base.PatientKG(DF[DF['ACTIVITY_IDENTIFIER']==item],Event_list_TV,['TOTAL_LOS','HPS_AGE_AT_ADMISSION_DATE'],reference,Columns_Data_Spec_Dic_TV)
    test.add_nodes_by_period(period=None)
    test.add_full_edges()
    test.add_linked_edges()
    # graph1 = test.graph
    # hv.opts.defaults(
    #     opts.EdgePaths(**defaults), opts.Graph(**defaults), opts.Nodes(**defaults))
    # simple_graph=hv.Graph.from_networkx(graph1, test.timeline_layout(dict((int(v),k) for k,v in test.node_dic.items())))
    # print(graph1.degree)
    # bokeh_server = pn.Row(simple_graph).show(port=12345)
    # bokeh_server.stop()
    # show(simple_graph)
    return test

def test_compose(item = '5008674'):    
    try:
        test_wl = test_pu_example_waterlow(item )
    except:
        pass
    try:
        test_ws = test_pu_example_wardstay(item )
        print(test_ws)
    except:
        pass    

    try:
        test_blood = test_pu_example_labtests(item )
    except:
        pass
    
    #test_skin = test_pu_example_skinassess('5008674')
    
    try:
        test_pp = test_pu_example_pp(item)
    except:
        pass
    try:
        test_ws.compose_pKG(test_wl)
    except:
        pass
    try:
        test_ws.compose_pKG(test_blood)
    except:
        pass
    try: 
        test_ws.compose_pKG(test_pp)
    except:
        pass
    print("test")
    
    test_ws.add_full_edges()
    test_ws.add_linked_edges()
    test_ws.standard_compliance()
    total_los = test_ws.graph.graph['TOTAL_LOS']        
    sum_degree = sum([val for (node, val) in test_ws.graph.degree()])
    sum_degree2 = sum([val['size'] for (node, val) in test_ws.graph.nodes(data=True)])
    global_central = nx.global_reaching_centrality(test_ws.graph)
    waterlow_standard = test_ws.graph.graph['waterlow_standard_compliance'] 
    careplan = test_ws.graph.graph['care_plan_compliance'] 
    #compose doesnt work as no unique name of node in each graph
    #composed = nx.compose(test_wl.graph,test_ws.graph)
    #print(test_wl.node_dic.items())
    #z = test_wl.node_dic.copy()
    #z.update(test_ws.node_dic)
    #print(z)
    label = dict((int(v),k) for k,v in test_ws.node_dic.items())
    combined = test_ws.graph
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
    pos = test_ws.timeline_layout(label)
    optss = dict(node_size='size', edge_line_width=0.5 ,node_color='color', cmap=['#30a2da','yellow','red','green','black'])
    simple_graph=hv.Graph.from_networkx(combined, pos).options(**optss)
    labels = hv.Labels(simple_graph.nodes, ['x', 'y'], 'abv')
    #print(simple_graph.nodes)
    #print(test_ws.graph.degree)
    #bokeh_server = pn.Row(simple_graph* labels.opts(text_font_size='10pt', text_color='white', bgcolor='gray')).show(port=12345)
    #bokeh_server.stop()
    #show(simple_graph* labels.opts(text_font_size='16pt', text_color='white', bgcolor='gray'))
    return pn.Row(simple_graph* labels.opts(text_font_size='10pt', text_color='white', bgcolor='gray'))

def test_reopen(item='./GraphCalculationResults/Ward_Stay/KG_4307898'):
    test = utils_pickle.read(item)
    graph1 = test.graph
    hv.opts.defaults(
        opts.EdgePaths(**defaults), opts.Graph(**defaults), opts.Nodes(**defaults))
    simple_graph=hv.Graph.from_networkx(graph1, test.timeline_layout(dict((int(v),k) for k,v in test.node_dic.items())))
    print(graph1.degree)
    bokeh_server = pn.Row(simple_graph).show(port=12345)
    bokeh_server.stop()
    show(simple_graph)


if __name__ == '__main__':
    test_hrg_example()