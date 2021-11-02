#Use this carefully as it will loop through all patient data, modelling as knowledge graph and save in pickle file
#Already modelled activity identifier stored in the table AdvancedAnalytics.dbo.Modelled_Activity_Identifier
#
import pyodbc
import numpy as np
import pandas as pd
from tqdm import tqdm
import logging
from patientKG import *
from patientKG.config.bedrock_connection import *
from patientKG.priorKnowledge import labturnaround
from patientKG.tests.test_graphs import *
from PU.pu_events import *
from patientKG import utils_pickle
import multiprocessing as mp
from contextlib import closing
import os
import re
import ast
import json
from patientKG.tests.test_graphs import *
from datetime import datetime
logging.basicConfig(filename="test.log", level=logging.DEBUG)

def check_events(**events_data):
    base = pd.DataFrame()
    try:
        base = events_data.pop('Base')
        base['modelled_events'] = ''
    except:
        print("No base data suppilied")
        return
    
    for event_name,value in events_data.items():        
        base = pd.merge(base, value, on=['ACTIVITY_IDENTIFIER'], how='left', indicator='Exist')
        base.drop(value.columns[~value.columns.isin(['ACTIVITY_IDENTIFIER'])],inplace=True,axis =1)
        base['modelled_events'] = np.where(base.Exist == 'both', base['modelled_events'].astype(str) +','+ str(event_name), base['modelled_events'].astype(str))
        base.drop('Exist',inplace=True,axis =1)
        base.drop_duplicates(keep=False, inplace=True)
    base.drop_duplicates(keep=False, inplace=True)
    base = base.reset_index(drop=True)
    base.to_csv('test.csv')
    return base
    #print(events.values())

def single_worker(arg):
    log = logging.getLogger(__name__)
    _id, modelled, date = arg
    try:
        Event_list_WS, Columns_Data_Spec_Dic_WS = pu_wardstay()
        kg_ws = graphs_base.PatientKG(DF_WS[DF_WS['ACTIVITY_IDENTIFIER']==_id],Event_list_WS,['TOTAL_LOS','HPS_AGE_AT_ADMISSION_DATE'],reference,Columns_Data_Spec_Dic_WS)
        kg_ws.add_nodes_by_period(period=None)
        events = modelled.split(",")
        for event in events:
            if event != '':
                if event == 'LabTest':
                    Event_list_LT, Columns_Data_Spec_Dic_LT = pu_labtests()
                    kg_lt = graphs_base.PatientKG(DF_LT[DF_LT['ACTIVITY_IDENTIFIER']==_id],Event_list_LT,['TOTAL_LOS','HPS_AGE_AT_ADMISSION_DATE'],reference,Columns_Data_Spec_Dic_LT)
                    kg_lt.add_nodes_by_period(period=None)
                    kg_lt.add_full_edges()
                    kg_lt.add_linked_edges()
                    kg_ws.compose_pKG(kg_lt)

                if event == 'Waterlow':
                    Event_list_WL, Columns_Data_Spec_Dic_WL = pu_waterlow()
                    kg_wl = graphs_base.PatientKG(DF_WL[DF_WL['ACTIVITY_IDENTIFIER']==_id],Event_list_WL,['TOTAL_LOS','HPS_AGE_AT_ADMISSION_DATE'],reference,Columns_Data_Spec_Dic_WL)
                    kg_wl.add_nodes_by_period(period=None)
                    kg_wl.add_full_edges()
                    kg_wl.add_linked_edges()
                    kg_ws.compose_pKG(kg_wl)

                if event == 'SkinAssess':
                    Event_list_SA, Columns_Data_Spec_Dic_SA = pu_skinassess()
                    kg_sa = graphs_base.PatientKG(DF_SA[DF_SA['ACTIVITY_IDENTIFIER']==_id],Event_list_SA,['TOTAL_LOS','HPS_AGE_AT_ADMISSION_DATE'],reference,Columns_Data_Spec_Dic_SA)
                    kg_sa.add_nodes_by_period(period=None)
                    kg_sa.add_full_edges()
                    kg_sa.add_linked_edges()
                    kg_ws.compose_pKG(kg_sa)

                if event == 'TVReferral':
                    Event_list_TV, Columns_Data_Spec_Dic_TV = pu_tv()
                    kg_tv = graphs_base.PatientKG(DF_TV[DF_TV['ACTIVITY_IDENTIFIER']==_id],Event_list_TV,['TOTAL_LOS','HPS_AGE_AT_ADMISSION_DATE'],reference,Columns_Data_Spec_Dic_TV)
                    kg_tv.add_nodes_by_period(period=None)
                    kg_tv.add_full_edges()
                    kg_tv.add_linked_edges()
                    kg_ws.compose_pKG(kg_tv)
                if event == 'PatientPosition':
                    Event_list_PP, Columns_Data_Spec_Dic_PP = pu_pp()
                    kg_pp = graphs_base.PatientKG(DF_PP[DF_PP['ACTIVITY_IDENTIFIER']==_id],Event_list_PP,['TOTAL_LOS','HPS_AGE_AT_ADMISSION_DATE'],reference,Columns_Data_Spec_Dic_PP)
                    kg_pp.add_nodes_by_period(period=None)
                    kg_pp.add_full_edges()
                    kg_pp.add_linked_edges()
                    kg_ws.compose_pKG(kg_pp)
        kg_ws.add_full_edges()
        kg_ws.add_linked_edges()
        kg_ws.standard_compliance()


        total_los = kg_ws.graph.graph['TOTAL_LOS']        
        sum_degree = sum([val for (node, val) in kg_ws.graph.degree()])
        sum_degree2 = sum([val['size'] for (node, val) in kg_ws.graph.nodes(data=True)])
        global_central = nx.global_reaching_centrality(kg_ws.graph)
        waterlow_standard = kg_ws.graph.graph['waterlow_standard_compliance'] 
        careplan = kg_ws.graph.graph['care_plan_compliance'] 
        careplan_ontime = kg_ws.graph.graph['care_plan_ontime']
        #distance_to_base = nx.graph_edit_distance(kg.graph, base.graph)
        
        utils_pickle.write(kg_ws,"GraphCalculationResults/Ward_Stay/KG_{}".format(str(_id)))
        #calculate_results.update({str(_id):[sum_degree, global_central,total_los, sum_degree2]})
        return [str(_id),sum_degree, global_central,total_los, sum_degree2,waterlow_standard,careplan,careplan_ontime]
    except Exception as e:
        log.debug("ID: {}, Exception:{}".format(_id,e))
    
def initializer():
    Red004_Conn = Red004()
    global DF_WS
    global DF_WL
    global DF_LT
    global DF_SA
    global DF_TV
    global DF_PP
    global reference
    DF_WS = pd.read_sql_query('SELECT * FROM [AdvancedAnalytics].[dbo].[Patient_Episode_Ward_Stay]',Red004_Conn)
    DF_WL = pd.read_sql_query('SELECT * FROM [AdvancedAnalytics].[dbo].[Waterlow_Events_Pivot]',Red004_Conn)
    DF_LT = pd.read_sql_query('SELECT * FROM [AdvancedAnalytics].[dbo].[Labtests_Events_Pivot]',Red004_Conn)
    DF_SA = pd.read_sql_query('SELECT * FROM [AdvancedAnalytics].[dbo].[Skinassess_Events_Pivot]',Red004_Conn)
    DF_TV = pd.read_sql_query('SELECT * FROM [AdvancedAnalytics].[dbo].[TVReferral_Events_Pivot]',Red004_Conn)
    DF_PP = pd.read_sql_query('SELECT * FROM [AdvancedAnalytics].[dbo].[PatientPosition_Events_Pivot]',Red004_Conn)
    reference = labturnaround.LabTurnAround().get_reference_from_db()
    Red004_Conn.close()   

def main_modelling_parral():
    log = logging.getLogger(__name__)
    
    Red004_Conn = Red004()
    modelled_events = 'WardStay,LabTest,Waterlow,SkinAssess,TVReferral,PatientPosition'
    query = "SET NOCOUNT ON; EXEC [AdvancedAnalytics].[dbo].[usp_PU_Analysis_Modelling_Process] '{0}'".format(modelled_events)
    #DF = pd.read_sql_query('SELECT * FROM [AdvancedAnalytics].[dbo].[Patient_Episode_Ward_Stay]',Red004_Conn)
    base = pd.read_sql_query(query,Red004_Conn)    
    
    # DF_WS = pd.read_sql_query('SELECT * FROM [AdvancedAnalytics].[dbo].[Patient_Episode_Ward_Stay]',Red004_Conn)
    
    # DF_WL = pd.read_sql_query('SELECT * FROM [AdvancedAnalytics].[dbo].[Waterlow_Events_Pivot]',Red004_Conn)    
    
    # DF_LT = pd.read_sql_query('SELECT * FROM [AdvancedAnalytics].[dbo].[Labtests_Events_Pivot]',Red004_Conn)

    # DF_SA = pd.read_sql_query('SELECT * FROM [AdvancedAnalytics].[dbo].[Skinassess_Events_Pivot]',Red004_Conn)

    # DF_TV = pd.read_sql_query('SELECT * FROM [AdvancedAnalytics].[dbo].[TVReferral_Events_Pivot]',Red004_Conn)

    # DF_PP = pd.read_sql_query('SELECT * FROM [AdvancedAnalytics].[dbo].[PatientPosition_Events_Pivot]',Red004_Conn)
    
    #base = check_events(Base=DF_Base,WardStay=DF_WS, Waterlow=DF_WL, Labtest=DF_LT,SkinAssess=DF_SA, TVReferral=DF_TV)
    # DF = pd.read_csv('MockPatient.csv')
    
    path_to_file = 'GraphCalculationResults/Ward_Stay/'
    ids = []
    for filename in os.listdir(path_to_file):
        if re.search(r'\bKG_\d*\b',filename):
            ids.append(filename.split("_")[1])
    print('Total: {}'.format(len(base)))
    base=base[~base['activity_identifier'].isin(ids)]
    print('Resume:{}'.format(len(base)))
    Red004_Conn.commit()
    Red004_Conn.close()
    calculate_results =[]
    # for index,row in tqdm(base.iterrows()):
    #     try:
    #         single_worker(row)
    #     except Exception as e:
    #         log.debug("ID: {}, Exception:{}".format(row['ACTIVITY_IDENTIFIER'],e))
    #         continue
    # for i in range(0,base.shape[0]):
    #     pool.apply_async(single_worker, args=(base.loc[i]))
    # pool.close()
    # pool.join()
    num_workers = mp.cpu_count()-1
    process = list(tuple(map(tuple,base.to_records(index=False))))    
    with closing(mp.Pool(num_workers,initializer)) as pool:
        max_ = len(process)
        with tqdm(total=max_) as pbar:
            #joined = pool.imap_unordered(single_worker, process)
            for i, res in enumerate(pool.imap_unordered(single_worker, process)):
                pbar.update()
                calculate_results.append(res)
    
    
    
    engine = Red004_AA_engine()
    #base.to_sql("MODELLED_ACTIVITY_IDENTIFIER", schema="dbo", con = engine, index=False, if_exists='append')    
    utils_pickle.write(calculate_results,"GraphCalculationResults/Ward_Stay/KG_Calculated_Degree")
    print("Success!")

def write_list_sql(results):
    calculate_results_df = pd.DataFrame.from_records([x for x in results if x is not None], columns=['ACTIVITY_IDENTIFIER','Complex_Degree','Global_Central','Total_LOS','Turnaround_Degree','Waterlow_Standard','Careplan','DT_ATRISK','ward_move','careplan_ontime','numberof_repositioning','carplan_numberof_repositioning','careplan_compliance_degree'])
    engine = Red004_AA_engine()
    mask = (calculate_results_df['Waterlow_Standard'] != 'No waterlow')
    z_valid= calculate_results_df[mask]
    #calculate_results_df.loc[mask,'Waterlow_Standard']=str(z_valid['Waterlow_Standard'])

    calculate_results_df['Waterlow_Standard'] = calculate_results_df['Waterlow_Standard'].astype(str)
    calculate_results_df['Careplan'] = calculate_results_df['Careplan'].astype(str)
    calculate_results_df['DT_ATRISK'] = calculate_results_df['DT_ATRISK'].astype(str)
    calculate_results_df['ward_move'] = calculate_results_df['ward_move'].astype(str)
    calculate_results_df['careplan_ontime'] = calculate_results_df['careplan_ontime'].astype(str)
    
    calculate_results_df['numberof_repositioning'] = calculate_results_df['numberof_repositioning'].astype(str)
    calculate_results_df['carplan_numberof_repositioning'] = calculate_results_df['carplan_numberof_repositioning'].astype(str)
    calculate_results_df['careplan_compliance_degree'] = calculate_results_df['careplan_compliance_degree'].astype(str)
    
    calculate_results_df.to_sql("MODELLED_RESULTS", schema="dbo", con = engine, index=False, if_exists='append')       
    print("Done!")

def check_all(arg):
    _id = arg    
    kg_ws =utils_pickle.read("GraphCalculationResults/Ward_Stay/KG_{}".format(str(_id)))
    total_los = kg_ws.graph.graph['TOTAL_LOS']        
    sum_degree = sum([val for (node, val) in kg_ws.graph.degree()])
    sum_degree2 = sum([val['size'] for (node, val) in kg_ws.graph.nodes(data=True)])
    global_central = nx.global_reaching_centrality(kg_ws.graph)    
    waterlow_standard = kg_ws.graph.graph['waterlow_standard_compliance'] 
    careplan = kg_ws.graph.graph['care_plan_compliance'] 
    dt_atrisk = check_dt_atrisk(kg_ws.graph)
    ward_move = check_ward_move(kg_ws.graph)
    careplan_ontime = check_careplan_ontime(kg_ws.graph)
    numberof_repositioning = check_reposition(kg_ws.graph)
    carplan_numberof_repositioning = calculate_numberof_reposition(kg_ws.graph)
    if careplan in (['No careplan', "Risk but no careplan!","Notyet"]):
        careplan_compliance_degree = 0 
    elif careplan in (['No risk',"0,0"]):
        careplan_compliance_degree = 1
    else:
        careplan_compliance_degree = float(careplan.split(",")[1])/carplan_numberof_repositioning
    #careplan_ontime = kg_ws.graph.graph['care_plan_ontime']
        #distance_to_base = nx.graph_edit_distance(kg.graph, base.graph)
        #calculate_results.update({str(_id):[sum_degree, global_central,total_los, sum_degree2]})
    return [str(_id),sum_degree, global_central,total_los, sum_degree2,waterlow_standard,careplan,dt_atrisk,ward_move,careplan_ontime,numberof_repositioning,carplan_numberof_repositioning,careplan_compliance_degree]

def check_reposition(kg):    
    try:
        nodesOfCarePlan = {x:y for x,y in kg.nodes(data=True) if y['name'] == 'Patient Position' and ('Pressure Ulcer' not in y['PATIENTPOSITION'])}
        return len(nodesOfCarePlan)
    except:
        return 0

def calculate_numberof_reposition(kg):
    nodesOfCarePlan,dt_atrisk=check_careplan_init(kg)
    if dt_atrisk == '0000' or dt_atrisk =='No Risk!':
        return 0
    else:
        nodesOfSpellEnd = {x:y for x,y in kg.nodes(data=True) if y['name'] == 'Spell_End'}
        difference = datetime.strptime(nodesOfSpellEnd[1]['activity_end_time'],'%Y.%m.%d %H:%M:%S')-datetime.strptime(dt_atrisk,'%Y.%m.%d %H:%M:%S')
        duration_in_s = difference.total_seconds()
        return duration_in_s/(4*3600)

def hour4(event_start_dt ,event_start_dt2):
    #waterlow_start_dt = row[kg.cd_spec['EventStartDT']]
    #waterlow_result_dt = row[kg.cd_spec['EventEndDT']]
    #spell_start = row[kg.cd_spec['SpellStartDT']]
    difference = event_start_dt - event_start_dt2
    duration_in_s = difference.total_seconds()
    hours = divmod(duration_in_s, 3600)[0]
    if hours <= 4:
        return 'Pass'
    else:
        return 'Fail'

def check_careplan_ontime(kg):
    nodesOfCarePlan,dt_atrisk=check_careplan_init(kg)
    attrs={}
    if dt_atrisk == '0000':
        return '1'
    elif dt_atrisk !='0000' and len(nodesOfCarePlan) ==0:
        return '0' 
    elif dt_atrisk !='0000' and len(nodesOfCarePlan) >0:
        try:
            check = hour4(datetime.strptime(nodesOfCarePlan['activity_start_time'],'%Y.%m.%d %H:%M:%S'), datetime.strptime(dt_atrisk,'%Y.%m.%d %H:%M:%S'))
            if check =='Pass':
                return '1'
            else:
                return '0'
        except:
            return '0' 

def check_careplan_init(kg):
    nodesOfWaterlow = {x:y for x,y in kg.nodes(data=True) if y['name'] == 'Waterlow Assess'}
    sorted_ = {k:v for k,v in sorted(nodesOfWaterlow .items(), key=lambda item: item[1]['activity_start_time'])}
    dt_atrisk = 'No Risk!'
    try:
        nodesOfCarePlan = {x:y for x,y in kg.nodes(data=True) if y['name'] == 'Patient Position' and ('Pressure Ulcer' in y['PATIENTPOSITION'])}
    except:
        nodesOfCarePlan = ''
    try:
        for key,value in sorted_.items():
            if value['WL - Waterlow Score'] !='None' and int(value['WL - Waterlow Score']) > 10:
                dt_atrisk = value['activity_end_time']
                break
            else:
                dt_atrisk = '0000'
    except:
        dt_atrisk = 'No Risk!'
    return nodesOfCarePlan,dt_atrisk
def check_ward_move(kg):
    nodesOfWardStay = {x:y for x,y in kg.nodes(data=True) if y['name'] == 'Ward Stay'}
    stay_location = []
    for key,value in nodesOfWardStay.items():
        if 'virtual' not in value['WARD STAY LOCATION'].lower():
            stay_location.append(value['WARD STAY LOCATION'].lower())
    return len(stay_location)

def check_dt_atrisk(kg):
    nodesOfWaterlow = {x:y for x,y in kg.nodes(data=True) if y['name'] == 'Waterlow Assess'}
    sorted_ = {k:v for k,v in sorted(nodesOfWaterlow .items(), key=lambda item: item[1]['activity_start_time'])}       
    dt_atrisk = 'No Risk!'
    try:
        for key,value in sorted_.items():
            if int(value['WL - Waterlow Score']) > 10:
                dt_atrisk = value['activity_end_time']
                break
            else:
                dt_atrisk = '0000' 
    except:
        dt_atrisk = '0000'
    return dt_atrisk

def main_reload_all_degree():
    calculate_results =[]
    path_to_file = 'GraphCalculationResults/Ward_Stay/'
    ids = []
    for filename in os.listdir(path_to_file):
        if re.search(r'\bKG_\d*\b',filename):
            ids.append(filename.split("_")[1])
    num_workers = mp.cpu_count()-1      
    #process = list(tuple(map(tuple,base.to_records(index=False))))    
    with closing(mp.Pool(num_workers)) as pool:
        max_ = len(ids)
        with tqdm(total=max_) as pbar:
            #joined = pool.imap_unordered(single_worker, process)            
            for i, res in enumerate(pool.imap_unordered(check_all, ids)):                
                pbar.update()
                calculate_results.append(res)
    utils_pickle.write(calculate_results,"GraphCalculationResults/Ward_Stay/KG_Calculated_Degree")

def debug_single_worker():
    Red004_Conn = Red004()    
    DF_WS = pd.read_sql_query('SELECT * FROM [AdvancedAnalytics].[dbo].[Patient_Episode_Ward_Stay]',Red004_Conn)
    DF_WL = pd.read_sql_query('SELECT * FROM [AdvancedAnalytics].[dbo].[Waterlow_Events_Pivot]',Red004_Conn)
    DF_LT = pd.read_sql_query('SELECT * FROM [AdvancedAnalytics].[dbo].[Labtests_Events_Pivot]',Red004_Conn)
    DF_SA = pd.read_sql_query('SELECT * FROM [AdvancedAnalytics].[dbo].[Skinassess_Events_Pivot]',Red004_Conn)
    DF_TV = pd.read_sql_query('SELECT * FROM [AdvancedAnalytics].[dbo].[TVReferral_Events_Pivot]',Red004_Conn)
    DF_PP = pd.read_sql_query('SELECT * FROM [AdvancedAnalytics].[dbo].[PatientPosition_Events_Pivot]',Red004_Conn)
    reference = labturnaround.LabTurnAround().get_reference_from_db()
    Red004_Conn.close() 
    single_worker(['4307898',])  


def modelled_fixing(list_):
    path_to_file = 'GraphCalculationResults/Ward_Stay/'
    for item in list_:
        try:
            test_wl = test_pu_example_waterlow(item)
        except:
            pass
        try:
            test_ws = test_pu_example_wardstay(item)
        except:
            pass    

        try:
            test_blood = test_pu_example_labtests(item)
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
        test_ws.add_full_edges()
        test_ws.add_linked_edges()
        test_ws.standard_compliance()
        total_los = test_ws.graph.graph['TOTAL_LOS']        
        sum_degree = sum([val for (node, val) in test_ws.graph.degree()])
        sum_degree2 = sum([val['size'] for (node, val) in test_ws.graph.nodes(data=True)])
        global_central = nx.global_reaching_centrality(test_ws.graph)
        waterlow_standard = test_ws.graph.graph['waterlow_standard_compliance'] 
        careplan = test_ws.graph.graph['care_plan_compliance']
        utils_pickle.write(test_ws,"GraphCalculationResults/Ward_Stay/KG_{}".format(str(item)))
  

if __name__ == '__main__':
    #Whole Modelling took approx 90 hours
    main_modelling_parral()
    
    #Whole reload calculated results took approx 5 minutes
    #modelled_fixing(list_)
    
    main_reload_all_degree()
    results = utils_pickle.read("GraphCalculationResults/Ward_Stay/KG_Calculated_Degree")
    write_list_sql(results)
    # kg_ws =utils_pickle.read("GraphCalculationResults/Ward_Stay/KG_{}".format(str(5566733)))
    # total_los = kg_ws.graph.graph['TOTAL_LOS']        
    # sum_degree = sum([val for (node, val) in kg_ws.graph.degree()])
    # sum_degree2 = sum([val['size'] for (node, val) in kg_ws.graph.nodes(data=True)])
    # global_central = nx.global_reaching_centrality(kg_ws.graph)    
    # waterlow_standard = kg_ws.graph.graph['waterlow_standard_compliance'] 
    # careplan = kg_ws.graph.graph['care_plan_compliance'] 
    # dt_atrisk = check_dt_atrisk(kg_ws.graph)
    # ward_move = check_ward_move(kg_ws.graph)
    # careplan_ontime = check_careplan_ontime(kg_ws.graph)
    # numberof_repositioning = check_reposition(kg_ws.graph)
    # carplan_numberof_repositioning = calculate_numberof_reposition(kg_ws.graph)
    # if careplan in (['No careplan', "Risk but no careplan!","Notyet"]):
    #     careplan_compliance_degree = 0 
    # elif careplan in (['No risk',"0,0"]):
    #     careplan_compliance_degree = 1
    # else:
    #     careplan_compliance_degree = float(careplan.split(",")[1])/carplan_numberof_repositioning
        
    