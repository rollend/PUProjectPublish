#Rule 1: Pressure Ulcer Care Plan Initiated for The patients At Risk
#Rule 2: Pressure Ulcer 4 Hourly Evidence of Repositioning
#Rule 3: PU repositioning evidence 4 hourly in last 3 days
#Rule 4: PU repositioning evidence 6 hoursly in last 3 days

#Extracted from 
#Who is at risk?
#PRESSURE_ULCERS_RISK_DT_TM (SCENARIO 1)
#Last WLWATERLOWSCORE > 10

#PRESSURE_ULCERS_RISK_DT_TM (SCENARIO 2)
#WARD_STAY_START_DATE_TIME > previous 72 hours (3 days)

#PRESSURE_ULCERS_RISK_DT_TM (SCENARIO 3)
#Use the date of the first WLWATERLOWSCORE > 10 if it is after the ward stay start date
#and the first WLWATERLOWSCORE of the ward stay is <= 10 

#FIRST PRESSURE_ULCERS_POSITIONING_LAST_3_DAYS_FLAG (SCENARIO 1)
#Set flag to 0 for the following parameters
#Patient at risk of ulcer
#Patient has not been repositioned within 4 hours since the date_time patient was flagged as high risk and the date_time the data was collated

#PRESSURE_ULCERS_POSITIONING_LAST_3_DAYS_FLAG EXCLUDING FIRST EVENT (SCENARIO 2)
#Set flag to 0 for the following parameters
#Patient at risk of ulcer
#Patient has not been repositioned every 4 hours since the date_time the patient was flagged as high risk and the date_time the data was collated

#LAST PRESSURE_ULCERS_POSITIONING_LAST_3_DAYS_FLAG (SCENARIO 3)
#Set flag to 0 for the following parameters
#Patient at risk of ulcer
#Patient has not been repositioned for the last 4 hours before the data was collated

#PRESSURE_ULCERS_POSITIONING_LAST_3_DAYS_FLAG (SCENARIO 4)
#Set flag to 0 for the following parameters
#Patient is not at risk of ulcers

#PRESSURE_ULCERS_POSITIONING_LAST_3_DAYS_FLAG (SCENARIO 5)
#Set flag to 0 for the following parameters
#Patient is at risk of ulcers
#Patient has never been repositioned
#Patient was last repositioned more than 3 days ago

#FIRST PRESSURE_ULCERS_POSITIONING_6HOURLY_LAST_3_DAYS_FLAG (SCENARIO 1)
#Set flag to 0 for the following parameters
#Patient at risk of ulcer
#Patient has not been repositioned within 6 hours since the date_time patient was flagged as high risk

#PRESSURE_ULCERS_POSITIONING_6HOURLY_LAST_3_DAYS_FLAG EXCLUDING FIRST EVENT (SCENARIO 2)
#Set flag to 0 for the following parameters
#Patient at risk of ulcer
#Patient has not been repositioned every 6 hours since the date_time the patient was flagged as high risk

#LAST PRESSURE_ULCERS_POSITIONING_6HOURLY_LAST_3_DAYS_FLAG (SCENARIO 3)
#Set flag to 0 for the following parameters
#Patient at risk of ulcer
#Patient has not been repositioned every 6 hours since the date_time the patient was flagged as high risk

#PRESSURE_ULCERS_POSITIONING_6HOURLY_LAST_3_DAYS_FLAG (SCENARIO 4)
#Set flag to 0 for the following parameters
#Patient is not at risk of ulcers

#PRESSURE_ULCERS_POSITIONING_LAST_3_DAYS_FLAG (SCENARIO 5)
#Set flag to 0 for the following parameters
#Patient is at risk of ulcers
#Patient has never been repositioned
#Patient was last repositioned more than 3 days ago


#Put it simple, Risk DT Time is the first datetime waterlow score > 10
#Once patient at high risk, within 4 hours must be first time repositioned and every 4 hours after, if not, flag
#Once patient at risk, within 6 hours must be first time repositioned and every 6 hours after, if not, flag

#

#Nice standard states that 
# 'For safety reasons, repositioning is recommended at least every 6 hours for adults at risk,
# and every 5 hours for adults at high risk. For children and yound people at risk, repositioning is recommended at least every 4 hours, 
# and more frequently for those at high risk.'

#RBH Local implementation is that we are aiming every 4 hours repositioning of patient above risk score 10, 6hours is a counter measure that to double check and as 4 hours policy is 
# too strict to compliant. Note: This logic is attached to the real time application which moving forward as time pointer. Retrospective analysis speaking, as there is no real time pointer moving
#it is not possible to implement as a real time flag. 

#Together with NICE standard, current implementation follows 4 hours and 6 hours, node color updates on patient position node, 4 hours indicated by yellow, 6 hours indicated by red;
#Thus each node represent a compliance with 4/6 hours policy; Overall plan compliance records number of breaches accordingly.  


from datetime import datetime
import networkx as nx

def policy_compliance(kg):
    nodesOfCarePlan,dt_atrisk=check_careplan_init(kg)
    attrs={}
    if dt_atrisk == '0000':
        kg.graph.update({'care_plan_compliance': 'No risk'})
        kg.graph.update({'care_plan_ontime': 1})
    elif dt_atrisk !='0000' and len(nodesOfCarePlan) ==0:
        kg.graph.update({'care_plan_compliance': 'Risk but no careplan!'})
        kg.graph.update({'care_plan_ontime': 0})
    elif dt_atrisk !='0000' and len(nodesOfCarePlan) >0:
        check = hour4(datetime.strptime(nodesOfCarePlan['activity_start_time'],'%Y.%m.%d %H:%M:%S'), datetime.strptime(dt_atrisk,'%Y.%m.%d %H:%M:%S'))
        if check =='Pass':
            kg.graph.update({'care_plan_ontime': 1})
        else:
            kg.graph.update({'care_plan_ontime': 0})    
        try:
            kg = careplan_compliance(kg)
        except:
            kg.graph.update({'care_plan_compliance': 'Error!'})
    return kg

def check_careplan_init(kg):
    nodesOfWaterlow = {x:y for x,y in kg.nodes(data=True) if y['name'] == 'Waterlow Assess'}
    sorted_ = {k:v for k,v in sorted(nodesOfWaterlow .items(), key=lambda item: item[1]['activity_start_time'])}
    try:
        nodesOfCarePlan = {x:y for x,y in kg.nodes(data=True) if y['name'] == 'Patient Position' and ('Pressure Ulcer' in y['PATIENTPOSITION'])}
    except:
        nodesOfCarePlan = ''
    for key,value in sorted_.items():
        if value['WL - Waterlow Score'] !='None' and int(value['WL - Waterlow Score']) > 10:
            dt_atrisk = value['activity_end_time']
            break
        else:
            dt_atrisk = '0000'
    
    return nodesOfCarePlan,dt_atrisk

def careplan_compliance(kg):
    fourhour_breach = 0
    sixhour_breach = 0
    attrs = {}
    nodesOfWaterlow = {x:y for x,y in kg.nodes(data=True) if y['name'] == 'Waterlow Assess'}
    sorted_waterlow = {k:v for k,v in sorted(nodesOfWaterlow.items(), key=lambda item: item[1]['activity_start_time'])}
    nodesOfPatientPosition = {x:y for x,y in kg.nodes(data=True) if y['name'] == 'Patient Position'}
    sorted_PatientPosition = {k:v for k,v in sorted(nodesOfPatientPosition.items(), key=lambda item: item[1]['activity_start_time'])}
    for key,value in sorted_waterlow.items():
        if int(value['WL - Waterlow Score']) > 10:
            dt_firstatrisk = value['activity_end_time']
            filtered_watlerlow = {k:v for k,v in sorted_waterlow.items() if datetime.strptime(v['activity_start_time'],'%Y.%m.%d %H:%M:%S') >= datetime.strptime(value['activity_end_time'],'%Y.%m.%d %H:%M:%S')}
            break
    for key, value in  filtered_watlerlow.items():
        if int(value['WL - Waterlow Score']) <10:
            dt_recover = value['activity_end_time']
            leftover_waterlow = {k:v for k,v in sorted_waterlow.items() if datetime.strptime(v['activity_start_time'],'%Y.%m.%d %H:%M:%S') >= datetime.strptime(value['activity_end_time'],'%Y.%m.%d %H:%M:%S')}
        else:
            dt_recover = kg.nodes[1]['activity_end_time']
            leftover_waterlow = ''
    if dt_recover == kg.nodes[1]['activity_end_time']:
        only_after_risk = {k:v for k,v in sorted_PatientPosition.items() if datetime.strptime(v['activity_start_time'],'%Y.%m.%d %H:%M:%S') >= datetime.strptime(value['activity_end_time'],'%Y.%m.%d %H:%M:%S')}
        key_list = list(only_after_risk.keys())
        for index in range(len(key_list)):
            #print(index, key_list[index])
            #print(only_after_risk[key_list[index]])
            if index == 0:
                check4 = hour4(datetime.strptime(only_after_risk[key_list[index]]['activity_start_time'],'%Y.%m.%d %H:%M:%S'),datetime.strptime(dt_firstatrisk,'%Y.%m.%d %H:%M:%S'))
                check6 = hour6(datetime.strptime(only_after_risk[key_list[index]]['activity_start_time'],'%Y.%m.%d %H:%M:%S'),datetime.strptime(dt_firstatrisk,'%Y.%m.%d %H:%M:%S'))
                value = only_after_risk[key_list[index]]
                if check4 =='Pass' and check6 == 'Pass':
                    value.update({'size':30
                        ,'color':0
                    })
                    attrs.update({key_list[index]:value})
                elif check4 == 'Pass' and check6 == 'Fail':
                    value.update({'size':30
                        ,'color':0
                    })
                    sixhour_breach += 1
                    attrs.update({key_list[index]:value})
                elif check4 == 'Fail' and check6 == 'Pass':
                    value.update({'size':40
                        ,'color':1
                    })
                    fourhour_breach += 1
                    attrs.update({key_list[index]:value})
                elif check4 == 'Fail' and check6 == 'Fail':
                    value.update({'size':40
                        ,'color':2
                    })
                    fourhour_breach += 1
                    sixhour_breach += 1
                    attrs.update({key_list[index]:value})                
            else:
                check4 = hour4(datetime.strptime(only_after_risk[key_list[index]]['activity_start_time'],'%Y.%m.%d %H:%M:%S'),datetime.strptime(only_after_risk[key_list[index-1]]['activity_start_time'],'%Y.%m.%d %H:%M:%S'))
                check6 = hour6(datetime.strptime(only_after_risk[key_list[index]]['activity_start_time'],'%Y.%m.%d %H:%M:%S'),datetime.strptime(only_after_risk[key_list[index-1]]['activity_start_time'],'%Y.%m.%d %H:%M:%S'))
                value = only_after_risk[key_list[index]]
                if check4 =='Pass' and check6 == 'Pass':
                    value.update({'size':30
                        ,'color':0
                    })
                    attrs.update({key_list[index]:value})
                elif check4 == 'Pass' and check6 == 'Fail':
                    value.update({'size':30
                        ,'color':0
                    })
                    sixhour_breach += 1
                    attrs.update({key_list[index]:value})
                elif check4 == 'Fail' and check6 == 'Pass':
                    value.update({'size':40
                        ,'color':1
                    })
                    fourhour_breach += 1
                    attrs.update({key_list[index]:value})
                elif check4 == 'Fail' and check6 == 'Fail':
                    value.update({'size':40
                        ,'color':2
                    })
                    fourhour_breach += 1
                    sixhour_breach += 1
                    attrs.update({key_list[index]:value})            
    nx.set_node_attributes(kg,attrs)
    kg.graph.update({'care_plan_compliance': '{},{}'.format(fourhour_breach,sixhour_breach)})        
    return kg

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

def hour6(event_start_dt ,event_start_dt2):
    #waterlow_start_dt = row[kg.cd_spec['EventStartDT']]
    #waterlow_result_dt = row[kg.cd_spec['EventEndDT']]
    #spell_start = row[kg.cd_spec['SpellStartDT']]
    difference = event_start_dt - event_start_dt2
    duration_in_s = difference.total_seconds()
    hours = divmod(duration_in_s, 3600)[0]
    if hours <= 6:
        return 'Pass'
    else:
        return 'Fail'
