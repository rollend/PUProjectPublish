#The policy is translated based on Pressure Ulcer Prevention Policy  v4.0  CG325
#As each event is modelled as Node, so checking is based on particular node information, process is KG+Node -> Rule check -> return node indicator
#Document Section 5.1 Risk assessment

#5.1 Risk Assessment
#1. • The Trusts chosen pressure ulcer risk assessment tool is the Waterlow Risk Assessment tool
#2. • All in-patients will have a documented risk pressure ulcer assessment (Waterl completed within 4 hours of admission to the Trust.
#3. • Patients in Emergency Department and the Acute Medical Unit will have a documented pressure ulcer risk assessment completed at the first point of contact (within 4 hours).
#4. • All in-patients will have a documented pressure ulcer risk assessment completed within 4 hours of transfer to any ward or department.
#5. • Initial pressure ulcer risk assessment must be undertaken by a registered nurse or midwife or by an Assistant Practitioner. Student nurses/midwives can undertake the assessment under the guidance and supervision of a registered nurse or midwife and the documentation must be countersigned.
#6. • Reassessment can be completed by the Registered Nurse or Midwife or Assistant Practitioner. Nursing Associate trainees and Nursing Associates can undertake the reassessment but are advised to discuss findings with Registered Nurse
#7. • Reassessment of risk should be ongoing and will take place:-
#       Immediately if the patient has undergone any operative procedure or their clinical condition changes
#       As a minimum every seven days.

#8. • Adult general patients must be assessed using informed clinical judgement and the Waterlow risk assessment tool. (Appendix 1)
#9. • Women in the Maternity Unit must be assessed using the adapted Waterlow risk assessment tool. (Appendix 1a)
#10. • Children must be assessed using informed clinical judgement and the Glamorgan risk assessment tool (Appendix 2).
#11. • Neonates must be assessed using informed clinical judgement and the locally (and CSSHA Neonatal network) agreed assessment tool.
#12. • Risk assessment tools act as a formal tool identifying patients at risk of developing pressure ulcers but should be used in conjunction with clinical judgement and expertise.
#13. • The documented pressure ulcer risk assessment should be accessible to all members of the health care team involved in the patient’s care.

#Explain
#rule 1 no doubt will all compliance; 
#rule 2 indicator compare datetime of spell start and earlist waterlow assessment,  should update spell start node (single)
#rule 3 indicator compare datetime of AE start and earlist waterlow assessment, should update spell start node (single)
#rule 4 indicator compare datetime of Ward stay start and closest waterlow assessment, should update each ward stay nodes (multiple)
#rule 5 indicator should update waterlow assess node (not implemented yet, need further information of registered nurse table)
#rule 6 indicator should update waterlow assess node (not implemented yet, need further information of registered nurse table)
#rule 7 indicator compare datetime of OP finish and closest waterlow assessment, should update each OP node 
#rule 7 part two if there is nothing in between, indicator should update each waterlow assess nodes
#rule 8 not yet, age info is in the kg
#rule 9 not yet, gender info is in the kg
#rule 10 not yet, age info is in the kg
from datetime import datetime
import networkx as nx

def policy_compliance(kg):
    indicator = 0
    attrs={}
    nodeOfSpellstart = [0]
    nodesAll = {x:y for x,y in kg.nodes(data=True) if y['name'] == 'Waterlow Assess' or y['name'] == 'Ward Stay'}
    #nodesAll[0]=kg.nodes[0]
    sorted_ = {k:v for k,v in sorted(nodesAll.items(), key=lambda item: item[1]['activity_start_time'])}
    sorted_={**{0:kg.nodes[0]},**sorted_}
    nodesOfWaterlow = [x for x,y in kg.nodes(data=True) if y['name'] == 'Waterlow Assess']
    nodesOfWardStay = [x for x,y in kg.nodes(data=True) if y['name'] == 'Ward Stay']
    #nodesOfOP = [x for x,y in kg.nodes(data=True) if y['name'] == 'Surgery']
    spell_start = {0:kg.nodes[0]}
    nodesOfWaterlow = {x:y for x,y in kg.nodes(data=True) if y['name'] == 'Waterlow Assess'}
    nodesOfWardStay = {x:y for x,y in kg.nodes(data=True) if y['name'] == 'Ward Stay'}
    kg.graph['waterlow_standard_compliance'] = {'rule 1':'Pass'}
    #Rule 2 implementation
    indicator = rule_2(spell_start,nodesOfWaterlow)
    if indicator == 'Pass':
        spell_start[0].update({'size':30
                     ,'color':0
        })
        kg.graph['waterlow_standard_compliance'].update({'rule 2':'Pass'})
    else:
        spell_start[0].update({'size':40
                     ,'color':2
        })
        kg.graph['waterlow_standard_compliance'].update({'rule 2':'Fail'})
    attrs.update(spell_start)
    
    #Rule 3 implementation
    indicator = rule_3(spell_start,nodesOfWaterlow)
    if indicator == 'Pass':
        spell_start.update({'size':30
                     ,'color':0
        })
        kg.graph['waterlow_standard_compliance'].update({'rule 3':'Pass'})
    else:
        spell_start.update({'size':40
                     ,'color':2
        })
        kg.graph['waterlow_standard_compliance'].update({'rule 3':'Fail'})
    attrs.update(spell_start)
    
    #Rule 4 implementation
    virtual_wards = {}
    for key,value in nodesOfWardStay.items():
        if 'virtual' not in value['WARD STAY LOCATION'].lower(): #or 'recovery' in value['WARD STAY LOCATION'].lower():
            indicator = rule_4(value,nodesOfWaterlow)
            if indicator == 'Pass':
                value.update({'size':30
                            ,'color':0
                })
                try:
                    kg.graph['waterlow_standard_compliance']['rule 4']
                except:                   
                    kg.graph['waterlow_standard_compliance'].update({'rule 4':'Pass'})
            else:
                value.update({'size':40
                            ,'color':2
                })
                kg.graph['waterlow_standard_compliance'].update({'rule 4':'Fail'})
            attrs.update({key:value})
        else:
            virtual_wards.update({key:value})
    if len(virtual_wards) > 0:
        sorted_ = {k:v for k,v in sorted(virtual_wards.items(), key=lambda item: item[1]['activity_start_time'],reverse=True)}
        values_view = sorted_.values()
        value_iterator = iter(values_view)
        value = next(value_iterator)
        keys_view = sorted_.keys()
        key_iterator = iter(keys_view)
        key = next(key_iterator)
        indicator = rule_4(value,nodesOfWaterlow)
        if indicator == 'Pass':
            value.update({'size':30
                        ,'color':0
            })
            try:
                kg.graph['waterlow_standard_compliance']['rule 4']
            except:                   
                kg.graph['waterlow_standard_compliance'].update({'rule 4':'Pass'})
        else:
            value.update({'size':40
                        ,'color':2
            })
            kg.graph['waterlow_standard_compliance'].update({'rule 4':'Fail'})
        attrs.update({key:value})
    nx.set_node_attributes(kg,attrs)  
   
    return kg

def rule_1(kg):
    return

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

def hour1(event_start_dt ,event_start_dt2):
    #waterlow_start_dt = row[kg.cd_spec['EventStartDT']]
    #waterlow_result_dt = row[kg.cd_spec['EventEndDT']]
    #spell_start = row[kg.cd_spec['SpellStartDT']]
    difference = event_start_dt - event_start_dt2
    duration_in_s = difference.total_seconds()
    hours = divmod(duration_in_s, 3600)[0]
    if hours <= 1:
        return 'Pass'
    else:
        return 'Fail'

def rule_2(spell_start, waterlowAll):
    sorted_ = {k:v for k,v in sorted(waterlowAll.items(), key=lambda item: item[1]['activity_start_time'])}
    values_view = sorted_.values()
    value_iterator = iter(values_view)
    first_value = datetime.strptime(next(value_iterator)['activity_start_time'],'%Y.%m.%d %H:%M:%S')
    second_value = datetime.strptime(spell_start[0]['activity_start_time'],'%Y.%m.%d %H:%M:%S')
    check = hour4(first_value, second_value)
    return check
  
def rule_3(spell_start, waterlowAll):
    sorted_ = {k:v for k,v in sorted(waterlowAll.items(), key=lambda item: item[1]['activity_start_time'])}
    values_view = sorted_.values()
    value_iterator = iter(values_view)
    first_value = datetime.strptime(next(value_iterator)['activity_start_time'],'%Y.%m.%d %H:%M:%S')
    second_value = datetime.strptime(spell_start[0]['activity_start_time'],'%Y.%m.%d %H:%M:%S')
    check = hour4(first_value, second_value)
    return check

def rule_4(ward_value,nodesOfWaterlow):
    #waterlow_start_dt = row[kg.cd_spec['EventStartDT']]
    #waterlow_result_dt = row[kg.cd_spec['EventEndDT']]
    #Episode_start = row['EPISODE_START_DATE_TIME']
    event_start = datetime.strptime(ward_value['activity_start_time'],'%Y.%m.%d %H:%M:%S')
    sorted_ = {k:v for k,v in sorted(nodesOfWaterlow.items(), key=lambda item: item[1]['activity_start_time'])}
    filtered_watlerlow = {k:v for k,v in sorted_.items() if datetime.strptime(v['activity_start_time'],'%Y.%m.%d %H:%M:%S') >= event_start}
    if len(filtered_watlerlow)> 0 :
        values_view = filtered_watlerlow.values()
        value_iterator = iter(values_view)
        first_value = datetime.strptime(next(value_iterator)['activity_start_time'],'%Y.%m.%d %H:%M:%S')
        check = hour1(first_value, event_start)
    else:
        return 'Fail'
    return check

def rule_5(kg,row):
    return

def rule_6(kg,row):
    return

def rule_7(waterlow_start_dt, Surgic_finish):
    difference = waterlow_start_dt - Surgic_finish
    duration_in_s = difference.total_seconds()
    hours = divmod(duration_in_s, 3600)[0]
    if hours <= 1:
        return 'Pass'
    else:
        return 'Fail'
    

def rule_8(kg,row):
    return

def rule_9(kg,row):
    return

def rule_10(kg,row):
    return

def rule_11(kg,row):
    return

def rule_12(kg,row):
    return

def rule_13(kg,row):
    return