from .priorKnowledge import FBC
from .priorKnowledge import Albumin
from .priorKnowledge import Creactive
from .priorKnowledge import Hb1AC
from .priorKnowledge import labturnaround
from .priorKnowledge import waterlow_assess_policy
from .priorKnowledge import care_plan_policy
class NodeDecorator:

    def __init__(self):
        self.node_size_reference =  [20, 40, 80, 160]
        self.cmap =['#30a2da','yellow','red','green','black']

    def return_size_color(self, kg,row):
        if row[kg.cd_spec['EventCatalog']] in ['C-reactive protein level, blood','Albumin level, blood','Full blood count, blood','Haemoglobin A1c (Monitoring), blood']:
                indicator = self.get_range_lab(kg,row)                
                if indicator > 0 :
                        size = 30
                        color = 0
                else:
                        size = 40
                        color = 2
        else:
                size = 30
                color = 0
        # if row[kg.cd_spec['EventCatalog']] in ['Waterlow Assess']:
        #         indicator = waterlow_assess_policy.policy_compliance(kg)                
        #         if indicator > 0 :
        #                 size = 30
        #                 color = 0
        #         else:
        #                 size = 40
        #                 color = 2
        
        # if turn_around == range_[0]:
        #         size = 20
        #         color = 0
        # elif range_[0] < turn_around <= range_[1]:
        #         size = 30
        #         color = 0
        # elif range_[1] < turn_around <= range_[2]:
        #         size = 40
        #         color = 0
        # elif range_[2] < turn_around <= range_[3]:
        #         size = 50
        #         color = 1
        # elif turn_around > range_[3]:
        #         size = 60
        #         color = 2
        return size, color

    def get_range_lab(self,kg, row):
        Reference_Range,input_node_fields = FBC.FBC_inputs_reference()
        gender = row['PERSON_GENDER_CODE']
        indicator = 0
        if row[kg.cd_spec['EventCatalog']] == 'C-reactive protein level, blood':                
                range_ = Reference_Range['C-reactive protein']
                input_node_fields = 'C-reactive protein'                
                if int(gender) == 1:
                        range_ = range_['Male']
                elif int(gender) == 2:
                        range_ = range_['Female']
                else:
                        range_=range_['Unknown']
                value = row['C-reactive protein']
                if float(range_['min']) <= value  <= float(range_['max']):
                        indicator +=1
                else:
                        indicator += -1
        if row[kg.cd_spec['EventCatalog']] == 'Full blood count, blood':
                for item in input_node_fields:
                        range_ = Reference_Range[item]
                        if int(gender) == 1:
                                range_ = range_['Male']
                        elif int(gender) == 2:
                                range_ = range_['Female']
                        else:
                                range_=range_['Unknown']
                        value = row[item]
                        if float(range_['min']) <= value  <= float(range_['max']):
                                indicator += 1
                        else:
                                indicator += -1
        if row[kg.cd_spec['EventCatalog']] == 'Albumin level, blood':
                Reference_Range,input_node_fields = Albumin.Albumin_inputs_reference()
                for item in input_node_fields:
                        range_ = Reference_Range[item]
                        if int(gender) == 1:
                                range_ = range_['Male']
                        elif int(gender) == 2:
                                range_ = range_['Female']
                        else:
                                range_=range_['Unknown']
                        value = row[item]
                        if float(range_['min']) <= value  <= float(range_['max']):
                                indicator += 1
                        else:
                                indicator += -1

        if row[kg.cd_spec['EventCatalog']] == 'Haemoglobin A1c (Monitoring), blood':
                Reference_Range,input_node_fields = Hb1AC.Hb1AC_inputs_reference()
                for item in input_node_fields:
                        range_ = Reference_Range[item]
                        if int(gender) == 1:
                                range_ = range_['Male']
                        elif int(gender) == 2:
                                range_ = range_['Female']
                        else:
                                range_=range_['Unknown']
                        value = row[item]
                        if float(range_['min']) <= value  <= float(range_['max']):
                                indicator += 1
                        else:
                                indicator += -1
        return indicator

    def get_value_lab(self, kg,row):
        if row[kg.cd_spec['EventCatalog']] == 'C-reactive protein level, blood':
                value = row['C-reactive protein']
        return value
        
        


