# Copyright 2018 The GraphNets Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

#This is priorknowledge from EPR
#ORDER_CATALOG_DESCRIPTION	ORDER_EVENT
#Full blood count, blood	Absolute Blast Count
#Full blood count, blood	Absolute Metamyelocyte Count
#Full blood count, blood	Absolute Myelocyte Count
#Full blood count, blood	Absolute Promyelocyte Count
#Full blood count, blood	Basophil count
#Full blood count, blood	Blood Culture Sample
#Full blood count, blood	Eosinophil count
#Full blood count, blood	Haematocrit
#Full blood count, blood	Haemoglobin
#Full blood count, blood	HCT
#Full blood count, blood	Lymphocyte count
#Full blood count, blood	Mean cell haemoglobin
#Full blood count, blood	Mean cell haemoglobin conc
#Full blood count, blood	Mean cell volume
#Full blood count, blood	Monocyte count
#Full blood count, blood	Neutrophil count
#Full blood count, blood	Platelet count
#Full blood count, blood	Red blood cell count
#Full blood count, blood	White blood cell count

def Albumin_inputs_reference():
        Reference_Range = {
            'Albumin':{'Male':{'min':'34','max':'54'}
                                        ,'Female':{'min':'34','max':'54'}
                                        ,'Unknown':{'min':'34','max':'54'}
                                        ,'Unspecified':{'min':'34','max':'54'}}            
                        }
        input_node_fields = ['Albumin']
                            
        return Reference_Range,input_node_fields
