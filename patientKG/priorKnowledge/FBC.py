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

def FBC_inputs_reference():
        Reference_Range = {
            'Haemoglobin':{'Male':{'min':'130','max':'180'}
                                        ,'Female':{'min':'115','max':'165'}
                                        ,'Unknown':{'min':'115','max':'180'}
                                        ,'Unspecified':{'min':'115','max':'180'}},
            'White blood cell count':{'Male':{'min':'4','max':'11'}
                                    ,'Female':{'min':'4','max':'11'}
                                    ,'Unknown':{'min':'4','max':'11'}
                                    ,'Unspecified':{'min':'4','max':'11'}},
            'Platelet count':{'Male':{'min':'150','max':'450'}
                            ,'Female':{'min':'150','max':'450'}
                            ,'Unknown':{'min':'150','max':'450'}
                            ,'Unspecified':{'min':'150','max':'450'}},
            'Lymphocyte count':{'Male':{'min':'1.5','max':'4.5'}
                                ,'Female':{'min':'1.5','max':'4.5'}
                            ,'Unknown':{'min':'1.5','max':'4.5'}
                                ,'Unspecified':{'min':'1.5','max':'4.5'}},
            'C-reactive protein':{'Male':{'min':'0','max':'5'}
                                ,'Female':{'min':'0','max':'5'}
                            ,'Unknown':{'min':'0','max':'5'}
                                ,'Unspecified':{'min':'0','max':'5'}},
            'Red blood cell count':{'Male':{'min':'4.7','max':'6.1'}
                                ,'Female':{'min':'4.2','max':'5.4'}
                            ,'Unknown':{'min':'4.2','max':'6.1'}
                                ,'Unspecified':{'min':'4.2','max':'6.1'}},
            'Mean cell haemoglobin':{'Male':{'min':'27','max':'32'}
                                ,'Female':{'min':'27','max':'32'}
                            ,'Unknown':{'min':'27','max':'32'}
                                ,'Unspecified':{'min':'27','max':'32'}},
            'Mean cell volume':{'Male':{'min':'80','max':'100'}
                                ,'Female':{'min':'80','max':'100'}
                            ,'Unknown':{'min':'80','max':'100'}
                                ,'Unspecified':{'min':'80','max':'100'}}
            ,'Mean cell haemoglobin conc':{'Male':{'min':'320','max':'360'}
                                ,'Female':{'min':'320','max':'360'}
                            ,'Unknown':{'min':'320','max':'360'}
                                ,'Unspecified':{'min':'320','max':'360'}}
            ,'Monocyte count':{'Male':{'min':'0.2','max':'0.8'}
                                ,'Female':{'min':'0.2','max':'0.8'}
                            ,'Unknown':{'min':'0.2','max':'0.8'}
                                ,'Unspecified':{'min':'0.2','max':'0.8'}}
            ,'Neutrophil count':{'Male':{'min':'2.0','max':'7.5'}
                                ,'Female':{'min':'2.0','max':'7.5'}
                            ,'Unknown':{'min':'2.0','max':'7.5'}
                                ,'Unspecified':{'min':'2.0','max':'7.5'}}
            ,'Eosinophil count':{'Male':{'min':'0','max':'0.4'}
                                ,'Female':{'min':'0','max':'0.4'}
                            ,'Unknown':{'min':'0','max':'0.4'}
                                ,'Unspecified':{'min':'0','max':'0.4'}}
            ,'Basophil count':{'Male':{'min':'0','max':'0.1'}
                                ,'Female':{'min':'0','max':'0.1'}
                            ,'Unknown':{'min':'0','max':'0.1'}
                                ,'Unspecified':{'min':'0','max':'0.1'}}
            ,'Haematocrit':{'Male':{'min':'0.4','max':'0.52'}
                                ,'Female':{'min':'0.37','max':'0.47'}
                            ,'Unknown':{'min':'0.37','max':'0.52'}
                                ,'Unspecified':{'min':'0.37','max':'0.52'}}
                        }
        input_node_fields = ['Red blood cell count'
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
        ]
        return Reference_Range,input_node_fields
