import numpy as np
import logging
log = logging.getLogger(__name__)
def return_clean(DF,cd_spec): 
    
    
    encode_list=[#'Chief Complaint SNOMED Code'
            #,'PRESENTING_COMPLAINT'
            cd_spec['EventCatalog'] #'ORDER_CATALOG_DESCRIPTION'
            ,'WARD STAY LOCATION'
            ,'ETHNIC_CATEGORY_CODE'
            ,'PERSON_MARITAL_STATUS_CODE'
            ,'RELIGIOUS_OR_OTHER_BELIEF_SYSTEM_AFFILIATION'
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
      ,'WL - Waterlow Score'
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
      ,'Referral - Tissue Viability','PATIENTPOSITION']
    for column in DF[DF.columns.difference(encode_list)]:
        try:
            DF[column] = DF[column].replace(' ', np.NaN).replace('----',np.NaN).replace('----',np.NaN, regex=True).replace('[a-zA-Z]',np.NaN,regex=True).astype(float)
        except Exception as e:
            if column == 'C-REACTIVE PROTEIN':
                DF[column] = DF[column].replace('<1', 0.5).replace('<0.2', 0.5).replace('<0.3', 0.5).replace(' ', np.NaN).replace('[a-zA-Z]',np.NaN,regex=True).astype(float)
            elif e == 'cannot astype a datetimelike from [datetime64[ns]] to [float64]':
                pass
            else:
                log.debug("ID: {}, Column: {}, Exception:{}".format(DF['ACTIVITY_IDENTIFIER'].unique(),column,e))
            continue        
    return DF