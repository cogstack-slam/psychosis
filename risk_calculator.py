#!/usr/bin/env python
# coding: utf-8

# In[1]:



# In[2]:


from elasticsearch import Elasticsearch
from elasticsearch_dsl import Search
import pandas as pd
import numpy as np
import json
from datetime import date, datetime
import logging
from elastic_util import *
import time


# # Load data from Cogstack

# In[3]:


logger = logging.getLogger('psychosis_risk_cal')
handler = logging.StreamHandler()
formatter = logging.Formatter(
        '%(asctime)s %(name)-12s %(levelname)-8s : %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.DEBUG)


# In[26]:


def read_elas_index(client, index, doc_type='doc'):
    while not client.indices.exists(index):
        time.sleep(30)
    s = Search(using=client, index=index, doc_type=doc_type).filter('match', coeff_validated='0')
    logger.info('Loaded %s documents' %s.count())
    df = pd.DataFrame((d.to_dict() for d in s.scan()))
    logger.info('Documents shape: %s %s' %(df.shape))
    
    logger.info('Change data types')
    df['first_primary_diagnosis_date'] = pd.to_datetime(df['first_primary_diagnosis_date'])
    df['patient_date_of_birth'] = pd.to_datetime(df['patient_date_of_birth'])
    df['first_primary_diagnosis_recorded_date'] = pd.to_datetime(df['first_primary_diagnosis_recorded_date'])
#     print(df.dtypes)
#     df['patient_id'] = df['patient_id'].astype(int)
#     df['referral_id'] = df['referral_id'].astype(int)
    logger.info('Documents shape after changing data types: %s %s' %(df.shape))
    return df


# # Ethnicity (changed to lower for string match and added two mapping rules)

# In[5]:


# Added the following two lines in original list
# Scottish (CB) --- British (A) 
# African --- African (N)
def read_eth_mapping(file_path='ethnicity_mapping.txt'):
    eth_map = {}
    with open(file_path, 'r') as fr:
        for line in fr.readlines():
            tokens = line.strip().split('---')
            eth_map[tokens[0].strip().lower()] = tokens[1].strip().lower()
    return eth_map


# In[6]:


def ethnicity_coeff(df):
    eth_map = read_eth_mapping()
    logger.info('Mapping ePJS ethnicity names to CRIS ethnicity names')
    df['patient_demography_ethnicity_cris'] = df['patient_demography_ethnicity'].str.lower()
    df['patient_demography_ethnicity_cris'] = df['patient_demography_ethnicity_cris'].map(eth_map)
#     print(set(df['patient_demography_ethnicity_cris']))
    
    logger.info('Calcuating ethnicity coeffs')
    df['eth_coeff'] = None
    black = [s.lower() for s in ['Caribbean (M)','African (N)','Any other black background (P)']]
    white = [s.lower() for s in ['British (A)','Irish (B)','Any other white background (C)']]
    asian = [s.lower() for s in ['Indian (H)','Pakistani (J)','Any other Asian background (L)','Chinese (R)','Bangladeshi (K)']]
    mixed = [s.lower() for s in ['White and black Caribbean (D)','White and Black African (E)','White and Asian (F)','Any other mixed background (G)']]
    other = [s.lower() for s in ['Any other ethnic group (S)']]

    df.loc[df['patient_demography_ethnicity_cris'].isin(asian), 'eth_coeff'] = 0.5143438
    df.loc[df['patient_demography_ethnicity_cris'].isin(black), 'eth_coeff'] = 1.037915
    df.loc[df['patient_demography_ethnicity_cris'].isin(mixed), 'eth_coeff'] = 0.6044039
    df.loc[df['patient_demography_ethnicity_cris'].isin(other), 'eth_coeff'] = 0.4081036
    df.loc[df['patient_demography_ethnicity_cris'].isin(white), 'eth_coeff'] = 0.0
    df['eth_coeff'] = df['eth_coeff'].astype(float)
    
    # CRIS ethnicity categories that are not mapped in the code
#     print(set(df['patient_demography_ethnicity_cris']) - set(black) - set(white) - set(asian) - set(mixed) - set(other))
    logger.info('Finished calcuating ethnicity coeffs')
    return df


# # Gender

# In[7]:


def gender_coeff(df):
    logger.info('Calcuating gender coeffs')
    df['gender_coeff'] = None
    df.loc[df['patient_demography_gender']=='Female', 'gender_coeff'] = 0.0 
    df.loc[df['patient_demography_gender']=='Male', 'gender_coeff'] = 0.5681779    
    df['gender_coeff'] = df['gender_coeff'].astype(float)
    logger.info('Finished calcuating gender coeffs')
    return df


# # Age

# In[8]:


# np.floor(13.2)


# In[9]:


# Age should be int, not float. using floor. 
def age_coeff(df):
    logger.info('Calcuating age coeffs')
    df['age_coeff'] = None

    df['age_at_index_diagnosis'] = np.floor((df['first_primary_diagnosis_date'] - df['patient_date_of_birth'])/np.timedelta64(1, 'Y'))
#     Remove age < 0 to remove system error in 2009-11-27/28
    df = df.copy()
    df = df.loc[df['age_at_index_diagnosis'] >= 0]
    df['age_coeff'] = df['age_at_index_diagnosis']*0.0117113
    df['age_coeff'] = df['age_coeff'].astype(float)
    logger.info('Finished calcuating age coeffs')
    return df


# # Gender & Age

# In[10]:


def gender_age_coeff(df):
    logger.info('Calcuating gender*age coeffs')
    df['gender_age_coeff'] = None
    df.loc[df['patient_demography_gender']=='Female', 'gender_age_coeff'] = 0.0
    df.loc[df['patient_demography_gender']=='Male', 'gender_age_coeff'] = 0.0121931*df['age_at_index_diagnosis']
    df['gender_age_coeff'] = df['gender_age_coeff'].astype(float)
    logger.info('Finished calcuating gender*age coeffs')
    return df


# # Diagnosis index (added exclusions)

# In[11]:


def diag_coeff(df):
    # Diagnosis mapping
    
    Psychotic = tuple(['F20', 'F25.0', 'F25.1', 'F25.2', 'F25.8', 'F25.9', 'F22.0', 'F22.8', 'F22.9', 'F24', 'F28', 'F29',
                      'F30.2', 'F31.2', 'F31.5', 'F32.3', 'F33.3', 'F53.1'] + 
                     ['F1'+ str(i) + '.' + str(j) for i in range(10) for j in [4, 5, 7]])
    Psychotic_exclude = tuple(['F20.7'])
    
    Acute = tuple(['F23'])

    Substance = tuple(['F1'])
    Substance_exclude = tuple(['F1'+ str(i) + '.' + str(j) for i in range(10) for j in [4, 5, 7]])

    Bipolar = tuple(['F31', 'F34.0', 'F30'])
    Bipolar_exclude = tuple(['F31.2', 'F31.5', 'F30.2'])

    Non_bipolar = tuple(['F32', 'F33', 'F34.1', 'F34.8', 'F34.9', 'F38', 'F39'])
    Non_bipolar_exclude = tuple(['F32.3', 'F33.3'])

    Anxiety = tuple(['F40', 'F41', 'F42', 'F43', 'F44', 'F45', 'F48'])

    Personality = tuple(['F60', 'F61', 'F62', 'F68', 'F69', 'F21', 'F63', 'F64', 'F65', 'F66'])

    Developmental = tuple(['F80', 'F81', 'F82', 'F83', 'F84', 'F88', 'F89'])

    Childhood = tuple(['F90', 'F91', 'F92', 'F93', 'F94', 'F98', 'F95'])

    Physiological = tuple(['F50', 'F51', 'F52', 'F53', 'F54', 'F55', 'F59'])
    Physiological_exclude = tuple(['F53.1'])

    Mental = tuple(['F70', 'F71', 'F72', 'F73', 'F78', 'F79'])
    
    Organic = tuple(['F0'])
    
    logger.info('Mapping ICD-10 codes to disorder categories')
    df['diagnosis_group'] = None
# 
    df.loc[df['first_primary_diagnosis'].notna(), 'diagnosis_group'] = 'Other' # All other diagnoses not in list of interest
    df.loc[df['first_primary_diagnosis'].str.startswith(Psychotic) & (~df['first_primary_diagnosis'].str.startswith(Psychotic_exclude)), 'diagnosis_group'] = 'Psychotic disorder'
    df.loc[df['first_primary_diagnosis'].str.startswith(Acute), 'diagnosis_group'] = 'Acute and transient psychotic disorders'
    df.loc[df['first_primary_diagnosis'].str.startswith(Substance) & (~df['first_primary_diagnosis'].str.startswith(Substance_exclude)), 'diagnosis_group'] = 'Substance use disorders'
    df.loc[df['first_primary_diagnosis'].str.startswith(Bipolar) & (~df['first_primary_diagnosis'].str.startswith(Bipolar_exclude)), 'diagnosis_group'] = 'Bipolar mood disorders'
    df.loc[df['first_primary_diagnosis'].str.startswith(Non_bipolar) & (~df['first_primary_diagnosis'].str.startswith(Non_bipolar_exclude)), 'diagnosis_group'] = 'Non bipolar mood disorders'
    df.loc[df['first_primary_diagnosis'].str.startswith(Anxiety), 'diagnosis_group'] = 'Anxiety disorders'
    df.loc[df['first_primary_diagnosis'].str.startswith(Personality), 'diagnosis_group'] = 'Personality disorders'
    df.loc[df['first_primary_diagnosis'].str.startswith(Developmental), 'diagnosis_group'] = 'Developmental disorders'
    df.loc[df['first_primary_diagnosis'].str.startswith(Childhood), 'diagnosis_group'] = 'Childhood/adolescence onset disorders'
    df.loc[df['first_primary_diagnosis'].str.startswith(Physiological) & (~df['first_primary_diagnosis'].str.startswith(Physiological_exclude)), 'diagnosis_group'] = 'Physiological syndromes'
    df.loc[df['first_primary_diagnosis'].str.startswith(Mental), 'diagnosis_group'] = 'Mental retardation'
    df.loc[df['first_primary_diagnosis'].str.startswith(Organic), 'diagnosis_group'] = 'Organic mental disorder'
    logger.info('Finshed mapping ICD-10 codes to disorder categories')
    
    logger.info('Calculating diagnosis coeffs')
    df['diagnosis_group_coeff'] = None
#     df.loc[df['diagnosis_group'] == 'Psychotic disorder', 'diagnosis_group_coeff'] = 0.0 # Comment this for excluding 'Psychotic disorder'
    df.loc[df['diagnosis_group'] == 'Acute and transient psychotic disorders', 'diagnosis_group_coeff'] = 0.9867204
    df.loc[df['diagnosis_group'] == 'Substance use disorders', 'diagnosis_group_coeff'] = -1.925903
    df.loc[df['diagnosis_group'] == 'Bipolar mood disorders', 'diagnosis_group_coeff'] = -0.1754082
    df.loc[df['diagnosis_group'] == 'Non bipolar mood disorders', 'diagnosis_group_coeff'] = -1.886428
    df.loc[df['diagnosis_group'] == 'Anxiety disorders', 'diagnosis_group_coeff'] = -2.235825
    df.loc[df['diagnosis_group'] == 'Personality disorders', 'diagnosis_group_coeff'] = -1.547794
    df.loc[df['diagnosis_group'] == 'Developmental disorders', 'diagnosis_group_coeff'] = -3.466732
    df.loc[df['diagnosis_group'] == 'Childhood/adolescence onset disorders', 'diagnosis_group_coeff'] = -3.25382
    df.loc[df['diagnosis_group'] == 'Physiological syndromes', 'diagnosis_group_coeff'] = -2.463145
    df.loc[df['diagnosis_group'] == 'Mental retardation', 'diagnosis_group_coeff'] = -2.450679
    
    df['diagnosis_group_coeff'] = df['diagnosis_group_coeff'].astype(float)
    logger.info('Finished calculating diagnosis coeffs')
    
    
    logger.info('Labeling records that do not miss any predictors')
#     df['coeff_validated'] = False
    df.loc[df['diagnosis_group'].notna() & df['age_coeff'].notna()
           & df['eth_coeff'].notna() & df['gender_coeff'].notna(), 'coeff_validated'] = '1'
    return df


# # Risk scores

# In[12]:


def risk_score(df):
    logger.info('Calculating risk scores per year')
    
    df['PI'] = None
    df.loc[df['diagnosis_group_coeff'].notna() & df['age_coeff'].notna()
           & df['eth_coeff'].notna() & df['gender_coeff'].notna(), 'PI'] = df['gender_coeff'] + df['age_coeff'] - df['gender_age_coeff'] + df['eth_coeff'] + df['diagnosis_group_coeff']
    df['risk_calculated_dttm'] = datetime.now()
#     dfvald = df.loc[df['coeff_validated'] == True] # Records that with "coeff_validated == True" can hava diagnoses in 'Psychotic disorder','Organic mental disorder' and others. Risk scores of these records are not calculated
    
    df.loc[df['PI'].notna(), 'exist_risk_score'] = '1'
    
    dfpi = df.loc[df['exist_risk_score'] == '1']
    dfpi = dfpi.copy()
    dfpi['PI'] = dfpi['PI'].astype(float)
    
    # https://data.princeton.edu/wws509/notes/c7.pdf Eq. 7.11  
    pi_exp = np.exp(dfpi['PI'].astype(float))
    dfpi.loc[dfpi['PI'].notna(),'h_1_year'] = 1 - (0.9714991 ** pi_exp)
    dfpi.loc[dfpi['PI'].notna(),'h_2_year'] = 1 - (0.9540228 ** pi_exp)
    dfpi.loc[dfpi['PI'].notna(),'h_3_year'] = 1 - (0.9403899 ** pi_exp)
    dfpi.loc[dfpi['PI'].notna(),'h_4_year'] = 1 - (0.9273409 ** pi_exp)
    dfpi.loc[dfpi['PI'].notna(),'h_5_year'] = 1 - (0.9160071 ** pi_exp)
    dfpi.loc[dfpi['PI'].notna(),'h_6_year'] = 1 - (0.9086922 ** pi_exp)
    
    
    logger.info('Finished calculating risk scores per year')
    return df, dfpi


# In[ ]:





# In[ ]:





# In[25]:


while True:
    logger.info('Connect to host.....')
    client = Elasticsearch(['http://10.16.31.65:9200/'], request_timeout=600)

    df = read_elas_index(client, index='psychosis_referral_base', doc_type='doc')
    df = ethnicity_coeff(df)
    df = gender_coeff(df)
    df = age_coeff(df)
    df = gender_age_coeff(df)
    df = diag_coeff(df)
    df, df_risk = risk_score(df)

    logger.info('Write patients at risk into Cogstack')
    logger.info('Numbers of risky patients: %s %s' %(df_risk.shape))
    if df_risk.shape[0] > 0:
        df_risk['alerted'] = False
        df_risk['alerted_dttm'] = datetime.now()
        INDEX="psychosis_risk"
        TYPE= "doc"
        doc_index = 'patient_id'
        res = bulk_insert(client, df_risk, INDEX, TYPE, doc_index)
        if res:
            logger.info('Write sucessfully!')
        else:
            logger.info('Write UNsucessfully!!!!')

    logger.info('Update records in source table')
    if df.shape[0] > 0:
        INDEX="psychosis_referral_base"
        TYPE= "doc"
        doc_index = 'patient_id'
        df_update = df[['patient_id','coeff_validated', 'risk_calculated_dttm', 'exist_risk_score']]
        logger.info('Numbers of records that are needed to update: %s %s' %(df_update.shape))
        res = bulk_update(client, df_update, INDEX, TYPE, doc_index)
        if res:
            logger.info('Updated sucessfully!')
        else:
            logger.info('Updated UNsucessfully!!!!')
    time.sleep(12 * 60* 60)


# In[ ]:





# In[ ]:




