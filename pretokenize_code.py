import datetime
import json
import argparse
import pickle
import math
import numpy as np
import datetime
import sys
from collections import Counter

"""

It is better to write all functions into a single spark submit file. 
Otherwise, you have to use --py_files arguments when submitting spark job.
See: https://spark.apache.org/docs/latest/api/python/user_guide/python_packaging.html

For me, I choose the more convenient way - put everything into one file.

Usage: 
spark-submit --master spark://ghjk3695-gu-spark-master-125:7077 --executor-memory 60g pretokenize.py s3a://dataset/nhird-all/all_visits s3a://dataset/nhird-all/all_visits_tokenized vocabulary.json 4096

"""

YEARS = {"_<5y": 5, **{f"_{i - 5}-{i}y": i for i in range(10, 101, 5)}}
YEAR_NAMES = list(YEARS.keys()) + ["_>100y"]
YEAR_BINS = list(YEARS.values())

VISIT_DAYS = {"_<5d": 5, **{f"_{i - 5}-{i}d": i for i in range(10, 101, 5)}}
VISIT_NAMES = list(VISIT_DAYS.keys()) + ["_>100d"]
VISIT_BINS = list(VISIT_DAYS.values())

SEX_SYMBOLS = {"M":"SEX_M", "F":"SEX_F","U":"SEX_U"}
LLM_SYMBOLS = ["PAD", "SEP", "EOS"]

def day_difference(date1,date2):
    date1 = datetime.datetime.strptime(date1,"%Y%m%d")
    if len(date2) == 6:  # Format: yyyymm
        date2 += "01"
    elif len(date2) == 4:  # Format: yyyy
        date2 += "0101"

    date2 = datetime.datetime.strptime(date2, "%Y%m%d")
    return abs((date2-date1).days)

def ordered_day_difference(date1, date2, difference):
    # Compare if date1 < date2 in "difference" days

    date1 = datetime.datetime.strptime(date1,"%Y%m%d")
    if len(date2) == 6:  # Format: yyyymm
        date2 += "01"
    elif len(date2) == 4:  # Format: yyyy
        date2 += "0101"

    date2 = datetime.datetime.strptime(date2, "%Y%m%d")
    
    if date1 < date2:
        if abs((date2-date1).days) >= difference:
            return True

    return False


def calculate_age(birth, visit_date):
    # Extract year and month from the birthday and fee_ym strings
    birth_year = int(birth[:4])
    fee_year = int(visit_date[:4])
    year_diff = fee_year - birth_year
    # The total age is just the year difference, as months are not needed for approximate age
    return year_diff

def standardize_codes(input_list):

    # Helper function to recursively flatten the list
    def flatten(lst):
        for item in lst:
            if isinstance(item, list):
                yield from flatten(item)  # Recursively flatten if the item is a list
            else:
                yield item

    # Flatten the list and filter out None and empty strings, then convert everything to string
    flattened_list = list(flatten(input_list))
    standardized_list = [str(item) for item in flattened_list if item not in [None, ""]]
    
    return standardized_list

def generate_codes(visit):
    icd_diag_codes = visit["acode_icd9_list"]
    icd_op_codes = visit["icd_op_code_list"]
    drug_codes = [do["order_code"] for do in visit["order_info"]]

    return standardize_codes(icd_diag_codes),standardize_codes(icd_op_codes),standardize_codes(drug_codes)
    
def age_to_year_token(age):
    i = np.digitize(age, YEAR_BINS)
    return YEAR_NAMES[i]

def visit_to_day_token(visit_days):
    i = np.digitize(visit_days, VISIT_BINS)
    return VISIT_NAMES[i]

def history_to_ids_sequence(history, vocab):
    visit_ids_sequence = []
    visit_pos_sequence = [0,0]
    
    prev_visit_day = 0
    for i, visit in enumerate(history):
        try:
            cur_visit_day = visit["visit_date"]
        
            icd_diag_codes,icd_op_codes,drug_codes = generate_codes(visit)
            codes = icd_diag_codes + icd_op_codes + drug_codes
            
            #ids = [vocab["code2idx"][code] for code in codes] # this is for nhird we have
            ids = [vocab["code2idx"].get(code, 10216) for code in codes] # this is for FuJen 1M dataset

            
            if i == 0:
                ids = [vocab["code2idx"]["SEP"]]+ids
            else:
                visit_day_interval = day_difference(cur_visit_day,prev_visit_day)
                visit_day_code = visit_to_day_token(visit_day_interval)
                ids = [vocab["code2idx"][visit_day_code]]+ids

            prev_visit_day = cur_visit_day
            
            visit_ids_sequence.extend(ids)
            visit_pos_sequence.extend([i+1]*len(ids))
        except Exception as e:
            print(f"Error processing record and visit: {str(e)} and {visit}")
            visit_ids_sequence.extend([])
            visit_pos_sequence.extend([])
       
    #generate age code:
    age = calculate_age(history[0]["id_birthday"], history[0]["visit_date"])
    age_code = age_to_year_token(age)
    visit_ids_sequence = [vocab["code2idx"][age_code]]+ visit_ids_sequence
    #generate gender code:    
    if "id_sex" not in history[0]:
        gender = history[0]['sex']
    else:
        gender = history[0]['id_sex']
    if gender not in ['M','F']:
        gender = 'U'
    visit_ids_sequence  = [vocab["code2idx"]["SEX_"+gender]]+ visit_ids_sequence
    visit_ids_sequence.append(vocab["code2idx"]["EOS"])  
    visit_pos_sequence.append(visit_pos_sequence[-1]+1)
    
    assert len(visit_ids_sequence) == len(visit_pos_sequence)    
    return visit_ids_sequence,visit_pos_sequence
