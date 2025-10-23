import json
from tqdm import tqdm
from joblib import Parallel, delayed
import argparse


def args_parser():
    parser = argparse.ArgumentParser(description="Convert EHRShot patients to NHIRD format")
    parser.add_argument("--this_version", type=int, required=True, help="This version of the mapping")
    parser.add_argument("--mapping_version", type=int, default=7, help="Mapping version to use")
    parser.add_argument("--icd_score_threshold", type=float, default=0.85, help="Score threshold for ICD codes")
    parser.add_argument("--order_score_threshold", type=float, default=0.85, help="Score threshold for order codes")
    parser.add_argument("--n_jobs", type=int, default=8, help="Number of parallel jobs")
    
    return parser.parse_args()


def get_gender(patient):
    for event in patient["events"]:
        if "Gender" in event["code"]:
            return event["code"].replace("Gender/", "")

def group_events(patient):
    visits = {}

    for event in patient["events"]:
        event_time = event["time"].split(" ")[0].replace("-", "")
        visits.setdefault(event_time, []).append(event)
    visits = dict(sorted(visits.items(), key=lambda x:x[0]))

    return visits

def mapping_to_nhird(codes, mapping):
    nhird_codes = []

    for code in codes:
        if code in mapping:
            if len(mapping[code]):
                nhird_codes.append(mapping[code])

    return nhird_codes

def get_valid_nhird_icd_op_codes(icd10pcs_mapping):
    valid_codes = []
    for entry in icd10pcs_mapping.values():
        icd9_code = entry.get("icd9_op_code")
        # print(type(icd9_code))
        # if isinstance(icd9_code, float):
        #     icd9_code = str(icd9_code)
        if icd9_code:
            valid_codes.append(icd9_code)
    return valid_codes

def create_nhird_order_list(order_codes, order_key_name):
    order_info = []
    for order_code in order_codes:
        order_info.append({
            order_key_name: order_code,
        })
    return order_info

def get_order_code_for_visit_occurrence(visit_events):
    for event in visit_events:
        if event["code"] in ["Domain/OMOP generated", "Medicare Specialty/A0"]:
            return ["00156A"]
    return []

def create_visit(visit_date, visit_events, pid, birth, gender, mapping, mapping_version=0, icd_score_threshold=0.7, order_score_threshold=0.7):
    snomed_codes = [e["code"].replace("SNOMED/", "") for e in visit_events if "SNOMED" in e["code"]]
    icdo3_codes = [e["code"].replace("ICDO3/", "").split("-")[1] if len(e["code"].replace("ICDO3/", "").split("-")) > 1 else e["code"].replace("ICDO3/", "").split("-")[0] for e in visit_events if "ICDO3" in e["code"]]
    lonic_codes = [e["code"].replace("LONIC/", "") for e in visit_events if "LONIC" in e["code"]]
    icd10pcs_codes = [e["code"].replace("ICD10PCS/", "") for e in visit_events if "ICD10PCS" in e["code"]]
    icd9proc_codes_a = [e["code"].replace("ICD9Proc/", "") for e in visit_events if "ICD9Proc" in e["code"]]
    icd9proc_codes_b = [e["code"].replace("ICD9Proc/", "").replace(".", "") for e in visit_events if "ICD9Proc" in e["code"]]
    cpt4_codes = [e["code"].replace("CPT4/", "") for e in visit_events if "CPT4" in e["code"]]
    hcpcs_codes = [e["code"].replace("HCPCS/", "") for e in visit_events if "HCPCS" in e["code"]]
    rxnorm_codes = [e["code"].replace("RxNorm/", "") for e in visit_events if "RxNorm" in e["code"]]
    event_time = visit_events[-1]["time"]
    """
    v1 - snomed
    v2 - snomed, rxnorm
    v3 - snomed, icd10pcs, rxnorm
    v4 - snomed, icd10pcs, rxnorm, icdo3
    v5 - snomed, icd10pcs, rxnorm, icdo3, cpt4
    v6 - snomed, icd10pcs, rxnorm, icdo3, cpt4, lonic
    v7 - snomed, icd10pcs, rxnorm, icdo3, cpt4, lonic, hcpcs
    """
    
    nhird_codes = []
    if mapping_version >= 1:
        nhird_codes += mapping_to_nhird(snomed_codes, mapping["snomed"])
    if mapping_version >= 2:
        nhird_codes += mapping_to_nhird(rxnorm_codes, mapping["rxnorm"])
    if mapping_version >= 3:
        nhird_codes += mapping_to_nhird(icd10pcs_codes, mapping["icd10pcs"])
    if mapping_version >= 4:
        nhird_codes += mapping_to_nhird(icdo3_codes, mapping["icdo3"])
    if mapping_version >= 5:
        nhird_codes += mapping_to_nhird(cpt4_codes, mapping["cpt4"])
    if mapping_version >= 6:
        nhird_codes += mapping_to_nhird(lonic_codes, mapping["lonic"])
    if mapping_version >= 7:        
        nhird_codes += mapping_to_nhird(hcpcs_codes, mapping["hcpcs"])
    
    icd9_codes = list(set([code["icd9_code"].replace(".", "") for code in nhird_codes if "icd9_code" in code and float(code["diag_score"]) > icd_score_threshold]))    
    valid_op_codes = get_valid_nhird_icd_op_codes(mapping["icd10pcs"])
    # print([code["icd9_op_code"] for code in nhird_codes if "icd9_op_code" in code and float(code["op_score"]) > icd_score_threshold])
    icd9_op_codes = [str(code["icd9_op_code"]).replace(".", "") for code in nhird_codes if "icd9_op_code" in code and float(code["op_score"]) > icd_score_threshold]
    icd9_op_codes += [code.replace(".", "") for code in icd9proc_codes_a if code in valid_op_codes]
    icd9_op_codes += [code.replace(".", "") for code in icd9proc_codes_b if code in valid_op_codes]
    icd9_op_codes = list(set(icd9_op_codes)) 
    order_codes = list(set([code["order_code"] for code in nhird_codes if "order_code" in code if float(code["order_score"]) > order_score_threshold]))
    visit_codes = get_order_code_for_visit_occurrence(visit_events)


    diag_scores = [float(c["diag_score"]) for c in nhird_codes if "diag_score" in c and float(c["diag_score"]) >= icd_score_threshold]
    op_scores = [float(c["op_score"]) for c in nhird_codes if "op_score" in c and float(c["op_score"]) >= icd_score_threshold]
    order_scores = [float(c["order_score"]) for c in nhird_codes if "order_score" in c and float(c["order_score"]) >= order_score_threshold]
    all_scores = diag_scores + op_scores + order_scores
    avg_score_all = sum(all_scores) / len(all_scores) if all_scores else None

    # --- Construct final visit dict ---
    visit = {
        "id": pid,
        "id_birthday": birth,
        "id_sex": gender,
        "acode_icd9_list": icd9_codes,
        "icd_op_code_list": icd9_op_codes,
        "order_info": create_nhird_order_list(order_codes + visit_codes, "order_code"),
        "func_date": visit_date,
        "visit_date": visit_date,
        "visit_type": "dd",
        "ehrshot_event_time": event_time,

        # new meta fields
        "avg_score_all_codes": avg_score_all,
        "n_diag_codes": len(diag_scores),
        "n_op_codes": len(op_scores),
        "n_order_codes": len(order_scores),
        "n_total_codes": len(all_scores),
        "sum_score_diag": sum(diag_scores),
        "sum_score_op": sum(op_scores),
        "sum_score_order": sum(order_scores),
    }

    return visit



    # visit = {
    #     "id": pid,
    #     "id_birthday": birth,
    #     "id_sex": gender,
    #     "acode_icd9_list": icd9_codes,
    #     "icd_op_code_list": icd9_op_codes,
    #     "order_info": create_nhird_order_list(order_codes + visit_codes, "order_code"),
    #     "func_date": visit_date,
    #     "visit_date": visit_date,
    #     "visit_type": "dd",
    #     "ehrshot_event_time": event_time
    # }

    # return visit
    
def create_nhird_patient(ehrshot_patient, ehrshot_patient_births, mapping, mapping_version=0, icd_score_threshold=0.7, order_score_threshold=0.7): 
    patient_id = str(ehrshot_patient["pid"])
    id_birthday = ehrshot_patient_births[patient_id].replace("-", "")[:6]
    id_sex = get_gender(ehrshot_patient)
    grouped_visits = group_events(ehrshot_patient)
    
    history = []
    time_index = {}
    index = 0
    for visit_date, visit_events in grouped_visits.items():
        visit = create_visit(visit_date, visit_events, patient_id, id_birthday, id_sex, mapping, mapping_version, icd_score_threshold, order_score_threshold)
        time_index[visit["ehrshot_event_time"]] = index
        history.append(visit)
        index += 1
    
    nhird_patient = {
        "patient_id": patient_id,
        "history": history,
        "time_index": time_index
    }

    return nhird_patient 

def file_io(filename, patients, mapping, mapping_version=0, icd_score_threshold=0.7, order_score_threshold=0.7):
    with open(filename, "a") as f:
    #with open(f"ehrshot_in_nhird_patients_v{mapping_version}.json", "a") as f:
        for ehrshot_patient in tqdm(patients, desc="Processing patients"):
            # print(ehrshot_patient)
            nhird_patient = create_nhird_patient(ehrshot_patient, ehrshot_patient_births, mapping, mapping_version, icd_score_threshold, order_score_threshold)
            f.write(json.dumps(nhird_patient) + "\n") 

def patient_batches(patients, n_jobs):
    batch_size = len(patients) // n_jobs
    batches = [patients[i:i + batch_size] for i in range(0, len(patients), batch_size)]
    if len(patients) % n_jobs != 0:
        batches.append(patients[batch_size * n_jobs:])
    return batches


if __name__ == "__main__":
    args = args_parser()
    mapping_version = args.mapping_version
    mapping = {
        "snomed": json.load(open("mapping/snomed_to_nhird_code.json")),
        "icdo3": json.load(open("mapping/icdo3_to_nhird_code.json")),
        "lonic": json.load(open("mapping/lonic_to_nhird_code.json")),
        "icd10pcs": json.load(open("mapping/icd10pcs_to_nhird_code.json")),
        "cpt4": json.load(open("mapping/cpt_to_nhird_code.json")),
        "rxnorm": json.load(open("mapping/rxnorm_to_nhird_code.json")),
        "hcpcs": json.load(open("mapping/hcpcs_to_nhird_code.json"))
    }  

    ehrshot_patients = json.load(open("ehrshot_patient.json"))
    ehrshot_patient_births = json.load(open("ehrshot_patient_birth.json"))

    Parallel(n_jobs=args.n_jobs)(delayed(file_io)(f"ehrshot_in_nhird_patients_v{args.this_version}.json", patients, mapping, args.mapping_version, args.icd_score_threshold, args.order_score_threshold) for patients in patient_batches(ehrshot_patients, args.n_jobs))
