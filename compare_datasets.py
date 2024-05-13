import pandas as pd
from fuzzywuzzy import fuzz

ERP_DATA_PATH = "datasets/customer_erp_data.csv"
BROKER_DATA_PATH = "datasets/data_broker_dataset.csv"
RESULT_DATA_PATH = "datasets/data_broker_existing_clients_data.csv"

def pre_process_postcode(postcode):
    return pre_process_name(postcode).replace(" ","")

def pre_process_name(name):
    return name.lower().strip().replace(",","")

def preprocess_broker_data(broker_data):
    broker_data["postcode_norm"] = broker_data["postcode"].apply(pre_process_postcode)
    broker_data["name_norm"] = broker_data["name"].apply(pre_process_name)
    broker_data["address_norm"] = broker_data["address"].apply(pre_process_name)
    return broker_data

def preprocess_erp_data(erp_data):
    erp_data["postcode_norm"] = erp_data["postcode"].apply(pre_process_postcode)
    erp_data["name_norm"] = erp_data["name"].apply(pre_process_name)
    erp_data["address_norm"] = erp_data["address"].apply(pre_process_name)
    return erp_data

def pre_process_data():
    erp_data = pd.read_csv(ERP_DATA_PATH)
    broker_data = pd.read_csv(BROKER_DATA_PATH)

    erp_data = preprocess_erp_data(erp_data)
    broker_data = preprocess_broker_data(broker_data)

    return erp_data, broker_data    

def get_candidates_based_on_postcode(broker_data, erp_entry):
    candidates = broker_data[broker_data["postcode_norm"] == erp_entry["postcode_norm"]]
    if candidates.empty:
        candidates = broker_data.copy()
    return candidates

def find_best_candidate(candidates, erp_entry):
    candidates["name_score"] = candidates["name_norm"].apply(lambda x: fuzz.token_set_ratio(x, erp_entry["name_norm"]))
    candidates["address_score"] = candidates["address_norm"].apply(lambda x: fuzz.token_set_ratio(x, erp_entry["address_norm"]))
    candidates["final_score"] = candidates[['name_score', 'address_score']].mean(axis=1)
    best_candidate = candidates[candidates["final_score"] == candidates["final_score"].max()]
    return best_candidate

def find_best_matching_indices(erp_data, broker_data):
    broker_link_indices = []
    for i in range (len(erp_data)):
        erp_entry = erp_data.loc[i]
        candidates = get_candidates_based_on_postcode(broker_data, erp_entry)
        best_candidate = find_best_candidate(candidates, erp_entry)
        broker_link_indices.append(best_candidate.index[0])
    return broker_link_indices

def find_existing_customer_data_at_broker(erp_data, broker_data):
    broker_link_indices = find_best_matching_indices(erp_data, broker_data)
    broker_data = pd.read_csv(BROKER_DATA_PATH)
    return broker_data.iloc[broker_link_indices]

def main():
    erp_data, broker_data = pre_process_data()
    existing_customers_broker_data = find_existing_customer_data_at_broker(erp_data, broker_data)
    existing_customers_broker_data.to_csv(RESULT_DATA_PATH)

main()