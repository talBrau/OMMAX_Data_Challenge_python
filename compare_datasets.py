import pandas as pd
from fuzzywuzzy import fuzz

# Paths to datasets
ERP_DATA_PATH = "datasets/customer_erp_data.csv"
BROKER_DATA_PATH = "datasets/data_broker_dataset.csv"
RESULT_DATA_PATH = "datasets/data_broker_existing_clients_data.csv"

def preprocess_postcode(postcode):
    """
    Preprocesses postcode by removing spaces.

    Args:
        postcode (str): The postcode to preprocess.

    Returns:
        str: Preprocessed postcode.
    """
    return preprocess_name(postcode).replace(" ", "")

def preprocess_name(name):
    """
    Preprocesses name by converting to lowercase and stripping whitespace.

    Args:
        name (str): The name to preprocess.

    Returns:
        str: Preprocessed name.
    """
    return name.lower().strip().replace(",", "")

def preprocess_broker_data(broker_data):
    """
    Preprocesses broker data by normalizing postcode, name, and address.

    Args:
        broker_data (DataFrame): DataFrame containing broker data.

    Returns:
        DataFrame: Preprocessed broker data.
    """
    broker_data["postcode_norm"] = broker_data["postcode"].apply(preprocess_postcode)
    broker_data["name_norm"] = broker_data["name"].apply(preprocess_name)
    broker_data["address_norm"] = broker_data["address"].apply(preprocess_name)
    return broker_data

def preprocess_erp_data(erp_data):
    """
    Preprocesses ERP data by normalizing postcode, name, and address.

    Args:
        erp_data (DataFrame): DataFrame containing ERP data.

    Returns:
        DataFrame: Preprocessed ERP data.
    """
    erp_data["postcode_norm"] = erp_data["postcode"].apply(preprocess_postcode)
    erp_data["name_norm"] = erp_data["name"].apply(preprocess_name)
    erp_data["address_norm"] = erp_data["address"].apply(preprocess_name)
    return erp_data

def pre_process_data():
    """
    Reads ERP and broker data from CSV files and preprocesses them.

    Returns:
        tuple: A tuple containing preprocessed ERP and broker data DataFrames.
    """
    erp_data = pd.read_csv(ERP_DATA_PATH)
    broker_data = pd.read_csv(BROKER_DATA_PATH)

    erp_data = preprocess_erp_data(erp_data)
    broker_data = preprocess_broker_data(broker_data)

    return erp_data, broker_data

def get_candidates_based_on_postcode(broker_data, erp_entry):
    """
    Filters broker data based on the postcode of the ERP entry.

    Args:
        broker_data (DataFrame): DataFrame containing broker data.
        erp_entry (Series): Series representing a single ERP entry.

    Returns:
        DataFrame: Filtered DataFrame of broker data.
    """
    candidates = broker_data[broker_data["postcode_norm"] == erp_entry["postcode_norm"]]
    if candidates.empty:
        candidates = broker_data.copy()
    return candidates

def find_best_candidate(candidates, erp_entry):
    """
    Finds the best candidate broker for a given ERP entry based on name and address similarity.

    Args:
        candidates (DataFrame): DataFrame containing candidate broker data.
        erp_entry (Series): Series representing a single ERP entry.

    Returns:
        DataFrame: DataFrame containing the best candidate broker.
    """
    candidates["name_score"] = candidates["name_norm"].apply(lambda x: fuzz.token_set_ratio(x, erp_entry["name_norm"]))
    candidates["address_score"] = candidates["address_norm"].apply(lambda x: fuzz.token_set_ratio(x, erp_entry["address_norm"]))
    candidates["final_score"] = candidates[['name_score', 'address_score']].mean(axis=1)
    best_candidate = candidates[candidates["final_score"] == candidates["final_score"].max()]
    return best_candidate

def find_best_matching_indices(erp_data, broker_data):
    """
    Finds the best matching broker indices for each ERP entry.

    Args:
        erp_data (DataFrame): DataFrame containing ERP data.
        broker_data (DataFrame): DataFrame containing broker data.

    Returns:
        list: List of indices corresponding to the best matching broker data for each ERP entry.
    """
    broker_link_indices = []
    for i in range(len(erp_data)):
        erp_entry = erp_data.loc[i]
        candidates = get_candidates_based_on_postcode(broker_data, erp_entry)
        best_candidate = find_best_candidate(candidates, erp_entry)
        broker_link_indices.append(best_candidate.index[0])
    return broker_link_indices

def find_existing_customer_data_at_broker(erp_data, broker_data):
    """
    Finds existing customer data at the broker for each ERP entry.

    Args:
        erp_data (DataFrame): DataFrame containing ERP data.
        broker_data (DataFrame): DataFrame containing broker data.

    Returns:
        DataFrame: DataFrame containing existing customer data at the broker.
    """
    broker_link_indices = find_best_matching_indices(erp_data, broker_data)
    broker_data = pd.read_csv(BROKER_DATA_PATH)
    return broker_data.iloc[broker_link_indices]

def main():
    """
    Main function to preprocess data, find existing customers at the broker, and save the results.
    """
    erp_data, broker_data = pre_process_data()
    existing_customers_broker_data = find_existing_customer_data_at_broker(erp_data, broker_data)
    existing_customers_broker_data.to_csv(RESULT_DATA_PATH)

if __name__ == "__main__":
    main()
