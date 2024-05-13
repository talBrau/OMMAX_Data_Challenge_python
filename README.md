# OMMAX_Data_Challenge

This Python package matches ERP (Enterprise Resource Planning) customer data with existing customer data from a broker dataset. It utilizes pandas for data manipulation and fuzzywuzzy for string matching.

## Installation

1. Clone or download the repository to your local machine.

2. Install the required dependencies using pip:

    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. Prepare your datasets:
    - Ensure your ERP customer data and broker data are in the datasets folder

2. Run the main script:
    - Execute the `main.py` script to perform the data matching process:

    ```bash
    python main.py
    ```

3. Results:
    - The matched customer data from the broker dataset will be saved as `data_broker_existing_clients_data.csv` in the `datasets` directory.

## Configuration

- Paths to the input datasets and the result file can be modified in the `main.py` script by changing the values of the `ERP_DATA_PATH`, `BROKER_DATA_PATH`, and `RESULT_DATA_PATH` variables.

## Method
- The method used here to flag the existing customer data is using the postcode to find suitable candidates in the broker dataset, then assigns a score based on string similarity of the name and address of the customer.

## Notes

- Ensure that your input datasets have consistent column names and formats to avoid errors during the preprocessing and matching process.
