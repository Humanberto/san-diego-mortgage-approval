"""Send one sample applicant to the running service and print the answer.

Start the service first (locally or in Docker), then in a second terminal:
    uv run python test_predict.py
"""
import json
import sys
from urllib.request import Request, urlopen

url = sys.argv[1] if len(sys.argv) > 1 else 'http://localhost:9696/predict'

# A real 2025 San Diego application from the test set (it was approved).
# FHA purchase of a $685,000 home, 96.5% loan-to-value, $138,000 income, DTI 49%.
applicant = {
    "lei": "549300REENMLD2RN3W58",
    "conforming_loan_limit": "C",
    "derived_loan_product_type": "FHA:First Lien",
    "derived_dwelling_category": "Single Family (1-4 Units):Site-Built",
    "loan_type": "fha",
    "loan_purpose": "purchase",
    "lien_status": "first",
    "reverse_mortgage": "2",
    "open-end_line_of_credit": "2",
    "business_or_commercial_purpose": "2",
    "negative_amortization": "2",
    "interest_only_payment": "2",
    "balloon_payment": "2",
    "other_nonamortizing_features": "2",
    "construction_method": "1",
    "occupancy_type": "primary",
    "manufactured_home_secured_property_type": "3",
    "manufactured_home_land_property_interest": "5",
    "total_units": "1",
    "debt_to_income_ratio": "49",
    "applicant_credit_score_type": "1",
    "co-applicant_credit_score_type": "9",
    "submission_of_application": "2",
    "aus-1": "3",
    "loan_amount": 675000,
    "loan_to_value_ratio": 96.5,
    "loan_term": 360,
    "property_value": 685000,
    "income": 138,
    "is_adjustable": 0,
    "tract_population": 5346,
    "tract_to_msa_income_percentage": 101,
    "tract_owner_occupied_units": 1634,
    "tract_one_to_four_family_homes": 1879,
    "tract_median_age_of_housing_units": 42,
}

request = Request(url, data=json.dumps(applicant).encode('utf-8'),
                  headers={'Content-Type': 'application/json'}, method='POST')
with urlopen(request) as response:
    print('as sent:          ', json.load(response))

# Same person, but with debt-to-income above 60%.
applicant['debt_to_income_ratio'] = '>60%'
request = Request(url, data=json.dumps(applicant).encode('utf-8'),
                  headers={'Content-Type': 'application/json'}, method='POST')
with urlopen(request) as response:
    print('with DTI above 60%:', json.load(response))
