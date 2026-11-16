"""Train the San Diego mortgage approval model and save it to a file.

Run from the project folder:
    uv run python train.py
    uv run python train.py --data hmda_2026_san_diego.csv --year 2026   (next year's data)
"""
import argparse
import pickle

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

# Columns the model must not see (see the notebook for why).
LEAK = ['action_taken', 'purchaser_type', 'interest_rate', 'rate_spread', 'hoepa_status',
        'total_loan_costs', 'total_points_and_fees', 'origination_charges',
        'discount_points', 'lender_credits',
        'denial_reason-1', 'denial_reason-2', 'denial_reason-3', 'denial_reason-4',
        'initially_payable_to_institution', 'preapproval']
CONSTANT = ['activity_year', 'state_code', 'county_code', 'derived_msa-md', 'census_tract',
            'ffiec_msa_md_median_family_income']
MOSTLY_EMPTY = ['prepayment_penalty_term', 'multifamily_affordable_units',
                'aus-2', 'aus-3', 'aus-4', 'aus-5']

NUMERICAL = ['loan_amount', 'loan_to_value_ratio', 'loan_term', 'property_value', 'income',
             'is_adjustable', 'tract_population', 'tract_to_msa_income_percentage',
             'tract_owner_occupied_units', 'tract_one_to_four_family_homes',
             'tract_median_age_of_housing_units']
LOG_COLUMNS = ['loan_amount', 'property_value', 'income']

# The settings you picked in friday_trees.ipynb. Change them here if you picked others.
MODEL_PARAMS = dict(n_estimators=200, min_samples_leaf=5, random_state=1, n_jobs=-1)


def load_and_clean(path):
    """Read the HMDA file and apply every cleaning step from the notebook."""
    df = pd.read_csv(path, low_memory=False)
    df['approved'] = (df.action_taken == 1).astype(int)

    protected = [c for c in df.columns
                 if c.startswith(('derived_', 'applicant_', 'co-applicant_'))
                 and any(w in c for w in ['ethnicity', 'race', 'sex', 'age'])]
    protected.append('tract_minority_population_percent')
    df = df.drop(columns=LEAK + CONSTANT + MOSTLY_EMPTY + protected)

    df['is_adjustable'] = pd.to_numeric(df.intro_rate_period, errors='coerce').notnull().astype(int)
    df = df.drop(columns=['intro_rate_period'])

    for col in ['loan_to_value_ratio', 'loan_term', 'property_value']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df['income'] = df.income.clip(lower=0)
    df.loc[df.loan_to_value_ratio > 200, 'loan_to_value_ratio'] = np.nan

    df['loan_type'] = df.loan_type.map({1: 'conventional', 2: 'fha', 3: 'va', 4: 'usda'})
    df['loan_purpose'] = df.loan_purpose.map({1: 'purchase', 2: 'improvement', 31: 'refinance',
                                              32: 'cash_out_refi', 4: 'other', 5: 'na'})
    df['occupancy_type'] = df.occupancy_type.map({1: 'primary', 2: 'second_home', 3: 'investment'})
    df['lien_status'] = df.lien_status.map({1: 'first', 2: 'second'})

    categorical = [c for c in df.columns if c not in NUMERICAL + ['approved']]
    for col in categorical:
        df[col] = df[col].astype(str).replace('nan', 'missing')
    return df, categorical


def prepare(df, categorical, medians):
    """Fill blanks with the training medians, log the money columns, return dictionaries."""
    df = df.copy()
    df[NUMERICAL] = df[NUMERICAL].fillna(medians)
    for col in LOG_COLUMNS:
        df[col] = np.log1p(df[col])
    return df[categorical + NUMERICAL].to_dict(orient='records')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='hmda_2025_san_diego.csv')
    parser.add_argument('--year', default='2025')
    args = parser.parse_args()

    df, categorical = load_and_clean(args.data)
    df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)
    y_full_train = df_full_train.approved.values
    y_test = df_test.approved.values

    medians = df_full_train[NUMERICAL].median()
    dv = DictVectorizer(sparse=False)
    X_full_train = dv.fit_transform(prepare(df_full_train, categorical, medians))
    X_test = dv.transform(prepare(df_test, categorical, medians))

    model = RandomForestClassifier(**MODEL_PARAMS)
    model.fit(X_full_train, y_full_train)

    auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    print(f'test AUC: {auc:.3f}')

    bundle = {
        'dv': dv,
        'model': model,
        'medians': medians.to_dict(),
        'categorical': categorical,
        'numerical': NUMERICAL,
        'log_columns': LOG_COLUMNS,
        'year': args.year,
        'test_auc': round(auc, 3),
    }
    output = f'model_{args.year}.bin'
    with open(output, 'wb') as f_out:
        pickle.dump(bundle, f_out)
    print(f'saved {output}')


if __name__ == '__main__':
    main()
