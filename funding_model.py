import pandas as pd
from data.sources import Sources
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from scipy.optimize import minimize, LinearConstraint
from tensorflow.keras.models import load_model
import joblib

# This returns the saved model as well as the function used to format the data to get it into model format
def get_funding_model():
    df = get_main_data(["2015","2016","2017","2018","2019"])
    _, _, _, _, df_to_model_format, scaler= get_train_test_data(df)
    
    model = load_model('models/dense_regression_model.h5')
    reg = joblib.load('models/linear_regression_model.pkl')

    return model, reg,df_to_model_format, scaler

# This gets the demo functions that are necessary for our frontend. Uses the 2020 dataset
# as an example of our model on "new" data
def get_demo_functions():
    model, reg, df_to_model_format,scaler = get_funding_model()
    # Helper function to go from model format to the original values
    def inverse_scaling(number, index):
        return number * (scaler.data_max_[index] - scaler.data_min_[index]) + scaler.data_min_[index]
    
    df_2020 = get_main_data()
    school_name_list = df_2020['School name']
    
    # Function that lets you input a school, what they want the graduation rate to be, and what they might want to adjust, and it will output how much more of that feature will be necessary to reach that graduation %. This can be used to help policy makers best determine how much funds / where to use the funds / what features might be most relevant to them.
    def how_much_more(school_name,wanted_grad_rate, col_name = "per_pupil_funding"):

        coefficients = list(reg.coef_)
        intercept = reg.intercept_
        school = df_2020[df_2020['School name'] == school_name]
        t = df_to_model_format(school)[0]
        index_to_exclude = t.columns.get_loc(col_name)
        relevant_coefficients = coefficients[:index_to_exclude] + coefficients[index_to_exclude + 1:]
        inputs = list(np.array(t)[0])
        relevant_inputs = inputs[:index_to_exclude] + inputs[index_to_exclude + 1:]
        necessary_val = (wanted_grad_rate - np.dot(relevant_inputs, relevant_coefficients) - intercept)/coefficients[index_to_exclude]

        return (inverse_scaling(necessary_val, index_to_exclude) - school[col_name]).values[0]
    
    def objective_function_dense(params):
        out = 0
        X_modified = df_2020.copy()
        X_modified['per_pupil_funding'] = params / X_modified['COHORT']
        X_modified,_  = df_to_model_format(X_modified)

        out = out + np.dot(model.predict(X_modified,verbose = 0).flatten() , np.array(df_2020['COHORT']).reshape(-1,1)) / np.sum(df_2020['COHORT'])
        return -1 * out
    
    # Gets our predicted funding distribution given a set amount of funding for our 1 layer NN model using numerical optimization.
    def getFundingDistributionDense(max_funding = 50000000):
        initial_guess = df_2020['per_pupil_funding'] * df_2020['COHORT'] * max_funding / np.sum(df_2020['per_pupil_funding'] * df_2020['COHORT'])

        linear_constraint = LinearConstraint(np.ones_like(initial_guess), lb=max_funding, ub=max_funding)


        # Perform optimization
        result = minimize(
            objective_function_dense,
            initial_guess,
            constraints=[linear_constraint],
            method='SLSQP',  # You can choose different optimization methods
        )

        optimized_params = result.x


        funding_df_dense = pd.concat([df_2020['School name'].reset_index(drop = True), pd.DataFrame(optimized_params, columns = ['Funding']).reset_index(drop = True), pd.DataFrame(optimized_params/df_2020['COHORT']).reset_index(drop = True)],axis = 1)
        funding_df_dense = funding_df_dense.rename(columns = {"COHORT": "Per Pupil Funding"})
        print("ex. 5 schools from df")
        print(funding_df_dense.head())
        print("Objective Function Value:", -result.fun)
        return funding_df_dense



    def objective_function_lr(params):
        # higher values on penalty enforces that the funding is more equal
        # make this really high and they will all be equal
        penalty_weight = 110000
        out = 0
        X_modified = df_2020.copy()
        X_modified['per_pupil_funding'] = params / X_modified['COHORT']
        X_modified,_  = df_to_model_format(X_modified)
        mean = np.mean(X_modified['per_pupil_funding'])
        penalty = -1 * np.sum(np.power((X_modified['per_pupil_funding']-mean),2))/len(X_modified) * penalty_weight
        out = out + np.sum(reg.predict(X_modified).flatten())/len(X_modified) + penalty
        return -1 * out

    # Gets our predicted funding distribution given a set amount of funding for our 1 layer NN model using numerical optimization.
    def getFundingDistributionLr(max_funding = 50000000):
        initial_guess = df_2020['per_pupil_funding'] * df_2020['COHORT'] * max_funding / np.sum(df_2020['per_pupil_funding'] * df_2020['COHORT'])
        bounds = [(0, max_funding)] * len(df_2020)
        linear_constraint = LinearConstraint(np.ones_like(initial_guess), lb=0, ub=max_funding)

        # Perform optimization
        result = minimize(
            objective_function_lr,
            initial_guess,
            constraints=[linear_constraint],
            bounds = bounds,
            method='SLSQP',  # You can choose different optimization methods
        )

        optimized_params = result.x

        # the results are to make all of the funding basically the same
        # scaling the variables in preprocessing might get the differentiation
        # but effect size for funding is small so idk if we would get good results even if we do that

        funding_df_lr = pd.concat([df_2020['School name'].reset_index(drop = True), pd.DataFrame(optimized_params, columns = ['Funding']).reset_index(drop = True), pd.DataFrame(optimized_params/df_2020['COHORT']).reset_index(drop = True)],axis = 1)
        funding_df_lr = funding_df_lr.rename(columns = {"COHORT": "Per Pupil Funding"})

        print("ex. 5 schools from df")
        print(funding_df_lr.head())
        print("Objective Function Value:", -result.fun)
        return funding_df_lr
    

    return how_much_more, getFundingDistributionLr, school_name_list


#####################################
# Everything below here are just helper functions to get the model data and scalers
# Function that converts graduation rates into a single rate
def rate_to_number(rate):
    chars = ["GE","LT","LE"]
    if "-" in rate:
        return int(rate.split("-")[0])
    elif any(i in rate for i in chars):
        for i in chars:
            rate = rate.replace(i, "")
        return int(rate)
    elif "PS" in rate:
        return None
    return int(rate)

# Function to identify columns to conver to numeric
def represents_float(s):
    if s is None:
        return True
    try: 
        float(s)
    except ValueError:
        return False
    else:
        return True

# This gets the main dataset for a specific subset of years. It fetches the data from the MySQL database and does some
# parsing and data conversions to get the relevant columns and data for our model.
def get_main_data(years = ["2020"]):
    data = Sources("localhost", "aipi510_project", "root", "password") 

    query = f"""
    SELECT * FROM AIPI510_PROJECT.SSLE AS SSLE
    INNER JOIN AIPI510_PROJECT.SAIPE AS SAIPE
    ON SAIPE.`LEA ID` = SSLE.`LEA ID`
    
    INNER JOIN aipi510_project.graduation_rate_category AS GRAD_CAT
    ON GRAD_CAT.`NCESSCH` = SSLE.`NCES ID` AND GRAD_CAT.`CATEGORY` = 'ALL' AND GRAD_CAT.`year` = SAIPE.`year`
    
    INNER JOIN AIPI510_PROJECT.NCES_BY_YEAR AS NCES
    ON NCES.`school_id_nces_assigned` = SSLE.`NCES ID` AND NCES.`year` = SAIPE.`year`
    
    INNER JOIN AIPI510_PROJECT.title_i_max_funding AS funding
    on funding.`LEA ID` = SSLE.`LEA ID` AND funding.`year` = SAIPE.`year`
    
    INNER JOIN AIPI510_PROJECT.per_pupil_data as per_pupil_data
    on per_pupil_data.`site_name` = SSLE.`School name` AND per_pupil_data.`year` = SAIPE.`year`
    WHERE SSLE.`Title I eligibility`="Title I" AND SAIPE.`year` in ({','.join(years)}) AND per_pupil_data.`reporting_category` = "Per Pupil Expenses"

    """
    df = pd.read_sql_query(query, data.engine)
    
    not_relevant_columns = "index ,NCES ID, State code, State abbreviation, LEA name, Grades offered lowest, Grades offered highest, District Name, State Code, SCHOOL_YEAR, STNAM, ST_LEAID, NCESSCH, ST_SCHID, SCHNAM, CATEGORY, DATE_CUR, school_name, state_name, Title I eligibility, School operational status,school_wide_title_i, title_i_eligible_school, title_i_school_status,grade_10_offered,updated_status,charter_school, school_type,school_system_code,school_system_name,site_code,reporting_category,reporting_subcategory,Personnel salaries at school level - total,Personnel salaries at school level - instructional staff only,Personnel salaries at school level - teachers only,Non-personnel expenditures at school level,Student enrollment,Per-pupil expenditure - total personnel salaries,Number free and reduced price lunch students"
    not_relevant_columns = not_relevant_columns.split(",")
    not_relevant_columns = [i.strip() for i in not_relevant_columns]

    df = df.T.drop_duplicates().T
    df = df.loc[:,~df.columns.duplicated()].copy()
    df = df.drop(columns = not_relevant_columns)

    df = df.map(lambda x: x if x != "–" else 0)

    df['Charter status'] = df['Charter status'].replace({'No': 0, 'Yes': 1})
    df['magnet_school'] = df['magnet_school'].replace({'2-No': 0, '1-Yes': 1})

    df['RATE'] = df['RATE'].apply(rate_to_number)
    df = df[df['RATE'].notna()]
    df['RATE'] = df['RATE'] / 100
    
    numeric_filter = df.applymap(represents_float).sum() == len(df)
    numeric_cols = list(numeric_filter[numeric_filter].index)
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric)
    
    df = df.rename(columns = {"amount_or_percent": "per_pupil_funding"})
    variables_to_scale_str = "grade_9_students_american_indian_alaska_native_female,grade_9_students_american_indian_alaska_native_male,grade_9_students_asian_or_asian_pacific_islander_female,grade_9_students_asian_or_asian_pacific_islander_male,total_students_all_grades_excludes_ae,grade_9_students_black_or_african_american_female,grade_9_students_black_or_african_american_male,grade_9_students_hispanic_female,grade_9_students_hispanic_male,grade_9_students_nat_pacific_isl_female,grade_9_students_nat_pacific_isl_male,grade_9_students_two_or_more_races_male,grade_9_students_white_female,grade_9_students_white_male,grade_10_students_american_indian_alaska_native_female,grade_10_students_american_indian_alaska_native_male,grade_10_students_asian_or_asian_pacific_islander_female,grade_10_students_asian_or_asian_pacific_islander_male,grade_9_students_two_or_more_races_female,grade_10_students_black_or_african_american_female,grade_10_students_black_or_african_american_male,grade_10_students_hispanic_female,grade_10_students_hispanic_male,grade_10_students_white_male,grade_10_students_nat_pacific_isl_female,grade_10_students_nat_pacific_isl_male,grade_10_students_two_or_more_races_female,grade_10_students_two_or_more_races_male,grade_10_students_white_female,grade_11_students_american_indian_alaska_native_male,grade_11_students_american_indian_alaska_native_female,grade_11_students_asian_or_asian_pacific_islander_female,grade_11_students_asian_or_asian_pacific_islander_male,grade_11_students_black_or_african_american_male,grade_11_students_hispanic_female,grade_11_students_hispanic_male,grade_11_students_black_or_african_american_female,grade_11_students_nat_pacific_isl_female,grade_11_students_nat_pacific_isl_male,grade_11_students_two_or_more_races_male,grade_11_students_white_female,grade_11_students_white_male,grade_11_students_two_or_more_races_female,grade_12_students_american_indian_alaska_native_female,grade_12_students_american_indian_alaska_native_male,grade_12_students_asian_or_asian_pacific_islander_female,grade_12_students_asian_or_asian_pacific_islander_male,grade_12_students_hispanic_male,grade_12_students_black_or_african_american_female,grade_12_students_black_or_african_american_male,grade_12_students_hispanic_female,grade_12_students_nat_pacific_isl_male,grade_12_students_white_female,grade_12_students_white_male,free_lunch_eligible,grade_12_students_nat_pacific_isl_female,grade_12_students_two_or_more_races_female,grade_12_students_two_or_more_races_male,free_and_reduced_lunch_students,reduced_price_lunch_eligible_students"
    variables_to_scale = variables_to_scale_str.split(",")
    variables_to_scale = [i.strip() for i in variables_to_scale]
    df[variables_to_scale] = df[variables_to_scale].div(df['COHORT'], axis=0)

    return df

# This is just to rearrange columns so that we can use them more easily
def move_column_inplace(df, col, pos):
    col = df.pop(col)
    df.insert(pos, col.name, col)
    
# Helper function to one hot encode based on our encoder   
def one_hot_encode(df, onehot_enc, onehot_cols):
    onehot_encoded = onehot_enc.transform(df[onehot_cols]).toarray()
    colnames = list(onehot_enc.get_feature_names_out(input_features = onehot_cols))
    enc_df = pd.DataFrame(onehot_encoded, columns = colnames, index = df.index)
    return pd.concat([df,enc_df], axis = 1).drop(columns = onehot_cols)

# Split the data and apply one hot encoding and MinMaxScaling. This function also returns a helper function 
# (df_to_model_format) that allows us to convert new data (not just training/test) into the right format 
# and also the scaler to help us retrieve the original values. 
def get_train_test_data(df):
    onehot_cols = ['School level', 'School type', 'locale']
    drop_cols =  ['LEA ID', 'Year', 'School ID', 'School name']
    df = df.drop(columns = drop_cols)
    X = df.drop(columns = ['RATE'])
    y = df['RATE']
    move_column_inplace(X, 'per_pupil_funding', 0)
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2, random_state = 121)

    onehot_enc = OneHotEncoder(handle_unknown ='ignore' ).fit(X_train[onehot_cols])
    X_train = one_hot_encode(X_train, onehot_enc,onehot_cols)

    X_test = one_hot_encode(X_test, onehot_enc,onehot_cols)
    sc = MinMaxScaler().set_output(transform="pandas")
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    
    def df_to_model_format(input_df):
        input_df = input_df.drop(columns = drop_cols)
        X = input_df.drop(columns = ['RATE'])
        y = input_df['RATE']
        X = one_hot_encode(X, onehot_enc,onehot_cols)
        move_column_inplace(X, 'per_pupil_funding', 0)
        X = sc.transform(X)
        return X, y
    return X_train, X_test, y_train, y_test, df_to_model_format, sc

