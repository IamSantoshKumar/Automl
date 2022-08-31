import streamlit as st
import pandas as pd
import numpy as np
import base64
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import load_diabetes
from sklearn import model_selection
import xgboost
from sklearn.metrics import mean_squared_error
from functools import partial
import optuna

#---------------------------------#
# Page layout
## Page expands to full width
st.set_page_config(page_title='Automated Machine Learning App',
    layout='wide')

#---------------------------------#
st.write("""
# Automated Machine Learning App
**(Regression Edition)**
XGBoost Algorithm (Optuna Backbone).
""")

#---------------------------------#
# Sidebar - Collects user input features into dataframe
st.sidebar.header('Upload your CSV data')
uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
st.sidebar.markdown("""
[Example CSV input file](https://raw.githubusercontent.com/dataprofessor/data/master/delaney_solubility_with_descriptors.csv)
""")

# Sidebar - Specify parameter settings
st.sidebar.header('Set Parameters')
split_size = st.sidebar.slider('Data split ratio (% for Training Set)', 10, 90, 80, 5)

st.sidebar.subheader('Learning Parameters')
parameter_n_estimators = st.sidebar.slider('Number of estimators (n_estimators)', 0, 500, (10,50), 50)
parameter_n_estimators_step = st.sidebar.number_input('Step size for n_estimators', 10)
st.sidebar.write('---')
n_trials = st.sidebar.slider('Number of trials (n_trials)', 0, 500, (10,5), 5)
n_trials = st.sidebar.number_input('Step size for n_trials', 5)
st.sidebar.write('---')
parameter_max_features = st.sidebar.slider('Max features (max_features)', 1, 50, (1,3), 1)
st.sidebar.number_input('Step size for max_features', 1)
st.sidebar.write('---')
parameter_min_samples_split = st.sidebar.slider('Minimum number of samples required to split an internal node (min_samples_split)', 1, 10, 2, 1)
parameter_min_samples_leaf = st.sidebar.slider('Minimum number of samples required to be at a leaf node (min_samples_leaf)', 1, 10, 2, 1)

st.sidebar.subheader('General Parameters')
parameter_random_state = st.sidebar.slider('Seed number (random_state)', 0, 1000, 42, 1)
parameter_criterion = st.sidebar.select_slider('Performance measure (criterion)', options=['mse', 'mae'])
parameter_bootstrap = st.sidebar.select_slider('Bootstrap samples when building trees (bootstrap)', options=[True, False])
#parameter_oob_score = st.sidebar.select_slider('Whether to use out-of-bag samples to estimate the R^2 on unseen data (oob_score)', options=[False, True])
#parameter_n_jobs = st.sidebar.select_slider('Number of jobs to run in parallel (n_jobs)', options=[1, -1])


n_estimators_range = np.arange(parameter_n_estimators[0], parameter_n_estimators[1]+parameter_n_estimators_step, parameter_n_estimators_step)
max_features_range = np.arange(parameter_max_features[0], parameter_max_features[1]+1, 1)
param_grid = dict(max_features=max_features_range, n_estimators=n_estimators_range)

#---------------------------------#
# Main panel

# Displays the dataset
st.subheader('Dataset')



#---------------------------------#
# Model building

def filedownload(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # strings <-> bytes conversions
    href = f'<a href="data:file/csv;base64,{b64}" download="model_performance.csv">Download CSV File</a>'
    return href
	
from sklearn import model_selection
import xgboost
from sklearn.metrics import mean_squared_error
from functools import partial
import optuna

def optimize(trial,x,y):
    n_estimators=trial.suggest_int('n_estimators',100,1500)
    max_depth=trial.suggest_int('max_depth',3,15)
    learning_rate=trial.suggest_loguniform('learning_rate',0.01, 1.0)
    subsample=trial.suggest_float('subsample',0.01,1.0,log=True)
    colsample_bytree=trial.suggest_float('colsample_bytree',0.01,1.0,log=True)
    #criterion=trial.suggest_categorical('criterion',['gini','entropy'])
    NFOLDS = 5
    model=xgboost.XGBRegressor(n_jobs=-1,
             n_estimators=n_estimators,
             max_depth=max_depth,
             learning_rate=learning_rate,
             subsample=subsample,
             colsample_bytree=colsample_bytree ,
             objective='reg:squarederror',
             tree_method = 'gpu_hist'
        )

    accuracies=[]
    skf=model_selection.KFold(n_splits=5, shuffle=True, random_state=42)
    for idx in skf.split(X=x, y=y):
        train_idx, test_idx = idx[0], idx[1]
        xtrain = x[train_idx]
        ytrain = y[train_idx]
        xtest = x[test_idx]
        ytest = y[test_idx]
        model.fit(xtrain,ytrain)
        predicted =model.predict(xtest)
        
        accuracy=np.sqrt(mean_squared_error(ytest,predicted))
        accuracies.append(accuracy)
        
    return np.mean(accuracies)	

def build_model(df):
    X = df.iloc[:,:-1] # Using all column except for the last column as X
    Y = df.iloc[:,-1] # Selecting the last column as Y

    st.markdown('A model is being built to predict the following **Y** variable:')
    st.info(Y.name)

    # Data splitting
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=split_size)
    #X_train.shape, Y_train.shape
    #X_test.shape, Y_test.shape

    optimize_func=partial(optimize, x=X_train.values,y=Y_train.values)
    study = optuna.create_study(direction='minimize')
    study.optimize(optimize_func,n_trials=n_trials)
    predictions = np.zeros((len(X_test)), dtype=np.float64)
    xgb_model = xgboost.XGBRegressor(**study.best_params)
    xgb_model.fit(X_train, Y_train)
    Y_pred_test = xgb_model.predict(X_test)

    st.subheader('Model Performance')

    st.write('Coefficient of determination ($R^2$):')
    st.info( r2_score(Y_test, Y_pred_test) )

    st.write('Error (MSE or MAE):')
    st.info( mean_squared_error(Y_test, Y_pred_test) )

    st.write("The best parameters are %s with a score of %0.2f"
      % (study.best_params, study.best_value))

    st.subheader('Model Parameters')
    st.write(study.best_params)


    st.markdown(filedownload(study.trials_dataframe()), unsafe_allow_html=True)

#---------------------------------#
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write(df)
    build_model(df)
else:
    st.info('Awaiting for CSV file to be uploaded.')
    if st.button('Press to use Example Dataset'):
        diabetes = load_diabetes()
        X = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
        Y = pd.Series(diabetes.target, name='response')
        df = pd.concat( [X,Y], axis=1 )

        st.markdown('The **Diabetes** dataset is used as the example.')
        st.write(df.head(5))

        build_model(df)