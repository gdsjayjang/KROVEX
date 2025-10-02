import numpy as np
import pandas as pd
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.config import SET_SEED, SEED, DATASET_PATH
from utils import MolecularFeatureExtractor

SET_SEED()

df = pd.read_csv(DATASET_PATH + '.csv')
smiles_list = df['smiles'].tolist()
target = df.iloc[:,-1]

print('Processing descriptor selection...')
extractor = MolecularFeatureExtractor()
df_all_features = extractor.extract_molecular_features(smiles_list)

df_all_features['target'] = target


num_all_features = df_all_features.shape[1] - 1 
df_all_features[df_all_features.isna().any(axis = 1)]
df_removed_features = df_all_features.dropna()
num_removed_features = df_removed_features.shape[1] - 1
unique_columns = list(df_removed_features.loc[:, df_removed_features.nunique() == 1].columns)
df_removed_features = df_removed_features.drop(columns = unique_columns).copy()
num_removed_features = df_removed_features.shape[1] - 1
low_variances = sorted(df_removed_features.var())
columns_low_variances = []
for i in low_variances:
    if i < 0.001:
        column = df_removed_features.loc[:, df_removed_features.var() == i].columns
        columns_low_variances.append(column)
columns_low_variances = [item for index in columns_low_variances for item in index]
columns_low_variances = list(set(columns_low_variances))
df_removed_features = df_removed_features.drop(columns = columns_low_variances).copy()
num_removed_features = df_removed_features.shape[1] - 1


# ISIS
print('Processing ISIS...')
from isis import ISIS
screening = ISIS(df_removed_features)
selected_feats = screening.fit()
df_screening = screening.transform()

# Elastic Net
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split

X_ISIS = df_screening.drop(columns = ['target'])
y_ISIS = df_screening['target']

X_train, X_test, y_train, y_test = train_test_split(X_ISIS, y_ISIS, test_size = 0.2, random_state = SEED)

print('Processing Elastic Net...')
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(X_train)

X_train_scaling = scaler.transform(X_train)
X_test_scaling = scaler.transform(X_test)

iter = 5000
en = ElasticNet(max_iter = iter)

param_grid = {
    'alpha': np.linspace(0.01, 1.0, 300),
    'l1_ratio': np.linspace(0.1, 0.9, 30)
}
kfold = KFold(n_splits = 5, shuffle = True, random_state = SEED)

grid_search = GridSearchCV(
    estimator = en,
    param_grid = param_grid,
    scoring = 'neg_mean_squared_error',
    cv = kfold,
    verbose = 0,
    n_jobs = -1
)
grid_search.fit(X_train_scaling, y_train)
best_params = grid_search.best_params_
best_en = ElasticNet(
    alpha = best_params['alpha'],
    l1_ratio = best_params['l1_ratio'],
    max_iter = iter,
    fit_intercept=True
)

best_en.fit(X_train_scaling, y_train)

coefficients = best_en.coef_
coefficients.size

selected_features_elastic = list(X_train.loc[:, best_en.coef_ != 0].columns)

X_train_None0 = X_train.drop(columns=X_train.columns[best_en.coef_ == 0])
coefficients_None0 = coefficients[coefficients!=0]

df_final_selected_features = pd.DataFrame({'Feature' : X_train_None0.columns,
                                       'Coefficient' : coefficients_None0})
final_selected_features = abs(df_final_selected_features['Coefficient']).sort_values(ascending = False)
final_selected_features_index = final_selected_features.index
X_train_None0 = X_train_None0.iloc[:, final_selected_features_index]

print('Selected descriptors:', X_train_None0.columns)
print('Done')