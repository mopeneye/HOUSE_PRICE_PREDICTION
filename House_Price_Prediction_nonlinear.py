import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.exceptions import ConvergenceWarning
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from lightgbm import LGBMRegressor
import xgboost as xgb
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from datetime import datetime
from warnings import filterwarnings
filterwarnings('ignore')

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter("ignore", category=ConvergenceWarning)

pd.pandas.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)


# GET DATA FROM PICKLE FILE

train_df = pd.read_pickle(r'E:\PROJECTS\dsmlbc\House_Price_Prediction_odev\datasets\prepared_data\train_df.pkl')
test_df = pd.read_pickle(r'E:\PROJECTS\dsmlbc\House_Price_Prediction_odev\datasets\prepared_data\test_df.pkl')

def clean_dataset(df):
    assert isinstance(df, pd.DataFrame), "df needs to be a pd.DataFrame"
    df.dropna(inplace=True)
    indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
    return df[indices_to_keep].astype(np.float64)


train_df.replace([np.inf, -np.inf], np.nan, inplace=True)

# all = [train_df, test_df]
# drop_list = ['index', 'Id']
#
# for data in all:
#     data.drop(drop_list, axis=1, inplace=True)
#     for col in data.columns:
#         data[col] = data[col].astype(int)


X = train_df.drop('SalePrice', axis=1)
y = np.ravel(train_df[["SalePrice"]])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state =123)
y_train = np.ravel(y_train)  # dimension setting

# CART Modelling and Prediction

cart_model = DecisionTreeRegressor(random_state=123)
cart_model.fit(X_train, y_train)
y_pred = cart_model.predict(X_test)
print('CART base: ', np.sqrt(mean_squared_error(y_test, y_pred)))  #

# CART Tuning

cart_params = {'max_depth': [5, 10, None],
               'min_samples_leaf': range(1, 6),
               'min_samples_split': range(2, 5),
               'criterion': ['mse', 'friedman_mse', 'mae'],
               'max_leaf_nodes': range(2, 11)}

cart_model = DecisionTreeRegressor()
print('CART Baslangic zamani: ', datetime.now())
cart_cv_model = GridSearchCV(cart_model, cart_params, cv =10, verbose=2, n_jobs=-1).fit(X_train, y_train)  # ???
print('CART Bitis zamani: ', datetime.now())
print('CART Best params: ', cart_cv_model.best_params_)

# Final model
print('CART Baslangic zamani: ', datetime.now())
cart_tuned = DecisionTreeRegressor(**cart_cv_model.best_params_).fit(X_train, y_train)  # ???
print('CART Bitis zamani: ', datetime.now())
y_pred = cart_tuned.predict(X_test)
print('CART tuned: ', np.sqrt(mean_squared_error(y_test, y_pred)))


# Decision rules

#pip install skompiler
#pip install astor
from skompiler import skompile

print(skompile(cart_tuned.predict).to('python/code'))

# Random Forests Modelling and Prediction

rf_model = RandomForestRegressor(random_state=123)
rf_model.fit(X_train, y_train)
y_pred = rf_model.predict(X_test)
print('RF base: ', np.sqrt(mean_squared_error(y_test, y_pred)))  # ???

# RF Tuning


rf_params = {'max_depth': [100, 200, None],
             'max_features': [100, 200],
             'max_leaf_nodes': [None, 10],
             'min_samples_split': range(10,  25),
             'min_samples_leaf': range(3, 10),
             'n_estimators': [300, 400, 500]}

rf_model = RandomForestRegressor()
print('RF Baslangic zamani: ', datetime.now())
rf_cv_model = GridSearchCV(rf_model,rf_params, cv =10, verbose=2, n_jobs=-1).fit(X_train, y_train)  # ???
print('RF Bitis zamani: ', datetime.now())
    print('RF Best params: ', rf_cv_model.best_params_)

# Final model
print('RF Baslangic zamani: ', datetime.now())
rf_tuned = RandomForestRegressor(**rf_cv_model.best_params_).fit(X_train, y_train)  # ???
print('RF Bitis zamani: ', datetime.now())
y_pred = rf_tuned.predict(X_test)
print('RF tuned: ', np.sqrt(mean_squared_error(y_test, y_pred)))

# Feature Importance

print(rf_tuned.feature_importances_)
Importance = pd.DataFrame({'Importance' : rf_tuned.feature_importances_ * 100,
                          'Feature': X_train.columns})

plt.figure(figsize=(10, 30))
sns.barplot(x="Importance", y="Feature", data=Importance.sort_values(by="Importance",ascending=False))
plt.title('Feature Importance')
plt.show()


# XGB Modelling and Prediction

xgb_model = XGBRegressor()
xgb_model.fit(X_train, y_train)
y_pred = xgb_model.predict(X_test)
print('XGB base: ', np.sqrt(mean_squared_error(y_test, y_pred)))  # ???

# XGB Tuning

# xgb_params = {"colsample_bytree": [0.08, 0.05, 0.03, 0.1, 0.001, 0.5, 1],
#              'learning_rate': [0.01, 0.05, 0.1, 0.3, 0.5, 0.8, 1],
#              'n_estimators': [100, 300, 500, 1000]}

xgb_params = {"colsample_bytree": [0.05, 0.1, 0.5, 1],
              'max_depth': np.arange(1, 11),
              'subsample': [0.5, 1],
              'learning_rate': [0.01, 0.05, 0.1, 0.5, 1],
              'n_estimators': [100, 500, 1000]}

xgb_model = XGBRegressor()
print('XGB Baslangic zamani: ', datetime.now())
xgb_cv_model = GridSearchCV(xgb_model, xgb_params, cv =10, verbose=2, n_jobs=-1).fit(X_train, y_train)  # ???
print('XGB Bitis zamani: ', datetime.now())
print('XGB Best params: ', xgb_cv_model.best_params_)

# Final model
print('XGB Baslangic zamani: ', datetime.now())
xgb_tuned = XGBRegressor(**xgb_cv_model.best_params_).fit(X_train, y_train)  # ???
print('XGB Bitis zamani: ', datetime.now())
y_pred = xgb_tuned.predict(X_test)
print('XGB tuned: ', np.sqrt(mean_squared_error(y_test, y_pred)))


# LightGBM Modelling and Prediction

lgbm_model = LGBMRegressor()
lgbm_model.fit(X_train, y_train)
y_pred = lgbm_model.predict(X_test)
print('LGBM base: ', np.sqrt(mean_squared_error(y_test, y_pred)))  # ???

# LGBM Tuning

# lgbm_params = {'max_depth': np.arange(1, 11),
#              "colsample_bytree": [1, 0.07, 0.05, 0.03, 0.1, 0.001],
#              'learning_rate': [0.01, 0.05, 0.1, 0.3, 0.5, 0.8, 1],
#              'n_estimators': np.arange(500, 5500, 500)}

lgbm_params = {'boosting_type': ['gbdt', 'dart', 'goss', 'rf'],
               'learning_rate': [0.05, 0.1, 0.5],
               'n_estimators': np.arange(500, 2000, 500),
               'num_leaves': [31, 50, 100, 200],
               # 'max_depth': np.arange(1, 7),
                "colsample_bytree": [0.05, 0.1,  0.5, 1]}

lgbm_model = LGBMRegressor()
print('LGBM Baslangic zamani: ', datetime.now())
lgbm_cv_model = GridSearchCV(lgbm_model, lgbm_params, cv =10, verbose=2, n_jobs=-1).fit(X_train, y_train)  # ???
print('LGBM Bitis zamani: ', datetime.now())
print('LGBM Best params: ', lgbm_cv_model.best_params_)

# Final model

print('LGBM Baslangic zamani: ', datetime.now())
lgbm_tuned = LGBMRegressor(**lgbm_cv_model.best_params_).fit(X_train, y_train)  # ???
print('LGBM Bitis zamani: ', datetime.now())
y_pred = lgbm_tuned.predict(X_test)
print('LGBM tuned: ', np.sqrt(mean_squared_error(y_test, y_pred)))

#Festure Importances

print(lgbm_tuned.feature_importances_)
Importance = pd.DataFrame({'Importance': lgbm_tuned.feature_importances_ * 100,
                          'Feature': X_train.columns})

plt.figure(figsize=(10, 30))
sns.barplot(x="Importance", y="Feature", data=Importance.sort_values(by="Importance",ascending=False))
plt.title('Feature Importance')
plt.show()
plt.savefig('lgbm_importances.png')

# CatBoost Modelling and Prediction

catb_model = CatBoostRegressor()
catb_model.fit(X_train, y_train)

plt.savefig('lgbm_importances.png')
y_pred = catb_model.predict(X_test)
print('CATB base: ', np.sqrt(mean_squared_error(y_test, y_pred)))  # ???

# Catb Tuning

catb_params = {'iterations': np.arange(500, 1000, 100),
             "depth": np.arange(1, 11),
             'learning_rate': [0.01, 0.05, 0.1, 0.5]}

catb_model = CatBoostRegressor()
print('CATB Baslangic zamani: ', datetime.now())
catb_cv_model = GridSearchCV(catb_model, catb_params, cv =10, verbose=2, n_jobs=-1).fit(X_train, y_train)  # ???
print('CATB Bitis zamani: ', datetime.now())
print('CATB Best params: ', catb_cv_model.best_params_)

# Final model
print('CATB Baslangic zamani: ', datetime.now())
catb_tuned = CatBoostRegressor (**catb_cv_model.best_params_).fit(X_train, y_train)  # ???
print('CATB Bitis zamani: ', datetime.now())
y_pred = catb_tuned.predict(X_test)
print('CATB tuned: ', np.sqrt(mean_squared_error(y_test, y_pred)))


# KNN Modelling and Prediction

RMSE = []

knn_model = KNeighborsRegressor().fit(X_train, y_train)
knn_model.get_params()
y_pred = knn_model.predict(X_test)
print('KNN Base: ', np.sqrt(mean_squared_error(y_test, y_pred)))

for k in range(20):
    k += 2
    knn_model = KNeighborsRegressor(n_neighbors = k).fit(X_train, y_train)
    y_pred = knn_model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    RMSE.append(rmse)
    print('KNN for k = ', k, ' RMSE value: ', rmse)

# Find optimum k value with GridSearch

knn_params = {'n_neighbors': np.arange(2, 30, 1)}
knn_model = KNeighborsRegressor()
print('KNN Baslangic zamani: ', datetime.now())
knn_cv_model = GridSearchCV(knn_model, knn_params, cv =10, verbose=2, n_jobs=-1).fit(X_train, y_train)
print('KNN Bitis zamani: ', datetime.now())
print('KNN Best params: ', knn_cv_model.best_params_)

# Final model

print('KNN Baslangic zamani: ', datetime.now())
knn_tuned = KNeighborsRegressor(**knn_cv_model.best_params_,).fit(X_train, y_train)
print('KNN Bitis zamani: ', datetime.now())
y_pred = knn_tuned.predict(X_test)
print('KNN tuned: ', np.sqrt(mean_squared_error(y_test, y_pred)))

# SVR Modelling and Prediction

svr_model = SVR('linear').fit(X_train, y_train)
y_pred = svr_model.predict(X_test)
print('SVM Base: ', np.sqrt(mean_squared_error(y_test, y_pred)))

# SVR Tuning

svr_params = {'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
                'C': [0.001, 0.01, 0.1, 0.5, 1, 10, 100, 500, 1000, 2000, 3000, 4000, 5000, 10000, 20000, 50000, 100000]}
svr_model = SVR()
print('SVR Baslangic zamani: ', datetime.now())
svr_cv_model = GridSearchCV(svr_model, svr_params, cv =10, verbose=2, n_jobs=-1).fit(X_train, y_train)  # 8 min.
print('SVR Bitis zamani: ', datetime.now())
print('SVR Best params: ', svr_cv_model.best_params_)

# Final model
print('SVR Baslangic zamani: ', datetime.now())
svr_tuned = SVR(**svr_cv_model.best_params_,).fit(X_train, y_train)  # 43086.867
print('SVR Bitis zamani: ', datetime.now())
y_pred = svr_tuned.predict(X_test)
print('SVR tuned: ', np.sqrt(mean_squared_error(y_test, y_pred)))

# GBM Modelling and Prediction

gbm_model = GradientBoostingRegressor()
gbm_model.fit(X_train, y_train)
y_pred = gbm_model.predict(X_test)
print('GBM base: ', np.sqrt(mean_squared_error(y_test, y_pred)))  # ???

# GBM Tuning

# gbm_params = {'max_depth': [10,  15, 20, 100, None],
#              'subsample': [1, 0.08, 0.05, 0.03, 0.1, 0.001],
#              'learning_rate ': {0.001, 0.01, 0.05, 0.1],
#              'loss': ["ls", "lad", "quantile"],
#              'n_estimators': np.arange(500, 3500, 500)}

gbm_params = {'max_depth': [20, 100, None],
             'subsample': [1, 0.5, 0.1],
             'learning_rate': [0.05, 0.1, 0.5],
             'loss': ['ls', 'lad', 'quantile'],
             'n_estimators': np.arange(500, 2000, 500)}

gbm_model = GradientBoostingRegressor()
print('GBM Baslangic zamani: ', datetime.now())
gbm_cv_model = GridSearchCV(gbm_model, gbm_params, cv =10, verbose=2, n_jobs=-1).fit(X_train, y_train)  # ???
print('GBM Bitis zamani: ', datetime.now())
print('GBM Best params: ', gbm_cv_model.best_params_)

# Final model
print('GBM Baslangic zamani: ', datetime.now())
gbm_tuned = GradientBoostingRegressor(**gbm_cv_model.best_params_).fit(X_train, y_train)  # ???
print('GBM Bitis zamani: ', datetime.now())
y_pred = gbm_tuned.predict(X_test)
print('GBM tuned: ', np.sqrt(mean_squared_error(y_test, y_pred)))


# ANN Modelling and Prediction

ann_model = MLPRegressor().fit(X_train, y_train)
y_pred = ann_model.predict(X_test)
print('ANN base: ', np.sqrt(mean_squared_error(y_test, y_pred)))  # ????

# ANN Tuning

ann_params = {'alpha': [0.1, 0.01, 0.02, 0.001, 0.0001],
             'hidden_layer_sizes': [(5, 5), (20, 20), (100, 100), (1000, 100, 10), (300, 200, 150)],
             'activation': ['relu', 'logistic', 'tanh'],
             'solver': ['sgd', 'adam', 'libfgs']}
print('ANN Baslangic zamani: ', datetime.now())
ann_cv_model = GridSearchCV(ann_model, ann_params, cv =10, verbose=2, n_jobs=-1).fit(X_train, y_train)  # ???
print('ANN Bitis zamani: ', datetime.now())
print('ANN Best params: ', ann_cv_model.best_params_)

# Final model
print('ANN Baslangic zamani: ', datetime.now())
ann_tuned = MLPRegressor(**ann_cv_model.best_params_,).fit(X_train, y_train)  # ???
print('ANN Bitis zamani: ', datetime.now())
y_pred = ann_tuned.predict(X_test)
print('ANN tuned: ', np.sqrt(mean_squared_error(y_test, y_pred)))


# XGB
# XGB base:  28308.087436255122
# XGB Baslangic zamani:  2020-10-21 13:01:55.972516
# Fitting 10 folds for each of 1200 candidates, totalling 12000 fits
# [Parallel(n_jobs=-1)]: Using backend LokyBackend with 8 concurrent workers.
# XGB Bitis zamani:  2020-10-21 16:04:13.662263
# XGB Best params:  {'colsample_bytree': 0.5, 'learning_rate': 0.01, 'max_depth': 3, 'n_estimators': 500, 'subsample': 1}
# XGB Baslangic zamani:  2020-10-21 16:04:13.662263
# XGB Bitis zamani:  2020-10-21 16:04:15.302472
# XGB tuned:  27726.60232397898
#
# KNN
# KNN Base:  27484.482915326545
# KNN Baslangic zamani:  2020-10-21 16:07:21.729235
# Fitting 10 folds for each of 28 candidates, totalling 280 fits
# [Parallel(n_jobs=-1)]: Using backend LokyBackend with 8 concurrent workers.
# [Parallel(n_jobs=-1)]: Done 280 out of 280 | elapsed:   13.4s finished
# KNN Bitis zamani:  2020-10-21 16:07:35.439538
# KNN Best params:  {'n_neighbors': 19}
# KNN Baslangic zamani:  2020-10-21 16:07:35.439538
# KNN Bitis zamani:  2020-10-21 16:07:35.502642
# KNN tuned:  29272.874385027957
#
# SVR
# SVM Base:  40379.63385917046
# SVR Baslangic zamani:  2020-10-21 16:09:54.931391
# Fitting 10 folds for each of 68 candidates, totalling 680 fits
# [Parallel(n_jobs=-1)]: Using backend LokyBackend with 8 concurrent workers.
# SVR Bitis zamani:  2020-10-21 17:00:22.892214
# SVR Best params:  {'C': 500, 'kernel': 'linear'}
# SVR Baslangic zamani:  2020-10-21 17:00:22.892214
# SVR Bitis zamani:  2020-10-21 17:00:24.986984
# SVR tuned:  28323.8060709585
#
# CART
# CART base:  34950.64464757155
# CART Baslangic zamani:  2020-10-21 17:10:32.933366
# Fitting 10 folds for each of 1215 candidates, totalling 12150 fits
# [Parallel(n_jobs=-1)]: Using backend LokyBackend with 8 concurrent workers.
# CART Bitis zamani:  2020-10-21 17:20:23.900661
# CART Best params:  {'criterion': 'mse', 'max_depth': 5, 'max_leaf_nodes': 9, 'min_samples_leaf': 5, 'min_samples_split': 2}
# CART Baslangic zamani:  2020-10-21 17:20:23.900661
# CART Bitis zamani:  2020-10-21 17:20:23.912965
# CART tuned:  32584.985467668583
#
# LGBM
# LGBM base:  28577.02441597784
# LGBM Baslangic zamani:  2020-10-21 17:39:34.748791
# Fitting 10 folds for each of 576 candidates, totalling 5760 fits
# [Parallel(n_jobs=-1)]: Using backend LokyBackend with 8 concurrent workers.
# LGBM Bitis zamani:  2020-10-21 18:50:35.471950
# LGBM Best params:  {'boosting_type': 'dart', 'colsample_bytree': 0.5, 'learning_rate': 0.05, 'n_estimators': 500, 'num_leaves': 31}
# LGBM Baslangic zamani:  2020-10-21 18:50:35.471950
# LGBM Bitis zamani:  2020-10-21 18:50:38.388709
# LGBM tuned:  28554.953583258684
#
# CATB
# CATB base:  26382.004111864477
# CATB Baslangic zamani:  2020-10-21 19:07:08.836122
# Fitting 10 folds for each of 200 candidates, totalling 2000 fits
# [Parallel(n_jobs=-1)]: Using backend LokyBackend with 8 concurrent workers.
# CATB Bitis zamani:  2020-10-22 00:05:59.058166
# CATB Best params:  {'depth': 3, 'iterations': 900, 'learning_rate': 0.01}
# CATB Baslangic zamani:  2020-10-22 00:05:59.058166
# CATB Bitis zamani:  2020-10-22 00:06:01.549549
# CATB tuned:  27049.585972283177
#
# RF base:  26762.98198719678
# RF Baslangic zamani:  2020-10-23 23:42:17.470707
# Fitting 10 folds for each of 3780 candidates, totalling 37800 fits
# [Parallel(n_jobs=-1)]: Using backend LokyBackend with 8 concurrent workers.
# RF Bitis zamani:  2020-10-24 04:34:54.438990
# RF Best params:  {'max_depth': 200, 'max_features': 100, 'max_leaf_nodes': None, 'min_samples_leaf': 3, 'min_samples_split': 13, 'n_estimators': 300}
# RF Baslangic zamani:  2020-10-24 04:34:54.438990
# RF Bitis zamani:  2020-10-24 04:34:58.452366
# RF tuned:  27000.040479177864




