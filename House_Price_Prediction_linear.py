import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import LassoCV
from sklearn.linear_model import ElasticNetCV
from sklearn.neighbors import LocalOutlierFactor

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter("ignore", category=ConvergenceWarning)

pd.pandas.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)


def load_House_Price_data():
    train = pd.read_csv(r"datasets\train.csv")
    test = pd.read_csv(r"datasets\test.csv")
    data = train.append(test).reset_index()

    return data


df = load_House_Price_data()

# GENERAL

print(df.head())

print(df.tail())

print(df.info())

print(df.columns)

print(df.index)

print(df.describe().T)

print(df.isnull().values.any())

print(df.isnull().sum().sort_values(ascending=False))

# EDA

df.drop(['Id', 'index'], axis=1, inplace=True)

# CATEGORICAL VARIABLE ANALYSIS
cat_cols = [col for col in df.columns if df[col].dtypes == 'O']
print('Categorical Variable Count: ', len(cat_cols))


def cat_summary(data, categorical_cols, target, number_of_classes=10):
    var_count = 0
    vars_more_classes = []
    for var in categorical_cols:
        if len(df[var].value_counts()) <= number_of_classes:  # sınıf sayısına göre seç
            print(pd.DataFrame({var: data[var].value_counts(),
                                "Ratio": 100 * data[var].value_counts() / len(data),
                                "TARGET_MEDIAN": data.groupby(var)[target].median()}), end="\n\n\n")
            sns.countplot(x=var, data=data)
            plt.show()
            var_count += 1
        else:
            vars_more_classes.append(data[var].name)
    print('%d categorical variables have been described' % var_count, end="\n\n")
    print('There are', len(vars_more_classes), "variables have more than", number_of_classes, "classes", end="\n\n")
    print('Variable names have more than %d classes:' % number_of_classes, end="\n\n")
    print(vars_more_classes)


cat_summary(df, cat_cols, "SalePrice")

# variables have more than 10 classes:

for col in ['Neighborhood', 'Exterior1st', 'Exterior2nd']:
    print(df[col].value_counts())

#NUMERICAL VARIABLES ANALYSIS

# df.loc[(df["YearBuilt"]==df["YearRemodAdd"]),'Modified'] = 0
# df.loc[(df["YearBuilt"] != df["YearRemodAdd"]),'Modified'] = 1
# df['House_Age'] = 2010 - df['YearRemodAdd']
# df['Garage_Age'] = 2010 - df['GarageYrBlt']
#
# df.drop('YearBuilt', axis = 1, inplace = True)
# df.drop('GarageYrBlt', axis = 1, inplace = True)
# df.drop('YearRemodAdd', axis = 1, inplace = True)
# df.drop('YrSold', axis = 1, inplace = True)


num_cols = [col for col in df.columns if df[col].dtypes != 'O' and col not in "Id"]
print('Numerical Variables Count: ', len(num_cols))


def hist_for_nums(data, numeric_cols):
    col_counter = 0
    data = data.copy()
    for col in numeric_cols:
        data[col].hist(bins=20)
        plt.xlabel(col)
        plt.title(col)
        plt.show()
        sns.boxplot(x=df[col]);
        plt.show()
        col_counter += 1
    print(col_counter, "variables have been plotted")


hist_for_nums(df, num_cols)


def correlation_matrix(df, cols):
    fig = plt.gcf()
    fig.set_size_inches(10, 8)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    fig = sns.heatmap(df[cols].corr(), annot=True, linewidths=0.5, annot_kws={'size': 12}, linecolor='w',
                      cmap='RdBu')
    plt.show()


correlation_matrix(df, (num_cols[0:10] + ['SalePrice']))
correlation_matrix(df, (num_cols[10:20] + ['SalePrice']))
correlation_matrix(df, (num_cols[20:30] + ['SalePrice']))
correlation_matrix(df, (num_cols[30:] + ['SalePrice']))


# TARGET ANALYSE

def find_correlation(dataframe, corr_limit=0.30):
    high_correlations = []
    low_correlations = []
    for col in num_cols:
        if col == "SalePrice":
            pass

        else:
            correlation = dataframe[[col, "SalePrice"]].corr().loc[col, "SalePrice"]
            print(col, correlation)
            if abs(correlation) > corr_limit:
                high_correlations.append(col)
            else:
                low_correlations.append(col)
    return low_correlations, high_correlations


low_corrs, high_corrs = find_correlation(df)

print('Variables have low correlation with target:')
print('-' * 44)
print(low_corrs)
print('Variables have high correlation with target:')
print('-' * 44)
print(high_corrs)

# Pairplot of variables that have high correlation with target data

sns.set()
sns.pairplot(df[high_corrs], height=2.5)
plt.show()


# 3. DATA PREPROCESSING & FEATURE ENGINEERING

# RARE ANALYZER

def rare_analyser(dataframe, target, rare_perc):
    rare_columns = [col for col in df.columns if len(df[col].value_counts()) <= 20
                    and (df[col].value_counts() / len(df) < rare_perc).any(axis=None)]
    for var in rare_columns:
        print(var, ":", len(dataframe[var].value_counts()))
        print(pd.DataFrame({"COUNT": dataframe[var].value_counts(),
                            "RATIO": dataframe[var].value_counts() / len(dataframe),
                            "TARGET_MEDIAN": dataframe.groupby(var)[target].median()}), end="\n\n\n")

rare_analyser(df, "SalePrice", 0.01)
#
# # Operation after Rare Analyzer
# # Get rid of a class
# df = df[~(df['MSSubClass'] == 150)]
# #
# # df = df[~(df['Exterior1st'] == 'ImStucc')]
# # df = df[~(df['Exterior1st'] == 'AsphShn')]
# # df = df[~(df['Exterior1st'] == 'CBlock')]
# # df = df[~(df['Exterior1st'] == 'Stone')]
# #
# # df = df[~(df['Exterior2nd'] == 'Other')]
# df = df[~(df['Exterior2nd'] == 'CBlock')]
# #
# # df = df[~(df['ExterCond'] == 'Po')]
# #
# # df = df[~(df['Foundation'] == 'Wood')]
# #
# df = df[~(df['FullBath'] == 4 )]
# #
# # df = df[~(df['KitchenAbvGr'] == 0 )]
# # df = df[~(df['KitchenAbvGr'] == 3 )]
# #
# # df = df[~(df['TotRmsAbvGrd'] == 2)]
# df = df[~(df['TotRmsAbvGrd'] == 13)]
# df = df[~(df['TotRmsAbvGrd'] == 15)]
# #
# df = df[~(df['Fireplaces'] == 4)]
# #
# df = df[~(df['GarageCars'] == 5)]
# # # Append to another class
# df.loc[(df['MSSubClass'] == 40), 'MSSubClass'] = 85
# # df.loc[(df['MSSubClass'] == 45), 'MSSubClass'] = 30
# df.loc[(df['MSSubClass'] == 75), 'MSSubClass'] = 80
# # df.loc[(df['MSSubClass'] == 180), 'MSSubClass'] = 30
# # df.loc[(df['MSSubClass'] == 70), 'MSSubClass'] = 20
# #
# df.loc[(df['LotConfig'] == 'FR3'), 'LotConfig'] = 'CulDSac'
# #
# df.loc[(df['LandSlope'] == 'Sev'), 'LandSlope'] = 'Mod'
# #
# # df.loc[(df['Condition1'] == 'RRAe'), 'Condition1'] = 'Feedr'
# df.loc[(df['Condition1'] == 'RRNn'), 'Condition1'] = 'PosA'
# #
# # df.loc[(df['HouseStyle'] == '2.5Fin'), 'HouseStyle'] = '2Story'
# df.loc[(df['HouseStyle'] == '2.5Unf'), 'HouseStyle'] = 'SFoyer'
# #
# df.loc[(df['RoofStyle'] == 'Mansard'), 'RoofStyle'] = 'Hip'
# #
# # df.loc[(df['Exterior2nd'] == 'AsphShn'), 'Exterior2nd'] = 'HdBoard '
# # df.loc[(df['Exterior2nd'] == 'Brk Cmn'), 'Exterior2nd'] = 'Stucco'
# # df.loc[(df['Exterior2nd'] == 'Stone'), 'Exterior2nd'] = 'Wd Sdng'
# #
# # df.loc[(df['MasVnrType'] == 'BrkCmn'), 'MasVnrType'] = 'None'
# #
# # df.loc[(df['ExterCond'] == 'Ex'), 'ExterCond'] = 'TA'
# #
# df.loc[(df['Foundation'] == 'Stone'), 'Foundation'] = 'BrkTil'
# #
# df.loc[(df['BsmtHalfBath'] == 2), 'BsmtHalfBath'] = 1
# #
# df.loc[(df['BsmtFullBath'] == 3), 'BsmtFullBath'] = 1
# #
# # df.loc[(df['BedroomAbvGr'] == 0), 'BedroomAbvGr'] = 8
# #
# df.loc[(df['TotRmsAbvGrd'] == 14), 'TotRmsAbvGrd'] = 8
# df.loc[(df['TotRmsAbvGrd'] == 12), 'TotRmsAbvGrd'] = 8
# #
# df.loc[(df['Fireplaces'] == 3), 'Fireplaces'] = 2
# #
# # df.loc[(df['GarageCars'] == 4), 'GarageCars'] = 3

def rare_encoder(dataframe, rare_perc):
    temp_df = dataframe.copy()
    rare_columns = [col for col in temp_df.columns if temp_df[col].dtypes == 'O'
                    and (temp_df[col].value_counts() / len(temp_df) < rare_perc).any(axis=None)]
    for var in rare_columns:
        tmp = temp_df[var].value_counts() / len(temp_df)
        rare_labels = tmp[tmp < rare_perc].index
        temp_df[var] = np.where(temp_df[var].isin(rare_labels), 'Rare', temp_df[var])
    return temp_df


df = rare_encoder(df, 0.01)
rare_analyser(df, "SalePrice", 0.01)

drop_list = ["Street", "Utilities", "LandSlope", "PoolQC", "MiscFeature", "Condition2", "RoofMatl", "3SsnPorch", "PoolArea"]
             # "OverallCond", "BsmtFinSF2", "LowQualFinSF", "BsmtHalfBath", "MoSold", "YrSold"]

# drop_list2 = ['MSSubClass', 'OverallCond', 'BsmtFinSF2', 'LowQualFinSF', 'BsmtHalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'EnclosedPorch',
#             'ScreenPorch', 'MiscVal', 'MoSold', 'YrSold']
drop_list_3 = ['MSSubClass', 'LotArea', 'OverallCond', 'BsmtFinSF2', 'BsmtUnfSF', 'LowQualFinSF',
               'BsmtFullBath', 'BsmtHalfBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'EnclosedPorch',  'ScreenPorch', 'MiscVal',
               'MoSold', 'YrSold']

cat_cols = [col for col in df.columns if df[col].dtypes == 'O'
            and col not in drop_list and col not in drop_list_3]

for col in drop_list:
    df.drop(col, axis=1, inplace=True)

for col in drop_list_3:
    df.drop(col, axis=1, inplace=True)

rare_analyser(df, "SalePrice", 0.01)

# LABEL ENCODING & ONE-HOT ENCODING
def one_hot_encoder(dataframe, categorical_cols, nan_as_category=False):
    original_columns = list(dataframe.columns)
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, dummy_na=nan_as_category, drop_first=True)
    new_columns = [c for c in dataframe.columns if c not in original_columns]
    return dataframe, new_columns


df, new_cols_ohe = one_hot_encoder(df, cat_cols)
cat_summary(df, new_cols_ohe, "SalePrice")

# [col for col in df.columns if 'Gar' in col]

# drop_list2 = ['MSZoning_nan',
#  'Alley_nan',
#  'LotShape_nan',
#  'LandContour_nan',
#  'LotConfig_nan',
#  'Neighborhood_nan',
#  'Condition1_nan',
#  'Condition2_nan',
#  'BldgType_nan',
#  'HouseStyle_nan',
#  'RoofStyle_nan',
#  'RoofMatl_nan',
#  'Exterior1st_nan',
#  'Exterior2nd_nan',
#  'MasVnrType_nan',
#  'ExterQual_nan',
#  'ExterCond_nan',
#  'Foundation_nan',
#  'BsmtQual_nan',
#  'BsmtCond_nan',
#  'BsmtExposure_nan',
#  'BsmtFinType1_nan',
#  'BsmtFinType2_nan',
#  'Heating_nan',
#  'HeatingQC_nan',
#  'CentralAir_nan',
#  'Electrical_nan',
#  'KitchenQual_nan',
#  'Functional_nan',
#  'FireplaceQu_nan',
#  'GarageType_nan',
#  'GarageFinish_nan',
#  'GarageQual_nan',
#  'GarageCond_nan',
#  'PavedDrive_nan',
#  'Fence_nan',
#  'SaleType_nan',
#  'SaleCondition_nan']
#
# for col in drop_list2:
#     df.drop(col, axis=1, inplace=True)

# MISSING_VALUES
def missing_values_table(dataframe):
    variables_with_na = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]
    n_miss = dataframe[variables_with_na].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[variables_with_na].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df)
    return variables_with_na


missing_values_table(df)
df = df.apply(lambda x: x.fillna(x.median()), axis=0)
missing_values_table(df)


# OUTLIERS
def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.05)
    quartile3 = dataframe[variable].quantile(0.95)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


def has_outliers(dataframe, num_col_names, plot=False):
    variable_names = []
    for col in num_col_names:
        low_limit, up_limit = outlier_thresholds(dataframe, col)
        if dataframe[(dataframe[col] > up_limit) | (dataframe[col] < low_limit)].any(axis=None):
            number_of_outliers = dataframe[(dataframe[col] > up_limit) | (dataframe[col] < low_limit)].shape[0]
            print(col, ":", number_of_outliers)
            variable_names.append(col)
            if plot:
                sns.boxplot(x=dataframe[col])
                plt.show()
    return variable_names

num_cols = [col for col in df.columns if df[col].dtypes != 'O']

has_outliers(df, num_cols)


def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit


for col in num_cols:
    replace_with_thresholds(df, col)

clf = LocalOutlierFactor(n_neighbors = 20, contamination=0.1)

clf.fit_predict(df)

df_scores = clf.negative_outlier_factor_

np.sort(df_scores)[0:1000]

esik_deger = np.sort(df_scores)[1000]

aykiri_tf = df_scores > esik_deger
new_df = df[df_scores > esik_deger]


new_df.shape

df_scores.mean()

baski_deger = df[df_scores == esik_deger]
aykirilar = df[~aykiri_tf]

baski_deger

res = aykirilar.to_records(index = False)
res[:] = baski_deger.to_records(index = False)

df[~aykiri_tf] = pd.DataFrame(res, index = df[~aykiri_tf].index)

has_outliers(df, num_cols)

df[~aykiri_tf]

# STANDARTLASTIRMA

df.head()
like_num = [col for col in df.columns if df[col].dtypes != 'O' and len(df[col].value_counts()) < 20]
cols_need_scale = [col for col in df.columns if col not in new_cols_ohe
                   and col not in "Id"
                   and col not in "SalePrice"
                   and col not in like_num]

df[cols_need_scale].head()
df[cols_need_scale].describe([0.05, 0.10, 0.25, 0.50, 0.75, 0.80, 0.90, 0.95, 0.99]).T
hist_for_nums(df, cols_need_scale)


def robust_scaler(variable):
    var_median = variable.median()
    quartile1 = variable.quantile(0.25)
    quartile3 = variable.quantile(0.75)
    interquantile_range = quartile3 - quartile1
    if int(interquantile_range) == 0:
        quartile1 = variable.quantile(0.05)
        quartile3 = variable.quantile(0.95)
        interquantile_range = quartile3 - quartile1
        z = (variable - var_median) / interquantile_range
        return round(z, 3)
    else:
        z = (variable - var_median) / interquantile_range
    return round(z, 3)


for col in cols_need_scale:
    df[col] = robust_scaler(df[col])

df[cols_need_scale].head()
df[cols_need_scale].describe().T
hist_for_nums(df, cols_need_scale)

# son kontrol
missing_values_table(df)
has_outliers(df, num_cols)

# Veriyi kaydetme

train_df = df[df['SalePrice'].notnull()]
test_df = df[df['SalePrice'].isnull()]

train_df.to_pickle("datasets/prepared_data/train_df.pkl")
test_df.to_pickle("datasets/prepared_data/test_df.pkl")

# MODELLING

X = train_df.drop('SalePrice', axis=1)
y = train_df[["SalePrice"]]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=46)

models = [('LinearRegression', LinearRegression()),
          ('Ridge', Ridge()),
          ('Lasso', Lasso()),
          ('ElasticNet', ElasticNet())]

# evaluate each model in turn
results = []
names = []

print('Base Regression Models')
print('-' * 23)

for name, model in models:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    result = np.sqrt(mean_squared_error(y_test, y_pred))
    results.append(result)
    names.append(name)
    msg = "%s: %f" % (name, result)
    print(msg)

#tuned models

print('\n')
print('Tuned Regression Models')
print('-' * 23)

for name, model in models:
    if (name == 'LinearRegression'):
                model.fit(X_train, y_train)
                result = np.sqrt(-cross_val_score(model,  # bu da test icin rmse'li cross validation
                             X_test,
                             y_test,
                             cv=10,
                             scoring="neg_mean_squared_error")).mean()
                results.append(result)
                names.append(name)
                msg = "%s: %f" % (name, result)
                print(msg)
    if (name == 'Ridge'):
        lambdalar = 10 ** np.linspace(10, -2, 100) * 0.5
        ridge_cv = RidgeCV(alphas=lambdalar,
                           scoring="neg_mean_squared_error",
                           normalize=True)
        ridge_cv.fit(X_train, y_train)
        ridge_tuned = Ridge(alpha=ridge_cv.alpha_,
                            normalize=True).fit(X_train, y_train)
        result = np.sqrt(mean_squared_error(y_test, ridge_tuned.predict(X_test)))
        results.append(result)
        names.append(name)
        msg = "%s: %f" % (name, result)
        print(msg)
    if (name == 'Lasso'):
        lasso_cv_model = LassoCV(alphas=None,
                                 cv=10,
                                 max_iter=10000,
                                 normalize=True)
        lasso_cv_model.fit(X_train, y_train)
        lasso_tuned = Lasso(alpha = lasso_cv_model.alpha_)
        lasso_tuned.fit(X_train, y_train)
        y_pred = lasso_tuned.predict(X_test)
        result = np.sqrt(mean_squared_error(y_test, y_pred))
        results.append(result)
        names.append(name)
        msg = "%s: %f" % (name, result)
        print(msg)
    if (name == 'ElasticNet'):
        enet_cv_model = ElasticNetCV(cv = 10, random_state = 0).fit(X_train, y_train)
        enet_tuned = ElasticNet(alpha=enet_cv_model.alpha_).fit(X_train, y_train)
        y_pred = enet_tuned.predict(X_test)
        result = np.sqrt(mean_squared_error(y_test, y_pred))
        results.append(result)
        names.append(name)
        msg = "%s: %f" % (name, result)
        print(msg)
#
# Base Regression Models
# -----------------------
# LinearRegression: 25011.694958
# Ridge: 24978.086590
# Lasso: 25001.499789
# ElasticNet: 25087.428992

# Tuned Regression Models
# -----------------------
# LinearRegression: 97408.034919
# Ridge: 24829.386053
# Lasso: 24935.720865
# ElasticNet: 38140.852125


[col for col in df.columns if 'Year' in col or 'Yr' in col]

df[['GarageYrBlt', 'YrSold']]