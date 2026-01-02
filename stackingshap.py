
import pandas as pd

df = pd.read_csv("1.0.csv")


df = df.sort_values(by=["bulk_modulus"], ascending=False)

df = df[(df["bulk_modulus"] < 1000) & (df["bulk_modulus"] > -1000)]

df = df[(df["bulk_modulus"] < 350) & (df["bulk_modulus"] > -500)]

df = df[df["bulk_modulus"] > -100]
#df = df[(df["shear_modulus"] < 200) & (df["bulk_modulus"] > 0)]

df = df.dropna()

df = df.drop(columns=[ "structure", "brgoch_feats", "suspect_value", "composition"])
df = df.drop(columns=["composition_oxid"])
df = df.drop(columns=["is_centrosymmetric",'crystal_system'])
df = df.drop(columns=["compound possible"])
df = df.drop(columns=['MagpieData minimum NfUnfilled',
'MagpieData maximum NfUnfilled',
'MagpieData range NfUnfilled',
'MagpieData mean NfUnfilled',
'MagpieData avg_dev NfUnfilled',
'MagpieData mode NfUnfilled'])

df = df[~df.isin([float("inf"), -float("inf")]).any(axis=1)]

for col in df.columns:
    if df[col].nunique() < 0.01 * len(df):
        df = df.drop(columns=[col])

new_columns = [col.replace('MagpieData ', '') for col in df.columns]
df.columns = new_columns


data_target = df.iloc[:, 0:3]
#data_features = df.iloc[:, 3:]
data_target['Flexibility'] = data_target['bulk_modulus'] / data_target['shear_modulus']
data_features = df.iloc[:, 3:]

data_target['Flexibility_class'] = (data_target['Flexibility'] > 1.75).astype(int)


print("分类结果统计:")
print(data_target['Flexibility_class'].value_counts())
print("\n分类比例:")
print(data_target['Flexibility_class'].value_counts(normalize=True).round(4) * 100, "%")


import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(12, 5))


plt.subplot(1, 2, 1)
sns.histplot(data_target['Flexibility'], bins=30, kde=True,)
plt.axvline(x=1.75, color='red', linestyle='--', label='B/G=1.75')
plt.xlabel('Flexibility (B/G)', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.title('Flexibility', fontsize=16)
plt.legend()


plt.subplot(1, 2, 2)
sns.scatterplot(x='shear_modulus', y='bulk_modulus', 
                hue='Flexibility_class', data=data_target,
                palette={0:'blue', 1:'red'}, legend=False)
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='B/G<1.75'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='B/G>1.75')
]

plt.legend(handles=legend_elements, loc='best')

plt.title('Mechanical flexibility classification', fontsize=16)
plt.xlabel('shear_modulus (GPa)', fontsize=14)
plt.ylabel('bulk_modulus (GPa)', fontsize=14)
import numpy as np
import re
from typing import Dict, List, Tuple

class FeatureNameCleaner:
    """Utility for cleaning feature names."""

    def __init__(self):
        self.feature_mapping = {}
        self.original_names = []

    def clean_feature_names(self, df: pd.DataFrame, method: str = 'clean') -> pd.DataFrame:
        """
        Clean feature names in a DataFrame.

        Parameters:
        df: input DataFrame
        method: cleanup method ('clean', 'simple', 'numbered')

        Returns:
        Cleaned DataFrame
        """

        self.original_names = df.columns.tolist()

        if method == 'clean':
            df_cleaned = self._clean_method(df)
        elif method == 'simple':
            df_cleaned = self._simple_method(df)
        elif method == 'numbered':
            df_cleaned = self._numbered_method(df)
        else:
            raise ValueError("method must be 'clean', 'simple', or 'numbered'")

        return df_cleaned

    def _clean_method(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean names while preserving meaning."""
        df_copy = df.copy()
        new_columns = []

        for col in df_copy.columns:

            col_str = str(col)


            cleaned_name = col_str.replace('[', '').replace(']', '').replace('<', '').replace('>', '')


            cleaned_name = re.sub(r'[^a-zA-Z0-9_\u4e00-\u9fff]', '_', cleaned_name)


            cleaned_name = re.sub(r'_+', '_', cleaned_name)


            cleaned_name = cleaned_name.strip('_')


            if not cleaned_name:
                cleaned_name = f'feature_{len(new_columns)}'

            new_columns.append(cleaned_name)


        final_columns = self._handle_duplicates(new_columns)


        self.feature_mapping = {final_columns[i]: self.original_names[i] 
                               for i in range(len(final_columns))}

        df_copy.columns = final_columns
        return df_copy

    def _simple_method(self, df: pd.DataFrame) -> pd.DataFrame:
        """Simple renaming method."""
        df_copy = df.copy()
        new_columns = [f'feature_{i}' for i in range(len(df_copy.columns))]


        self.feature_mapping = {new_columns[i]: self.original_names[i] 
                               for i in range(len(new_columns))}

        df_copy.columns = new_columns
        return df_copy

    def _numbered_method(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prefixed numbering method."""
        df_copy = df.copy()
        new_columns = [f'col_{i:03d}' for i in range(len(df_copy.columns))]


        self.feature_mapping = {new_columns[i]: self.original_names[i] 
                               for i in range(len(new_columns))}

        df_copy.columns = new_columns
        return df_copy

    def _handle_duplicates(self, columns: List[str]) -> List[str]:
        """Handle duplicate column names."""
        final_columns = []
        seen = {}

        for col in columns:
            if col in seen:
                seen[col] += 1
                new_col = f"{col}_{seen[col]}"
            else:
                seen[col] = 0
                new_col = col
            final_columns.append(new_col)

        return final_columns

    def get_mapping(self) -> Dict[str, str]:
        """Return the feature name mapping."""
        return self.feature_mapping

    def print_mapping(self):
        """Print the feature name mapping."""
        print("特征名称映射:")
        print("-" * 60)
        for new_name, old_name in self.feature_mapping.items():
            print(f"{new_name:<20} -> {old_name}")

    def save_mapping(self, filepath: str):
        """Save the mapping to a file."""
        mapping_df = pd.DataFrame([
            {'new_name': k, 'original_name': v} 
            for k, v in self.feature_mapping.items()
        ])
        mapping_df.to_csv(filepath, index=False, encoding='utf-8')
        print(f"映射已保存到: {filepath}")

def check_feature_names(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """
    Check feature names for unsupported characters.

    Returns:
    (is_valid, problematic_columns)
    """
    problematic_chars = ['[', ']', '<', '>']
    problematic_cols = []

    for col in df.columns:
        col_str = str(col)
        if any(char in col_str for char in problematic_chars):
            problematic_cols.append(col)

    is_valid = len(problematic_cols) == 0
    return is_valid, problematic_cols

def fix_data_features(df: pd.DataFrame, method: str = 'clean', 
                     save_mapping: bool = True, mapping_file: str = None) -> pd.DataFrame:
    """
    Full feature-name cleanup workflow.

    Parameters:
    df: input DataFrame
    method: cleanup method ('clean', 'simple', 'numbered')
    save_mapping: whether to save the mapping
    mapping_file: mapping file path

    Returns:
    Cleaned DataFrame
    """
    print("=" * 60)
    print("特征名称修复工具")
    print("=" * 60)


    print(f"原始数据形状: {df.shape}")
    print(f"原始特征数量: {len(df.columns)}")


    is_valid, problematic_cols = check_feature_names(df)

    if is_valid:
        print("✅ 所有特征名称都符合要求，无需修复")
        return df.copy()

    print(f"❌ 发现 {len(problematic_cols)} 个包含问题字符的特征名称")
    print("问题特征名称示例:")
    for i, col in enumerate(problematic_cols[:5]):
        print(f"  {i+1}. {col}")
    if len(problematic_cols) > 5:
        print(f"  ... 还有 {len(problematic_cols) - 5} 个")


    cleaner = FeatureNameCleaner()
    print(f"\n🔧 使用 '{method}' 方法修复特征名称...")

    df_fixed = cleaner.clean_feature_names(df, method=method)

    print("✅ 修复完成!")
    print(f"修复后特征数量: {len(df_fixed.columns)}")


    if len(cleaner.get_mapping()) <= 20:
        cleaner.print_mapping()
    else:
        print(f"映射包含 {len(cleaner.get_mapping())} 个特征，前5个示例:")
        mapping_items = list(cleaner.get_mapping().items())[:5]
        for new_name, old_name in mapping_items:
            print(f"  {new_name:<20} -> {old_name}")


    if save_mapping:
        if mapping_file is None:
            mapping_file = "feature_name_mapping.csv"
        cleaner.save_mapping(mapping_file)

    return df_fixed


def quick_fix_feature_names(df: pd.DataFrame) -> pd.DataFrame:
    """Quick helper to clean feature names."""
    return fix_data_features(df, method='clean', save_mapping=False)


def prepare_for_ml(df: pd.DataFrame, model_type: str = 'xgboost') -> pd.DataFrame:
    """
    Prepare data for ML by cleaning feature names.

    Parameters:
    df: input DataFrame
    model_type: model type ('xgboost', 'lightgbm', 'sklearn')
    """
    if model_type.lower() in ['xgboost', 'xgb']:

        return fix_data_features(df, method='clean', save_mapping=True)
    elif model_type.lower() in ['lightgbm', 'lgb']:

        return fix_data_features(df, method='clean', save_mapping=True)
    else:

        return fix_data_features(df, method='simple', save_mapping=True)
features_to_keep = data_features[['vpa',
 'mean NUnfilled',
 'maximum MeltingT',
 'mode NValence',
 'max packing efficiency',
 'minimum MeltingT',
 'range MendeleevNumber',
 'structural complexity per cell',
 'mean MendeleevNumber',
 'mode CovalentRadius',
 'avg_dev NUnfilled']]

X = features_to_keep
#X_0 = data_features[['MagpieData minimum CovalentRadius', 'MagpieData maximum CovalentRadius', 'MagpieData mean CovalentRadius',
#              'MagpieData minimum Electronegativity', 'MagpieData maximum Electronegativity', 'MagpieData mean Electronegativity',
#              'MagpieData minimum MeltingT', 'MagpieData maximum MeltingT', 'MagpieData mean MeltingT',
#             'minimum oxidation state', 'maximum oxidation state', 'density', 'vpa', 'packing fraction', 'mean Average bond angle','mean Average bond length']]
y1 = data_target['bulk_modulus']
y2 = data_target['shear_modulus']


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)

from sklearn.model_selection import train_test_split
X_train, X_test, y1_train, y1_test, y2_train, y2_test = train_test_split(
    X_scaled, y1, y2, test_size=0.2, random_state=42)


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import warnings
warnings.filterwarnings("ignore")
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler 
from sklearn.pipeline import Pipeline

pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('knn', KNeighborsRegressor())
])

# ---------------------------------------------------------

# ---------------------------------------------------------
knn_param_grid = {




    'knn__n_neighbors': list(range(5, 51, 2)), 





    'knn__weights': ['uniform'], 


    'knn__p': [1, 2]
}


knn_cv = KFold(n_splits=5, shuffle=True, random_state=42)


knn_grid_search = GridSearchCV(
    estimator=pipe,
    param_grid=knn_param_grid, 
    cv=knn_cv,
    scoring='neg_mean_squared_error',
    n_jobs=-1,
    verbose=1
)


print("Fitting KNN GridSearchCV with Standardization...")

knn_grid_search.fit(X_train, y2_train)


print("KNN Best parameters found: ", knn_grid_search.best_params_)
knn_best_score = -knn_grid_search.best_score_
print("KNN Best CV MSE: ", knn_best_score)


knn_best_model = knn_grid_search.best_estimator_
knn_y_pred = knn_best_model.predict(X_test)

# ---------------------------------------------------------

# ---------------------------------------------------------
knn_rmse = np.sqrt(mean_squared_error(y2_test, knn_y_pred))
knn_r2 = r2_score(y2_test, knn_y_pred)

print("-" * 30)
print(f"KNN Test RMSE: {knn_rmse:.4f}")
print(f"KNN Test R²:   {knn_r2:.4f}")


train_pred = knn_best_model.predict(X_train)
knn_train_r2 = r2_score(y2_train, train_pred)
print(f"KNN Train R²:  {knn_train_r2:.4f}")
print("-" * 30)


if knn_train_r2 > 0.98 and knn_r2 < 0.85:
    print("【诊断】警告：模型依然严重过拟合。可能是特征维度太高，或者 weights='distance' 导致的。")
elif knn_train_r2 - knn_r2 < 0.1:
    print("【诊断】状态良好：训练集和测试集差距不大，过拟合已控制。")
else:
    print("【诊断】中度过拟合，可尝试进一步增大 n_neighbors。")

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import mean_squared_error, r2_score



rf_model = RandomForestRegressor(random_state=42, oob_score=True)

# ---------------------------------------------------------

# ---------------------------------------------------------
rf_param_grid = {

    'n_estimators': [100, 200, 300],




    'max_depth': [5, 8, 10],




    'min_samples_split': [5, 10, 15],




    'min_samples_leaf': [ 4, 8,12],





    'max_features': ['sqrt', 0.3, 0.5],



    'bootstrap': [True]
}


rf_cv = KFold(n_splits=5, shuffle=True, random_state=42)


rf_grid_search = GridSearchCV(
    estimator=rf_model,
    param_grid=rf_param_grid,
    cv=rf_cv,
    scoring='neg_mean_squared_error',
    n_jobs=-1,
    verbose=1
)


print("Fitting Random Forest GridSearchCV...")
rf_grid_search.fit(X_train, y2_train)

# ---------------------------------------------------------

# ---------------------------------------------------------
print("\nRF Best parameters found: ", rf_grid_search.best_params_)

rf_best_model = rf_grid_search.best_estimator_
rf_y_pred = rf_best_model.predict(X_test)


rf_rmse = np.sqrt(mean_squared_error(y2_test, rf_y_pred))
rf_r2 = r2_score(y2_test, rf_y_pred)

print(f"RF Test RMSE: {rf_rmse:.4f}")
print(f"RF Test R²:   {rf_r2:.4f}")


rf_train_pred = rf_best_model.predict(X_train)
rf_train_r2 = r2_score(y2_train, rf_train_pred)
print(f"RF Train R²:  {rf_train_r2:.4f}")

print("-" * 30)
if rf_train_r2 - rf_r2 > 0.1:
    print("【诊断】：模型可能仍然有点过拟合，请尝试增大 min_samples_leaf 或减小 max_depth")
else:
    print("【诊断】：过拟合控制得不错，泛化能力较强")

import optuna
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_squared_error, r2_score
import warnings


warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ---------------------------------------------------------

# ---------------------------------------------------------
def objective(trial):



    param = {

        'random_state': 42,
        'n_jobs': -1,
        'booster': 'gbtree',


        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.05), 
        'n_estimators': trial.suggest_int('n_estimators', 300, 500),


        'max_depth': trial.suggest_int('max_depth',1, 3,),



        'min_child_weight': trial.suggest_int('min_child_weight', 15, 30),



        'gamma': trial.suggest_float('gamma', 2.0, 4.0),


        'subsample': trial.suggest_float('subsample', 0.6, 0.8),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 0.8),



        'reg_alpha': trial.suggest_float('reg_alpha', 0.1, 2.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1.0, 5.0, log=True)
    }


    model = XGBRegressor(**param)



    cv = KFold(n_splits=5, shuffle=True, random_state=42)



    scores = cross_val_score(model, X_train, y2_train, cv=cv, scoring='neg_mean_squared_error', n_jobs=-1)

    return -scores.mean()

# ---------------------------------------------------------

# ---------------------------------------------------------
print("开始 XGBoost Optuna 贝叶斯搜索 (Trying 50 times)...")
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=500)

# ---------------------------------------------------------

# ---------------------------------------------------------
print("\nBest CV MSE:", study.best_value)
print("Best params:", study.best_params)



best_params = study.best_params

best_params['random_state'] = 42
best_params['n_jobs'] = -1


xgb_best_model = XGBRegressor(**best_params)

xgb_best_model.fit(X_train, y2_train)

# ---------------------------------------------------------

# ---------------------------------------------------------
xgb_y_pred = xgb_best_model.predict(X_test)


xgb_rmse = np.sqrt(mean_squared_error(y2_test, xgb_y_pred))
xgb_r2 = r2_score(y2_test, xgb_y_pred)

print("-" * 30)
print(f"XGBoost Test RMSE: {xgb_rmse:.4f}")
print(f"XGBoost Test R²:   {xgb_r2:.4f}")


xgb_train_r2 = r2_score(y2_train, xgb_best_model.predict(X_train))
print(f"XGBoost Train R²:  {xgb_train_r2:.4f}")
print("-" * 30)


diff = xgb_train_r2 - xgb_r2
if diff > 0.1:
    print(f"【诊断】(差值 {diff:.2f}) 还是有点过拟合。")
    print("建议：在 objective 函数中调大 min_child_weight 的下限 (例如 20-30) 或调大 gamma (例如 3-5)")
else:
    print(f"【诊断】(差值 {diff:.2f}) 过拟合控制良好！")

from lightgbm import LGBMRegressor


lgbm_model = LGBMRegressor(random_state=42, verbose=-1)

lgbm_param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [2,3],
    'learning_rate': [0.01, 0.1],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0],
    'min_child_samples': [10, 20],
    'reg_alpha': [0, 0.1],
    'reg_lambda': [0, 0.1]
}


lgbm_cv = KFold(n_splits=5, shuffle=True, random_state=42)

lgbm_grid_search = GridSearchCV(
    estimator=lgbm_model,
    param_grid=lgbm_param_grid,
    cv=lgbm_cv,
    scoring='neg_mean_squared_error',
    n_jobs=-1,
    verbose=2
)

print("Fitting LGBM GridSearchCV...")
lgbm_grid_search.fit(X_train, y2_train)

print("LGBM Best parameters found: ", lgbm_grid_search.best_params_)

lgbm_best_score = -lgbm_grid_search.best_score_
print("LGBM Best CV MSE: ", lgbm_best_score)

lgbm_best_model = lgbm_grid_search.best_estimator_
lgbm_y_pred = lgbm_best_model.predict(X_test)

lgbm_mse = mean_squared_error(y2_test, lgbm_y_pred)
lgbm_rmse = np.sqrt(lgbm_mse)
lgbm_r2 = r2_score(y2_test, lgbm_y_pred)
print("LGBM Test RMSE: ", lgbm_rmse)
print("LGBM Test R²: ", lgbm_r2)

import optuna
import numpy as np
from catboost import CatBoostRegressor
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_squared_error, r2_score
import warnings

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ---------------------------------------------------------

# ---------------------------------------------------------
def objective(trial):

    param = {

        'loss_function': 'RMSE',
        'random_state': 42,
        'verbose': 0,
        'bootstrap_type': 'Bernoulli',






        'depth': trial.suggest_int('depth', 1, 3),



        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'iterations': trial.suggest_int('iterations', 500, 1000),



        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 10, 15.0, log=True),



        'subsample': trial.suggest_float('subsample', 0.6, 0.8),
        'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.6, 0.8),



        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 10,20),

        # 6. Random Strength

        'random_strength': trial.suggest_float('random_strength', 1.0, 5.0)
    }


    model = CatBoostRegressor(**param)


    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X_train, y2_train, cv=cv, scoring='neg_mean_squared_error', n_jobs=-1)

    return -scores.mean()
4
# ---------------------------------------------------------

# ---------------------------------------------------------
print("开始 CatBoost Optuna 贝叶斯搜索...")
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=200)

print("-" * 30)
print("Best CV MSE:", study.best_value)
print("Best parameters found:", study.best_params)

# ---------------------------------------------------------

# ---------------------------------------------------------

best_params = study.best_params


final_params = {
    'loss_function': 'RMSE',
    'random_state': 42,
    'verbose': 0,
    'bootstrap_type': 'Bernoulli',
    **best_params
}


catboost_best_model = CatBoostRegressor(**final_params)
catboost_best_model.fit(X_train, y2_train)
print("catboost_best_model 已重新训练完成。")

# ---------------------------------------------------------

# ---------------------------------------------------------
catboost_y_pred = catboost_best_model.predict(X_test)
catboost_train_pred = catboost_best_model.predict(X_train)


catboost_rmse = np.sqrt(mean_squared_error(y2_test, catboost_y_pred))
catboost_r2 = r2_score(y2_test, catboost_y_pred)
catboost_train_r2 = r2_score(y2_train, catboost_train_pred)

print("-" * 30)
print(f"CatBoost Test RMSE: {catboost_rmse:.4f}")
print(f"CatBoost Test R²:   {catboost_r2:.4f}")
print(f"CatBoost Train R²:  {catboost_train_r2:.4f}")
print("-" * 30)


diff = catboost_train_r2 - catboost_r2
if diff > 0.1:
    print(f"【诊断】(差值 {diff:.2f}) 存在过拟合风险。")
    print("建议：调大 l2_leaf_reg 下限，或调大 min_data_in_leaf。")
else:
    print(f"【诊断】(差值 {diff:.2f}) 过拟合控制良好。")

import optuna
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import warnings


warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ---------------------------------------------------------

# ---------------------------------------------------------
def objective(trial):



    C = trial.suggest_float("svm__C", 10, 100.0, log=True)



    gamma = trial.suggest_float("svm__gamma", 0.001, 0.8, log=True)


    epsilon = trial.suggest_float("svm__epsilon", 0.01, 2)


    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('svm', SVR(C=C, gamma=gamma, epsilon=epsilon))
    ])


    scores = cross_val_score(pipeline, X_train, y2_train, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)

    return -scores.mean()

# ---------------------------------------------------------

# ---------------------------------------------------------

print("开始 SVM Optuna 贝叶斯搜索...")
study = optuna.create_study(direction="minimize") 
study.optimize(objective, n_trials=300) 

print("-" * 30)
print("Best CV MSE:", study.best_value)
print("Best parameters found:", study.best_params)

# ---------------------------------------------------------


# ---------------------------------------------------------
best_params = study.best_params

svm_best_model = Pipeline([
    ('scaler', StandardScaler()),
    ('svm', SVR(
        C=best_params["svm__C"], 
        gamma=best_params["svm__gamma"], 
        epsilon=best_params["svm__epsilon"]
    ))
])


svm_best_model.fit(X_train, y2_train)
print("svm_best_model 已重新训练完成。")

# ---------------------------------------------------------

# ---------------------------------------------------------
svm_y_pred = svm_best_model.predict(X_test)
svm_train_pred = svm_best_model.predict(X_train)


svm_rmse = np.sqrt(mean_squared_error(y2_test, svm_y_pred))
svm_r2 = r2_score(y2_test, svm_y_pred)
svm_train_r2 = r2_score(y2_train, svm_train_pred)

print("-" * 30)
print(f"SVM Test RMSE: {svm_rmse:.4f}")
print(f"SVM Test R²:   {svm_r2:.4f}")
print(f"SVM Train R²:  {svm_train_r2:.4f}")
print("-" * 30)


diff = svm_train_r2 - svm_r2

if diff > 0.15:
    print(f"【严重警告】检测到过拟合！(Train - Test = {diff:.2f})")
    print("建议：")
    print(f"1. 将 objective 函数中 C 的上限降至 {best_params['svm__C'] * 0.5:.1f} 以下")
    print(f"2. 将 gamma 的上限降至 {best_params['svm__gamma'] * 0.5:.3f} 以下")
elif diff > 0.1:
    print(f"【注意】存在轻微过拟合 (Train - Test = {diff:.2f})")
else:
    print(f"【成功】模型泛化能力良好 (Train - Test = {diff:.2f})")

import seaborn as sns


models_info = [
    ("KNN", y2_train, knn_best_model.predict(X_train), y2_test, knn_y_pred, r2_score(y2_train, knn_best_model.predict(X_train)), knn_r2),
    ("RF", y2_train, rf_best_model.predict(X_train), y2_test, rf_y_pred, r2_score(y2_train, rf_best_model.predict(X_train)), rf_r2),
    ("XGBoost", y2_train, xgb_best_model.predict(X_train), y2_test, xgb_y_pred, r2_score(y2_train, xgb_best_model.predict(X_train)), xgb_r2),
    ("LGBM", y2_train, lgbm_best_model.predict(X_train), y2_test, lgbm_y_pred, r2_score(y2_train, lgbm_best_model.predict(X_train)), lgbm_r2),
    ("CatBoost", y2_train, catboost_best_model.predict(X_train), y2_test, catboost_y_pred, r2_score(y2_train, catboost_best_model.predict(X_train)), catboost_r2),
    #("MLP", y2_train, mlp_best_model.predict(X_train), y2_test, mlp_y_pred, r2_score(y2_train, mlp_best_model.predict(X_train)), mlp_r2),
    ("SVM", y2_train, svm_best_model.predict(X_train), y2_test, svm_y_pred, r2_score(y2_train, svm_best_model.predict(X_train)), svm_r2)
]

palette = {'Train': '#b4d4e1', 'Test': '#f4ba8a'}
fig, axes = plt.subplots(2, 3, figsize=(18, 10), dpi=300)
axes = axes.flatten()

for i, (model_name, y_train_true, y_train_pred, y_test_true, y_test_pred, r2_train, r2_test) in enumerate(models_info):

    df_train = pd.DataFrame({'True': y_train_true, 'Predicted': y_train_pred, 'Data Set': 'Train'})
    df_test = pd.DataFrame({'True': y_test_true, 'Predicted': y_test_pred, 'Data Set': 'Test'})
    df_all = pd.concat([df_train, df_test])

    ax = axes[i]
    sns.scatterplot(data=df_all, x="True", y="Predicted", hue="Data Set", palette=palette, alpha=0.5, ax=ax)
    sns.regplot(data=df_train, x="True", y="Predicted", scatter=False, ax=ax, color=palette['Train'], label='Train Regression Line')
    sns.regplot(data=df_test, x="True", y="Predicted", scatter=False, ax=ax, color=palette['Test'], label='Test Regression Line')


    min_val = min(df_all['True'].min(), df_all['Predicted'].min())
    max_val = max(df_all['True'].max(), df_all['Predicted'].max())
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.6)


    ax.set_title(f"{model_name} Model", fontsize=14)
    ax.text(0.95, 0.15, f"Train $R^2$ = {r2_train:.3f}", transform=ax.transAxes,
            fontsize=10, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="white"))
    ax.text(0.95, 0.08, f"Test $R^2$ = {r2_test:.3f}", transform=ax.transAxes,
            fontsize=10, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="white"))
    ax.set_xlabel("DFT Calculation")
    ax.set_ylabel("Predicted")
    ax.legend()

plt.tight_layout()
plt.savefig("All_Models_True_vs_Predicted.pdf", format='pdf', bbox_inches='tight',dpi=1200)
plt.show()

from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import LinearRegression


base_learners = [
    ("KNN", knn_best_model),
    ("RF", rf_best_model),
    ("XGB", xgb_best_model),
    ("LGBM", lgbm_best_model),
    ("CatBoost", catboost_best_model),
    ("SVM", svm_best_model)
    #("MLP", mlp_best_model)
]


meta_model = LinearRegression()
#meta_model = XGBRegressor()

stacking_regressor = StackingRegressor(
    estimators=base_learners,
    final_estimator=meta_model,
    cv=5,
    n_jobs=-1,
    passthrough=False
)


print("Training StackingRegressor...")
stacking_regressor.fit(X_train, y2_train)

from sklearn import metrics


y_train_true = y2_train.values
y_test_true = y2_test.values


y2_pred_train = stacking_regressor.predict(X_train)
y2_pred_test = stacking_regressor.predict(X_test)


mse_train = metrics.mean_squared_error(y_train_true, y2_pred_train)
rmse_train = np.sqrt(mse_train)
mae_train = metrics.mean_absolute_error(y_train_true, y2_pred_train)
r2_train = metrics.r2_score(y_train_true, y2_pred_train)


mse_test = metrics.mean_squared_error(y_test_true, y2_pred_test)
rmse_test = np.sqrt(mse_test)
mae_test = metrics.mean_absolute_error(y_test_true, y2_pred_test)
r2_test = metrics.r2_score(y_test_true, y2_pred_test)




print("训练集评价指标:")
print("均方误差 (MSE):", mse_train)
print("均方根误差 (RMSE):", rmse_train)
print("平均绝对误差 (MAE):", mae_train)
print("拟合优度 (R-squared):", r2_train)

print("\n测试集评价指标:")
print("均方误差 (MSE):", mse_test)
print("均方根误差 (RMSE):", rmse_test)
print("平均绝对误差 (MAE):", mae_test)
print("拟合优度 (R-squared):", r2_test)





import numpy as np
import pandas as pd


residuals_train = y_train_true - y2_pred_train
residuals_test = y_test_true - y2_pred_test


abs_error_train = np.abs(residuals_train)
abs_error_test = np.abs(residuals_test)


train_errors_df = pd.DataFrame({
    "True": y_train_true,
    "Pred": y2_pred_train,
    "Residual": residuals_train,
    "AbsError": abs_error_train
})

test_errors_df = pd.DataFrame({
    "True": y_test_true,
    "Pred": y2_pred_test,
    "Residual": residuals_test,
    "AbsError": abs_error_test
})


print("训练集误差最大的样本:")
print(train_errors_df.sort_values(by="AbsError", ascending=False).head(10))

print("\n测试集误差最大的样本:")
print(test_errors_df.sort_values(by="AbsError", ascending=False).head(10))

import scipy.stats as stats

scale_factor = 1.5
confidence = 0.95

z_train = np.polyfit(y2_train, y2_pred_train, 1)
p_train = np.poly1d(z_train)
predicted_values_train = p_train(y1_train)
residuals_train = y2_pred_train - predicted_values_train
mean_error_train = np.mean(residuals_train**2)
t_value_train = stats.t.ppf((1 + confidence) / 2., len(y2_train) - 1)
ci_train = t_value_train * scale_factor * np.sqrt(mean_error_train) * np.sqrt(1 / len(y2_train) + (y2_train - np.mean(y2_train))**2 / np.sum((y2_train - np.mean(y2_train))**2))
x_extended_train = np.linspace(min(y2_train), max(y2_train), 100)
predicted_extended_train = p_train(x_extended_train)
ci_extended_train = t_value_train * scale_factor * np.sqrt(mean_error_train) * np.sqrt(1 / len(y2_train) + (x_extended_train - np.mean(y2_train))**2 / np.sum((y2_train - np.mean(y2_train))**2))


z_test = np.polyfit(y2_test, y2_pred_test, 1)
p_test = np.poly1d(z_test)
predicted_values_test = p_test(y2_test)
residuals_test = y2_pred_test - predicted_values_test
mean_error_test = np.mean(residuals_test**2)
t_value_test = stats.t.ppf((1 + confidence) / 2., len(y2_test) - 1)
ci_test = t_value_test * scale_factor * np.sqrt(mean_error_test) * np.sqrt(1 / len(y2_test) + (y2_test - np.mean(y2_test))**2 / np.sum((y2_test - np.mean(y2_test))**2))
x_extended_test = np.linspace(min(y2_test), max(y2_test), 100)
predicted_extended_test = p_test(x_extended_test)
ci_extended_test = t_value_test * scale_factor * np.sqrt(mean_error_test) * np.sqrt(1 / len(y1_test) + (x_extended_test - np.mean(y1_test))**2 / np.sum((y1_test - np.mean(y1_test))**2))



train_color = '#1f77b4'
test_color = '#ff7f0e'

confidence_train_color = '#aec7e8'
confidence_test_color = '#ffbb78'


fig = plt.figure(figsize=(8, 7), dpi=1200)
gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)
ax_main = fig.add_subplot(gs[1:, :-1])
ax_hist_x = fig.add_subplot(gs[0, :-1], sharex=ax_main)
ax_hist_y = fig.add_subplot(gs[1:, -1], sharey=ax_main)


ax_main.scatter(y2_train, y2_pred_train, color=train_color, label="Training Predicted Values", alpha=0.6)
ax_main.plot(y2_train, p_train(y2_train), color=train_color, alpha=0.9, label=f"Training Line of Best Fit\n$R^2$ = {r2_train:.2f}, MAE = {mae_train:.2f}")
#ax_main.fill_between(x_extended_train, predicted_extended_train - ci_extended_train, predicted_extended_train + ci_extended_train, 
#                     color=confidence_train_color, alpha=0.5, label="Training 95% Confidence Interval")


ax_main.scatter(y2_test, y2_pred_test, color=test_color, label="Testing Predicted Values", alpha=0.6)
ax_main.plot(y2_test, p_test(y2_test), color=test_color, alpha=0.9, label=f"Testing Line of Best Fit\n$R^2$ = {r2_test:.2f}, MAE = {mae_test:.2f}")
#ax_main.fill_between(x_extended_test, predicted_extended_test - ci_extended_test, predicted_extended_test + ci_extended_test, 
#                     color=confidence_test_color, alpha=0.5, label="Testing 95% Confidence Interval")


ax_main.plot([min(y2_train.min(), y2_test.min()), max(y2_train.max(), y2_test.max())], 
             [min(y2_train.min(), y2_test.min()), max(y2_train.max(), y2_test.max())], 
             color='grey', linestyle='--', alpha=0.6, label="1:1 Line")


ax_main.set_xlabel("Observed Values", fontsize=12)
ax_main.set_ylabel("Predicted Values", fontsize=12)
ax_main.legend(loc="upper left", fontsize=10)


ax_hist_x.hist(y2_train, bins=20, color=train_color, alpha=0.7, edgecolor='black', label="Training Observed Distribution")
ax_hist_x.hist(y2_test, bins=20, color=test_color, alpha=0.7, edgecolor='black')
ax_hist_x.tick_params(labelbottom=False)


ax_hist_y.hist(y2_pred_train, bins=20, orientation='horizontal', color=train_color, alpha=0.7, edgecolor='black')
ax_hist_y.hist(y2_pred_test, bins=20, orientation='horizontal', color=test_color, alpha=0.7, edgecolor='black')
ax_hist_y.set_xlabel("Frequency", fontsize=12)
ax_hist_y.tick_params(labelleft=False)
ax_hist_y.grid(False)

plt.savefig('train_test_combined_with_histograms_and_confidence_intervals_new_colors.pdf', format='pdf', bbox_inches='tight')
plt.show()

import scipy.stats as stats

scale_factor = 1.5
confidence = 0.95

z_train = np.polyfit(y2_train, y2_pred_train, 1)
p_train = np.poly1d(z_train)
predicted_values_train = p_train(y2_train)
residuals_train = y2_pred_train - predicted_values_train
mean_error_train = np.mean(residuals_train**2)
t_value_train = stats.t.ppf((1 + confidence) / 2., len(y2_train) - 1)
ci_train = t_value_train * scale_factor * np.sqrt(mean_error_train) * np.sqrt(1 / len(y2_train) + (y2_train - np.mean(y2_train))**2 / np.sum((y2_train - np.mean(y2_train))**2))
x_extended_train = np.linspace(min(y2_train), max(y2_train), 100)
predicted_extended_train = p_train(x_extended_train)
ci_extended_train = t_value_train * scale_factor * np.sqrt(mean_error_train) * np.sqrt(1 / len(y2_train) + (x_extended_train - np.mean(y2_train))**2 / np.sum((y2_train - np.mean(y2_train))**2))


z_test = np.polyfit(y2_test, y2_pred_test, 1)
p_test = np.poly1d(z_test)
predicted_values_test = p_test(y2_test)
residuals_test = y2_pred_test - predicted_values_test
mean_error_test = np.mean(residuals_test**2)
t_value_test = stats.t.ppf((1 + confidence) / 2., len(y2_test) - 1)
ci_test = t_value_test * scale_factor * np.sqrt(mean_error_test) * np.sqrt(1 / len(y2_test) + (y2_test - np.mean(y2_test))**2 / np.sum((y2_test - np.mean(y2_test))**2))
x_extended_test = np.linspace(min(y2_test), max(y2_test), 100)
predicted_extended_test = p_test(x_extended_test)
ci_extended_test = t_value_test * scale_factor * np.sqrt(mean_error_test) * np.sqrt(1 / len(y1_test) + (x_extended_test - np.mean(y1_test))**2 / np.sum((y1_test - np.mean(y1_test))**2))



train_color = '#1f77b4'
test_color = '#ff7f0e'

confidence_train_color = '#aec7e8'
confidence_test_color = '#ffbb78'


fig = plt.figure(figsize=(8, 7), dpi=300)
gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)
ax_main = fig.add_subplot(gs[1:, :-1])
ax_hist_x = fig.add_subplot(gs[0, :-1], sharex=ax_main)
ax_hist_y = fig.add_subplot(gs[1:, -1], sharey=ax_main)


ax_main.scatter(y2_train, y2_pred_train, color=train_color, label="Training Predicted Values", alpha=0.6)
ax_main.plot(y2_train, p_train(y2_train), color=train_color, alpha=0.9, label=f"Train Regression Line\n$R^2$ = {r2_train:.2f}, MAE = {mae_train:.2f}")
#ax_main.fill_between(x_extended_train, predicted_extended_train - ci_extended_train, predicted_extended_train + ci_extended_train, 
#                     color=confidence_train_color, alpha=0.5, label="Training 95% Confidence Interval")

ax_main.set_xlim([-10, 260])
ax_main.set_ylim([-10, 260])


ax_main.scatter(y2_test, y2_pred_test, color=test_color, label="Testing Predicted Values", alpha=0.6)
ax_main.plot(y2_test, p_test(y2_test), color=test_color, alpha=0.9, label=f"Test Regression Line\n$R^2$ = {r2_test:.2f}, MAE = {mae_test:.2f}")
#ax_main.fill_between(x_extended_test, predicted_extended_test - ci_extended_test, predicted_extended_test + ci_extended_test, 
#                     color=confidence_test_color, alpha=0.5, label="Testing 95% Confidence Interval")

ax_main.text(-0.2, 1.05, "(c)", transform=ax_main.transAxes,
             fontsize=20, va='top', ha='left')#fontweight="bold",


ax_main.plot([min(y1_train.min(), y1_test.min()), max(y1_train.max(), y1_test.max())], 
             [min(y1_train.min(), y1_test.min()), max(y1_train.max(), y1_test.max())], 
             color='grey', linestyle='--', alpha=0.6, label="1:1 Line")


ax_main.set_xlabel("DFT Values(Shear Modulus)", fontsize=20)
ax_main.set_ylabel("Predicted Values", fontsize=20)
ax_main.legend(loc="upper left", fontsize=10)
ax_main.tick_params(axis='both', labelsize=16)


ax_hist_x.hist(y2_train, bins=20, color=train_color, alpha=0.7, edgecolor='black', label="Training Observed Distribution")
ax_hist_x.hist(y2_test, bins=20, color=test_color, alpha=0.7, edgecolor='black')
ax_hist_x.tick_params(labelbottom=False, labelsize=15)


ax_hist_y.hist(y2_pred_train, bins=20, orientation='horizontal', color=train_color, alpha=0.7, edgecolor='black')
ax_hist_y.hist(y2_pred_test, bins=20, orientation='horizontal', color=test_color, alpha=0.7, edgecolor='black')
ax_hist_y.set_xlabel("Frequency", fontsize=15)
ax_hist_y.tick_params(labelleft=False, labelsize=15)
ax_hist_y.grid(False)

plt.savefig('train_test_combined_with_histograms_and_confidence_intervals_new_colors.pdf', format='pdf', bbox_inches='tight')
plt.show()

knn_mse = mean_squared_error(y2_test, knn_y_pred)
knn_rmse = np.sqrt(knn_mse)
knn_r2 = r2_score(y2_test, knn_y_pred)
rf_mse = mean_squared_error(y2_test, rf_y_pred)
rf_rmse = np.sqrt(rf_mse)
rf_r2 = r2_score(y2_test, rf_y_pred)
train_results_df = pd.DataFrame(train_results)
train_results.append({
        "R²": r2_train,
        "RMSE": rmse_train,
        "MAE": mae_train,
        "MAPE": mape_train,
        "EV": ev_train,
        "Model": model_name
    })


from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, explained_variance_score


train_results = []
test_results = []


def evaluate_model(model_name, model, X_train, y_train, X_test, y_test, cv=5):

    kf = KFold(n_splits=cv, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_train, y_train, cv=kf, scoring='r2')
    mean_cv_r2 = np.mean(cv_scores)
    std_cv_r2 = np.std(cv_scores)


    model.fit(X_train, y_train)


    y_pred_test = model.predict(X_test)

    y_pred_train = model.predict(X_train)


    r2_test = r2_score(y_test, y_pred_test)
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
    mae_test = mean_absolute_error(y_test, y_pred_test)
    mape_test = np.mean(np.abs((y_test - y_pred_test) / y_test)) if np.all(y_test != 0) else np.nan
    ev_test = explained_variance_score(y_test, y_pred_test)
    test_results.append({
        #"CV Mean R²": mean_cv_r2,
        #"CV Std R²": std_cv_r2,
        "R²": r2_test,
        "RMSE": rmse_test,
        "MAE": mae_test,
        #"MAPE": mape_test,
        #"EV": ev_test,
        "Model": model_name
    })


    r2_train = r2_score(y_train, y_pred_train)
    rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
    mae_train = mean_absolute_error(y_train, y_pred_train)
    mape_train = np.mean(np.abs((y_train - y_pred_train) / y_train)) if np.all(y_train != 0) else np.nan
    ev_train = explained_variance_score(y_train, y_pred_train)
    train_results.append({
        "R²": r2_train,
        "RMSE": rmse_train,
        "MAE": mae_train,
        #"MAPE": mape_train,
        #"EV": ev_train,
        "Model": model_name
    })



evaluate_model("Random Forest", rf_best_model, X_train, y2_train, X_test, y2_test)
evaluate_model("XGBoost", xgb_best_model, X_train, y2_train, X_test, y2_test)
evaluate_model("LightGBM", lgbm_best_model, X_train, y2_train, X_test, y2_test)
evaluate_model("KNN", knn_best_model, X_train, y2_train, X_test, y2_test)
evaluate_model("SVM", svm_best_model, X_train, y2_train, X_test, y2_test)
#evaluate_model("Gradient Boosting", GradientBoostingRegressor(random_state=42), X_train, y_train, X_test, y_test)
#evaluate_model("AdaBoost", AdaBoostRegressor(random_state=42), X_train, y_train, X_test, y_test)
evaluate_model("CatBoost", catboost_best_model, X_train, y2_train, X_test, y2_test)

train_results_df = pd.DataFrame(train_results)

import seaborn as sns
def plot_model_performance(results_df, dataset_type="Train", save_path=None):
    """
    Plot performance metrics for a set of models.

    Parameters:
    - results_df: DataFrame with model metrics (train_results_df or test_results_df)
    - dataset_type: dataset label ("Train" or "Test")
    - save_path: optional output path
    """
    colors = sns.color_palette("Set2", len(results_df))
    long_format = results_df.melt(
        id_vars=["Model"], 
        var_name="Metric", 
        value_name="Value"
    )

    plt.figure(figsize=(12, 6))
    sns.barplot(data=long_format, x="Metric", y="Value", hue="Model", palette=colors)
    plt.title(f"{dataset_type} Set Performance Metrics", fontsize=16)
    plt.ylabel("Value", fontsize=12)
    plt.xlabel("Metrics", fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.ylim(0, 12)
    plt.legend(title="Models", bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    if save_path:
        plt.savefig(save_path, format="pdf", bbox_inches="tight", dpi=300)
    plt.tight_layout()
    plt.show()



plot_model_performance(train_results_df, dataset_type="Train", save_path="train_metrics.pdf")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_model_performance_dual_y(
    results_df: pd.DataFrame,
    dataset_type: str = "Train",
    r2_like_keys = ("r2", "r^2", "r2_score"),
    r2_metrics: list | None = None,
    save_path: str | None = None
):
    """
    Dual-axis bar chart: errors on the left axis, R2-like metrics on the right.

    Parameters
    ----------
    results_df : DataFrame
        Example:
        Model | MAE | RMSE | R2
        ----- | --- | ---- | ---
        RF    | ... | ...  | ...
        XGB   | ... | ...  | ...
    dataset_type : str
        "Train" or "Test", used in the title.
    r2_like_keys : tuple[str]
        Keywords to detect R2-like columns (case-insensitive).
    r2_metrics : list[str] | None
        If provided, use these columns as R2-like metrics.
    save_path : str | None
        Output path (e.g., "metrics_dual_axis.pdf").
    """
    assert "Model" in results_df.columns, "results_df 必须包含 'Model' 列"
    models = results_df["Model"].tolist()
    metrics_all = [c for c in results_df.columns if c != "Model"]


    if r2_metrics is None:
        r2_cols = [m for m in metrics_all if any(k in m.lower() for k in r2_like_keys)]
    else:
        r2_cols = list(r2_metrics)


    left_cols = [m for m in metrics_all if m not in r2_cols]

    if len(left_cols) == 0 or len(r2_cols) == 0:
        raise ValueError(
            "需要同时存在左轴(误差类)与右轴(R²类)指标。"
            f"当前 left_cols={left_cols}, r2_cols={r2_cols}"
        )


    metrics_order = left_cols + r2_cols

    n_models = len(models)
    n_groups = len(metrics_order)
    bar_width = 0.8 / max(n_models, 1)

    palette = sns.color_palette("Set2", n_models)

    fig, ax_left = plt.subplots(figsize=(8, 5))
    ax_right = ax_left.twinx()


    legend_handles = []


    for j, model in enumerate(models):
        color = palette[j]

        legend_label = model


        for gi, metric in enumerate(metrics_order):

            value = results_df.loc[results_df["Model"] == model, metric].values[0]

            x = gi + j * bar_width - (n_models - 1) * bar_width / 2

            if metric in r2_cols:

                bar = ax_right.bar(
                    x, value,
                    width=bar_width,
                    color=color,
                    alpha=0.45,
                    edgecolor="black",
                    linewidth=0.7
                )
            else:

                bar = ax_left.bar(
                    x, value,
                    width=bar_width,
                    color=color,
                    alpha=0.90
                )

                if gi == 0:
                    legend_handles.append(bar[0])


    ax_left.set_xticks(range(n_groups))
    ax_left.set_xticklabels(metrics_order, rotation=0, ha="right", fontsize=20)


    ax_left.set_xlabel("Metrics", fontsize=15)
    ax_left.set_ylabel("MAE/RMSE metrics ", fontsize=15)
    ax_right.set_ylabel("R² metrics", fontsize=15)
    ax_left.set_title(f"{dataset_type} Set Performance Metrics of Shear Modulus", fontsize=20)


    ax_left.grid(axis="y", linestyle="--", alpha=0.4)

    ax_left.set_ylim(0, 25)
    ax_left.set_yticks(np.arange(0, 25, 5))
    ax_left.set_yticklabels([f"{v:.2f}" for v in np.arange(0, 25, 5)], fontsize=15)

    ax_right.set_ylim(0.75, 0.92)

    ax_right.set_yticks(np.arange(0.8, 0.9, 0.02))
    ax_right.set_yticklabels([f"{v:.2f}" for v in np.arange(0.8, 0.9, 0.02)], fontsize=15)


    ax_left.legend(legend_handles, models, bbox_to_anchor=(1.1, 1), loc="upper left", fontsize=10)

    plt.tight_layout()
    #if save_path:
    #    plt.savefig(save_path, format="pdf", bbox_inches="tight", dpi=300)

    ax_left.text(0.05, 0.95, "(d)", transform=ax_left.transAxes, fontsize=20, color='black', va='top', ha='left')

    plt.show()


plot_model_performance_dual_y(
    train_results_df,
    dataset_type="Train",
    r2_metrics=["R²"],
)

plot_model_performance_dual_y(
    test_results_df,
    dataset_type="Test",
    r2_metrics=["R²"],
)

test_results_df = pd.DataFrame(test_results)


plot_model_performance(test_results_df.iloc[:,2::], dataset_type="Test", save_path="test_metrics.pdf")

import sys
import os

sys.stdout = open(os.devnull, 'w')
import shap



shap_dfs = {}


for name, model in stacking_regressor.named_estimators_.items():
    try:
        if not hasattr(model, "predict"):
            continue

        print(f"Computing SHAP for model: {name}")


        if isinstance(model, (RandomForestRegressor,XGBRegressor, LGBMRegressor, CatBoostRegressor)):
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_test)
        else:
            explainer = shap.KernelExplainer(model.predict, X_test)
            shap_values = explainer.shap_values(X_test)


        shap_df = pd.DataFrame(shap_values, columns=X_test.columns)
        shap_dfs[name] = shap_df

    except Exception as e:

        sys.stdout = original_stdout
        print(f"[Warning] SHAP failed for model '{name}': {e}")
        sys.stdout = open(os.devnull, 'w')

print("SHAP analysis completed for models:", list(shap_dfs.keys()))

import sys
import os
import shap
import pandas as pd


original_stdout = sys.stdout


shap_dfs = {}

for name, model in stacking_regressor.named_estimators_.items():
    try:
        if not hasattr(model, "predict"):
            continue


        sys.stdout = original_stdout
        print(f"Computing SHAP for model: {name}")


        sys.stdout = open(os.devnull, 'w')


        if isinstance(model, (RandomForestRegressor, XGBRegressor, LGBMRegressor, CatBoostRegressor)):
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_test)
        else:
            explainer = shap.KernelExplainer(model.predict, X_test)
            shap_values = explainer.shap_values(X_test)


        sys.stdout = original_stdout

        shap_df = pd.DataFrame(shap_values, columns=X_test.columns)
        shap_dfs[name] = shap_df

    except Exception as e:

        sys.stdout = original_stdout
        print(f"[Warning] SHAP failed for model '{name}': {e}")

sys.stdout = original_stdout

print("SHAP analysis completed for models:", list(shap_dfs.keys()))

fig, axes = plt.subplots(2, 3, figsize=(15, 10))  
axes = axes.flatten()  

for i, (name, shap_df) in enumerate(shap_dfs.items()):
    try:
        if i >= len(axes):  
            break
        plt.sca(axes[i])  
        show_color_bar = (i % 3 == 2)  
        shap.summary_plot(
            shap_values=shap_df.to_numpy(),
            features=X_test,
            feature_names=X_test.columns,
            plot_type="dot",
            show=False,
            color_bar=show_color_bar  
        )
        plt.xlabel('')  
        axes[i].set_title(name, fontsize=10)  
        axes[i].tick_params(axis='y', labelsize=8)  
        if i % 3 != 0:  # 
            axes[i].set_ylabel("")
    except Exception as e:
        print(f"Error plotting SHAP summary for model {name}: {e}")
fig.text(0.5, 0.004, "SHAP value (impact on model output)", ha="center", fontsize=12)
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])
plt.tight_layout(rect=[0, 0, 1, 1])
fig.suptitle("SHAP Feature Importance Analysis of Base Learners in the First Layer of Stacking Model", fontsize=16, y=1.02)
plt.savefig("summary_plot.pdf", format='pdf', bbox_inches='tight')
plt.show()

import shap

# build a list of arrays and a matching list of names
shap_arrays = [df.values for df in shap_dfs.values()]
model_names = list(shap_dfs.keys())

# single call to summary_plot in classification‐mode:
shap.summary_plot(
    shap_values=shap_arrays,
    features=    X_test,
    feature_names=X_test.columns,
    class_names= model_names,    # titles each subplot
    plot_type=   "bar",
    show=        True
)


fig, axes = plt.subplots(2, 3, figsize=(15, 10))  
axes = axes.flatten()  

for i, (name, shap_df) in enumerate(shap_dfs.items()):
    try:
        if i >= len(axes):  
            break
        plt.sca(axes[i])  
        show_color_bar = (i % 3 == 2)  
        shap.summary_plot(
            shap_values=shap_df.to_numpy(),
            features=X_test,
            feature_names=X_test.columns,
            plot_type="dot",
            show=False,
            color_bar=show_color_bar  
        )
        plt.xlabel('')  
        axes[i].set_title(name, fontsize=10)  
        axes[i].tick_params(axis='y', labelsize=8)  
        if i % 3 != 0:  # 
            axes[i].set_ylabel("")
    except Exception as e:
        print(f"Error plotting SHAP summary for model {name}: {e}")
fig.text(0.5, 0.004, "SHAP value (impact on model output)", ha="center", fontsize=12)
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])
plt.tight_layout(rect=[0, 0, 1, 1])
fig.suptitle("SHAP Feature Importance Analysis of Base Learners in the First Layer of Stacking Model", fontsize=16, y=1.02)
#plt.savefig("summary_plot.pdf", format='pdf', bbox_inches='tight')
plt.show()


print("=== 生成单独的SHAP dot图 ===")
for name, shap_df in shap_dfs.items():
    try:
        plt.figure(figsize=(8, 6))
        shap.summary_plot(
            shap_values=shap_df.to_numpy(),
            features=X_test,
            feature_names=X_test.columns,
            plot_type="dot",
            show=False,
            color_bar=True
        )
        plt.title(f"{name} - SHAP Feature Importance (Dot Plot)", fontsize=14, pad=20)
        plt.tight_layout()
        plt.savefig(f"shap_dot_{name}.pdf", format='pdf', bbox_inches='tight', dpi=300)
        plt.show()
        plt.close()
    except Exception as e:
        print(f"Error plotting SHAP dot plot for model {name}: {e}")

print("\n=== 生成单独的SHAP bar图 ===")
for name, shap_df in shap_dfs.items():
    try:
        plt.figure(figsize=(8, 6))
        shap.summary_plot(
            shap_values=shap_df.to_numpy(),
            features=X_test,
            feature_names=X_test.columns,
            plot_type="bar",
            show=False
        )
        plt.title(f"{name} - SHAP Feature Importance (Bar Plot)", fontsize=14, pad=20)
        plt.tight_layout()
        plt.savefig(f"shap_bar_{name}.pdf", format='pdf', bbox_inches='tight', dpi=300)
        plt.show()
        plt.close()
    except Exception as e:
        print(f"Error plotting SHAP bar plot for model {name}: {e}")


print("\n=== 生成手动SHAP特征重要性组合图 ===")
n_models = len(shap_dfs)
n_cols = 3  
n_rows = (n_models + n_cols - 1) // n_cols  
fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))


if n_rows == 1:
    axes = axes.reshape(1, -1)
axes = axes.flatten()

for i, (name, shap_df) in enumerate(shap_dfs.items()):
    try:

        feature_importance = np.abs(shap_df.values).mean(axis=0)
        feature_names = shap_df.columns


        sorted_idx = np.argsort(feature_importance)
        sorted_importance = feature_importance[sorted_idx]
        sorted_names = feature_names[sorted_idx]


        bars = axes[i].barh(range(len(sorted_importance)), sorted_importance, 
                           color='steelblue', alpha=0.7)
        axes[i].set_yticks(range(len(sorted_names)))
        axes[i].set_yticklabels(sorted_names, fontsize=8)
        axes[i].set_title(f"{name}", fontsize=12, fontweight='bold')
        axes[i].set_xlabel('Mean |SHAP value|', fontsize=10)


        axes[i].grid(axis='x', alpha=0.3)


        for j, v in enumerate(sorted_importance):
            axes[i].text(v + max(sorted_importance)*0.01, j, f'{v:.3f}', 
                        va='center', fontsize=7)

    except Exception as e:
        print(f"Error plotting manual SHAP plot for model {name}: {e}")

        axes[i].text(0.5, 0.5, f"Error: {name}\n{str(e)}", 
                    ha='center', va='center', transform=axes[i].transAxes,
                    fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="red", alpha=0.3))
        axes[i].set_title(f"{name} (Error)", fontsize=12, color='red')


for j in range(len(shap_dfs), len(axes)):
    fig.delaxes(axes[j])


fig.text(0.5, 0.02, "mean(|SHAP value|) (average impact on model output magnitude)", 
         ha="center", fontsize=14, fontweight='bold')

plt.tight_layout(rect=[0, 0.05, 1, 0.95]) 
fig.suptitle("SHAP Feature Importance Analysis of Base Learners in Stacking Model", 
             fontsize=16, fontweight='bold', y=0.98)
plt.savefig("Manual_SHAP_Feature_Importance.pdf", format='pdf', bbox_inches='tight', dpi=300)
plt.show()


print("\n=== 生成模型对比图 ===")
try:
    shap_values_list = [df.values for df in shap_dfs.values()]
    model_names = list(shap_dfs.keys())

    plt.figure(figsize=(12, 8))
    shap.summary_plot(
        shap_values=shap_values_list,
        features=X_test,
        feature_names=X_test.columns,
        class_names=model_names,
        plot_type="bar",
        show=False
    )
    plt.title("SHAP Feature Importance Comparison Across All Base Learners", fontsize=16, pad=20)
    plt.tight_layout()
    plt.savefig("All_Models_SHAP_Comparison.pdf", format='pdf', bbox_inches='tight', dpi=300)
    plt.show()

except Exception as e:
    print(f"Error creating comparison plot: {e}")

print("SHAP分析完成！")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

print("=== 生成SHAP Dot图组合 ===")

n_models = len(shap_dfs)
n_cols = 3  
n_rows = (n_models + n_cols - 1) // n_cols  
fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))


if n_rows == 1:
    axes = axes.reshape(1, -1)
axes = axes.flatten()


colors = ['#0000FF', '#FFFFFF', '#FF0000']
n_bins = 100
cmap = LinearSegmentedColormap.from_list('shap', colors, N=n_bins)

for i, (name, shap_df) in enumerate(shap_dfs.items()):
    try:
        ax = axes[i]


        shap_values = shap_df.values
        feature_values = X_test.values
        feature_names = shap_df.columns


        feature_importance = np.abs(shap_values).mean(axis=0)
        sorted_idx = np.argsort(feature_importance)


        top_features = sorted_idx[-10:]

        y_pos = np.arange(len(top_features))

        for j, feat_idx in enumerate(top_features):

            shap_vals = shap_values[:, feat_idx]
            feat_vals = feature_values[:, feat_idx]


            if feat_vals.std() > 0:
                feat_vals_norm = (feat_vals - feat_vals.min()) / (feat_vals.max() - feat_vals.min())
            else:
                feat_vals_norm = np.ones_like(feat_vals) * 0.5


            y_jitter = np.random.normal(0, 0.1, len(shap_vals))
            y_positions = np.full_like(shap_vals, j) + y_jitter


            scatter = ax.scatter(shap_vals, y_positions, c=feat_vals_norm, 
                               cmap=cmap, alpha=0.6, s=16, edgecolors='none')


        ax.set_yticks(y_pos)
        ax.set_yticklabels([feature_names[idx] for idx in top_features], fontsize=8)
        ax.set_xlabel('SHAP value (impact on model output)', fontsize=10)
        ax.set_title(f"{name}", fontsize=12, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        ax.axvline(x=0, color='black', linestyle='-', alpha=0.3, linewidth=0.8)


        if i % n_cols == n_cols - 1 or i == len(shap_dfs) - 1:
            cbar = plt.colorbar(scatter, ax=ax, aspect=30, shrink=0.8)
            cbar.set_label('Feature value\n(Low → High)', fontsize=8, rotation=270, labelpad=15)
            cbar.ax.tick_params(labelsize=7)

    except Exception as e:
        print(f"Error plotting SHAP dot plot for model {name}: {e}")

        ax.text(0.5, 0.5, f"Error: {name}\n{str(e)}", 
                ha='center', va='center', transform=ax.transAxes,
                fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="red", alpha=0.3))
        ax.set_title(f"{name} (Error)", fontsize=12, color='red')


for j in range(len(shap_dfs), len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
fig.suptitle("SHAP Dot Plot Analysis of Base Learners in Stacking Model", 
             fontsize=16, fontweight='bold', y=0.98)


fig.text(0.5, 0.01, 
         "Each dot represents a sample. Color indicates feature value (blue=low, red=high). "
         "X-axis shows SHAP impact on model output.", 
         ha="center", fontsize=10, style='italic')

plt.savefig("SHAP_Dot_Plot_Combined.pdf", format='pdf', bbox_inches='tight', dpi=300)
plt.show()

print("SHAP Dot图组合完成！")



print("=== 生成SHAP分析可视化 ===")


print("1. 生成单独的SHAP dot图...")
for name, shap_df in shap_dfs.items():
    try:
        plt.figure(figsize=(8, 6))
        shap.summary_plot(
            shap_values=shap_df.to_numpy(),
            features=X_test,
            feature_names=X_test.columns,
            plot_type="dot",
            show=False,
            color_bar=True
        )
        plt.title(f"{name} - SHAP Feature Importance (Dot Plot)", fontsize=14, pad=20)
        plt.tight_layout()
        plt.savefig(f"shap_dot_{name}.pdf", format='pdf', bbox_inches='tight', dpi=300)
        plt.show()
    except Exception as e:
        print(f"Error plotting SHAP dot plot for model {name}: {e}")


print("\n2. 生成单独的SHAP bar图...")
for name, shap_df in shap_dfs.items():
    try:
        plt.figure(figsize=(8, 6))
        shap.summary_plot(
            shap_values=shap_df.to_numpy(),
            features=X_test,
            feature_names=X_test.columns,
            plot_type="bar",
            show=False
        )
        plt.title(f"{name} - SHAP Feature Importance (Bar Plot)", fontsize=14, pad=20)
        plt.tight_layout()
        plt.savefig(f"shap_bar_{name}.pdf", format='pdf', bbox_inches='tight', dpi=300)
        plt.show()
    except Exception as e:
        print(f"Error plotting SHAP bar plot for model {name}: {e}")


print("\n3. 生成模型对比图...")
try:
    shap_values_list = [df.values for df in shap_dfs.values()]
    model_names = list(shap_dfs.keys())

    plt.figure(figsize=(12, 8))
    shap.summary_plot(
        shap_values=shap_values_list,
        features=X_test,
        feature_names=X_test.columns,
        class_names=model_names,
        plot_type="bar",
        show=False
    )
    plt.title("SHAP Feature Importance Comparison Across All Base Learners", fontsize=16, pad=20)
    plt.tight_layout()
    plt.savefig("All_Models_SHAP_Comparison.pdf", format='pdf', bbox_inches='tight', dpi=300)
    plt.show()

except Exception as e:
    print(f"Error creating comparison plot: {e}")

print("\n=== SHAP分析完成！===")



import matplotlib.pyplot as plt
import numpy as np

n_models = len(shap_dfs)
n_cols = 3  
n_rows = (n_models + n_cols - 1) // n_cols  
fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))


if n_rows == 1:
    axes = axes.reshape(1, -1)
axes = axes.flatten()

for i, (name, shap_df) in enumerate(shap_dfs.items()):
    try:

        feature_importance = np.abs(shap_df.values).mean(axis=0)
        feature_names = shap_df.columns


        sorted_idx = np.argsort(feature_importance)
        sorted_importance = feature_importance[sorted_idx]
        sorted_names = feature_names[sorted_idx]


        bars = axes[i].barh(range(len(sorted_importance)), sorted_importance, 
                           color='steelblue', alpha=0.7)
        axes[i].set_yticks(range(len(sorted_names)))
        axes[i].set_yticklabels(sorted_names, fontsize=8)
        axes[i].set_title(f"{name}", fontsize=12, fontweight='bold')
        axes[i].set_xlabel('Feature Importance', fontsize=10)


        axes[i].grid(axis='x', alpha=0.3)


        for j, v in enumerate(sorted_importance):
            axes[i].text(v + max(sorted_importance)*0.01, j, f'{v:.3f}', 
                        va='center', fontsize=7, fontweight='bold')

    except Exception as e:
        print(f"Error plotting SHAP bar plot for model {name}: {e}")

        axes[i].text(0.5, 0.5, f"Error: {name}\n{str(e)}", 
                    ha='center', va='center', transform=axes[i].transAxes,
                    fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="red", alpha=0.3))
        axes[i].set_title(f"{name} (Error)", fontsize=12, color='red')


for j in range(len(shap_dfs), len(axes)):
    fig.delaxes(axes[j])


fig.text(0.5, 0.02, "mean(|SHAP value|) (average impact on model output magnitude)", 
         ha="center", fontsize=14, fontweight='bold')

plt.tight_layout(rect=[0, 0.05, 1, 0.95]) 
fig.suptitle("SHAP Feature Importance Analysis of Base Learners in Stacking Model", 
             fontsize=16, fontweight='bold', y=0.98)
plt.savefig("Manual_SHAP_Feature_Importance.pdf", format='pdf', bbox_inches='tight', dpi=300)
plt.show()


print("=== 生成单独的SHAP dot图（用于组合） ===")


fig_width = 15
fig_height = 10
n_cols = 3
n_rows = 2


fig = plt.figure(figsize=(fig_width, fig_height))

for i, (name, shap_df) in enumerate(shap_dfs.items()):
    try:

        ax = plt.subplot(n_rows, n_cols, i + 1)


        current_fig = plt.gcf()
        current_ax = plt.gca()


        temp_fig, temp_ax = plt.subplots(figsize=(5, 4))


        shap.summary_plot(
            shap_values=shap_df.to_numpy(),
            features=X_test,
            feature_names=X_test.columns,
            plot_type="dot",
            show=False,
            color_bar=(i % n_cols == n_cols - 1)
        )


        temp_ax.set_title(name, fontsize=12, fontweight='bold')
        temp_ax.set_xlabel('SHAP value (impact on model output)', fontsize=10)


        temp_fig.savefig(f'temp_shap_{name}.png', dpi=150, bbox_inches='tight')
        plt.close(temp_fig)


        plt.figure(current_fig.number)
        plt.sca(current_ax)


        from PIL import Image
        img = Image.open(f'temp_shap_{name}.png')
        ax.imshow(img)
        ax.axis('off')
        ax.set_title(name, fontsize=12, fontweight='bold')

    except Exception as e:
        print(f"Error plotting SHAP dot plot for model {name}: {e}")
        ax.text(0.5, 0.5, f"Error: {name}\n{str(e)}", 
                ha='center', va='center', transform=ax.transAxes,
                fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="red", alpha=0.3))

plt.tight_layout()
fig.suptitle("SHAP Dot Plot Analysis of Base Learners in Stacking Model", 
             fontsize=16, fontweight='bold', y=0.98)
plt.savefig("SHAP_Dot_Combined_Alternative.pdf", format='pdf', bbox_inches='tight', dpi=300)
plt.show()


import os
for name in shap_dfs.keys():
    try:
        os.remove(f'temp_shap_{name}.png')
    except:
        pass

print("SHAP Dot图组合（替代方案）完成！")

import matplotlib.pyplot as plt
import numpy as np


n_models = len(shap_dfs)
n_cols = 3  
n_rows = (n_models + n_cols - 1) // n_cols  
fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))


if n_rows == 1:
    axes = axes.reshape(1, -1)
axes = axes.flatten()

for i, (name, shap_df) in enumerate(shap_dfs.items()):
    try:

        feature_importance = np.abs(shap_df.values).mean(axis=0)
        feature_names = shap_df.columns


        sorted_idx = np.argsort(feature_importance)
        sorted_importance = feature_importance[sorted_idx]
        sorted_names = feature_names[sorted_idx]


        axes[i].barh(range(len(sorted_importance)), sorted_importance)
        axes[i].set_yticks(range(len(sorted_names)))
        axes[i].set_yticklabels(sorted_names, fontsize=8)
        axes[i].set_title(f"{name}", fontsize=12)
        axes[i].set_xlabel('')


        for j, v in enumerate(sorted_importance):
            axes[i].text(v + 0.001, j, f'{v:.3f}', va='center', fontsize=7)

    except Exception as e:
        print(f"Error plotting SHAP bar plot for model {name}: {e}")

        axes[i].text(0.5, 0.5, f"Error: {name}\n{str(e)}", 
                    ha='center', va='center', transform=axes[i].transAxes)
        axes[i].set_title(f"{name} (Error)", fontsize=12)


for j in range(len(shap_dfs), len(axes)):
    fig.delaxes(axes[j])


fig.text(
    0.5, 0.02,  
    "mean(|SHAP value|) (average impact on model output magnitude)", 
    ha="center", 
    fontsize=14
)

plt.tight_layout(rect=[0, 0.05, 1, 0.95]) 
fig.suptitle("SHAP Sorted Feature Importance of Base Learners in the First Layer of Stacking Model", 
             fontsize=16, y=0.98)
plt.savefig("Sorted Feature Importance.pdf", format='pdf', bbox_inches='tight', dpi=300)
plt.show()

from sklearn.exceptions import NotFittedError
import sys
import os
import shap
import pandas as pd

try:
    meta_model = stacking_regressor.final_estimator_
except AttributeError:
    raise NotFittedError("The final estimator (meta_model) is not yet fitted. Ensure that the StackingRegressor is trained.")


meta_features = stacking_regressor.transform(X_test)


explainer_meta = shap.KernelExplainer(meta_model.predict, meta_features)


shap_values_meta = explainer_meta.shap_values(meta_features)


shap_df = pd.DataFrame(shap_values_meta, columns=[name for name, _ in base_learners])

meta_features_df = pd.DataFrame(meta_features, columns=shap_df.columns)


plt.figure()
shap.summary_plot(np.array(shap_df), meta_features_df, feature_names=shap_df.columns, plot_type="dot", show=False)
plt.title("SHAP Contribution Analysis for the Meta-Learner in the Second Layer of Stacking Regressor", fontsize=16, y=1.02)
plt.savefig("SHAP Contribution Analysis for the Meta-Learner in the Second Layer of Stacking Regressor.pdf", format='pdf', bbox_inches='tight')
plt.show()

plt.figure(figsize=(10, 5), dpi=1200)
shap.summary_plot(np.array(shap_df), meta_features_df, plot_type="bar", show=False)
plt.tight_layout()
plt.title("Bar Plot of SHAP Feature Contributions for the Meta-Learner in Stacking Regressor", fontsize=16, y=1.02)
plt.savefig("Bar Plot of SHAP Feature Contributions for the Meta-Learner in Stacking Regressor.pdf", format='pdf', bbox_inches='tight')
plt.show()


stacking_explainer = shap.KernelExplainer(stacking_regressor.predict, X_test)

stacking_shap_values = stacking_explainer.shap_values(X_test)
stacking_shap_df = pd.DataFrame(stacking_shap_values, columns=X_test.columns)

plt.figure()
shap.summary_plot(np.array(stacking_shap_df), X_test, feature_names=stacking_shap_df.columns, plot_type="dot", show=False)
plt.title("Based on the overall feature contribution analysis of SHAP to the stacking model", fontsize=16, y=1.02)
plt.savefig("Based on the overall feature contribution analysis of SHAP to the stacking model.pdf", format='pdf', bbox_inches='tight')
plt.show()

plt.figure(figsize=(10, 5), dpi=1200)
shap.summary_plot(np.array(stacking_shap_df), X_test.iloc[0:100, :], plot_type="bar", show=False)
plt.tight_layout()
plt.title("SHAP-based Stacking Model Feature Contribution Histogram Analysis", fontsize=16, y=1.02)
plt.savefig("SHAP-based Stacking Model Feature Contribution Histogram Analysis.pdf", format='pdf', bbox_inches='tight')
plt.show()

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import shap


S = np.asarray(stacking_shap_df)  # (n_samples, n_features)
feat_names = np.array(getattr(X_test, "columns", range(S.shape[1])))
imp = np.abs(S).mean(axis=0)
order = np.argsort(imp)[::1]

S_ord = S[:, order]
X_ord = X_test.iloc[:, order]
names_ord = feat_names[order]
imp_ord = imp[order]
ypos = np.arange(len(order))


fig, ax = plt.subplots(figsize=(7, 8), dpi=300)
shap.summary_plot(
    S_ord, X_ord,
    feature_names=names_ord,
    plot_type="dot",
    show=False, color_bar=True
)


ax.set_position([0.20, 0.18, 0.63, 0.70])


for i in range(len(order)):
    ax.axhspan(i - 0.5, i + 0.5, color="#eaf4ff", alpha=0.5, zorder=-2)
ax.axvline(0, color="0.6", lw=1, zorder=-1)


ax_top = ax.twiny()
ax_top.set_ylim(ax.get_ylim())
ax_top.barh(
    ypos, imp_ord, height=0.82,
    color=mcolors.to_rgba("#4a90e2", 0.5),
    edgecolor="none", zorder=-1
)


ax.set_xlabel("Shapley Value Contribution", fontsize=15)
ax.set_ylabel("Stacking Model features of Shear Modulus", fontsize=17)

ax_top.set_xlabel("Mean |Shapley Value|", fontsize=14, labelpad=5)
ax_top.xaxis.set_label_position('top')
ax_top.xaxis.tick_top()


ax.set_xlim(-100, 100)
ax_top.set_xlim(0.0, 30.0)


ax.tick_params(axis='x', labelsize=14)
ax.tick_params(axis='y', labelsize=14, pad=-10)
ax_top.tick_params(axis='x', labelsize=14)


ax.text(-0.3, 1.02, "(d)", transform=ax.transAxes,
        fontsize=20, color='black', va='top', ha='left')

plt.tight_layout()
plt.savefig("shap_overlay.pdf", bbox_inches="tight")
plt.show()


import statsmodels.api as sm
from statsmodels.nonparametric.smoothers_lowess import lowess


features = stacking_shap_df.columns.tolist()


fig, axes = plt.subplots(3, 3, figsize=(10, 10), dpi=1200)
axes = axes.flatten()


for i in range(len(axes)):
    if i < len(features):
        feature = features[i]
        if feature in X_test.columns and feature in stacking_shap_df.columns:
            ax = axes[i]


            ax.scatter(X_test[feature], stacking_shap_df[feature], s=10, color="#6A9ACE")
            ax.axhline(y=0, color='red', linestyle='-.', linewidth=1)


            lowess_fit = lowess(stacking_shap_df[feature], X_test[feature], frac=0.3)
            ax.plot(lowess_fit[:, 0], lowess_fit[:, 1], color='#B5B5B5', linewidth=2)


            ax.set_xlabel(feature, fontsize=10)
            ax.set_ylabel(f'SHAP value for\n{feature}', fontsize=10)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
        else:

            axes[i].axis('off')
    else:

        axes[i].axis('off')


plt.tight_layout()
#plt.savefig("with_lowess.pdf", format='pdf', bbox_inches='tight')
fig.suptitle("SHAP Values for Shear Modulus", fontsize=18, y=1.02)
plt.show()

import shap
import matplotlib.pyplot as plt



shap.initjs()
sample_idx = 0
shap.force_plot(
    base_value=stacking_explainer.expected_value,
    shap_values=stacking_shap_values[sample_idx],
    features=X_test.iloc[sample_idx],
    feature_names=X_test.columns
)


shap.force_plot(
    base_value=stacking_explainer.expected_value,
    shap_values=stacking_shap_values[:100],
    features=X_test.iloc[:100],
    feature_names=X_test.columns,

)



feature_name = stacking_shap_df.columns[0]
shap.dependence_plot(
    ind=feature_name,
    shap_values=stacking_shap_values,
    features=X_test,
    feature_names=X_test.columns
)


# for feature_name in stacking_shap_df.columns:
#     shap.dependence_plot(feature_name, stacking_shap_values, X_test, feature_names=X_test.columns)

for feature_name in stacking_shap_df.columns:
    shap.dependence_plot(feature_name, stacking_shap_values, X_test, feature_names=X_test.columns, interaction_index='vpa')


import shap
import matplotlib.pyplot as plt


X_test_reset = X_test.reset_index(drop=True)


sample_idx = X_test_reset[X_test_reset['vpa'] == -0.3452643848715692].index[0]


shap.force_plot(
    base_value=stacking_explainer.expected_value,
    shap_values=stacking_shap_values[sample_idx],
    features=X_test_reset.iloc[sample_idx],
    feature_names=X_test.columns,
    matplotlib=True
)
plt.show()

stacking_shap_df.to_csv("stacking_shap_values2.csv", index=False)
