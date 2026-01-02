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
df_fomula = df["formula"]
for col in df.columns:
    if df[col].nunique() < 0.01 * len(df):
        df = df.drop(columns=[col])
new_columns = [col.replace('MagpieData ', '') for col in df.columns]
df.columns = new_columns
data = df.iloc[:, 3:]
data_target = df.iloc[:, 0:3]
#data_features = df.iloc[:, 3:]
data_target['Flexibility'] = data_target['bulk_modulus'] / data_target['shear_modulus']
data_features = data
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


import pandas as pd
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


if __name__ == "__main__":

    test_data = {
        'feature[1]': [1, 2, 3],
        'feature<2>': [4, 5, 6],
        'feature[3]<test>': [7, 8, 9],
        'normal_feature': [10, 11, 12],
        '特征[中文]': [13, 14, 15]
    }

    df_test = pd.DataFrame(test_data)

    print("测试数据:")
    print(df_test.head())
    print(f"原始列名: {df_test.columns.tolist()}")


    print("\n" + "="*60)
    print("方法1: 智能清理")
    df_clean = fix_data_features(df_test, method='clean')
    print(f"清理后列名: {df_clean.columns.tolist()}")


    print("\n" + "="*60)
    print("方法2: 简单重命名")
    df_simple = fix_data_features(df_test, method='simple', save_mapping=False)
    print(f"简单重命名后列名: {df_simple.columns.tolist()}")


    print("\n" + "="*60)
    print("方法3: 编号重命名")
    df_numbered = fix_data_features(df_test, method='numbered', save_mapping=False)
    print(f"编号重命名后列名: {df_numbered.columns.tolist()}")


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


X = data_features
X_0 = data_features[['minimum CovalentRadius', 'maximum CovalentRadius', 'mean CovalentRadius',
              'minimum Electronegativity', 'maximum Electronegativity', 'mean Electronegativity',
              'minimum MeltingT', 'maximum MeltingT', 'mean MeltingT',
              'density', 'vpa', 'packing fraction', 'mean Average bond angle','mean Average bond length']]

y1 = data_target['bulk_modulus']
y2 = data_target['shear_modulus']
y3 = data_target['Flexibility']
y4 = data_target['Flexibility_class']


X_0 = X_0.fillna(0)
X = X.fillna(0)
X= fix_data_features(X, method='clean')


X = X[~X.isin([np.inf, -np.inf]).any(axis=1)]
X_0 = X_0[~X_0.isin([np.inf, -np.inf]).any(axis=1)]
#X_best = X_best[~X_best.isin([np.inf, -np.inf]).any(axis=1)]

import xgboost as xgb
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X, y4, test_size=0.001, 
                                                    random_state=2)

xgb_model = xgb.XGBRegressor(random_state=42)
xgb_model.fit(X_train, y_train)


feature_importance = xgb_model.feature_importances_


feature_importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': feature_importance
})


feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
feature_importance_df

import xgboost as xgb
from sklearn.model_selection import train_test_split
import pandas as pd


X_train, X_test, y_train, y_test = train_test_split(X, y4, test_size=0.001, random_state=2)


xgb_model = xgb.XGBClassifier(random_state=42)
xgb_model.fit(X_train, y_train)


feature_importance = xgb_model.feature_importances_


feature_importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': feature_importance
})


feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
feature_importance_df

import xgboost as xgb
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X, y4, test_size=0.001, 
                                                    random_state=42)

xgb_model = xgb.XGBClassifier(random_state=42)
xgb_model.fit(X_train, y_train)


feature_importance = xgb_model.feature_importances_


feature_importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': feature_importance
})


feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
feature_importance_df

import matplotlib.cm as cm
import matplotlib.colors as mcolors

# Get top 30 features
top_features = feature_importance_df.head(30)
importances = top_features['Importance']
features = top_features['Feature']

# Configure color scheme
norm = mcolors.Normalize(vmin=importances.min(), vmax=importances.max())
colors = cm.viridis(norm(importances))  

# Create horizontal bar chart for single model feature importance
plt.figure(figsize=(8, 12), dpi=1200)  # Adjust figure size for larger fonts
plt.barh(features, importances, color=colors)
plt.title("Top 30 Feature Importances", fontsize=20)  # Set title font size
plt.xlabel("Importance", fontsize=15)
plt.ylabel("Features", fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.gca().invert_yaxis()
plt.tight_layout()
#plt.savefig("1.pdf", format='pdf', bbox_inches='tight', dpi=1200)
plt.show()

X[list(top_features['Feature'])]

from matplotlib.patches import Wedge

corr = X[list(top_features['Feature'])].corr()
p_values = X[list(top_features['Feature'])].corr(method="pearson")  


fig, ax = plt.subplots(figsize=(18, 12), dpi=300)
fig.subplots_adjust(left=0.50)
cmap = plt.cm.RdBu
norm = plt.Normalize(vmin=-1, vmax=1)


for i in range(len(corr.columns)):
    for j in range(len(corr.columns)):
        if i >= j:
            ax.scatter(i, j, s=1, color='white')
        else:
            coeff = corr.iloc[i, j]
            abs_coeff = np.abs(coeff)


            start_angle = 90
            end_angle = 90 + abs_coeff * 360


            x, y = i, j


            wedge = Wedge(center=(x, y), r=0.4, theta1=start_angle, theta2=end_angle,
                          facecolor=cmap(norm(coeff)), edgecolor='black', alpha=0.75)
            ax.add_patch(wedge)


            bg_wedge = Wedge(center=(x, y), r=0.4, theta1=end_angle, theta2=start_angle + 360,
                             facecolor='white', edgecolor='black', alpha=0.5)
            ax.add_patch(bg_wedge)


            if abs_coeff > 0.85:
                font_color = 'white'
            else:
                font_color = 'black'


            ax.text(x, y, f'{coeff:.2f}', ha='center', va='center', fontsize=10, color=font_color)


            p_value = p_values.iloc[i, j]
            if p_value < 0.001:
                sig = '***'
            elif p_value < 0.01:
                sig = '**'
            elif p_value < 0.05:
                sig = '*'
            else:
                sig = ''


            ax.text(x, y - 0.3, sig, ha='center', va='center', fontsize=10, color=font_color)


ax.set_xticks(range(len(corr.columns)))
ax.set_xticklabels(corr.columns, rotation=45, ha='right', fontsize=10)
ax.set_yticks(range(len(corr.columns)))
ax.set_yticklabels(corr.columns, fontsize=10)


sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, orientation='vertical', ticks=[-1, -0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1],pad=0.1)

cbar.ax.tick_params(width=0.01)


cbar.ax.tick_params(labelsize=15, width=2)

for spine in ax.spines.values():
    spine.set_visible(False)


ax.set_xticks([])
ax.set_yticks([])
ax.set_xticklabels([])
ax.set_yticklabels([])
#for i in range(len(corr.columns)):
#    ax.text(i+0.1, i-0.02, X[list(top_features['Feature'])].columns[i], ha='center', va='center', fontsize=8, color='black', fontweight='bold',rotation=90)
for i in range(1, len(corr.columns)):
    ax.text(i+0.1, i-0.02, X[list(top_features['Feature'])].columns[i], 
            ha='left', va='center', fontsize=10, color='black', 
            fontweight='bold')
for i in [0]:
    ax.text(i+0.1, i-0.02, X[list(top_features['Feature'])].columns[i], 
            ha='center', va='center', fontsize=10, color='black', 
            fontweight='bold')
ax.text(23, 7, "Blue → Positive correlation", ha='center', va='top', fontsize=18, color='blue', fontweight='bold')
ax.text(23, 6, "Red → Negative correlation", ha='center', va='top', fontsize=18, color='red', fontweight='bold')
ax.text(23, 5, "The filled fraction of each \npie chart represents the absolute value of \nthe Pearson correlation coefficient\n\nAsterisks indicate statistical significance\n(* p<0.05; ** p<0.01; *** p<0.001)",
        ha='center', va='top', fontsize=18, color='black', fontweight='bold', wrap=True)
#plt.savefig("2.pdf", format='pdf', bbox_inches='tight', dpi=1200)
#ax.text(0, 30, "(a)",  fontsize=30, color='black')
plt.tight_layout()
plt.show()


X[list(top_features['Feature'])]

top_n_feature_names = list(X[list(top_features['Feature'])].columns)

threshold = 0.58
features_to_keep = top_n_feature_names[:]  
features_to_remove_collinearity = [] 
importance_dict = feature_importance_df.set_index('Feature')['Importance'].to_dict()


cols_to_check = top_n_feature_names[:]
already_removed_in_this_pass = set()

for i in range(len(cols_to_check)):
    for j in range(i + 1, len(cols_to_check)):
        feat1 = cols_to_check[i]
        feat2 = cols_to_check[j]


        if feat1 in already_removed_in_this_pass or feat2 in already_removed_in_this_pass:
            continue


        if feat1 in corr.index and feat2 in corr.columns:
            correlation = corr.loc[feat1, feat2]
        else:
            continue

        if abs(correlation) > threshold:
            print(f"发现高度相关特征对: '{feat1}' 和 '{feat2}', 相关系数: {correlation:.4f}")

            importance1 = importance_dict.get(feat1, 0)
            importance2 = importance_dict.get(feat2, 0)

            if importance1 < importance2:
                if feat1 in features_to_keep:
                    features_to_keep.remove(feat1)
                    features_to_remove_collinearity.append(feat1)
                    already_removed_in_this_pass.add(feat1)
                    print(f"  剔除 '{feat1}' (重要性: {importance1:.4f})，保留 '{feat2}' (重要性: {importance2:.4f})")
            elif importance2 < importance1:
                if feat2 in features_to_keep:
                    features_to_keep.remove(feat2)
                    features_to_remove_collinearity.append(feat2)
                    already_removed_in_this_pass.add(feat2)
                    print(f"  剔除 '{feat2}' (重要性: {importance2:.4f})，保留 '{feat1}' (重要性: {importance1:.4f})")
            else:
                if feat2 in features_to_keep:
                    features_to_keep.remove(feat2)
                    features_to_remove_collinearity.append(feat2)
                    already_removed_in_this_pass.add(feat2)
                    print(f"  重要性相同。剔除 '{feat2}' (索引靠后/任意选择)，保留 '{feat1}'")

print("-" * 30)
print("最终保留的特征 (经过多重共线性剔除后):")
print(features_to_keep)
print(f"共保留 {len(features_to_keep)} 个特征。")
print("-" * 30)
print("因多重共线性而被剔除的特征 (来自Top N列表):")
print(features_to_remove_collinearity)
print(f"共剔除 {len(features_to_remove_collinearity)} 个特征。")
print("-" * 30)


feature_names = top_n_feature_names
feature_importances = feature_importance_df.iloc[0:30]['Importance']
selected_features = features_to_keep


colors = ['red' if feature in selected_features else 'blue' for feature in feature_names]



plt.subplots(figsize=(8, 4.5), dpi=200)
bars = plt.bar(feature_names, feature_importances, color=colors, edgecolor='black', alpha=0.85)



red_patch = plt.Line2D([0], [0], color='red', lw=4, label=' Retained features')
blue_patch = plt.Line2D([0], [0], color='blue', lw=4, label='Removed features')

plt.legend(handles=[red_patch, blue_patch], loc='upper right', fontsize=15)


#plt.title('Feature Importances of Shear Modulus', fontsize=20)
plt.title('Test Set Performance Metrics of Shear Modulus', fontsize=20)
#plt.xlabel('Features ', fontsize=20)
plt.ylabel('Feature-Importances', fontsize=15)
plt.xticks(rotation=40, fontsize=10, ha='right')
plt.yticks(fontsize=15)
plt.tight_layout()
plt.text(-3, 0.25, "(b)",  fontsize=20, color='black')
#plt.savefig("3.pdf", format='pdf', bbox_inches='tight', dpi=1200)
plt.show()

import itertools

final_features = features_to_keep 


X_train, X_test, y_train, y_test = train_test_split(X, y1, test_size=0.001, 
                                                    random_state=42)

all_combinations = []
for r in range(1, len(final_features) + 1):
    combinations = list(itertools.combinations(final_features, r))
    all_combinations.extend(combinations)

all_combinations 

len(all_combinations)

from sklearn.model_selection import cross_val_score, KFold
from joblib import Parallel, delayed
import numpy as np
import pandas as pd
import xgboost as xgb


def evaluate_feature_combination(combination, X_train, y_train, random_state, kf):
    """
    Evaluate a single feature combination.

    Parameters:
    combination: feature tuple
    X_train: training data
    y_train: training targets
    random_state: random seed
    kf: KFold object

    Returns:
    dict: feature list and CV R^2 score
    """
    selected_features = list(combination)
    X_train_subset = X_train[selected_features]


    rf_model = xgb.XGBRegressor(random_state=random_state)


    cv_scores = cross_val_score(rf_model, X_train_subset, y_train, cv=kf, scoring='r2')
    mean_cv_score = np.mean(cv_scores)

    return {
        'Features': selected_features,
        'Mean CV R^2': mean_cv_score
    }


random_state = 5
kf = KFold(n_splits=5, shuffle=True, random_state=random_state)


results = Parallel(n_jobs=-1, verbose=1)(
    delayed(evaluate_feature_combination)(
        combination, X_train, y_train, random_state, kf
    ) for combination in all_combinations
)

print("并行计算完成，正在整理结果...")


results_df = pd.DataFrame(results)
results_df = results_df.sort_values(by='Mean CV R^2', ascending=False).reset_index(drop=True)

print(f"计算完成！共评估了 {len(results_df)} 个特征组合")
print("前10个最佳特征组合：")
print(results_df.head(10))

results_df


results_df['Num Features'] = results_df['Features'].apply(len)


global_best_idx = results_df['Mean CV R^2'].idxmax()
global_best_score = results_df.loc[global_best_idx, 'Mean CV R^2']
global_best_num_features = len(results_df.loc[global_best_idx, 'Features'])
global_best_features = results_df.loc[global_best_idx, 'Features']


optimal_scores = results_df[results_df['Num Features'] != 20].groupby('Num Features')['Mean CV R^2'].max()


plt.figure(figsize=(8,7 ), dpi=200)


for num_features, group in results_df[results_df['Num Features'] != 20].groupby('Num Features'):
    plt.scatter(
        [num_features] * len(group), group['Mean CV R^2'], 
        color='none', edgecolor='blue', label='Other combinations' if num_features == 1 else "", 
        marker='s', alpha=0.7
    )


for num_features, value in optimal_scores.items():
    if num_features != global_best_num_features:
        plt.scatter(
            num_features, value, 
            color='red', edgecolor='black', label='Best per feature count' if num_features == 1 else "", 
            marker='o', s=150, alpha=0.8
        )


plt.scatter(
    global_best_num_features, global_best_score, 
    color='red', edgecolor='black', label='Highest', 
    marker='*', s=300, alpha=1.0
)


#plt.text(

#    f"Best features:\n{', '.join(global_best_features)}", 
#    fontsize=19, color='black', ha='center', va='center', fontweight='bold'
#)


plt.title('Feature Selection of Bulk Modulus', fontsize=23)
plt.xlabel('Number of Features', fontsize=23)
plt.ylabel('Coefficient of Determination $R^2$', fontsize=18)
plt.xticks(np.arange(1, len(results_df['Num Features'].unique()) + 1), fontsize=20)
plt.yticks(fontsize=20)


plt.legend(handles=[plt.Line2D([0], [0], color='red', marker='*', markersize=20, label='Highest', linestyle='')], 
           fontsize=25, loc='lower right', frameon=False)
#plt.savefig("4.pdf", format='pdf', bbox_inches='tight', dpi=1200)
plt.text(-1.1, 1.1, "(a)",  fontsize=30, color='black')
plt.tight_layout()
plt.show()

global_best_features

from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
X_train, X_test, y_train, y_test = train_test_split(X[list(top_features['Feature'])], y1, test_size=0.2, 
                                                    random_state=42)



xgb_reg = xgb.XGBRegressor(random_state=42)


rfe = RFE(estimator=xgb_reg, n_features_to_select=1, step=1)
rfe.fit(X_train, y_train)


feature_ranking = rfe.ranking_


rfe_features = pd.DataFrame({
    'Feature': X_train.columns,
    'Ranking': feature_ranking
}).sort_values(by='Ranking')
rfe_features


selected_features = []


selection_results = pd.DataFrame(columns=['Feature', 'Importance', 'MSE', 'R2'])


for i in range(len(rfe_features)):

    current_feature = rfe_features.iloc[i]['Feature']
    selected_features.append(current_feature)


    X_train_subset = X_train[selected_features]
    X_test_subset = X_test[selected_features]


    xgb_reg = xgb.XGBRegressor(random_state=42)
    xgb_reg.fit(X_train_subset, y_train)


    if len(selected_features) == 1:
        importance = xgb_reg.feature_importances_[0]
    else:

        importance = xgb_reg.feature_importances_[-1]


    y_pred = xgb_reg.predict(X_test_subset)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)


    selection_results.loc[len(selection_results)] = [
        current_feature,
        importance,
        mse,
        r2
    ]

selection_results

selection_results = selection_results.iloc[0:130]

n_features = 16
fig, ax1 = plt.subplots(figsize=(30, 16))

norm = plt.Normalize(selection_results['Importance'].min(), selection_results['Importance'].max())
colors = plt.cm.Blues(norm(selection_results['Importance']))
ax1.bar(selection_results['Feature'], selection_results['Importance'], color=colors, label='Feature Importance')
ax1.set_xlabel("Features", fontsize=18, fontweight='bold')
ax1.set_ylabel("Feature Importance", fontsize=18, fontweight='bold')
ax1.tick_params(axis='y', labelsize=15, width=1.5)
x_labels = selection_results['Feature']
x_colors = ['red' if i < n_features else 'black' for i in range(len(x_labels))]
for tick_label, color in zip(ax1.get_xticklabels(), x_colors):
    tick_label.set_color(color)
ax1.tick_params(axis='x', rotation=90, labelsize=15, width=1.5)
ax2 = ax1.twinx()
ax2.plot(
    selection_results['Feature'][:n_features + 1],
    selection_results['R2'][:n_features + 1],
    color="red", marker='o', linestyle='-', label="Cumulative R² (Top Features)"
)
ax2.plot(
    selection_results['Feature'][n_features:],
    selection_results['R2'][n_features:],
    color="black", marker='o', linestyle='-', label="Cumulative R² (Other Features)"
)
ax2.set_ylabel("Cumulative R²", fontsize=18, fontweight='bold')
ax2.tick_params(axis='y', labelsize=15, width=1.5)
ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.3f}'))

plt.title(f"Feature Contribution and R² Performance (Top {n_features} Features - RF)", fontsize=18, fontweight='bold')
fig.tight_layout()
plt.savefig("RF-RFE-R2.pdf", format='pdf', bbox_inches='tight', dpi=1200)
plt.show()


fig, ax1 = plt.subplots(figsize=(30, 16))


norm = plt.Normalize(selection_results['Importance'].min(), selection_results['Importance'].max())
colors = plt.cm.Blues(norm(selection_results['Importance']))
ax1.bar(selection_results['Feature'], selection_results['Importance'], color=colors, label='Feature Importance')
ax1.set_xlabel("Features", fontsize=18, fontweight='bold')
ax1.set_ylabel("Feature Importance", fontsize=18, fontweight='bold')
ax1.tick_params(axis='y', labelsize=15, width=1.5)


x_labels = selection_results['Feature']
x_colors = ['red' if i < n_features else 'black' for i in range(len(x_labels))]
for tick_label, color in zip(ax1.get_xticklabels(), x_colors):
    tick_label.set_color(color)
ax1.tick_params(axis='x', rotation=90, labelsize=15, width=1.5)


ax2 = ax1.twinx()


ax2.plot(
    selection_results['Feature'][:n_features + 1],
    selection_results['MSE'][:n_features + 1],
    color="red", marker='o', linestyle='-', label="Cumulative MSE (Top Features)"
)


ax2.plot(
    selection_results['Feature'][n_features:],
    selection_results['MSE'][n_features:],
    color="black", marker='o', linestyle='-', label="Cumulative MSE (Other Features)"
)


ax2.set_ylabel("Cumulative MSE", fontsize=18, fontweight='bold')
ax2.tick_params(axis='y', labelsize=15, width=1.5)
ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.3f}'))


plt.title(f"Feature Contribution and MSE Performance (Top {n_features} Features - RF)", fontsize=18, fontweight='bold')


fig.tight_layout()


#plt.savefig("RF-RFE-MSE.pdf", format='pdf', bbox_inches='tight', dpi=1200)

plt.show()
