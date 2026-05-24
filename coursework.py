import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (classification_report, confusion_matrix,
roc_curve, auc, precision_recall_curve, f1_score,
roc_auc_score, precision_score, recall_score)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from imblearn.metrics import classification_report_imbalanced

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
import shap

sns.set_theme(style="whitegrid", font_scale=1.1)
plt.rcParams['figure.figsize'] = (10, 6)
print(" Библиотеки загружены")

df = pd.read_csv('online_shoppers_intention.csv')

expected_cols = [
'Administrative', 'Administrative_Duration', 'Informational', 'Informational_Duration',
'ProductRelated', 'ProductRelated_Duration', 'BounceRates', 'ExitRates', 'PageValues',
'SpecialDay', 'Month', 'OperatingSystems', 'Browser', 'Region', 'TrafficType',
'VisitorType', 'Weekend', 'Revenue'
]
if len(df.columns) == len(expected_cols) and df.columns[0] != 'Administrative':
    df.columns = expected_cols

print(f" Размер датасета: {df.shape[0]} строк, {df.shape[1]} колонок")
display(df.head(3))
print("\n Информация о данных:")
display(df.info())
print("\n Пропуски:")
display(df.isnull().sum())
print("\n Дубликаты:", df.duplicated().sum())

plt.figure(figsize=(6,4))
sns.countplot(data=df, x='Revenue', palette='pastel')
plt.title('Распределение целевой переменной (Revenue)')
plt.xlabel('Совершил покупку')
plt.ylabel('Количество сессий')
plt.show()

ratio = df['Revenue'].value_counts(normalize=True) * 100
print(f" Дисбаланс классов: FALSE = {ratio[False]:.1f}%, TRUE = {ratio[True]:.1f}%")

num_cols = df.select_dtypes(include='number').columns.tolist()
corr_matrix = df[num_cols + ['Revenue']].corr(numeric_only=True)

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', vmin=-1, vmax=1,
            cbar_kws={'shrink': 0.8}, linewidths=0.5)
plt.title('Корреляционная матрица (числовые признаки)')
plt.show()

print(" Ключевые наблюдения:")
print("- PageValues имеет сильную положительную корреляцию с покупкой")
print("- BounceRates и ExitRates отрицательно связаны с конверсией")
print("- Длительность и количество просмотров продуктов также важны")

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
sns.boxplot(data=df, x='Revenue', y='PageValues', ax=axes[0], palette='Set2')
axes[0].set_title('PageValues vs Revenue')
sns.boxplot(data=df, x='Revenue', y='BounceRates', ax=axes[1], palette='Set2')
axes[1].set_title('BounceRates vs Revenue')
sns.boxplot(data=df, x='Revenue', y='ProductRelated_Duration', ax=axes[2], palette='Set2')
axes[2].set_title('ProductRelated Duration vs Revenue')
plt.tight_layout()
plt.show()

df_temp = df.copy()
df_temp['Revenue_int'] = df_temp['Revenue'].astype(int)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
month_order = ['Jan','Feb','Mar','Apr','May','June','Jul','Aug','Sep','Oct','Nov','Dec']
sns.barplot(data=df_temp, x='Month', y='Revenue_int', order=month_order, ax=axes[0], ci=None, palette='viridis')
axes[0].set_title('Конверсия по месяцам')
axes[0].tick_params(axis='x', rotation=45)

sns.barplot(data=df_temp, x='VisitorType', y='Revenue_int', ax=axes[1], ci=None, palette='muted')
axes[1].set_title('Конверсия по типу посетителя')
plt.tight_layout()
plt.show()

data = df.copy()

data['Weekend'] = data['Weekend'].astype(int)
data['Revenue'] = data['Revenue'].astype(int)

data = pd.get_dummies(data, columns=['Month', 'VisitorType'], drop_first=True)

X = data.drop('Revenue', axis=1)
y = data['Revenue']

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

scaler = StandardScaler()
num_features = X_train.select_dtypes(include='number').columns
X_train[num_features] = scaler.fit_transform(X_train[num_features])
X_val[num_features] = scaler.transform(X_val[num_features])
X_test[num_features] = scaler.transform(X_test[num_features])

print(f" Train: {X_train.shape[0]} | Val: {X_val.shape[0]} | Test: {X_test.shape[0]}")
print(f" Доля покупок в train: {y_train.mean():.3f}")

n_neg, n_pos = np.bincount(y_train)
total = len(y_train)
weight_for_0 = total / (2 * n_neg)
weight_for_1 = total / (2 * n_pos)
class_weights = {0: weight_for_0, 1: weight_for_1}
print(f" Веса классов: 0 → {weight_for_0:.3f}, 1 → {weight_for_1:.3f}")

lr = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
lr.fit(X_train, y_train)

rf = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)

xgb = XGBClassifier(
    n_estimators=200, max_depth=5, learning_rate=0.05,
    scale_pos_weight=weight_for_1/weight_for_0, random_state=42, eval_metric='logloss'
)
xgb.fit(X_train, y_train)

print(" Классические модели обучены")

def build_mlp(input_dim):
    model = Sequential([
        Dense(64, activation='relu', input_shape=(input_dim,)),
        BatchNormalization(),
        Dropout(0.3),
        Dense(32, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss='binary_crossentropy',
                  metrics=[tf.keras.metrics.AUC(name='auc')])
    return model

mlp = build_mlp(X_train.shape[1])
early_stop = EarlyStopping(monitor='val_auc', patience=7, restore_best_weights=True, mode='max')

history = mlp.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=50, batch_size=64,
    class_weight=class_weights,
    callbacks=[early_stop],
    verbose=1
)

plt.figure(figsize=(8,4))
plt.plot(history.history['auc'], label='Train AUC')
plt.plot(history.history['val_auc'], label='Val AUC')
plt.title('MLP Learning Curve (AUC)')
plt.xlabel('Epoch')
plt.ylabel('AUC')
plt.legend()
plt.show()
print(" Нейросеть обучена")

def evaluate_model(model, X, y, name, is_nn=False):
    if is_nn:
        probs = model.predict(X).ravel()
    else:
        probs = model.predict_proba(X)[:, 1]
    preds = (probs >= 0.5).astype(int)

    roc = roc_auc_score(y, probs)
    prec, rec, _ = precision_recall_curve(y, probs)
    pr_auc = auc(rec, prec)
    f1 = f1_score(y, preds)
    prec_score = precision_score(y, preds)
    rec_score = recall_score(y, preds)

    return {
        'Model': name, 'ROC-AUC': roc, 'PR-AUC': pr_auc,
        'Precision': prec_score, 'Recall': rec_score, 'F1': f1
    }, probs, preds

results = []
for mdl, nm, nn in [(lr, 'LogReg', False), (rf, 'RandomForest', False), (xgb, 'XGBoost', False), (mlp, 'MLP (Keras)', True)]:
    res, probs, preds = evaluate_model(mdl, X_test, y_test, nm, nn)
    results.append(res)

df_metrics = pd.DataFrame(results).set_index('Model')
display(df_metrics.style.highlight_max(color='lightgreen', axis=0))

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
for mdl, nm, nn in [(lr, 'LogReg', False), (rf, 'RF', False), (xgb, 'XGB', False), (mlp, 'MLP', True)]:
    _, probs, _ = evaluate_model(mdl, X_test, y_test, nm, nn)
    fpr, tpr, _ = roc_curve(y_test, probs)
    plt.plot(fpr, tpr, label=f"{nm} (AUC={auc(fpr,tpr):.3f})")
plt.plot([0,1], [0,1], 'k--')
plt.title('ROC Curves')
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.legend()

plt.subplot(1, 2, 2)
for mdl, nm, nn in [(lr, 'LogReg', False), (rf, 'RF', False), (xgb, 'XGB', False), (mlp, 'MLP', True)]:
    _, probs, _ = evaluate_model(mdl, X_test, y_test, nm, nn)
    prec, rec, _ = precision_recall_curve(y_test, probs)
    plt.plot(rec, prec, label=f"{nm} (PR-AUC={auc(rec,prec):.3f})")
plt.title('Precision-Recall Curves')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend()
plt.tight_layout()
plt.show()

cm = confusion_matrix(y_test, (xgb.predict_proba(X_test)[:,1] >= 0.5).astype(int))
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['No Purchase', 'Purchase'],
            yticklabels=['No Purchase', 'Purchase'])
plt.title('Confusion Matrix (XGBoost)')
plt.ylabel('True')
plt.xlabel('Predicted')
plt.show()