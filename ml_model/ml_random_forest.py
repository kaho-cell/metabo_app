import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import numpy as np # RandomizedSearchCVで均等な分布を作成するために使用
from sklearn.preprocessing import StandardScaler # 標準化のために追加


# データの読み込み
df = pd.read_csv("metabo_app/health_check_simulated.csv")

# 説明変数：1年目のデータ
X = df[[col for col in df.columns if "_1" in col]]

# 目的変数：メタボ判定
y = df["metabo"]

# データ分割（学習70%、テスト30%）
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# --- 説明変数の標準化 ---
scaler = StandardScaler()
X_train_scaled  = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#データフレームに適用
X_train = pd.DataFrame(X_train_scaled, columns=X_train.columns)
X_test = pd.DataFrame(X_test_scaled, columns=X_test.columns)

#GridSearchCV を使ったハイパーパラメータチューニング

#パラメータの調整
param_grid = {
    'n_estimators': [100, 200, 300], # 決定木の数
    'max_depth': [None, 10, 20, 30], # 各決定木の最大深度
    'min_samples_split': [2, 5, 10], # ノードを分割するために必要な最小サンプル数
    'min_samples_leaf': [1, 2, 4],   # リーフノードに必要な最小サンプル数
    'max_features': ['sqrt', 'log2', 0.8] # 各決定木が考慮する特徴量の最大数
}

# モデル構築
model = RandomForestClassifier(random_state=42)
random_search = GridSearchCV(estimator=model, param_grid=param_grid, 
                                   cv=5, scoring='accuracy', 
                                   n_jobs=-1, verbose=2)

random_search.fit(X_train, y_train)

# 最適なハイパーパラメータとスコアを表示
print("\n--- RandomizedSearchCV 結果 ---")
print(f"ベストスコア (accuracy): {random_search.best_score_:.4f}")
print("最適なハイパーパラメータ:", random_search.best_params_)

# 最適なモデルを取得
best_random_model = random_search.best_estimator_

# 予測
y_pred_random = best_random_model.predict(X_test)

# 評価
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_random))
print("\nClassification Report:")
print(classification_report(y_test, y_pred_random))
print("\nAccuracy Score:", accuracy_score(y_test, y_pred_random))

###########################################################
#結果
#ベストスコア (accuracy): 0.8900
#最適なハイパーパラメータ: {'max_depth': 10, 'max_features': 'sqrt', 'min_samples_leaf': 4, 'min_samples_split': 2, 'n_estimators': 300}
#Confusion Matrix:
#[[212  10]
# [ 29  49]]
#
#Classification Report:
#              precision    recall  f1-score   support
#
#          0       0.88      0.95      0.92       222 
#           1       0.83      0.63      0.72        78 
#    accuracy                           0.87       300
#   macro avg       0.86      0.79      0.82       300
#weighted avg       0.87      0.87      0.86       300
#
#
#Accuracy Score: 0.87