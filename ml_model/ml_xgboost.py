import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import xgboost as xgb # XGBoost をインポート

# データの読み込み
df = pd.read_csv("metabo_app/health_check_simulated.csv")

# 説明変数：1年目のデータ
X = df[[col for col in df.columns if "_1" in col]]

# 目的変数：メタボ判定
y = df["metabo"]

# データ分割（学習70%、テスト30%）
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# GridSearchCV を使ったハイパーパラメータチューニング
# XGBoost のパラメータ調整
param_grid = {
    'n_estimators': [100, 200, 300], # ブースティングする決定木の数
    'max_depth': [3, 6, 9], # 各決定木の最大深度
    'learning_rate': [0.01, 0.1, 0.2], # 各ブースティングステップにおける重みの縮小率
    'subsample': [0.7, 0.8, 0.9], # 決定木を構築する際のデータのサブサンプル比率
    'colsample_bytree': [0.7, 0.8, 0.9], # 決定木を構築する際の列（特徴量）のサブサンプル比率
    'gamma': [0, 0.1, 0.2], # 葉ノードの分割で必要な最小損失削減
    'reg_alpha': [0, 0.005, 0.01] # L1正則化項
}

# モデル構築
# objective='binary:logistic' は二値分類問題のロジスティック回帰
# use_label_encoder=False はXGBoostの将来のバージョンでの警告を避けるため
# eval_metric='logloss' は評価指標をログロスに設定
model = xgb.XGBClassifier(objective='binary:logistic', use_label_encoder=False, eval_metric='logloss', random_state=42)

grid_search = GridSearchCV(estimator=model, param_grid=param_grid,
                                cv=5, scoring='accuracy',
                                n_jobs=-1, verbose=2)

grid_search.fit(X_train, y_train)

# 最適なハイパーパラメータとスコアを表示
print("\n--- GridSearchCV 結果 ---")
print(f"ベストスコア (accuracy): {grid_search.best_score_:.4f}")
print("最適なハイパーパラメータ:", grid_search.best_params_)

# 最適なモデルを取得
best_xgb_model = grid_search.best_estimator_

# 予測
y_pred_xgb = best_xgb_model.predict(X_test)

# 評価
print("\n--- モデル評価 ---")
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_xgb))
print("\nClassification Report:")
print(classification_report(y_test, y_pred_xgb))
print("\nAccuracy Score:", accuracy_score(y_test, y_pred_xgb))

###################################################################################
#結果
#Confusion Matrix:
#[[211  11]
# [ 29  49]]
#
#Classification Report:
#              precision    recall  f1-score   support
#
#           0       0.88      0.95      0.91       222
#           1       0.82      0.63      0.71        78
#
#    accuracy                           0.87       300
#   macro avg       0.85      0.79      0.81       300
#weighted avg       0.86      0.87      0.86       300
#
#
#Accuracy Score: 0.8666666666666667