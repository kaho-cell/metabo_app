import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# データの読み込み
df = pd.read_csv("予測アプリ2/health_check_simulated.csv")

# 説明変数：1年目のデータ
X = df[[col for col in df.columns if "_1" in col]]

# 目的変数：メタボ判定
y = df["metabo"]

# データ分割（学習70%、テスト30%）
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# モデル構築
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# 予測
y_pred = model.predict(X_test)

# 評価
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("\nAccuracy Score:", accuracy_score(y_test, y_pred))
