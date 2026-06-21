import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    auc
)
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
from sklearn.tree import plot_tree

sns.set_theme(style="whitegrid")



# merged csv laden
df = pd.read_csv("movies_merged.csv")
print(df.info())
print(df.head())



# FEATURES ERSTELLEN
df["genre"] = df["genre"].fillna("Unknown")
X = df[["release_year", "vote_count", "duration_min", "genre"]].copy()
X = pd.get_dummies(X, columns=["genre"], drop_first=True)                  # Genre in Dummy-Variablen umwandeln

# Zielvariablen
y_top = df["is_top_100"]
y_rating = df["imdb_rating"]



# TRAIN-TEST-SPLIT
# Klassifikation
X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(
    X,
    y_top,
    test_size=0.2,
    random_state=42,
    stratify=y_top)

# Regression
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X,
    y_rating,
    test_size=0.2,
    random_state=42)



# NORMALISIERUNG
scaler_clf = StandardScaler()
scaler_reg = StandardScaler()

num_cols = ["release_year", "vote_count", "duration_min"]

X_train_clf = X_train_clf.copy()
X_test_clf = X_test_clf.copy()

X_train_reg = X_train_reg.copy()
X_test_reg = X_test_reg.copy()

X_train_clf[num_cols] = scaler_clf.fit_transform(X_train_clf[num_cols])
X_test_clf[num_cols] = scaler_clf.transform(X_test_clf[num_cols])

X_train_reg[num_cols] = scaler_reg.fit_transform(X_train_reg[num_cols])
X_test_reg[num_cols] = scaler_reg.transform(X_test_reg[num_cols])



# KLASSIFIKATION
print("KLASSIFIKATION")

# Logistische Regression
logreg = LogisticRegression(max_iter=1000, random_state=42)
logreg.fit(X_train_clf, y_train_clf)
y_pred_log = logreg.predict(X_test_clf)

print("\nLogistische Regression")
print("Accuracy:", accuracy_score(y_test_clf, y_pred_log))
print(classification_report(y_test_clf, y_pred_log))


# Random Forest
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train_clf, y_train_clf)
y_pred_rf = rf.predict(X_test_clf)

print("\nRandom Forest")
print("Accuracy:", accuracy_score(y_test_clf, y_pred_rf))
print(classification_report(y_test_clf, y_pred_rf))



# HYPERPARAMETER-TUNING
print("\nGridSearch startet...")

param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [10, 20, None],
    "min_samples_split": [2, 5]}

grid = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=-1)

grid.fit(X_train_clf, y_train_clf)
best_rf = grid.best_estimator_

print("Beste Parameter:")
print(grid.best_params_)

y_pred_best = best_rf.predict(X_test_clf)

print("\nAccuracy Tuned RF:", accuracy_score(y_test_clf, y_pred_best))
print(classification_report(y_test_clf, y_pred_best))



# REGRESSION
print("REGRESSION")

# Lineare Regression
linreg = LinearRegression()
linreg.fit(X_train_reg, y_train_reg)
pred_lin = linreg.predict(X_test_reg)

print("\nLineare Regression")
print("MAE:", mean_absolute_error(y_test_reg, pred_lin))
print("RMSE:", np.sqrt(mean_squared_error(y_test_reg, pred_lin)))
print("R²:", r2_score(y_test_reg, pred_lin))



# Random Forest Regressor
rf_reg = RandomForestRegressor(random_state=42)
rf_reg.fit(X_train_reg, y_train_reg)
pred_rf = rf_reg.predict(X_test_reg)

print("\nRandom Forest Regressor")
print("MAE:", mean_absolute_error(y_test_reg, pred_rf))
print("RMSE:", np.sqrt(mean_squared_error(y_test_reg, pred_rf)))
print("R²:", r2_score(y_test_reg, pred_rf))



# CLUSTERING
print("CLUSTERING")

X_cluster = df[["imdb_rating", "vote_count", "duration_min", "release_year"]].copy()

scaler_cluster = StandardScaler()
X_cluster_scaled = scaler_cluster.fit_transform(X_cluster)

inertia = []

for k in range(1, 11):
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    model.fit(X_cluster_scaled)
    inertia.append(model.inertia_)

plt.figure(figsize=(8, 6))
plt.plot(range(1, 11), inertia, marker="o")
plt.title("Elbow-Methode")
plt.xlabel("Anzahl Cluster")
plt.ylabel("Inertia")
plt.show()


# Beispiel: 3 Cluster
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_cluster_scaled)
df["cluster"] = clusters

plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x="imdb_rating", y="vote_count", hue="cluster", palette="Set2")
plt.title("K-Means Cluster")
plt.show()



# VISUALISIERUNG KLASSIFIKATION
# Konfusionsmatrix

cm = confusion_matrix(y_test_clf, y_pred_best)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Normal", "Top100"], yticklabels=["Normal", "Top100"])
plt.title("Konfusionsmatrix")
plt.xlabel("Vorhersage")
plt.ylabel("Wahr")
plt.show()


# ROC-Kurve
y_prob = best_rf.predict_proba(X_test_clf)[:, 1]
fpr, tpr, _ = roc_curve(y_test_clf, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(7, 6))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
plt.plot([0, 1], [0, 1], "--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC-Kurve")
plt.legend()
plt.show()



# FEATURE IMPORTANCE
importances = best_rf.feature_importances_
indices = np.argsort(importances)[::-1][:10]
top_features = X_train_clf.columns[indices]

plt.figure(figsize=(10, 6))
sns.barplot(x=importances[indices],y=top_features)
plt.title("Top 10 Feature Importances")
plt.show()



# EINEN BAUM VISUALISIEREN
plt.figure(figsize=(20, 10))
plot_tree(best_rf.estimators_[0], feature_names=X_train_clf.columns, class_names=["Normal", "Top100"], filled=True, max_depth=3, rounded=True)
plt.title("Ein Entscheidungsbaum des Random Forest")
plt.show()



# REGRESSIONSVORHERSAGE
plt.figure(figsize=(7, 6))
plt.scatter(y_test_reg, pred_rf, alpha=0.5)
plt.xlabel("Tatsächliches IMDb-Rating")
plt.ylabel("Vorhergesagtes IMDb-Rating")
plt.title("Vorhersage vs. Realität")
plt.show()

