import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
)
from sklearn.tree import plot_tree


def load_and_prepare_datasets(basis_path, top_path, flop_path):
    """Schritt 1 & 2: Datensätze laden, Spalten harmonisieren und Targets erstellen."""

    # 1. Datensätze einlesen
    print("Lade Datensätze...")
    df_basis = pd.read_csv(basis_path)
    df_top = pd.read_csv(top_path)
    df_flop = pd.read_csv(flop_path)

    # 2. Harmonisierung & Target Engineering: Basis-Datensatz (Movies_dataset.csv)
    # Erwartete Spalten laut Datei: id, title, overview, popularity, release_date, vote_average, vote_count
    df_basis_prep = pd.DataFrame()
    df_basis_prep["film_title"] = df_basis["title"]
    df_basis_prep["imdb_rating"] = df_basis["vote_average"]
    df_basis_prep["vote_count"] = df_basis["vote_count"]
    # Jahr aus dem Datum extrahieren (YYYY-MM-DD -> YYYY)
    df_basis_prep["release_year"] = pd.to_datetime(
        df_basis["release_date"], errors="coerce"
    ).dt.year
    df_basis_prep["genre"] = "Unknown" # Platzhalter, falls nicht im Basis-Set direkt gepflegt
    df_basis_prep["duration_min"] = np.nan # Wird später per Median im Gesamtset gefüllt
    df_basis_prep["is_top_100"] = 0 # Standard-Label für die Klassifikation

    # 3. Harmonisierung & Target Engineering: Top-100 (imdb_top_movies.csv)
    # Erwartete Spalten laut Datei: Rank, Title, Description, Genre, Rating, Year, IMDB_ID...
    df_top_prep = pd.DataFrame()
    df_top_prep["film_title"] = df_top["Title"]
    df_top_prep["imdb_rating"] = df_top["Rating"]
    df_top_prep["vote_count"] = 150000 # Synthetischer hoher Wert, da in dieser Datei kein vote_count vorhanden ist
    df_top_prep["release_year"] = df_top["Year"]
    df_top_prep["genre"] = df_top["Genre"]
    df_top_prep["duration_min"] = np.nan # Wird im Gesamtschritt mitberechnet
    df_top_prep["is_top_100"] = 1 # Wichtiges Ziel-Label für Klassifikation

    # 4. Harmonisierung & Target Engineering: Flop-100 (lowest_ranked_movies_data.csv)
    # Erwartete Spalten laut Datei: rank, name, year, certification, duration, rating, review_count...
    df_flop_prep = pd.DataFrame()
    df_flop_prep["film_title"] = df_flop["name"]
    df_flop_prep["imdb_rating"] = df_flop["rating"]

    # Bereinigung des review_count (z.B. '94K' -> 94000)
    def clean_review_count(val):
        if pd.isna(val):
            return 0
        val = str(val).upper().strip()
        if "K" in val:
            return int(float(val.replace("K", "")) * 1000)
        return int(float(val))

    df_flop_prep["vote_count"] = df_flop["review_count"].apply(clean_review_count)
    df_flop_prep["release_year"] = df_flop["year"]
    df_flop_prep["genre"] = df_flop["genre"]

    # Textuelle Laufzeit konvertieren (z.B. "1h 27m" -> 87) via Regex
    def convert_duration_to_minutes(duration_str):
        if pd.isna(duration_str):
            return np.nan
        duration_str = str(duration_str)
        hours = re.search(r"(\d+)\s*h", duration_str)
        minutes = re.search(r"(\d+)\s*m", duration_str)
        total_mins = 0
        if hours:
            total_mins += int(hours.group(1)) * 60
        if minutes:
            total_mins += int(minutes.group(1))
        return total_mins if total_mins > 0 else np.nan

    df_flop_prep["duration_min"] = df_flop["duration"].apply(
        convert_duration_to_minutes
    )
    df_flop_prep["is_top_100"] = 0 # Flops gehören nicht zu den Top-100

    return df_basis_prep, df_top_prep, df_flop_prep


def pipeline_data_preparation(df_basis, df_top, df_flop):
    """Schritt 3 bis 7: Zusammenführen, Bereinigen, Imputieren und Splitten."""

    # Schritt 3: Vertikale Fusionierung (Concat)
    print("Führe Datensätze vertikal zusammen...")
    df_all = pd.concat([df_top, df_flop, df_basis], ignore_index=True)
    # Duplikatbereinigung (Priorisierung der echten Top-100 Einträge)
    df_all = df_all.sort_values(by="is_top_100", ascending=False)
    df_all = df_all.drop_duplicates(subset=["film_title", "release_year"], keep="first")

    # Schritt 4: Filterung von Biases (Mindestanzahl an Nutzerbewertungen)
    # Schließt Filme aus, die durch zu wenige Stimmen das Rating verzerren
    min_votes_threshold = 100
    df_all = df_all[df_all["vote_count"] >= min_votes_threshold]
    # Schritt 5 & 6: Umgang mit fehlenden Einträgen (Missing Values)
    # Kategorialer Platzhalter
    df_all["genre"] = df_all["genre"].fillna("Unknown")

    # Numerischer Median für Laufzeit (wird berechnet, um NaNs zu füllen)
    global_duration_median = df_all["duration_min"].median()
    if pd.isna(global_duration_median):
        global_duration_median = 105 # Standard-Ausfallwert, falls Median nicht berechenbar
    df_all["duration_min"] = df_all["duration_min"].fillna(global_duration_median)
    # Zeilen löschen, bei denen die kritischen numerischen Werte (Jahr/Rating) fehlen
    df_all = df_all.dropna(subset=["release_year", "imdb_rating"])

    # Feature Selection: Unstrukturierte Textspalten/IDs entfernen (Vermeidung von Overfitting)
    # Der Filmtitel wird ab hier nicht mehr für mathematische Berechnungen benötigt
    df_features = df_all[
        ["release_year", "vote_count", "duration_min", "genre"]
    ].copy()
    # Kategoriale Variablen umwandeln (One-Hot-Encoding für Genres)
    df_features = pd.get_dummies(df_features, columns=["genre"], drop_first=True)

    # Zielvariablen trennen
    y_regression = df_all["imdb_rating"]
    y_classification = df_all["is_top_100"]

    # Schritt 7: Train-Test-Split (Vermeidung von Data Leakage)
    print("Führe Train-Test-Split durch (80:20)...")
    X_train, X_test, y_train_reg, y_test_reg, y_train_clf, y_test_clf = train_test_split(
        df_features,
        y_regression,
        y_classification,
        test_size=0.2,
        random_state=42,
        stratify=y_classification,
    )

    # Normalisierung / Skalierung (Erst NACH dem Split lernen!)
    print("Normalisiere numerische Merkmale mit StandardScaler...")
    scaler = StandardScaler()
    # Die Spalten, die skaliert werden sollen (wichtig, da One-Hot-Spalten bereits 0 oder 1 sind)
    num_cols = ["release_year", "vote_count", "duration_min"]

     # Fit & Transform auf das Trainingsset
    X_train_scaled = X_train.copy()
    X_train_scaled[num_cols] = scaler.fit_transform(X_train[num_cols])

    # Reines Transformieren auf das Testset (Kein Fit!)
    X_test_scaled = X_test.copy()
    X_test_scaled[num_cols] = scaler.transform(X_test[num_cols])

    print("--- Datenaufbereitung erfolgreich abgeschlossen! ---")
    return X_train_scaled, X_test_scaled, y_train_clf, y_test_clf


def train_and_evaluate_models(X_train, y_train, X_test, y_test):
    """Trainiert Modelle, optimiert Hyperparameter und vergleicht sie."""
    print("\n" + "="*50)
    print("MODELL-TRAINING & EVALUATION")
    print("="*50)

    # Logistische Regression (Baseline Modell)
    print("\nTrainiere Logistische Regression (Baseline)...")
    logreg = LogisticRegression(max_iter=1000, random_state=42)
    logreg.fit(X_train, y_train)
    y_pred_logreg = logreg.predict(X_test)
    acc_logreg = accuracy_score(y_test, y_pred_logreg)
    print(f"Genauigkeit (Accuracy) LogReg: {acc_logreg:.4f}")

    # Random Forest (Komplexes Modell)
    print("\nTrainiere Random Forest (Standard)...")
    rf_base = RandomForestClassifier(random_state=42)
    rf_base.fit(X_train, y_train)
    y_pred_rf_base = rf_base.predict(X_test)
    acc_rf_base = accuracy_score(y_test, y_pred_rf_base)
    print(f"Genauigkeit (Accuracy) Random Forest (Base): {acc_rf_base:.4f}")

    # Hyperparameter-Optimierung für Random Forest
    print("\nStarte Hyperparameter-Optimierung für Random Forest via GridSearchCV...")
    # Suchraum für Parameter (klein gehalten, damit es schnell durchläuft)
    param_grid = {
        'n_estimators': [50, 100, 150],  # Anzahl der Bäume
        'max_depth': [None, 10, 20],     # Maximale Tiefe der Bäume
        'min_samples_split': [2, 5]      # Mindestanzahl an Samples zum Teilen
    }

    grid_search = GridSearchCV(
        estimator=RandomForestClassifier(random_state=42),
        param_grid=param_grid,
        cv=3, # 3-fache Kreuzvalidierung
        scoring='accuracy',
        n_jobs=-1 # Nutzt alle CPU-Kerne
    )
    
    grid_search.fit(X_train, y_train)
    best_rf = grid_search.best_estimator_
    
    print(f"Beste gefundene Parameter: {grid_search.best_params_}")
    
    y_pred_rf_tuned = best_rf.predict(X_test)
    acc_rf_tuned = accuracy_score(y_test, y_pred_rf_tuned)
    print(f"Genauigkeit (Accuracy) Random Forest (Tuned): {acc_rf_tuned:.4f}")

    # Detaillierter Report für das beste Modell
    print("\nKlassifikationsbericht für das beste Modell (Tuned Random Forest):")
    print(classification_report(y_test, y_pred_rf_tuned))

    return best_rf, y_pred_rf_tuned


def plot_visualizations(model, X_train, X_test, y_test, y_pred):
    """Erstellt alle benötigten Visualisierungen für den Projektbericht."""
    print("\nErstelle Visualisierungen... (Fenster schließen, um fortzufahren)")
    sns.set_theme(style="whitegrid")

    # Visualisierung des Modells: Feature Importance
    plt.figure(figsize=(10, 6))
    importances = model.feature_importances_
    # Die wichtigsten 10 Features filtern
    indices = np.argsort(importances)[::-1][:10]
    top_features = X_train.columns[indices]
    
    sns.barplot(x=importances[indices], y=top_features, palette="viridis")
    plt.title("Visualisierung des Modells: Feature Importance (Top 10)")
    plt.xlabel("Wichtigkeit des Merkmals")
    plt.ylabel("Merkmal (Feature)")
    plt.tight_layout()
    plt.show()

    # Visualisierung eines einzelnen Baumes aus dem Random Forest
    plt.figure(figsize=(20, 10))
    plot_tree(model.estimators_[0], feature_names=X_train.columns, 
              class_names=["Flop/Normal", "Top 100"], filled=True, max_depth=3, rounded=True)
    plt.title("Visualisierung des Modells: Ein Entscheidungsbaum des Random Forests (Tiefe max. 3)")
    plt.show()

    # Visualisierung der Vorhersagen: Konfusionsmatrix (Confusion Matrix)
    plt.figure(figsize=(6, 5))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Normal', 'Top 100'], yticklabels=['Normal', 'Top 100'])
    plt.title("Vorhersagen: Konfusionsmatrix (Testdaten)")
    plt.xlabel("Vorhergesagte Klasse (KI)")
    plt.ylabel("Tatsächliche Klasse (Realität)")
    plt.tight_layout()
    plt.show()

    # Visualisierung der Vorhersagen: ROC Curve
    plt.figure(figsize=(7, 6))
    # Wahrscheinlichkeiten der Vorhersage abrufen
    y_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Kurve (Fläche = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Falsch-Positiv-Rate (False Positive Rate)')
    plt.ylabel('Richtig-Positiv-Rate (True Positive Rate)')
    plt.title('Vorhersagen: ROC-Kurve')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.show()


# =====================================================================
# HAUPTPROGRAMM / AUSFÜHRUNG
# =====================================================================
if __name__ == "__main__":

    # Hier die Pfade anpassen
    PATH_BASIS = "/Users/Lian/Documents/Movies_dataset.csv"
    PATH_TOP = "/Users/Lian/Documents/imdb_top_movies.csv"
    PATH_FLOP = "/Users/Lian/Documents/lowest_ranked_movies_data.csv"

    # Prüfen, ob die Dateien am angegebenen Ort existieren
    files_exist = True
    for p in [PATH_BASIS, PATH_TOP, PATH_FLOP]:
        if not os.path.exists(p):
            print(f"FEHLER: Datei '{p}' nicht gefunden. Bitte prüfen Sie den Pfad!")
            files_exist = False

    if files_exist:
        # Datenaufbereitung
        df_b, df_t, df_f = load_and_prepare_datasets(PATH_BASIS, PATH_TOP, PATH_FLOP)
        X_train_final, X_test_final, y_train_c, y_test_c = pipeline_data_preparation(df_b, df_t, df_f)
        
        # Bereinigte Daten speichern
        # pd.DataFrame(X_train_final).to_csv("Train_Clean.csv", index=False)

        # Modelle trainieren, evaluieren und optimieren
        best_model, predictions = train_and_evaluate_models(
            X_train_final, y_train_c, X_test_final, y_test_c
        )

        # Modelle und Vorhersagen visualisieren
        plot_visualizations(best_model, X_train_final, X_test_final, y_test_c, predictions)