import os
import re
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_and_prepare_datasets(basis_path, top_path, flop_path):
    """Schritt 1 & 2: Datensätze laden, Spalten harmonisieren und Targets erstellen."""

    print("Lade Datensätze...")
    df_basis = pd.read_csv(basis_path)
    df_top = pd.read_csv(top_path)
    df_flop = pd.read_csv(flop_path)

    # Basis-Datensatz (Movies_dataset.csv)
    df_basis_prep = pd.DataFrame()
    df_basis_prep["film_title"] = df_basis["title"]
    df_basis_prep["imdb_rating"] = df_basis["vote_average"]
    df_basis_prep["vote_count"] = df_basis["vote_count"]
    df_basis_prep["release_year"] = pd.to_datetime(
        df_basis["release_date"], errors="coerce"
    ).dt.year
    df_basis_prep["genre"] = "Unknown"
    df_basis_prep["duration_min"] = np.nan
    df_basis_prep["is_top_100"] = 0

    # Top-100 (imdb_top_movies.csv)
    df_top_prep = pd.DataFrame()
    df_top_prep["film_title"] = df_top["Title"]
    df_top_prep["imdb_rating"] = df_top["Rating"]
    df_top_prep["vote_count"] = 150000  # Synthetischer hoher Wert
    df_top_prep["release_year"] = df_top["Year"]
    df_top_prep["genre"] = df_top["Genre"]
    df_top_prep["duration_min"] = np.nan
    df_top_prep["is_top_100"] = 1

    # Flop-100 (lowest_ranked_movies_data.csv)
    df_flop_prep = pd.DataFrame()
    df_flop_prep["film_title"] = df_flop["name"]
    df_flop_prep["imdb_rating"] = df_flop["rating"]

    def clean_review_count(val):
        if pd.isna(val):
            return 0
        val = str(val).upper().strip()
        if "K" in val:
            return int(float(val.replace("K", "")) * 1000)
        return int(float(val))

    df_flop_prep["vote_count"] = df_flop["review_count"].apply(
        clean_review_count
    )
    df_flop_prep["release_year"] = df_flop["year"]
    df_flop_prep["genre"] = df_flop["genre"]

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
    df_flop_prep["is_top_100"] = 0

    return df_basis_prep, df_top_prep, df_flop_prep


def pipeline_data_preparation(df_basis, df_top, df_flop):
    """Schritt 3 bis 7: Zusammenführen, Bereinigen, Imputieren und Splitten."""

    print("Führe Datensätze vertikal zusammen...")
    # Hier wurden die Variablennamen harmonisiert, um den NameError zu beheben
    df_all = pd.concat([df_top, df_flop, df_basis], ignore_index=True)

    df_all = df_all.sort_values(by="is_top_100", ascending=False)
    df_all = df_all.drop_duplicates(
        subset=["film_title", "release_year"], keep="first"
    )

    min_votes_threshold = 100
    df_all = df_all[df_all["vote_count"] >= min_votes_threshold]

    df_all["genre"] = df_all["genre"].fillna("Unknown")

    global_duration_median = df_all["duration_min"].median()
    if pd.isna(global_duration_median):
        global_duration_median = 105
    df_all["duration_min"] = df_all["duration_min"].fillna(
        global_duration_median
    )

    df_all = df_all.dropna(subset=["release_year", "imdb_rating"])

    df_features = df_all[
        ["release_year", "vote_count", "duration_min", "genre"]
    ].copy()
    df_features = pd.get_dummies(
        df_features, columns=["genre"], drop_first=True
    )

    y_regression = df_all["imdb_rating"]
    y_classification = df_all["is_top_100"]

    print("Führe Train-Test-Split durch (80:20)...")
    X_train, X_test, y_train_reg, y_test_reg, y_train_clf, y_test_clf = (
        train_test_split(
            df_features,
            y_regression,
            y_classification,
            test_size=0.2,
            random_state=42,
            stratify=y_classification,
        )
    )

    print("Normalisiere numerische Merkmale mit StandardScaler...")
    scaler = StandardScaler()
    num_cols = ["release_year", "vote_count", "duration_min"]

    X_train_scaled = X_train.copy()
    X_train_scaled[num_cols] = scaler.fit_transform(X_train[num_cols])

    X_test_scaled = X_test.copy()
    X_test_scaled[num_cols] = scaler.transform(X_test[num_cols])

    print("--- Datenaufbereitung erfolgreich abgeschlossen! ---")
    return (
        X_train_scaled,
        X_test_scaled,
        y_train_reg,
        y_test_reg,
        y_train_clf,
        y_test_clf,
    )

# HAUPTPROGRAMM / AUSFÜHRUNG

if __name__ == "__main__":

    # Hier die Pfade anpassen
    PATH_BASIS = "/Users/Lian/Documents/Movies_dataset.csv"
    PATH_TOP = "/Users/Lian/Documents/imdb_top_movies.csv"
    PATH_FLOP = "/Users/Lian/Documents/lowest_ranked_movies_data.csv"

    # Prüfen, ob die Dateien am angegebenen Ort existieren
    files_exist = True
    for p in [PATH_BASIS, PATH_TOP, PATH_FLOP]:
        if not os.path.exists(p):
            print(
                f"FEHLER: Datei '{p}' nicht gefunden. Bitte prüfen Sie den Pfad!"
            )
            files_exist = False

    if files_exist:
        # Pipeline ausführen
        df_b, df_t, df_f = load_and_prepare_datasets(
            PATH_BASIS, PATH_TOP, PATH_FLOP
        )
        (
            X_train_final,
            X_test_final,
            y_train_r,
            y_test_r,
            y_train_c,
            y_test_c,
        ) = pipeline_data_preparation(df_b, df_t, df_f)

        print("\n=== Starte Modell-Training und Evaluation ===")

        # MODELL 1: Logistische Regression
        #Komplexität: Sehr gering. Es handelt sich um ein lineares Modell, das einfach zu interpretieren ist. Es berechnet für jedes Feature ein Gewicht (Koeffizient)."
        #Performanz: Dient als hervorragende Baseline. Wenn die Grenze zwischen Top-Filmen und normalen Filmen (z. B. anhand der Stimmenanzahl vote_count) annähernd linear verläuft, liefert es solide und extrem schnelle Ergebnisse. Es neigt kaum zu Overfitting.'
        print("\n[Modell 1] Trainiere Logistische Regression...")
        log_reg = LogisticRegression(
            random_state=42, class_weight="balanced", max_iter=1000
        )
        log_reg.fit(X_train_final, y_train_c)

        y_pred_log = log_reg.predict(X_test_final)
        y_proba_log = log_reg.predict_proba(X_test_final)[:, 1]

        # MODELL 2: Random Forest
        #Komplexität: Mittel bis hoch. Ein Ensemble-Modell, das aus vielen verschiedenen Entscheidungsbäumen besteht. Es kann nicht-lineare Zusammenhänge und komplexe Interaktionen zwischen Features (z. B. Kombination aus bestimmtem Genre und Laufzeit) ohne manuelle Transformationen lernen.
        #Performanz: Erwartungsgemäß sehr hoch. Random Forests sind sehr robust gegenüber Ausreißern und können die ungleichen Klassenverhältnisse (nur wenige Top-100 Filme im Vergleich zu normalen Filmen) meist deutlich besser abbilden als lineare Modelle.
        print("[Modell 2] Trainiere Random Forest...")
        rand_forest = RandomForestClassifier(
            random_state=42, class_weight="balanced", n_estimators=100
        )
        rand_forest.fit(X_train_final, y_train_c)

        y_pred_rf = rand_forest.predict(X_test_final)
        y_proba_rf = rand_forest.predict_proba(X_test_final)[:, 1]

        # EVALUATION
        print("\n" + "=" * 40)
        print("EVALUATIONSERGEBNISSE: LOGISTISCHE REGRESSION")
        print("=" * 40)
        print(classification_report(y_test_c, y_pred_log))
        print(f"ROC-AUC Score: {roc_auc_score(y_test_c, y_proba_log):.4f}")

        print("\n" + "=" * 40)
        print("EVALUATIONSERGEBNISSE: RANDOM FOREST")
        print("=" * 40)
        print(classification_report(y_test_c, y_pred_rf))
        print(f"ROC-AUC Score: {roc_auc_score(y_test_c, y_proba_rf):.4f}")

        # ------------------------------------------
        # VISUALISIERUNG
        # ------------------------------------------
        print("\nGeneriere Visualisierungen...")
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))

        # 1. Konfusionsmatrix Logistische Regression
        cm_log = confusion_matrix(y_test_c, y_pred_log)
        ConfusionMatrixDisplay(
            confusion_matrix=cm_log, display_labels=["Normal/Flop", "Top 100"]
        ).plot(ax=axes[0, 0], cmap="Blues", values_format="d")
        axes[0, 0].set_title("Konfusionsmatrix: Logistische Regression")

        # 2. Konfusionsmatrix Random Forest
        cm_rf = confusion_matrix(y_test_c, y_pred_rf)
        ConfusionMatrixDisplay(
            confusion_matrix=cm_rf, display_labels=["Normal/Flop", "Top 100"]
        ).plot(ax=axes[0, 1], cmap="Greens", values_format="d")
        axes[0, 1].set_title("Konfusionsmatrix: Random Forest")

        # 3. ROC-Kurve
        fpr_log, tpr_log, _ = roc_curve(y_test_c, y_proba_log)
        fpr_rf, tpr_rf, _ = roc_curve(y_test_c, y_proba_rf)

        axes[1, 0].plot(
            fpr_log,
            tpr_log,
            label=f"Log. Regression (AUC = {roc_auc_score(y_test_c, y_proba_log):.2f})",
            color="blue",
        )
        axes[1, 0].plot(
            fpr_rf,
            tpr_rf,
            label=f"Random Forest (AUC = {roc_auc_score(y_test_c, y_proba_rf):.2f})",
            color="green",
        )
        axes[1, 0].plot([0, 1], [0, 1], "k--", label="Zufall (AUC = 0.50)")
        axes[1, 0].set_xlabel("False Positive Rate")
        axes[1, 0].set_ylabel("True Positive Rate")
        axes[1, 0].set_title("ROC-Kurve im Vergleich")
        axes[1, 0].legend(loc="lower right")
        axes[1, 0].grid(True)

        # 4. Feature Importance
        importances = rand_forest.feature_importances_
        indices = np.argsort(importances)[::-1][:10]
        feature_names = X_train_final.columns

        sns.barplot(
            x=importances[indices],
            y=feature_names[indices],
            ax=axes[1, 1],
            palette="viridis",
            hue=feature_names[indices],
            legend=False,
        )
        axes[1, 1].set_title("Top 10 wichtige Features (Random Forest)")
        axes[1, 1].set_xlabel("Relative Wichtigkeit")

        plt.tight_layout()
        plt.show()