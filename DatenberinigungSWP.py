import os
import re
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


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
    df_basis_prep["genre"] = (
        "Unknown"  # Platzhalter, falls nicht im Basis-Set direkt gepflegt
    )
    df_basis_prep["duration_min"] = (
        np.nan
    )  # Wird später per Median im Gesamtset gefüllt
    df_basis_prep["is_top_100"] = 0  # Standard-Label für die Klassifikation

    # 3. Harmonisierung & Target Engineering: Top-100 (imdb_top_movies.csv)
    # Erwartete Spalten laut Datei: Rank, Title, Description, Genre, Rating, Year, IMDB_ID...
    df_top_prep = pd.DataFrame()
    df_top_prep["film_title"] = df_top["Title"]
    df_top_prep["imdb_rating"] = df_top["Rating"]
    df_top_prep["vote_count"] = (
        150000  # Synthetischer hoher Wert, da in dieser Datei kein vote_count vorhanden ist
    )
    df_top_prep["release_year"] = df_top["Year"]
    df_top_prep["genre"] = df_top["Genre"]
    df_top_prep["duration_min"] = (
        np.nan
    )  # Wird im Gesamtschritt mitberechnet
    df_top_prep["is_top_100"] = 1  # Wichtiges Ziel-Label für Klassifikation

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

    df_flop_prep["vote_count"] = df_flop["review_count"].apply(
        clean_review_count
    )
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
    df_flop_prep["is_top_100"] = 0  # Flops gehören nicht zu den Top-100

    return df_basis_prep, df_top_prep, df_flop_prep


def pipeline_data_preparation(df_basis, df_top, df_flop):
    """Schritt 3 bis 7: Zusammenführen, Bereinigen, Imputieren und Splitten."""

    # Schritt 3: Vertikale Fusionierung (Concat)
    print("Führe Datensätze vertikal zusammen...")
    df_all = pd.concat([df_top, df_flop, df_basis], ignore_index=True)

    # Duplikatbereinigung (Priorisierung der echten Top-100 Einträge)
    df_all = df_all.sort_values(by="is_top_100", ascending=False)
    df_all = df_all.drop_duplicates(
        subset=["film_title", "release_year"], keep="first"
    )

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
        global_duration_median = (
            105  # Standard-Ausfallwert, falls Median nicht berechenbar
        )
    df_all["duration_min"] = df_all["duration_min"].fillna(
        global_duration_median
    )

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
    print(f"Größe Trainingsdatensatz: {X_train_scaled.shape[0]} Filme")
    print(f"Größe Testdatensatz: {X_test_scaled.shape[0]} Filme")

    return (
        X_train_scaled,
        X_test_scaled,
        y_train_reg,
        y_test_reg,
        y_train_clf,
        y_test_clf,
    )


# =====================================================================
# HAUPTPROGRAMM / AUSFÜHRUNG
# =====================================================================
if __name__ == "__main__":

    #Hier die Pfade anpassen
    PATH_BASIS = "/Users/Lian/Documents/Movies_dataset.csv"
    PATH_TOP = "/Users/Lian/Documents/imdb_top_movies.csv"
    PATH_FLOP = (
        "/Users/Lian/Documents/lowest_ranked_movies_data.csv"
    )

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