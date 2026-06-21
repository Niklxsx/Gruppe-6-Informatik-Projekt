import os
import re
import numpy as np
import pandas as pd


def load_and_prepare_datasets(basis_path, top_path, flop_path):
    """Datensätze laden, Spalten harmonisieren und Targets erstellen."""

    df_basis = pd.read_csv(basis_path)
    df_top = pd.read_csv(top_path)
    df_flop = pd.read_csv(flop_path)


    def clean_genres(genre):
        if pd.isna(genre):
            return np.nan

        genre = str(genre).strip()

        # ['Action', 'Drama', 'Musical']
        if genre.startswith("["):
            genre = genre.strip("[]")
            genres = [
                g.strip().strip("'\"")
                for g in genre.split(",")
            ]
            return "|".join(genres)

        # Animation, Drama, Fantasy
        elif "," in genre:
            genres = [g.strip() for g in genre.split(",")]
            return "|".join(genres)

        # bereits Action|Comedy|Music
        return genre

    # Basis-Datensatz (movies.csv)
    df_basis_prep = pd.DataFrame()
    df_basis_prep["film_title"] = df_basis["title_x"]
    df_basis_prep["imdb_rating"] = df_basis["imdb_rating"]
    df_basis_prep["vote_count"] = df_basis["imdb_votes"]
    df_basis_prep["release_year"] = df_basis["year_of_release"]
    df_basis_prep["genre"] = df_basis["genres"].apply(clean_genres)
    df_basis_prep["duration_min"] = df_basis["runtime"]
    df_basis_prep["is_top_100"] = 0 # Standard-Label für die Klassifikation

    # Top-100 (imdb_top_movies.csv)
    df_top_prep = pd.DataFrame()
    df_top_prep["film_title"] = df_top["Title"]
    df_top_prep["imdb_rating"] = df_top["Rating"]
    df_top_prep["vote_count"] = np.nan
    df_top_prep["release_year"] = df_top["Year"]
    df_top_prep["genre"] = df_top["Genre"].apply(clean_genres)
    df_top_prep["duration_min"] = np.nan
    df_top_prep["is_top_100"] = 1 # Wichtiges Ziel-Label für Klassifikation

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

    df_flop_prep["vote_count"] = df_flop["review_count"].apply(clean_review_count)
    df_flop_prep["release_year"] = df_flop["year"]
    df_flop_prep["genre"] = df_flop["genre"].apply(clean_genres)

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

    df_flop_prep["duration_min"] = df_flop["duration"].apply(convert_duration_to_minutes)
    df_flop_prep["is_top_100"] = 0 # Flops gehören nicht zu den Top-100

    return df_basis_prep, df_top_prep, df_flop_prep


def merge_and_save(df_basis, df_top, df_flop, output_file="movies_merged.csv"):
    """Führt die drei Datensätze zusammen und speichert sie als CSV."""

    # Vertikal zusammenfügen
    df_all = pd.concat([df_top, df_flop, df_basis], ignore_index=True)

    # Top-100-Einträge priorisieren
    df_all = df_all.sort_values(by="is_top_100", ascending=False)

    # Duplikate entfernen, vote_count und duration aus basis übernehemen, data aus top100 zuerst/priorisiert
    def first_valid(series):
        s = series.dropna()
        return s.iloc[0] if not s.empty else np.nan

    df_all = (
        df_all.groupby(["film_title", "release_year"], as_index=False)
              .agg({
                  "imdb_rating": first_valid,
                  "vote_count": first_valid,
                  "duration_min": first_valid,
                  "genre": first_valid,
                  "is_top_100": "max"
              })
    )

    # Fehlende Genres ersetzen
    df_all["genre"] = df_all["genre"].fillna("Unknown")

    # Fehlende Laufzeiten ersetzen
    df_all["duration_min"] = (df_all["duration_min"].replace(r"\N", np.nan))
    df_all["duration_min"] = pd.to_numeric(df_all["duration_min"],errors="coerce")
    median_duration = df_all["duration_min"].median()
    df_all["duration_min"] = df_all["duration_min"].fillna(median_duration)

    # Fehlende Bewertungen ersetzen
    df_all["vote_count"] = df_all["vote_count"].fillna(df_all["vote_count"].median())

    # Fehlende Ratings oder Jahreszahlen entfernen
    df_all = df_all.dropna(subset=["release_year", "imdb_rating"])

    print(df_all.isnull().sum())

    # Als CSV speichern
    df_all.to_csv(output_file, index=False, encoding="utf-8")

    print(f"Datei erfolgreich gespeichert: {output_file}")

    return df_all


if __name__ == "__main__":

    # Hier die Pfade evtl anpassen
    PATH_BASIS = "movies.csv"
    PATH_TOP = "imdb_top_movies.csv"
    PATH_FLOP = "lowest_ranked_movies_data.csv"

    # Prüfen, ob die Dateien am angegebenen Ort existieren
    files_exist = True
    for p in [PATH_BASIS, PATH_TOP, PATH_FLOP]:
        if not os.path.exists(p):
            print(f"FEHLER: Datei '{p}' nicht gefunden. Bitte prüfen Sie den Pfad!")
            files_exist = False

    if files_exist:
        # Datenaufbereitung
        df_b, df_t, df_f = load_and_prepare_datasets(PATH_BASIS, PATH_TOP, PATH_FLOP)
        df_merged = merge_and_save(df_b, df_t, df_f, "movies_merged.csv")
        
