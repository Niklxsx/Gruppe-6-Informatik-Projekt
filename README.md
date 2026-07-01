# Gruppe-6-Informatik-Projekt
IMDb Movie Rating & Blockbuster Predictor

In diesem Projekt entwickeln wir KI-Modelle, die selbstständig Vorhersagen über den Erfolg und die genaue Bewertung von Filmen treffen. Das Modell lernt aus historischen Filmdaten, versteckte Muster in objektiven Eigenschaften (wie Genre, Laufzeit, Erscheinungsjahr) zu erkennen.

Um das Potenzial eines Films vollständig zu analysieren, kombinieren wir zwei Methoden des maschinellen Lernens:
1. Klassifikation (Blockbuster vs. Normaler Film) - Kann das Modell vorhersagen, ob ein Film in die Liste der "Top 100 Filme" aufsteigt? (Binäres Label: 1 für Hit, 0 für normal).
2. Regression (Exakte Rating-Vorhersage): Wie präzise kann das Modell die IMDb-Bewertung (auf einer Skala von 1.0 bis 10.0) einschätzen?

Die Datengrundlage bilden historische Filmdaten der Plattform IMDb, die primär durch Web Scraping erhoben und über Kaggle zur Verfügung gestellt wurden. Wir nutzen drei Basis-Datensätze, die wir zu einer Datei gemerged haben:
1. 10.000 "normale" Filme (Movies_dataset.csv)
2. Die Top-100 besten bewerteten Filme (imdb_top_movies.csv)
3. Die Flop-100 am schlechtesten bewerteten Filme (lowest_ranked_movies_data.csv)

Folgende Biases sind hierbei relevant:
1. Durch das Web Scraping fehlen bei weniger populären Filmen häufiger Informationen.
2. Nutzerbewertungen häufig nicht objektiv, sondern unterliegen Hypes oder Review-Bombing aus verschiedenen Motiven.

Der ursprüngliche Zustand der Daten bestand aus drei unverbundenen, textlastigen Tabellen. Unsere Preprocessing-Pipeline führt folgende Schritte durch:
1. Vereinheitlichung der Spaltennamen und vertikales Merging (pd.concat).
2. Umwandlung textueller Laufzeiten (z. B. "1h 27m") in numerische Minuten via Regex.
3. One-Hot-Encoding für das kategoriale Feature Genre.
4. Entfernung von Filmen mit unter 100 Bewertungen, um statistische Ausreißer zu filtern.
5. Entfernung irrelevanter Spalten (IDs, URLs, Titel), um Overfitting zu vermeiden.

Für den Test-Training-Split wird der Datensatz im Verhältnis 80:20 aufgeteilt. Für die Klassifikation nutzen wir das stratify-Verfahren, um die Klassenverteilung stabil zu halten. Die Normalisierung erfolgt mittels Z-Transformation (StandardScaler). Um Data Leakage zu verhindern, wird der Scaler ausschließlich auf den Trainingsdaten berechnet (fit_transform) und nur transformativ auf die Testdaten angewendet (transform).

Bevor die Modelle trainiert wurden, ergab die explorative Analyse folgende Ergebnisse:
1. Das Attribut is_top_100 ist extrem ungleich verteilt. Auch bei den Genres dominieren Drama, Comedy und Action.
2. Es gibt eine positive Korrelation zwischen der Anzahl der abgegebenen Stimmen und der eigentlichen Bewertung. Zwischen dem Erscheinungsjahr und der Filmlänge besteht hingegen praktisch keine Korrelation.

Trainierte Modelle:
1. Klassifikation:
    - Logistische Regression: Ein lineares Modell mit geringer Komplexität. Es zieht eine gerade Trennlinie zwischen die Klassen und ist stark interpretierbar.
    - Random Forest Classifier: Ein hochkomplexes, nicht-parametrisches Ensemble-Modell ("Bagging"). Es baut Hunderte spezialisierte Entscheidungsbäume auf Teilmengen der Daten und lässt diese abstimmen. Es macht keine Annahmen über lineare Zusammenhänge und kann komplexe Muster erfassen (Gefahr von Overfitting bei falscher Konfiguration).
2. Regression