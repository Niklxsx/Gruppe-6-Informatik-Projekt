import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# Laden der bereinigten csv Datei in ein DataFrame
df = pd.read_csv('movies_merged.csv')  

# Überblick über die Daten
print('Erste 5 Zeilen: ', df.head())
print('Dateninformationen: ', df.info())
print('Statistische Zusammenfassung: ', df.describe())

# Fehlende Werte prüfen
print("Fehlende Werte:", df.isnull().sum())



#Trainings und Testdaten aufteilen, Zielvariable ist "is_top_100"
X = df[["release_year", "vote_count", "duration_min"]]
y = df["is_top_100"]

# Zielvariable is IMDb-Bewertung
# y_rating = df["imdb_rating"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
# X_train_rating, X_test_rating, y_train_rating, y_test_rating = train_test_split(X, y_rating, test_size=0.2, random_state=42)
# kein stratify, da das Rating kontinuierlich

# Trainingsdaten normalisieren
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)



# Verteilung der Zielvariablen
plt.figure(figsize=(8, 6))
sns.countplot(x='is_top_100', data=df)        # untersucht Class Imbalance (binäre Variable)
plt.title('Verteilung der Zielvariablen')
plt.xlabel('Ist in Top 100')
plt.ylabel('Anzahl')
plt.show()


# Class Imbalance
print(df["is_top_100"].value_counts())
print('Class Imbalance, is_top_100')
print(df['is_top_100'].value_counts(normalize=True) * 100)

genres = (df["genre"].str.split("|").explode())
print('Class Imbalance, genre')
print(genres.value_counts())


# Verteilung der IMDb-Bewertungen
plt.figure(figsize=(8, 6))
sns.histplot(df['imdb_rating'], bins=20, kde=True)
plt.title('Verteilung der IMDb-Bewertungen')
plt.xlabel('IMDb-Bewertung')
plt.ylabel('Anzahl Filme')
plt.show()

# Verteilung der Filmdauer
plt.figure(figsize=(8,5))
sns.histplot(df["duration_min"], bins=30)
plt.title("Verteilung der Filmdauer")
plt.xlabel("Dauer in Minuten")
plt.ylabel("Anzahl Filme")
plt.show()


# Korrelation zwischen Attributen
corr = df[["imdb_rating", "vote_count", "release_year", "duration_min", "is_top_100"]].corr()    # bestimmte Attribute gewählt
print('Korrelation: ', corr)

plt.figure(figsize=(8, 6))
sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Korrelationsmatrix')
plt.show()



# weitere Plots

# Erscheinungsjahre
plt.figure(figsize=(8, 6))
sns.histplot(df['release_year'], bins=20, kde=True)
plt.title('Verteilung der Erscheinungsjahre')
plt.xlabel('Erscheinungsjahr')
plt.ylabel('Anzahl Filme')
plt.show()

# Filmdauer
plt.figure(figsize=(8, 6))
sns.boxplot(x=df['duration_min'])
plt.title('Verteilung der Filmdauer')
plt.xlabel('Filmdauer (Minuten)')
plt.show()

# Stimmzahl vs IMDb-Bewertung
plt.figure(figsize=(8, 6))
sns.scatterplot(x='vote_count', y='imdb_rating', data=df)
plt.title('Stimmzahl vs IMDb-Bewertung')
plt.xlabel('Anzahl Stimmen')
plt.ylabel('IMDb-Bewertung')
plt.show()

# Genre-Verteilung
plt.figure(figsize=(12, 8))
sns.countplot(y=genres, data=df, order=genres.value_counts().index)  # sortiert nach Häufigkeit der Genres
plt.title('Verteilung der Genres')
plt.xlabel('Anzahl Filme')
plt.ylabel('Genre')
plt.show()
