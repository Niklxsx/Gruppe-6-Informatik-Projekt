import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Laden der bereinigten csv Datei in ein DataFrame
df = pd.read_csv('filme.csv')    # Dateiname anpassen, falls nötig

# Überblick über die Daten
print('Erste 5 Zeilen: ', df.head())
print('Dateninformationen: ', df.info())

# Verteilung der Zielvariablen
plt.figure(figsize=(8, 6))
sns.countplot(x='is_top_100', data=df)
plt.title('Verteilung der Zielvariablen')
plt.xlabel('Ist in Top 100')
plt.ylabel('Anzahl')
plt.show()

plt.figure(figsize=(8, 6))
sns.histplot(df['imdb_rating'], bins=20, kde=True)
plt.title('Verteilung der IMDb-Bewertungen')
plt.xlabel('IMDb-Bewertung')
plt.ylabel('Anzahl Filme')
plt.show()


# Korrelation zwischen Attributen
numeric_df = df.select_dtypes(include=[np.number])
corr = numeric_df.corr()
print('Korrelation: ', corr)

plt.figure(figsize=(8, 6))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Korrelationsmatrix')
plt.show()


# Class Imbalance?

class_counts = df['is_top_100'].value_counts()



# Plots
# IMDb-Bewertungen
plt.figure(figsize=(8, 6))
sns.histplot(df['imdb_rating'], bins=20, kde=True)
plt.title('Verteilung der IMDb-Bewertungen')
plt.xlabel('IMDb-Bewertung')
plt.ylabel('Anzahl Filme')
plt.show()

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
sns.countplot(y='genre', data=df, order=df['genre'].value_counts().index)  # ???
plt.title('Verteilung der Genres')
