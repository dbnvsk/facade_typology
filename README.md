# Typologie Fasad — Klasteryzacja Metodą Warda z Walidacją Bootstrap

Projekt wykonany w ramach zajęć z **Machine Learning** i dotyczy
zautomatyzowanego wydzielania **typologii fasad architektonicznych**
na podstawie pomiarów geometrycznych, proporcji, cech rytmicznych,
stopnia perforacji i ornamentyki.

Celem projektu jest stworzenie **obiektywnej, mierzalnej klasyfikacji fasad**,
która może wspierać analizy urbanistyczne, inwentaryzacje, badania historyczne
oraz projektowanie oparte na danych.

---

## Kontekst projektu

Fasady zostały zmierzone i opisane zestawem cech opisujących m.in.:

- proporcje (wysokość, szerokość, proporcja),
- rytm i symetrię (częstotliwość, warianty rytmu),
- perforację (okna, drzwi, długości otworów),
- warstwowość fasad (wykusze, ryzality)
- cechy geometryczne powierzchni fasady.

Łącznie analizowano dziesiątki cech numerycznych i binarnych.

---

## Cele analizy

1. **Wybrać istotne cechy fasad** na podstawie wariancji i korelacji.  
2. **Przeprowadzić klasteryzację** dwiema metodami:
   - K-Means (porównawczo),
   - **Ward’s Hierarchical Clustering (metoda główna)**.
3. **Wyznaczyć optymalną liczbę klastrów** na podstawie dendrogramu.
4. **Oszacować stabilność klastrów** przy pomocy *bootstrap resampling*.
5. **Zinterpretować typologie fasad** na podstawie profili cech.
6. **Zmapować wyniki** (folium) w przestrzeni miasta (Gdańsk).

Rezultatem jest **7-klastrowa typologia fasad** o zróżnicowanej stabilności.

---

## Wykorzystane metody i pipeline

### 1. Przygotowanie danych
- usunięcie kolumn opisowych (adres, typ, kontekst),
- imputacja braków (mediana),
- standaryzacja cech (`StandardScaler`),
- usunięcie cech o niskiej wariancji,
- usunięcie cech wzajemnie skorelowanych > 0.8,
- kodowanie kategorii do postaci numerycznej.

### 2. Klasteryzacja K-Means (k=4)
- szybkie porównanie działania metody,
- metoda *flat*, niehierarchiczna,
- wykres liczebności klastrów.

### 3. Klasteryzacja metodą Warda
- podejście hierarchiczne,
- analiza dendrogramu,
- próg odległości = **15**, liczba klastrów = **7**,
- klasteryzacja najlepiej oddająca różnice typologiczne.

### 4. Bootstrap stabilności klastrów (100 replikacji)
Dla każdej replikacji:
- losowanie próby ze zwracaniem,
- klasteryzacja Ward,
- obliczenie macierzy współwystępowania,
- wyznaczenie stabilności klastrów.

Zakres obserwowany:
- od **0.48** (klaster niestabilny),
- do **1.00** (klaster bardzo stabilny).

### 5. Profilowanie klastrów
Dla każdego klastra obliczono średnie wartości:
- proporcji,
- perforacji,
- ornamentyki,
- rytmu,
- geometrii.

To pozwoliło nadać **architektoniczne interpretacje** klastrom.

### 6. Mapa klastrów (folium)
Każda fasada została wyświetlona na mapie Gdańska w kolorze odpowiadającym klastrowi.

---

## Struktura repozytorium
typologie_fasad/<br>
│<br>
├── README.md<br>
├── requirements.txt<br>
│<br>
├── data/<br>
│ DANE.xlsx<br>
│ cechy.png<br>
│ proba badawcza.png<br>
│<br>
├── src/<br>
│ typologie_ward.py<br>
│<br>
└── output/<br>
mapa_klastry_ward.html<br>

## Uruchamianie projektu

### 1. Sklonuj repo:
git clone https://github.com/dbnvsk/typologie_fasad.git

### 2. Zainstaluj wymagania:
pip install -r requirements.txt

### 3. Uruchom analizę:
python src/typologie_ward.py


Wyniki pojawią się w folderze `output/`.

---

## Wymagania (requirements)

Wszystkie zależności znajdują się w `requirements.txt`.

Najważniejsze biblioteki:
- pandas  
- numpy  
- matplotlib  
- seaborn  
- scikit-learn  
- scipy  
- tqdm  
- folium  
- plotly  
- openpyxl  

---

## Wyniki analizy

- **7 klastrów fasad** o zróżnicowanych cechach i charakterze architektonicznym,  
- stabilność klastrów oceniona bootstrapem,  
- analizy proporcji, warstwowości, perforacji, rytmu i symetrii,  
- mapa fasad i ich typologii w przestrzeni miasta,  
- możliwość dalszej interpretacji urbanistycznej i historycznej.

---

## Możliwe zastosowania architektoniczne

- systematyzacja zasobów zabytkowych,  
- analiza spójności stylistycznej kwartałów,  
- wykrywanie typowych i nietypowych fasad,  
- wsparcie w decyzjach konserwatorskich,  
- tworzenie narzędzi do automatycznej kategoryzacji zabudowy.

