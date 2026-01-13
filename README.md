# Opis projektu

Celem projektu jest zaprojektowanie i implementacja systemu symulującego dane pomiarowe w sieciach przesyłowych (np. wodociągowych lub gazowych) w celu analizy bilansu przepływów w czasie rzeczywistym. System ma na celu odtworzenie realistycznego zachowania sieci, w tym występowania losowych zakłóceń pomiarowych i anomalii, takich jak wycieki lub błędy liczników.

Projekt stanowi etap przygotowawczy do budowy pełnego systemu detekcji anomalii – w niniejszej wersji skupia się wyłącznie na generowaniu i wizualizacji danych symulacyjnych uruchamianych z poziomu CLI.

System w obecnej wersji umożliwia:
*   tworzenie wirtualnej topologii sieci przesyłowej w postaci grafu (drzewo/siatka/losowa),
*   generowanie syntetycznych szeregów czasowych przepływów wraz z przepływami na krawędziach,
*   dodawanie szumu pomiarowego oraz symulowanie anomalii (wycieki stałe/narastające, błędy liczników),
*   zapisywanie danych pomiarowych oraz statyczną wizualizację wyników.

Projekt wykorzystuje język Python oraz zestaw narzędzi do analizy danych i symulacji.

# Zakres projektu

## Zakres funkcjonalny
*   **Generowanie topologii sieci**
    Umożliwienie tworzenia wirtualnej sieci przesyłowej w postaci grafu, w którym węzły reprezentują punkty poboru, a krawędzie – połączenia przesyłowe. System pozwala na definiowanie liczby węzłów, źródeł zasilania oraz długości połączeń.

*   **Konfiguracja parametrów symulacji**
    Użytkownik może określić parametry symulacji, takie jak liczba węzłów, częstotliwość próbkowania, długość okresu symulacji, poziom szumu oraz częstość występowania anomalii.

*   **Generowanie danych pomiarowych**
    Dla każdego węzła generowane są syntetyczne szeregi czasowe przepływów, które podlegają losowym wahaniom i cyklicznym wzorcom zużycia. Dane są zniekształcane przez dodanie szumu z rozkładu normalnego.

*   **Symulacja anomalii**
    System losowo wprowadza dwa typy anomalii:
    *   wyciek w sieci – utrata części przepływu na losowym odcinku, o charakterze stałym lub narastającym,
    *   błąd licznika – stały offset lub dryf w pomiarach wybranego punktu.
    Anomalie występują w losowych momentach trwania symulacji.

*   **Zapisywanie danych**
    Wszystkie dane pomiarowe wraz z metadanymi (czas, identyfikator punktu, typ anomalii) są zapisywane jako csv.

*   **Wizualizacja wyników**
    Po zakończeniu symulacji generowane są statyczne wykresy (matplotlib) zapisywane jako pliki PNG: serie czasowe z zaznaczonymi anomaliami, statystyki przepływów, rozkład anomalii oraz graf siły przedstawiający topologię.

*   **Eksport danych**
    System umożliwia eksport wyników symulacji do formatu CSV lub JSON w celu dalszej analizy zewnętrznej.

*   **Raport symulacji**
    Po zakończeniu symulacji generowany jest raport zawierający podstawowe statystyki, takie jak średni przepływ, odchylenie standardowe, liczba i rodzaj wprowadzonych anomalii.

## Ograniczenia projektu
*   Wersja nie zawiera serwera API ani interaktywnego dashboardu – interakcja odbywa się wyłącznie przez CLI i pliki wyjściowe.
*   Projekt nie obejmuje implementacji algorytmów detekcji anomalii.
*   Dane są generowane w sposób syntetyczny i nie pochodzą z rzeczywistych urządzeń pomiarowych.
*   System nie realizuje jeszcze logiki wnioskowania (np. klasyfikacji typu awarii).
*   Aplikacja ma charakter prototypowy i działa w środowisku lokalnym.

## Wymagania techniczne
*   Python 3.8+
*   Biblioteki: numpy, pandas, networkx, matplotlib, scipy

# Implementacja

Projekt został przebudowany do architektury produkcyjnej z modułową strukturą.

## Struktura projektu

```
project/
├── __init__.py                     # Główny punkt wejścia modułu
├── config.py                       # Konfiguracja symulacji (SimulationConfig)
├── simulator.py                    # Orkiestracja symulacji (FlowSimulator)
│
├── anomaly_simulator.py           # Wrapper dla anomalii
├── data_generator.py              # Wrapper dla generatora danych
├── network_topology.py            # Wrapper dla topologii
├── visualizer.py                  # Wrapper dla wizualizacji
│
├── anomalies/                     # Moduł anomalii
│   ├── __init__.py
│   └── anomaly_injector.py        # Implementacja AnomalySimulator
│
├── simulation/                    # Moduł symulacji przepływów
│   ├── __init__.py
│   └── flow_simulator.py          # Implementacja FlowDataGenerator
│
├── topology/                      # Moduł topologii sieci
│   ├── __init__.py
│   └── graph_generator.py         # Implementacja NetworkTopology
│
├── visualization/                 # Moduł wizualizacji
│   ├── __init__.py
│   └── plotter.py                 # Implementacja FlowVisualizer
│
└── utils/                         # Narzędzia pomocnicze
    ├── __init__.py
    └── data_saver.py              # Zapis danych do CSV/JSON
```

## Główne komponenty

*   **config.py** – dataclass z parametrami symulacji
*   **simulator.py** – orkiestracja batchowa (setup → run → save → visualize → report)
*   **topology/** – konfigurowalne topologie z wieloma źródłami i długościami krawędzi
*   **simulation/** – silnik przepływów per węzeł i krawędź
*   **anomalies/** – wycieki stałe/narastające oraz błędy liczników z propagacją po grafie
*   **visualization/** – generowanie wykresów statycznych (matplotlib) oraz wizualizacji grafu sieci
*   **utils/** – zapis danych do plików CSV/JSON

## Uruchomienie

1. Instalacja zależności:
```bash
pip install -r requirements.txt
```

2. Uruchomienie podstawowej symulacji:
```bash
python main.py
```

3. Podgląd wszystkich opcji CLI:
```bash
python main.py --help
```

4. Uruchomienie przykładów z gotowymi konfiguracjami:
```bash
python examples.py [numer_przykładu]
```

## Tryb działania

Repozytorium udostępnia wyłącznie tryb wsadowy sterowany parametrami CLI. Typowy przebieg obejmuje:
1. Budowę topologii grafu (`FlowSimulator.setup`), w tym rozmieszczenie źródeł, węzłów pośrednich i odbiorców.
2. Generowanie przebiegów czasowych na węzłach i krawędziach wraz z szumem pomiarowym.
3. Opcjonalne wstrzyknięcie anomalii (wycieki, błędy liczników) zgodnie z parametrami.
4. Zapis danych do `output/` w formacie CSV lub JSON oraz wygenerowanie wykresów PNG i raportu zbiorczego.

Szczegółowy opis parametrów i plików wyjściowych znajduje się w `USAGE.md`.

# Wymagania niefunkcjonalne
*   **Wydajność** – System powinien generować dane dla minimum 100 punktów pomiarowych z częstotliwością 1 Hz w rozsądnym czasie pojedynczego uruchomienia CLI.
*   **Skalowalność** – Architektura systemu powinna umożliwiać łatwe rozszerzenie liczby węzłów i zwiększenie złożoności topologii sieci.
*   **Spójność danych** – Dane zapisywane muszą zachowywać bilans przepływów zgodny z topologią sieci (poza przypadkami anomalii).
*   **Bezpieczeństwo** – Dane zapisywane są lokalnie w plikach CSV/JSON, dlatego ważne jest kontrolowanie uprawnień do katalogu `output/`.
*   **Czytelność kodu** – Kod projektu powinien być modularny i opatrzony komentarzami zgodnymi.
*   **Przenośność** – System powinien działać w sposób identyczny w różnych środowiskach systemowych.
*   **Niezależność sieciowa** – Symulacja musi być możliwa do uruchomienia w trybie offline.
*   **Możliwość wizualizacji** – Po każdym przebiegu generowane są statyczne wykresy PNG (brak interaktywnego UI).
