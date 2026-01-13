# AI Graph Flow Simulator – pełna dokumentacja (PL)

## 1. Wprowadzenie i cel
System generuje syntetyczne dane pomiarowe dla sieci przesyłowych (np. woda, gaz) w formie grafu. Pozwala testować bilans przepływów, symulować zakłócenia i szybko wizualizować wyniki. W aktualnej wersji narzędzia dostępne są wyłącznie w trybie wsadowym (CLI) oraz w formie interfejsu programistycznego (pakiet `project`).

Najważniejsze możliwości:
- budowa topologii grafu (drzewo, siatka/mesh, radial, grid, losowa) z wieloma źródłami,
- generowanie szeregów czasowych z cyklami dobowymi/tygodniowymi i szumem,
- symulacja anomalii: wycieki (stałe i narastające) oraz błędy liczników (add/mul/drift),
- zapis wyników do CSV/JSON i tworzenie raportu statystyk,
- wizualizacje statyczne (PNG), w tym graf siłowy z wielkością węzłów/żył zależną od przepływów.

## 2. Architektura i przepływ danych
Przepływ w trakcie jednego uruchomienia:
```
CLI (main.py) / kod Python →
FlowSimulator (project/simulator.py) →
  NetworkTopology      (topology/graph_generator.py)
  FlowDataGenerator    (simulation/flow_simulator.py)
  AnomalySimulator     (anomalies/anomaly_injector.py)
  SimulationDataSaver  (utils/data_saver.py)
  FlowVisualizer       (visualization/plotter.py)
→ zapisy plików → wizualizacje → raport JSON → podsumowanie w konsoli
```

Warstwy:
- **Wejście**: argumenty CLI lub konfiguracja `SimulationConfig`.
- **Model domenowy**: graf sieci z typami węzłów (source/hub/consumer), krawędziami z długością.
- **Silnik danych**: generacja bazowych przepływów, propagacja w grafie, wtrysk anomalii.
- **Wyjście**: pliki CSV/JSON oraz wizualizacje PNG.

## 3. Wymagania i instalacja
- Python 3.8+.
- Zależności: `numpy`, `pandas`, `networkx`, `matplotlib`, `scipy` (szczegóły w `requirements.txt`).

Instalacja:
```bash
pip install -r requirements.txt
```

## 4. Szybki start (CLI)
```bash
python main.py                 # Domyślna symulacja (20 węzłów, 24h, 0.1 Hz)
python main.py --help          # Lista opcji
python main.py --nodes 50 --topology mesh --duration 48 --anomaly-prob 0.15
python examples.py 1           # Gotowy scenariusz (pełna lista w examples.py)
```
Standardowy przebieg: `setup → run → save_data → visualize → print_summary`. Wyniki domyślnie trafiają do `output/` (można zmienić `--output`).

## 5. Konfiguracja – parametry (CLI i SimulationConfig)
### Topologia
- `--nodes` (`num_nodes`, domyślnie 20) – liczba węzłów odbiorczych.
- `--sources` (`num_sources`, domyślnie 1) – liczba źródeł.
- `--topology {tree,mesh,random,radial,grid}` (`topology_type`, domyślnie `tree`).
- `edge_length_range` – długość krawędzi (m) losowana z przedziału `(5.0, 25.0)`.

### Czas
- `--duration` (`duration_hours`, domyślnie 24) – czas trwania w godzinach.
- `--sampling` (`sampling_frequency_hz`, domyślnie 0.1) – częstotliwość próbkowania w Hz.
- `--start-time` (`start_time`) – początek w ISO; brak → północ dnia uruchomienia.

### Przepływy
- `--base-flow` (`base_flow_rate`, domyślnie 100.0 m³/h) – średni przepływ.
- `--flow-variation` (`flow_variation`, domyślnie 0.2) – zakres wahań (ułamek bazowego).
- `--noise-std` (`noise_std`, domyślnie 2.0) – odchylenie standardowe szumu.

### Anomalie
- `--anomaly-prob` (`anomaly_probability`, domyślnie 0.1) – prawdopodobieństwo wystąpienia w godzinie.
- `--anomaly-rate-multiplier` (`anomaly_rate_multiplier`, domyślnie 1.0) – mnożnik liczby zdarzeń.
- `--anomaly-severity` (`anomaly_severity`, domyślnie 1.0) – skala wielkości anomalii.
- `--progressive-leak-prob` (`progressive_leak_probability`, domyślnie 0.25) – udział wycieków narastających.
- `--disable-leaks` / `--disable-meter-errors` – wyłączenie danego typu anomalii.
- Zakresy wielkości: `leak_magnitude_range=(5,15) m³/h`, `meter_error_range=(-5,5) m³/h` (skalowane przez severity).

### Wyjście
- `--output` (`output_dir`, domyślnie `output`) – folder wynikowy (tworzony automatycznie).
- `--export {csv,json}` (`export_format`, domyślnie `csv`).
- `--no-visualize` – pominięcie generowania wykresów (przyspiesza duże przebiegi).

## 6. Topologie – jak są budowane
- **tree**: źródła → huby (≈ `num_nodes/5`) → konsumenci `c001..`. Każdy hub podpięty do źródła; konsumenci rozdzieleni równomiernie.
- **mesh**: baza jak `tree` + dodatkowe krawędzie między hubami oraz losowe drugie połączenie konsumentów zwiększające redundancję.
- **random**: konsumenci losowo oznaczeni jako huby; każdy węzeł ma co najmniej jednego rodzica (źródło lub hub). Dodawane są też losowe krawędzie między konsumentami.
- **radial**: pierścień hubów zasilany z wielu źródeł; konsumenci rozdzieleni po hubach; huby łączone w pierścień dla redundancji.
- **grid**: siatka kwadratowa węzłów konsumenckich `gXX_YY`, połączenia prawo/dół, źródła podpięte do górnego rzędu. Nadaje się do imitacji sieci miejskiej.

Wszystkie krawędzie otrzymują unikalny `edge_id` (`e_{source}_{target}`) i losową długość z zakresu konfiguracji.

## 7. Generowanie danych przepływu (FlowDataGenerator)
- Dla każdego węzła konsumenckiego tworzony jest bazowy przebieg: suma dwóch sinusów (dobowy + tygodniowy) z losowym współczynnikiem per węzeł.
- Dodawany jest szum gaussowski `N(0, noise_std)`.
- Przepływy w węzłach wewnętrznych/hubach powstają przez sumowanie przepływów dzieci w kolejności topologicznej (zabezpieczenie na grafy z cyklami).
- Przepływy na krawędziach są kopiowane z węzła docelowego (w obecnym modelu przepływ na krawędzi = przepływ konsumpcyjny węzła docelowego).
- Oś czasu: `DatetimeIndex` z krokiem `max(1s, 1/sampling_frequency_hz)`; łączna liczba próbek = `duration_hours * 3600 * sampling_frequency_hz`.
- Tryb strumieniowy: `stream_time_series` zwraca generator porcji (timestamp + snapshot węzłów i krawędzi); może symulować czas rzeczywisty przez `real_time=True`.

## 8. Anomalie (AnomalySimulator)
### Generowanie
- Liczba anomalii ≈ `duration_hours * 60 * anomaly_probability * anomaly_rate_multiplier / 60`.
- Typy aktywne zależą od flag `enable_leaks` / `enable_meter_errors`.
- Każda anomalia ma: `id`, `type`, `target_type` (node/edge), `target_id`, `start_time`, `duration_minutes`, `magnitude`, `mode`.

### Typy
- **Leak (wyciek)**: cel = krawędź; mode `const` (stały ubytek) lub `progressive` (liniowy wzrost). Skala wielkości = `leak_magnitude_range * anomaly_severity`. Redukcja propaguje do węzła docelowego i dalej po potomkach.
- **Meter error (błąd licznika)**: cel = węzeł; mode:
  - `add` – offset addytywny,
  - `mul` – mnożnik (1 ± losowy odchył),
  - `drift` – liniowy dryf w czasie trwania anomalii.
  Wielkość skalowana przez `meter_error_range * anomaly_severity`.

### Aplikacja
- Dla każdego rekordu w oknie czasowym ustawiane są `anomaly_type` oraz `anomaly_active=True`.
- Wyciek: wartości przepływu są obcinane do zera (clip), a następnie propagowane w dół grafu.
- Błąd licznika: modyfikacja wartości pomiaru w węźle (add/mul/drift).

## 9. Zapisywanie danych (SimulationDataSaver)
Pliki powstają w katalogu `output_dir`:
- `flow_measurements.csv|json` – wszystkie próbki węzłowe (timestamp, node_id, flow, node_type, anomaly_type, anomaly_active).
- `edge_flows.csv|json` – próbki na krawędziach (timestamp, edge_id, source, target, length, flow, anomaly_*).
- `anomalies.csv` – tabela wygenerowanych anomalii (id, type, target_id, mode, magnitude, start_time, duration_minutes, target_type).
- `topology_info.json` – metadane grafu, lista węzłów (id, type) i krawędzi (id, source, target, length).
- `simulation_report.json` – statystyki symulacji (patrz sekcja 11).

Format wybierany przez `export_format` (CSV/JSON) dla danych pomiarowych i krawędzi.

## 10. Wizualizacje (FlowVisualizer)
- `flow_plot.png` – wykresy szeregów czasowych dla próby węzłów konsumenckich (domyślnie pierwsze 5). Anomalie oznaczone czerwonym scatterem.
- `flow_statistics.png` – cztery wykresy słupkowe (mean, std, min, max) na węzeł.
- `anomaly_distribution.png` – histogram wielkości + rozkład typów (tworzony tylko gdy są anomalie).
- `force_directed_graph.png` – graf siłowy (Kamada-Kawai, fallback spring) z:
  - kolory: źródło (czerwony), hub (niebieski), konsument (zielony),
  - rozmiar węzła ~ całkowity przepływ,
  - grubość krawędzi ~ średni przepływ,
  - etykiety krawędzi: łączny i średni przepływ,
  - legenda typów węzłów.

Flaga `--no-visualize` pomija generowanie obrazów (przydatne dla długich przebiegów).

## 11. Raport i podsumowanie
- `generate_report()` tworzy `simulation_report.json` z sekcjami:
  - `simulation_info`: start/end, duration_hours, sampling_frequency_hz, total_samples,
  - `topology_info`: liczby węzłów/źródeł/krawędzi/hubów/konsumentów, typ topologii,
  - `flow_statistics` i `edge_statistics`: mean/std/min/max, liczba pomiarów,
  - `anomaly_statistics`: liczba anomalii, liczba wycieków/błędów, procent próbek z flagą anomalii.
- `print_summary()` wypisuje skrót raportu w konsoli po zakończeniu symulacji.

## 12. Programistyczne użycie pakietu `project`
```python
from project import FlowSimulator, SimulationConfig

config = SimulationConfig(
    num_nodes=30,
    topology_type="mesh",
    duration_hours=24,
    sampling_frequency_hz=0.2,
    anomaly_probability=0.15,
    output_dir="my_output",
    export_format="json",
)

sim = FlowSimulator(config)
sim.setup()
sim.run()

# Dostęp do danych
node_df = sim.get_node_dataframe()   # lub sim.time_series (dict node_id -> DataFrame)
edge_df = sim.get_edge_dataframe()
anomalies = sim.anomalies            # lista dictów
topology = sim.topology.get_topology_info()

sim.save_data()
sim.visualize()
sim.print_summary()
```
W trybie strumieniowym można użyć `FlowDataGenerator.stream_time_series()` jako generatora snapshotów.

## 13. Scenariusze przykładowe (examples.py)
- `python examples.py 1` – podstawowa symulacja (10 węzłów, 12 h, 0.1 Hz).
- `python examples.py 2` – wysoka częstotliwość (1 Hz, krótkie trwanie).
- `python examples.py 3` – duża sieć (50 węzłów, 24 h, więcej anomalii).
- `python examples.py 4` – wysoka częstość anomalii.
- `python examples.py 5` – demonstracja dostępu programistycznego (drukuje statystyki w konsoli).
- `python examples.py 6` – eksport JSON zamiast CSV.

## 14. Wydajność i tuningi
- Rozmiar danych: `num_nodes * duration_hours * 3600 * sampling_frequency_hz`.  
  100 węzłów @ 1 Hz @ 24 h ≈ 8.6 mln rekordów.
- Aby skrócić czas/pamięć:
  - zmniejsz `sampling_frequency_hz` (np. 0.05–0.2 dla długich symulacji),
  - skróć `duration_hours` lub liczbę węzłów,
  - wyłącz wizualizacje (`--no-visualize`) przy dużych zbiorach,
  - zapis do CSV jest szybszy niż JSON.
- Wizualizacje ograniczają się do próbki węzłów, by nie przeciążać wykresów.

## 15. Rozszerzanie systemu
- **Nowy typ topologii**: dodaj metodę w `topology/graph_generator.py`, zarejestruj w słowniku `builders` w `_create_topology`.
- **Nowy typ anomalii**: dodaj wariant w `generate_anomalies` i obsługę w `apply_anomalies` w `anomalies/anomaly_injector.py`; rozważ dopisanie oznaczeń kolumn.
- **Dodatkowa wizualizacja**: dopisz funkcję w `visualization/plotter.py` i wywołaj ją w `FlowSimulator.visualize`.
- **Nowe pola danych**: zaktualizuj `SimulationDataSaver` tak, by uwzględnić zapis w CSV/JSON.

## 16. Struktura repozytorium
```
project/
├── simulator.py           # Orkiestracja przebiegu
├── config.py              # Dataclass konfiguracji
├── network_topology.py    # Wrapper do graph_generator
├── anomaly_simulator.py   # Wrapper do anomaly_injector
├── data_generator.py      # Wrapper do flow_simulator
├── visualizer.py          # Wrapper do plotter
├── simulation/flow_simulator.py   # Generator szeregów
├── anomalies/anomaly_injector.py  # Anomalie
├── topology/graph_generator.py    # Topologie
├── visualization/plotter.py       # Wykresy
└── utils/data_saver.py            # Zapis plików
main.py, examples.py               # Wejścia CLI/przykłady
README.md, USAGE.md                # Opis i skrócona instrukcja
```

## 17. Rozwiązywanie problemów (quick-checklista)
- **Brak zależności**: uruchom `pip install -r requirements.txt`.
- **Brak plików wyjściowych**: sprawdź uprawnienia/katalog `--output`; upewnij się, że `save_data()` zostało wywołane.
- **Puste wykresy anomalii**: jeżeli flaga `--disable-leaks` i `--disable-meter-errors` są aktywne, nie powstają anomalie.
- **Błędny format czasu**: `--start-time` musi być w ISO (np. `2025-01-01T00:00:00`).
- **Za duże zużycie pamięci**: obniż `sampling_frequency_hz`, skróć czas, użyj `--no-visualize`.

## 18. Dobre praktyki pracy z danymi
- Katalog `output/` można sprzątać między uruchomieniami, aby uniknąć mieszania wyników.
- Przy analizie zewnętrznej: kolumny `anomaly_type` i `anomaly_active` umożliwiają filtrowanie normalnych/anomalnych próbek.
- Topologię oraz słownik krawędzi (`edge_id`, `source`, `target`) można łączyć z danymi pomiarowymi po `edge_id`/`node_id`.

## 19. Notatki o jakości i testowaniu
- Funkcje są opatrzone docstringami; typowanie (typing) obejmuje kluczowe klasy.
- Podstawowa walidacja wejścia odbywa się w `SimulationConfig.__post_init__` (normalizacja źródeł, liczby próbek, dodatnie severity).
- Brak zautomatyzowanych testów w repo; do szybkiej walidacji użyj `examples.py` lub jednolinijkowego testu:
  ```bash
  python - <<'PY'
  from project import FlowSimulator, SimulationConfig
  sim = FlowSimulator(SimulationConfig(num_nodes=5, duration_hours=1, sampling_frequency_hz=0.2))
  sim.setup(); sim.run(); sim.save_data()
  print("✓ symulacja OK")
  PY
  ```

## 20. API referencyjne (moduły i klasy)
- `project/simulator.py` – **FlowSimulator**
  - `setup()` – buduje topologię, inicjalizuje generator, symulator anomalii i wizualizator.
  - `run()` – generuje szeregi, tworzy listę anomalii, aplikuje je do danych.
  - `save_data()` – zapisuje dane węzłów/krawędzi, anomalii i topologii.
  - `visualize()` – zapisuje wykresy PNG.
  - `generate_report()` – zwraca słownik z podsumowaniem i zapisuje `simulation_report.json`.
  - `print_summary()` – wypisuje raport w konsoli.
  - `get_node_dataframe()`, `get_edge_dataframe()` – scalamy dict DataFrame’ów w jeden DataFrame.

- `project/config.py` – **SimulationConfig** (dataclass)
  - Topologia: `num_nodes`, `num_sources`, `topology_type`, `edge_length_range`, `source_nodes`.
  - Czas: `start_time`, `duration_hours`, `sampling_frequency_hz`.
  - Przepływy: `base_flow_rate`, `flow_variation`, `noise_std`.
  - Anomalie: `anomaly_probability`, `progressive_leak_probability`, `leak_magnitude_range`, `meter_error_range`, `anomaly_rate_multiplier`, `anomaly_severity`, `enable_leaks`, `enable_meter_errors`.
  - Wyjście: `output_dir`, `export_format`.
  - Własność: `total_samples`, `time_step_seconds`, `end_time`.
  - Uwaga: jeżeli potrzebujesz deterministycznych wyników, możesz ręcznie dodać atrybut `seed` (np. `config.seed = 42`), który zostanie wykorzystany przez generator danych.

- `project/topology/graph_generator.py` – **NetworkTopology**
  - `get_nodes()`, `get_edges()`, `get_consumers()`, `get_node_type(node)`, `get_edge_id(src, tgt)`, `get_topology_info()`.
  - Metody `_build_*` tworzą konkretne układy grafu (tree/mesh/random/radial/grid).
  - Graf to `networkx.DiGraph` z atrybutami węzłów: `node_type`, `demand`; oraz krawędzi: `edge_id`, `length`.

- `project/simulation/flow_simulator.py` – **FlowDataGenerator**
  - `generate_time_series(topology=None, node_ids=None)` – zwraca dict węzłów i krawędzi → DataFrame.
  - `stream_time_series(topology=None, real_time=None)` – generator snapshotów (timestamp + listy słowników).
  - Pomocnicze: `generate_base_flow()`, `add_noise()`, `_propagate_internal_nodes()`, `_build_edge_series()`.

- `project/anomalies/anomaly_injector.py` – **AnomalySimulator**
  - `generate_anomalies(node_ids, edge_catalog)` – buduje listę zdarzeń anomalii.
  - `apply_anomalies(time_series, edge_flows)` – modyfikuje DataFrame’y, ustawia flagi.
  - `get_anomaly_report()` – zwraca DataFrame z anomaliami.
  - Mechanizmy: propagacja wycieku w dół grafu, profile `const`/`progressive`, tryby liczników `add/mul/drift`.

- `project/utils/data_saver.py` – **SimulationDataSaver**
  - `save_node_data()`, `save_edge_data()`, `save_anomalies()`, `save_topology()`.
  - `_resolve_path()` wybiera rozszerzenie na bazie `export_format`.

- `project/visualization/plotter.py` – **FlowVisualizer**
  - `plot_node_flows()`, `plot_anomaly_distribution()`, `plot_flow_statistics()`, `plot_force_directed_graph()`.
  - Wykresy zapisują pliki PNG w `output_dir`.

- Wrappery ułatwiają import: `project/network_topology.py`, `project/data_generator.py`, `project/anomaly_simulator.py`, `project/visualizer.py`.

## 21. Struktura danych w pamięci
- `FlowSimulator.time_series` – dict `node_id -> DataFrame` z kolumnami: `timestamp`, `node_id`, `flow`, `node_type`, `anomaly_type`, `anomaly_active`.
- `FlowSimulator.edge_series` – dict `edge_id -> DataFrame` z kolumnami: `timestamp`, `edge_id`, `source`, `target`, `length`, `flow`, `anomaly_type`, `anomaly_active`.
- `FlowSimulator.anomalies` – lista słowników (id, type, target_id, mode, magnitude, start_time, duration_minutes, target_type).
- `NetworkTopology.graph` – `networkx.DiGraph` z atrybutami węzłów/krawędzi, używany do propagacji i wizualizacji.
- `SimulationDataSaver` łączy DataFrame’y w pliki; `FlowVisualizer` przyjmuje scalone DataFrame’y (z `get_node_dataframe` / `get_edge_dataframe`).

## 22. Szczegółowy lifecycle jednego przebiegu
1) **Konfiguracja**: CLI → `SimulationConfig` (walidacja w `__post_init__`: uzupełnia `source_nodes`, oblicza `total_samples`).
2) **setup()**:
   - `NetworkTopology` buduje graf (węzły + krawędzie z długością).
   - `FlowDataGenerator` i `AnomalySimulator` dostają konfigurację; symulator anomalii otrzymuje referencję do grafu.
   - `FlowVisualizer` przygotowuje katalog wyjściowy.
3) **run()**:
   - `generate_time_series()` tworzy szeregi dla konsumentów, następnie propaguje w górę do hubów/źródeł, buduje dane krawędzi.
   - `generate_anomalies()` tworzy listę zdarzeń na podstawie prawdopodobieństwa, czasu trwania i wybranych celów (węzły/krawędzie).
   - `apply_anomalies()` modyfikuje przepływy, ustawia flagi i propaguje wycieki w dół grafu.
4) **save_data()**: zapisuje pomiary, anomalie, topologię i zwraca ścieżki do plików.
5) **visualize()**: tworzy wykresy; jeśli brak anomalii, pomija ich rozkład.
6) **generate_report() / print_summary()**: oblicza statystyki (mean/std/min/max, procent próbek z anomaliami) i zapisuje `simulation_report.json`.

## 23. Integracja i rozszerzenia w kodzie
- **Jako biblioteka ETL**: po `sim.run()` użyj `sim.get_node_dataframe()` i `sim.get_edge_dataframe()` jako wejścia do dalszych pipeline’ów (np. agregacje w Pandas/Polars lub zapis do bazy).
- **Symulacja strumieniowa**: `FlowDataGenerator.stream_time_series(real_time=True)` może zasilać kolejkę/Kafkę; w trybie testowym `real_time=False` emituje bez opóźnień.
- **Kontrola losowości**: dodaj `config.seed = 123` przed `FlowSimulator(config)`, aby uzyskać powtarzalny wynik (wykorzystywane w generatorze danych i anomalii).
- **Własne metryki/wizualizacje**: rozbuduj `visualization/plotter.py`, dodaj nowe funkcje i wywołaj je w `FlowSimulator.visualize()`.
- **Nowe typy danych w plikach**: dopisz kolumny do DataFrame’ów w generatorze/anomaliach, następnie uwzględnij je w `SimulationDataSaver` oraz ewentualnie w wizualizacjach.

## 24. Schemat kolumn plików (dokładniej)
- `flow_measurements.*`
  - `timestamp` (ISO), `node_id`, `node_type` (`source|hub|consumer`), `flow` (m³/h), `anomaly_type` (`none|leak|meter_error`), `anomaly_active` (bool).
- `edge_flows.*`
  - `timestamp`, `edge_id`, `source`, `target`, `length` (m), `flow` (m³/h), `anomaly_type`, `anomaly_active`.
- `anomalies.csv`
  - `id`, `type` (`leak|meter_error`), `target_type` (`edge|node`), `target_id`, `magnitude`, `mode` (`const|progressive|add|mul|drift`), `start_time`, `duration_minutes`.
- `topology_info.json`
  - `metadata` (liczby węzłów/krawędzi/źródeł/konsumentów/hubów, typ topologii),
  - `nodes`: `id`, `type`,
  - `edges`: `id`, `source`, `target`, `length`.
- `simulation_report.json`
  - `simulation_info`, `topology_info`, `flow_statistics`, `edge_statistics`, `anomaly_statistics` (procent próbek z anomaliami = średnia z `anomaly_active` * 100).

## 25. Sanity-checki i diagnostyka danych
- **Brak anomalii**: sprawdź, czy `anomaly_probability` > 0 i nie używasz `--disable-*`.
- **Bilans przepływów**: dla węzłów konsumenckich suma przepływów powinna równać się przepływowi na węźle nadrzędnym (poza anomaliami). W Pandas: grupuj po czasie i porównuj z sumą dzieci.
- **Anomaly coverage**: `all_data['anomaly_active'].mean()` daje udział próbek dotkniętych anomalią.
- **Rozkład typów**: `all_data['anomaly_type'].value_counts()`; analogicznie dla `anomalies.csv`.
- **Spójność grafu**: `topology.get_topology_info()` pozwala zweryfikować liczbę węzłów/źródeł względem ustawień.

## 26. Zaawansowane przykłady użycia
### A) Krótka symulacja testowa do CI
```python
from project import FlowSimulator, SimulationConfig
config = SimulationConfig(num_nodes=3, duration_hours=0.1, sampling_frequency_hz=1.0, anomaly_probability=0.2)
sim = FlowSimulator(config)
sim.setup(); sim.run()
print(sim.get_node_dataframe().head())  # szybka weryfikacja
```

### B) Własny pipeline agregujący dobowe sumy
```python
import pandas as pd
from project import FlowSimulator, SimulationConfig

sim = FlowSimulator(SimulationConfig(num_nodes=20, duration_hours=48, sampling_frequency_hz=0.1))
sim.setup(); sim.run()
node_df = sim.get_node_dataframe()
daily = node_df.set_index("timestamp").groupby("node_id").resample("24H")["flow"].sum()
print(daily.head())
```

### C) Wyłączenie wizualizacji dla dużego przebiegu
```bash
python main.py --nodes 120 --duration 24 --sampling 0.05 --anomaly-prob 0.05 --no-visualize
```
