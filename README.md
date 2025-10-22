# Opis projektu

Celem projektu jest zaprojektowanie i implementacja systemu symulującego dane pomiarowe w sieciach przesyłowych (np. wodociągowych lub gazowych) w celu analizy bilansu przepływów w czasie rzeczywistym. System ma na celu odtworzenie realistycznego zachowania sieci, w tym występowania losowych zakłóceń pomiarowych i anomalii, takich jak wycieki lub błędy liczników.

Projekt stanowi etap przygotowawczy do budowy pełnego systemu detekcji anomalii – w niniejszej wersji skupia się wyłącznie na generowaniu i wizualizacji danych symulacyjnych.

System będzie umożliwiał:
*   tworzenie wirtualnej topologii sieci przesyłowej w postaci grafu,
*   generowanie syntetycznych szeregów czasowych przepływów,
*   dodawanie szumu pomiarowego oraz symulowanie anomalii,
*   zapisywanie i wizualizację danych w czasie rzeczywistym.

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
    Dane są prezentowane w środowisku wizualnym w postaci wykresów przepływów w czasie, z możliwością filtrowania według punktu pomiarowego i zakresu czasowego.

*   **Eksport danych**
    System umożliwia eksport wyników symulacji do formatu CSV lub JSON w celu dalszej analizy zewnętrznej.

*   **Raport symulacji**
    Po zakończeniu symulacji generowany jest raport zawierający podstawowe statystyki, takie jak średni przepływ, odchylenie standardowe, liczba i rodzaj wprowadzonych anomalii.

## Ograniczenia projektu
*   Projekt nie obejmuje implementacji algorytmów detekcji anomalii.
*   Dane są generowane w sposób syntetyczny i nie pochodzą z rzeczywistych urządzeń pomiarowych.
*   System nie realizuje jeszcze logiki wnioskowania (np. klasyfikacji typu awarii).
*   Aplikacja ma charakter prototypowy i działa w środowisku lokalnym.

## Wymagania techniczne
*   

# Wymagania niefunkcjonalne
*   **Wydajność** – System powinien generować dane dla minimum 100 punktów pomiarowych z częstotliwością 1 Hz w czasie rzeczywistym.
*   **Skalowalność** – Architektura systemu powinna umożliwiać łatwe rozszerzenie liczby węzłów i zwiększenie złożoności topologii sieci.
*   **Spójność danych** – Dane zapisywane muszą zachowywać bilans przepływów zgodny z topologią sieci (poza przypadkami anomalii).
*   **Bezpieczeństwo** – Dostęp do bazy danych powinien być chroniony hasłem lub tokenem uwierzytelniającym.
*   **Czytelność kodu** – Kod projektu powinien być modularny i opatrzony komentarzami zgodnymi.
*   **Przenośność** – System powinien działać w sposób identyczny w różnych środowiskach systemowych.
*   **Niezależność sieciowa** – Symulacja musi być możliwa do uruchomienia w trybie offline.
*   **Możliwość wizualizacji** – Dane muszą być dostępne w czasie rzeczywistym, a wykresy powinny umożliwiać interaktywną analizę (powiększanie, wybór zakresu czasowego).
