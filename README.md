# Smart Valve Fault Detection

Random Forest classification of five industrial pump operating states from motor current and pressure. Based on the supplied **ML mini project** report. This repository is a **software simulation**, not a completed hardware experiment.

## Schnellstart in VS Code (Windows)

1. Diesen Ordner in VS Code mit **Datei > Ordner öffnen** öffnen.
2. Python 3.12 und die Microsoft-Python-Erweiterung verwenden.
3. Im VS-Code-Terminal ausführen:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe demo.py
```

Die Aktivierung der Umgebung ist nicht nötig. So vermeidest du Probleme mit der PowerShell-Ausführungsrichtlinie. Bei Bedarf über **Python: Select Interpreter** `.venv` auswählen. F5 startet ebenfalls `demo.py`.

`demo.py` erzeugt 1.500 simulierte Messfenster (300 pro Klasse), trainiert das Modell und speichert Ergebnisse. Es braucht keinen ESP32 und nach der Installation keine Internetverbindung.

## Outputs

- `data/synthetic_valve_dataset.csv`: zwölf Features und ein Label pro Sekunde.
- `data/synthetic_valve_dataset.metadata.json`: Herkunft und Zufallsstartwert.
- `results/model_results.png`: Konfusionsmatrix und Feature-Bedeutung.
- `results/evaluation.md`: tatsächlich berechnete Bewertung.
- `results/metrics.json`: maschinenlesbare Metriken und Daten-Prüfsumme.
- `results/valve_model.joblib`: trainierte Pipeline inklusive Skalierung.
- `results/predictions.csv`: Vorhersagen mit maximaler Klassenwahrscheinlichkeit; diese ist keine kalibrierte Zuverlässigkeitsgarantie.


## Method

Classes: `all_closed`, `valve1_open`, `valve2_open`, `valve3_open`, `leak_detected`.

Each simulated window contains 1,000 current samples at 1,000 Hz and one pressure value. Signals combine a 50 Hz fundamental, 150/250 Hz harmonics, class-dependent amplitude, noise and pressure. These are illustrative assumptions, not measured pump physics. Feature extraction uses RMS, mean absolute current, standard deviation, peak, crest factor, variance, dominant frequency/magnitude, third/fifth harmonic energy, total energy and pressure.

For compatibility with the report firmware, dominant frequency is searched at 40..300 Hz in 5 Hz steps; magnitude is `abs(DFT)/1000`; harmonic energy is the sum of squared normalized magnitudes in 145..155 and 245..255 Hz. `total_energy` is the time-domain sum of squared samples. It is **not** the unnormalized FFT energy claimed in the report's feature table. Standard deviation describes the AC waveform as well as noise; it is not a noise-only measurement.

The split is stratified 80/20 with seed 42. Five-fold shuffled stratified CV runs only on training data. `StandardScaler` is inside the pipeline and fitted independently in each fold. The Random Forest has 100 trees, square-root feature selection and seed 42. Feature importance uses impurity reduction; correlated features can share importance.

High scores on this synthetic distribution are not evidence of real-world accuracy. The report's 100% result is not assumed or copied. Adjacent windows from real experiments should be evaluated with run/session groups and a separate deployment test. The scikit-learn model runs on the laptop; no embedded classifier deployment is claimed.

## Einzelne Schritte

```powershell
python generate_dataset.py
python train_model.py
python predict.py --data data/synthetic_valve_dataset.csv
python -m unittest discover -s tests -v
```

Only load trusted local `.joblib` files. They contain executable Python serialization.

## Optional: echte ESP32-Daten

Firmware: `firmware/esp32_valve_sensor/esp32_valve_sensor.ino`. Copy `secrets.example.h` to `secrets.h`, enter Wi-Fi and laptop IP locally, and open the sketch in Arduino IDE with an ESP32 board package. Private `secrets.h` is excluded from Git. GPIO 34: conditioned CT signal; GPIO 35: conditioned pressure signal. Calibration constants must be determined on the actual rig. Do not connect mains or a current-output CT directly to an ADC: suitable isolation, burden resistor, bias and voltage protection are required. Pressure input must stay within the ESP32 ADC input range.

Set one of the five labels via Serial Monitor at 115200 baud, e.g. `L:valve1_open`. Collect on a trusted local network:

```powershell
python receiver.py --host 0.0.0.0 --out data/measured_valve_dataset.csv
python train_model.py --data data/measured_valve_dataset.csv --source measured --out results_measured
```

The default receiver only listens on localhost. UDP is unauthenticated and can lose packets. Firmware acquisition has software timing and a processing gap between windows; it is not a hard real-time 1-row-per-second guarantee. Firmware has not been compiled or hardware-tested in this delivery.

## Kurzvortrag

„Mein Projekt klassifiziert fünf Ventilzustände anhand von Strom- und Druckmerkmalen. Für diese Software-Demo simuliere ich 1.500 einsekündige Messfenster. Zwölf Merkmale beschreiben Amplitude, Signalform, Frequenzanteile und Druck. Ein Random Forest wird mit einem getrennten Testdatensatz und fünffacher Kreuzvalidierung bewertet. Die angezeigten Ergebnisse stammen aus meiner Simulation; für industrielle Aussagen fehlen reale Messungen. Die ESP32-Schnittstelle ermöglicht später die Aufnahme echter Daten.“

## Source

User-provided report: *Smart Valve Fault Detection in Industrial Pump Systems Using Random Forest Classification*, 23IR602, PSG College of Technology. Implementation adapted and corrected from its described approach; no claim is made to reproduce its original experimental dataset.
