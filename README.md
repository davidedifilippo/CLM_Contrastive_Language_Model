# Contrastive Language Models (CLM) — Real-Time 60 FPS Decision Benchmark

Questo repository contiene un'indagine empirica e un benchmark sulle prestazioni in **tempo reale (60 FPS)** dell'architettura **CLM (Contrastive Language Models)** applicata al controllo decisionale autonomo di un gioco a scorrimento, eseguito in locale su **Apple Silicon (Mac M4)**.

---

## 🎯 Obiettivo del Progetto

I Large Language Model (LLM) generativi tradizionali prendono decisioni generando testo token per token, introducendo latenze di 1.000–3.000 ms incompatibili con l'interazione in tempo reale a 60 fotogrammi al secondo (budget di 16,6 ms per frame).

Questa sperimentazione valuta come l'architettura **Contrastive Language Model** (sviluppata da ricercatori di Stanford e NVIDIA) risolve il problema disaggregando lo **Stato** dalle **Azioni**:
1. Le azioni possibili (`UP`, `DOWN`) vengono codificate e inserite nella cache vettoriale (`VectorArena`) **una sola volta all'avvio**.
2. Ad ogni frame, lo stato corrente viene proiettato in uno spazio a 512 dimensioni.
3. La decisione viene presa calcolando la **prossimità geometrica (prodotto scalare / cosine similarity)** tra lo stato e le azioni in cache, con tempi di risposta nell'ordine dei **microsecondi**.

---

## 🕹️ Componenti Inclusi nel Repository

### 1. `bird_clm_game.py` (Gioco Interattivo a 60 FPS)
Un gioco a scorrimento fluido in PyQt6 in cui l'uccellino deve superare ostacoli a fessura variabile.
- Include un pannello laterale di **Telemetria Live** con indicatori grafici dei tasti premuti da CLM, distribuzione delle probabilità Softmax, latenza in ms e Hit Rate della cache.
- Modalità selezionabili dal vivo con i tasti `1`, `2`, `3`:
  - **`[1] CLM Fast Head`**: Projection head addestrata con loss InfoNCE sulla fisica dei varchi (~0,02 ms su cache hit, 100% precisione).
  - **`[2] CLM-8B Zero-Shot`**: Interroga il modello generale da 8 miliardi di parametri (`Qwen3-8B` su GPU Metal via `llama-server`).
  - **`[3] Giocatore Umano`**: Controllo manuale tramite frecce della tastiera (`SU` / `GIÙ`).

### 2. `bird_clm_vision_game.py` (Computer Vision + CLM)
Un'architettura completa a circuito chiuso in cui l'AI **non ha accesso alle coordinate interne del gioco**:
- **Cattura Schermo (`mss`)**: cattura i pixel grezzi della finestra a 60 FPS.
- **Segmentazione (`OpenCV`)**: isola l'uccellino (maschera colore HSV) e calcola il centro del varco tra i tubi ostacolo escludendo il terreno, determinando distanza, quota e velocità verticale in ~0,4 ms.
- **Cervello Decisionale (`CLM`)**: converte l'osservazione visiva in stato testuale e invia l'azione al gioco in ~0,02 ms.
- Finestra a tre riquadri: **Gioco Reale**, **Vista a Raggi X di OpenCV**, **Telemetria CLM**.

### 3. `benchmark_m4_real.py` (Script di Benchmark Terminale)
Script di test automatizzato che:
- Connette `llama-server` (con `--embeddings --pooling last` su GPU Metal) ai pesi di `Qwen3-8B`.
- Misura la latenza a freddo (cache miss dell'encoder 8B) e a caldo (cache hit nella `VectorArena`).
- Esegue una simulazione completa di 60 secondi del gioco T-Rex senza scudo di sicurezza.

---

## 📊 Risultati Misurati su Apple M4

Test eseguiti su **Apple M4 (24 GB Unified Memory)** a 60 FPS (budget: 16,6 ms per fotogramma):

| Componente | Latenza Mediana (p50) | Note |
| :--- | :--- | :--- |
| **Estrazione Visiva (OpenCV + mss)** | **0,40 ms** | Tracciamento contorni, filtro HSV e calcolo varco |
| **Decisione CLM (Cache HIT)** | **0,017 ms** *(17 µs)* | Prodotto scalare e Softmax su `VectorArena` |
| **Decisione CLM (Cache MISS, Qwen3-8B)** | **~176 ms** | Calcolo embedding 4096-d su Metal GPU |
| **VectorArena Cache Hit Rate** | **> 92%** | Oltre 9 decisioni su 10 sono servite istantaneamente dalla cache |
| **Tempo totale ciclo per frame (Vision + CLM)** | **< 1,0 ms** | **Consuma meno del 6% del budget dei 60 FPS** |
| **Tasso di successo (Crash)** | **0 morti** | Volo continuo perfetto attraverso tutti gli ostacoli |

---

## 🚀 Come Eseguire il Progetto

### Requisiti
- macOS con Apple Silicon (M1/M2/M3/M4)
- Python 3.10+ (consigliato Python gestito tramite `uv`)
- Librerie: `PyQt6`, `opencv-python`, `mss`, `torch`, `contrastive-lm`, `numpy`

### 1. Configurazione Ambiente
```bash
git clone https://github.com/davidedifilippo/CLM_Contrastive_Language_Model.git
cd CLM_Contrastive_Language_Model

# Crea il virtual environment e installa le dipendenze
uv venv .venv --system-site-packages
source .venv/bin/activate
uv pip install numpy requests httpx fastapi uvicorn torch
uv pip install --no-deps --no-build-isolation -e .
```

### 2. Avviare la Demo Interattiva dell'Uccellino (Direct Mode)
```bash
python bird_clm_game.py
```

### 3. Avviare la Demo con Computer Vision (mss + OpenCV + CLM)
```bash
python bird_clm_vision_game.py
```

### 4. Eseguire il Benchmark su Terminale
```bash
python benchmark_m4_real.py
```

---

## 📚 Riferimenti e Crediti

- **Architettura CLM originale:** Stanford University & NVIDIA ([Contrastive-LM/CLM](https://github.com/Contrastive-LM/CLM)).
- **Modello Encoder:** Alibaba Cloud `Qwen/Qwen3-8B`.
- **Implementazione Benchmark & Real-Time Bird Player:** Ing. Davide Di Filippo.

---

## ⚖️ Licenza e Disclaimer (License & Disclaimer)

Questo progetto è rilasciato sotto licenza [Apache 2.0](LICENSE).

### Disclaimer
```text
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

> **Nota:** Questo software è fornito esclusivamente a scopo di ricerca, studio e benchmark prestazionale sperimentale. L'autore non si assume alcuna responsabilità per utilizzi al di fuori di tali scopi.

