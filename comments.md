Sì. Prima di aggiungere altre feature, renderei lo **smoke test una specie di tutorial diagnostico del sistema**. In questo momento funziona come test automatico della pipeline, ma non è progettato per essere capito da una persona guardando la GUI.

### Cosa sta testando davvero lo smoke test attuale

Il synthetic genera **12 trial**, nella sequenza:

```text
LEFT(13 Hz) → RIGHT(21 Hz) → NONE
LEFT(13 Hz) → RIGHT(21 Hz) → NONE
... ×4
```

su due canali `Oz/O1` a 128 Hz. LEFT e RIGHT ricevono praticamente una sinusoide pura alla frequenza corrispondente; NONE riceve solo un piccolo segnale a 7 Hz. Quindi è intenzionalmente un problema facilissimo: verifica che event → window → FFT/features → model funzioni, non simula ancora EEG realistico.

Con 4 esempi per classe e lo split stratificato 50/25/25, ottieni per ciascuna classe:

```text
2 calibration
1 validation
1 test
```

quindi in totale 6 calibration + 3 validation + 3 test.

Il primo fit può quindi avvenire dopo il sesto trial di calibrazione, perché `batch_size_trials=6`.

### Perché va velocissimo

Non simula il tempo reale. Il source synthetic restituisce **8 campioni alla volta a 128 Hz**, cioè avanza di 62.5 ms di EEG ad ogni poll, mentre l'engine aspetta soltanto 10 ms tra i poll. Quindi gira circa 6× più veloce del tempo simulato.

Questo va bene per CI/testing, ma è pessimo per capire visivamente cosa succede.

---

# Perché nello smoke vedi quasi sempre `NONE`

Qui c'è un problema concettuale reale.

La decision policy richiede attualmente:

```text
posterior > 0.85
AND
2 finestre consecutive
```

prima di emettere una command.

Ma l'engine produce **una sola prediction per trial**.

E i trial di validation/test arrivano:

```text
LEFT
RIGHT
NONE
LEFT
RIGHT
NONE
```

Quindi non hai mai:

```text
LEFT
LEFT
```

oppure:

```text
RIGHT
RIGHT
```

su due prediction consecutive.

Di conseguenza **`NONE` è quasi il risultato atteso per costruzione**.

Questo non significa che il classifier non funzioni. Significa che abbiamo applicato una decision policy pensata per finestre temporali consecutive a un engine che produce una sola prediction per stimulus trial.

Inoltre hai già:

```yaml
inference_stride_seconds: 0.25
```

ma l'attuale realtime engine non lo usa per produrre sliding predictions.

Questa è la prima cosa scientificamente importante che sistemerei.

---

# E il `model v0` del tuo screenshot reale?

Quello è un problema separato.

Lo screenshot mostra:

```text
BCI-EEG-Replay | 9 ch @ 256 Hz
```

quindi è Kalunga/MOABB replay, non synthetic.

`phase: CALIBRATING` non dimostra che siano già arrivati trial: l'engine entra in `CALIBRATING` **immediatamente dopo la connessione**, prima di ricevere il primo evento.

Perciò:

```text
CALIBRATING
model v0
```

potrebbe significare indifferentemente:

```text
0 eventi ricevuti
```

oppure:

```text
3 trial ricevuti, aspetto il batch
```

oppure:

```text
6 trial ricevuti ma condizioni di fit non soddisfatte
```

La GUI attuale non te lo dice. Questo è il problema principale.

L'engine pubblica già `TrialStarted`, `TrialCompleted` e `CalibrationBatchReady`, ma `MainWindow` non usa `TrialStarted` né `CalibrationBatchReady`.

Quindi stiamo buttando via proprio le informazioni necessarie per capire perché siamo a `v0`.

---

# V1.1: farei uno “Transparent Smoke Test”

Il prossimo step lo imposterei così:

1. **Separare due synthetic mode.** `classifier_smoke` deve verificare feature + model e usare una decisione semplice (`consecutive_windows=1`). `controller_smoke` deve invece avere stimulus block lunghi e sliding windows ogni 250 ms, così testa davvero evidence accumulation e latency.

2. **Aggiungere un clock controllabile dalla GUI:** `Pause`, `Play`, `Step trial`, `Restart`, e velocità `0.25× / 0.5× / 1× / 2× / 5×`. Lo smoke tutorial partirebbe a `1×`, non alla massima velocità.

3. **Mostrare esplicitamente il trial corrente.** Per esempio `stimulus: LEFT | 13 Hz`, `event 4/12`, `split: calibration`, `window collecting: 1.12 / 1.50 s`.

4. **Mostrare perché il modello non è ancora trained.** Qualcosa come `Calibration: LEFT 2/2 | RIGHT 1/2 | NONE 0/2` e `next fit: waiting for RIGHT +1, NONE +2`. Quando il fit avviene: `MODEL v1 TRAINED on 6 trials`.

5. **Sostituire il falso “spectrum”.** Il grafico attuale ha asse x = `feature index`: non è uno spettro, mostra semplicemente il vettore `feature.values`.  Io mostrerei davvero `frequency [Hz] → log PSD`, con linee verticali a 13, 21, 26, 39, 42 Hz. Sotto, tre barre riassuntive `LEFT evidence`, `RIGHT evidence`, `NONE`.

6. **Implementare veramente il latent plot.** Attualmente `LatentPanel` mostra solo del testo; non disegna punti, centri o covarianze.  L'engine calcola già `decoder.diagnostics(...)`, ma poi conserva solo `separation`.  Passerei l'intero `DecoderDiagnostics` alla GUI e mostrerei calibration points + class centers + current sample.

7. **Separare posterior ed evidence accumulator.** Il pannello dovrebbe mostrare contemporaneamente `model posterior: LEFT=.94` e `temporal evidence: LEFT=.72`, con threshold `.85`. Ora mostra le posterior bars e poi soltanto il testo finale della decisione.  Se viene emesso `NONE`, voglio sapere perché: `below threshold`, `waiting 1/2 consecutive windows`, `refractory`, oppure `model unavailable`.

8. **Restart dalla GUI**, ma ricreando completamente experiment/engine/sources/model. Il controller corrente costruisce un singolo `ManagedExperiment` e un singolo worker thread all'avvio.   Non riutilizzerei lo stesso engine dopo uno stop: Restart deve chiamare di nuovo `build_realtime_experiment()`, creare un nuovo artifact directory e riportare il modello a `v0`.

---

## Come vorrei vedere lo smoke test

All'inizio:

```text
SYNTHETIC TUTORIAL
Trial 1 / 18

Ground truth: LEFT
Stimulus: 13 Hz
Phase: CALIBRATION

Calibration
LEFT   0 / 3
RIGHT  0 / 3
NONE   0 / 3

Model: not trained
Reason: collecting calibration examples
```

Poi durante il trial:

```text
Spectrum

         LEFT
          ↓
          │
          /\                         RIGHT
         /  \                          ↓
________/    \_________________________│_________
       13 Hz                         21 Hz
```

e:

```text
Spectral evidence
LEFT    ███████████  +2.8
RIGHT   ██           +0.3
```

Dopo il primo batch:

```text
MODEL v1

latent space

 LEFT ● ● ●


            ○ ○ ○ RIGHT

       × × ×
        NONE
```

Poi in test:

```text
TRUE: LEFT

Model posterior
LEFT     0.96
RIGHT    0.02
NONE     0.02

Temporal evidence
LEFT     0.89

Decision
LEFT

Correct ✓
```

Questo rende il sistema quasi autoesplicativo.

---

## Farei anche lo synthetic leggermente migliore

Non ancora “realistico”, perché sarebbe prematuro. Ma smetterei di usare sinusoidali quasi perfette.

Per ogni trial userei:

[
x_c(t)=
A_{c} \sin(2\pi f t+\phi)
+
B_c\sin(4\pi f t+\phi_2)
+
\epsilon_c(t)
]

con:

* fase casuale per trial;
* ampiezza diversa per canale;
* piccolo secondo armonico;
* rumore gaussiano;
* piccolo background 1/f o low-frequency drift.

Il seed rimane fisso.

Poi nella GUI metterei un controllo:

```text
Synthetic difficulty

[Perfect] [Easy] [Noisy]
```

`Perfect` deve essere quasi 100% accurato e serve per il wiring. `Easy` testa calibration. `Noisy` serve successivamente per confrontare modelli.

---

## Per il controller smoke farei una cosa diversa

Qui vogliamo finalmente testare il `consecutive_windows=2`.

Non trial da 1.5 s con una sola prediction, ma:

```text
LEFT stimulus: 4 s
```

con:

```text
window = 1.5 s
stride = 0.25 s
```

ottenendo:

```text
0.00–1.50
0.25–1.75
0.50–2.00
...
```

e quindi:

```text
P(LEFT) .82
P(LEFT) .91
P(LEFT) .96
             ↓
         emit LEFT
```

Quello sarebbe finalmente un test corretto della catena:

[
\mathrm{EEG}
\rightarrow
p(a|EEG)
\rightarrow
\text{temporal evidence}
\rightarrow
\text{command}.
]

E ci permetterebbe di misurare anche **command latency**, che in prospettiva è molto più importante dell'accuracy pura.

---

### Prima diagnosticherei anche il tuo `model v0`

Con queste modifiche sarà immediato. Nel frattempo, nel run reale la domanda da risolvere è molto precisa:

> dopo che la GUI ha ricevuto **6 trial di calibration validi**, il modello passa a `v1`?

Se sì, il comportamento attuale era soltanto opaco. Se resta `v0`, allora abbiamo un bug nella ricezione annotations → trial reconstruction → split/calibration.

Non toccherei ancora il decoder finché non sappiamo questo.

La V1.1 quindi non è principalmente “fare una GUI più bella”: è fare una **GUI che renda osservabile l'intero esperimento e uno synthetic test con risultati attesi espliciti**. È esattamente ciò che ci serve prima di iniziare a giudicare il modello sui dati EEG reali.
