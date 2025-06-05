# Handibur – Real-Time Hebrew Sign Language Translation 📲🖐️💬

> **Flagship project of the WMSM research line** — a mobile-first, WebRTC-enabled framework that recognises single-word **gestures at 98 % top-1 accuracy** and assembles full sentences that reach **21.15 BLEU-4 / 25.40 SacreBLEU**, all while running in real time on commodity hardware.

---

## 🧭 Why Handibur?

Millions of Deaf and hard-of-hearing Israelis rely on HSL, yet qualified interpreters are scarce and expensive. Handibur removes that barrier with an app-centric, cloud-assisted pipeline that:

* Captures a signer’s video on any modern smartphone.
* Streams it over low-latency WebRTC to an **AI server** that recognises single-word gestures and assembles full sentences in real time.
* Returns live text through the same peer connection, so hearing users can read instantly.

No depth cameras, sensor gloves, or multi-cam rigs required.

---

## 📌 Key Contributions

* **WMSM: Word-Model + Sentence-Mechanism.** A two-tier design that replaces heavyweight Seq2Seq transformers with a light word recogniser and a consensus-based sentence module, slashing latency below 100 ms round trip
* **Three-phase augmentation pipeline** (video-, frame- and landmark-level) that inflates a 2 342-clip dataset into **252 712 augmented samples** without GPUs, overcoming data scarcity and over-fitting.
* **Custom attention layer, confidence-scaling activation & multi-term loss** to boost fine-grained gesture separation while enforcing temporal smoothness.
* **Dual validation** (clean + augmented) logged every epoch to catch over-fitting early and guarantee robustness to real-world noise.

---

# 🔬 ML Pipeline

## 1 · Data & Augmentation

| Stage                     | What happens                                                                                                        | Output size |
| ------------------------- | ------------------------------------------------------------------------------------------------------------------- | ----------- |
| **Raw corpus**            | 2 342 single‐gesture MP4s (one word / clip)                                                                         | 2 342       |
| **Video-level aug.**      | ±5° rotation, ±5 px shift, 2 % zoom, shear, brightness / contrast, 11 % left-hand flips                             | 25 762      |
| **Frame & landmark aug.** | ≤ 40 frames / video, isotropic scale, colour jitter, ±6° landmark rotation, Gaussian noise, palm-centred transforms | **252 712** |
| **On-the-fly aug.**       | Timing warps, random jog, per-finger noise (training only)                                                          | —           |

*All augmentation runs on CPU (`ThreadPoolExecutor`) to avoid long-run GPU packet loss.*

## 2 · Word-Level Model (WMSM-Word)

```
Input 40×42  →  TimeDistributed Conv1D ×3  →  BN / MaxPool
            →  Custom Attention →  Bi-LSTM  →  Dense512 + Dropout
            →  Dense(2 342) + CustomConfidenceActivation
```

* **12 layers · 2.26 M params**
* **Loss:** CE + Confidence + Temporal-Consistency
* **Optimiser:** Adam 1 e-4, gradient-clipping, EarlyStopping, ReduceLROnPlateau

## 3 · Sentence Mechanism (WMSM-Sent)

1. **Windowing** – 40-frame tensors every 5 frames.
2. **Sliding consensus** (size 5) – majority class + mean confidence.
3. **Dynamic threshold** – base 0.65, +0.05 per repetition, +0.25 for historically over-predicted classes (needs 3 / 5; 4 / 5 for noisy signs).
4. **Token emission** – validated gesture appended to sentence string.

*Sliding windows feed the word model; the consensus aggregator promotes stable tokens and suppresses noise, keeping server inference ≤ 20 ms and WebRTC round-trip ≤ 100 ms.*

---

# 📊 Model Performance

| Variant            | Gesture Top-1 | Top-3 / 5 / 10          | GAP     | BLEU-4 (sent.)                                         |
| ------------------ | ------------- | ----------------------- | ------- | ------------------------------------------------------ |
| Regular CNN + LSTM | **97.84 %**   | 99.72 / 99.93 / 99.96 % | 0.9881  | —                                                      |
| Landmark-Augmented | 97.68 %       | 99.72 / 99.97 / 99.99 % | 0.988 ≈ | —                                                      |
| **Combined**       | **98.50 %**   | —                       | —       | **21.15** (BLEU-1 55.66 · BLEU-2 41.32 · BLEU-3 30.16) |

* Sentence-level SacreBLEU **25.40** confirms robustness.
* AUC ≈ 1.0 for top-50 classes on ROC curve.
* Training: **24 GPU-hrs** on one NVIDIA A100; augmentation: **72 CPU-hrs** on Ryzen 7 3700X (32-thread, 350 GB RAM).

---

### 🏗️ System Architecture

| Tier                                 | Key Tech                                                                                | Role                                                         |
| ------------------------------------ | --------------------------------------------------------------------------------------- | ------------------------------------------------------------ |
| **Mobile App** (Expo + React Native) | React Hooks, Context Providers, Jest & Maestro                                          | Cross-platform UI, 40 FPS video capture, dark / light themes |
| **Signaller**                        | Express 18 on AWS EC2                                                                   | ICE negotiation, health checks, AI-server routing            |
| **AI Server**                        | Python 3.10, aiortc 1.5, TensorFlow 2.17, Keras 3.4.1                                   | Landmark extraction, inference, sentence assembly            |
| **TURN / STUN**                      | CoTurn (Docker)                                                                         | Media relay when P2P fails                                   |
| **Auth + DB**                        | Supabase (Postgres + Realtime)                                                          | Accounts, sessions, analytics                                |
| **Ops / Scaling**                    | Docker 24 + Compose, GitHub Actions, manager-daemon that spawns AI containers on demand | Automated rollout and horizontal scaling                     |

---

### ⚙️ Tech Stack

| Layer                     | Primary Tools                                                          |
| ------------------------- | ---------------------------------------------------------------------- |
| **CV / Landmarking**      | MediaPipe Hands 0.10, OpenCV 4.9, YOLOv8 (research baseline)           |
| **Deep Learning**         | TensorFlow 2.17, Keras 3.4.1, Optuna 2.10, NLTK 3.8, SacreBLEU 2.4     |
| **Backend**               | Python 3.10, aiortc 1.5, FastAPI (prototypes), Express 18              |
| **Mobile**                | React Native 0.74, Expo 50, react-native-webrtc, Jest 30, Maestro 1.29 |
| **DevOps**                | Docker 24, Docker Compose, GitHub Actions, AWS EC2 (A100), Route 53    |
| **Data Ops / Automation** | Selenium 4, Pandas 2.2, NumPy 1.26, TQDM 4.66, ThreadPoolExecutor      |

(This collapses the previous “Tools Used” bullets and “Software Stack” table into one non-duplicated view.)

---

### 🗺️ End-to-End Pipeline

Smartphone Cam → WebRTC (H.264) → Signaller (Express) → AI Server (which sends live text to the video chat)

1. **Capture** — Expo / React Native app streams 40 FPS video via WebRTC.
2. **Landmarking** — AI server runs MediaPipe Hands (21 key-points / frame).
3. **Pre-processing** — sliding window (40 frames, stride 5) → time-aligned tensors.
4. **Word Recognition** — WMSM-Word model processes each tensor.
5. **Sentence Assembly** — WMSM-Sent consensus module produces Hebrew text.

TURN/STUN fallback (CoTurn) maintains low latency behind NAT; Supabase logs sessions and anonymised analytics.

---

### 📚 Key Design Choices

* **Hand-only landmarks**: ignore z-depth, switch Conv2D → Conv1D, shrinking FLOPs.
* **Custom Confidence Activation**: sharper logits, better separation of subtle signs.
* **Dual validation**: evaluate on clean & augmented dev sets each epoch to catch over-fitting instantly.
* **Consensus aggregator**: replaces Seq2Seq, keeping round-trip < 100 ms.
* **CPU-safe preprocessing**: avoids GPU packet loss during multi-day augmentation runs.

---


## ⚡ Hardware Footprint

| Stage        | CPU                              | GPU      | Runtime |
| ------------ | -------------------------------- | -------- | ------- |
| Augmentation | Ryzen 7 3700X, 32-thread         | —        | 72 h    |
| Training     | AMD EPYC via AWS, **A100 40 GB** | 1 × A100 | 24 h    |

---

## 👥 Authors & Acknowledgements

* Eyal Pasha, Nir Tuttnauer, Eva Karkar — Computer Science Faculty, Colman.
* Supervisor **Dr. Galit Haim**.

---

## ⚖️ License

Project assets and source code are released under the **MIT License**. See `LICENSE` for full text.
