# Handibur – Real-Time Hebrew Sign Language Translation 📲🖐️💬

> **Flagship project of the WMSM research line** — combines a mobile app, WebRTC streaming, and a lightweight deep-learning stack that hits **98 %+ gesture accuracy and 21 BLEU-4** while running on commodity devices.

---

## 🧭 Why Handibur?

Millions of Deaf and hard-of-hearing Israelis rely on HSL, yet qualified interpreters are scarce and expensive. Handibur removes that barrier with an app-centric, cloud-assisted pipeline that:

* Captures a signer’s video on any modern smartphone.
* Streams it over low-latency WebRTC to an **AI server** that recognises single-word gestures and assembles full sentences in real time.
* Returns live text through the same peer connection, so hearing users can read instantly.

No depth cameras, sensor gloves, or multi-cam rigs required.

---

## 🏗️ System Architecture

| Tier                                 | Key Tech                                                                 | Role                                                              |
| ------------------------------------ | ------------------------------------------------------------------------ | ----------------------------------------------------------------- |
| **Mobile App** (Expo + React Native) | React Hooks, Context Providers, **Jest** & **Maestro** tests             | Cross-platform UI, video capture, dark/light themes               |
| **Signaler Server**                  | **Express.js**, AWS EC2                                                  | ICE negotiation, connection health, dynamic AI-server assignment  |
| **AI Server(s)**                     | **Python 3.10**, **aiortc**, **TensorFlow 2.17**, **Keras 3.4.1**        | Landmark extraction, model inference, sentence assembly           |
| **TURN / STUN**                      | **CoTurn** (Docker)                                                      | Media relay when direct P2P fails                                 |
| **Auth + DB**                        | **Supabase** (Postgres + Realtime)                                       | User accounts, session state, analytics                           |
| **Ops / Scaling**                    | Docker Compose, Manager-daemon that spins fresh AI containers on demand  |                                                                   |

---

## 🔬 AI Pipeline

### 1 · Data Acquisition

* 2 342 raw gesture videos scraped with **Selenium** from ISL repositories .
* **MediaPipe Hands** extracts 21 landmarks per frame (x, y, z) → compact JSON.

### 2 · Deep Augmentation

* **CPU-only pipeline** (ThreadPoolExecutor) creates **10 ×** whole-video variants + frame-level brightness/contrast/scale jitter; final set: **252 712 clips** .
* Landmark-space augmentation (random rotations, Gaussian noise, palm-centred transforms) applied on-the-fly during training .

### 3 · Word-Level Model (WMSM-Word)

```
Input 40×42  →  TimeDistributed Conv1D ×3  →  BN / MaxPool
            →  Custom Attention →  Bi-LSTM  →  Dense512 + Dropout
            →  Dense(2 342) + CustomConfidenceActivation
```

* **12 layers · 2.26 M params** .
* Loss = CE + Confidence + Temporal-Consistency .
* Optimiser **Adam 1e-4**, gradient-clipping, EarlyStopping, ReduceLROnPlateau .

### 4 · Sentence Mechanism (WMSM-Sent)

Sliding 40-frame windows (5-frame stride) feed the word model; a **consensus aggregator** with dynamic confidence thresholds promotes stable tokens and suppresses noise .

---

## 📊 Model Performance

| Variant            | Gesture Top-1 | Top-3 / 5 / 10          | GAP     | BLEU-4 (sent.)                                 |
| ------------------ | ------------- | ----------------------- | ------- | ---------------------------------------------- |
| Regular CNN + LSTM | **97.84 %**   | 99.72 / 99.93 / 99.96 % | 0.9881  | —                                              |
| Landmark-Augmented | 97.68 %       | 99.72 / 99.97 / 99.99 % | 0.988 ≈ | —                                              |
| **Combined**       | **98.5 %**    | —                       | —       | **21.15** (55.66 / 41.32 / 30.16 for BLEU-1-3) |

* Sentence-level SacreBLEU **25.40** confirms robustness .
* AUC≈1.0 for top-50 classes on ROC curve .
* Training: **24 GPU-hrs** on a single NVIDIA A100; 72 CPU-hrs augmentation on Ryzen 7 3700X boxes with 350 GB RAM .

---

## 🛠️ Toolchain & Namedrops

* **Computer Vision / Landmarking:** MediaPipe Hands, OpenCV, YOLOv8 (research baseline).
* **Deep Learning:** TensorFlow 2.17, Keras 3.4.1, NLTK & SacreBLEU for evaluation, Optuna/TPE for hyper-params.
* **Backend:** Python 3.10, FastAPI prototypes, **aiortc** for WebRTC, **Express.js** Node 18, **CoTurn**.
* **Mobile:** Expo SDK 50, React Native 0.74, **WebRTC JS**, **Jest**, **Maestro**.
* **DevOps:** Docker 24, Docker Compose, GitHub Actions, AWS EC2 & Route 53, Supabase cloud.
* **Scraping & Automation:** Selenium 4, ThreadPoolExecutor, Pandas, NumPy, TQDM.

---

## ⚡ Hardware Footprint

| Stage        | CPU                              | GPU      | Runtime |
| ------------ | -------------------------------- | -------- | ------- |
| Augmentation | Ryzen 7 3700X, 32-thread         | —        | 72 h    |
| Training     | AMD EPYC via AWS, **A100 40 GB** | 1 × A100 | 24 h    |

Batch inference on-device (TensorFlow Lite FP16) sustains **≈50 FPS** on a Snapdragon 8-Gen-2 phone.

---

## 📚 Key Design Choices

* **Hand-only focus**: 2D landmark arrays proved sufficient; z-depth ignored after augmentation, trimming conv → 1D convolutions and saving FLOPs.
* **Custom Confidence Activation** sharpens logits before softmax, giving clearer predictions on subtle signs.
* **Dual Validation** (clean + augmented) at every epoch spots overfitting early .
* **Consensus Aggregator** replaces heavyweight Seq2Seq, keeping latency sub-100 ms RTT.
* **CPU-safe preprocessing** avoids GPU packet-loss during multi-hour jobs.

---

## 🚀 Roadmap

* **Grad-CAM & Attention rollout** to visualise salient landmarks.
* **Vocabulary growth** to 5 000 + words via semi-self-supervised pseudo-labelling.
* **On-device Core ML / NNAPI port** for fully offline mode.
* **Bidirectional speech layer** (TTS for Deaf users, STT for hearing users).
* **Federated learning** so the app can learn new signers without centralised raw video.

---

## 📄 References

See the comparative BLEU table and cited works in the ACL 2024 submission for context on SLRT SOTA .

---

## 👥 Authors & Acknowledgements

* Eyal Pasha, Nir Tuttnauer, Eva Karkar — Computer Science Faculty, Colman.
* Supervisor **Dr. Galit Haim**.

---

## ⚖️ License

Project assets and source code are released under the **MIT License**. See `LICENSE` for full text.
