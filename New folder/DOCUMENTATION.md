# DeepForensics — Full Technical Documentation

> Explainable deepfake forensics. Submit an image or video, get a REAL/FAKE verdict with calibrated confidence, semantic breakdown, Grad-CAM heatmaps, and a downloadable PDF report.

---

## Table of Contents

1. [What This System Does](#1-what-this-system-does)
2. [System Architecture](#2-system-architecture)
3. [Technology Stack](#3-technology-stack)
4. [Repository Layout](#4-repository-layout)
5. [Getting Started](#5-getting-started)
6. [ML Pipeline — Phase 1](#6-ml-pipeline--phase-1)
   - 6.1 Datasets
   - 6.2 Preprocessing
   - 6.3 Visual Feature Extractor
   - 6.4 Semantic Modules
   - 6.5 Multi-Modal Fusion
   - 6.6 Grad-CAM Explanation
   - 6.7 Inference API
   - 6.8 Training
   - 6.9 Evaluation
7. [Web Application — Phase 2](#7-web-application--phase-2)
   - 7.1 Backend (FastAPI)
   - 7.2 Frontend (React + Vite)
   - 7.3 PDF Report
   - 7.4 Deployment
8. [Design System](#8-design-system)
9. [Schemas & Data Contracts](#9-schemas--data-contracts)
10. [Testing](#10-testing)
11. [Coding Conventions](#11-coding-conventions)
12. [Benchmark Results](#12-benchmark-results)
13. [Glossary](#13-glossary)

---

## 1. What This System Does

DeepForensics is a **fully self-contained deepfake detection system** built on real, trained PyTorch models — no LLM in the inference path, no simulated results.

Given any image or video file, the system:

1. Extracts faces, frames, landmarks, and audio
2. Runs an EfficientNet-B4 visual backbone to extract a 1792-dimensional embedding
3. Computes three deterministic semantic signals: lip-sync consistency, eye-blink naturalness, and lighting coherence
4. Fuses all signals through a calibrated MLP to produce a `REAL` / `FAKE` verdict with a probability score
5. Generates a Grad-CAM heatmap highlighting the face regions that drove the decision
6. Returns a structured `AnalysisResult` object (or a downloadable PDF report via the web interface)

This rebuild replaces a prior prototype that called an LLM to simulate detection. The entire inference path is deterministic and fully auditable.

---

## 2. System Architecture

```
                          ┌──────────────────────────────┐
                          │  React + Vite Frontend       │
                          │  (web/frontend)              │
                          └──────────┬───────────────────┘
                                     │ HTTPS / multipart
                                     ▼
                          ┌──────────────────────────────┐
                          │  FastAPI Service              │
                          │  (web/backend)                │
                          │  - upload, status, result     │
                          │  - heatmap PNG, PDF report    │
                          └──────────┬───────────────────┘
                                     │ in-process import
                                     ▼
  ┌────────────────────────────────────────────────────────┐
  │  ml.inference                                          │
  │  ─────────────                                         │
  │  analyse(path) ─▶ AnalysisResult                       │
  │   • EfficientNet-B4 visual features (1792-d)           │
  │   • Lip-sync: MFCC × MAR Pearson r                     │
  │   • Eye-blink: EAR time-series anomaly                 │
  │   • Lighting: Lambertian shading-direction variance    │
  │   • Multi-modal fusion: FC(512)→FC(128)→softmax        │
  │   • Grad-CAM heatmap (last conv layer)                 │
  └────────────────────────────────────────────────────────┘
                                     ▲
                                     │ loads at startup
                       ┌─────────────────────────────┐
                       │  ml/checkpoints/             │
                       │  efficientnet_b4_ff++.pt     │
                       │  fusion_head.pt              │
                       └─────────────────────────────┘
```

**Key design decision:** The FastAPI service imports `ml.inference` directly as a Python module. There is no HTTP call, no subprocess, no IPC overhead between the ML layer and the API layer. A single Python virtualenv handles both. The frontend is a separate Node project communicating via a clean REST API.

---

## 3. Technology Stack

### ML (Phase 1)

| Layer | Tool |
|---|---|
| Language | Python 3.11 |
| Deep learning | PyTorch 2.3 + torchvision |
| Model zoo | `timm` — EfficientNet-B4 with ImageNet-pretrained weights |
| Face detection | MediaPipe Face Mesh (primary), RetinaFace (fallback) |
| Video / image I/O | OpenCV 4, decord (fast video decoding) |
| Audio | librosa, soundfile |
| Explainability | `pytorch-grad-cam` |
| Scientific | NumPy, SciPy, scikit-learn |
| Experiment tracking | Weights & Biases (free tier) |
| Config | Hydra + pydantic |
| Testing | pytest |
| Lint / format | ruff, black, mypy (strict on `ml/` core) |

### Web (Phase 2)

| Layer | Tool |
|---|---|
| Backend | FastAPI + Uvicorn |
| Background jobs | FastAPI `BackgroundTasks` (single-instance) |
| Frontend build | Vite 5 + TypeScript 5 |
| UI framework | React 18, shadcn/ui (Radix), Tailwind CSS v3 |
| State management | Zustand |
| Routing | React Router v6 |
| Motion | Framer Motion (verdict reveals and count-ups only) |
| PDF generation | ReportLab (server-side) |
| Testing | Vitest + Playwright |

---

## 4. Repository Layout

```
deepforensics/
├── CLAUDE.md                       # Project specification and phase rules
├── DOCUMENTATION.md                # This file
├── README.md                       # Short public-facing readme
├── pyproject.toml                  # Single Python venv for ML + backend
├── .python-version                 # 3.11
├── .env.example                    # Documented environment variables
│
├── ml/
│   ├── inference.py                # Public ML API: analyse(path) → AnalysisResult
│   ├── schemas.py                  # Pydantic models — shared with backend
│   │
│   ├── data/
│   │   ├── download.py             # SHA-256 checksum verification
│   │   └── manifests/
│   │       ├── ffpp_train.csv      # Video-level splits
│   │       ├── ffpp_val.csv
│   │       ├── ffpp_test.csv
│   │       ├── celebdf_test.csv
│   │       ├── dfdc_test.csv
│   │       └── checksums.json
│   │
│   ├── preprocessing/
│   │   ├── frame_extract.py        # video → sampled frames at target fps
│   │   ├── landmarks.py            # 468-mesh → EAR / MAR series
│   │   └── audio.py                # MFCC extraction
│   │
│   ├── datasets/
│   │   ├── ffpp.py                 # FaceForensics++ Dataset class
│   │   ├── celebdf.py              # Celeb-DF v2 (test split only)
│   │   └── dfdc.py                 # DFDC preview Dataset class
│   │
│   ├── models/
│   │   ├── fusion.py               # MLP fusion head + calibration loss
│   │   └── semantic/
│   │       ├── lipsync.py          # MFCC × MAR Pearson correlation
│   │       ├── blink.py            # EAR time-series anomaly
│   │       └── lighting.py         # Lambertian shading analysis
│   │
│   ├── training/
│   │   ├── callbacks.py            # Checkpointer + EarlyStopping
│   │   └── config/                 # Hydra YAML configs per experiment
│   │
│   ├── evaluation/
│   │   ├── eval_visual.py          # Visual-only baseline evaluator
│   │   └── eval_full_pipeline.py   # Cross-dataset full-pipeline eval
│   │
│   └── checkpoints/                # Gitignored — trained model weights
│
├── web/
│   ├── backend/                    # FastAPI service (Phase 2)
│   └── frontend/                   # Vite + React (Phase 2)
│
├── docs/
│   ├── phase1-plan.md
│   ├── phase2-plan.md
│   ├── results.md                  # Benchmark results table
│   └── decisions/                  # ADRs
│
├── scripts/
│   ├── setup_env.sh
│   ├── download_ffpp_subset.sh
│   └── smoke_test_inference.sh
│
└── tests/
    └── ml/                         # Cross-cutting integration tests
```

---

## 5. Getting Started

### Prerequisites

- Python 3.11 (use `pyenv` or `.python-version`)
- `ffmpeg` available on `PATH` (required by OpenCV for video decoding)
- A CUDA-capable GPU is recommended for training; inference runs on CPU

### Installation

```bash
# 1. Clone and enter the project
git clone <repo-url>
cd deepforensics

# 2. Create and activate the virtual environment
scripts/setup_env.sh
source .venv/bin/activate        # Linux / macOS
# .venv\Scripts\activate         # Windows

# 3. Copy environment config
cp .env.example .env
# Fill in WANDB_API_KEY and optionally DEEPFORENSICS_DATA_ROOT
```

### Environment Variables

| Variable | Default | Description |
|---|---|---|
| `DEEPFORENSICS_DATA_ROOT` | `~/datasets` | Root directory for dataset storage |
| `DEEPFORENSICS_CACHE_ROOT` | `$DATA_ROOT/_cache` | Preprocessed tensor cache location |
| `WANDB_API_KEY` | — | Weights & Biases API key (free tier) |

### Download the Dev Subset

```bash
# Download a 100-video FF++ subset for development
scripts/download_ffpp_subset.sh

# Verify checksums
python -m ml.data.download --dataset ffpp --verify
```

### Run a Smoke Test

```bash
# Requires a trained checkpoint at ml/checkpoints/
scripts/smoke_test_inference.sh
```

---

## 6. ML Pipeline — Phase 1

### 6.1 Datasets

Three datasets are used at different stages of training and evaluation:

| Dataset | Videos | Role |
|---|---|---|
| **FaceForensics++ c23** | 1,000 real + 4,000 fake | Primary train / val / test (60/20/20 video-level split) |
| **Celeb-DF v2** | 590 real + 5,639 fake | Cross-dataset test only — no training samples |
| **DFDC preview** | 5,000 clips | Stress test under compression artefacts |

**Split discipline:** All splits are at the **video level** — never frame level. Frame-level splits leak near-identical frames from the same video across train and test, inflating reported accuracy by 10–20 points. The manifests in `ml/data/manifests/` are version-controlled and must not be altered once training has begun. Any change requires archiving the old manifest with a date suffix and noting it in `docs/results.md`.

**Checksum verification:**

```bash
python -m ml.data.download --dataset all --verify
```

Each dataset's key files are SHA-256 hashed against `ml/data/manifests/checksums.json`. A mismatch surfaces immediately before any training run.

---

### 6.2 Preprocessing

The preprocessing pipeline converts raw video into cached tensors that training reads directly. It runs once; subsequent training reads from cache.

#### Frame extraction — `ml/preprocessing/frame_extract.py`

```python
from ml.preprocessing.frame_extract import extract_frames

for frame in extract_frames("video.mp4", sample_fps=5.0):
    # frame.index, frame.timestamp, frame.bgr
    ...
```

- Decodes at native frame rate; samples every `round(native_fps / sample_fps)` frames
- Yields `Frame(index, timestamp, bgr)` dataclass objects
- Falls back gracefully on decode errors (skips the frame, logs a warning)
- Default: **5 fps** for inference, **1 fps** for fast training

#### Landmark extraction — `ml/preprocessing/landmarks.py`

Converts MediaPipe's 468-point face mesh into biometrically meaningful ratios:

```python
from ml.preprocessing.landmarks import compute_ear_mar_series

ear_series, mar_series = compute_ear_mar_series(landmarks_sequence)
```

- **EAR (Eye Aspect Ratio):** ratio of vertical to horizontal eye span — used for blink detection
- **MAR (Mouth Aspect Ratio):** analogous ratio for the outer lip — used for lip-sync analysis
- Missing frames (failed detection) are linearly interpolated from neighbours
- Indices are mapped from MediaPipe's 468-mesh to dlib's 68-point convention internally

#### Face detection

- **Primary:** MediaPipe Face Mesh (fast, CPU-friendly)
- **Fallback:** RetinaFace (handles occluded or low-quality frames where MediaPipe fails)
- If detection fails on more than 30% of frames, the fallback is triggered automatically
- Each detected face is cropped to a 1.3× expanded bounding box, resized to **224 × 224**, ImageNet-normalised

#### Audio features

- Mono 16 kHz waveform extracted with `soundfile`
- 13-coefficient MFCCs: 25 ms windows, 10 ms hop
- If no audio track is present, the MFCC tensor is zeroed and the modality is marked missing in `AnalysisResult`

#### Cache layout

```
$DEEPFORENSICS_CACHE_ROOT/
└── ffpp/
    └── <video_id>/
        └── <preprocessing_version>/
            ├── frames.pt       # (T, 3, 224, 224) float32
            ├── mfcc.pt         # (13, M) float32
            ├── landmarks.pt    # (T, 468, 2) float32
            ├── ear.pt          # (T,) float32
            └── mar.pt          # (T,) float32
```

Cache is invalidated by a `preprocessing.VERSION` constant in `ml/preprocessing/__init__.py`. Bumping it forces a full re-cache on next run.

---

### 6.3 Visual Feature Extractor

**Backbone:** EfficientNet-B4 from `timm`, pretrained on ImageNet-21k.

The classifier head is replaced with a 2-class head (REAL / FAKE). Fine-tuning is two-stage:

| Stage | What unfreezes | Epochs | LR |
|---|---|---|---|
| 1 | Last block + classifier | 2 | `3e-4` |
| 2 | Entire network | 10 | `1e-5` |

The two-stage approach prevents catastrophic forgetting of ImageNet features early in training. Jumping to a low learning rate from the start under-utilises the final block; unfreezing everything immediately with a high learning rate destroys the pretrained features.

**Optimiser:** AdamW, cosine schedule, weight decay `1e-4`. Mixed precision via `torch.amp.autocast`. Batch size 16 on a 16 GB GPU; gradient accumulation if memory is tight.

**Outputs:**
- `logits` — (B, 2) — visual-only prediction
- `embedding` — (B, 1792) — penultimate layer, passed to the fusion head

---

### 6.4 Semantic Modules

All three semantic modules are **deterministic** — no learnable parameters. They output fixed 4-dimensional feature vectors used as fusion inputs alongside the visual embedding.

#### Lip-Sync Analysis — `ml/models/semantic/lipsync.py`

Measures whether audio energy and mouth movement are correlated over time.

```python
from ml.models.semantic.lipsync import compute_lipsync_features

features, modality_missing = compute_lipsync_features(
    mar_series,     # (T,) mouth aspect ratio per frame
    mfcc_energy,    # (M,) MFCC L2 energy per MFCC frame
    fps=5.0,
)
# features: [r_mean, r_min, r_std, frac_below_threshold]
```

**How it works:**

1. MFCC energy is resampled to match the MAR frame rate via linear interpolation
2. Sliding 1-second windows compute the **Pearson correlation `r`** between MAR and MFCC energy
3. `r_mean < 0.4` is the heuristic threshold for a strong mismatch signal
4. If no audio is present, the function returns zeros and sets `modality_missing = True`

**Output vector:** `[r_mean, r_min, r_std, fraction_of_windows_below_0.4]`

---

#### Eye-Blink Analysis — `ml/models/semantic/blink.py`

Detects unnatural blink patterns — too infrequent, too frequent, or too uniform — which are characteristic of GAN-generated faces.

```python
from ml.models.semantic.blink import compute_blink_features

features, modality_missing = compute_blink_features(ear_series, fps=5.0)
# features: [blink_rate_per_min, mean_duration_frames, ibi_std, anomaly_z_score]
```

**How it works:**

1. A blink event is detected when EAR drops below **0.21** for 2–7 consecutive frames at 30 fps (scaled to `sample_fps`)
2. Inter-blink interval (IBI) standard deviation captures regularity — deepfakes often blink at unnaturally uniform intervals
3. **Anomaly z-score** = `(blink_rate - 16) / 4` where the natural range is 12–20 blinks/minute

**Output vector:** `[blink_rate, mean_duration, ibi_std, anomaly_z_score]`

---

#### Lighting Consistency — `ml/models/semantic/lighting.py`

Checks whether lighting direction is physically consistent across face regions. Face composites often have mismatched shading because the fake face was rendered under different illumination than the background.

```python
from ml.models.semantic.lighting import compute_lighting_features

features, modality_missing = compute_lighting_features(frames_rgb, landmarks_sequence)
# features: [inter_region_var, max_pairwise_angle, mean_intensity_residual, shadow_softness]
```

**How it works:**

1. Four face regions are defined by landmark indices: forehead, left cheek, right cheek, chin
2. Per-region dominant light direction is estimated via PCA on pixel intensities (Lambertian proxy)
3. Directions are averaged across frames, then inter-region variance and max pairwise angle are computed
4. **High inter-region variance** → inconsistent lighting → likely composite

**Output vector:** `[inter_region_var, max_pairwise_angle, mean_intensity_residual, shadow_softness]`

---

### 6.5 Multi-Modal Fusion

`ml/models/fusion.py` combines the visual embedding and all three semantic features into a final verdict.

#### Architecture

```
Input(1804)  =  visual(1792) + lipsync(4) + blink(4) + lighting(4)

→ Linear(1804 → 512) + GELU + Dropout(0.3)
→ Linear(512 → 128)  + GELU + Dropout(0.2)
→ Linear(128 → 2)    [REAL / FAKE logits]
```

The visual backbone is **frozen during fusion training** to prevent it from forgetting the patterns learned during visual pre-training.

#### Calibration Loss

```python
class FusionLoss(nn.Module):
    def __init__(self, label_smoothing: float = 0.05):
        self.ce = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
```

Label smoothing of 0.05 acts as a calibration regulariser — it penalises overconfident predictions, ensuring that a 90% confidence score genuinely corresponds to ~90% empirical accuracy.

#### Loading a trained checkpoint

```python
from ml.models.fusion import load_fusion_model

model = load_fusion_model("ml/checkpoints/fusion_head.pt", device="cuda")
```

Checkpoints store: model weights, epoch, metric value, git SHA, config hash, and W&B run URL.

---

### 6.6 Grad-CAM Explanation

`ml/explain/gradcam.py` generates per-frame saliency heatmaps using Gradient-weighted Class Activation Mapping.

**Target layer:** Last convolutional block of EfficientNet-B4 — the deepest layer that still has spatial resolution worth overlaying.

**For videos:**
- Heatmaps are generated for every frame individually
- The **top-5 most confident-fake frames** are selected (not all frames — averaging over uncertain frames dilutes the signal)
- Heatmaps are averaged across those 5 frames and overlaid at 40% alpha on the face crop

**Region mapping:**
The peak activation coordinate is mapped to a coarse facial region dictionary (eyes, nose, mouth, cheeks, boundary) to generate the `Explanation.top_regions` field in the result.

---

### 6.7 Inference API

`ml/inference.py` is the **only public surface** of the ML layer. The FastAPI backend and all evaluation scripts call this and nothing else.

```python
from ml.schemas import AnalysisResult

def analyse(
    media_path: str,
    *,
    sample_fps: int = 5,
    return_heatmap: bool = True,
    visual_checkpoint: str | None = None,
    fusion_checkpoint: str | None = None,
) -> AnalysisResult: ...
```

**What `analyse()` does internally:**

1. Detect whether input is an image or video
2. Extract frames at `sample_fps`
3. Detect faces, crop and normalise to 224×224
4. Extract landmarks → compute EAR/MAR series
5. Extract audio → compute MFCC energy
6. Run EfficientNet-B4 → get embedding (1792-d) + visual logit
7. Compute lip-sync, blink, lighting feature vectors
8. Run fusion head on `[embedding ∥ semantic_features]` → FAKE probability
9. Generate Grad-CAM heatmap (if `return_heatmap=True`)
10. Build and return `AnalysisResult`

**Performance target:** < 30 seconds per video on a development machine (CPU inference).

---

### 6.8 Training

#### Visual backbone

```bash
python -m ml.training.train_visual \
  --config config/visual_ffpp.yaml
```

- Logs every epoch to Weights & Biases
- Saves checkpoints to `ml/checkpoints/visual_b4_<timestamp>.pt`
- Best checkpoint is tracked by `val_auc`; last 3 are kept on disk
- Early stopping: patience = 5 epochs

#### Fusion head

```bash
python -m ml.training.train_fusion \
  --config config/fusion_ffpp.yaml
```

Uses cached visual features from the trained backbone — no re-extraction needed.

#### Reproducibility

Every training config sets `seed: 42` and enables `torch.use_deterministic_algorithms(True)`. Every checkpoint records the git SHA and config hash so any run can be reproduced exactly by checking out the same commit and running the same config.

#### Callbacks — `ml/training/callbacks.py`

```python
class Checkpointer:
    """Saves best checkpoint by val_auc; prunes old ones to keep last N."""

class EarlyStopping:
    """Stops training when metric stops improving for `patience` epochs."""
```

---

### 6.9 Evaluation

#### Visual-only baseline

```bash
python -m ml.evaluation.eval_visual \
  ml/checkpoints/visual_b4_best.pt \
  --split val \
  --append-results
```

Evaluates EfficientNet-B4 alone (no semantic features, no fusion) on the FF++ val or test split. Results are appended to `docs/results.md`.

#### Full pipeline evaluation (cross-dataset)

```bash
python -m ml.evaluation.eval_full_pipeline \
  --visual-checkpoint ml/checkpoints/visual_b4_best.pt \
  --fusion-checkpoint ml/checkpoints/fusion_head_best.pt \
  --append-results
```

Runs `ml.inference.analyse()` on every video in all three test manifests:

| Metric | Description |
|---|---|
| Accuracy | Fraction correctly classified |
| AUC | Area under the ROC curve |
| F1 | Harmonic mean of precision and recall |
| ECE | Expected Calibration Error — confidence alignment |

**Robustness variants:** The evaluator also tests JPEG-compressed (Q=30) and 112 px downscaled frames to simulate social-media re-encoding.

**Phase 1 exit targets:**

| Criterion | Target |
|---|---|
| FF++ val accuracy (visual-only) | ≥ 94% |
| Cross-dataset AUC (Celeb-DF) | reported (≥ 70% target) |
| Fusion vs visual-only improvement | ≥ +2 pt on cross-dataset |
| ECE | < 0.10 |
| Grad-CAM spot-check (20 samples) | logged in docs/results.md |
| Reproducibility | metrics within 0.5 pt on full re-run |

---

## 7. Web Application — Phase 2

The web layer wraps the ML pipeline in a production-ready service with a polished UI.

### 7.1 Backend (FastAPI)

#### API Contract

| Endpoint | Method | Description |
|---|---|---|
| `/api/upload` | POST | Multipart upload — returns `{ job_id }` |
| `/api/analyze` | POST | Start analysis with `{ job_id, options }` |
| `/api/status/{job_id}` | GET | `{ stage, progress_pct, eta_seconds }` |
| `/api/result/{job_id}` | GET | Full `AnalysisResult` JSON |
| `/api/heatmap/{job_id}` | GET | Grad-CAM overlay PNG |
| `/api/report/{job_id}` | GET | Downloadable PDF report |
| `/api/health` | GET | `{ status, model_versions, gpu }` |

#### Job Lifecycle

```
queued → preprocessing → visual → semantic → fusion → explanation → ready
                                                                   ↘ failed (with error)
```

The frontend polls `/api/status/{job_id}` every 1 second, with exponential backoff after 30 seconds.

#### Design decisions

- **Single worker, single GPU slot.** Concurrency = 1 job at a time; extras queue. For a demo, this is correct — there is no reason to introduce Celery complexity.
- **In-memory job store** guarded by `asyncio.Lock`. A restart clears jobs, which is acceptable for a demo deployment.
- **File cleanup:** `tmp/deepforensics/jobs/<job_id>/` is removed after 1 hour.
- **Limits:** max upload 200 MB, max video 60 seconds — enforced at the route layer before any processing begins.
- **CORS:** locked to the deployed frontend origin in production.

---

### 7.2 Frontend (React + Vite)

Six pages, zero marketing fluff:

| Page | Purpose |
|---|---|
| **Home** | Short hero, "Try it" CTA, methodology summary |
| **Analyze** | Drag-and-drop upload, live progress (stage + percent + ETA), preview |
| **Result** | Verdict panel, semantic score breakdown, evidence highlights, heatmap viewer, report download |
| **Metrics** | Model benchmarks rendered from `docs/results.md` |
| **About** | Team, methodology, academic references |
| **Help** | Capabilities, limitations, ethics statement |

**State management:** Zustand (`useJobStore`, `useResultStore`)

**TypeScript types** are generated automatically from `ml/schemas.py` via `pydantic-to-typescript` at build time. The TS types are never hand-written — they are always regenerated from the Python source of truth.

---

### 7.3 PDF Report

Generated server-side with ReportLab. No HTML-to-PDF — that approach is fragile and produces inconsistent layout across environments.

**Sections:**

1. Verdict + confidence (large, prominent)
2. Job metadata: timestamp, model checkpoint hashes, input file SHA-256
3. Semantic scores table (lip-sync, blink, lighting, visual-only)
4. Grad-CAM heatmap (most confident-fake frame)
5. Natural-language explanation (`AnalysisResult.explanation.natural_language`)
6. Methodology footnote + model version string

Filename: `deepforensics_report_<job_id>.pdf`

---

### 7.4 Deployment

| Component | Platform | Notes |
|---|---|---|
| Backend | Hugging Face Spaces (Docker, GPU tier) | Preferred — GPU available, Docker-based |
| Backend (fallback) | Fly.io | CPU-only, free tier |
| Frontend | Vercel | Set `VITE_API_BASE_URL` at build time |

The public URL goes into `README.md` once deployed.

---

## 8. Design System

DeepForensics is a **lab instrument**, not a marketing site. The reference aesthetic is Linear, Stripe Docs, Vercel dashboards — precise, dense, calm.

### Typography

| Role | Font | Weight |
|---|---|---|
| Display headings (h1, h2) | **Syne** | 500 / 700 |
| Body and UI text | **DM Sans** | 400 / 500 |
| Numbers, hashes, code | **JetBrains Mono** | 400 |

Fonts are loaded via `fontsource-*` npm packages — not Google Fonts CDN. This ensures privacy and deterministic builds.

### Color Tokens

All tokens are defined in `web/frontend/src/index.css` as HSL variables. Components reference tokens only — never raw hex, never Tailwind palette classes.

| Token | HSL | Role |
|---|---|---|
| `--background` | `228 30% 6%` | App canvas (deep navy) |
| `--card` | `225 24% 10%` | Surface |
| `--bg-elevated` | `228 22% 13%` | Modal / popover |
| `--foreground` | `228 50% 95%` | Primary text |
| `--muted-foreground` | `230 12% 60%` | Secondary text |
| `--primary` / `--accent-cyan` | `197 100% 50%` | Signal cyan — actions, links, focus |
| `--secondary` / `--accent-blue` | `222 100% 62%` | Secondary actions |
| `--risk-red` | `351 100% 62%` | FAKE verdicts only |
| `--risk-amber` | `43 100% 50%` | Caution band (50–70% confidence) |
| `--safe-green` | `160 100% 42%` | REAL verdicts only |
| `--border` | `228 16% 18%` | Hairlines, dividers |

**Hard rules:**
- Never write `text-white`, `bg-[#0af]`, `text-red-500`, or any raw color literal in a component
- Verdict colors (`--risk-red`, `--safe-green`) are reserved — do not reuse them for unrelated UI states
- Always HSL in token definitions; reference as `hsl(var(--token))` or via the Tailwind class

### Motion

- Verdict reveal: entrance animation, `--ease-out-expo`, 600 ms max
- Metric count-up: scroll-triggered, `CountUp` component in `src/components/common/`
- Nothing else animates. No looping animations. No shimmer.

### Anti-Patterns

| Forbidden | Instead |
|---|---|
| Purple-to-pink hero gradients | Solid `--background` with single subtle radial cyan glow |
| Glow / neon card outlines | 1 px `border-border` hairline |
| "AI-powered" / "revolutionary" copy | Plain technical claims |
| AI-flavor emojis (🤖✨🚀) in UI | None |
| Generic stock photos | Heatmap thumbnails, code snippets, benchmark plots |
| `shadow-2xl` everywhere | `--shadow-card` and `--shadow-elevated` tokens only |
| Toast for every action | Toast for failures and PDF/heatmap ready events only |

---

## 9. Schemas & Data Contracts

`ml/schemas.py` is the **canonical schema file**. The FastAPI backend imports it directly. TypeScript types are generated from it. Nothing is duplicated.

```python
class ScoreDetail(BaseModel):
    score: float           # normalised 0–1 signal strength
    raw: list[float]       # raw 4-d feature vector from the module
    modality_missing: bool = False

class SemanticScores(BaseModel):
    visual:   ScoreDetail
    lip_sync: ScoreDetail
    blink:    ScoreDetail
    lighting: ScoreDetail

class Explanation(BaseModel):
    top_regions:      list[str]   # face regions with highest Grad-CAM activation
    natural_language: str         # one-sentence plain-English verdict summary

class AnalysisResult(BaseModel):
    job_id:           str
    label:            Literal["REAL", "FAKE"]
    confidence:       float                    # softmax of fusion head
    scores:           SemanticScores
    explanation:      Explanation
    heatmap_png_path: str | None = None
    model_versions:   dict[str, str]           # checkpoint name → SHA-256 prefix
    elapsed_ms:       int
```

When the schema changes, regenerate TypeScript types:

```bash
cd web/frontend
pnpm run generate-types
```

---

## 10. Testing

Tests are co-located with the code they test. Cross-cutting integration tests live in `tests/ml/`.

### Unit tests

Each semantic module ships with a focused test suite:

**`ml/models/semantic/test_lipsync.py`**
- `test_returns_4d_float_array` — shape and dtype check on realistic inputs
- `test_missing_audio` — zeros returned, `modality_missing = True`
- `test_frac_below_threshold` — anti-correlated signal produces high `frac_below`

**`ml/models/semantic/test_lighting.py`**
- `test_returns_4d_float_array` — shape and dtype on synthetic frames + landmarks
- `test_all_none_landmarks_returns_missing` — graceful missing-data handling

**`ml/models/test_fusion.py`**
- `test_forward_shape` — logits are (B, 2) for batch of 4
- `test_loss_finite` — calibration loss is finite on random inputs

**`tests/ml/test_schemas.py`**
- Pydantic field validation and serialisation round-trips

### Running tests

```bash
# All tests
pytest

# Just semantic modules
pytest ml/models/semantic/

# With coverage
pytest --cov=ml --cov-report=term-missing
```

### Smoke test

```bash
scripts/smoke_test_inference.sh
```

Calls `ml.inference.analyse()` on one cached sample and asserts that the result is a valid `AnalysisResult` with all required fields populated.

---

## 11. Coding Conventions

### Python

- **Full type hints** on all public functions in `ml/inference.py`, `ml/schemas.py`, and `web/backend/`. Strict mypy on these modules.
- Format with `black`. Lint with `ruff`. CI enforces both.
- No `print()` in library code — use `logging`. Logger is configured once in `ml/__init__.py`.
- No global mutable state. Pass Hydra configs down explicitly.
- Tests live next to the code they test: `ml/models/fusion.py` ↔ `ml/models/test_fusion.py`

### TypeScript / React

- No `any` unless interfacing with an untyped third-party value — narrow immediately.
- Path alias `@/` maps to `src/`. No `../../..` chains.
- Pages are default-exported. Primitives are named-exported.
- Hooks: `useThing.ts`, named export.
- Forms: `react-hook-form` + `zod`.
- Every interactive element has a label or `aria-*`. Focus rings are never removed.

### Commit messages

```
<type>(<scope>): <imperative summary>

feat(ml): add fusion head with calibration loss
fix(preprocessing): handle audio-less video in MFCC extractor
train(visual): EfficientNet-B4 baseline — FF++ val AUC 0.97
docs(results): append cross-dataset eval table
```

Types: `feat`, `fix`, `refactor`, `docs`, `test`, `chore`, `train`

---

## 12. Benchmark Results

Results are appended to `docs/results.md` as training runs complete, with W&B run links and config hashes for full reproducibility.

| Criterion | Target | Status |
|---|---|---|
| FF++ val accuracy (visual-only) | ≥ 94% | complete |
| Celeb-DF AUC (visual-only) | reported | complete |
| Fusion ≥ +2 pt cross-dataset vs visual-only | ≥ +2 pt | complete |
| ECE calibration | < 0.10 | complete |
| Grad-CAM spot-check (20 samples) | logged | complete |
| Reproducibility within 0.5 pt | verified | complete |

Full metric tables, W&B run URLs, and config hashes are in [docs/results.md](docs/results.md).

---

## 13. Glossary

| Term | Definition |
|---|---|
| **EAR** | Eye Aspect Ratio — ratio of vertical to horizontal eye landmark distances; used for blink detection |
| **MAR** | Mouth Aspect Ratio — analogous ratio for the outer lip; used for lip-sync analysis |
| **MFCC** | Mel-Frequency Cepstral Coefficients — standard audio feature for speech characterisation |
| **Grad-CAM** | Gradient-weighted Class Activation Mapping — highlights CNN regions that drove a prediction |
| **Fusion score** | Final softmax output of the multi-modal MLP classifier |
| **ECE** | Expected Calibration Error — how well predicted confidences match observed accuracies |
| **Job** | One upload + analysis run, identified by a UUID `job_id` |
| **c23 / c40** | FaceForensics++ compression levels: c23 is light compression, c40 is heavy |
| **EfficientNet-B4** | CNN architecture from the EfficientNet family, ~19M parameters; strong accuracy/efficiency trade-off |
| **Lambertian** | Reflectance model assuming diffuse surfaces; used here as a proxy for estimating light direction from pixel intensity |
| **IBI** | Inter-Blink Interval — time between successive blink events; unnaturally regular IBI is a deepfake signal |

---

*Documentation generated: 2026-05-25. Maintained alongside the codebase — update when schemas, training configs, or API contracts change.*
