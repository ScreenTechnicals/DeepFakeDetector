# CLAUDE.md — DeepForensics

> **Purpose.** This file is the single source of truth for the DeepForensics rebuild. It is read at the start of every Claude Code session. Read §0 and §1 before writing a single line of code.

---

## 0. Read this first

**DeepForensics** is an explainable deepfake forensics system. It accepts an image or video and returns a REAL/FAKE verdict with calibrated confidence, semantic scores (lip-sync, eye-blink, lighting), Grad-CAM heatmaps, and a downloadable PDF report.

**This rebuild differs from the previous Lovable version in one critical way:** the previous version *simulated* the ML pipeline by prompting Gemini. This rebuild **trains and serves the actual PyTorch models**. There is no LLM in the inference path.

The project is **phased**. Do not skip phases.

---

## 1. Phases — non-negotiable

| Phase | Focus | Output |
| --- | --- | --- |
| **1. ML (weeks 1–10)** | Datasets, preprocessing, training, evaluation | `ml.inference.analyse(path) -> AnalysisResult` works end-to-end on real trained checkpoints |
| **2. Web (weeks 11–15)** | FastAPI service + React frontend + PDF report + deploy | Public URL, polished UX, professional design |

**Rules of engagement:**

1. **During Phase 1: no frontend code, no design tweaks, no deployment work.** If the user asks for any of these during Phase 1, surface the phase conflict in one sentence and ask whether to defer.
2. **During Phase 2: no model retraining, no architecture changes.** If a model change becomes necessary, treat it as scope-changing and confirm before proceeding.
3. **Phase 1 exits only when every box in §12 is checked.** Self-graded "good enough" is not an exit.
4. **Frontend assets, design tokens, and architectural decisions for Phase 2 may be *drafted* in `docs/phase2-plan.md` during Phase 1, but no code in `web/` is written until Phase 1 exits.**

---

## 2. Architecture

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
                                             │
                              ┌──────────────────────────────┐
                              │  ml/checkpoints/             │
                              │  efficientnet_b4_ff++.pt     │
                              │  fusion_head.pt              │
                              │  ...                          │
                              └──────────────────────────────┘
```

**Mono-repo, single Python virtualenv for ML + backend.** The FastAPI service `import`s `ml.inference` directly — no HTTP between the two. The frontend is a separate Node project.

---

## 3. Tech Stack

### Phase 1 (ML)
| Layer | Tool |
| --- | --- |
| Language | Python 3.11 |
| DL framework | PyTorch 2.3 + torchvision |
| Model zoo | `timm` (EfficientNet-B4 with ImageNet-pretrained weights) |
| Face detection / landmarks | MediaPipe Face Mesh (primary), RetinaFace (fallback for hard cases) |
| Video / image I/O | OpenCV 4, decord (faster video decoding when available) |
| Audio | librosa, soundfile |
| Explainability | `pytorch-grad-cam` |
| Scientific | NumPy, SciPy, scikit-learn |
| Experiment tracking | Weights & Biases (`wandb`) — free tier is sufficient |
| Config | Hydra (or plain YAML + `pydantic` if Hydra feels heavy) |
| Test | pytest |
| Lint / format | ruff, black, mypy (strict on `ml/` core, lax on scripts) |

### Phase 2 (Web)
| Layer | Tool |
| --- | --- |
| Backend | FastAPI + Uvicorn |
| Background jobs | FastAPI `BackgroundTasks` (sufficient for single-instance demo); upgrade to Celery + Redis only if multi-worker is needed |
| Build | Vite 5 + TypeScript 5 |
| UI | React 18, shadcn/ui (Radix), Tailwind CSS v3 |
| State | Zustand |
| Routing | React Router v6 |
| Motion | Framer Motion (used sparingly — see §7) |
| PDF | ReportLab (server-side) — NOT a JS lib; the report is generated on the backend |
| Test | Vitest + Playwright |

### Do NOT introduce
- Next.js / Remix / Nuxt — Vite is sufficient.
- Express / Node backend — FastAPI is the only backend.
- Lovable, Supabase, edge functions — this is a self-hosted project.
- LangChain / LlamaIndex — there is no LLM in the inference path.
- Generic AI image-style frameworks; the ML is hand-rolled PyTorch.

---

## 4. Repository Layout

```
deepforensics/
├── CLAUDE.md                       # this file
├── README.md                       # short public-facing readme
├── pyproject.toml                  # ML + backend Python deps (single venv)
├── .python-version                 # 3.11
├── .env.example                    # documented env vars
│
├── ml/
│   ├── data/
│   │   ├── download.py             # scripts to download / verify datasets
│   │   ├── manifests/              # CSV/JSON splits — version-controlled
│   │   │   ├── ffpp_train.csv
│   │   │   ├── ffpp_val.csv
│   │   │   ├── ffpp_test.csv
│   │   │   ├── celebdf_test.csv
│   │   │   └── dfdc_test.csv
│   │   └── README.md               # how to obtain datasets (NOT the data itself)
│   │
│   ├── datasets/
│   │   ├── ffpp.py                 # FaceForensics++ Dataset class
│   │   ├── celebdf.py
│   │   └── dfdc.py
│   │
│   ├── preprocessing/
│   │   ├── face_detect.py          # MediaPipe + RetinaFace fallback
│   │   ├── frame_extract.py        # video → frames @ sample_fps
│   │   ├── landmarks.py            # 68/468 landmarks + EAR/MAR helpers
│   │   └── audio.py                # MFCC extraction
│   │
│   ├── models/
│   │   ├── visual.py               # EfficientNet-B4 backbone wrapper
│   │   ├── fusion.py               # MLP fusion head
│   │   └── semantic/
│   │       ├── lipsync.py          # MFCC × MAR correlation
│   │       ├── blink.py            # EAR time-series anomaly
│   │       └── lighting.py         # Lambertian shading analysis
│   │
│   ├── explain/
│   │   └── gradcam.py              # Grad-CAM on last conv layer
│   │
│   ├── training/
│   │   ├── train_visual.py         # train/fine-tune EfficientNet-B4
│   │   ├── train_fusion.py         # train fusion head on cached features
│   │   ├── config/                 # Hydra configs (one YAML per experiment)
│   │   └── callbacks.py            # checkpointing, early stop, W&B logging
│   │
│   ├── evaluation/
│   │   ├── metrics.py              # acc, AUC, F1, ECE (calibration)
│   │   ├── eval_visual.py
│   │   ├── eval_full_pipeline.py
│   │   └── reports/                # generated eval reports go here (gitignored)
│   │
│   ├── inference.py                # the ONLY public ML API: analyse(path) -> AnalysisResult
│   ├── schemas.py                  # pydantic models — shared with backend
│   └── checkpoints/                # gitignored, but documented in README
│
├── web/
│   ├── backend/
│   │   ├── main.py                 # FastAPI app
│   │   ├── routes/
│   │   │   ├── analyze.py
│   │   │   ├── result.py
│   │   │   ├── heatmap.py
│   │   │   └── report.py
│   │   ├── jobs.py                 # in-memory job store + BackgroundTasks
│   │   ├── pdf_report.py           # ReportLab template
│   │   └── tests/
│   │
│   └── frontend/
│       ├── package.json
│       ├── vite.config.ts
│       ├── tailwind.config.ts
│       ├── tsconfig.json
│       ├── index.html
│       ├── src/
│       │   ├── main.tsx
│       │   ├── App.tsx
│       │   ├── index.css           # design tokens (HSL) — single source of truth
│       │   ├── routes/index.tsx
│       │   ├── components/
│       │   │   ├── common/         # CountUp, ScrollReveal, etc.
│       │   │   ├── layout/         # Navbar, Footer
│       │   │   └── ui/             # shadcn primitives — do not hand-edit
│       │   ├── features/
│       │   │   ├── analysis/       # upload + run
│       │   │   ├── results/        # verdict, scores, evidence
│       │   │   ├── heatmap/        # Grad-CAM viewer
│       │   │   ├── report/         # PDF preview/download
│       │   │   └── metrics/        # model benchmark page
│       │   ├── pages/              # Home, About, Help, NotFound
│       │   ├── services/api.ts     # ForensicsAPI client
│       │   ├── store/              # Zustand
│       │   ├── lib/utils.ts
│       │   └── hooks/
│       └── tests/                  # Vitest + Playwright
│
├── docs/
│   ├── phase1-plan.md              # detailed Phase 1 plan (drafted week 0)
│   ├── phase2-plan.md              # drafted late in Phase 1
│   ├── api-contract.md             # canonical API spec
│   ├── design-system.md            # expanded design notes
│   ├── results.md                  # benchmark results, written as Phase 1 progresses
│   └── decisions/                  # ADRs — one MD per non-trivial decision
│
├── scripts/
│   ├── setup_env.sh
│   ├── download_ffpp_subset.sh
│   └── smoke_test_inference.sh
│
└── tests/
    └── ml/                         # cross-cutting tests; module-level tests live with code
```

### Files Claude must never modify without explicit instruction
- Anything in `ml/checkpoints/` (it's data, not code)
- `ml/data/manifests/*.csv` once Phase 1 has begun (changing splits invalidates results)
- `pyproject.toml` lockfile-equivalent regions
- `web/frontend/src/components/ui/*` (shadcn primitives — extend via `cva` variants)

---

## 5. ML Pipeline — Phase 1 specifications

This section is the contract. Every module below must exist, be tested, and be wired into `ml.inference.analyse` before Phase 1 exits.

### 5.1 Datasets

| Dataset | Size | Role | Source |
| --- | --- | --- | --- |
| FaceForensics++ (c23) | 1,000 real + 4,000 fake videos | **Primary train/val/test** (60/20/20 video-level split) | https://github.com/ondyari/FaceForensics |
| Celeb-DF v2 | 590 real + 5,639 fake | Cross-dataset test (no training) | https://github.com/yuezunli/celeb-deepfakeforensics |
| DFDC preview | 5,000 clips | Stress test under compression | https://ai.facebook.com/datasets/dfdc/ |

**Acquisition workflow:**
1. Request access where required; document the date in `ml/data/README.md`.
2. Download to a local path (not committed). The default expected path is `~/datasets/<name>/` — override via `DEEPFORENSICS_DATA_ROOT` env var.
3. Run `python -m ml.data.download --verify` to compute checksums against `ml/data/manifests/checksums.json`.
4. **For development, start with a 100-video subset of FF++ c23.** Full-dataset training only after the pipeline is bug-free on the subset.

**Split discipline.**
- Splits are at the **video level**, never frame level — leakage across splits is the #1 cause of inflated deepfake numbers.
- Splits are committed as CSV manifests in `ml/data/manifests/`. Changing a manifest mid-Phase-1 invalidates every reported metric; if it must change, archive the old manifest with a date suffix and note it in `docs/results.md`.

### 5.2 Preprocessing

For each video:
1. Decode at the native frame rate; sample at `sample_fps` (default **5 fps**) for inference, **1 fps** for fast training pass.
2. Detect face on each sampled frame (MediaPipe Face Mesh, single-face mode). If detection fails on >30% of frames, fall back to RetinaFace.
3. Crop to a 1.3× expanded bounding box around the face, resize to **224 × 224**, ImageNet-normalise.
4. Extract 68-point landmarks (dlib-style; derive from MediaPipe's 468-mesh) for EAR/MAR.
5. If audio present: extract mono 16 kHz waveform, then 13-coef MFCCs with 25 ms windows, 10 ms hop.

Cache preprocessed tensors to disk under `~/datasets/_cache/<dataset>/<video_id>/{frames.pt, mfcc.pt, landmarks.pt}`. Cache is invalidated by a `preprocessing.version` constant — bump it when preprocessing changes.

### 5.3 Visual feature extractor

- Backbone: **EfficientNet-B4** from `timm`, ImageNet-pretrained.
- Replace the classifier head with a 2-class head (REAL / FAKE).
- **Fine-tune the last block + classifier first** (2 epochs), then unfreeze the full network (10 epochs). Use mixed precision (`torch.amp`).
- Output the 1792-d penultimate embedding for fusion, plus the 2-class logits for the visual-only baseline.

Optimiser: AdamW, lr `3e-4` (head) → `1e-5` (full network), cosine schedule, weight decay `1e-4`. Batch size 16 on a single 16 GB GPU; gradient accumulation if memory is tight.

### 5.4 Lip-sync analysis

For a video segment:
1. Compute **MAR (Mouth Aspect Ratio)** per sampled frame from landmarks.
2. Compute **MFCC energy envelope** time-aligned to the frame timestamps.
3. Compute the **Pearson correlation `r`** between the two series over sliding 1-second windows.
4. Report `r_mean`, `r_min`. Heuristic: `r_mean < 0.4` is a strong mismatch signal.
5. Emit a 4-d feature vector for fusion: `[r_mean, r_min, r_std, fraction_below_threshold]`.

If the video has no audio, emit zeros and mark the modality as missing in the result.

### 5.5 Eye-blink (EAR) analysis

1. Compute **EAR (Eye Aspect Ratio)** per frame for both eyes; take the minimum (the eye that's blinking).
2. A blink = EAR drops below 0.21 for 2–7 consecutive frames at 30 fps (scale to `sample_fps`).
3. Report blink rate (blinks/minute), mean duration, inter-blink-interval std.
4. Anomaly score = z-score against natural human range (12–20 blinks/min). Output a 4-d feature.

### 5.6 Lighting consistency

1. Estimate the dominant light direction per face region (forehead, left cheek, right cheek, chin) via Lambertian reflectance: solve `n · l = I` over the face mesh normals.
2. Report inter-region variance of the estimated direction vectors. High variance ⇒ inconsistent lighting ⇒ likely composite.
3. Emit a 4-d feature: `[inter_region_var, max_pairwise_angle, mean_intensity_residual, shadow_softness]`.

### 5.7 Multi-modal fusion

Inputs:
- Visual embedding: 1792-d (EfficientNet-B4 penultimate)
- Semantic features: 12-d (lip-sync 4 + blink 4 + lighting 4)
- (Optional) audio-only features: 20-d (mean/std of MFCCs)

Architecture (`ml/models/fusion.py`):
```
Input(1812) → Linear(512) + GELU + Dropout(0.3)
            → Linear(128) + GELU + Dropout(0.2)
            → Linear(2)   (REAL / FAKE logits)
```

Train with cross-entropy + a calibration term (label smoothing 0.05). Freeze the visual backbone during fusion training to avoid catastrophic forgetting of the visual baseline.

### 5.8 Grad-CAM explanation

- Target layer: last conv block of EfficientNet-B4.
- Generate per-frame heatmaps; for videos, **average heatmaps across the top-5 most confident-fake frames**, not all frames.
- Overlay at 40% alpha on the original face crop.
- Identify the dominant region by mapping peak coordinates to a coarse facial region map (eyes / nose / mouth / cheeks / boundary).

### 5.9 Inference API — the only public surface of `ml/`

```python
# ml/inference.py
from ml.schemas import AnalysisResult

def analyse(
    media_path: str,
    *,
    sample_fps: int = 5,
    return_heatmap: bool = True,
) -> AnalysisResult: ...
```

`AnalysisResult` is a pydantic model in `ml/schemas.py`. It is **the canonical schema**. The FastAPI service must not invent its own — it imports this one. When the schema changes, the frontend types are regenerated.

Required fields (sketch — finalise in code before training):
```python
class AnalysisResult(BaseModel):
    job_id: str
    label: Literal["REAL", "FAKE"]
    confidence: float                    # softmax of fusion head
    scores: SemanticScores               # lip_sync, blink, lighting, visual
    explanation: Explanation             # top regions, natural-language summary
    heatmap_png_path: str | None
    model_versions: dict[str, str]       # checkpoint hashes
    elapsed_ms: int
```

### 5.10 Training procedure

1. Cache preprocessed features for the full FF++ c23 set (one-time, hours).
2. `python -m ml.training.train_visual --config config/visual_ffpp.yaml`
   - Logs to W&B; saves best checkpoint to `ml/checkpoints/visual_b4_<timestamp>.pt`.
3. Run `python -m ml.evaluation.eval_visual` to confirm baseline metrics.
4. `python -m ml.training.train_fusion --config config/fusion_ffpp.yaml` — uses cached visual features + computed semantic features.
5. Run full-pipeline evaluation: `python -m ml.evaluation.eval_full_pipeline`.
6. Cross-dataset eval on Celeb-DF and DFDC — report in `docs/results.md`.

**Reproducibility rules:**
- Every config sets `seed: 42` and `torch.use_deterministic_algorithms(True)` where possible.
- Every checkpoint records: git SHA, config hash, dataset manifest hash, W&B run URL.
- No "I changed a thing and reran" tweaks without bumping the config version.

### 5.11 Evaluation protocol

- **Within-dataset (FF++):** accuracy, AUC, F1, confusion matrix.
- **Cross-dataset (Celeb-DF, DFDC):** same metrics — expect a 5–15 pt drop; that's the literature norm, not a bug.
- **Calibration:** Expected Calibration Error (ECE) — confidence should mean what it says.
- **Robustness:** evaluate on JPEG-compressed (Q=30) and downscaled (112 px) variants.
- **Explanation sanity:** spot-check 20 Grad-CAM outputs; the heatmap should peak on visually plausible regions, not background.

Results table goes in `docs/results.md`, updated as runs complete.

---

## 6. Web Application — Phase 2 specifications

Begin only when §12 is fully checked.

### 6.1 API contract

| Path | Method | Purpose |
| --- | --- | --- |
| `/api/upload` | POST | multipart upload, returns `{ job_id }` |
| `/api/analyze` | POST | start analysis with `{ job_id, options }` |
| `/api/status/{job_id}` | GET | `{ stage, progress_pct, eta_seconds }` |
| `/api/result/{job_id}` | GET | full `AnalysisResult` JSON |
| `/api/heatmap/{job_id}` | GET | PNG (Grad-CAM overlay) |
| `/api/report/{job_id}` | GET | PDF report |
| `/api/health` | GET | `{ status, model_versions, gpu }` |

**Job lifecycle:** `queued → preprocessing → visual → semantic → fusion → explanation → ready` (or `failed` with `error`). The frontend polls `/status` every 1 s with exponential backoff after 30 s.

Schemas live in `ml/schemas.py` and are mirrored to TypeScript via `pydantic-to-typescript` at build time. **Do not hand-write the TS types** — regenerate them.

### 6.2 Backend (FastAPI)

- Single-instance, single-worker by default. Concurrency = 1 GPU job at a time; queued otherwise.
- Job store: in-memory dict guarded by an `asyncio.Lock`. Persistence is out of scope for Phase 2 v1 — a refresh loses jobs, and that's acceptable for a demo.
- File storage: `/tmp/deepforensics/jobs/<job_id>/` (uploads, heatmaps, reports). Cleaned up after 1 hour.
- CORS: locked to the deployed frontend origin.
- Hard limits: max upload 200 MB, max video duration 60 s, rejected at the route layer.
- No auth in v1; add it only if the project goes public.

### 6.3 Frontend (Vite + React)

Pages:
1. **Home** — short hero, "Try it" CTA, one-paragraph explanation. No carousel, no testimonial section.
2. **Analyze** — drag-and-drop upload, live progress (stage + percent + ETA), inline preview.
3. **Result** — verdict panel, semantic score breakdown, evidence highlights, heatmap viewer, report download.
4. **Metrics** — model benchmarks rendered from `docs/results.md` or a JSON snapshot.
5. **About** — team, methodology, references. Plain text, no marketing fluff.
6. **Help** — what the tool can and cannot do, limitations, ethics statement.

Routing in `src/routes/index.tsx`. State in Zustand stores (`useJobStore`, `useResultStore`).

### 6.4 PDF report

Generated server-side with ReportLab. Sections:
1. Verdict + confidence (large, prominent)
2. Job metadata (timestamp, model versions, file hash)
3. Semantic scores table
4. Grad-CAM heatmap (selected frame)
5. Natural-language explanation
6. Methodology footnote + version

Filename: `deepforensics_report_<job_id>.pdf`. No HTML-to-PDF — that's fragile.

### 6.5 Deployment

- **Backend:** Hugging Face Spaces (Docker, GPU tier) is the path of least resistance for a student project. Fly.io if a CPU-only deployment is acceptable for the demo.
- **Frontend:** Vercel or Netlify, free tier.
- Set `VITE_API_BASE_URL` at build time.
- Domain (optional): a `.tech` or `.dev` student domain is fine.

---

## 7. Design System

DeepForensics is a **lab instrument**, not a marketing site. Think Stripe Docs, Linear, Vercel dashboards — not generic SaaS templates.

### 7.1 Aesthetic principles

- **Tone:** investigative, precise, calm. Never breathless. Never "powered by AI."
- **Density:** information-rich but breathable. Generous gutters, tight rows.
- **Motion:** purposeful only — verdict reveals, scroll-triggered count-ups for headline metrics. Never decorative. No looping animations. No "shimmer."
- **Color:** restrained. Cyan is the only signal accent; verdict colors are reserved (see §7.3).
- **Typography:** display sans for headings, neutral sans for body, monospaced for data. Never decorative scripts. Never more than 3 font weights.

### 7.2 Typography

| Role | Font | Notes |
| --- | --- | --- |
| Display (h1, h2) | **Syne** 500/700 | Geometric, slightly technical |
| Body (p, ui) | **DM Sans** 400/500 | Neutral, screen-optimised |
| Mono (IDs, hashes, scores, code) | **JetBrains Mono** 400 | Use for any numeric readout |

Load via `fontsource-*` packages in the Vite project, not Google Fonts CDN — better privacy, deterministic builds.

### 7.3 Color tokens (HSL, in `web/frontend/src/index.css`)

| Token | HSL | Role |
| --- | --- | --- |
| `--background` | `228 30% 6%` | App canvas (deep navy, near-black) |
| `--card` | `225 24% 10%` | Surface |
| `--bg-elevated` | `228 22% 13%` | Modal / popover |
| `--foreground` | `228 50% 95%` | Primary text |
| `--muted-foreground` | `230 12% 60%` | Secondary text |
| `--primary` / `--accent-cyan` | `197 100% 50%` | Signal cyan — actions, focus, links |
| `--secondary` / `--accent-blue` | `222 100% 62%` | Secondary actions only |
| `--risk-red` | `351 100% 62%` | **FAKE verdicts only** |
| `--risk-amber` | `43 100% 50%` | Caution band (50–70% confidence) |
| `--safe-green` | `160 100% 42%` | **REAL verdicts only** |
| `--border` | `228 16% 18%` | Hairlines, dividers |

**Hard rules:**
1. **Never** write `text-white`, `bg-[#0af]`, `text-red-500`, or any raw hex/Tailwind palette color in a component. Always use the semantic token (`text-foreground`, `bg-card`, `text-destructive`).
2. **Always HSL** in tokens. Reference as `hsl(var(--token))` or via the Tailwind class that already wraps it.
3. Verdict colors are reserved — don't reuse `--risk-red` for a delete button.

### 7.4 Radii, shadows, motion

- Radii: `--radius-sm` (chips, 6 px), `--radius-md` (buttons/inputs, 8 px), `--radius-lg` (cards, 12 px), `--radius-xl` (hero panels, 16 px).
- Shadows: only `--shadow-card` and `--shadow-elevated`. Avoid Tailwind's default `shadow-*` scale.
- Easing: `--ease-out-expo` for entrances, `--ease-in-out-sine` for state changes.
- Durations: `fast 150 ms`, `normal 250 ms`, `slow 400 ms`, `entrance 600 ms`. No motion exceeds 600 ms.

### 7.5 Component conventions

- Build new visuals on shadcn primitives via `cva` variants — **never fork the primitive**.
- Reusable display logic (CountUp, ScrollReveal) lives in `src/components/common/`.
- Layout chrome (Navbar, Footer) lives in `src/components/layout/`.
- Feature-specific composites live next to their feature page under `src/features/<domain>/`.
- One file per component. PascalCase filenames for components, camelCase for hooks/utilities.

### 7.6 Anti-patterns — explicitly disallowed

| Don't | Do |
| --- | --- |
| Hero gradient swooshes (`bg-gradient-to-br from-purple-500 to-pink-500`) | Solid `--background` with a single subtle radial cyan glow if needed |
| Glow / neon outlines on cards | A 1 px hairline `border-border` |
| Animated everything on scroll | Reveal headings and metrics only |
| "AI-flavor" emojis (🤖✨🚀) in UI | None |
| Marketing buzzwords ("revolutionary," "AI-powered") | Plain technical claims |
| Generic stock photos | Custom visual elements: code snippets, heatmap thumbnails, plots |
| Tailwind default `shadow-2xl` everywhere | Tokens only |
| Toast for every action | Toast for failures and PDF/heatmap ready events; nothing else |

### 7.7 Reference

Look at these when in doubt:
- https://linear.app — density, restraint
- https://stripe.com/docs — typography, navigation
- https://vercel.com — dashboards
- https://www.factory.ai — modern dark forensic look

Do NOT look at: generic SaaS landing pages, Dribbble shots labelled "AI Dashboard 2024."

---

## 8. Coding Conventions

### Python

- **Type-hint everything** in `ml/inference.py`, `ml/schemas.py`, and `web/backend/`. Strict mypy on these modules.
- Scripts in `ml/training/` and `scripts/` may relax typing; use `from __future__ import annotations`.
- Format with `black`, lint with `ruff`. CI runs both.
- No `print` in library code — use `logging`. Configure once in `ml/__init__.py`.
- Avoid global state; use Hydra configs and pass them down.
- Tests next to code: `ml/models/fusion.py` ↔ `ml/models/test_fusion.py`. Cross-cutting tests in `tests/ml/`.

### TypeScript / React

- No `any` unless interfacing with an untyped third-party value — narrow immediately.
- Path alias `@/` → `src/`. No `../../..`.
- Components default-exported when they're a page, named-exported for primitives.
- Hooks: `useThing.ts`, named export.
- State: Zustand stores in `src/store/`, one per cohesive domain.
- Async: `async/await`; surface errors via `toast` from `@/hooks/use-toast`.
- Forms: `react-hook-form` + `zod`.
- Accessibility: every interactive element has a label / `aria-*`. Preserve focus rings.

### Commits

`<type>(<scope>): <imperative summary>` — e.g. `feat(ml): add fusion head with calibration loss`. Types: `feat`, `fix`, `refactor`, `docs`, `test`, `chore`, `train` (for training runs that produced a reportable checkpoint).

---

## 9. Workflow

### Starting a session
1. Pull latest.
2. Read this file's §1 (phase status) and the latest entry in `docs/results.md` or `docs/phase2-plan.md`.
3. Confirm the current phase before doing anything cross-phase.

### Making changes
1. **Plan only when needed.** Small, scoped → just implement. Large/ambiguous → write the plan in a 5-line comment or a `docs/decisions/<date>-<topic>.md` ADR first.
2. **Edit surgically.** Touch only what's needed.
3. **Test.** ML: at minimum, a smoke test that the new module runs on one cached sample. Web: Vitest unit + a Playwright happy-path.
4. **Log.** ML changes that affect a metric → re-run eval, append to `docs/results.md`.

### Communication style
- Under two sentences unless the user asked for depth.
- No third-person recaps of what was done.
- When uncertain, flag the assumption and proceed; don't ping-pong on clarifications.

---

## 10. Common pitfalls

| Pitfall | Correct approach |
| --- | --- |
| Frame-level train/test split | Video-level split — always |
| Training EfficientNet-B4 from scratch | Use `timm` ImageNet weights; fine-tune in two stages |
| Reporting FF++ numbers without cross-dataset eval | Always report FF++ + Celeb-DF side-by-side |
| Inflating accuracy by overfitting to c23 compression | Evaluate on c40 and resized variants too |
| Generating Grad-CAM from a random frame | Average across top-5 confident-fake frames |
| Writing the API contract on the frontend first | Schemas live in `ml/schemas.py`, TS is generated |
| Storing model checkpoints in git | Gitignored; document download/regenerate in README |
| Forking a shadcn primitive to restyle | Add a `cva` variant or wrap it |
| Using `framer-motion` on every element | Reserve for verdict reveals and metric count-ups |
| Hard-coding `#0af` or `text-white` | Use semantic tokens |
| Adding a Next.js or Node backend | FastAPI is the only backend |
| Starting Phase 2 work during Phase 1 | Defer; note in `docs/phase2-plan.md` |

---

## 11. Glossary

- **EAR** — Eye Aspect Ratio; ratio of vertical to horizontal eye landmark distances; used for blink detection.
- **MAR** — Mouth Aspect Ratio; analogous for lips.
- **MFCC** — Mel-Frequency Cepstral Coefficients; standard audio feature for speech.
- **Grad-CAM** — Gradient-weighted Class Activation Mapping; highlights regions that drove a CNN's prediction.
- **Fusion score** — final softmax output of the multi-modal classifier.
- **ECE** — Expected Calibration Error; how well predicted confidences match observed accuracies.
- **Job** — one upload + analysis run, identified by `job_id` (UUID).
- **c23 / c40** — FaceForensics++ compression levels (lighter / heavier).

---

## 12. Phase 1 exit criteria

Phase 1 is **not** complete until every box is checked **and** the line items are reflected in `docs/results.md`.

- [ ] FF++ c23 dataset downloaded, manifests committed, checksums verified.
- [ ] Preprocessing pipeline runs end-to-end on a 10-video sample without errors.
- [ ] Visual-only EfficientNet-B4 baseline: **≥ 94% accuracy** on FF++ val set.
- [ ] Cross-dataset eval on Celeb-DF v2: reported (target ≥ 70%; honest is more important than high).
- [ ] Lip-sync, blink, and lighting modules each produce a 4-d feature vector on one sample, with a unit test.
- [ ] Fusion head trained; achieves **≥ 2 pt** improvement over the visual-only baseline on cross-dataset eval.
- [ ] Grad-CAM heatmaps generated for 20 spot-checked samples; visual inspection logged.
- [ ] `ml.inference.analyse('sample.mp4')` returns a complete `AnalysisResult` in under 30 s on the dev machine.
- [ ] Reproducibility: deleting `ml/checkpoints/` and re-running the configs produces metrics within 0.5 pt of the original.
- [ ] `docs/results.md` contains a final table with all metrics + W&B run links.

---

## 13. Phase 2 exit criteria

- [ ] FastAPI service runs locally; `/api/health` returns model versions.
- [ ] Upload → analyse → result → heatmap → PDF flow works on Chrome, Firefox, Safari.
- [ ] Mobile-responsive (375 px and up).
- [ ] Lighthouse: Performance ≥ 85, Accessibility ≥ 95, Best Practices ≥ 95.
- [ ] All semantic tokens in use; zero raw color literals in the diff (grep check).
- [ ] Deployed to a public URL; the URL is in `README.md`.
- [ ] A 90-second demo video exists in `docs/`.

---

## 14. Practical notes for limited compute

This is a student project; you likely don't have a multi-GPU rig. Plan accordingly:

- **Start tiny.** Train the visual baseline on 100 videos first to debug the pipeline. Only scale to full FF++ once the training loop is bug-free.
- **Use mixed precision** (`torch.amp.autocast`) — roughly halves memory and time on modern GPUs.
- **Cache aggressively.** Preprocessing is expensive; do it once, save tensors, train against the cache.
- **Colab Pro is acceptable** for training runs, but the final inference must run on the deployment target (likely a CPU). Test inference on CPU before claiming Phase 1 done.
- **Don't train the visual backbone end-to-end if you don't have to.** Fine-tuning the last block + head is often within 1 pt of full fine-tuning at a fraction of the cost.
- **Save checkpoints every epoch.** Crashes happen.

---

*Last updated: 2026-05-23. Keep §12 and §13 in sync with reality. When Phase 1 closes, set the date here and append a one-paragraph postmortem in `docs/decisions/`.*
