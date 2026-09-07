# Xynth

**Synthetic X-ray Augmentation using DCGAN**

Xynth is a research-style GenAI lab project that trains a Deep Convolutional GAN (DCGAN) on minority-class chest X-rays, generates synthetic samples, builds an augmented training set, and compares CNN classification performance with vs. without synthetic augmentation.

---

## Overview

Medical imaging datasets are often class-imbalanced. Xynth focuses on the **COVID** class from the COVID-19 Radiography Database and asks:

> Does DCGAN-based synthetic augmentation improve multi-class X-ray classification?

Pipeline:

1. Load & preprocess radiographs (grayscale, 64×64, [-1, 1])
2. Train a DCGAN on the target minority class
3. Generate synthetic PNGs
4. Build `augmented_dataset/` (original + synthetic for the target class only)
5. Train identical CNNs on baseline vs augmented data
6. Compare accuracy, precision, recall, F1, and confusion matrices in the web UI

---

## Architecture

```
Xynth/
├── backend/
│   ├── app/           # FastAPI, config, utils
│   ├── models/        # DCGAN + CNN architectures
│   ├── training/      # train / generate / evaluate scripts
│   ├── data/          # loaders & transforms
│   └── outputs/       # checkpoints, generated images, metrics, plots
├── frontend/          # React + Vite + TypeScript + Tailwind
├── dataset/           # optional symlink / configured path (.gitignore)
├── COVID-19_Radiography_Dataset/   # local Kaggle dump
├── .env.example
└── README.md
```

- **Generator / Discriminator** — Radford et al. (2016) DCGAN for 1×64×64 grayscale
- **Classifier** — lightweight CNN, identical hyperparameters for both experiments
- **API** — FastAPI with background jobs for train / generate / classify
- **UI** — minimalist white / blue dashboard (Home, DCGAN, Generated, Results)

---

## Dataset

[COVID-19 Radiography Database](https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database)

Expected classes:

- `COVID/`
- `Normal/`
- `Viral Pneumonia/`
- `Lung_Opacity/`

Both flat (`Class/*.png`) and Kaggle nested (`Class/images/*.png`) layouts are supported.

Configure the path in `.env`:

```env
DATASET_PATH=COVID-19_Radiography_Dataset
```

---

## Setup

### Prerequisites

- Python **3.11**
- Node.js 18+
- (Optional) CUDA or Apple MPS for faster training

### Backend

```bash
cd Xynth
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements.txt
cp .env.example .env
```

### Frontend

```bash
cd frontend
npm install
```

---

## Commands

Run all Python commands from the **project root** with the venv activated.

### API server

```bash
source .venv/bin/activate
uvicorn backend.app.api:app --reload --host 0.0.0.0 --port 8000
```

### Frontend

```bash
cd frontend
npm run dev
```

Open http://localhost:5173 — Vite proxies `/api` and `/static` to the FastAPI server.

### Train DCGAN

```bash
python -m backend.training.train_dcgan --epochs 50
# quick smoke test:
python -m backend.training.train_dcgan --epochs 1
```

### Generate synthetic images + augmented dataset

```bash
python -m backend.training.generate_images --num 500 --build-augmented
```

### Train baseline & augmented classifiers

```bash
python -m backend.training.train_classifier --epochs 10
# quick demo (subset):
python -m backend.training.train_classifier --epochs 2 --max-per-class 200
```

### Health check

```bash
curl http://127.0.0.1:8000/health
```

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Liveness |
| GET | `/dataset/stats` | Per-class counts |
| POST | `/train/dcgan` | Queue DCGAN training |
| POST | `/generate` | Queue image generation |
| POST | `/train/classifier` | Queue CNN experiments |
| GET | `/results` | Metrics, plots, gallery |
| GET | `/jobs/{name}` | Job status (`dcgan`, `generate`, `classifier`) |

Training endpoints start background jobs and return immediately. Poll `/jobs/...` or refresh the UI.

---

## Configuration

See `.env.example`:

```env
DATASET_PATH=COVID-19_Radiography_Dataset
OUTPUT_PATH=backend/outputs
TARGET_CLASS=COVID
LATENT_DIM=100
IMAGE_SIZE=64
BATCH_SIZE=64
EPOCHS=50
CLASSIFIER_EPOCHS=10
NUM_GENERATED=500
```

---

## Screenshots

> Place UI screenshots here after a full run.

| Page | Screenshot |
|------|------------|
| Home | `docs/screenshots/home.png` |
| DCGAN | `docs/screenshots/dcgan.png` |
| Generated | `docs/screenshots/generated.png` |
| Results | `docs/screenshots/results.png` |

---

## Outputs

| Path | Contents |
|------|----------|
| `backend/outputs/checkpoints/` | `dcgan_latest.pt`, epoch checkpoints, CNN weights |
| `backend/outputs/generated/` | Sample PNGs, grids, preview collage |
| `backend/outputs/metrics/` | JSON / CSV metrics |
| `backend/outputs/plots/` | Loss curves, confusion matrices |
| `augmented_dataset/` | Merged real + synthetic (gitignored) |

**Committed:** architecture code under `backend/models/`  
**Not committed:** `*.pth` / `*.pt` / `*.ckpt`, `dataset/`, `backend/outputs/` contents, `.env`

---

## UI design

Extremely minimalist:

- White background, blue palette only (`#1E5EFF`, `#0A3D91`)
- Flat, sharp edges — no glassmorphism, gradients, rounded cards, or shadows
- Inter + JetBrains Mono
- Subtle hover border states only

---

## References

Implementation is original code inspired by these works (do not treat as copies of their text or code):

1. Radford, Metz & Chintala — *Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks* (2016). https://arxiv.org/abs/1511.06434
2. Kingma & Welling — *Auto-Encoding Variational Bayes* (2013). https://arxiv.org/abs/1312.6114
3. Frid-Adar et al. — GAN-based synthetic medical image augmentation for liver lesion classification (2018). https://arxiv.org/abs/1803.01229
4. Chawla et al. — SMOTE. https://doi.org/10.1613/jair.953
5. He et al. — ADASYN. https://doi.org/10.1109/IJCNN.2008.4633969
6. NVIDIA MAISI — synthetic medical imaging. https://developer.nvidia.com/blog/addressing-medical-imaging-limitations-with-synthetic-data-generation/
7. Project MONAI GenerativeModels. https://github.com/Project-MONAI/GenerativeModels

---

## License

Academic / research use. Respect the COVID-19 Radiography Database license when redistributing data.
