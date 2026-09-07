# Xynth Project Progress Report

**Report date:** 7 September 2026  
**Project:** Synthetic X-ray Augmentation using DCGAN

## Executive summary

Xynth is implemented as an end-to-end research prototype: it can load the COVID-19 Radiography Dataset, train a DCGAN on COVID chest X-rays, generate synthetic images, build an augmented dataset, train baseline and augmented CNN classifiers on a shared test split, and display the outputs in a React dashboard.

The model pipeline has completed a **short local experiment**, but it is **not a final fully trained model**. The completed run used 2 epochs for the DCGAN and 2 epochs for each classifier, generated 64 synthetic COVID images, and used Apple MPS acceleration. The project defaults target 50 DCGAN epochs, 10 classifier epochs, and 500 generated images.

The short run is useful as a functional proof of concept. Its metrics should not be presented as clinically meaningful or as final research results.

## Current completion status

- Project structure and configuration: complete
- Dataset discovery and preprocessing: complete
- DCGAN generator and discriminator implementation: complete
- DCGAN training and checkpointing pipeline: complete
- Synthetic image generation pipeline: complete
- Augmented dataset builder: complete
- Baseline and augmented CNN training pipeline: complete
- Evaluation metrics and confusion matrices: complete
- FastAPI backend: complete
- React/Vite dashboard: complete
- Local application startup: verified
- Short end-to-end model experiment: complete
- Full-duration model training: pending
- Automated project tests: not currently included
- Clinical validation, external validation, and production deployment: not performed

## Dataset

The configured local dataset is the COVID-19 Radiography Database. The loader supports both `Class/*.png` and Kaggle's `Class/images/*.png` layouts.

Detected image counts:

- COVID: 3,616
- Normal: 10,192
- Viral Pneumonia: 1,345
- Lung Opacity: 6,012
- Total: 21,165

COVID is the configured DCGAN target class. The data and generated augmented dataset are intentionally excluded from Git because they are large local artifacts and may be subject to dataset licensing restrictions.

## Implemented machine-learning pipeline

### Preprocessing

- Images are loaded with Pillow.
- Inputs are resized to 64×64.
- DCGAN images are converted to one grayscale channel and normalized to `[-1, 1]`.
- Classifier transforms use the shared preprocessing implementation.
- Supported image extensions include PNG, JPEG, BMP, TIFF, and WebP.

### DCGAN

The PyTorch DCGAN follows the standard 64×64 architecture:

- Generator input: 100-dimensional latent vector
- Generator output: 1×64×64 grayscale image
- Generator: transposed convolutions, batch normalization, ReLU, and final Tanh
- Discriminator: strided convolutions, batch normalization, LeakyReLU, and final Sigmoid
- DCGAN weight initialization is applied to convolution and batch-normalization layers.
- Training supports checkpoint creation, periodic image grids, configurable epochs and batch size, and checkpoint resume.

### Classifier

The classifier is a lightweight four-class PyTorch CNN:

- Four convolution stages with 32, 64, 128, and 256 channels
- Batch normalization and ReLU activations
- Max pooling followed by adaptive average pooling
- Dropout of 0.3
- Four-class linear output for COVID, Normal, Viral Pneumonia, and Lung Opacity
- Cross-entropy loss and Adam optimization

### Fair baseline-versus-augmentation comparison

- A reproducible stratified 80/20 split uses random seed 42.
- Baseline and augmented experiments use the same held-out real-image test indices.
- Synthetic COVID images are added only to the augmented training set.
- Synthetic images do not enter the test set.
- Accuracy, macro precision, macro recall, macro F1, training history, checkpoints, and confusion matrices are saved.

## Completed local experiment

### DCGAN run

- Status: completed
- Epochs: 2
- Device: Apple MPS
- Target class: COVID
- Batches per epoch: 56
- Final generator loss: 6.27695
- Final discriminator loss: 1.06611
- Latest checkpoint: `backend/outputs/checkpoints/dcgan_latest.pt`
- Loss plot: `backend/outputs/plots/dcgan_loss.png`
- Training grids were produced for epochs 1 and 2.

The losses fluctuate considerably, which is common in GAN training but is also evidence that a 2-epoch run is too short for a strong convergence claim. Image quality should be inspected after a longer run and assessed with suitable quantitative measures.

### Synthetic generation

- Status: completed
- Synthetic COVID images generated: 64
- Individual samples were written under `backend/outputs/generated/samples/`.
- Preview collage and 4×4 preview images were produced.
- The materialized augmented dataset contains 3,680 COVID images: 3,616 original plus 64 synthetic.
- Other augmented-dataset counts remain Normal 10,192, Viral Pneumonia 1,345, and Lung Opacity 6,012.

### Classifier experiment

Both classifiers completed 2 epochs on Apple MPS. The held-out test set contained 160 real images, with 40 images from each class. This was a quick, downsampled demonstration rather than a full-dataset experiment.

Baseline results:

- Accuracy: 26.875%
- Macro precision: 11.964%
- Macro recall: 26.875%
- Macro F1: 14.167%
- Final training accuracy: 68.906%
- Final training loss: 0.78276

Augmented results:

- Accuracy: 47.500%
- Macro precision: 55.561%
- Macro recall: 47.500%
- Macro F1: 47.461%
- Final training accuracy: 70.170%
- Final training loss: 0.80112

Observed short-run differences:

- Accuracy increased by 20.625 percentage points.
- Macro F1 increased by approximately 33.294 percentage points.
- COVID correct predictions increased from 4/40 to 18/40.
- Viral Pneumonia correct predictions increased from 0/40 to 15/40.
- Lung Opacity correct predictions decreased from 39/40 to 30/40.

These values demonstrate that the comparison pipeline works, but they do not establish that synthetic augmentation reliably improves the classifier. The run is too short and small, only one random seed was evaluated, the baseline is severely undertrained, and the synthetic addition is only 64 images.

## Backend implementation

The FastAPI application exposes:

- `GET /health` for liveness
- `GET /dataset/stats` for class counts
- `POST /train/dcgan` to queue DCGAN training
- `POST /generate` to generate synthetic images and optionally build the augmented dataset
- `POST /train/classifier` to train baseline and augmented classifiers
- `GET /jobs/{name}` for in-memory job state
- `GET /results` for aggregated metrics, plots, checkpoints, and gallery data
- `GET /files/plot/{name}` for plot downloads

Generated images and result plots are served as static files. CORS is configured for the local Vite frontend.

Job state is in memory, so generation and classifier jobs can appear `idle` after the API server restarts even when persisted result files prove that those runs completed.

## Frontend implementation

The React 19, TypeScript, Vite, Tailwind CSS application includes:

- Home page with dataset and pipeline overview
- DCGAN page for training controls and loss visualization
- Generated page with synthetic-image gallery and generation controls
- Results page with baseline-versus-augmented metrics and confusion matrices
- Reusable navigation, buttons, statistics, charts, pipeline, and confusion-matrix components
- API service and results-fetching hook
- Vite proxying for backend API and static resources

The UI follows the intended minimalist white-and-blue visual design.

## Local runtime verification

- Backend health check returned HTTP 200 with `{"status":"ok","project":"Xynth"}`.
- Frontend development server started successfully at `http://127.0.0.1:5173/`.
- Backend API was available at `http://127.0.0.1:8000/`.
- Backend Python source passed bytecode compilation.
- The frontend TypeScript production build completed successfully.
- Frontend lint completed with one non-blocking React warning in `src/hooks/useResults.ts` about synchronous state initialization inside an effect.
- Vite reported a non-blocking bundle-size warning because the main minified JavaScript chunk is approximately 636 kB.

There is currently no project-owned automated test suite, so this verification covers syntax, compilation, linting, API health, and local startup rather than behavioral test coverage.

## Generated local artifacts

The completed run produced:

- DCGAN latest checkpoint
- Baseline CNN checkpoint
- Augmented CNN checkpoint
- DCGAN loss history and plot
- Generated image samples and preview grids
- Baseline and augmented metric JSON files
- Comparison JSON and CSV files
- Reproducible split indices
- Baseline and augmented confusion-matrix plots
- Augmented-dataset metadata and materialized dataset

These files are under ignored output/data paths and will not be committed by a normal `git add .`. This is appropriate for large model weights, generated data, and local datasets. The report records the results that were present locally on the report date.

## Remaining work before calling the model final

1. Train the DCGAN for the planned duration, starting with the configured 50 epochs and reviewing generated samples at checkpoints.
2. Generate the configured 500 or a scientifically justified number of synthetic images.
3. Train both classifiers for at least the configured 10 epochs on the intended dataset scope.
4. Repeat experiments with multiple random seeds and report mean values and variability.
5. Evaluate synthetic-image quality and diversity; check for memorization and near-duplicates.
6. Tune class balancing so augmentation does not address COVID while leaving Viral Pneumonia as the smallest class.
7. Add validation-based model selection and consider early stopping.
8. Add automated unit and integration tests for data loading, model tensor shapes, API endpoints, and split leakage.
9. Record dependency versions, hardware, runtime, random seeds, and full experiment parameters for reproducibility.
10. Perform external and clinical validation before any medical-use claim.

## Recommended full-run commands

Run these from the project root after activating the virtual environment:

```bash
source .venv/bin/activate
python -m backend.training.train_dcgan --epochs 50
python -m backend.training.generate_images --num 500 --build-augmented
python -m backend.training.train_classifier --epochs 10
```

GAN and classifier results should be reviewed after these commands finish. Longer training does not automatically guarantee better output, so checkpoints, generated images, loss behavior, and held-out metrics all need inspection.

## Conclusion

The software and end-to-end experimental workflow are implemented and have been exercised successfully with a short run. The available checkpoints and metrics are **demo/smoke-test artifacts**, not final research-grade models. A full controlled training and evaluation run remains necessary before the project can accurately claim a completed model or a reliable augmentation benefit.
