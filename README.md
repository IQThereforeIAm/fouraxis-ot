# Four-Axis OT – Reproducibility Bundle

This repo re-creates the experiments from  
**“A Four-Axis Positive-Simplex Coordinate Framework with Optimal-Transport Acceleration”** (TOMS 2025).

## Quick start (GPU)

```bash
git clone https://github.com/<your-user>/fouraxis-ot.git
cd fouraxis-ot
docker build -t fouraxis-ot .
docker run --gpus all fouraxis-ot
