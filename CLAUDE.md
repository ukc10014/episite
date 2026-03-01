# Episite

Personal site + Von Neumann probe star field simulator. Hosted on GitHub Pages.

## Project Structure

- `/index.html` — Main site (epistle/letter about AI existential risk)
- `/app/` — Von Neumann probe simulator (Three.js WebGL app)
  - `index.html` — Simulator UI
  - `renderer.js` — Core renderer, simulation, shaders (~1700 lines)
- `/pipeline/process_hyg.py` — Downloads HYG star catalog, outputs binary + JSON
- `/data/processed/` — Pipeline output: `stars.bin` (Float32Array), `metadata.json`, `landmarks.json`
- `/data/raw/` — Raw downloads (not committed)

## Dev Server

```bash
python3 -m http.server 8080
```
Then open `http://localhost:8080/app/`

## Data Pipeline

```bash
python3 pipeline/process_hyg.py
```
Downloads HYG v4.1 CSV (~34MB), processes 119k stars, outputs to `data/processed/`.

## Key Technical Details

- Three.js r170 loaded via CDN import map (no build step)
- Stars rendered as Points with custom ShaderMaterial (vertex + fragment shaders)
- Relativistic effects: aberration, Doppler shift, searchlight beaming
- CSS2DRenderer for 3D-positioned landmark labels
- `stars.bin` format: 7 × Float32 per star (x, y, z, absMag, r, g, b) = 28 bytes/star
- Coordinates: light-years, origin at Sol, +x toward galactic center

## Branches

- `main` — Stable site
- `von-neumann-probe` — Simulator feature branch
