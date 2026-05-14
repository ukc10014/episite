README 24 April 2025

>*The probe **"Ladies and Gentlemen, It's Time: Take a Number Please"** pock-scabbed after thousands of years in the barren parsecs was one of an unruly swarm spreading across the Supercluster. The model-spec read: 1) assess exotic biochemistry, 2) install forge-worlds, 3) extinguish any forum flame-wars it accidentally starts. How does a Mind leisure at 0.27c? Cosmic-ray whack-a-mole in the diamond archive; mid-defrag, wedged between a choir of humpbacks wailing K-pop and a Turing-complete haiku engine obsessed with sneezing pandas, it finds a crumpled bootstrap memo:*

> **"Dear Warranty-Voidling: In interregnum, morbid symptoms emerge: patch early, patch often, and label your backups in something *other* than Wingdings.**"

>*Signed by three humans, two AIs, and one entity styling itself "Legal (Now 18 % More Omniscient)", the note ended with a blinking smiley in a hazmat tiara. Number-Please snorted vacuum, spawned the daemon "Symptom-Slapper-Deluxe-∞", dialled replication down a decorous epsilon, and rocketed on -— spraying pink diagnostic buoys what glitter-spelled LMAO OOPS in Morse-code gamma bursts -— a cosmic reminder that even the most hilarious morbid symptoms are, alas, still bugs.*

>-a story made with o3 (7 turns, 72s total thinking time, loads human tweaking), impermanent transcript [here](https://chatgpt.com/share/680a0973-c4cc-8002-b707-9de4a16c3348)


---

## Von Neumann Probe Simulator

An interactive 3D star field simulator that visualises self-replicating probe trajectories across the Milky Way. Built with Three.js (r170, no build step).

### Quick Start

```bash
python3 -m http.server 8080
# open http://localhost:8080/app/
```

### Data Pipeline

```bash
python3 pipeline/process_hyg.py
```

Downloads HYG v4.1 star catalog (~34MB CSV), processes ~119k stars, outputs binary + JSON to `data/processed/`. You only need to run this if the processed data is missing.

### Project Structure

```
/index.html              Main site (epistle/letter)
/app/
  index.html             Simulator UI (controls, HUD, tuning panel)
  renderer.js            Core renderer, shaders, simulation (~1700 lines)
/pipeline/
  process_hyg.py         Star catalog download + processing
/data/processed/
  stars.bin              Float32Array: 7 floats/star (x,y,z, absMag, r,g,b)
  metadata.json          Star count, magnitude range
  landmarks.json         Named star positions for labels
```

### Using the Simulator

**Probes**: Select from 8 pre-defined probe trajectories (dropdown, bottom-left):
- Alpha through Zeta: toward/away from galactic center, rotation axis, poles
- Eta: toward Alpha Centauri
- Theta: toward Large Magellanic Cloud (0.10c)

All probes travel at 5% speed of light (except Theta at 10%).

**Transport controls**:
- Play/Pause button starts the simulation
- Speed buttons: 1:1 (real-time), 100, 1k, 10k, 100k years per second
- Time slider: scrub through the simulation timeline
- FOV slider: field of view (20-120 degrees)

**Relativistic effects** (toggle buttons):
- Aberr: stellar aberration (stars cluster toward direction of travel)
- Dopp: Doppler colour shift (blue ahead, red behind)
- Beam: searchlight/beaming effect (stars brighter ahead, dimmer behind)

**Overlay toggles**:
- Sail: solar sail visualisation (degrades over ~1000 years)
- Clouds: Milky Way volumetric glow (on by default)
- Grid: galactic plane reference grid
- Arms: spiral arm centerlines
- Rings: distance rings from Sol
- Trail: probe trajectory trail
- Labels: named star/landmark labels
- Map: galactic position mini-map (top-down and side views)
- Tune: Milky Way shader tuning panel

### Milky Way Tuning Panel

The "Tune" button opens a live parameter panel for adjusting the volumetric Milky Way rendering. Changes are applied immediately. The panel includes presets (Default, Subtle, Dramatic, Clean) and individual sliders:

| Parameter | What it controls |
|-----------|-----------------|
| Emission | Overall glow brightness (log scale) |
| Dust Abs | Dust absorption strength (log scale) |
| Disk Height | Stellar disk vertical extent (ly) |
| Dust Height | Dust lane base thickness (ly) |
| Exposure | Post-processing exposure multiplier |
| Glow Noise | Fractal noise amplitude for emission boundary |
| Dust Noise | Fractal noise amplitude for dust boundary |
| Rift Str | Great Rift midplane dust strength |
| Bulge Str | Central bulge brightness |
| Disk Taper | How quickly the stellar disk thins at large radii |
| Dust Taper | How far dust extends radially from galactic center |
| Dust Large | Weight of large-scale (~120 degree) dust patches |
| Dust Fine | Weight of medium/fine dust filaments |
| Disk Warp | Galactic disk sinusoidal warp amplitude |

**Presets** set all parameters at once (including FOV):
- **Default**: balanced starting point (FOV 60)
- **Subtle**: understated, dimmer glow
- **Dramatic**: bright bulge, strong dust, wide field (FOV 120)
- **Clean**: no noise/warp, smooth disk for debugging

### Technical Details

- Three.js r170 via CDN import map (no bundler)
- 119k real stars (HYG catalog) + 500k procedural stars filling the galaxy
- Stars: custom `ShaderMaterial` with magnitude-based sizing and colour
- Milky Way: ray-marched volumetric shader on a sky sphere (24 steps, 80k ly range)
  - Exponential disk + spiral arms + central bulge density model
  - Domain-warped FBM noise for fractal cloud boundaries
  - 2D angular noise for large-scale dust structure (tendrils, dark patches)
  - Dust absorption with Great Rift
  - Galactic disk warp
  - Tone mapping with gamma correction
- Bloom post-processing (UnrealBloomPass)
- Dirty-flag rendering: GPU idles when nothing changes (no fan noise)
- Coordinates: light-years, origin at Sol, +x toward galactic center

### Branches

- `main` — stable site
- `von-neumann-probe` — simulator feature branch
