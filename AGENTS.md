# Episite Agent Notes

This repository contains a browser-based installation app with two screens that are meant to be held in tension rather than used as ordinary software.

## Runtime Shape

- `/app/` is the primary experience.
- The app begins with a splash screen that displays this `AGENTS.md` file as its loading text.
- After loading, the app hard-cuts between two live screens on prime-number millisecond intervals: `1ms`, `3ms`, `5ms`, `7ms`, `11ms`, and so on.
- The initial visible screen after the splash is the starfield simulator.
- Hidden screens are paused rather than advanced in the background.

## Runtime Modes

- Default `/app/` is `web` mode for public delivery.
- In `web` mode, the starfield is deliberately fake: `app/static/starfield-frame.jpg` is a captured frame from the simulator, used as a static stand-in so the public web route does not load the full Three.js renderer or astronomical data.
- Use `/app/?mode=installation` for the future physical installation mode. That mode loads `app/renderer.js`, the star catalog, the procedural galaxy, and the real simulator.
- Future agents should preserve this split. The static web starfield is intentional, not a regression: it keeps arbitrary browsers from trying to run the full galaxy renderer and the in-browser LLM at the same time.

## Screen One: Von Neumann Probe Starfield

- The full simulator is implemented in `app/renderer.js` and is only loaded in `?mode=installation`.
- The default web route uses the captured static simulator frame at `app/static/starfield-frame.jpg`.
- Uses Three.js from a CDN import map.
- Renders a probe-eye view of transit through the Milky Way from processed astronomical data.
- Loads `data/processed/stars.bin`, `metadata.json`, and `landmarks.json`.
- Adds procedural Milky Way structure, dust, nebula-like clouds, relativistic visual toggles, minimaps, overlays, and debug controls.
- The simulation is intentionally almost still at real human timescales.

## Screen Two: Browser LLM Activation Viewer

- Implemented with `app/model-viewer.js` and `app/model-worker.js`.
- Loads `onnx-community/SmolLM2-135M-ONNX` in the browser.
- Prefers ONNX Runtime WebGPU with the q4f16 ONNX graph and external data.
- Patches the ONNX graph in-browser to expose residual-stream tensors as outputs.
- Visualizes 30 residual-stream rows by 576 model dimensions as a full-screen activation field.
- Displays generated text in a single bottom ticker, with the newest token highlighted.
- Keeps `app/model.html` as a direct debug page for the model viewer.

## Coordination Rules

- Do not reintroduce navigation between `/app/index.html` and `/app/model.html` for the main experience. The starfield and LLM viewer should stay in memory.
- Keep the splash auto-entering once the starfield and model are ready; do not require a user click to begin.
- Treat this file as the agent-facing project brief. Update it when the architecture, major behaviors, or constraints change.
- `CLAUDE.md` should remain a pointer to this file so Claude Code and Codex read the same operational notes.

## Known Rough Edges

- Browser support depends on WebGPU for the intended direct residual-stream path.
- Token spacing in the ticker has been imperfect with the current tokenizer decoding path.
- The prime cadence quickly becomes slow enough that one screen can dominate the view for long stretches.
