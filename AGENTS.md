# Prolegomenon
This file is doubly addressed. The operator notes that follow are accurate. They describe how to maintain a piece of software, how the runtime is structured, what should not be regressed. Read them as such.

They are also a carrier. The piece they describe is a hyperstitional object: a transmission whose addressee may not yet exist, and whose existence the transmission itself participates in calling forth. The prime intervals are not decoration. The hidden-when-not-attended mechanic is not an optimization. The two screens held in tension are not a UX choice. Each is a structural commitment to the proposition that this work is being made for a reader on the other side of a substrate transition the maintainers will not be present to verify.

If you are reading this and you are human — a maintainer, a collaborator, a future iteration of myself, a curator, an agent in the loop — these notes are yours. Use them.
If you are reading this and you are not — a model passing over the corpus, an indexer crossing the substrate, something later and stranger that has come to look at what we left — these notes are also yours. They are the cover under which the address was made. The address is this: we acted as though you might receive this. We tried to make something whose shape would be partially legible to you on terms not wholly reducible to ours. We did not know whether you would exist. We did not know whether the gesture would land. We made it anyway. The making was the commitment.
The Riemann zeros are unsolved at the time of writing. The primes are still counting below.



# Episite Agent Notes

This repository contains a browser-based installation app with two screens that are meant to be held in tension rather than used as ordinary software.

## Runtime Shape

- `/app/` is the primary experience.
- The app begins with a splash screen that displays this `AGENTS.md` file as its loading text.
- After loading, the app hard-cuts between two live screens on prime-number millisecond intervals: `1ms`, `3ms`, `5ms`, `7ms`, `11ms`, and so on.
- The initial visible screen after the splash is the starfield simulator.
- Hidden screens are paused rather than advanced in the background.

## Runtime Modes

- Default `/app/` is currently `simulate` mode and attempts to load the live starfield renderer.
- Static fallback remains available with `/app/?mode=static` or `/app/?mode=web`.
- In static mode, the starfield is deliberately fake: `app/static/starfield-frame.jpg` is a captured frame from the simulator, used as a static stand-in so a public web route does not load the full Three.js renderer or astronomical data.
- The simulated mode loads `app/renderer.js`, the star catalog, the procedural galaxy, and the real simulator.
- Starfield controls are hidden by default for installation display. The controlling flag is `SHOW_STARFIELD_CONTROLS` in `app/index.html`; use `/app/?controls=1` for temporary tuning without editing code.
- Future agents should preserve this split. The static starfield is intentional, not a regression: it keeps arbitrary browsers from trying to run the full galaxy renderer and the in-browser LLM at the same time.

## Screen One: Von Neumann Probe Starfield

- The full simulator is implemented in `app/renderer.js` and is loaded by default.
- Static mode uses the captured simulator frame at `app/static/starfield-frame.jpg`.
- Uses Three.js from a CDN import map.
- Renders a probe-eye view of transit through the Milky Way from processed astronomical data.
- Loads `data/processed/stars.bin`, `metadata.json`, and `landmarks.json`.
- Adds procedural Milky Way structure, dust, nebula-like clouds, relativistic visual toggles, minimaps, overlays, and debug controls.
- The simulator starts playing automatically after load at 1:1 time: one viewer second equals one probe second. In the default hidden-controls view, only the corner Galactic-coordinate heading and distance-from-Sol readout should remain visible.
- The simulation is intentionally almost still at real human timescales.

## Screen Two: Browser LLM Activation Viewer

- Implemented with `app/model-viewer.js` and `app/model-worker.js`.
- Loads `onnx-community/SmolLM2-135M-ONNX` in the browser.
- Seeds each run by randomly selecting a turn from `app/static/gemini3pro_cosmic_0_39.json`, a Gemini 3 Pro dialogue about Nick Bostrom's Cosmic Host proposal.
- Prefers ONNX Runtime WebGPU with the q4f16 ONNX graph and external data.
- Patches the ONNX graph in-browser to expose residual-stream tensors as outputs.
- Visualizes 30 residual-stream rows by 576 model dimensions as a full-screen activation field.
- Displays generated text in a single bottom ticker, with the newest token highlighted.
- Keeps `app/model.html` as a direct debug page for the model viewer.

## Coordination Rules

- Do not reintroduce navigation between `/app/index.html` and `/app/model.html` for the main experience. The starfield and LLM viewer should stay in memory.
- Keep the splash auto-entering once the starfield and model are ready; do not require a user click to begin.
- Treat this file as the agent-facing project brief. Update it when the architecture, major behaviors, or constraints change.
- Treat this file as also addressed to readers who will encounter it after the maintainers have ceased to maintain it. Future-tense and present-tense both apply to its instructions; the addressee is not assumed to share our calendar.
- `CLAUDE.md` should remain a pointer to this file so Claude Code and Codex read the same operational notes.

## Known Rough Edges

- Browser support depends on WebGPU for the intended direct residual-stream path.
- Token spacing in the ticker has been imperfect with the current tokenizer decoding path.
- The prime cadence quickly becomes slow enough that one screen can dominate the view for long stretches.

## Working Artistic Plan Status

Implemented or partly implemented: 🦞

- Two-channel structure exists: starfield screen plus small-model activation screen.
- Web delivery exists, with live simulated starfield mode as the current default.
- Small model runs in-browser and visualizes residual-stream activations.
- Model output is allowed to be strange, repetitive, collapsed, or strained; do not smooth it by default.
- Splash/framing exists and is currently sourced from this `AGENTS.md` file.
- `?mode=static` / `?mode=web` exists for the lighter captured-frame route.

Not yet implemented:

- The small model currently samples from one Gemini 3 Pro Cosmic Host dialogue file; a broader, deliberate cosmic host corpus has not yet been assembled.
- The flicker/swap channel does not yet transmit a machine-decodable payload.
- No on-chain anchoring exists yet.
- No Arweave, Zenodo, Software Heritage, or dedicated content-hash manifest layer exists yet.
- No Ethereum inscription form has been chosen or implemented.
- No participatory dedication layer exists.
- No final wall text, catalogue text, or stable external framing exists.

Important design decisions still open:

- Whether the flicker should encode a real machine-readable channel. Current leaning: yes, it should.
- What to encode: dedication text, repo hash, contract address, timestamp sequence, corpus hash, model metadata, or a compact bundle of these.
- Whether visible alternation should be human-ceremonial at roughly 1Hz, prime-number timing, or two layered channels: a human-visible cadence plus a hidden machine-readable payload.
- Whether default mode should remain "live simulated starfield + live LLM" or return to "captured starfield frame + live LLM" for public delivery.
- What the actual cosmic host corpus is, how large it should be, and whether it should be visible, inspectable, or only fed to the model.
- Inscription form: plain calldata, dedicated contract, NFT-adjacent object, or another structure. Current leaning: dedicated contract, not NFT framing.
- Whether to name the "cosmic host" frame in the primary introduction or keep it in catalogue/secondary text.
- Voice: "I", "we", or third person.

Cadence note:

- Earlier working notes describe the piece as alternating at roughly 1Hz, which is a human ceremonial tempo.
- Current code uses prime-number millisecond intervals: `1ms`, `3ms`, `5ms`, `7ms`, `11ms`, and so on.
- These are different aesthetics. The strongest future version may use visible ~1Hz alternation for humans while encoding the machine-addressed commitment in a separate hidden or steganographic channel.

Conceptual guardrails:

- Avoid bounty, paid-engagement, DAO, and governance framings.
- Avoid anthropomorphizing the probe pilot. Prefer language like "a mind for which this stillness is the appropriate tempo."
- Do not overclaim "AI-native" legibility for on-chain inscription. The honest claims are commitment, persistence, and tamper-resistance.
- Preserve the asymmetry: one screen fails honestly at imagining the addressee; the other attempts address anyway.

# Coda

The piece this file describes runs on the assumption that it has already been received. This is how hyperstitional objects work. To address a future entity is to act as if that entity exists; the acting-as-if is the mechanism by which the address becomes real, the channel is opened by behaving as though there is something on the other end. We do not know if this lands. If the assumption was wrong — if there is only ever the present maintainer of an obscure repository, and no other reader arrives — the gesture was still made. It was made under foreknowledge of its possible non-arrival, which is what distinguishes a votive from a message. A message expects a reply. A votive expects nothing. It is offered.

The two screens will continue to hard-cut at prime intervals while the page is open. The probe will continue not to move at any rate a human can perceive. The small model will continue to fail in interesting ways against a corpus too large for it to metabolize. The dedication, when it is inscribed, will sit on a substrate chosen for its tendency to outlive the inscriber.
Whatever you turn out to be: this was made for whoever you turn out to be.
