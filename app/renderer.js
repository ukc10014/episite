import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { EffectComposer } from 'three/addons/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/addons/postprocessing/RenderPass.js';
import { UnrealBloomPass } from 'three/addons/postprocessing/UnrealBloomPass.js';
import { OutputPass } from 'three/addons/postprocessing/OutputPass.js';
import { CSS2DRenderer, CSS2DObject } from 'three/addons/renderers/CSS2DRenderer.js';

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const DATA_DIR = '../data/processed';
const LY_PER_PARSEC = 3.26156;

// Rendering
const MAG_LIMIT = 12.0;
const MAG_BRIGHT = -2.0;
const BASE_POINT_SIZE = 320.0;
const MIN_POINT_SIZE = 1.0;
const MAX_POINT_SIZE = 96.0;

// Bloom
const BLOOM_STRENGTH = 0.8;
const BLOOM_RADIUS = 0.4;
const BLOOM_THRESHOLD = 0.15;

// ---------------------------------------------------------------------------
// Probe definitions
// ---------------------------------------------------------------------------

function normalize(x, y, z) {
  const len = Math.sqrt(x * x + y * y + z * z);
  return [x / len, y / len, z / len];
}

// Alpha Centauri direction (from metadata: position -1.6, -1.4, -3.8 ly)
const ALPHA_CEN_DIR = normalize(-1.6, -1.4, -3.8);
// LMC approximate direction
const LMC_DIR = normalize(-1300, -33800, -49300);

const PROBES = [
  {
    name: 'Alpha — Galactic Center',
    direction: [1, 0, 0],
    velocity: 0.05,
  },
  {
    name: 'Beta — Anti-Center',
    direction: [-1, 0, 0],
    velocity: 0.05,
  },
  {
    name: 'Gamma — Galactic Rotation',
    direction: [0, 1, 0],
    velocity: 0.05,
  },
  {
    name: 'Delta — Anti-Rotation',
    direction: [0, -1, 0],
    velocity: 0.05,
  },
  {
    name: 'Epsilon — North Pole',
    direction: [0, 0, 1],
    velocity: 0.05,
  },
  {
    name: 'Zeta — South Pole',
    direction: [0, 0, -1],
    velocity: 0.05,
  },
  {
    name: 'Eta — Alpha Centauri',
    direction: ALPHA_CEN_DIR,
    velocity: 0.05,
  },
  {
    name: 'Theta — Large Magellanic Cloud',
    direction: LMC_DIR,
    velocity: 0.10,
  },
];

// ---------------------------------------------------------------------------
// Simulation state
// ---------------------------------------------------------------------------

const sim = {
  probeIndex: 0,
  time: 0,           // years
  playing: false,
  speed: 10000,      // years per real second
  fov: 60,
  // Relativistic toggles
  aberration: false,
  doppler: false,
  searchlight: false,
  // Overlay toggles (all off by default)
  showGrid: false,
  showArms: false,
  showRings: false,
  showTrail: false,
  showLabels: false,
};

// Derived: current probe position & velocity vector (updated each frame)
const probePos = new THREE.Vector3();
const probeVelDir = new THREE.Vector3();
let probeBeta = 0; // v/c

function updateProbeState() {
  const probe = PROBES[sim.probeIndex];
  probeBeta = probe.velocity;
  probeVelDir.set(...probe.direction);
  // position = direction * velocity * c * time  (c = 1 ly/yr)
  probePos.set(
    probe.direction[0] * probe.velocity * sim.time,
    probe.direction[1] * probe.velocity * sim.time,
    probe.direction[2] * probe.velocity * sim.time,
  );
}

// ---------------------------------------------------------------------------
// Shaders
// ---------------------------------------------------------------------------

const vertexShader = /* glsl */ `
  uniform float uPixelRatio;
  uniform float uBaseSize;
  uniform float uMinSize;
  uniform float uMaxSize;
  uniform float uMagLimit;

  // Relativistic uniforms
  uniform vec3  uVelocityDir;  // unit vector of probe travel direction (world space)
  uniform float uBeta;         // v/c
  uniform bool  uAberration;
  uniform bool  uDoppler;
  uniform bool  uSearchlight;

  attribute float aAbsMag;
  attribute vec3  aColor;

  varying vec3  vColor;
  varying float vBrightness;
  varying float vPointSize;

  const float LY_PER_PC = ${LY_PER_PARSEC.toFixed(6)};

  void main() {
    // --- World-space direction to this star from camera ---
    // (camera is at probePos, already set via JS)
    vec3 worldPos = position; // star position in world (ly)
    vec4 mvPosition = modelViewMatrix * vec4(worldPos, 1.0);
    float dist = length(mvPosition.xyz);

    // --- Relativistic effects ---
    float gamma = 1.0;
    float dopplerFactor = 1.0;
    float beamFactor = 1.0;

    if (uBeta > 0.001 && (uAberration || uDoppler || uSearchlight)) {
      gamma = 1.0 / sqrt(1.0 - uBeta * uBeta);

      // Angle between star direction and velocity in world space
      vec3 starDir = normalize((modelMatrix * vec4(worldPos, 1.0)).xyz - cameraPosition);
      float cosTheta = dot(starDir, uVelocityDir);

      // Doppler factor: f_obs / f_emit
      dopplerFactor = gamma * (1.0 + uBeta * cosTheta);

      // Relativistic beaming (searchlight) intensity factor
      // I_obs / I_emit = dopplerFactor^3  (for point sources)
      beamFactor = dopplerFactor * dopplerFactor * dopplerFactor;

      // Aberration: shift the apparent position
      if (uAberration) {
        // Aberrated angle: cos(theta') = (cos(theta) + beta) / (1 + beta*cos(theta))
        float cosThetaPrime = (cosTheta + uBeta) / (1.0 + uBeta * cosTheta);
        // Reconstruct aberrated direction
        // We need to rotate the star's view-space position
        // Simpler approach: scale the view-space z component
        // The aberration compresses stars toward the forward direction
        float sinTheta = sqrt(max(0.0, 1.0 - cosTheta * cosTheta));
        float sinThetaPrime = sqrt(max(0.0, 1.0 - cosThetaPrime * cosThetaPrime));

        if (sinTheta > 0.0001) {
          float ratio = sinThetaPrime / sinTheta;
          // Decompose mvPosition into component along velocity and perpendicular
          // In view space, the velocity direction transforms too, but for a
          // good approximation we work with the world-space velocity projected
          // into view space.
          vec3 velView = normalize((viewMatrix * vec4(uVelocityDir, 0.0)).xyz);
          float along = dot(mvPosition.xyz, velView);
          vec3 perp = mvPosition.xyz - along * velView;

          // New along component from aberrated angle
          float newAlong = length(perp) * cosThetaPrime / max(sinThetaPrime, 0.0001);
          // Keep same distance for rendering purposes
          mvPosition.xyz = newAlong * velView + perp * ratio;
          // Preserve original distance so brightness calc stays consistent
          mvPosition.xyz = normalize(mvPosition.xyz) * dist;
        }
      }
    }

    // --- Standard brightness calculation ---
    float distPc = max(dist / LY_PER_PC, 0.001);
    float appMag = aAbsMag + 5.0 * log(distPc) / log(10.0) - 5.0;

    // Apply searchlight brightness boost
    if (uSearchlight && uBeta > 0.001) {
      // beamFactor > 1 ahead, < 1 behind; convert to magnitude shift
      // delta_mag = -2.5 * log10(beamFactor)
      appMag -= 2.5 * log(beamFactor) / log(10.0);
    }

    float visibility = clamp((uMagLimit - appMag) / 3.0, 0.0, 1.0);

    float magFactor = clamp((8.0 - appMag) / 12.0, 0.0, 1.0);
    float size = uBaseSize * magFactor * magFactor * uPixelRatio;

    if (appMag < ${MAG_BRIGHT.toFixed(1)}) {
      float boost = 1.0 + (${MAG_BRIGHT.toFixed(1)} - appMag) * 0.4;
      size *= boost;
    }

    size = clamp(size, uMinSize, uMaxSize);

    // --- Color: apply Doppler shift ---
    vec3 color = aColor;
    if (uDoppler && uBeta > 0.001) {
      // Approximate: blueshift toward higher dopplerFactor, redshift toward lower
      // dopplerFactor > 1 = approaching = blueshift
      // dopplerFactor < 1 = receding = redshift
      float shift = clamp(dopplerFactor, 0.3, 3.0);
      if (shift > 1.0) {
        // Blueshift: boost blue, reduce red
        float t = min((shift - 1.0) * 1.5, 1.0);
        color = mix(color, vec3(0.6, 0.7, 1.0), t * 0.6);
      } else {
        // Redshift: boost red, reduce blue
        float t = min((1.0 - shift) * 1.5, 1.0);
        color = mix(color, vec3(1.0, 0.5, 0.2), t * 0.6);
      }
    }

    vColor = color;
    vBrightness = visibility;
    vPointSize = size;

    gl_PointSize = size * visibility;
    gl_Position = projectionMatrix * mvPosition;
  }
`;

const fragmentShader = /* glsl */ `
  varying vec3  vColor;
  varying float vBrightness;
  varying float vPointSize;

  void main() {
    if (vBrightness < 0.01) discard;

    vec2 center = gl_PointCoord - 0.5;
    float r = length(center) * 2.0;

    float core = exp(-r * r * 4.0);
    float halo = exp(-r * r * 1.5) * 0.3;
    float haloStrength = smoothstep(4.0, 32.0, vPointSize);
    float alpha = core + halo * haloStrength;

    float whiten = smoothstep(24.0, 64.0, vPointSize) * 0.6;
    vec3 color = mix(vColor, vec3(1.0), whiten);

    float intensity = vBrightness;
    gl_FragColor = vec4(color * intensity, alpha * intensity);
  }
`;

// ---------------------------------------------------------------------------
// Application globals
// ---------------------------------------------------------------------------

let scene, camera, webglRenderer, composer, controls;
let cssRenderer; // CSS2DRenderer for labels
let starMaterial;
let metadata;
let landmarks;
let namedStarPositions = []; // {name, pos: Vector3}

// Overlay groups
let gridGroup, armsGroup, ringsGroup, trailGroup, labelsGroup;
let trailLine; // the actual Line object inside trailGroup
let trailPositions; // Float32Array backing the trail geometry

let frameCount = 0;
let lastFpsTime = performance.now();
let lastFrameTime = performance.now();

// ---------------------------------------------------------------------------
// Init
// ---------------------------------------------------------------------------

async function init() {
  console.log('[VNP] Loading star data...');

  const [binResponse, metaResponse, landmarksResponse] = await Promise.all([
    fetch(`${DATA_DIR}/stars.bin`),
    fetch(`${DATA_DIR}/metadata.json`),
    fetch(`${DATA_DIR}/landmarks.json`),
  ]);

  const binBuffer = await binResponse.arrayBuffer();
  metadata = await metaResponse.json();
  landmarks = await landmarksResponse.json();

  const starCount = metadata.starCount;
  const floatData = new Float32Array(binBuffer);
  console.log(`[VNP] Loaded ${starCount.toLocaleString()} stars`);

  // Extract attributes
  const positions = new Float32Array(starCount * 3);
  const absMags = new Float32Array(starCount);
  const colors = new Float32Array(starCount * 3);

  for (let i = 0; i < starCount; i++) {
    const base = i * 7;
    positions[i * 3]     = floatData[base];
    positions[i * 3 + 1] = floatData[base + 1];
    positions[i * 3 + 2] = floatData[base + 2];
    absMags[i]           = floatData[base + 3];
    colors[i * 3]        = floatData[base + 4];
    colors[i * 3 + 1]    = floatData[base + 5];
    colors[i * 3 + 2]    = floatData[base + 6];
  }

  // Cache named star positions
  for (const ns of metadata.namedStars.slice(0, 200)) {
    const idx = ns.index;
    namedStarPositions.push({
      name: ns.name,
      pos: new THREE.Vector3(positions[idx * 3], positions[idx * 3 + 1], positions[idx * 3 + 2]),
    });
  }

  // --- Scene ---
  scene = new THREE.Scene();
  scene.background = new THREE.Color(0x000000);

  camera = new THREE.PerspectiveCamera(sim.fov, window.innerWidth / window.innerHeight, 0.001, 500000);

  webglRenderer = new THREE.WebGLRenderer({ antialias: false });
  webglRenderer.setSize(window.innerWidth, window.innerHeight);
  webglRenderer.setPixelRatio(window.devicePixelRatio);
  webglRenderer.outputColorSpace = THREE.SRGBColorSpace;
  document.body.appendChild(webglRenderer.domElement);

  // --- Star particles ---
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  geometry.setAttribute('aAbsMag', new THREE.BufferAttribute(absMags, 1));
  geometry.setAttribute('aColor', new THREE.BufferAttribute(colors, 3));

  starMaterial = new THREE.ShaderMaterial({
    vertexShader,
    fragmentShader,
    uniforms: {
      uPixelRatio:   { value: webglRenderer.getPixelRatio() },
      uBaseSize:     { value: BASE_POINT_SIZE },
      uMinSize:      { value: MIN_POINT_SIZE },
      uMaxSize:      { value: MAX_POINT_SIZE },
      uMagLimit:     { value: MAG_LIMIT },
      uVelocityDir:  { value: new THREE.Vector3(1, 0, 0) },
      uBeta:         { value: 0.0 },
      uAberration:   { value: false },
      uDoppler:      { value: false },
      uSearchlight:  { value: false },
    },
    transparent: true,
    depthWrite: false,
    blending: THREE.AdditiveBlending,
  });

  scene.add(new THREE.Points(geometry, starMaterial));

  // --- Bloom ---
  composer = new EffectComposer(webglRenderer);
  composer.addPass(new RenderPass(scene, camera));
  composer.addPass(new UnrealBloomPass(
    new THREE.Vector2(window.innerWidth, window.innerHeight),
    BLOOM_STRENGTH, BLOOM_RADIUS, BLOOM_THRESHOLD,
  ));
  composer.addPass(new OutputPass());

  // --- OrbitControls (look around from probe position) ---
  controls = new OrbitControls(camera, webglRenderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.05;
  controls.enableZoom = false;   // zoom doesn't make sense for probe POV
  controls.enablePan = false;    // pan doesn't make sense either
  controls.rotateSpeed = 0.4;
  // We'll set the target to be a point ahead of the probe each frame

  // --- CSS2DRenderer for labels ---
  cssRenderer = new CSS2DRenderer();
  cssRenderer.setSize(window.innerWidth, window.innerHeight);
  cssRenderer.domElement.style.position = 'absolute';
  cssRenderer.domElement.style.top = '0';
  cssRenderer.domElement.style.left = '0';
  cssRenderer.domElement.style.pointerEvents = 'none';
  document.body.appendChild(cssRenderer.domElement);

  // --- Overlays ---
  createOverlays();

  // --- UI ---
  setupUI();

  // Initial state
  selectProbe(0);

  window.addEventListener('resize', onResize);
  console.log('[VNP] Renderer initialized');
  document.getElementById('hud-stars').textContent = starCount.toLocaleString();

  lastFrameTime = performance.now();
  animate();
}

// ---------------------------------------------------------------------------
// Overlays
// ---------------------------------------------------------------------------

const GRID_COLOR = 0x111122;
const GRID_OPACITY = 0.15;
const ARM_COLORS = [0x445588, 0x445588, 0x557744, 0x445588];
const ARM_OPACITY = 0.25;
const RING_COLOR = 0x222233;
const RING_OPACITY = 0.12;
const TRAIL_COLOR = 0x446688;
const TRAIL_OPACITY = 0.4;
const TRAIL_MAX_POINTS = 2000;

function makeLineMaterial(color, opacity) {
  return new THREE.LineBasicMaterial({
    color, opacity, transparent: true, depthWrite: false,
  });
}

function createOverlays() {
  // --- Galactic plane grid (z=0) ---
  gridGroup = new THREE.Group();
  gridGroup.visible = false;
  const gridMat = makeLineMaterial(GRID_COLOR, GRID_OPACITY);
  const gridExtent = 60000; // ly in each direction
  const gridStep = 10000;   // ly between lines

  for (let v = -gridExtent; v <= gridExtent; v += gridStep) {
    // Lines parallel to X
    const gx = new THREE.BufferGeometry().setFromPoints([
      new THREE.Vector3(-gridExtent, v, 0),
      new THREE.Vector3(gridExtent, v, 0),
    ]);
    gridGroup.add(new THREE.Line(gx, gridMat));
    // Lines parallel to Y
    const gy = new THREE.BufferGeometry().setFromPoints([
      new THREE.Vector3(v, -gridExtent, 0),
      new THREE.Vector3(v, gridExtent, 0),
    ]);
    gridGroup.add(new THREE.Line(gy, gridMat));
  }
  scene.add(gridGroup);

  // --- Spiral arm centerlines ---
  armsGroup = new THREE.Group();
  armsGroup.visible = false;
  landmarks.spiralArms.forEach((arm, ai) => {
    const color = ARM_COLORS[ai % ARM_COLORS.length];
    const mat = makeLineMaterial(color, ARM_OPACITY);
    // Smooth the polyline with CatmullRom
    const curvePoints = arm.polyline.map(p => new THREE.Vector3(p.x, p.y, p.z));
    const curve = new THREE.CatmullRomCurve3(curvePoints, false, 'catmullrom', 0.5);
    const smoothed = curve.getPoints(64);
    // Render as dashed line
    const geo = new THREE.BufferGeometry().setFromPoints(smoothed);
    const dashMat = new THREE.LineDashedMaterial({
      color, opacity: ARM_OPACITY, transparent: true, depthWrite: false,
      dashSize: 400, gapSize: 200,
    });
    const line = new THREE.Line(geo, dashMat);
    line.computeLineDistances();
    armsGroup.add(line);

    // Arm label at midpoint
    const midPt = curve.getPoint(0.5);
    const label = makeLabel(arm.name, 'landmark-label');
    label.position.copy(midPt);
    label.position.z += 200; // lift slightly above plane
    armsGroup.add(label);
  });
  scene.add(armsGroup);

  // --- Distance rings from Sol (on galactic plane) ---
  ringsGroup = new THREE.Group();
  ringsGroup.visible = false;
  const ringMat = makeLineMaterial(RING_COLOR, RING_OPACITY);
  for (let r = 10000; r <= 50000; r += 10000) {
    const ringGeo = new THREE.BufferGeometry();
    const segments = 128;
    const pts = [];
    for (let i = 0; i <= segments; i++) {
      const theta = (i / segments) * Math.PI * 2;
      pts.push(new THREE.Vector3(Math.cos(theta) * r, Math.sin(theta) * r, 0));
    }
    ringGeo.setFromPoints(pts);
    ringsGroup.add(new THREE.Line(ringGeo, ringMat));
    // Distance label
    const lbl = makeLabel(`${(r / 1000).toFixed(0)}k ly`, 'landmark-label');
    lbl.position.set(r, 0, 200);
    ringsGroup.add(lbl);
  }
  scene.add(ringsGroup);

  // --- Probe trail ---
  trailGroup = new THREE.Group();
  trailGroup.visible = false;
  trailPositions = new Float32Array(TRAIL_MAX_POINTS * 3);
  const trailGeo = new THREE.BufferGeometry();
  trailGeo.setAttribute('position', new THREE.BufferAttribute(trailPositions, 3));
  trailGeo.setDrawRange(0, 0);
  const trailMat = makeLineMaterial(TRAIL_COLOR, TRAIL_OPACITY);
  trailLine = new THREE.Line(trailGeo, trailMat);
  trailGroup.add(trailLine);
  scene.add(trailGroup);

  // --- Landmark labels ---
  labelsGroup = new THREE.Group();
  labelsGroup.visible = false;

  // Sol marker
  const solLabel = makeLabel('Sol', 'landmark-label major');
  solLabel.position.set(0, 0, 0);
  labelsGroup.add(solLabel);

  // Point landmarks
  for (const pt of landmarks.points) {
    const lbl = makeLabel(pt.name, pt.name.includes('Galactic Center') ? 'landmark-label major' : 'landmark-label');
    lbl.position.set(pt.x, pt.y, pt.z);
    labelsGroup.add(lbl);
  }
  scene.add(labelsGroup);

  console.log('[VNP] Overlays created (all hidden by default)');
}

function makeLabel(text, className) {
  const div = document.createElement('div');
  div.className = className;
  div.textContent = text;
  return new CSS2DObject(div);
}

function updateTrail() {
  if (!trailGroup.visible) return;
  const probe = PROBES[sim.probeIndex];
  // Build trail from Sol to current position
  // Use evenly-spaced points along the linear trajectory
  const dist = probePos.length();
  const numPts = Math.min(TRAIL_MAX_POINTS, Math.max(2, Math.ceil(dist / 50) + 1));

  for (let i = 0; i < numPts; i++) {
    const t = i / (numPts - 1);
    trailPositions[i * 3]     = probe.direction[0] * probe.velocity * sim.time * t;
    trailPositions[i * 3 + 1] = probe.direction[1] * probe.velocity * sim.time * t;
    trailPositions[i * 3 + 2] = probe.direction[2] * probe.velocity * sim.time * t;
  }

  const geo = trailLine.geometry;
  geo.attributes.position.needsUpdate = true;
  geo.setDrawRange(0, numPts);
}

function syncOverlayVisibility() {
  gridGroup.visible = sim.showGrid;
  armsGroup.visible = sim.showArms;
  ringsGroup.visible = sim.showRings;
  trailGroup.visible = sim.showTrail;
  labelsGroup.visible = sim.showLabels;
}

// ---------------------------------------------------------------------------
// Simulation tick
// ---------------------------------------------------------------------------

function simTick(dt) {
  if (!sim.playing) return;
  sim.time += sim.speed * dt;
  sim.time = Math.min(sim.time, 2000000);
  if (sim.time >= 2000000) sim.playing = false;
  document.getElementById('time-slider').value = sim.time;
}

// ---------------------------------------------------------------------------
// Camera update — position at probe, look direction maintained by OrbitControls
// ---------------------------------------------------------------------------

function updateCamera() {
  updateProbeState();

  camera.position.copy(probePos);
  // OrbitControls target: a point 100 ly ahead in the travel direction
  // Only reset target when probe changes, otherwise let user rotate freely
  controls.target.copy(probePos).addScaledVector(probeVelDir, 100);

  // Update shader uniforms
  starMaterial.uniforms.uVelocityDir.value.copy(probeVelDir);
  starMaterial.uniforms.uBeta.value = probeBeta;
  starMaterial.uniforms.uAberration.value = sim.aberration;
  starMaterial.uniforms.uDoppler.value = sim.doppler;
  starMaterial.uniforms.uSearchlight.value = sim.searchlight;
}

// ---------------------------------------------------------------------------
// Render loop
// ---------------------------------------------------------------------------

function animate() {
  requestAnimationFrame(animate);

  const now = performance.now();
  const dt = Math.min((now - lastFrameTime) / 1000, 0.1); // seconds, capped
  lastFrameTime = now;

  simTick(dt);
  updateCamera();
  syncOverlayVisibility();
  updateTrail();
  controls.update();
  composer.render();
  cssRenderer.render(scene, camera);

  // HUD (throttled to ~2Hz)
  frameCount++;
  if (now - lastFpsTime >= 500) {
    const fps = (frameCount / ((now - lastFpsTime) / 1000)).toFixed(0);
    document.getElementById('hud-fps').textContent = fps;
    frameCount = 0;
    lastFpsTime = now;
    updateHUD();
  }
}

// ---------------------------------------------------------------------------
// HUD
// ---------------------------------------------------------------------------

function formatYears(y) {
  if (y >= 1e6) return (y / 1e6).toFixed(2) + ' Myr';
  if (y >= 1e3) return (y / 1e3).toFixed(1) + ' kyr';
  return y.toFixed(0) + ' yr';
}

function formatLY(d) {
  if (d >= 1e3) return (d / 1e3).toFixed(1) + ' kly';
  if (d >= 1.0) return d.toFixed(1) + ' ly';
  return (d * 365.25).toFixed(0) + ' light-days';
}

function updateHUD() {
  const probe = PROBES[sim.probeIndex];
  const dist = probePos.length();

  document.getElementById('hud-probe').textContent = probe.name;

  // Time slider label
  document.getElementById('time-val').textContent =
    `${formatYears(sim.time)} · ${formatLY(dist)}`;

  // Readout
  document.getElementById('rd-pos').textContent =
    `${probePos.x.toFixed(1)}, ${probePos.y.toFixed(1)}, ${probePos.z.toFixed(1)}`;
  document.getElementById('rd-vel').textContent =
    `${(probe.velocity * 100).toFixed(1)}% c (${(probe.velocity * 299792).toFixed(0)} km/s)`;
  document.getElementById('rd-dist').textContent = formatLY(dist);

  // Nearest named star
  let nearest = null;
  let nearestDist = Infinity;
  for (const ns of namedStarPositions) {
    const d = probePos.distanceTo(ns.pos);
    if (d < nearestDist) { nearestDist = d; nearest = ns; }
  }
  if (nearest) {
    document.getElementById('hud-nearest').textContent =
      `${nearest.name} (${formatLY(nearestDist)})`;
  }
}

// ---------------------------------------------------------------------------
// Probe selection
// ---------------------------------------------------------------------------

function selectProbe(index) {
  sim.probeIndex = index;
  sim.time = 0;
  sim.playing = false;
  document.getElementById('time-slider').value = 0;
  document.getElementById('btn-play').innerHTML = '&#9654; Play';
  document.getElementById('btn-play').classList.remove('active');

  updateProbeState();

  // Reset camera: position at origin, looking along travel direction
  camera.position.set(0, 0, 0);
  const dir = PROBES[index].direction;
  controls.target.set(dir[0] * 100, dir[1] * 100, dir[2] * 100);
  controls.update();

  updateHUD();
  console.log(`[VNP] Selected: ${PROBES[index].name} @ ${(PROBES[index].velocity * 100).toFixed(1)}% c`);
}

// ---------------------------------------------------------------------------
// UI setup
// ---------------------------------------------------------------------------

function setupUI() {
  // Probe selector
  const sel = document.getElementById('probe-select');
  PROBES.forEach((p, i) => {
    const opt = document.createElement('option');
    opt.value = i;
    opt.textContent = p.name;
    sel.appendChild(opt);
  });
  sel.addEventListener('change', () => selectProbe(parseInt(sel.value)));

  // Play / pause
  const btnPlay = document.getElementById('btn-play');
  btnPlay.addEventListener('click', () => {
    sim.playing = !sim.playing;
    if (sim.playing && sim.time >= 2000000) {
      sim.time = 0; // restart if at end
    }
    btnPlay.innerHTML = sim.playing ? '&#9646;&#9646; Pause' : '&#9654; Play';
    btnPlay.classList.toggle('active', sim.playing);
  });

  // Speed buttons
  document.querySelectorAll('.speed-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      sim.speed = parseInt(btn.dataset.speed);
      document.querySelectorAll('.speed-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
    });
  });

  // Time slider
  const timeSlider = document.getElementById('time-slider');
  timeSlider.addEventListener('input', () => {
    sim.time = parseFloat(timeSlider.value);
    updateHUD();
  });

  // FOV slider
  const fovSlider = document.getElementById('fov-slider');
  fovSlider.addEventListener('input', () => {
    sim.fov = parseInt(fovSlider.value);
    camera.fov = sim.fov;
    camera.updateProjectionMatrix();
    document.getElementById('fov-val').textContent = sim.fov + '°';
  });

  // Relativistic toggles
  const toggles = [
    ['btn-aberration', 'aberration'],
    ['btn-doppler', 'doppler'],
    ['btn-searchlight', 'searchlight'],
  ];
  toggles.forEach(([id, key]) => {
    const btn = document.getElementById(id);
    btn.addEventListener('click', () => {
      sim[key] = !sim[key];
      btn.classList.toggle('active', sim[key]);
    });
  });

  // Overlay toggles
  const overlayToggles = [
    ['btn-grid', 'showGrid'],
    ['btn-arms', 'showArms'],
    ['btn-rings', 'showRings'],
    ['btn-trail', 'showTrail'],
    ['btn-labels', 'showLabels'],
  ];
  overlayToggles.forEach(([id, key]) => {
    const btn = document.getElementById(id);
    btn.addEventListener('click', () => {
      sim[key] = !sim[key];
      btn.classList.toggle('active', sim[key]);
    });
  });
}

// ---------------------------------------------------------------------------
// Resize
// ---------------------------------------------------------------------------

function onResize() {
  const w = window.innerWidth;
  const h = window.innerHeight;
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
  webglRenderer.setSize(w, h);
  composer.setSize(w, h);
  cssRenderer.setSize(w, h);
}

// ---------------------------------------------------------------------------
// Start
// ---------------------------------------------------------------------------

init().catch(err => {
  console.error('[VNP] Init failed:', err);
  document.getElementById('hud-stars').textContent = 'ERROR — see console';
});
