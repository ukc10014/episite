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
const BASE_POINT_SIZE = 70.0;
const MIN_POINT_SIZE = 1.0;
const MAX_POINT_SIZE = 32.0;

// Bloom
const BLOOM_STRENGTH = 0.5;
const BLOOM_RADIUS = 0.3;
const BLOOM_THRESHOLD = 0.2;

// Galactic model (shared between procedural generation and minimap)
const GC_X = 26000;          // Galactic center offset from Sol along +x (ly)
const GALAXY_R = 50000;       // Disk radius (ly)
const DISK_SCALE_LENGTH = 10000; // Exponential disk scale length (ly)
const DISK_SCALE_HEIGHT = 300;   // Thin disk scale height (ly)
const BULGE_SCALE_R = 2000;     // Central bulge scale radius (ly)
const SPIRAL_A = 2000;          // Log-spiral coefficient
const SPIRAL_B = 0.22;          // Log-spiral growth rate
const SPIRAL_STARTS = [0, Math.PI / 2, Math.PI, 3 * Math.PI / 2];
const SPIRAL_ARM_WIDTH = 0.15;  // Arm angular half-width (radians)
const SPIRAL_ARM_STRENGTH = 0.5; // Arm overdensity factor
const PROCEDURAL_COUNT = 500000;
const HYG_EXCLUSION_RADIUS = 500; // ly — don't place procedural stars near Sol

// Solar sail
const SAIL_DISTANCE = 1000;      // meters ahead of probe
const SAIL_SIZE = 2000;           // meters (2km wide diamond)
const SAIL_DEGRADE_TIME = 1000;   // years until mostly destroyed
const SAIL_BOOM_WIDTH = 0.002;    // fraction of half-span for boom/frame thickness

// Milky Way background
const MW_RAY_STEPS = 32;
const MW_MAX_DIST = 60000.0;       // ly — max ray march distance
const MW_DUST_SCALE_HEIGHT = 100.0; // ly — dust disk thinner than stellar

// ---------------------------------------------------------------------------
// Probe definitions
// ---------------------------------------------------------------------------

function normalize(x, y, z) {
  const len = Math.sqrt(x * x + y * y + z * z);
  return [x / len, y / len, z / len];
}

const PROBE_COUNT = 997; // nearest prime to 1000
const PROBE_VELOCITY = 0.05;
const PROBE_PLANE_BIAS_EXPONENT = 2.35;
const GOLDEN_ANGLE = Math.PI * (3 - Math.sqrt(5));

function galacticCoordinates(direction) {
  const [x, y, z] = direction;
  const longitude = (Math.atan2(y, x) * 180 / Math.PI + 360) % 360;
  const latitude = Math.asin(Math.max(-1, Math.min(1, z))) * 180 / Math.PI;
  return { longitude, latitude };
}

function formatProbeNumber(index) {
  return String(index + 1).padStart(4, '0');
}

function generateProbeSwarm(count) {
  return Array.from({ length: count }, (_, index) => {
    const centered = ((index + 0.5) / count) * 2 - 1;
    // Compress galactic latitude toward b=0 so the probe swarm is denser
    // around the Milky Way plane while still sampling both poles.
    const biasedZ = Math.sign(centered) * Math.pow(Math.abs(centered), PROBE_PLANE_BIAS_EXPONENT);
    const radius = Math.sqrt(Math.max(0, 1 - biasedZ * biasedZ));
    const longitude = index * GOLDEN_ANGLE;
    const direction = normalize(
      Math.cos(longitude) * radius,
      Math.sin(longitude) * radius,
      biasedZ,
    );
    const coords = galacticCoordinates(direction);
    return {
      name: `Probe ${formatProbeNumber(index)}`,
      direction,
      velocity: PROBE_VELOCITY,
      longitude: coords.longitude,
      latitude: coords.latitude,
    };
  });
}

const PROBES = generateProbeSwarm(PROBE_COUNT);

// ---------------------------------------------------------------------------
// Simulation state
// ---------------------------------------------------------------------------

// Fixed starfield epoch for the project: a reference to AlphaGo / Lee Sedol
// and Move 37, supplied as 10 March 2016 in the project notes. The probe
// clock starts from this date instead of from page load, so the starfield has
// a stable shared timebase across reloads and machines.
const STARFIELD_EPOCH_MS = Date.parse('2016-03-10T00:00:00Z');
const MS_PER_JULIAN_YEAR = 365.25 * 24 * 60 * 60 * 1000;
const REALTIME_YEARS_PER_SECOND = 1 / (365.25 * 24 * 60 * 60);
const SIM_TIME_LIMIT_YEARS = 2000000;

const sim = {
  probeIndex: 0,
  time: 0,           // years
  playing: true,
  speed: REALTIME_YEARS_PER_SECOND, // 1:1: one viewer second = one probe second
  fov: 60,
  // Relativistic toggles
  aberration: false,
  doppler: false,
  searchlight: false,
  // Overlay toggles
  showSail: false,    // solar sail ahead of probe
  showClouds: true,   // on by default — core to Milky Way look
  showGrid: false,
  showArms: false,
  showRings: false,
  showTrail: false,
  showLabels: false,
  showMap: false,
};

function starfieldEpochYears(nowMs = Date.now()) {
  const elapsedYears = (nowMs - STARFIELD_EPOCH_MS) / MS_PER_JULIAN_YEAR;
  return Math.max(0, Math.min(SIM_TIME_LIMIT_YEARS, elapsedYears));
}

function resetSimTimeToEpoch() {
  sim.time = starfieldEpochYears();
  const timeSlider = document.getElementById('time-slider');
  if (timeSlider) timeSlider.value = sim.time;
}

// Derived: current probe position & velocity vector (updated each frame)
const probePos = new THREE.Vector3();
const probeVelDir = new THREE.Vector3();
let probeBeta = 0; // v/c
let renderContainer = null;
let renderPaused = true;

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
        // Approaching stars bend toward the green-white side of the flag palette.
        float t = min((shift - 1.0) * 1.5, 1.0);
        color = mix(color, vec3(0.45, 1.0, 0.62), t * 0.6);
      } else {
        // Receding stars bend toward the red side of the flag palette.
        float t = min((1.0 - shift) * 1.5, 1.0);
        color = mix(color, vec3(0.94, 0.10, 0.14), t * 0.6);
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

    float core = exp(-r * r * 6.0);           // tighter core — more point-like
    float halo = exp(-r * r * 2.0) * 0.15;    // subtler halo
    float haloStrength = smoothstep(6.0, 24.0, vPointSize);
    float alpha = core + halo * haloStrength;

    float whiten = smoothstep(20.0, 32.0, vPointSize) * 0.3;
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

// Solar sail (separate near-field scene)
let sailScene, sailCamera, sailMesh, sailMaterial;
let mwSkyMaterial; // Milky Way sky sphere material

// Overlay groups
let gridGroup, armsGroup, ringsGroup, trailGroup, labelsGroup, nebulaGroup;
let trailLine; // the actual Line object inside trailGroup
let trailPositions; // Float32Array backing the trail geometry

let frameCount = 0;
let lastFpsTime = performance.now();
let lastFrameTime = performance.now();
let needsRender = true;  // dirty flag — skip rendering when nothing changed

function markDirty() { needsRender = true; }

// ---------------------------------------------------------------------------
// Procedural star generation
// ---------------------------------------------------------------------------

// Mulberry32 PRNG — fast, deterministic, 32-bit state
function mulberry32(seed) {
  let s = seed | 0;
  return function() {
    s = (s + 0x6D2B79F5) | 0;
    let t = Math.imul(s ^ (s >>> 15), 1 | s);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// B-V color index → [r, g, b] in [0,1] (Tanner Helland / Ballesteros)
function bvToRgb(bv) {
  bv = Math.max(-0.4, Math.min(2.0, bv));
  const temp = 4600.0 * (1.0 / (0.92 * bv + 1.7) + 1.0 / (0.92 * bv + 0.62));
  const t = Math.max(1000.0, Math.min(40000.0, temp));
  const x = t / 100.0;

  let r, g, b;
  if (x <= 66.0) r = 255.0;
  else r = 329.698727446 * Math.pow(x - 60.0, -0.1332047592);

  if (x <= 66.0) g = 99.4708025861 * Math.log(x) - 161.1195681661;
  else g = 288.1221695283 * Math.pow(x - 60.0, -0.0755148492);

  if (x >= 66.0) b = 255.0;
  else if (x <= 19.0) b = 0.0;
  else b = 138.5177312231 * Math.log(x - 10.0) - 305.0447927307;

  return [
    Math.max(0, Math.min(255, r)) / 255,
    Math.max(0, Math.min(255, g)) / 255,
    Math.max(0, Math.min(255, b)) / 255,
  ];
}

function hsvToRgb01(h, s, v) {
  const i = Math.floor(h * 6);
  const f = h * 6 - i;
  const p = v * (1 - s);
  const q = v * (1 - f * s);
  const t = v * (1 - (1 - f) * s);
  switch (i % 6) {
    case 0: return [v, t, p];
    case 1: return [q, v, p];
    case 2: return [p, v, t];
    case 3: return [p, q, v];
    case 4: return [t, p, v];
    default: return [v, p, q];
  }
}

function mixRgb01(a, b, t) {
  return [
    a[0] + (b[0] - a[0]) * t,
    a[1] + (b[1] - a[1]) * t,
    a[2] + (b[2] - a[2]) * t,
  ];
}

function smoothstep01(edge0, edge1, value) {
  const t = Math.max(0, Math.min(1, (value - edge0) / (edge1 - edge0)));
  return t * t * (3 - 2 * t);
}

function flagStarRgbFromWarmth(warmth) {
  // Red, white, black, and green reference Gaza/Palestine.
  const w = Math.max(0, Math.min(1, warmth));
  const green = hsvToRgb01(141 / 360, 0.86, 1.0);
  const white = [1.0, 1.0, 1.0];
  const red = hsvToRgb01(356 / 360, 0.86, 1.0);
  if (w < 0.48) return mixRgb01(green, white, smoothstep01(0.34, 0.48, w));
  if (w > 0.52) return mixRgb01(white, red, smoothstep01(0.52, 0.66, w));
  return white;
}

function bvToFlagRgb(bv) {
  const warmth = (Math.max(-0.4, Math.min(2.0, bv)) + 0.4) / 2.4;
  return flagStarRgbFromWarmth(warmth);
}

function sourceRgbToFlagRgb(r, g, b) {
  const naturalWarmth = Math.max(0, Math.min(1, ((r - b) + 1) * 0.5));
  const direction = Math.sign(naturalWarmth - 0.5);
  const contrast = Math.pow(Math.abs(naturalWarmth - 0.5) * 2, 0.55);
  const warmth = 0.5 + direction * contrast * 0.5;
  return flagStarRgbFromWarmth(warmth);
}

// Galactic density at world-space point (Sol at origin, GC at +x = 26000)
function galacticDensity(x, y, z) {
  // Convert to galactocentric coordinates
  const gx = x - GC_X;
  const gy = y;
  const gz = z;
  const R = Math.sqrt(gx * gx + gy * gy); // cylindrical radius from GC

  // Exponential disk
  let disk = Math.exp(-R / DISK_SCALE_LENGTH) * Math.exp(-Math.abs(gz) / DISK_SCALE_HEIGHT);

  // Spiral arm modulation
  if (R > 200) {
    const theta = Math.atan2(gy, gx); // angle in galactic plane
    // For each arm, find angular distance
    let armFactor = 1.0;
    for (const startAngle of SPIRAL_STARTS) {
      // Invert log-spiral: at radius R, the arm is at angle = ln(R/a)/b + startAngle
      const armTheta = Math.log(R / SPIRAL_A) / SPIRAL_B + startAngle;
      // Angular difference (wrapped to [-pi, pi])
      let dTheta = theta - armTheta;
      dTheta = dTheta - Math.round(dTheta / (2 * Math.PI)) * 2 * Math.PI;
      // Gaussian overdensity
      armFactor += SPIRAL_ARM_STRENGTH * Math.exp(-0.5 * (dTheta / SPIRAL_ARM_WIDTH) * (dTheta / SPIRAL_ARM_WIDTH));
    }
    disk *= armFactor;
  }

  // Central bulge — oblate ellipsoid (flattened 2:1 in z)
  const bulgeR = Math.sqrt(R * R + 4 * gz * gz); // stretch z by 2× so bulge is squashed
  const bulge = 2.0 * Math.exp(-bulgeR / BULGE_SCALE_R);

  return disk + bulge;
}

// Sample absolute magnitude with bias toward luminous stars
function sampleAbsMag(rand) {
  const r = rand();
  if (r < 0.02) return -7 + rand() * 3;        // supergiants: -7 to -4
  if (r < 0.10) return -4 + rand() * 3;         // bright giants: -4 to -1
  if (r < 0.30) return -1 + rand() * 3;         // giants/bright MS: -1 to +2
  if (r < 0.65) return 2 + rand() * 3;          // sun-like: +2 to +5
  return 5 + rand() * 3;                         // K-type MS: +5 to +8
}

// Approximate B-V from absolute magnitude (main-sequence relation + scatter)
function absMagToBV(absMag, rand) {
  const bv = -0.3 + (absMag + 7) * 0.1 + (rand() - 0.5) * 0.2;
  return Math.max(-0.4, Math.min(2.0, bv));
}

function generateProceduralStars(count, seed) {
  const t0 = performance.now();
  const rand = mulberry32(seed);

  // Pre-compute max density for rejection sampling (near galactic center)
  const maxDensity = galacticDensity(GC_X, 0, 0);

  const data = new Float32Array(count * 7);
  let accepted = 0;
  let sampled = 0;
  const cylR = GALAXY_R;
  const cylZ = 3000; // half-height of sampling cylinder

  while (accepted < count) {
    sampled++;
    // Random point in cylinder centered on galactic center
    // Use galactocentric coords, then convert to world (Sol-centered)
    const gr = cylR * Math.sqrt(rand()); // uniform in disk area
    const gtheta = rand() * 2 * Math.PI;
    const gx = gr * Math.cos(gtheta);
    const gy = gr * Math.sin(gtheta);
    const gz = (rand() * 2 - 1) * cylZ;

    // World coords (Sol at origin)
    const wx = gx + GC_X;
    const wy = gy;
    const wz = gz;

    // HYG exclusion zone
    const distSol = Math.sqrt(wx * wx + wy * wy + wz * wz);
    if (distSol < HYG_EXCLUSION_RADIUS) continue;

    // Rejection sampling
    const density = galacticDensity(wx, wy, wz);
    if (rand() > density / maxDensity) continue;

    // Accepted — generate star properties
    const absMag = sampleAbsMag(rand);
    const bv = absMagToBV(absMag, rand);
    const rgb = bvToFlagRgb(bv);

    const base = accepted * 7;
    data[base]     = wx;
    data[base + 1] = wy;
    data[base + 2] = wz;
    data[base + 3] = absMag;
    data[base + 4] = rgb[0];
    data[base + 5] = rgb[1];
    data[base + 6] = rgb[2];
    accepted++;
  }

  const elapsed = (performance.now() - t0).toFixed(0);
  console.log(`[VNP] Generated ${count.toLocaleString()} procedural stars in ${elapsed}ms (${sampled.toLocaleString()} candidates sampled)`);
  return data;
}


// ---------------------------------------------------------------------------
// Milky Way sky sphere shaders
// ---------------------------------------------------------------------------

const mwVertexShader = /* glsl */ `
  varying vec3 vDir;
  void main() {
    vDir = position; // unit sphere direction = world-space view direction
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
  }
`;

const mwFragmentShader = /* glsl */ `
  uniform vec3 uProbePos;
  uniform float uEmission;      // emission coefficient
  uniform float uDustAbs;       // dust absorption coefficient
  uniform float uDiskHeight;    // stellar disk scale height (ly)
  uniform float uDustHeight;    // dust disk base height (ly)
  uniform float uExposure;      // exposure multiplier
  uniform float uNoiseAmp;      // noise amp for emission (glow-to-void boundary)
  uniform float uDustNoiseAmp;  // noise amp for dust (dust-to-glow boundary)
  uniform float uRiftStrength;  // Great Rift midplane dust strength
  uniform float uBulgeStr;      // bulge brightness multiplier
  uniform float uWarpStr;       // galactic disk warp strength
  uniform float uDiskTaper;     // disk edge taper — controls how quickly the stellar disk thins at large radii
  uniform float uDustRadTaper;  // dust radial taper — controls how far dust extends from GC
  uniform float uDustLarge;     // weight of large-scale (~120°) dust structure
  uniform float uDustFine;      // weight of medium+fine dust filaments
  varying vec3 vDir;

  const float PI = 3.14159265;
  const float GC_X_C = ${GC_X.toFixed(1)};
  const float DISK_SL = ${DISK_SCALE_LENGTH.toFixed(1)};
  const float BULGE_SR = ${BULGE_SCALE_R.toFixed(1)};
  const float SPIRAL_A_C = ${SPIRAL_A.toFixed(1)};
  const float SPIRAL_B_C = ${SPIRAL_B.toFixed(4)};

  // --- Noise functions ---
  float hash2(vec2 p) {
    return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453);
  }
  float hash3(vec3 p) {
    return fract(sin(dot(p, vec3(127.1, 311.7, 74.7))) * 43758.5453);
  }

  float noise2(vec2 p) {
    vec2 cell = floor(p);
    vec2 f = fract(p);
    f = f * f * (3.0 - 2.0 * f);
    return mix(mix(hash2(cell), hash2(cell + vec2(1.0, 0.0)), f.x),
               mix(hash2(cell + vec2(0.0, 1.0)), hash2(cell + vec2(1.0, 1.0)), f.x), f.y);
  }

  float noise3(vec3 p) {
    vec3 cell = floor(p);
    vec3 f = fract(p);
    f = f * f * (3.0 - 2.0 * f);
    float n000 = hash3(cell);
    float n100 = hash3(cell + vec3(1,0,0));
    float n010 = hash3(cell + vec3(0,1,0));
    float n110 = hash3(cell + vec3(1,1,0));
    float n001 = hash3(cell + vec3(0,0,1));
    float n101 = hash3(cell + vec3(1,0,1));
    float n011 = hash3(cell + vec3(0,1,1));
    float n111 = hash3(cell + vec3(1,1,1));
    return mix(mix(mix(n000, n100, f.x), mix(n010, n110, f.x), f.y),
               mix(mix(n001, n101, f.x), mix(n011, n111, f.x), f.y), f.z);
  }

  // Domain-warped FBM — feeds noise back into coordinates for filamentary,
  // tendril-like structures (like ink in water). This is the key to getting
  // the ribbon/filament shapes seen in the Gaia all-sky map.
  float fbm3(vec3 p) {
    float v = 0.0;
    v += noise3(p) * 0.5;
    v += noise3(p * 2.0 + 1.7) * 0.25;
    v += noise3(p * 4.0 + 3.3) * 0.125;
    v += noise3(p * 8.0 + 5.1) * 0.0625;
    return v;
  }

  float warpedFbm(vec3 p) {
    // First pass: get base FBM
    float f1 = fbm3(p);
    // Domain warp: offset coordinates by the noise itself
    // This creates filamentary, stretched structures
    vec3 warped = p + vec3(f1 * 1.5, f1 * 1.2, f1 * 0.8);
    // Second pass on warped domain
    return fbm3(warped);
  }

  // Spiral arm factor — unrolled 4 arms
  float armFactor(float R, float theta, float logR, float width) {
    float af = 1.0;
    float w2 = width * width;
    float dT0 = theta - logR;
    dT0 -= floor(dT0 / (2.0 * PI) + 0.5) * 2.0 * PI;
    af += exp(-0.5 * dT0 * dT0 / w2);
    float dT1 = theta - (logR + PI * 0.5);
    dT1 -= floor(dT1 / (2.0 * PI) + 0.5) * 2.0 * PI;
    af += exp(-0.5 * dT1 * dT1 / w2);
    float dT2 = theta - (logR + PI);
    dT2 -= floor(dT2 / (2.0 * PI) + 0.5) * 2.0 * PI;
    af += exp(-0.5 * dT2 * dT2 / w2);
    float dT3 = theta - (logR + PI * 1.5);
    dT3 -= floor(dT3 / (2.0 * PI) + 0.5) * 2.0 * PI;
    af += exp(-0.5 * dT3 * dT3 / w2);
    return af;
  }

  void main() {
    vec3 dir = normalize(vDir);
    vec3 color = vec3(0.0);
    float transmittance = 1.0;
    float maxDist = 80000.0;
    float stepSize = maxDist / 24.0;

    // Jitter to reduce banding
    float jitter = fract(sin(dot(gl_FragCoord.xy, vec2(12.9898, 78.233))) * 43758.5453) * stepSize;

    for (int step = 0; step < 24; step++) {
      float t = (float(step) + 0.5) * stepSize + jitter;
      vec3 sp = uProbePos + dir * t;

      float gx = sp.x - GC_X_C;
      float gy = sp.y;
      float R = length(vec2(gx, gy));
      float theta = atan(gy, gx);

      // --- Galactic warp (smooth onset) ---
      float warpR = smoothstep(8000.0, 30000.0, R);
      float warpOffset = uWarpStr * warpR * warpR * sin(theta) * 3000.0;
      float gz = sp.z - warpOffset;

      float logR = log(max(R, 1.0) / SPIRAL_A_C) / SPIRAL_B_C;

      // --- Large-scale fractal noise for cloud structure ---
      vec3 noiseCoord = vec3(gx, gy, gz * 1.5) / 8000.0;
      float fractal = warpedFbm(noiseCoord);
      float dustFractal = warpedFbm(noiseCoord + vec3(50.0, 23.0, 11.0));

      // Radial noise taper: full noise near GC, fading to zero at disk edge.
      // This makes outer regions converge to a clean, thin disc profile
      // while the central regions stay puffy and irregular.
      float noiseTaper = smoothstep(DISK_SL * 3.5, DISK_SL * 0.5, R);

      // --- Stellar emission density ---
      float baseDensity = exp(-R / DISK_SL);
      if (R > 200.0) {
        baseDensity *= armFactor(R, theta, logR, 0.25);
      }

      // Disk taper: disk gets thinner at large radii, giving lenticular shape.
      // uDiskTaper controls the falloff radius (in DISK_SL units).
      float diskTaper = smoothstep(DISK_SL * uDiskTaper, DISK_SL * uDiskTaper * 0.2, R);
      float effectiveHeight = uDiskHeight * (0.05 + 0.95 * diskTaper);

      // Fractal emission boundary (glow-to-void):
      // Near center: big irregular cloud protrusions.
      // At edges: smooth exponential falloff (thin disc).
      float noiseContrib = fractal * uNoiseAmp * noiseTaper;
      float cloudBoundary = effectiveHeight * (0.1 + noiseContrib);
      // Blend between fractal boundary and smooth exp falloff at edges
      float zFade = mix(
        exp(-abs(gz) / (effectiveHeight * 0.3)),  // smooth thin disc
        smoothstep(cloudBoundary, cloudBoundary * 0.2, abs(gz)),  // fractal boundary
        noiseTaper
      );
      float sd = baseDensity * zFade * diskTaper;

      // Brightness variation (also tapered)
      sd *= (0.3 + fractal * 1.4 * noiseTaper + (1.0 - noiseTaper) * 0.7);

      // Bulge
      float bulgeR = sqrt(R * R + 4.0 * gz * gz);
      sd += uBulgeStr * exp(-bulgeR / BULGE_SR);

      // --- Dust density with fractal filamentary edges ---
      float dustBase = exp(-R / (DISK_SL * 0.7));
      // Radial dust taper: dust confined near GC, fades at uDustRadTaper * DISK_SL
      float dustRadFade = smoothstep(DISK_SL * uDustRadTaper, DISK_SL * uDustRadTaper * 0.2, R);
      dustBase *= dustRadFade;
      if (R > 200.0) {
        dustBase *= armFactor(R, theta, logR, 0.12);
      }

      // Large-scale dust structure: 2D angular noise in (theta, R) space.
      // Creates continent-sized dark patches and tendrils that hold their
      // shape in projection (unlike 3D noise which averages out along rays).
      // Multiple scales: galaxy-wide patches + medium filaments + fine detail.
      vec2 dustAngCoord = vec2(theta * 3.0, R / 15000.0);  // ~120° patches
      float dustStruct = noise2(dustAngCoord + vec2(17.3, 5.7));
      // Medium scale: ~30° features with z-dependent offset for vertical tendrils
      vec2 dustMedCoord = vec2(theta * 12.0, R / 5000.0 + gz / 3000.0);
      float dustMed = noise2(dustMedCoord + vec2(41.0, 13.0));
      // Fine filaments: ~10° features, stronger z coupling for vertical wisps
      vec2 dustFineCoord = vec2(theta * 30.0, R / 2000.0 + gz / 1000.0);
      float dustFine = noise2(dustFineCoord + vec2(73.0, 29.0));
      // Combine: large shapes modulated by medium, with fine detail
      float totalW = uDustLarge + uDustFine + 0.001;
      float dustMask = (dustStruct * uDustLarge + (dustMed * 0.6 + dustFine * 0.4) * uDustFine) / totalW;
      // Sharpen into cloud-like patches: push toward 0 or 1
      dustMask = smoothstep(0.25, 0.65, dustMask);

      // Fractal dust boundary — tapered so outer disk has clean dust lane
      float dustNoiseContrib = dustFractal * uDustNoiseAmp * noiseTaper;
      float dustBoundary = uDustHeight + 400.0 * dustNoiseContrib;
      // Vertical boundary also modulated by structure mask — tendrils reach higher
      dustBoundary *= (0.5 + dustMask * 1.5);
      float dustZFade = smoothstep(dustBoundary, dustBoundary * 0.1, abs(gz));
      float dd = dustBase * dustZFade;

      // Apply structure mask to dust density
      dd *= (0.1 + dustMask * 0.9);

      // Filamentary dust clumping
      dd *= (0.2 + dustFractal * 1.8);

      // Great Rift
      float riftNoise = noise2(vec2(gx, gy) / 4000.0 + vec2(33.0, 7.0));
      float riftDetail = noise2(vec2(gx, gy) / 1500.0 + vec2(71.0, 3.0));
      float riftFractal = riftNoise * 0.6 + riftDetail * 0.4;
      float riftBoundary = 60.0 + 180.0 * riftFractal;
      float riftFade = smoothstep(riftBoundary, riftBoundary * 0.1, abs(gz));
      dd += uRiftStrength * riftFade * exp(-R / (DISK_SL * 1.2));

      // --- Accumulate ---
      // Red, white, black, and green reference Gaza/Palestine.
      // Black comes from the background and dust absorption; emission moves
      // from green through white into red as local stellar density rises.
      float warmth = smoothstep(0.08, 1.15, sd);
      vec3 flagGreen = vec3(0.00, 0.62, 0.24);
      vec3 flagWhite = vec3(1.00, 1.00, 1.00);
      vec3 flagRed = vec3(0.93, 0.08, 0.13);
      vec3 emColor = warmth < 0.5
        ? mix(flagGreen, flagWhite, smoothstep(0.10, 0.50, warmth))
        : mix(flagWhite, flagRed, smoothstep(0.50, 0.90, warmth));
      emColor = mix(emColor, flagRed, smoothstep(1.0, 2.0, sd) * 0.35);

      color += emColor * sd * stepSize * uEmission * transmittance;

      // Dust absorption
      float tau = dd * stepSize * uDustAbs;
      transmittance *= exp(-tau);

      if (transmittance < 0.01) break;
    }

    // Tone mapping
    color *= uExposure;
    color = color / (1.0 + color);
    color = pow(color, vec3(0.85));

    gl_FragColor = vec4(color, 1.0);
  }
`;

// ---------------------------------------------------------------------------
// Solar sail shaders
// ---------------------------------------------------------------------------

const sailVertexShader = /* glsl */ `
  varying vec2 vUv;
  varying vec3 vNormal;
  varying vec3 vViewDir;
  void main() {
    vUv = uv;
    vNormal = normalize(normalMatrix * normal);
    vec4 mvPos = modelViewMatrix * vec4(position, 1.0);
    vViewDir = normalize(-mvPos.xyz);
    gl_Position = projectionMatrix * mvPos;
  }
`;

const sailFragmentShader = /* glsl */ `
  uniform float uDamage;  // 0 = pristine, 1+ = mostly destroyed

  varying vec2 vUv;
  varying vec3 vNormal;
  varying vec3 vViewDir;

  // Hash-based noise
  float hash(vec2 p) {
    return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453123);
  }

  float noise(vec2 p) {
    vec2 i = floor(p);
    vec2 f = fract(p);
    f = f * f * (3.0 - 2.0 * f);
    float a = hash(i);
    float b = hash(i + vec2(1.0, 0.0));
    float c = hash(i + vec2(0.0, 1.0));
    float d = hash(i + vec2(1.0, 1.0));
    return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
  }

  float fbm(vec2 p) {
    float v = 0.0;
    float a = 0.5;
    for (int i = 0; i < 6; i++) {
      v += a * noise(p);
      p *= 2.0;
      a *= 0.5;
    }
    return v;
  }

  void main() {
    // Diamond shape: discard outside |u-0.5| + |v-0.5| > 0.5
    vec2 d = abs(vUv - 0.5);
    float diamond = d.x + d.y;
    if (diamond > 0.5) discard;

    // Structural booms: along diamond edges and diagonals
    float boomWidth = ${SAIL_BOOM_WIDTH.toFixed(4)};
    float edgeDist = abs(diamond - 0.5);                 // distance to diamond edge
    float diagH = abs(vUv.x - 0.5);                      // distance to vertical boom
    float diagV = abs(vUv.y - 0.5);                      // distance to horizontal boom
    bool onFrame = edgeDist < boomWidth || diagH < boomWidth * 0.6 || diagV < boomWidth * 0.6;

    // Damage holes (only in fabric, not frame)
    if (!onFrame && uDamage > 0.0) {
      float n = fbm(vUv * 80.0);               // fine-grained noise
      float n2 = fbm(vUv * 30.0 + 5.0);        // medium-scale variation
      float holeMask = n * 0.6 + n2 * 0.4;
      // Threshold decreases as damage increases
      float threshold = 1.0 - uDamage * 0.95;
      if (holeMask > threshold) discard;
    }

    // Gold color with view-angle-dependent sheen
    vec3 gold = vec3(0.85, 0.65, 0.12);
    vec3 brightGold = vec3(1.0, 0.85, 0.3);

    float fresnel = 1.0 - abs(dot(vNormal, vViewDir));
    fresnel = fresnel * fresnel;

    vec3 color = mix(gold, brightGold, fresnel * 0.5);

    // Frame is slightly darker, more structural
    if (onFrame) {
      color = vec3(0.15, 0.14, 0.13);
    }

    // Subtle variation across the fabric
    float fabric = noise(vUv * 200.0) * 0.1 + 0.9;
    if (!onFrame) color *= fabric;

    // Slight transparency at edges of remaining fabric near holes
    float alpha = 1.0;
    if (!onFrame && uDamage > 0.0) {
      float n = fbm(vUv * 80.0);
      float n2 = fbm(vUv * 30.0 + 5.0);
      float holeMask = n * 0.6 + n2 * 0.4;
      float threshold = 1.0 - uDamage * 0.95;
      float edgeFade = smoothstep(threshold - 0.05, threshold, holeMask);
      alpha = 1.0 - edgeFade * 0.4;
    }

    gl_FragColor = vec4(color, alpha);
  }
`;

// ---------------------------------------------------------------------------
// Init
// ---------------------------------------------------------------------------

async function init() {
  renderContainer = document.getElementById('galaxy-screen') || document.body;
  const status = document.getElementById('hud-stars');
  const setStatus = (msg) => { status.textContent = msg; };

  setStatus('Downloading star catalog...');
  console.log('[VNP] Loading star data...');

  const [binResponse, metaResponse, landmarksResponse] = await Promise.all([
    fetch(`${DATA_DIR}/stars.bin`),
    fetch(`${DATA_DIR}/metadata.json`),
    fetch(`${DATA_DIR}/landmarks.json`),
  ]);

  setStatus('Parsing star data...');
  const binBuffer = await binResponse.arrayBuffer();
  metadata = await metaResponse.json();
  landmarks = await landmarksResponse.json();

  const hygCount = metadata.starCount;
  const floatData = new Float32Array(binBuffer);
  console.log(`[VNP] Loaded ${hygCount.toLocaleString()} HYG stars`);

  // Generate procedural stars
  setStatus(`Generating ${PROCEDURAL_COUNT.toLocaleString()} procedural stars...`);
  await new Promise(r => setTimeout(r, 0)); // yield to render status update
  const procData = generateProceduralStars(PROCEDURAL_COUNT, 42);
  const totalCount = hygCount + PROCEDURAL_COUNT;

  // Allocate combined arrays
  const positions = new Float32Array(totalCount * 3);
  const absMags = new Float32Array(totalCount);
  const colors = new Float32Array(totalCount * 3);

  // Copy HYG stars
  for (let i = 0; i < hygCount; i++) {
    const base = i * 7;
    positions[i * 3]     = floatData[base];
    positions[i * 3 + 1] = floatData[base + 1];
    positions[i * 3 + 2] = floatData[base + 2];
    absMags[i]           = floatData[base + 3];
    const rgb = sourceRgbToFlagRgb(floatData[base + 4], floatData[base + 5], floatData[base + 6]);
    colors[i * 3]        = rgb[0];
    colors[i * 3 + 1]    = rgb[1];
    colors[i * 3 + 2]    = rgb[2];
  }

  // Append procedural stars
  for (let i = 0; i < PROCEDURAL_COUNT; i++) {
    const src = i * 7;
    const dst = hygCount + i;
    positions[dst * 3]     = procData[src];
    positions[dst * 3 + 1] = procData[src + 1];
    positions[dst * 3 + 2] = procData[src + 2];
    absMags[dst]           = procData[src + 3];
    colors[dst * 3]        = procData[src + 4];
    colors[dst * 3 + 1]    = procData[src + 5];
    colors[dst * 3 + 2]    = procData[src + 6];
  }

  console.log(`[VNP] Total: ${totalCount.toLocaleString()} stars (${hygCount.toLocaleString()} HYG + ${PROCEDURAL_COUNT.toLocaleString()} procedural)`);

  // Cache named star positions
  for (const ns of metadata.namedStars.slice(0, 200)) {
    const idx = ns.index;
    namedStarPositions.push({
      name: ns.name,
      pos: new THREE.Vector3(positions[idx * 3], positions[idx * 3 + 1], positions[idx * 3 + 2]),
    });
  }

  setStatus('Initializing renderer...');

  // --- Scene ---
  scene = new THREE.Scene();
  scene.background = new THREE.Color(0x000000);

  const width = renderContainer.clientWidth || window.innerWidth;
  const height = renderContainer.clientHeight || window.innerHeight;
  camera = new THREE.PerspectiveCamera(sim.fov, width / height, 0.001, 500000);
  camera.up.set(0, 0, 1); // Galactic north (+z) is "up" so disk band appears horizontal

  webglRenderer = new THREE.WebGLRenderer({ antialias: false });
  webglRenderer.setSize(width, height);
  webglRenderer.setPixelRatio(window.devicePixelRatio);
  webglRenderer.outputColorSpace = THREE.SRGBColorSpace;
  webglRenderer.domElement.classList.add('galaxy-webgl');
  renderContainer.appendChild(webglRenderer.domElement);

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

  // --- Milky Way background sky sphere ---
  setStatus('Initializing Milky Way background...');
  const mwGeo = new THREE.IcosahedronGeometry(1, 5);
  mwSkyMaterial = new THREE.ShaderMaterial({
    vertexShader: mwVertexShader,
    fragmentShader: mwFragmentShader,
    uniforms: {
      uProbePos:     { value: new THREE.Vector3() },
      uEmission:      { value: 0.0000562 },
      uDustAbs:       { value: 0.00398 },
      uDiskHeight:    { value: 1490.0 },
      uDustHeight:    { value: 90.0 },
      uExposure:      { value: 1.0 },
      uNoiseAmp:      { value: 3.25 },
      uDustNoiseAmp:  { value: 2.45 },
      uRiftStrength:  { value: 0.65 },
      uBulgeStr:      { value: 2.15 },
      uWarpStr:       { value: 1.0 },
      uDiskTaper:     { value: 3.0 },
      uDustRadTaper:  { value: 2.5 },
      uDustLarge:     { value: 0.8 },
      uDustFine:      { value: 0.95 },
    },
    side: THREE.BackSide,
    depthWrite: false,
    depthTest: false,
  });
  const mwMesh = new THREE.Mesh(mwGeo, mwSkyMaterial);
  mwMesh.scale.setScalar(100);
  mwMesh.renderOrder = -1000;
  mwMesh.frustumCulled = false;
  nebulaGroup = new THREE.Group();
  nebulaGroup.add(mwMesh);
  scene.add(nebulaGroup);

  // --- Bloom ---
  composer = new EffectComposer(webglRenderer);
  composer.addPass(new RenderPass(scene, camera));
  composer.addPass(new UnrealBloomPass(
    new THREE.Vector2(width, height),
    BLOOM_STRENGTH, BLOOM_RADIUS, BLOOM_THRESHOLD,
  ));
  composer.addPass(new OutputPass());

  // --- Solar sail (near-field scene) ---
  sailScene = new THREE.Scene();
  sailCamera = new THREE.PerspectiveCamera(sim.fov, width / height, 1, 10000);

  // Diamond sail: a flat plane, diamond shape carved in shader
  const sailGeo = new THREE.PlaneGeometry(SAIL_SIZE, SAIL_SIZE);
  sailMaterial = new THREE.ShaderMaterial({
    vertexShader: sailVertexShader,
    fragmentShader: sailFragmentShader,
    uniforms: {
      uDamage: { value: 0.0 },
    },
    transparent: true,
    side: THREE.DoubleSide,
    depthWrite: true,
  });
  sailMesh = new THREE.Mesh(sailGeo, sailMaterial);
  sailScene.add(sailMesh);

  // Faint ambient + directional light for the sail
  sailScene.add(new THREE.AmbientLight(0xffffff, 0.3));
  const sunLight = new THREE.DirectionalLight(0xffffee, 1.5);
  sunLight.position.set(0, 0, 1); // light from behind (from Sol direction)
  sailScene.add(sunLight);

  // --- OrbitControls (look around from probe position) ---
  controls = new OrbitControls(camera, webglRenderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.05;
  controls.enableZoom = false;   // zoom doesn't make sense for probe POV
  controls.enablePan = false;    // pan doesn't make sense either
  controls.rotateSpeed = 0.4;
  controls.addEventListener('change', markDirty); // re-render when user rotates
  // We'll set the target to be a point ahead of the probe each frame

  // --- CSS2DRenderer for labels ---
  cssRenderer = new CSS2DRenderer();
  cssRenderer.setSize(width, height);
  cssRenderer.domElement.style.position = 'absolute';
  cssRenderer.domElement.style.top = '0';
  cssRenderer.domElement.style.left = '0';
  cssRenderer.domElement.style.pointerEvents = 'none';
  cssRenderer.domElement.classList.add('galaxy-labels');
  renderContainer.appendChild(cssRenderer.domElement);

  // --- Overlays ---
  createOverlays();

  // --- UI ---
  setupUI();

  // Initial state
  selectProbe(0);

  window.addEventListener('resize', onResize);
  console.log('[VNP] Renderer initialized');
  document.getElementById('hud-stars').textContent =
    `${hygCount.toLocaleString()} + ${PROCEDURAL_COUNT.toLocaleString()}`;

  lastFrameTime = performance.now();
  animate();
  window.dispatchEvent(new Event('vnp:ready'));
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
  nebulaGroup.visible = sim.showClouds;
  gridGroup.visible = sim.showGrid;
  armsGroup.visible = sim.showArms;
  ringsGroup.visible = sim.showRings;
  trailGroup.visible = sim.showTrail;
  labelsGroup.visible = sim.showLabels;

  const popup = document.getElementById('minimap-popup');
  popup.classList.toggle('visible', sim.showMap);
}

// ---------------------------------------------------------------------------
// Mini-map drawing
// ---------------------------------------------------------------------------

// Mini-map layout (galaxy constants defined at top of file)
const GALAXY_THICK = 2000; // Disk thickness for edge-on view (ly)
const MAP_RANGE = 70000;   // Canvas shows ±70k ly from galactic center

function drawSpiralArm(ctx, gcCanvasX, gcCanvasY, scale, startAngle) {
  ctx.beginPath();
  let first = true;
  for (let theta = 0; theta < 6 * Math.PI; theta += 0.05) {
    const r = SPIRAL_A * Math.exp(SPIRAL_B * theta);
    if (r > GALAXY_R) break;
    const angle = theta + startAngle;
    // Galaxy coords (centered on GC): convert to canvas
    const gx = r * Math.cos(angle);
    const gy = r * Math.sin(angle);
    const canvasX = gcCanvasX + gx * scale;
    const canvasY = gcCanvasY - gy * scale;
    if (first) { ctx.moveTo(canvasX, canvasY); first = false; }
    else ctx.lineTo(canvasX, canvasY);
  }
  ctx.stroke();
}

function drawMiniMapXY(ctx, w, h) {
  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = '#0a0a10';
  ctx.fillRect(0, 0, w, h);

  // Map is centered on galactic center
  const scale = (w / 2 - 6) / MAP_RANGE;
  // Galactic center at canvas center
  const gcX = w / 2;
  const gcY = h / 2;
  // Sol position on canvas (GC is at +26k ly in world x, so Sol is at -26k from GC)
  const solCanvasX = gcX - GC_X * scale;
  const solCanvasY = gcY;

  // Galactic disk outline
  ctx.strokeStyle = '#222238';
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.arc(gcX, gcY, GALAXY_R * scale, 0, Math.PI * 2);
  ctx.stroke();

  // Spiral arms
  ctx.strokeStyle = '#1e2040';
  ctx.lineWidth = 3;
  ctx.globalAlpha = 0.6;
  for (const startAngle of SPIRAL_STARTS) {
    drawSpiralArm(ctx, gcX, gcY, scale, startAngle);
  }
  ctx.globalAlpha = 1.0;

  // Galactic center marker
  ctx.fillStyle = '#444';
  ctx.beginPath();
  ctx.arc(gcX, gcY, 2, 0, Math.PI * 2);
  ctx.fill();
  ctx.fillStyle = '#333';
  ctx.font = '8px monospace';
  ctx.textAlign = 'center';
  ctx.fillText('GC', gcX, gcY - 5);

  // Sol
  ctx.fillStyle = '#ffffff';
  ctx.beginPath();
  ctx.arc(solCanvasX, solCanvasY, 2.5, 0, Math.PI * 2);
  ctx.fill();
  ctx.fillStyle = '#888';
  ctx.font = '8px monospace';
  ctx.textAlign = 'center';
  ctx.fillText('Sol', solCanvasX, solCanvasY + 10);

  // Probe position (world coords: Sol at origin)
  const probeCanvasX = solCanvasX + probePos.x * scale;
  const probeCanvasY = solCanvasY - probePos.y * scale;
  ctx.fillStyle = '#ff3333';
  ctx.shadowColor = '#ff3333';
  ctx.shadowBlur = 6;
  ctx.beginPath();
  ctx.arc(probeCanvasX, probeCanvasY, 4, 0, Math.PI * 2);
  ctx.fill();
  ctx.shadowBlur = 0;

  // Scale bar: 10k ly
  const barLen = 10000 * scale;
  ctx.strokeStyle = '#444';
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(w - 6 - barLen, h - 10);
  ctx.lineTo(w - 6, h - 10);
  ctx.stroke();
  ctx.fillStyle = '#444';
  ctx.font = '8px monospace';
  ctx.textAlign = 'right';
  ctx.fillText('10k ly', w - 6, h - 14);
}

function drawMiniMapXZ(ctx, w, h) {
  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = '#0a0a10';
  ctx.fillRect(0, 0, w, h);

  const scale = (w / 2 - 6) / MAP_RANGE;
  const gcX = w / 2;
  const midY = h / 2;
  const solCanvasX = gcX - GC_X * scale;

  // Disk edge-on: ellipse centered on GC
  ctx.strokeStyle = '#222238';
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.ellipse(gcX, midY, GALAXY_R * scale, GALAXY_THICK * scale, 0, 0, Math.PI * 2);
  ctx.stroke();

  // Faint filled disk
  ctx.fillStyle = 'rgba(25, 25, 50, 0.3)';
  ctx.beginPath();
  ctx.ellipse(gcX, midY, GALAXY_R * scale, GALAXY_THICK * scale, 0, 0, Math.PI * 2);
  ctx.fill();

  // Galactic center
  ctx.fillStyle = '#444';
  ctx.beginPath();
  ctx.arc(gcX, midY, 2, 0, Math.PI * 2);
  ctx.fill();
  ctx.fillStyle = '#333';
  ctx.font = '8px monospace';
  ctx.textAlign = 'center';
  ctx.fillText('GC', gcX, midY - 5);

  // Sol
  ctx.fillStyle = '#ffffff';
  ctx.beginPath();
  ctx.arc(solCanvasX, midY, 2.5, 0, Math.PI * 2);
  ctx.fill();
  ctx.fillStyle = '#888';
  ctx.font = '8px monospace';
  ctx.textAlign = 'center';
  ctx.fillText('Sol', solCanvasX, midY + 10);

  // Probe (x → horizontal, z → vertical)
  const probeCanvasX = solCanvasX + probePos.x * scale;
  const probeCanvasY = midY - probePos.z * scale;
  ctx.fillStyle = '#ff3333';
  ctx.shadowColor = '#ff3333';
  ctx.shadowBlur = 6;
  ctx.beginPath();
  ctx.arc(probeCanvasX, probeCanvasY, 4, 0, Math.PI * 2);
  ctx.fill();
  ctx.shadowBlur = 0;

  // Scale bar
  const barLen = 10000 * scale;
  ctx.strokeStyle = '#444';
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(w - 6 - barLen, h - 10);
  ctx.lineTo(w - 6, h - 10);
  ctx.stroke();
  ctx.fillStyle = '#444';
  ctx.font = '8px monospace';
  ctx.textAlign = 'right';
  ctx.fillText('10k ly', w - 6, h - 14);
}

function drawMiniMap() {
  if (!sim.showMap) return;

  const xyCanvas = document.getElementById('minimap-xy');
  const xzCanvas = document.getElementById('minimap-xz');

  drawMiniMapXY(xyCanvas.getContext('2d'), xyCanvas.width, xyCanvas.height);
  drawMiniMapXZ(xzCanvas.getContext('2d'), xzCanvas.width, xzCanvas.height);
}

// ---------------------------------------------------------------------------
// Simulation tick
// ---------------------------------------------------------------------------

function simTick(dt) {
  if (!sim.playing) return;
  sim.time += sim.speed * dt;
  sim.time = Math.min(sim.time, SIM_TIME_LIMIT_YEARS);
  if (sim.time >= SIM_TIME_LIMIT_YEARS) sim.playing = false;
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
  if (renderPaused) {
    lastFrameTime = now;
    return;
  }
  const dt = Math.min((now - lastFrameTime) / 1000, 0.1); // seconds, capped
  lastFrameTime = now;

  // Simulation tick marks dirty if time advances
  if (sim.playing) {
    simTick(dt);
    needsRender = true;
  }

  // Controls damping may still be animating after user releases mouse.
  // OrbitControls.update() returns true in newer Three.js when it changed;
  // the 'change' event listener on controls also sets needsRender.
  controls.update();

  // Skip all rendering if nothing changed
  if (!needsRender) {
    // Still count frames for FPS (will show 0 when idle — that's fine)
    frameCount++;
    if (now - lastFpsTime >= 500) {
      document.getElementById('hud-fps').textContent = 'idle';
      frameCount = 0;
      lastFpsTime = now;
    }
    return;
  }
  needsRender = false;

  updateCamera();
  syncOverlayVisibility();
  updateTrail();
  drawMiniMap();

  // Update Milky Way sky sphere to follow camera
  if (nebulaGroup.visible) {
    nebulaGroup.children[0].position.copy(camera.position);
    mwSkyMaterial.uniforms.uProbePos.value.copy(probePos);
  }

  composer.render();

  // Render solar sail on top (near-field, separate scene)
  if (sim.showSail) {
    sailMesh.position.copy(probeVelDir).multiplyScalar(SAIL_DISTANCE);
    sailMesh.lookAt(0, 0, 0);
    sailMesh.rotateOnAxis(new THREE.Vector3(0, 0, 1), Math.PI / 4);

    sailCamera.quaternion.copy(camera.quaternion);
    sailCamera.fov = camera.fov;
    sailCamera.aspect = camera.aspect;
    sailCamera.updateProjectionMatrix();

    sailMaterial.uniforms.uDamage.value = Math.max(0, sim.time / SAIL_DEGRADE_TIME);

    webglRenderer.autoClear = false;
    webglRenderer.clearDepth();
    webglRenderer.render(sailScene, sailCamera);
    webglRenderer.autoClear = true;
  }

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
  return (d * MS_PER_JULIAN_YEAR / 1000).toLocaleString(undefined, { maximumFractionDigits: 0 }) + ' light-seconds';
}

function formatGalacticHeading(direction) {
  const { longitude, latitude } = galacticCoordinates(direction);
  return `l ${longitude.toFixed(1)}°, b ${latitude >= 0 ? '+' : ''}${latitude.toFixed(1)}°`;
}

function probeTargetName(probe) {
  return probe.name.replace(/^[^—]+—\s*/, '');
}

function updateHUD() {
  const probe = PROBES[sim.probeIndex];
  const dist = probePos.length();

  document.getElementById('hud-probe').textContent =
    `${formatGalacticHeading(probe.direction)} · ${probeTargetName(probe)}`;
  document.getElementById('hud-distance').textContent = formatLY(dist);

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

function selectProbe(index, options = {}) {
  const numericIndex = Number(index);
  const requestedIndex = Number.isFinite(numericIndex) ? numericIndex : 0;
  sim.probeIndex = ((requestedIndex % PROBES.length) + PROBES.length) % PROBES.length;
  resetSimTimeToEpoch();
  sim.playing = true;
  const probeSelect = document.getElementById('probe-select');
  if (probeSelect) probeSelect.value = sim.probeIndex;
  document.getElementById('btn-play').innerHTML = '&#9646;&#9646; Pause';
  document.getElementById('btn-play').classList.add('active');

  updateProbeState();

  // Reset camera: position at origin, looking along travel direction
  camera.position.set(0, 0, 0);
  const dir = PROBES[sim.probeIndex].direction;
  controls.target.set(dir[0] * 100, dir[1] * 100, dir[2] * 100);
  controls.update();

  updateHUD();
  markDirty();
  if (options.log) {
    console.log(`[VNP] Selected: ${PROBES[sim.probeIndex].name} @ ${(PROBES[sim.probeIndex].velocity * 100).toFixed(1)}% c`);
  }
}

function selectRandomProbe() {
  if (PROBES.length < 2) {
    selectProbe(0);
    return;
  }
  let nextIndex = sim.probeIndex;
  while (nextIndex === sim.probeIndex) {
    nextIndex = Math.floor(Math.random() * PROBES.length);
  }
  selectProbe(nextIndex);
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
  sel.addEventListener('change', () => selectProbe(parseInt(sel.value), { log: true }));

  // Play / pause
  const btnPlay = document.getElementById('btn-play');
  btnPlay.addEventListener('click', () => {
    sim.playing = !sim.playing;
    if (sim.playing && sim.time >= SIM_TIME_LIMIT_YEARS) {
      resetSimTimeToEpoch();
    }
    btnPlay.innerHTML = sim.playing ? '&#9646;&#9646; Pause' : '&#9654; Play';
    btnPlay.classList.toggle('active', sim.playing);
    markDirty();
  });

  // Speed buttons
  document.querySelectorAll('.speed-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      sim.speed = parseFloat(btn.dataset.speed);
      document.querySelectorAll('.speed-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
    });
  });

  // Time slider
  const timeSlider = document.getElementById('time-slider');
  timeSlider.addEventListener('input', () => {
    sim.time = parseFloat(timeSlider.value);
    markDirty();
    updateHUD();
  });

  // FOV slider
  const fovSlider = document.getElementById('fov-slider');
  fovSlider.addEventListener('input', () => {
    sim.fov = parseInt(fovSlider.value);
    camera.fov = sim.fov;
    camera.updateProjectionMatrix();
    document.getElementById('fov-val').textContent = sim.fov + '°';
    markDirty();
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
      markDirty();
    });
  });

  // Overlay toggles
  const overlayToggles = [
    ['btn-sail', 'showSail'],
    ['btn-clouds', 'showClouds'],
    ['btn-grid', 'showGrid'],
    ['btn-arms', 'showArms'],
    ['btn-rings', 'showRings'],
    ['btn-trail', 'showTrail'],
    ['btn-labels', 'showLabels'],
    ['btn-map', 'showMap'],
  ];
  overlayToggles.forEach(([id, key]) => {
    const btn = document.getElementById(id);
    btn.addEventListener('click', () => {
      sim[key] = !sim[key];
      btn.classList.toggle('active', sim[key]);
      markDirty();
    });
  });

  // MW Tune debug panel toggle
  const btnTune = document.getElementById('btn-mwtune');
  const tunePanel = document.getElementById('mw-debug');
  btnTune.addEventListener('click', () => {
    tunePanel.classList.toggle('visible');
    btnTune.classList.toggle('active');
  });

  // Wire up debug sliders → shader uniforms
  const dbgSliders = [
    { id: 'dbg-emission', uniform: 'uEmission', log: true, suffix: '' },
    { id: 'dbg-dust', uniform: 'uDustAbs', log: true, suffix: '' },
    { id: 'dbg-diskh', uniform: 'uDiskHeight', log: false, suffix: ' ly' },
    { id: 'dbg-dusth', uniform: 'uDustHeight', log: false, suffix: ' ly' },
    { id: 'dbg-exposure', uniform: 'uExposure', log: false, suffix: '' },
    { id: 'dbg-noise', uniform: 'uNoiseAmp', log: false, suffix: '' },
    { id: 'dbg-dustnoise', uniform: 'uDustNoiseAmp', log: false, suffix: '' },
    { id: 'dbg-rift', uniform: 'uRiftStrength', log: false, suffix: '' },
    { id: 'dbg-bulge', uniform: 'uBulgeStr', log: false, suffix: '' },
    { id: 'dbg-dusttaper', uniform: 'uDiskTaper', log: false, suffix: '' },
    { id: 'dbg-dustradtaper', uniform: 'uDustRadTaper', log: false, suffix: '' },
    { id: 'dbg-dustlarge', uniform: 'uDustLarge', log: false, suffix: '' },
    { id: 'dbg-dustfine', uniform: 'uDustFine', log: false, suffix: '' },
    { id: 'dbg-warp', uniform: 'uWarpStr', log: false, suffix: '' },
  ];

  dbgSliders.forEach(({ id, uniform, log, suffix }) => {
    const slider = document.getElementById(id);
    const valSpan = document.getElementById(id + '-val');
    slider.addEventListener('input', () => {
      const raw = parseFloat(slider.value);
      const val = log ? Math.pow(10, raw) : raw;
      mwSkyMaterial.uniforms[uniform].value = val;
      valSpan.textContent = (log ? val.toExponential(2) : val.toFixed(2)) + suffix;
      markDirty();
    });
  });

  // MW Presets
  const mwPresets = {
    'default':  { emission: -4.25, dust: -2.4, diskh: 1490, dusth: 90, exposure: 1.0, noise: 3.25, dustnoise: 2.45, rift: 0.65, bulge: 2.15, dusttaper: 3.0, dustradtaper: 2.5, dustlarge: 0.8, dustfine: 0.95, warp: 1.0, fov: 60 },
    'subtle':   { emission: -4.5, dust: -2.8, diskh: 800, dusth: 50, exposure: 0.8, noise: 1.0, dustnoise: 1.0, rift: 0.15, bulge: 0.5, dusttaper: 2.0, dustradtaper: 1.5, dustlarge: 0.3, dustfine: 0.3, warp: 0.3, fov: 60 },
    'dramatic': { emission: -4.0, dust: -2.6, diskh: 2970, dusth: 55, exposure: 1.5, noise: 1.65, dustnoise: 1.4, rift: 0.42, bulge: 2.7, dusttaper: 2.3, dustradtaper: 0.7, dustlarge: 0.5, dustfine: 0.5, warp: 1.35, fov: 120 },
    'clean':    { emission: -4.25, dust: -2.4, diskh: 1420, dusth: 90, exposure: 1.0, noise: 0.0, dustnoise: 0.0, rift: 0.42, bulge: 1.15, dusttaper: 3.0, dustradtaper: 2.5, dustlarge: 0.0, dustfine: 0.0, warp: 0.0, fov: 60 },
  };

  // Map preset keys → slider IDs
  const presetKeyToSlider = {
    emission: 'dbg-emission', dust: 'dbg-dust', diskh: 'dbg-diskh', dusth: 'dbg-dusth',
    exposure: 'dbg-exposure', noise: 'dbg-noise', dustnoise: 'dbg-dustnoise',
    rift: 'dbg-rift', bulge: 'dbg-bulge', dusttaper: 'dbg-dusttaper', dustradtaper: 'dbg-dustradtaper',
    dustlarge: 'dbg-dustlarge', dustfine: 'dbg-dustfine', warp: 'dbg-warp',
  };

  document.getElementById('dbg-preset').addEventListener('change', (e) => {
    const preset = mwPresets[e.target.value];
    if (!preset) return;
    for (const [key, sliderId] of Object.entries(presetKeyToSlider)) {
      const slider = document.getElementById(sliderId);
      slider.value = preset[key];
      slider.dispatchEvent(new Event('input')); // triggers uniform update + markDirty
    }
    // Apply FOV if preset includes it
    if (preset.fov != null) {
      const fovSlider = document.getElementById('fov-slider');
      fovSlider.value = preset.fov;
      fovSlider.dispatchEvent(new Event('input'));
    }
  });
}

// ---------------------------------------------------------------------------
// Resize
// ---------------------------------------------------------------------------

function onResize() {
  const w = renderContainer?.clientWidth || window.innerWidth;
  const h = renderContainer?.clientHeight || window.innerHeight;
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
  sailCamera.aspect = w / h;
  sailCamera.updateProjectionMatrix();
  webglRenderer.setSize(w, h);
  composer.setSize(w, h);
  cssRenderer.setSize(w, h);
  markDirty();
}

// ---------------------------------------------------------------------------
// Start
// ---------------------------------------------------------------------------

init().catch(err => {
  console.error('[VNP] Init failed:', err);
  document.getElementById('hud-stars').textContent = 'ERROR — see console';
});

window.vnpGalaxy = {
  pause() {
    renderPaused = true;
  },
  resume() {
    renderPaused = false;
    onResize();
    markDirty();
  },
  selectProbe(index) {
    selectProbe(index);
  },
  selectRandomProbe() {
    selectRandomProbe();
  },
  getProbeCount() {
    return PROBES.length;
  },
};
