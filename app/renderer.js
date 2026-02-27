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
const SAIL_BOOM_WIDTH = 0.008;    // fraction of half-span for boom/frame thickness

// Nebula gas clouds
const NEBULA_COUNT = 4000;
const NEBULA_SCALE_HEIGHT = 150;    // ly — gas is thinner than stellar disk
const NEBULA_BASE_SIZE = 200000.0;  // world-space — must be huge so blobs overlap into smooth band
const NEBULA_MIN_SIZE = 8.0;        // px — keep distant clouds visible
const NEBULA_MAX_SIZE = 512.0;      // px
const NEBULA_EXCLUSION_RADIUS = 50; // ly from Sol — Milky Way is visible from Earth

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
  // Overlay toggles
  showSail: true,     // solar sail ahead of probe
  showClouds: true,   // on by default — core to Milky Way look
  showGrid: false,
  showArms: false,
  showRings: false,
  showTrail: false,
  showLabels: false,
  showMap: false,
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

// Overlay groups
let gridGroup, armsGroup, ringsGroup, trailGroup, labelsGroup, nebulaGroup;
let trailLine; // the actual Line object inside trailGroup
let trailPositions; // Float32Array backing the trail geometry

let frameCount = 0;
let lastFpsTime = performance.now();
let lastFrameTime = performance.now();

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
    const rgb = bvToRgb(bv);

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
// Nebula cloud generation
// ---------------------------------------------------------------------------

function generateNebulaClouds(count, seed) {
  const t0 = performance.now();
  const rand = mulberry32(seed);

  const maxDensity = galacticDensity(GC_X, 0, 0);

  // 8 floats per cloud: x, y, z, size, r, g, b, opacity
  const data = new Float32Array(count * 8);
  let accepted = 0;
  const cylR = GALAXY_R;
  const cylZ = NEBULA_SCALE_HEIGHT * 10; // sampling half-height

  while (accepted < count) {
    const gr = cylR * Math.sqrt(rand());
    const gtheta = rand() * 2 * Math.PI;
    const gx = gr * Math.cos(gtheta);
    const gy = gr * Math.sin(gtheta);
    const gz = (rand() * 2 - 1) * cylZ;

    // World coords (Sol at origin)
    const wx = gx + GC_X;
    const wy = gy;
    const wz = gz;

    // Skip near Sol
    const distSol = Math.sqrt(wx * wx + wy * wy + wz * wz);
    if (distSol < NEBULA_EXCLUSION_RADIUS) continue;

    // Skip bulge region (no diffuse gas clouds near GC)
    const R_gc = Math.sqrt(gx * gx + gy * gy);
    if (R_gc < 3000) continue;

    // Use tighter z scale height for gas
    const density = galacticDensity(wx, wy, wz) *
      Math.exp(-Math.abs(wz) / NEBULA_SCALE_HEIGHT) /
      Math.exp(-Math.abs(wz) / DISK_SCALE_HEIGHT);
    if (rand() > density / maxDensity) continue;

    // Color palette: 60% blue-white, 25% warm white, 15% pink
    const colorRoll = rand();
    let r, g, b;
    if (colorRoll < 0.60) {
      r = 0.5; g = 0.6; b = 1.0;
    } else if (colorRoll < 0.85) {
      r = 0.8; g = 0.75; b = 0.7;
    } else {
      r = 1.0; g = 0.4; b = 0.5;
    }

    const size = NEBULA_BASE_SIZE * (0.5 + rand() * 1.5);
    const opacity = 0.03 + rand() * 0.06;

    const base = accepted * 8;
    data[base]     = wx;
    data[base + 1] = wy;
    data[base + 2] = wz;
    data[base + 3] = size;
    data[base + 4] = r;
    data[base + 5] = g;
    data[base + 6] = b;
    data[base + 7] = opacity;
    accepted++;
  }

  const elapsed = (performance.now() - t0).toFixed(0);
  console.log(`[VNP] Generated ${count} nebula clouds in ${elapsed}ms`);
  return data;
}

// ---------------------------------------------------------------------------
// Nebula shaders
// ---------------------------------------------------------------------------

const nebulaVertexShader = /* glsl */ `
  uniform float uPixelRatio;
  uniform float uMinSize;
  uniform float uMaxSize;
  attribute float aSize;
  attribute vec3  aColor;
  attribute float aOpacity;
  varying vec3  vColor;
  varying float vOpacity;
  void main() {
    vec4 mvPos = modelViewMatrix * vec4(position, 1.0);
    float dist = length(mvPos.xyz);
    float size = aSize * uPixelRatio / max(dist, 1.0);
    gl_PointSize = clamp(size, uMinSize, uMaxSize);
    vColor = aColor;
    vOpacity = aOpacity * smoothstep(uMaxSize, uMinSize, size);
    gl_Position = projectionMatrix * mvPos;
  }
`;

const nebulaFragmentShader = /* glsl */ `
  varying vec3  vColor;
  varying float vOpacity;
  void main() {
    vec2 c = gl_PointCoord - 0.5;
    float r = length(c) * 2.0;
    float alpha = exp(-r * r * 1.5);  // soft Gaussian
    float intensity = alpha * vOpacity;
    gl_FragColor = vec4(vColor * intensity, intensity);
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
      color = vec3(0.35, 0.30, 0.20);
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
    colors[i * 3]        = floatData[base + 4];
    colors[i * 3 + 1]    = floatData[base + 5];
    colors[i * 3 + 2]    = floatData[base + 6];
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

  camera = new THREE.PerspectiveCamera(sim.fov, window.innerWidth / window.innerHeight, 0.001, 500000);
  camera.up.set(0, 0, 1); // Galactic north (+z) is "up" so disk band appears horizontal

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

  // --- Nebula clouds ---
  setStatus('Generating nebula clouds...');
  await new Promise(r => setTimeout(r, 0));
  const nebulaData = generateNebulaClouds(NEBULA_COUNT, 123);
  const nebulaPositions = new Float32Array(NEBULA_COUNT * 3);
  const nebulaSizes = new Float32Array(NEBULA_COUNT);
  const nebulaColors = new Float32Array(NEBULA_COUNT * 3);
  const nebulaOpacities = new Float32Array(NEBULA_COUNT);

  for (let i = 0; i < NEBULA_COUNT; i++) {
    const src = i * 8;
    nebulaPositions[i * 3]     = nebulaData[src];
    nebulaPositions[i * 3 + 1] = nebulaData[src + 1];
    nebulaPositions[i * 3 + 2] = nebulaData[src + 2];
    nebulaSizes[i]             = nebulaData[src + 3];
    nebulaColors[i * 3]        = nebulaData[src + 4];
    nebulaColors[i * 3 + 1]    = nebulaData[src + 5];
    nebulaColors[i * 3 + 2]    = nebulaData[src + 6];
    nebulaOpacities[i]         = nebulaData[src + 7];
  }

  const nebulaGeo = new THREE.BufferGeometry();
  nebulaGeo.setAttribute('position', new THREE.BufferAttribute(nebulaPositions, 3));
  nebulaGeo.setAttribute('aSize', new THREE.BufferAttribute(nebulaSizes, 1));
  nebulaGeo.setAttribute('aColor', new THREE.BufferAttribute(nebulaColors, 3));
  nebulaGeo.setAttribute('aOpacity', new THREE.BufferAttribute(nebulaOpacities, 1));

  const nebulaMaterial = new THREE.ShaderMaterial({
    vertexShader: nebulaVertexShader,
    fragmentShader: nebulaFragmentShader,
    uniforms: {
      uPixelRatio: { value: webglRenderer.getPixelRatio() },
      uMinSize:    { value: NEBULA_MIN_SIZE },
      uMaxSize:    { value: NEBULA_MAX_SIZE },
    },
    transparent: true,
    depthWrite: false,
    blending: THREE.AdditiveBlending,
  });

  nebulaGroup = new THREE.Group();
  nebulaGroup.add(new THREE.Points(nebulaGeo, nebulaMaterial));
  scene.add(nebulaGroup);

  // --- Bloom ---
  composer = new EffectComposer(webglRenderer);
  composer.addPass(new RenderPass(scene, camera));
  composer.addPass(new UnrealBloomPass(
    new THREE.Vector2(window.innerWidth, window.innerHeight),
    BLOOM_STRENGTH, BLOOM_RADIUS, BLOOM_THRESHOLD,
  ));
  composer.addPass(new OutputPass());

  // --- Solar sail (near-field scene) ---
  sailScene = new THREE.Scene();
  sailCamera = new THREE.PerspectiveCamera(sim.fov, window.innerWidth / window.innerHeight, 1, 10000);

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
  document.getElementById('hud-stars').textContent =
    `${hygCount.toLocaleString()} + ${PROCEDURAL_COUNT.toLocaleString()}`;

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
  drawMiniMap();
  controls.update();
  composer.render();

  // Render solar sail on top (near-field, separate scene)
  if (sim.showSail) {
    // Position sail along probe velocity direction, 1km ahead
    sailMesh.position.copy(probeVelDir).multiplyScalar(SAIL_DISTANCE);
    // Sail faces back toward camera (probe)
    sailMesh.lookAt(0, 0, 0);
    // Rotate 45° to make it a diamond orientation
    sailMesh.rotateOnAxis(new THREE.Vector3(0, 0, 1), Math.PI / 4);

    // Sync sail camera with main camera orientation
    sailCamera.quaternion.copy(camera.quaternion);
    sailCamera.fov = camera.fov;
    sailCamera.aspect = camera.aspect;
    sailCamera.updateProjectionMatrix();

    // Damage accumulation
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
      sim.speed = parseFloat(btn.dataset.speed);
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
  sailCamera.aspect = w / h;
  sailCamera.updateProjectionMatrix();
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
