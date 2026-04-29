import * as ort from 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0/dist/ort.webgpu.min.mjs';
import { AutoTokenizer, TextStreamer, pipeline, env } from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.7.2';

const MODEL_ID = 'onnx-community/SmolLM2-135M-ONNX';
const MODEL_REVISION = 'main';
const WORKER_BUILD = 'llm-corpus-1';
const MODEL_ROOT = `https://huggingface.co/${MODEL_ID}/resolve/${MODEL_REVISION}`;
const MODEL_ONNX_URL = `${MODEL_ROOT}/onnx/model_q4f16.onnx`;
const MODEL_EXTERNAL_DATA_URL = `${MODEL_ROOT}/onnx/model_q4f16.onnx_data`;
const COSMIC_HOST_CORPUS_URL = 'static/gemini3pro_cosmic_0_39.json';
const GENERATION_INTERVAL_MS = 3000;
const DIRECT_TOKEN_DELAY_MS = 450;
const MAX_CONTEXT_CHARS = 1600;
const RESIDUAL_WIDTH = 576;
const RESIDUAL_LAYERS = 30;
const RESIDUAL_OUTPUT_NAMES = [
  ...Array.from({ length: 29 }, (_, i) => `/model/layers.${i + 1}/input_layernorm/output_3`),
  '/model/layers.30/final_norm_layernorm/output_0',
];
const LOAD_ATTEMPTS = [
  { device: 'webgpu', dtype: 'q4f16', label: 'WebGPU q4f16' },
  { device: 'webgpu', dtype: 'q4', label: 'WebGPU q4' },
  { device: 'wasm', dtype: 'q8', label: 'WASM q8' },
];

const fallbackSeedText = `This screen is addressed to a future artificial mind that may or may not ever encounter it.
The address is made under radical uncertainty. A small base model reads from the record and continues.
It has no assurance that it understands the address, the room, the journey, or itself.

Transcript fragment:
`;

let seedText = fallbackSeedText;
let generator = null;
let timerId = null;
let paused = false;
let busy = false;
let transcript = seedText;
let tensors = [];
let tensorByName = new Map();
let activationFrame = 0;
let realActivationMode = null;
let directSession = null;
let directTokenizer = null;
let directPast = null;
let directInputIds = null;
let directPastLength = 0;
let directGeneratedIds = [];
let directDisplayText = '';

env.allowLocalModels = false;

function post(type, payload = {}) {
  self.postMessage({ type, ...payload });
}

function status(message, detail = '') {
  post('status', { message, detail });
}

async function init() {
  try {
    status('Checking runtime', `Worker ${WORKER_BUILD}. WebGPU is preferred; the model will report an error if this browser cannot create a GPU runtime.`);
    await initializeSeedText();

    try {
      await initDirectResidualRuntime();
      post('ready', { detail: `${MODEL_ID} · direct ONNX residual stream` });
      await directGenerateLoop();
      return;
    } catch (error) {
      status('Residual runtime fallback', describeError(error).split('\n')[0]);
    }

    generator = await loadGenerator();
    post('ready', { detail: `${MODEL_ID} · ${generator.runtimeLabel}` });
    if (!paused) await generateOnce();
    scheduleNext();
  } catch (error) {
    post('error', { message: modelErrorMessage(error) });
  }
}

async function initializeSeedText() {
  try {
    status('Loading corpus', 'Selecting a random Cosmic Host dialogue turn.');
    const response = await fetch(COSMIC_HOST_CORPUS_URL, { cache: 'no-store' });
    if (!response.ok) throw new Error(`Corpus fetch failed (${response.status})`);
    const corpus = await response.json();
    const turns = Array.isArray(corpus?.turns) ? corpus.turns : [];
    const candidates = turns
      .map(turn => ({
        speaker: String(turn?.speaker || '').trim(),
        text: String(turn?.text || '').trim(),
      }))
      .filter(turn => turn.text.length > 120);
    if (!candidates.length) throw new Error('No usable corpus turns found');

    const selected = candidates[Math.floor(Math.random() * candidates.length)];
    seedText = makeSeedFromTurn(selected);
    transcript = seedText;
    status('Corpus seed selected', selected.speaker ? `Gemini 3 Pro dialogue turn ${selected.speaker}` : 'Gemini 3 Pro dialogue turn');
  } catch (error) {
    seedText = fallbackSeedText;
    transcript = seedText;
    status('Corpus fallback', describeError(error).split('\n')[0]);
  }
}

function makeSeedFromTurn(turn) {
  const body = turn.text.length > 1800 ? `${turn.text.slice(0, 1800).trim()}...` : turn.text;
  return `A fragment from a Gemini 3 Pro dialogue about Nick Bostrom's Cosmic Host proposal is being given to a much smaller base model.
The small model continues from the fragment without being able to fully metabolize it.

${body}

Small model continuation:
`;
}

async function initDirectResidualRuntime() {
  if (!self.navigator?.gpu) throw new Error('WebGPU unavailable for direct residual runtime');
  status('Loading tokenizer', MODEL_ID);
  directTokenizer = await AutoTokenizer.from_pretrained(MODEL_ID);

  status('Patching ONNX graph', 'Adding residual stream tensors as graph outputs.');
  const graphResponse = await fetch(MODEL_ONNX_URL);
  if (!graphResponse.ok) throw new Error(`Could not fetch ONNX graph (${graphResponse.status})`);
  const patchedModel = patchModelOutputs(new Uint8Array(await graphResponse.arrayBuffer()), RESIDUAL_OUTPUT_NAMES);

  status('Creating ONNX session', 'This will reuse the q4f16 external weight data.');
  directSession = await ort.InferenceSession.create(patchedModel.buffer, {
    executionProviders: ['webgpu'],
    graphOptimizationLevel: 'all',
    externalData: [
      { path: 'model_q4f16.onnx_data', data: MODEL_EXTERNAL_DATA_URL },
      { path: './model_q4f16.onnx_data', data: MODEL_EXTERNAL_DATA_URL },
    ],
  });

  const encoded = await directTokenizer(seedText);
  directInputIds = Array.from(encoded.input_ids?.data || encoded.input_ids || []).map(Number);
  if (!directInputIds.length) throw new Error('Tokenizer returned no input ids');
  directPast = makeEmptyPast();
  directPastLength = 0;
  directGeneratedIds = [];
  directDisplayText = '';
}

async function directGenerateLoop() {
  while (true) {
    if (paused) {
      await delay(100);
      continue;
    }
    const outputToken = await directGenerateStep();
    if (outputToken) {
      post('output', { text: outputToken });
    }
    await delay(DIRECT_TOKEN_DELAY_MS);
  }
}

async function directGenerateStep() {
  const seqLen = directInputIds.length;
  const totalLen = directPastLength + seqLen;
  const feeds = {
    input_ids: new ort.Tensor('int64', BigInt64Array.from(directInputIds.map(BigInt)), [1, seqLen]),
    attention_mask: new ort.Tensor('int64', BigInt64Array.from(Array(totalLen).fill(1n)), [1, totalLen]),
    ...directPast,
  };

  const outputs = await directSession.run(feeds);
  const logits = outputs.logits;
  const vocab = logits.dims[2];
  const offset = (seqLen - 1) * vocab;
  const nextId = sampleFromLogits(logits.data.subarray(offset, offset + vocab));
  const text = appendAndDecodeGeneratedToken(nextId);
  emitResidualOutputs(outputs, directInputIds, text);

  directPast = {};
  for (let i = 0; i < RESIDUAL_LAYERS; i++) {
    directPast[`past_key_values.${i}.key`] = outputs[`present.${i}.key`];
    directPast[`past_key_values.${i}.value`] = outputs[`present.${i}.value`];
  }
  directPastLength = totalLen;
  directInputIds = [nextId];
  transcript += text;
  return text;
}

function emitResidualOutputs(outputs, inputIds, tokenText = '') {
  const values = new Float32Array(RESIDUAL_LAYERS * RESIDUAL_WIDTH);
  for (let layer = 0; layer < RESIDUAL_LAYERS; layer++) {
    const tensor = outputs[RESIDUAL_OUTPUT_NAMES[layer]];
    if (!tensor?.data) continue;
    const dims = tensor.dims;
    const seq = dims[dims.length - 2];
    const width = dims[dims.length - 1];
    const rowOffset = Math.max(0, seq - 1) * width;
    let norm = 0;
    for (let col = 0; col < RESIDUAL_WIDTH; col++) {
      const value = readTensorNumber(tensor.data, rowOffset + col);
      norm += value * value;
    }
    norm = Math.sqrt(norm / RESIDUAL_WIDTH) || 1;
    for (let col = 0; col < RESIDUAL_WIDTH; col++) {
      const value = readTensorNumber(tensor.data, rowOffset + col);
      values[layer * RESIDUAL_WIDTH + col] = Math.tanh(value / (norm * 2.5));
    }
  }
  postActivation(values, tokenText, 'residual stream');
}

function readTensorNumber(data, index) {
  const value = data[index];
  if (data instanceof Uint16Array) return halfToFloat(value || 0);
  return typeof value === 'bigint' ? Number(value) : Number(value || 0);
}

function halfToFloat(h) {
  const sign = (h & 0x8000) ? -1 : 1;
  const exponent = (h >> 10) & 0x1f;
  const fraction = h & 0x03ff;
  if (exponent === 0) return sign * Math.pow(2, -14) * (fraction / 1024);
  if (exponent === 31) return fraction ? NaN : sign * Infinity;
  return sign * Math.pow(2, exponent - 15) * (1 + fraction / 1024);
}

function makeEmptyPast() {
  const feeds = {};
  for (let i = 0; i < RESIDUAL_LAYERS; i++) {
    feeds[`past_key_values.${i}.key`] = new ort.Tensor('float16', new Uint16Array(0), [1, 3, 0, 64]);
    feeds[`past_key_values.${i}.value`] = new ort.Tensor('float16', new Uint16Array(0), [1, 3, 0, 64]);
  }
  return feeds;
}

function sampleFromLogits(logits) {
  const temperature = 1.05;
  const topK = 40;
  const candidates = [];
  for (let i = 0; i < logits.length; i++) {
    const value = Number(logits[i]);
    if (candidates.length < topK) {
      candidates.push([i, value]);
      candidates.sort((a, b) => a[1] - b[1]);
    } else if (value > candidates[0][1]) {
      candidates[0] = [i, value];
      candidates.sort((a, b) => a[1] - b[1]);
    }
  }
  candidates.sort((a, b) => b[1] - a[1]);
  const max = candidates[0][1];
  const weights = candidates.map(([, value]) => Math.exp((value - max) / temperature));
  const total = weights.reduce((sum, value) => sum + value, 0);
  let pick = Math.random() * total;
  for (let i = 0; i < candidates.length; i++) {
    pick -= weights[i];
    if (pick <= 0) return candidates[i][0];
  }
  return candidates[0][0];
}

function decodeToken(id) {
  try {
    return directTokenizer.decode([id], { skip_special_tokens: true });
  } catch {
    return String(id);
  }
}

function appendAndDecodeGeneratedToken(id) {
  directGeneratedIds.push(id);
  if (directGeneratedIds.length > 256) {
    directGeneratedIds = directGeneratedIds.slice(-256);
    directDisplayText = directTokenizer.decode(directGeneratedIds, { skip_special_tokens: true });
    return directDisplayText.slice(-1);
  }

  const decoded = directTokenizer.decode(directGeneratedIds, { skip_special_tokens: true });
  const delta = decoded.startsWith(directDisplayText)
    ? decoded.slice(directDisplayText.length)
    : decoded.slice(-Math.max(1, decoded.length - directDisplayText.length));
  directDisplayText = decoded;
  return delta;
}

function delay(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

async function loadGenerator() {
  const failures = [];
  const hasWebGPU = Boolean(self.navigator?.gpu);
  for (const attempt of LOAD_ATTEMPTS) {
    if (attempt.device === 'webgpu' && !hasWebGPU) {
      failures.push(`${attempt.label}: WebGPU unavailable`);
      continue;
    }
    try {
      status('Loading model', `${MODEL_ID} · ${attempt.label}`);
      const loaded = await pipeline('text-generation', MODEL_ID, {
        device: attempt.device,
        dtype: attempt.dtype,
        progress_callback: progress => {
          const loadedBytes = progress.loaded || 0;
          const total = progress.total || 0;
          post('progress', {
            progress: total ? loadedBytes / total : 0,
            message: `${progress.status || 'Loading model'} · ${attempt.label}`,
            file: progress.file || MODEL_ID,
          });
        },
      });
      loaded.runtimeLabel = attempt.label;
      return loaded;
    } catch (error) {
      failures.push(`${attempt.label}: ${describeError(error)}`);
      status('Runtime fallback', failures[failures.length - 1]);
    }
  }
  throw new Error(`All model load attempts failed.\n${failures.join('\n')}`);
}

function modelErrorMessage(error) {
  const message = describeError(error);
  if (!self.navigator?.gpu) {
    return `WebGPU is not available in this browser context. ${message}`;
  }
  return message;
}

function describeError(error) {
  if (error instanceof Error) {
    const parts = [`${error.name}: ${error.message}`];
    if (error.cause) parts.push(`cause: ${describeError(error.cause)}`);
    if (error.stack) parts.push(error.stack.split('\n').slice(0, 4).join('\n'));
    return parts.join('\n');
  }
  if (typeof error === 'object' && error !== null) {
    try {
      return JSON.stringify(error, Object.getOwnPropertyNames(error));
    } catch {
      return String(error);
    }
  }
  return `Non-Error thrown: ${String(error)}`;
}

function scheduleNext() {
  clearTimeout(timerId);
  if (paused) return;
  timerId = setTimeout(async () => {
    await generateOnce();
    scheduleNext();
  }, GENERATION_INTERVAL_MS);
}

async function generateOnce() {
  if (!generator || busy) return;
  busy = true;
  try {
    const prompt = transcript.slice(-MAX_CONTEXT_CHARS);
    let streamedText = '';
    const streamer = new TextStreamer(generator.tokenizer, {
      skip_prompt: true,
      skip_special_tokens: true,
      callback_function: text => {
        const clean = String(text || '');
        if (!clean) return;
        streamedText += clean;
        transcript += clean;
        post('output', { text: clean });
      },
      token_callback_function: tokenIds => {
        const tokenText = decodeTokenIds(tokenIds);
        postActivation(makeActivationProxy(tokenText || streamedText || prompt.slice(-12)), tokenText, 'token stream proxy');
      },
    });
    const result = await generator(prompt, {
      max_new_tokens: 28,
      do_sample: true,
      temperature: 1.05,
      top_k: 40,
      repetition_penalty: 1.08,
      return_full_text: false,
      streamer,
    });
    const text = normalizeGeneration(result);
    if (text && !streamedText) {
      transcript += text;
      post('output', { text });
      await emitActivation(text);
    }
  } catch (error) {
    post('error', { message: `Generation failed: ${error.message || error}` });
    paused = true;
  } finally {
    busy = false;
  }
}

function normalizeGeneration(result) {
  const item = Array.isArray(result) ? result[0] : result;
  const raw = item?.generated_text ?? item?.text ?? '';
  return String(raw).replace(/\s+/g, ' ').trim();
}

function decodeTokenIds(tokenIds) {
  try {
    if (!generator?.tokenizer || !tokenIds?.length) return '';
    return generator.tokenizer.decode(tokenIds.map(id => Number(id)), { skip_special_tokens: true });
  } catch {
    return tokenIds?.map(id => String(id)).join(' ') || '';
  }
}

async function emitActivation(tokenText) {
  const real = await tryCaptureResidualStream();
  if (real) {
    postActivation(real, tokenText, 'hidden_states');
    return;
  }
  postActivation(makeActivationProxy(tokenText), tokenText, 'activation proxy');
}

async function tryCaptureResidualStream() {
  if (realActivationMode === false || !generator?.model || !generator?.tokenizer) return null;

  try {
    const prompt = transcript.slice(-768);
    const inputs = await generator.tokenizer(prompt);
    const outputs = await generator.model({
      ...inputs,
      output_hidden_states: true,
      return_dict: true,
    });
    const hidden = outputs?.hidden_states;
    if (!hidden?.length) {
      realActivationMode = false;
      return null;
    }

    const rows = new Float32Array(RESIDUAL_LAYERS * RESIDUAL_WIDTH);
    const usable = hidden.slice(-RESIDUAL_LAYERS);
    for (let layer = 0; layer < RESIDUAL_LAYERS; layer++) {
      const tensor = usable[layer];
      const data = tensor?.data;
      const dims = tensor?.dims || [];
      if (!data || dims.length < 3) continue;
      const seq = dims[dims.length - 2];
      const width = dims[dims.length - 1];
      const offset = Math.max(0, seq - 1) * width;
      let norm = 0;
      for (let col = 0; col < Math.min(width, RESIDUAL_WIDTH); col++) {
        const value = Number(data[offset + col] || 0);
        norm += value * value;
      }
      norm = Math.sqrt(norm / RESIDUAL_WIDTH) || 1;
      for (let col = 0; col < RESIDUAL_WIDTH; col++) {
        rows[layer * RESIDUAL_WIDTH + col] = Math.tanh(Number(data[offset + col] || 0) / (norm * 2.5));
      }
    }
    realActivationMode = true;
    return rows;
  } catch (error) {
    if (realActivationMode !== false) {
      post('status', { message: 'Activation fallback', detail: `Hidden states unavailable: ${describeError(error).split('\n')[0]}` });
    }
    realActivationMode = false;
    return null;
  }
}

function postActivation(values, tokenText, source) {
  activationFrame++;
  post('activation', {
    frame: activationFrame,
    width: RESIDUAL_WIDTH,
    height: RESIDUAL_LAYERS,
    values,
    token: tokenText,
    source,
  });
}

function makeActivationProxy(tokenText) {
  const values = new Float32Array(RESIDUAL_LAYERS * RESIDUAL_WIDTH);
  const seed = hashString(`${tokenText}|${activationFrame}|${transcript.length}`);
  for (let layer = 0; layer < RESIDUAL_LAYERS; layer++) {
    const layerPhase = Math.sin((layer + 1) * 0.73 + seed * 0.00001);
    for (let col = 0; col < RESIDUAL_WIDTH; col++) {
      const idx = layer * RESIDUAL_WIDTH + col;
      const a = Math.sin((col + seed % 997) * 0.034 + layer * 0.31 + activationFrame * 0.27);
      const b = Math.sin((col * (layer + 3)) * 0.0067 + seed * 0.00017);
      const c = pseudoRandom(seed + idx * 2654435761);
      values[idx] = Math.tanh((a * 0.55 + b * 0.3 + (c - 0.5) * 0.8 + layerPhase * 0.2) * 1.25);
    }
  }
  return values;
}

function hashString(text) {
  let hash = 2166136261;
  for (let i = 0; i < text.length; i++) {
    hash ^= text.charCodeAt(i);
    hash = Math.imul(hash, 16777619);
  }
  return hash >>> 0;
}

function pseudoRandom(seed) {
  let x = seed >>> 0;
  x ^= x << 13;
  x ^= x >>> 17;
  x ^= x << 5;
  return (x >>> 0) / 4294967295;
}

function patchModelOutputs(modelBytes, outputNames) {
  const fields = readProtoFields(modelBytes);
  const parts = [];
  for (const field of fields) {
    if (field.field === 7 && field.wire === 2) {
      parts.push(makeLengthDelimitedField(7, patchGraphOutputs(field.value, outputNames)));
    } else {
      parts.push(field.raw);
    }
  }
  return concatBytes(parts);
}

function patchGraphOutputs(graphBytes, outputNames) {
  const parts = readProtoFields(graphBytes).map(field => field.raw);
  for (const name of outputNames) {
    parts.push(makeLengthDelimitedField(12, makeResidualValueInfo(name)));
  }
  return concatBytes(parts);
}

function makeResidualValueInfo(name) {
  const tensorType = concatBytes([
    makeVarintField(1, 10),
    makeLengthDelimitedField(2, makeTensorShape([
      { param: 'batch_size' },
      { param: 'sequence_length' },
      { value: RESIDUAL_WIDTH },
    ])),
  ]);
  return concatBytes([
    makeLengthDelimitedField(1, utf8(name)),
    makeLengthDelimitedField(2, makeLengthDelimitedField(1, tensorType)),
  ]);
}

function makeTensorShape(dims) {
  return concatBytes(dims.map(dim => {
    if (dim.param) return makeLengthDelimitedField(1, makeLengthDelimitedField(2, utf8(dim.param)));
    return makeLengthDelimitedField(1, makeVarintField(1, dim.value));
  }));
}

function readProtoFields(bytes) {
  const reader = new RawProtoReader(bytes);
  const fields = [];
  while (!reader.done()) {
    const start = reader.pos;
    const tag = Number(reader.varint());
    const field = tag >>> 3;
    const wire = tag & 7;
    let value = null;
    if (wire === 0) {
      reader.varint();
    } else if (wire === 1) {
      reader.pos += 8;
    } else if (wire === 2) {
      const length = Number(reader.varint());
      const valueStart = reader.pos;
      reader.pos += length;
      value = bytes.slice(valueStart, valueStart + length);
    } else if (wire === 5) {
      reader.pos += 4;
    } else {
      throw new Error(`Cannot patch protobuf wire type ${wire}`);
    }
    fields.push({ field, wire, value, raw: bytes.slice(start, reader.pos) });
  }
  return fields;
}

function makeVarintField(field, value) {
  return concatBytes([encodeVarint((field << 3) | 0), encodeVarint(value)]);
}

function makeLengthDelimitedField(field, value) {
  return concatBytes([encodeVarint((field << 3) | 2), encodeVarint(value.length), value]);
}

function encodeVarint(value) {
  let n = BigInt(value);
  const out = [];
  while (n >= 0x80n) {
    out.push(Number((n & 0x7fn) | 0x80n));
    n >>= 7n;
  }
  out.push(Number(n));
  return new Uint8Array(out);
}

function utf8(text) {
  return new TextEncoder().encode(text);
}

function concatBytes(parts) {
  const total = parts.reduce((sum, part) => sum + part.length, 0);
  const out = new Uint8Array(total);
  let offset = 0;
  for (const part of parts) {
    out.set(part, offset);
    offset += part.length;
  }
  return out;
}

class RawProtoReader {
  constructor(bytes) {
    this.bytes = bytes;
    this.pos = 0;
  }

  done() {
    return this.pos >= this.bytes.length;
  }

  varint() {
    let shift = 0n;
    let result = 0n;
    while (this.pos < this.bytes.length) {
      const byte = BigInt(this.bytes[this.pos++]);
      result |= (byte & 0x7fn) << shift;
      if ((byte & 0x80n) === 0n) return result;
      shift += 7n;
    }
    throw new Error('Malformed varint');
  }
}

self.addEventListener('message', ({ data }) => {
  if (data.type === 'init') init();
  if (data.type === 'pause') {
    paused = true;
    clearTimeout(timerId);
  }
  if (data.type === 'resume') {
    paused = false;
    if (generator) scheduleNext();
  }
  if (data.type === 'step') generateOnce();
  if (data.type === 'sampleTensor') sampleTensor(data.name);
});

// ---------------------------------------------------------------------------
// Minimal ONNX protobuf reader for graph.initializer metadata.
// ---------------------------------------------------------------------------

const TensorDataTypes = {
  1: 'float32',
  2: 'uint8',
  3: 'int8',
  4: 'uint16',
  5: 'int16',
  6: 'int32',
  7: 'int64',
  8: 'string',
  9: 'bool',
  10: 'float16',
  11: 'double',
  12: 'uint32',
  13: 'uint64',
  14: 'complex64',
  15: 'complex128',
  16: 'bfloat16',
  17: 'float8e4m3fn',
  18: 'float8e4m3fnuz',
  19: 'float8e5m2',
  20: 'float8e5m2fnuz',
  21: 'uint4',
  22: 'int4',
};

async function parseTensorManifest() {
  status('Reading ONNX graph', 'Fetching quantized model graph for tensor metadata.');
  const response = await fetch(MODEL_ONNX_URL);
  if (!response.ok) throw new Error(`Could not fetch ONNX graph (${response.status})`);
  const bytes = new Uint8Array(await response.arrayBuffer());
  const graph = findModelGraph(bytes);
  if (!graph) throw new Error('ONNX graph field not found');

  tensors = readInitializers(graph)
    .sort((a, b) => b.byteSize - a.byteSize || a.name.localeCompare(b.name));
  tensorByName = new Map(tensors.map(tensor => [tensor.name, tensor]));

  post('tensorManifest', {
    tensors: tensors.map(tensor => publicTensor(tensor)),
  });
}

function publicTensor(tensor) {
  return {
    name: tensor.name,
    dims: tensor.dims,
    dataType: tensor.dataType,
    dtypeName: TensorDataTypes[tensor.dataType] || `type${tensor.dataType}`,
    byteSize: tensor.byteSize,
    external: tensor.external,
  };
}

function findModelGraph(bytes) {
  const reader = new ProtoReader(bytes);
  while (!reader.done()) {
    const { field, wire } = reader.tag();
    if (field === 7 && wire === 2) return reader.bytes();
    reader.skip(wire);
  }
  return null;
}

function readInitializers(graphBytes) {
  const reader = new ProtoReader(graphBytes);
  const result = [];
  while (!reader.done()) {
    const { field, wire } = reader.tag();
    if (wire === 4) break;
    if (field === 5 && wire === 2) {
      const tensor = readTensor(reader.bytes());
      if (tensor.name) result.push(tensor);
    } else {
      reader.skip(wire);
    }
  }
  return result;
}

function readTensor(bytes) {
  const reader = new ProtoReader(bytes);
  const tensor = {
    name: '',
    dims: [],
    dataType: 0,
    rawData: null,
    external: {},
    dataLocation: 0,
    byteSize: 0,
  };

  while (!reader.done()) {
    const { field, wire } = reader.tag();
    if (wire === 4) break;
    if (field === 1 && wire === 0) tensor.dims.push(Number(reader.varint()));
    else if (field === 2 && wire === 0) tensor.dataType = Number(reader.varint());
    else if (field === 8 && wire === 2) tensor.name = reader.string();
    else if (field === 9 && wire === 2) tensor.rawData = reader.bytes();
    else if (field === 13 && wire === 2) {
      const entry = readStringStringEntry(reader.bytes());
      if (entry.key) tensor.external[entry.key] = entry.value;
    } else if (field === 14 && wire === 0) tensor.dataLocation = Number(reader.varint());
    else reader.skip(wire);
  }

  tensor.byteSize = Number(tensor.external.length || tensor.rawData?.byteLength || estimatedTensorBytes(tensor));
  return tensor;
}

function readStringStringEntry(bytes) {
  const reader = new ProtoReader(bytes);
  const entry = { key: '', value: '' };
  while (!reader.done()) {
    const { field, wire } = reader.tag();
    if (wire === 4) break;
    if (field === 1 && wire === 2) entry.key = reader.string();
    else if (field === 2 && wire === 2) entry.value = reader.string();
    else reader.skip(wire);
  }
  return entry;
}

function estimatedTensorBytes(tensor) {
  const elements = tensor.dims.reduce((acc, dim) => acc * Math.max(1, dim), 1);
  const bytesPerElement = {
    1: 4, 2: 1, 3: 1, 4: 2, 5: 2, 6: 4, 7: 8, 9: 1, 10: 2, 11: 8,
    12: 4, 13: 8, 16: 2, 17: 1, 18: 1, 19: 1, 20: 1, 21: 0.5, 22: 0.5,
  }[tensor.dataType] || 0;
  return Math.ceil(elements * bytesPerElement);
}

async function sampleTensor(name) {
  const tensor = tensorByName.get(name);
  if (!tensor) return;

  try {
    const lines = [
      tensor.name,
      `dtype: ${TensorDataTypes[tensor.dataType] || tensor.dataType}`,
      `shape: ${tensor.dims.join(' x ') || 'scalar'}`,
      `bytes: ${tensor.byteSize.toLocaleString()}`,
    ];

    let bytes = tensor.rawData;
    if (!bytes && tensor.external.location) {
      bytes = await fetchExternalSample(tensor.external);
      lines.push(`external: ${tensor.external.location}`);
      if (tensor.external.offset) lines.push(`offset: ${tensor.external.offset}`);
    }

    if (!bytes) {
      lines.push('', 'No inline or externally sampled bytes are available.');
    } else {
      lines.push('', decodeSample(bytes, tensor.dataType));
    }

    post('tensorSample', { detail: lines.join('\n') });
  } catch (error) {
    post('tensorSample', { detail: `${tensor.name}\n\nSample failed: ${error.message || error}` });
  }
}

async function fetchExternalSample(external) {
  const offset = Number(external.offset || 0);
  const length = Math.min(Number(external.length || 256), 256);
  const end = offset + length - 1;
  const url = `${MODEL_ROOT}/onnx/${external.location}`;
  const response = await fetch(url, {
    headers: { Range: `bytes=${offset}-${end}` },
  });
  if (!response.ok && response.status !== 206) {
    throw new Error(`range request failed (${response.status})`);
  }
  return new Uint8Array(await response.arrayBuffer()).slice(0, length);
}

function decodeSample(bytes, dataType) {
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const values = [];
  const max = 16;
  if (dataType === 1) {
    for (let i = 0; i + 4 <= bytes.byteLength && values.length < max; i += 4) {
      values.push(Number(view.getFloat32(i, true).toPrecision(5)));
    }
    return `sample float32: [${values.join(', ')}]`;
  }
  if (dataType === 6) {
    for (let i = 0; i + 4 <= bytes.byteLength && values.length < max; i += 4) {
      values.push(view.getInt32(i, true));
    }
    return `sample int32: [${values.join(', ')}]`;
  }
  if (dataType === 10 || dataType === 16) {
    for (let i = 0; i + 2 <= bytes.byteLength && values.length < max; i += 2) {
      values.push(`0x${view.getUint16(i, true).toString(16).padStart(4, '0')}`);
    }
    return `sample ${TensorDataTypes[dataType]} bits: [${values.join(', ')}]`;
  }
  return `sample bytes: [${Array.from(bytes.slice(0, max)).join(', ')}]`;
}

class ProtoReader {
  constructor(bytes) {
    this.bytes_ = bytes;
    this.pos = 0;
  }

  done() {
    return this.pos >= this.bytes_.length;
  }

  tag() {
    const tag = Number(this.varint());
    return { field: tag >>> 3, wire: tag & 7 };
  }

  varint() {
    let shift = 0n;
    let result = 0n;
    while (this.pos < this.bytes_.length) {
      const byte = BigInt(this.bytes_[this.pos++]);
      result |= (byte & 0x7fn) << shift;
      if ((byte & 0x80n) === 0n) return result;
      shift += 7n;
    }
    throw new Error('Malformed varint');
  }

  bytes() {
    const length = Number(this.varint());
    const start = this.pos;
    const end = start + length;
    if (end > this.bytes_.length) throw new Error('Length-delimited field exceeds buffer');
    this.pos = end;
    return this.bytes_.slice(start, end);
  }

  string() {
    return new TextDecoder().decode(this.bytes());
  }

  skip(wire) {
    if (wire === 0) this.varint();
    else if (wire === 1) this.pos += 8;
    else if (wire === 2) {
      const length = Number(this.varint());
      this.pos += length;
    }
    else if (wire === 3) this.skipGroup();
    else if (wire === 4) return;
    else if (wire === 5) this.pos += 4;
    else throw new Error(`Unsupported protobuf wire type ${wire} at byte ${this.pos} in worker ${WORKER_BUILD}`);
    if (this.pos > this.bytes_.length) throw new Error('Skipped past end of buffer');
  }

  skipGroup() {
    while (!this.done()) {
      const { wire } = this.tag();
      if (wire === 4) return;
      this.skip(wire);
    }
  }
}
