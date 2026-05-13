const BUILD_ID = 'llm-corpus-1';

export function mountModelViewer(root, options = {}) {
  const canvas = root.querySelector('#activation-canvas');
  const ctx = canvas.getContext('2d');
  const runtime = root.querySelector('#runtime');
  const frame = root.querySelector('#frame');
  const token = root.querySelector('#token');
  const source = root.querySelector('#source');
  const error = root.querySelector('#error');
  const context = root.querySelector('#context-content');
  const latestToken = root.querySelector('#latest-token');

  let tokenPieces = [];
  const worker = new Worker(`model-worker.js?v=${BUILD_ID}`, { type: 'module' });

  // Red, white, black, and green reference Gaza/Palestine.
  const ZERO_COLOR = [3, 6, 5];
  const FLAG_RED_HUE = 356 / 360;
  const FLAG_GREEN_HUE = 141 / 360;

  function hsvToRgb(h, s, v) {
    const i = Math.floor(h * 6);
    const f = h * 6 - i;
    const p = v * (1 - s);
    const q = v * (1 - f * s);
    const t = v * (1 - (1 - f) * s);
    let r, g, b;
    switch (i % 6) {
      case 0: r = v; g = t; b = p; break;
      case 1: r = q; g = v; b = p; break;
      case 2: r = p; g = v; b = t; break;
      case 3: r = p; g = q; b = v; break;
      case 4: r = t; g = p; b = v; break;
      default: r = v; g = p; b = q; break;
    }
    return [r * 255, g * 255, b * 255];
  }

  function smoothstep(edge0, edge1, value) {
    const t = Math.max(0, Math.min(1, (value - edge0) / (edge1 - edge0)));
    return t * t * (3 - 2 * t);
  }

  function activationColor(value) {
    const v = Math.max(-1, Math.min(1, value || 0));
    const mag = Math.pow(Math.min(1, Math.abs(v)), 0.72);
    if (mag < 0.012) return ZERO_COLOR;

    const hue = v < 0 ? FLAG_RED_HUE : FLAG_GREEN_HUE;
    const whiteBloom = smoothstep(0.58, 0.98, mag);
    const saturation = 0.90 - whiteBloom * 0.86;
    const brightness = 0.08 + mag * 0.92;
    return hsvToRgb(hue, saturation, brightness);
  }

  function drawActivation(values, width, height) {
    if (canvas.width !== width || canvas.height !== height) {
      canvas.width = width;
      canvas.height = height;
    }

    const image = ctx.createImageData(width, height);
    for (let i = 0; i < width * height; i++) {
      const color = activationColor(values[i]);
      const base = i * 4;
      image.data[base] = color[0];
      image.data[base + 1] = color[1];
      image.data[base + 2] = color[2];
      image.data[base + 3] = 255;
    }
    ctx.putImageData(image, 0, 0);
  }

  function appendGeneratedToken(text) {
    tokenPieces.push(text);
    if (tokenPieces.length > 240) tokenPieces = tokenPieces.slice(-240);

    const priorText = tokenPieces.slice(0, -1).join('').replace(/\s+/g, ' ');
    context.textContent = priorText.slice(-900);
    latestToken.textContent = text || ' ';
  }

  worker.addEventListener('message', ({ data }) => {
    if (data.type === 'status') {
      runtime.textContent = data.detail || data.message;
      options.onStatus?.(data.detail || data.message || 'Loading model');
    }
    if (data.type === 'progress') {
      runtime.textContent = `${data.message || 'loading'} ${Math.round(data.progress * 100)}%`;
      options.onProgress?.({
        message: data.message || 'Loading model',
        progress: data.progress || 0,
      });
    }
    if (data.type === 'ready') {
      runtime.textContent = data.detail;
      options.onReady?.(data.detail);
    }
    if (data.type === 'output') {
      appendGeneratedToken(data.text);
      token.textContent = data.text.slice(0, 48) || ' ';
    }
    if (data.type === 'activation') {
      drawActivation(data.values, data.width, data.height);
      frame.textContent = data.frame;
      source.textContent = data.source;
      if (data.token) token.textContent = data.token.slice(0, 48);
    }
    if (data.type === 'error') {
      error.textContent = data.message;
      options.onError?.(data.message);
    }
  });
  worker.addEventListener('error', event => {
    const message = event.message || 'Model worker failed before it could report status';
    error.textContent = message;
    options.onError?.(message);
  });
  worker.addEventListener('messageerror', () => {
    const message = 'Model worker sent an unreadable message';
    error.textContent = message;
    options.onError?.(message);
  });

  worker.postMessage({ type: 'init' });
  worker.postMessage({ type: 'pause' });

  return {
    pause() {
      worker.postMessage({ type: 'pause' });
    },
    resume() {
      worker.postMessage({ type: 'resume' });
    },
  };
}
