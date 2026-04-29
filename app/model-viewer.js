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

  const ZERO_COLOR = [7, 8, 14];
  const NEGATIVE_MID = [115, 22, 20];
  const NEGATIVE_EXTREME = [255, 74, 34];
  const POSITIVE_MID = [42, 104, 206];
  const POSITIVE_EXTREME = [242, 250, 255];

  function mixColor(a, b, t) {
    return [
      a[0] + (b[0] - a[0]) * t,
      a[1] + (b[1] - a[1]) * t,
      a[2] + (b[2] - a[2]) * t,
    ];
  }

  function activationColor(value) {
    const v = Math.max(-1, Math.min(1, value || 0));
    const mag = Math.pow(Math.min(1, Math.abs(v)), 0.72);
    const midPoint = 0.55;
    if (v < 0) {
      return mag < midPoint
        ? mixColor(ZERO_COLOR, NEGATIVE_MID, mag / midPoint)
        : mixColor(NEGATIVE_MID, NEGATIVE_EXTREME, (mag - midPoint) / (1 - midPoint));
    }
    return mag < midPoint
      ? mixColor(ZERO_COLOR, POSITIVE_MID, mag / midPoint)
      : mixColor(POSITIVE_MID, POSITIVE_EXTREME, (mag - midPoint) / (1 - midPoint));
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
