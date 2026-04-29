const BUILD_ID = 'llm-splash-1';

export function mountModelViewer(root, options = {}) {
  const canvas = root.querySelector('#activation-canvas');
  const ctx = canvas.getContext('2d');
  const runtime = root.querySelector('#runtime');
  const frame = root.querySelector('#frame');
  const token = root.querySelector('#token');
  const source = root.querySelector('#source');
  const error = root.querySelector('#error');
  const ticker = root.querySelector('#ticker-content');

  let tokenPieces = [];
  const worker = new Worker(`model-worker.js?v=${BUILD_ID}`, { type: 'module' });

  function drawActivation(values, width, height) {
    if (canvas.width !== width || canvas.height !== height) {
      canvas.width = width;
      canvas.height = height;
    }

    const image = ctx.createImageData(width, height);
    for (let i = 0; i < width * height; i++) {
      const v = Math.max(-1, Math.min(1, values[i] || 0));
      const mag = Math.min(1, Math.abs(v));
      const base = i * 4;
      if (v >= 0) {
        image.data[base] = 40 + mag * 215;
        image.data[base + 1] = 44 + mag * 155;
        image.data[base + 2] = 58 + mag * 80;
      } else {
        image.data[base] = 34 + mag * 60;
        image.data[base + 1] = 54 + mag * 150;
        image.data[base + 2] = 72 + mag * 183;
      }
      image.data[base + 3] = 255;
    }
    ctx.putImageData(image, 0, 0);
  }

  function appendTicker(text) {
    tokenPieces.push(text);
    if (tokenPieces.length > 140) tokenPieces = tokenPieces.slice(-140);

    const visible = tokenPieces.slice();
    while (visible.join('').length > 260 && visible.length > 8) visible.shift();

    ticker.replaceChildren(...visible.map((piece, index) => {
      const span = document.createElement('span');
      const age = visible.length - 1 - index;
      span.className = 'token' +
        (age === 0 ? ' latest' : age === 1 ? ' prev1' : age === 2 ? ' prev2' : '');
      span.textContent = piece;
      return span;
    }));
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
      appendTicker(data.text);
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
