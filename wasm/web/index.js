import init, {keydown, size, tick} from '../pkg/wrl_wasm.js';
import {Reader, Terminal} from './lib.js';

const main = async () => {
  const wasm = await init();
  const [viewX, viewY] = size();
  const terminal = new Terminal();
  await terminal.init(viewX, viewY);
  document.body.appendChild(terminal.app.canvas);

  window.wasmCallbacks = {
    render: (ptr, sx, sy) => {
      const reader = new Reader(wasm.memory.buffer, ptr);
      for (let y = 0; y < viewY; y++) {
        for (let x = 0; x < viewX; x++) {
          const width = terminal.draw(x, y, reader);
          reader.consume(8 * (width - 1));
          x += width - 1;
        }
      }
    },
  };
  window.onkeydown = x => {
    const code = x.key.length === 1 ? x.key.charCodeAt(0) : x.keyCode;
    keydown(code, x.shiftKey);
    if (x.keyCode !== 9) return;
    x.preventDefault();
    x.stopPropagation();
  }
  terminal.app.ticker.add(() => { tick(); });
};

main();
