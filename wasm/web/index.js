import init, {keydown, size, tick} from '../pkg/wrl_wasm.js';
import {Terminal} from './lib.js';

const main = async () => {
  const wasm = await init();
  const [viewX, viewY] = size();
  const terminal = new Terminal();
  await terminal.init(viewX, viewY);

  window.wasmCallbacks = {
    render: (ptr, sx, sy) => {
      const view = new DataView(wasm.memory.buffer, ptr);
      for (let y = 0; y < viewY; y++) {
        for (let x = 0; x < viewX; x++) {
          const width = terminal.draw(x, y, view, 8 * (x + y * sx));
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
