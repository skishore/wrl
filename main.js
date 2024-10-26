import init, {tick} from './pkg/wrl.js';

const run = async () => {
  const wasm = await init();
  window.wasmCallbacks = {
    render: (ptr, sx, sy) => {
      const buffer = new Uint32Array(wasm.memory.buffer, ptr, sx * sy);
      const chars = [];
      for (let y = 1; y < sy - 2; y++) {
        for (let x = 1; x < sx - 1; x += 2) {
          chars.push(String.fromCharCode((buffer[x + y * sx] & 0xffff) - 0xff00 + 0x20));
        }
        chars.push('\n');
      }
      console.log(chars.join(''));
    },
  };
  tick();
};

run();
