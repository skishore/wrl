import init, {keydown, tick} from './pkg/wrl.js';

const main = async () => {
  const unit = 8;
  const view = 43;
  const app = new PIXI.Application();
  await app.init({width: unit * view, height: unit * view})
  app.renderer.background.color = 0x0;
  document.body.appendChild(app.canvas);

  await PIXI.Assets.load('images/aquarius_8x8.png');
  const font = PIXI.Texture.from('images/aquarius_8x8.png');

  const frame = (source, x, y) => {
    const frame = new PIXI.Rectangle(x * unit, y * unit, unit, unit);
    return new PIXI.Texture({source, frame});
  };

  const ascii = [];
  for (let i = 0; i < 256; i++) {
    ascii.push(frame(font, i & 0xf, i >> 4));
  }

  const tints = [];
  const grayScale = 17;
  const tintScale = 100;
  for (let i = 0; i < 16; i++) {
    tints.push(0);
  }
  for (let i = 0; i < 216; i++) {
    const b = Math.min(255, tintScale * (i % 6));
    const g = Math.min(255, tintScale * (((i / 6) | 0) % 6));
    const r = Math.min(255, tintScale * (((i / 36) | 0) % 6));
    tints.push((r << 16) | (g << 8) | b);
  }
  for (let i = 0; i < 24; i++) {
    const g = Math.min(255, grayScale * i);
    tints.push((g << 16) | (g << 8) | g);
  }

  const map = [];
  for (let y = 0; y < view; y++) {
    for (let x = 0; x < view; x++) {
      const sprite = new PIXI.Sprite(ascii[0x20]);
      sprite.scale = unit / sprite.texture.frame.width;
      sprite.x = unit * x;
      sprite.y = unit * y;
      app.stage.addChild(sprite);
      map.push(sprite);
    }
  }

  const wasm = await init();
  window.wasmCallbacks = {
    render: (ptr, sx, sy) => {
      const buffer = new Uint32Array(wasm.memory.buffer, ptr, sx * sy);
      for (let y = 0; y < view; y++) {
        for (let x = 0; x < view; x++) {
          const sprite = map[x + y * view];
          const glyph = buffer[(2 * x + 1) + (y + 1) * sx];
          const code = (glyph & 0xffff) - 0xff00 + 0x20;
          sprite.texture = ascii[code];
          sprite.tint = tints[(glyph >> 16) & 0xff];
        }
      }
    },
  };
  window.onkeydown = x => {
    const code = x.key.length === 1 ? x.key.charCodeAt(0) : x.keyCode;
    keydown(code, x.shiftKey);
  }
  app.ticker.add(() => { tick(); });
};

main();
