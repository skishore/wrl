import init, {keydown, tick} from './pkg/wrl.js';

const loadImage = async (uri) => new Promise((resolve, reject) => {
  const img = new Image();
  const canvas = document.createElement('canvas');
  const ctx = canvas.getContext('2d');

  img.crossOrigin = 'Anonymous';  // Handle CORS for cross-origin images

  img.onload = () => {
    canvas.width = img.width;
    canvas.height = img.height;
    ctx.drawImage(img, 0, 0);

    const imageData = ctx.getImageData(0, 0, img.width, img.height);
    resolve({rgba: imageData.data, width: img.width, height: img.height});
  };

  img.onerror = () => reject(new Error('Failed to load image'));
  img.src = uri;
});

const main = async () => {
  const image = await loadImage('images/aquarius_8x8.png');
  for (let i = 0; i < image.rgba.length; i += 4) {
    if (image.rgba[i + 0] === 0) image.rgba[i + 3] = 0;
  }
  for (let y = 0; y < image.height / 16; y++) {
    for (let x = 0; x < image.width / 16; x++) {
      for (let i = 0; i < 4; i++) {
        image.rgba[4 * (x + image.width * y) + i] = 0xff;
      }
    }
  }

  const unit = 8;
  const view = 50;
  const fgColor = (x) => x === 1 ? 0xffffff : x;
  const bgColor = (x) => x === 1 ? 0x1d1f21 : x;

  const app = new PIXI.Application();
  await app.init({width: unit * view, height: unit * view})
  app.renderer.background.color = bgColor(1);
  document.body.appendChild(app.canvas);

  const imageData = new ImageData(image.rgba, image.width, image.height);
  const canvas = document.createElement('canvas');
  canvas.width = image.width;
  canvas.height = image.height;
  const context = canvas.getContext('2d', {premultipliedAlpha: false});
  context.putImageData(imageData, 0, 0);
  const font = PIXI.Texture.from(canvas);

  const frame = (source, x, y) => {
    const frame = new PIXI.Rectangle(x * unit, y * unit, unit, unit);
    return new PIXI.Texture({source, frame});
  };

  const ascii = [];
  for (let i = 0; i < 256; i++) {
    ascii.push(frame(font, i & 0xf, i >> 4));
  }

  const map = [];
  for (let y = 0; y < view; y++) {
    for (let x = 0; x < view; x++) {
      for (let i = 0; i < 2; i++) {
        const sprite = new PIXI.Sprite(ascii[0x20]);
        sprite.scale = unit / sprite.texture.frame.width;
        sprite.x = unit * x;
        sprite.y = unit * y;
        app.stage.addChild(sprite);
        map.push(sprite);
      }
    }
  }

  const wasm = await init();
  window.wasmCallbacks = {
    render: (ptr, sx, sy) => {
      const buffer = new Uint32Array(wasm.memory.buffer, ptr, 2 * sx * sy);
      for (let y = 0; y < view; y++) {
        for (let x = 0; x < view; x++) {
          const spriteIndex = x + y * view;
          const backed = map[2 * spriteIndex + 0];
          const sprite = map[2 * spriteIndex + 1];
          const index = (2 * x + 1) + (y + 1) * sx;
          const glyph0 = buffer[2 * index + 0];
          const glyph1 = buffer[2 * index + 1];
          const fg = ((glyph0 >> 16) & 0xffff) | ((glyph1 & 0xff) << 16);
          const bg = (glyph1 >> 8) & 0xffffff;
          const code = (glyph0 & 0xffff) - 0xff00 + 0x20;
          backed.texture = ascii[0];
          backed.tint = bgColor(bg);
          sprite.texture = ascii[code];
          sprite.tint = fgColor(fg);
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
