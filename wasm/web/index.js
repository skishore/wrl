import init, {keydown, size, tick} from '../pkg/wrl_wasm.js';

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

const upscaleAndCenterFont = (image, tileSize) => {
  const {rgba: buffer, width, height} = image;
  const upscaledTileSize = tileSize * 2;
  const cols = Math.floor(width / tileSize);
  const rows = Math.floor(height / tileSize);
  const upscaledWidth = width * 2;
  const upscaledHeight = height * 2;
  const result = new Uint8ClampedArray(upscaledWidth * upscaledHeight * 4);

  for (let row = 0; row < rows; row++) {
    for (let col = 0; col < cols; col++) {
      // Find the original bounds of each glyph.
      let minX = tileSize, minY = tileSize, maxX = 0, maxY = 0;
      for (let y = 0; y < tileSize; y++) {
        for (let x = 0; x < tileSize; x++) {
          const srcIdx = ((row * tileSize + y) * width + col * tileSize + x) * 4;
          if (buffer[srcIdx + 3] === 0) continue;
          minX = Math.min(minX, x);
          minY = Math.min(minY, y);
          maxX = Math.max(maxX, x);
          maxY = Math.max(maxY, y);
        }
      }

      // Use the bounds to center the glyph. Offsets are upscaled coordinates.
      const spriteWidth = maxX - minX + 1;
      const spriteHeight = maxY - minY + 1;
      const xOffset = tileSize - spriteWidth;
      const yOffset = tileSize - spriteHeight;

      // Copy, upscale, and center this glyph.
      for (let y = minY; y <= maxY; y++) {
        for (let x = minX; x <= maxX; x++) {
          const srcIdx = ((row * tileSize + y) * width + col * tileSize + x) * 4;
          const dstX = col * upscaledTileSize + xOffset + (x - minX) * 2;
          const dstY = row * upscaledTileSize + yOffset + (y - minY) * 2;

          for (let dy = 0; dy < 2; dy++) {
            for (let dx = 0; dx < 2; dx++) {
              const dstIdx = ((dstY + dy) * upscaledWidth + dstX + dx) * 4;
              for (let i = 0; i < 4; i++) result[dstIdx + i] = buffer[srcIdx + i];
            }
          }
        }
      }
    }
  }
  return {rgba: result, width: upscaledWidth, height: upscaledHeight};
};

const normalizeFont = (image) => {
  for (let i = 0; i < image.rgba.length; i += 4) {
    if (image.rgba[i + 0] === 0) image.rgba[i + 3] = 0;
  }
  for (let y = 0; y < image.height / 16; y++) {
    for (let x = 0; x < image.width / 16; x++) {
      for (let i = 0; i < 4; i++) {
        image.rgba[4 * (x + y * image.width) + i] = 0xff;
      }
    }
  }
  return image;
};

const fontTexture = (image) => {
  const imageData = new ImageData(image.rgba, image.width, image.height);
  const canvas = document.createElement('canvas');
  canvas.width = image.width;
  canvas.height = image.height;
  const context = canvas.getContext('2d', {premultipliedAlpha: false});
  context.putImageData(imageData, 0, 0);
  return PIXI.Texture.from(canvas);
};

const main = async () => {
  let unifont = await loadImage('web/images/unifont_8x16.png');
  unifont = normalizeFont(unifont);
  let aquarius = await loadImage('web/images/aquarius_8x8.png');
  aquarius = normalizeFont(aquarius);
  aquarius = upscaleAndCenterFont(aquarius, aquarius.width / 16);

  const wasm = await init();
  const [viewX, viewY] = size();

  const unitX = 8;
  const unitY = 16;

  const app = new PIXI.Application();
  await app.init({width: unitX * viewX, height: unitY * viewY})
  app.renderer.background.color = 0;
  document.body.appendChild(app.canvas);
  app.canvas.classList.add('main');

  const aquariusFont = fontTexture(aquarius);
  const unifontFont = fontTexture(unifont);

  const frame = (source, x, y) => {
    const frame = new PIXI.Rectangle(x * unitX, y * unitY, unitX, unitY);
    return new PIXI.Texture({source, frame});
  };

  const aquariusFrames = [];
  for (let i = 0; i < 512; i++) {
    aquariusFrames.push(frame(aquariusFont, i & 0x1f, i >> 5));
  }

  const unifontFrames = [];
  for (let i = 0; i < 256; i++) {
    unifontFrames.push(frame(unifontFont, i & 0xf, i >> 4));
  }

  const map = [];
  for (let y = 0; y < viewY; y++) {
    for (let x = 0; x < viewX; x++) {
      const items = [];
      for (let i = 0; i < 2; i++) {
        const sprite = new PIXI.Sprite(aquariusFrames[0]);
        sprite.x = unitX * x;
        sprite.y = unitY * y;
        app.stage.addChild(sprite);
        items.push(sprite);
      }
      const [bg, fg] = items;
      map.push({bg, fg});
    }
  }

  const boxDrawingChars = {
    '┌': 0xda,
    '┐': 0xbf,
    '└': 0xc0,
    '┘': 0xd9,
    '│': 0xb3,
    '─': 0xc4,
  };
  const boxDrawingMap = new Map();
  for (const key in boxDrawingChars) {
    if (!boxDrawingChars.hasOwnProperty(key)) continue;
    boxDrawingMap.set(key.charCodeAt(0), boxDrawingChars[key]);
  }

  window.wasmCallbacks = {
    render: (ptr, sx, sy) => {
      const buffer = new Uint32Array(wasm.memory.buffer, ptr, 2 * sx * sy);
      for (let y = 0; y < viewY; y++) {
        for (let x = 0; x < viewX; x++) {
          const spriteIndex = x + y * viewX;
          const cell = map[spriteIndex];
          const index = x + y * sx;
          const glyph0 = buffer[2 * index + 0];
          const glyph1 = buffer[2 * index + 1];
          const fg = ((glyph0 >> 16) & 0xffff) | ((glyph1 & 0xff) << 16);
          const bg = (glyph1 >> 8) & 0xffffff;

          let code = glyph0 & 0xffff;
          code = boxDrawingMap.get(code) ?? code;

          if (code <= 0xff) {
            cell.fg.texture = unifontFrames[code];
            cell.fg.tint = fg;
            cell.bg.tint = bg;
          } else if (code >= 0xff00) {
            const wide = code - 0xff00 + 0x20;
            const next = map[spriteIndex + 1];
            cell.fg.texture = aquariusFrames[2 * wide + 0];
            next.fg.texture = aquariusFrames[2 * wide + 1];
            cell.fg.tint = next.fg.tint = fg;
            cell.bg.tint = next.bg.tint = bg;
            x++;
          }
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
  app.ticker.add(() => { tick(); });
};

main();
