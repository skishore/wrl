import {Reader, Terminal} from './lib.js';

class DebugTrace {
  constructor() {
    this.mapX = 100;
    this.mapY = 100;

    this.map = [];
    const w = this.mapX;
    const h = this.mapY;
    for (let i = 0; i < 2 * w * h; i++) this.map.push(-1);

    this.eid = '';
    this.ticks = [];
    this.tickState = {
      aiOutput: [],
      entities: [],
      map: [],
      sightings: [],
      utility: new Map(),
    };
    this.animBatch = [];
    this.animFrame = [];
    this.showAll = true;
    this.showSeen = true;
    this.showUtility = true;

    this.animIndex = 0;
    this.tickIndex = 0;
    this.reloading = false;
    this.lastAnimIndex = -1;
    this.lastTickIndex = -1;
    this.lastTicks = '';

    this.terminal = new Terminal();
    this.ui = {
      aiOutput: document.getElementById('ai-trace'),
      entities: document.getElementById('entities'),
      map: document.getElementById('map'),
      showAll: document.getElementById('show-all-entities'),
      showSeen: document.getElementById('show-sightings'),
      showUtility: document.getElementById('show-utility'),
      timeline: document.getElementById('timeline'),
      view: document.getElementById('view'),
    };

    window.onkeydown = this.onkeydown.bind(this);
    window.onmousedown = this.onmousedown.bind(this);
    window.onmousemove = this.onmousemove.bind(this);

    this.ui.showAll.onchange = this.onShowAllChange.bind(this);
    this.ui.showAll.checked = this.showAll;

    this.ui.showSeen.onchange = this.onShowSeenChange.bind(this);
    this.ui.showSeen.checked = this.showSeen;

    this.ui.showUtility.onchange = this.onShowUtilityChange.bind(this);
    this.ui.showUtility.checked = this.showUtility;
  }

  onShowAllChange() {
    this.showAll = this.ui.showAll.checked;
    this.markDirty();
  }

  onShowSeenChange() {
    this.showSeen = this.ui.showSeen.checked;
    this.markDirty();
  }

  onShowUtilityChange() {
    this.showUtility = this.ui.showUtility.checked;
    this.markDirty();
  }

  onkeydown(keyEvent) {
    const key = keyEvent.key;
    const code = key.length === 1 ? key.charCodeAt(0) : keyEvent.keyCode;

    if (key === 'r') {
      this.ui.showSeen.checked = !this.ui.showSeen.checked;
      this.onShowSeenChange();
      return;
    } else if (key === 's') {
      this.ui.showAll.checked = !this.ui.showAll.checked;
      this.onShowAllChange();
      return;
    } else if (key === 'u') {
      this.ui.showUtility.checked = !this.ui.showUtility.checked;
      this.onShowUtilityChange();
      return;
    }

    const eid = this.eid;
    const options = this.ticks.map((x, i) => [x, i]).filter(
        x => x[0].type === 'tick' && x[0].eid === eid);
    if (options.length === 0) return;

    const existing = options.findIndex(x => x[1] === this.tickIndex);
    const prev = existing < 0 ? 0 : existing;

    const next = key === 'j' ? prev + 1 : key === 'k' ? prev - 1 : -1;
    if (next < 0 || next >= options.length) return;

    this.animIndex = this.tickIndex;
    this.tickIndex = options[next][1];
    this.markDirty();
  }

  onmousedown(mouseEvent) {
    const eid = this.getEID(mouseEvent);
    if (eid === null) return;

    const options = this.ticks.map((x, i) => [x, i]).filter(
        x => x[0].type === 'tick' && x[0].eid === eid);
    if (options.length === 0) return;

    const best = options.filter(x => x[1] >= this.tickIndex)[0];
    const next = best ?? options[options.length - 1];

    this.eid = eid;
    this.tickIndex = next[1];
    this.animIndex = this.tickIndex;
    this.markDirty();
  }

  onmousemove(mouseEvent) {
    const eid = this.getEID(mouseEvent);
    const id = eid ? `entity-${eid}` : '';

    for (const entity of this.tickState.entities) {
      for (const element of this.ui.entities.children) {
        if (element.id !== id) element.classList.remove('mouseover');
        if (element.id === id) element.classList.add('mouseover');
      }
    }
  }

  getEID(mouseEvent) {
    const skip = mouseEvent.target !== this.terminal.app.canvas;
    const x = skip ? -1 : Math.floor(mouseEvent.layerX / (2 * this.terminal.unitX));
    const y = skip ? -1 : Math.floor(mouseEvent.layerY / this.terminal.unitY);

    const entity = this.tickState.entities.filter(
      e => e.particle.posX === x && e.particle.posY === y)[0];
    return entity ? entity.eid : null;
  }

  async init() {
    await this.terminal.init(2 * this.mapX, this.mapY);
    this.ui.map.appendChild(this.terminal.app.canvas);
    this.terminal.app.ticker.add(async () => { await this.reload(); });
  }

  async reload() {
    if (this.reloading) return;
    this.reloading = true;
    try {
      await this.reloadTicks();
      await this.reloadCurrentAnimation();
      await this.reloadCurrentTick();
      this.redraw();
    } catch (e) {
      console.error(e);
    }
    this.reloading = false;
  }

  async reloadTicks() {
    if (this.ticks.length > 0) return;

    const response = await fetch('debug/ticks.txt');
    const ticks = await response.text();
    if (ticks == this.lastTicks) return;

    this.ticks.length = 0;
    for (const line of ticks.trim().split('\n')) {
      try { this.ticks.push(JSON.parse(line)); } catch { break; }
    }
    this.lastTicks = ticks;
    this.dirty = true;

    const realTicks = this.ticks.filter(x => x.type === 'tick');
    if (!this.eid && realTicks.length > 0) this.eid = realTicks[0].eid;
  }

  async reloadDataFile(type, index) {
    const response = await fetch(`debug/${type}-${index}.bin.gz`);
    const inflated = new Response(response.body.pipeThrough(new DecompressionStream('gzip')));
    const data = await inflated.arrayBuffer();
    return new Reader(data);
  }

  async reloadAnimationBatch() {
    const {type, index} = this.ticks[this.animIndex];
    if (index === this.lastAnimIndex) return;

    const reader = await this.reloadDataFile(type, index);
    this.lastAnimIndex = index;

    const numFrames = reader.readInt();
    this.animBatch.length = 0;
    for (let i = 0; i < numFrames; i++) {
      const frame = [];
      const numParticles = reader.readInt();
      for (let j = 0; j < numParticles; j++) {
        frame.push(reader.readParticle());
      }
      this.animBatch.push(frame);
    }
  }

  async reloadCurrentAnimation() {
    if (this.animIndex === this.tickIndex) return;

    this.dirty = true;

    while (this.animIndex < this.tickIndex) {
      if (this.ticks[this.animIndex].type === 'animation') break;
      this.animIndex++;
    }
    if (this.animIndex >= this.tickIndex) {
      this.animIndex = this.tickIndex;
      this.animBatch.length = 0;
      this.animFrame.length = 0;
      this.lastAnimIndex = -1;
      return;
    }

    const prev = this.animIndex;

    await this.reloadAnimationBatch();

    if (this.animIndex !== prev) return;

    const frame = this.ticks[this.animIndex].frame;
    this.animFrame = this.animBatch[frame];
    this.animIndex++;
  }

  async reloadCurrentTick() {
    const eid = this.eid;
    const options = this.ticks.map((x, i) => [x, i]).filter(
        x => x[0].type === 'tick' && x[0].eid === eid);
    const best = options.filter(x => x[1] <= this.animIndex).pop();
    if (!best) return;

    const {type, index} = best[0];
    if (index === this.lastTickIndex) return;

    const reader = await this.reloadDataFile(type, index);
    this.lastTickIndex = index;
    this.dirty = true;

    const numEntities = reader.readInt();
    this.tickState.entities.length = 0;
    for (let i = 0; i < numEntities; i++) {
      const eid = reader.readStr();
      const name = reader.readStr();
      const health = reader.readDbl();
      const particle = reader.readParticle();
      this.tickState.entities.push({eid, name, health, particle});
    }

    const numSightings = reader.readInt();
    this.tickState.sightings.length = 0;
    for (let i = 0; i < numSightings; i++) {
      this.tickState.sightings.push(reader.readParticle());
    }

    const numLines = reader.readInt();
    this.tickState.aiOutput.length = 0;
    for (let i = 0; i < numLines; i++) {
      const color = reader.readInt();
      const depth = reader.readInt();
      const text = reader.readStr();
      this.tickState.aiOutput.push({color, depth, text});
    }

    const {mapX, mapY} = this;
    const numUtilityEntries = reader.readInt();
    this.tickState.utility.clear();
    for (let i = 0; i < numUtilityEntries; i++) {
      const value = reader.readInt();
      const x = reader.readInt();
      const y = reader.readInt();

      if (!(0 <= x && x < mapX && 0 <= y && y < mapY)) continue;
      const index = x + y * mapX;
      this.tickState.utility.set(index, value);
    }

    const w = reader.readInt();
    const h = reader.readInt();
    if (w > mapX || h > mapY) {
      throw Error(`Expected: max ${mapX} x ${mapY} map; got: ${w} x ${h}`);
    }
    this.tickState.map.length = 0;
    for (let y = 0; y < mapY; y++) {
      for (let x = 0; x < mapX; x++) {
        if (x < w && y < h) {
          this.tickState.map.push(reader.readInt());
          this.tickState.map.push(reader.readInt());
        } else {
          this.tickState.map.push(0xff00);
          this.tickState.map.push(0);
        }
      }
    }
  }

  drawGlyph(x, y, glyph0, glyph1) {
    const w = this.mapX;
    const h = this.mapY;
    if (!(0 <= x && x < w && 0 <= y && y < h)) return;

    const map = this.map;
    const index = 2 * (x + y * this.mapX);
    if (map[index] == glyph0 && map[index + 1] == glyph1) {
      return;
    }
    map[index] = glyph0;
    map[index + 1] = glyph1;
    this.terminal.drawGlyph(2 * x, y, glyph0, glyph1);
  }

  drawParticle(particle) {
    const {posX, posY, glyph0, glyph1} = particle;
    this.drawGlyph(posX, posY, glyph0, glyph1);
  }

  markDirty() {
    this.dirty = true;
  }

  redraw() {
    if (!this.dirty) return;
    this.dirty = false;

    this.redrawAITrace();
    this.redrawEntities();
    this.redrawMap();
    this.redrawTimeline();
  }

  redrawAITrace() {
    const traceElements = this.tickState.aiOutput.map(x => {
      const element = document.createElement('div');
      const color = x.color.toString(16);
      const zeros = '0'.repeat(6 - color.length);
      element.textContent = `${x.text}`;
      element.classList.add('ai-trace-line');
      element.style = `color: #${zeros}${color}; margin-left: ${12 * x.depth}px`;
      return element;
    });
    this.ui.aiOutput.replaceChildren(...traceElements);
  }

  redrawEntities() {
    const entityElements = this.tickState.entities.map(x => {
      const element = document.createElement('div');
      element.id = `entity-${x.eid}`;
      element.textContent = `${x.name} - ${Math.max(Math.floor(100 * x.health), 1)}%`;
      element.classList.add('entity');
      if (x.eid === this.eid) element.classList.add('highlighted');
      return element;
    });
    this.ui.entities.replaceChildren(...entityElements);
  }

  redrawMap() {
    let index = 0;
    const map = this.tickState.map;
    for (let y = 0; y < this.mapY; y++) {
      for (let x = 0; x < this.mapX; x++) {
        let glyph0 = map[index];
        let glyph1 = map[index + 1];

        if (this.showUtility) {
          const target = this.tickState.utility.get(index >> 1) ?? 0;
          const blue = (glyph1 >> 8) & 0xff;
          const util = (target >> 8) & 0xff;
          if (blue < util) glyph1 = (glyph1 & 0xffff00ff) | (util << 8);
        }

        this.drawGlyph(x, y, glyph0, glyph1);
        index += 2;
      }
    }
    if (this.animIndex < this.tickIndex) {
      for (const particle of this.animFrame) {
        this.drawParticle(particle);
      }
    } else {
      if (this.showAll) {
        for (const entity of this.tickState.entities) {
          this.drawParticle(entity.particle);
        }
      }
      if (this.showSeen) {
        for (const particle of this.tickState.sightings) {
          this.drawParticle(particle);
        }
      }
    }
  }

  redrawTimeline() {
    if (this.ticks.length === 0) {
      this.ui.timeline.replaceChildren();
      return;
    }

    const min = BigInt(this.ticks[0].time);
    const max = BigInt(this.ticks[this.ticks.length - 1].time);

    const elements = this.ticks.map((x, i) => {
      if (!(x.type === 'tick' && x.eid === this.eid)) return null;

      const element = document.createElement('div');
      const fraction = (() => {
        if (this.ticks.length === 1) return 0.5;
        const value = BigInt(x.time);
        return Number(10000n * (value - min) / (max - min)) / 100;
      })();
      element.style = `left: ${fraction}%`;
      element.classList.add('tick');
      if (i === this.tickIndex) element.classList.add('highlighted');
      return element;
    });
    this.ui.timeline.replaceChildren(...elements.filter(x => !!x));
  }
}

const main = async () => {
  const debug = new DebugTrace();
  await debug.init();
};

main();
