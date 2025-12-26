import {Reader, Terminal} from './lib.js';

class DebugTrace {
  constructor() {
    this.mapX = 100;
    this.mapY = 100;

    this.eid = '';
    this.index = 0;
    this.ticks = [];
    this.entities = [];
    this.aiOutput = [];
    this.lastIndex = -1;
    this.lastTicks = '';
    this.showAllEntities = true;

    this.reloading = false;

    this.terminal = new Terminal();
    this.ui = {
      aiOutput: document.getElementById('ai-trace'),
      checkbox: document.getElementById('show-all-entities'),
      entities: document.getElementById('entities'),
      map: document.getElementById('map'),
      timeline: document.getElementById('timeline'),
      view: document.getElementById('view'),
    };

    window.onkeydown = this.onkeydown.bind(this);
    window.onmousedown = this.onmousedown.bind(this);
    window.onmousemove = this.onmousemove.bind(this);

    this.ui.checkbox.onchange = this.onShowAllEntitiesChange.bind(this);
    this.ui.checkbox.checked = this.showAllEntities;
  }

  onShowAllEntitiesChange() {
    this.showAllEntities = this.ui.checkbox.checked;
    this.lastIndex = -1;
  }

  onkeydown(keyEvent) {
    const key = keyEvent.key;
    const code = key.length === 1 ? key.charCodeAt(0) : keyEvent.keyCode;

    if (key === 's') {
      this.ui.checkbox.checked = !this.ui.checkbox.checked;
      this.onShowAllEntitiesChange();
      return;
    }

    const options = this.ticks.map((x, i) => [x, i]).filter(x => x[0].eid === this.eid);
    if (options.length === 0) return;

    const existing = options.findIndex(x => x[1] === this.index);
    const prev = existing < 0 ? 0 : existing;

    const next = key === 'j' ? prev + 1 : key === 'k' ? prev - 1 : -1;
    if (next < 0 || next >= options.length) return;

    this.ui.timeline.children[prev].classList.remove('highlighted');
    this.ui.timeline.children[next].classList.add('highlighted');
    this.index = options[next][1];
  }


  onmousedown(mouseEvent) {
    const eid = this.getEID(mouseEvent);
    if (eid === null) return;

    this.eid = eid;
    this.lastIndex = -1;
    this.lastTicks = '';
  }

  onmousemove(mouseEvent) {
    const eid = this.getEID(mouseEvent);
    const id = eid ? `entity-${eid}` : '';

    for (const entity of this.entities) {
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

    const entity = this.entities.filter(e => e.posX === x && e.posY === y)[0];
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
      await this.reloadTimeline();
      await this.reloadView();
    } catch (e) {
      console.error(e);
    }
    this.reloading = false;
  }

  async reloadTimeline() {
    const response = await fetch('debug/ticks.txt');
    const ticks = await response.text();
    if (ticks === this.lastTicks) return;

    this.lastTicks = ticks;
    this.ticks = [];
    for (const line of ticks.trim().split('\n')) {
      try { this.ticks.push(JSON.parse(line)); } catch { break; }
    }
    this.index = Math.max(0, Math.min(this.index, this.ticks.length - 1));
    if (this.ticks.length === 0) return;

    const options = this.ticks.map((x, i) => [x, i]).filter(x => x[0].eid === this.eid);
    if (options.length === 0) {
      this.eid = this.ticks[this.index].eid;
    } else {
      let best = options.filter(x => x[1] >= this.index)[0];
      best = best ? best : options[options.length - 1];
      this.index = best[1];
    }

    const min = BigInt(this.ticks[0].time);
    const max = BigInt(this.ticks[this.ticks.length - 1].time);

    const elements = this.ticks.map((x, i) => {
      if (x.eid !== this.eid) return null;

      const element = document.createElement('div');
      const fraction = (() => {
        if (this.ticks.length === 1) return 0.5;
        const value = BigInt(x.time);
        return Number(10000n * (value - min) / (max - min)) / 100;
      })();
      element.style = `left: ${fraction}%`;
      element.classList.add('tick');
      if (i === this.index) element.classList.add('highlighted');
      return element;
    });
    this.ui.timeline.replaceChildren(...elements.filter(x => !!x));
  }

  async reloadView() {
    if (this.index === this.lastIndex) return;

    this.lastIndex = this.index;
    const response = await fetch(`debug/tick-${this.index}.bin.gz`);
    const inflated = new Response(response.body.pipeThrough(new DecompressionStream('gzip')));
    const data = await inflated.arrayBuffer();
    const reader = new Reader(data);

    const numEntities = reader.readInt();
    this.entities.length = 0;
    for (let i = 0; i < numEntities; i++) {
      const eid = reader.readStr();
      const name = reader.readStr();
      const health = reader.readDbl();
      const posX = reader.readInt();
      const posY = reader.readInt();
      const glyph0 = reader.readInt();
      const glyph1 = reader.readInt();
      this.entities.push({eid, name, health, posX, posY, glyph0, glyph1});
    }

    const entityElements = this.entities.map(x => {
      const element = document.createElement('div');
      element.id = `entity-${x.eid}`;
      element.textContent = `${x.name} - ${Math.max(Math.floor(100 * x.health), 1)}%`;
      element.classList.add('entity');
      if (x.eid === this.eid) element.classList.add('highlighted');
      return element;
    });
    this.ui.entities.replaceChildren(...entityElements);

    const numLines = reader.readInt();
    this.aiOutput.length = 0;
    for (let i = 0; i < numLines; i++) {
      const color = reader.readInt();
      const depth = reader.readInt();
      const text = reader.readStr();
      this.aiOutput.push({color, depth, text});
    }

    const traceElements = this.aiOutput.map(x => {
      const element = document.createElement('div');
      const color = x.color.toString(16);
      const zeros = '0'.repeat(6 - color.length);
      element.textContent = `${x.text}`;
      element.classList.add('ai-trace-line');
      element.style = `color: #${zeros}${color}; margin-left: ${12 * x.depth}px`;
      return element;
    });
    this.ui.aiOutput.replaceChildren(...traceElements);

    const w = reader.readInt();
    const h = reader.readInt();
    if (w !== this.mapX || h !== this.mapY) {
      throw Error(`Expected: ${this.mapX} x ${this.mapY} map; got: ${w} x ${h}`);
    }

    for (let y = 0; y < this.mapY; y++) {
      for (let x = 0; x < this.mapX; x++) {
        this.terminal.draw(2 * x, y, reader);
      }
    }
    if (this.showAllEntities) {
      for (const entity of this.entities) {
        const {posX, posY, glyph0, glyph1} = entity;
        this.terminal.drawGlyph(2 * posX, posY, glyph0, glyph1);
      }
    }
  }
}

const main = async () => {
  const debug = new DebugTrace();
  await debug.init();
};

main();
