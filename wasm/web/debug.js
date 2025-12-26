import {Terminal} from './lib.js';

class DebugTrace {
  constructor() {
    this.mapX = 100;
    this.mapY = 100;

    this.eid = '';
    this.index = 0;
    this.ticks = [];
    this.lastIndex = -1;
    this.lastTicks = '';

    this.reloading = false;

    this.terminal = new Terminal();
    this.timeline = document.getElementById('timeline');
  }

  onkeydown(e) {
    const code = e.key.length === 1 ? e.key.charCodeAt(0) : e.keyCode;
    const options = this.ticks.map((x, i) => [x, i]).filter(x => x[0].eid === this.eid);
    const prev = options.findIndex(x => x[1] === this.index);
    if (prev < 0) return;

    const next = e.key === 'j' ? prev + 1 : e.key === 'k' ? prev - 1 : -1;
    if (next < 0 || next >= options.length) return;

    this.timeline.children[prev].classList.remove('highlighted');
    this.timeline.children[next].classList.add('highlighted');
    this.index = options[next][1];
  }

  async init() {
    await this.terminal.init(2 * this.mapX, this.mapY);
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

    const existing = this.ticks.some(x => x.eid === this.eid);
    if (!existing) this.eid = this.ticks[this.index].eid;

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
    this.timeline.replaceChildren(...elements.filter(x => !!x));
  }

  async reloadView() {
    if (this.index === this.lastIndex) return;

    this.lastIndex = this.index;
    const response = await fetch(`debug/tick-${this.index}.bin`);
    const data = await response.arrayBuffer();
    const view = new DataView(data, 8);

    for (let y = 0; y < this.mapY; y++) {
      for (let x = 0; x < this.mapX; x++) {
        this.terminal.draw(2 * x, y, view, 8 * (x + y * this.mapX));
      }
    }
  }
}

const main = async () => {
  const debug = new DebugTrace();
  await debug.init();
  window.onkeydown = debug.onkeydown.bind(debug);
};

main();
