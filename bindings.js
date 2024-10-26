const render = (ptr, sx, sy) => {
  window.wasmCallbacks.render(ptr, sx, sy);
};

export {render};
