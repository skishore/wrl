const render = (mapData, mx, my, fovData, fovSize) => {
  window.wasmCallbacks.render(mapData, mx, my, fovData, fovSize);
};

export {render};
