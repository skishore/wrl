#!/bin/bash
pushd wasm
cargo build --release --target wasm32-unknown-unknown
popd
/Users/skishore/.cargo/bin/wasm-bindgen --out-dir wasm/pkg --target web target/wasm32-unknown-unknown/release/wrl_wasm.wasm
