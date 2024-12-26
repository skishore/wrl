#!/bin/bash
cargo build --release --target wasm32-unknown-unknown
/Users/skishore/.cargo/bin/wasm-bindgen --out-dir pkg --target web target/wasm32-unknown-unknown/release/wrl_wasm.wasm
