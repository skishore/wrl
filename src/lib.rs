#![feature(let_chains)]
#![feature(test)]

mod base;
mod game;
mod list;
mod effect;
mod entity;
mod knowledge;
mod pathing;

use std::cell::RefCell;

use wasm_bindgen::prelude::*;

use base::{Glyph, Matrix};
use game::State;

thread_local! {
    static STATE: RefCell<State> = State::new(None).into();
}

#[wasm_bindgen(module = "/bindings.js")]
extern "C" {
    fn render(ptr: *const Glyph, sx: i32, sy: i32) -> i32;
}

#[wasm_bindgen]
pub fn tick() {
    STATE.with_borrow_mut(|game|{
        game.update();
        let mut output = Matrix::default();
        let mut debug = String::default();
        game.render(&mut output, &mut debug);
        unsafe { render(output.data.as_ptr(), output.size.0, output.size.1) }
    });
}
