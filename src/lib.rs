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

use base::{Glyph, Matrix, Point};
use game::{Input, State};

thread_local! {
    static STATE: RefCell<State> = State::new(None).into();
}

#[wasm_bindgen(module = "/bindings.js")]
extern "C" {
    fn render(map_data: *const Glyph, mx: i32, my: i32,
              fov_data: *const i32, fov_size: i32) -> i32;
}

#[wasm_bindgen]
pub fn keydown(ch: i32, shift: bool) {
    let input = if ch == 27 {
        Input::Escape
    } else if ch == 9 && shift {
        Input::BackTab
    } else {
        Input::Char(ch as u8 as char)
    };
    STATE.with_borrow_mut(|game|{ game.add_input(input); });
}

#[wasm_bindgen]
pub fn tick() {
    STATE.with_borrow_mut(|game|{
        game.update();
        let mut output = Matrix::default();
        let mut debug = String::default();
        let mut fovs = vec![];
        game.render(&mut output, &mut fovs, &mut debug);

        let map_data = output.data.as_ptr();
        let Point(mx, my) = output.size;
        let fov_data = fovs.as_ptr();
        let fov_size = fovs.len() as i32;
        unsafe { render(map_data, mx, my, fov_data, fov_size); }
    });
}
