use std::cell::RefCell;

use wasm_bindgen::prelude::*;

use wrl_base::base::{Glyph, Matrix};
use wrl_base::game::{Input, State};

thread_local! {
    static STATE: RefCell<(State, Matrix<Glyph>)> = Default::default()
}

#[wasm_bindgen(module = "/web/bindings.js")]
extern "C" {
    fn render(ptr: *const Glyph, sx: i32, sy: i32) -> i32;
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
    STATE.with_borrow_mut(|(game, _)|{ game.add_input(input); });
}

#[wasm_bindgen]
pub fn tick() {
    STATE.with_borrow_mut(|(game, buffer)| {
        game.update();
        game.render(buffer);
        render(buffer.data.as_ptr(), buffer.size.0, buffer.size.1)
    });
}
