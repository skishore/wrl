use std::cmp::max;

use crate::base::{Buffer, Color, Glyph, LOS, Point, Rect, Slice};
use crate::effect::{Frame, self};
use crate::entity::Entity;
use crate::game::{UI_MAP_SIZE_X, UI_MAP_SIZE_Y, UI_MAP_MEMORY, UI_SHADE_FADE};
use crate::game::{State, show_item};
use crate::knowledge::{EntityKnowledge, Knowledge};

//////////////////////////////////////////////////////////////////////////////

// Constants

const UI_COL_SPACE: i32 = 2;
const UI_ROW_SPACE: i32 = 1;
const UI_KEY_SPACE: i32 = 4;

const UI_LOG_SIZE: i32 = 4;
const UI_CHOICE_SIZE: i32 = 40;
const UI_STATUS_SIZE: i32 = 30;
const UI_COLOR: (u8, u8, u8) = (255, 192, 0);

const UI_MOVE_ALPHA: f64 = 0.75;
const UI_TARGET_FRAMES: i32 = 20;

const UI_REMEMBERED: f64 = 0.15;

const PLAYER_KEY: char = 'a';
const SUMMON_KEYS: [char; 3] = ['s', 'd', 'f'];

//////////////////////////////////////////////////////////////////////////////

// Helpers

fn rivals<'a>(entity: &'a Entity) -> Vec<&'a EntityKnowledge> {
    let mut rivals = vec![];
    for other in &entity.known.entities {
        if other.age > 0 { break; }
        if other.eid != entity.eid { rivals.push(other); }
    }
    let pos = entity.pos;
    rivals.sort_by_cached_key(
        |x| ((x.pos - pos).len_l2_squared(), x.pos.0, x.pos.1));
    rivals
}

//////////////////////////////////////////////////////////////////////////////

// UI

pub struct LogLine {
    color: Color,
    menu: bool,
    text: String,
}

pub struct Menu {
    index: i32,
    summon: i32,
}

pub struct Target {
    error: String,
    frame: i32,
    okay_till: usize,
    source: Point,
    target: Point,
}

pub struct UI {
    pub log: Rect,
    pub map: Rect,
    pub choice: Rect,
    pub rivals: Rect,
    pub status: Rect,
    pub target: Rect,
    pub bounds: Point,
}

impl Default for UI {
    fn default() -> Self {
        let kl = Self::render_key('a').chars().count() as i32;
        assert!(kl == UI_KEY_SPACE);

        let ss = UI_STATUS_SIZE;
        let (x, y) = (UI_MAP_SIZE_X, UI_MAP_SIZE_Y);
        let (col, row) = (UI_COL_SPACE, UI_ROW_SPACE);
        let w = 2 * x + 2 + 2 * (ss + kl + 2 * col);
        let h = y + 2 + row + UI_LOG_SIZE + row + 1;

        let status = Rect {
            root: Point(col, row + 1),
            size: Point(ss + kl, y - 2 * row),
        };
        let map = Rect {
            root: Point(status.root.0 + status.size.0 + col + 1, 1),
            size: Point(2 * x, y),
        };
        let target = Rect {
            root: Point(map.root.0 + map.size.0 + col + 1, status.root.1),
            size: Point(ss + kl, 9),
        };
        let log = Rect {
            root: Point(0, map.root.1 + map.size.1 + row + 1),
            size: Point(w, UI_LOG_SIZE),
        };

        let (cw, ch) = (UI_CHOICE_SIZE, 6 * 5);
        let mut choice = Rect {
            root: Point((w - cw) / 2, (h - ch) / 2),
            size: Point(cw, ch),
        };
        if map.root.0 % 2 == choice.root.0 % 2 {
            choice.root.0 -= 1;
            choice.size.0 += 2;
        }
        if choice.size.0 % 2 != 0 { choice.size.0 += 1; }

        let ry = target.root.1 + target.size.1 + 2 * row + 1;
        let rivals = Rect {
            root: Point(target.root.0, ry),
            size: Point(ss + kl, status.root.1 + status.size.1 - ry),
        };

        Self { log, map, choice, rivals, status, target, bounds: Point(w, h) }
    }
}

impl UI {
    // Public entry points

    pub fn render_map(&self, state: &State, entity: &Entity,
                      frame: Option<&Frame>, offset: Point, slice: &mut Slice) {
        // Render each tile's base glyph, if it's known.
        let (known, player) = (&*entity.known, entity.player);
        let unseen = Glyph::wide(' ');

        let lookup = |point: Point| -> Glyph {
            let cell = known.get(point);
            let Some(tile) = cell.tile() else { return unseen; };

            let age_in_turns = (|| {
                if !player { return 0; }
                let age = cell.time_since_seen();
                for (turn, &turn_time) in state.turn_times.iter().enumerate() {
                    if age <= known.time - turn_time { return turn; }
                }
                return UI_MAP_MEMORY;
            })();
            if age_in_turns >= max(UI_MAP_MEMORY, 1) { return unseen; }

            let see_entity = cell.can_see_entity_at();
            let obscured = tile.limits_vision();
            let shadowed = cell.shade();

            let glyph = if see_entity && let Some(x) = cell.entity() {
                if obscured { x.glyph.with_fg(tile.glyph.fg()) } else { x.glyph }
            } else if let Some(x) = cell.items().last() {
                show_item(x)
            } else {
                tile.glyph
            };
            let mut color = glyph.fg();

            if !cell.visible() {
                let limit = max(UI_MAP_MEMORY, 1) as f64;
                let delta = (UI_MAP_MEMORY - age_in_turns) as f64;
                color = Color::white().fade(UI_REMEMBERED * delta / limit);
            } else if shadowed {
                color = color.fade(UI_SHADE_FADE);
            }
            glyph.with_fg(color)
        };

        // Render all currently-visible cells.
        slice.fill(Glyph::wide(' '));
        for y in 0..UI_MAP_SIZE_Y {
            for x in 0..UI_MAP_SIZE_X {
                let glyph = lookup(Point(x, y) + offset);
                slice.set(Point(2 * x, y), glyph);
            }
        }

        // Render ephemeral state: sounds we've heard and moves we've glimpsed.
        for entity in &known.entities {
            if !entity.heard { continue; }
            let Point(x, y) = entity.pos - offset;
            slice.set(Point(2 * x, y), Glyph::wide('?'));
        }
        if player {
            for (&k, v) in &state.moves {
                let Point(x, y) = k - offset;
                let p = Point(2 * x, y);
                if v.frame < 0 || !slice.contains(p) { continue; }

                let alpha = 1.0 - (v.frame as f64 / v.limit as f64);
                let color = v.color.fade(UI_MOVE_ALPHA * alpha);
                slice.set(p, slice.get(p).with_bg(color));
            }
        }

        // Render any animation that's currently running.
        if let Some(frame) = frame {
            for &effect::Particle { point, glyph } in frame {
                if !known.get(point).visible() { continue; }
                let Point(x, y) = point - offset;
                slice.set(Point(2 * x, y), glyph);
            }
        }

        // If we're still playing, render arrows showing NPC facing.
        if entity.cur_hp > 0 { self.render_arrows(state, known, offset, slice); }
    }

    fn render_arrows(&self, state: &State, known: &Knowledge,
                     offset: Point, slice: &mut Slice) {
        let arrow_length = 3;
        let sleep_length = 2;
        let mut arrows = vec![];
        for other in &known.entities {
            if other.friend || other.age > 0 { continue; }

            let (pos, dir) = (other.pos, other.dir);
            let mut ch = Glyph::ray(dir);
            let mut diff = dir.normalize(arrow_length as f64);
            if other.asleep { (ch, diff) = ('Z', Point(0, -sleep_length)); }
            arrows.push((ch, LOS(pos, pos + diff)));
        }

        for (ch, arrow) in &arrows {
            let speed = if *ch == 'Z' { 8 } else { 2 };
            let denom = if *ch == 'Z' { sleep_length } else { arrow_length };
            let index = (state.frame / speed) % (8 * denom as usize);
            if let Some(x) = arrow.get(index + 1) {
                let point = Point(2 * (x.0 - offset.0), x.1 - offset.1);
                slice.set(point, Glyph::wide(*ch));
            }
        }
    }

    pub fn render_log(&self, buffer: &mut Buffer, log: &[LogLine]) {
        let slice = &mut Slice::new(buffer, self.log);
        for line in log {
            slice.set_fg(Some(line.color)).write_str(&line.text).newline();
        }
    }

    pub fn render_rivals(&self, buffer: &mut Buffer,
                         entity: &Entity, target: Option<&Target>) {
        let slice = &mut Slice::new(buffer, self.rivals);
        let mut rivals = rivals(entity);
        rivals.truncate(max(slice.size().1, 0) as usize / 2);

        for rival in rivals {
            let EntityKnowledge { glyph, hp, name, .. } = *rival;
            let hp_color = Self::hp_color(hp);
            let hp_text = format!("{}%", max((100.0 * hp).floor() as i32, 1));
            let (sn, sh) = (name.chars().count(), hp_text.chars().count());
            let ss = max(16 - sn as i32 - sh as i32, 0) as usize;

            slice.newline();
            slice.write_chr(glyph).space().write_str(name);
            slice.spaces(ss).set_fg(Some(hp_color)).write_str(&hp_text).newline();

            let targeted = match &target {
                Some(x) => x.target == rival.pos,
                None => entity.known.focus == Some(rival.eid),
            };
            if targeted {
                let start = slice.get_cursor() - Point(0, 1);
                let (fg, bg) = (Color::default(), Color::gray(128));
                for x in 0..UI_STATUS_SIZE {
                    let p = start + Point(x, 0);
                    slice.set(p, Glyph::new(slice.get(p).ch(), fg, bg));
                }
            }
        }
    }

    pub fn render_status(&self, buffer: &mut Buffer, entity: &Entity,
                         menu: Option<&Menu>, summon: Option<&Entity>) {
        assert!(menu.is_some() == summon.is_some());
        let slice = &mut Slice::new(buffer, self.status);
        let known = &*entity.known;
        if let Some(view) = known.entity(entity.eid) {
            let key = if menu.is_some() { '-' } else { PLAYER_KEY };
            self.render_entity(Some(key), None, view, slice);
        }
        for (_, &key) in SUMMON_KEYS.iter().enumerate() {
            //let key = if menu.is_some() { '-' } else { *key };
            //let eid = entity.data.summons.get(i).map(|x| x.eid());
            //if let Some(view) = eid.and_then(|x| known.entity(x)) {
            //    self.render_entity(Some(key), None, view, slice);
            //    if let Some(x) = menu && x.summon == i as i32 {
            //        self.render_menu(slice, x.index, summon.unwrap());
            //    }
            //} else {
            //    self.render_empty_option(key, 0, slice);
            //}
            self.render_empty_option(key, 0, slice);
        }
    }

    pub fn render_target(&self, buffer: &mut Buffer,
                         entity: &Entity, target: Option<&Target>) {
        let slice = &mut Slice::new(buffer, self.target);
        let known = &*entity.known;
        if target.is_none() && known.focus.is_none() {
            let fg = Some(0x111.into());
            slice.newline();
            slice.set_fg(fg).write_str("No target selected.").newline();
            slice.newline();
            slice.set_fg(fg).write_str("[x] examine your surroundings").newline();
            return;
        }

        let (cell, view, header, seen) = match &target {
            Some(x) => {
                let cell = known.get(x.target);
                let (seen, view) = (cell.visible(), cell.entity());
                let header: String = "No target data yet...".into();
                //let header = match &x.data {
                //    TargetData::FarLook => "Examining...".into(),
                //    TargetData::Summon { index, .. } => {
                //        let name = match &entity.data.pokemon[*index] {
                //            PokemonEdge::In(y) => name(y),
                //            PokemonEdge::Out(_) => "?",
                //        };
                //        format!("Sending out {}...", name)
                //    }
                //};
                (cell, view, header, seen)
            }
            None => {
                let view = known.focus.and_then(|x| known.entity(x));
                let seen = view.map(|x| x.age == 0).unwrap_or(false);
                let cell = view.map(|x| known.get(x.pos)).unwrap_or(known.default());
                let header = if seen {
                    "Last target:"
                } else {
                    "Last target: (remembered)"
                }.into();
                (cell, view, header, seen)
            },
        };

        let fg = if target.is_some() || seen { None } else { Some(0x111.into()) };
        let text = if view.is_some() {
            if seen { "Standing on: " } else { "Stood on: " }
        } else {
            if seen { "You see: " } else { "You saw: " }
        };

        slice.newline();
        slice.set_fg(fg).write_str(&header).newline();

        if let Some(view) = view {
            self.render_entity(None, fg, view, slice);
        } else {
            slice.newline();
        }

        slice.set_fg(fg).write_str(text);
        if let Some(x) = cell.tile() {
            slice.write_chr(x.glyph).space();
            slice.write_chr('(').write_str(x.description).write_chr(')').newline();
        } else {
            slice.write_str("(unseen location)").newline();
        }
    }

    //fn render_choice(&self, buffer: &mut Buffer, trainer: &Trainer,
    //                 summons: Vec<&Pokemon>, choice: i32) {
    //    self.render_dialog(buffer, &self.choice);
    //    let slice = &mut Slice::new(buffer, self.choice);
    //    let options = &trainer.data.pokemon;
    //    for (i, key) in PARTY_KEYS.iter().enumerate() {
    //        let selected = choice == i as i32;
    //        match if i < options.len() { Some(&options[i]) } else { None } {
    //            Some(PokemonEdge::Out(x)) => {
    //                let pokemon = *summons.iter().find(|y| y.id() == *x).unwrap();
    //                let (me, pp) = (&*pokemon.data.me, get_pp(pokemon));
    //                self.render_option(*key, 1, selected, me, pp, slice);
    //            },
    //            Some(PokemonEdge::In(x)) =>
    //                self.render_option(*key, 0, selected, x, 1.0, slice),
    //            None => self.render_empty_option(*key, UI_COL_SPACE + 1, slice),
    //        }
    //    }
    //}

    // High-level private helpers

    //fn render_menu(&self, slice: &mut Slice, index: i32, summon: &Pokemon) {
    //    let spaces = UI::render_key('-').chars().count();

    //    for (i, key) in ATTACK_KEYS.iter().enumerate() {
    //        let prefix = if index == i as i32 { " > " } else { "  " };
    //        let attack = summon.data.me.attacks.get(i);
    //        let name = attack.map(|x| x.name).unwrap_or("---");
    //        let fg = if attack.is_some() { None } else { Some(0x111.into()) };
    //        slice.set_fg(fg).spaces(spaces).write_str(prefix);
    //        slice.write_str(&Self::render_key(*key)).write_str(name);
    //        slice.newline().newline();
    //    }

    //    let prefix = if index == ATTACK_KEYS.len() as i32 { " > " } else { "  " };
    //    slice.spaces(spaces).write_str(prefix);
    //    slice.write_str(&Self::render_key(RETURN_KEY)).write_str("Call back");
    //    slice.newline().newline();
    //}

    fn render_empty_option(&self, key: char, space: i32, slice: &mut Slice) {
        let n = space as usize;
        let fg = Some(0x111.into());
        let prefix = UI::render_key(key);

        slice.newline();
        slice.set_fg(fg).spaces(n).write_str(&prefix).write_str("---").newline();
        slice.newline().newline().newline();
    }

    //fn render_option(&self, key: char, out: i32, selected: bool,
    //                 me: &PokemonIndividualData, pp: f64, slice: &mut Slice) {
    //    let hp = get_hp(me);
    //    let (hp_color, pp_color) = (Self::hp_color(hp), 0x123.into());
    //    let fg = if out == 0 && hp > 0. { None } else { Some(0x111.into()) };

    //    let x = if selected { 1 } else { 0 };
    //    let arrow = if selected { '>' } else { ' ' };

    //    let prefix = UI::render_key(key);
    //    let n = prefix.chars().count() + (UI_COL_SPACE + 1) as usize;
    //    let w = slice.size().0 - (n as i32) - 2 * UI_COL_SPACE - 6;
    //    let status_bar_line = |p: &str, v: f64, c: Color, s: &mut Slice| {
    //        s.set_fg(fg).spaces(n + x).write_str(p);
    //        self.render_bar(v, c, w, s);
    //        s.newline();
    //    };

    //    slice.newline();
    //    slice.spaces(UI_COL_SPACE as usize).write_chr(arrow).spaces(x);
    //    slice.set_fg(fg).write_str(&prefix).write_str(me.species.name).newline();
    //    status_bar_line("HP: ", hp, hp_color, slice);
    //    status_bar_line("PP: ", pp, pp_color, slice);
    //    slice.newline();
    //}

    fn render_entity(&self, key: Option<char>, fg: Option<Color>,
                     entity: &EntityKnowledge, slice: &mut Slice) {
        let prefix = key.map(|x| UI::render_key(x)).unwrap_or(String::default());
        let n = prefix.chars().count();
        let w = UI_STATUS_SIZE - 6;
        let status_bar_line = |p: &str, v: f64, c: Color, s: &mut Slice| {
            self.render_bar(v, c, w, s.set_fg(fg).spaces(n).write_str(p));
            s.newline();
        };

        slice.newline();
        let (hp, pp) = (entity.hp, entity.pp);
        let (hp_color, pp_color) = (Self::hp_color(hp), 0x123.into());
        slice.set_fg(fg).write_str(&prefix).write_str(entity.name).newline();
        status_bar_line("HP: ", hp, hp_color, slice);
        status_bar_line("PP: ", pp, pp_color, slice);
        slice.newline();
    }

    // Private implementation details

    fn render_bar(&self, value: f64, color: Color, width: i32, slice: &mut Slice) {
        let count = if value > 0. { max(1, (width as f64 * value) as i32) } else { 0 };
        let glyph = Glyph::chfg('=', color);

        slice.write_chr('[');
        for _ in 0..count { slice.write_chr(glyph); }
        for _ in count..width { slice.write_chr(' '); }
        slice.write_chr(']');
    }

    fn render_box(&self, buffer: &mut Buffer, rect: &Rect) {
        let Point(w, h) = rect.size;
        let color: Color = UI_COLOR.into();
        buffer.set(rect.root + Point(-1, -1), Glyph::chfg('┌', color));
        buffer.set(rect.root + Point( w, -1), Glyph::chfg('┐', color));
        buffer.set(rect.root + Point(-1,  h), Glyph::chfg('└', color));
        buffer.set(rect.root + Point( w,  h), Glyph::chfg('┘', color));

        let tall = Glyph::chfg('│', color);
        let flat = Glyph::chfg('─', color);
        for x in 0..w {
            buffer.set(rect.root + Point(x, -1), flat);
            buffer.set(rect.root + Point(x,  h), flat);
        }
        for y in 0..h {
            buffer.set(rect.root + Point(-1, y), tall);
            buffer.set(rect.root + Point( w, y), tall);
        }
    }

    fn render_dialog(&self, buffer: &mut Buffer, rect: &Rect) {
        for x in 0..rect.size.0 {
            for y in 0..rect.size.1 {
                buffer.set(rect.root + Point(x, y), buffer.default);
            }
        }
        self.render_box(buffer, rect);
    }

    pub fn render_frame(&self, buffer: &mut Buffer) {
        let ml = self.map.root.0 - 1;
        let mw = self.map.size.0 + 2;
        let mh = self.map.size.1 + 2;
        let tt = self.target.root.1;
        let th = self.target.size.1;
        let rt = tt + th + UI_ROW_SPACE;
        let uw = self.bounds.0;
        let uh = self.bounds.1;

        self.render_title(buffer, ml, Point(0, 0), "Party");
        self.render_title(buffer, ml, Point(ml + mw, 0), "Target");
        self.render_title(buffer, ml, Point(ml + mw, rt), "Wild Pokemon");
        self.render_title(buffer, ml, Point(0, mh - 1), "Log");
        self.render_title(buffer, ml, Point(ml + mw, mh - 1), "");
        self.render_title(buffer, uw, Point(0, uh - 1), "");

        self.render_box(buffer, &self.map);
    }

    fn render_title(&self, buffer: &mut Buffer, width: i32, pos: Point, text: &str) {
        let shift = 2;
        let color: Color = UI_COLOR.into();
        let dashes = Glyph::chfg('-', color);
        let prefix_width = shift + text.chars().count() as i32;
        assert!(prefix_width <= width);
        for x in 0..shift {
            buffer.set(pos + Point(x, 0), dashes);
        }
        for (i, c) in text.chars().enumerate() {
            buffer.set(pos + Point(i as i32 + shift, 0), Glyph::chfg(c, color));
        }
        for x in prefix_width..width {
            buffer.set(pos + Point(x, 0), dashes);
        }
    }

    // Static helpers

    fn hp_color(hp: f64) -> Color {
        (if hp <= 0.25 { 0x300 } else if hp <= 0.50 { 0x330 } else { 0x020 }).into()
    }

    fn render_key(key: char) -> String {
        format!("[{}] ", key)
    }
}
