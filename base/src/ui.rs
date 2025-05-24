use std::cmp::{max, min};
use std::collections::VecDeque;

use rand::Rng;

use crate::ai::AIDebug;
use crate::base::{HashMap, LOS, Point, RNG, dirs};
use crate::base::{Buffer, Color, Glyph, Matrix, Rect, Slice};
use crate::effect::{Frame, self};
use crate::entity::{EID, Entity};
use crate::game::{WORLD_SIZE, FOV_RADIUS_NPC, FOV_RADIUS_PC_};
use crate::game::{Input, Tile, show_item};
use crate::knowledge::PLAYER_MAP_MEMORY;
use crate::knowledge::{EntityKnowledge, Knowledge};
use crate::pathing::Status;
use crate::shadowcast::{Vision, VisionArgs};

//////////////////////////////////////////////////////////////////////////////

// Constants

const UI_COL_SPACE: i32 = 2;
const UI_ROW_SPACE: i32 = 1;
const UI_KEY_SPACE: i32 = 4;

const UI_LOG_SIZE: i32 = 4;
const UI_DEBUG_SIZE: i32 = 60;
const UI_CHOICE_SIZE: i32 = 40;
const UI_STATUS_SIZE: i32 = 30;
const UI_COLOR: (u8, u8, u8) = (255, 192, 0);

const UI_MOVE_ALPHA: f64 = 0.75;
const UI_MOVE_FRAMES: i32 = 12;
const UI_TARGET_FRAMES: i32 = 20;

const UI_FOV_BRIGHTEN: f64 = 0.12;
const UI_REMEMBERED: f64 = 0.25;
const UI_SHADE_FADE: f64 = 0.50;

const UI_TARGET_SHADE: u8 = 192;
const UI_TARGET_FOV_SHADE: u8 = 32;

const UI_LOG_MENU: (u8, u8, u8) = (128, 192, 255);
const UI_LOG_FAILURE: (u8, u8, u8) = (255, 160, 160);

const PLAYER_KEY: char = 'a';
const SUMMON_KEYS: [char; 3] = ['s', 'd', 'f'];

//////////////////////////////////////////////////////////////////////////////

// Helpers

pub fn get_direction(ch: char) -> Option<Point> {
    match ch {
        'h' => Some(dirs::W),
        'j' => Some(dirs::S),
        'k' => Some(dirs::N),
        'l' => Some(dirs::E),
        'y' => Some(dirs::NW),
        'u' => Some(dirs::NE),
        'b' => Some(dirs::SW),
        'n' => Some(dirs::SE),
        '.' => Some(dirs::NONE),
        _ => None,
    }
}

fn rivals<'a>(entity: &'a Entity) -> Vec<&'a EntityKnowledge> {
    let mut rivals = vec![];
    for other in &entity.known.entities {
        if !other.visible { break; }
        if other.eid != entity.eid { rivals.push(other); }
    }
    let pos = entity.pos;
    rivals.sort_by_cached_key(
        |x| ((x.pos - pos).len_l2_squared(), x.pos.0, x.pos.1));
    rivals
}

fn within_map_bounds(ui: &UI, entity: &Entity, point: Point) -> bool {
    let size = ui.get_map_size();
    let Point(x, y) = point - ui.get_map_offset(entity);
    0 <= x && x < size.0 && 0 <= y && y < size.1
}

//////////////////////////////////////////////////////////////////////////////

// Layout

struct Layout {
    log: Rect,
    map: Rect,
    debug: Rect,
    choice: Rect,
    rivals: Rect,
    status: Rect,
    target: Rect,
    bounds: Point,
}

impl Default for Layout {
    fn default() -> Self {
        let side = 2 * FOV_RADIUS_PC_ + 1;
        Self::new(Point(side, side))
    }
}

impl Layout {
    fn new(size: Point) -> Self {
        let kl = UI::render_key('a').chars().count() as i32;
        assert!(kl == UI_KEY_SPACE);

        let Point(x, y) = size;
        let ss = UI_STATUS_SIZE;
        let (col, row) = (UI_COL_SPACE, UI_ROW_SPACE);
        let w = 2 * x + 2 + 2 * (ss + kl + 2 * col);
        let h = y + 2 + row + UI_LOG_SIZE + row + 1;

        let debug = Rect::default();
        let bounds = Point(w, h);

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

        Self { log, map, debug, choice, rivals, status, target, bounds }
    }

    fn full(size: Point) -> Self {
        let Point(x, y) = size;
        let w = UI_DEBUG_SIZE;
        let (col, row) = (UI_COL_SPACE, UI_ROW_SPACE);
        let map = Rect { root: Point(1, 1), size: Point(2 * x, y) };
        let debug = Rect {
            root: Point(map.root.0 + map.size.0 + col + 1, row + 1),
            size: Point(w, y - 2 * row),
        };
        let bounds = Point(debug.root.0 + debug.size.0, y + 2);

        let x = Rect::default();
        let (log, choice, rivals, status, target) = (x, x, x, x, x);

        Self { log, map, debug, choice, rivals, status, target, bounds }
    }
}

//////////////////////////////////////////////////////////////////////////////

// Effects

#[derive(Copy, Clone)]
pub struct MoveAnimation {
    pub color: Color,
    pub frame: i32,
    pub limit: i32,
}

struct RainDrop {
    frame: usize,
    point: Point,
}

struct Rainfall {
    ch: char,
    delta: Point,
    drops: VecDeque<RainDrop>,
    path: Vec<Point>,
    lightning: i32,
    thunder: i32,
}

//////////////////////////////////////////////////////////////////////////////

// Targeting UI

enum TargetData {
    FarLook,
    Summon { index: usize, range: i32 },
}

pub struct Target {
    data: TargetData,
    error: String,
    frame: i32,
    okay_until: usize,
    path: Vec<Point>,
    source: Point,
    target: Point,
}

fn can_target(entity: &EntityKnowledge) -> bool {
    entity.visible && !entity.friend
}

fn init_target(data: TargetData, source: Point, target: Point) -> Box<Target> {
    let (error, frame, okay_until, path) = ("".into(), 0, 0, vec![]);
    Box::new(Target { data, error, frame, okay_until, path, source, target })
}

//fn init_summon_target(player: &Trainer, data: TargetData) -> Box<Target> {
//    let (known, pos, dir) = (&*player.known, player.pos, player.dir);
//    let mut target = init_target(data, pos, pos);
//
//    if let Some(x) = defend_at_pos(pos, player) {
//        let line = LOS(pos, x);
//        for p in line.iter().skip(1).rev() {
//            update_target(known, &mut target, *p);
//            if target.error.is_empty() { return target; }
//        }
//    }
//
//    let mut okay = |p: Point| {
//        if !check_follower_square(known, player, p, false) { return false; }
//        update_target(known, &mut target, p);
//        target.error.is_empty()
//    };
//
//    let best = pos + dir.scale(2);
//    let next = pos + dir.scale(1);
//    if okay(best) { return target; }
//    if okay(next) { return target; }
//
//    let mut options: Vec<Point> = vec![];
//    for dx in -2..=2 {
//        for dy in -2..=2 {
//            let p = pos + Point(dx, dy);
//            if okay(p) { options.push(p); }
//        }
//    }
//
//    let update = (|| {
//        if options.is_empty() { return pos; }
//        *options.select_nth_unstable_by_key(0, |x| (*x - best).len_l2_squared()).1
//    })();
//    update_target(known, &mut target, update);
//    target
//}

fn update_target(known: &Knowledge, target: &mut Target, update: Point) {
    let los = LOS(target.source, update);

    target.error = "".into();
    target.frame = 0;
    target.path = los.into_iter().skip(1).collect();
    target.target = update;

    match &target.data {
        TargetData::FarLook => {
            for (i, &x) in target.path.iter().enumerate() {
                if known.get(x).visible() { target.okay_until = i + 1; }
            }
            if target.okay_until < target.path.len() {
                target.error = "You can't see a clear path there.".into();
            }
        }
        TargetData::Summon { range, .. } => {
            if target.path.is_empty() {
                target.error = "There's something in the way.".into();
            }
            for (i, &x) in target.path.iter().enumerate() {
                let cell = known.get(x);
                let status = cell.status();
                if status != Status::Free && status != Status::Unknown {
                    target.error = "There's something in the way.".into();
                } else if !(x - target.source).in_l2_range(*range) {
                    target.error = "You can't throw that far.".into();
                } else if !cell.visible() {
                    target.error = "You can't see a clear path there.".into();
                }
                if !target.error.is_empty() { break; }
                target.okay_until = i + 1;
            }
        }
    }
}

fn select_valid_target(ui: &mut UI, known: &Knowledge) -> Option<EID> {
    let target = ui.target.as_ref()?;
    let entity = known.get(target.target).entity();

    match &target.data {
        TargetData::FarLook => {
            let entity = entity?;
            if can_target(entity) { Some(entity.eid) } else { None }
        }
        TargetData::Summon { index: _index, .. } => {
            //state.input = Action::Summon(*index, target.target);
            ui.focus
        }
    }
}

//////////////////////////////////////////////////////////////////////////////

// UI inputs

fn process_ui_input(ui: &mut UI, entity: &Entity, input: Input) -> bool {
    let known = &*entity.known;
    let tab = input == Input::Char('\t') || input == Input::BackTab;
    let enter = input == Input::Char('\n') || input == Input::Char('.');

    let apply_tab = |prev: Option<EID>, off: bool| -> Option<EID> {
        let rivals = rivals(entity);
        if rivals.is_empty() { return None; }

        let t = input == Input::Char('\t');
        let n = rivals.len() + if off { 1 } else { 0 };

        let next = prev.and_then(|x| rivals.iter().position(|y| y.eid == x));
        let start = next.or_else(|| if off { Some(n - 1) } else { None });
        let index = start.map(|x| if t { x + n + 1 } else { x + n - 1 } % n)
                         .unwrap_or_else(|| if t { 0 } else { n - 1 });
        if index < rivals.len() { Some(rivals[index].eid) } else { None }
    };

    let get_initial_target = |source: Point| -> Point {
        let known = &*entity.known;
        let focus = ui.focus.and_then(|x| known.entity(x));
        if let Some(target) = focus && can_target(target) { return target.pos; }
        let rival = rivals(entity).into_iter().next();
        if let Some(rival) = rival { return rival.pos; }
        source
    };

    let get_updated_target = |target: Point| -> Option<Point> {
        if tab {
            let old_eid = known.get(target).entity().map(|x| x.eid);
            let new_eid = apply_tab( old_eid, false);
            return Some(known.entity(new_eid?)?.pos);
        }

        let ch = if let Input::Char(x) = input { Some(x) } else { None }?;
        let dir = get_direction(ch.to_lowercase().next().unwrap_or(ch))?;
        let scale = if ch.is_uppercase() { 4 } else { 1 };

        let mut prev = target;
        for _ in 0..scale {
            let next = prev + dir;
            if !within_map_bounds(ui, entity, prev + dir) { break; }
            prev = next;
        }
        Some(prev)
    };

    if let Some(x) = &ui.target {
        let update = get_updated_target(x.target);
        if let Some(update) = update && update != x.target {
            let mut target = ui.target.take();
            target.as_mut().map(|x| update_target(known, x, update));
            ui.target = target;
        } else if enter {
            if x.error.is_empty() {
                ui.focus = select_valid_target(ui, known);
                ui.target = None;
            } else {
                ui.log.log_menu(&x.error, UI_LOG_FAILURE);
            }
        } else if input == Input::Escape {
            if let TargetData::FarLook = x.data {
                let valid = x.error.is_empty();
                ui.focus = if valid { select_valid_target(ui, known) } else { None };
            }
            ui.log.log_menu("Canceled.", UI_LOG_MENU);
            ui.target = None;
        }
        return true;
    }

    if tab {
        ui.focus = apply_tab(ui.focus, true);
        return true;
    } else if input == Input::Escape {
        ui.focus = None;
        return true;
    }

    if input == Input::Char('x') {
        let source = entity.pos;
        let update = get_initial_target(source);
        let mut target = init_target(TargetData::FarLook, source, update);
        update_target(known, &mut target, update);
        ui.log.log_menu("Use the movement keys to examine a location:", UI_LOG_MENU);
        ui.target = Some(target);
        return true;
    }

    false
}

//////////////////////////////////////////////////////////////////////////////

// UI

pub struct LogLine {
    color: Color,
    menu: bool,
    text: String,
}

#[derive(Default)]
pub struct Log {
    lines: Vec<LogLine>,
}

impl Log {
    pub fn log<S: Into<String>>(&mut self, text: S) {
        self.log_color(text, Color::white());
    }

    pub fn log_color<S: Into<String>, T: Into<Color>>(&mut self, text: S, color: T) {
        let (color, text) = (color.into(), text.into());
        self.lines.push(LogLine { color, menu: false, text });
        if self.lines.len() as i32 > UI_LOG_SIZE { self.lines.remove(0); }
    }

    pub fn log_menu<S: Into<String>, T: Into<Color>>(&mut self, text: S, color: T) {
        let (color, text) = (color.into(), text.into());
        if self.lines.last().map(|x| x.menu).unwrap_or(false) { self.lines.pop(); }
        self.lines.push(LogLine { color, menu: true, text });
        if self.lines.len() as i32 > UI_LOG_SIZE { self.lines.remove(0); }
    }

    pub fn end_menu_logging(&mut self) {
        self.lines.last_mut().map(|x| x.menu = false);
    }
}

//////////////////////////////////////////////////////////////////////////////

// Focus

struct Focused {
    active: bool,
    vision: Vision,
}

impl Default for Focused {
    fn default() -> Self {
        Self { active: false, vision: Vision::new(FOV_RADIUS_NPC) }
    }
}

impl Focused {
    fn update(&mut self, entity: &Entity, target: &EntityKnowledge) {
        let known = &*entity.known;
        let (pos, dir) = (target.pos, target.dir);
        if target.asleep {
            self.vision.clear(target.pos);
        } else {
            let floor = Tile::get('.');
            let opacity_lookup = |x| known.get(x).tile().unwrap_or(floor).opacity();
            self.vision.compute(&VisionArgs { pos, dir, opacity_lookup });
        }
    }
}

//////////////////////////////////////////////////////////////////////////////

// UI

struct Menu {
    index: i32,
    summon: i32,
}

#[derive(Default)]
pub struct UI {
    frame: usize,
    layout: Layout,
    full: bool,

    // Public members
    pub debug: AIDebug,
    pub log: Log,

    // Animations
    moves: HashMap<Point, MoveAnimation>,
    rainfall: Option<Rainfall>,

    // Modal components
    focus: Option<EID>,
    target: Option<Box<Target>>,
    focused: Focused,
}

impl UI {
    // Rendering entry point

    pub fn render(&self, buffer: &mut Buffer, entity: &Entity,
                  frame: Option<&Frame>, entities: &[(Point, Glyph)]) {
        if buffer.data.is_empty() {
            *buffer = Matrix::new(self.layout.bounds, ' '.into());
        }
        buffer.fill(buffer.default);
        self.render_layout(buffer);

        self.render_log(buffer, entity);
        self.render_rivals(buffer, entity);
        self.render_status(buffer, entity, None, None);
        self.render_target(buffer, entity);

        // Render the base map, then the debug layer, then the weather:
        self.render_map(buffer, entity, frame);
        if !entity.player && frame.is_none() {
            self.render_debug_overlay(buffer, entity, entities);
        }
        self.render_weather(buffer, entity);

        if !entity.player && self.full {
            self.render_debug(buffer, entity);
        }
    }

    // Update entry points

    pub fn animate_move(&mut self, color: Color, delay: i32, point: Point) {
        let frame = -delay * UI_MOVE_FRAMES / 2;
        let manim = MoveAnimation { color, frame, limit: UI_MOVE_FRAMES };
        self.moves.insert(point, manim);
    }

    pub fn process_input(&mut self, entity: &Entity, input: Input) -> bool {
        process_ui_input(self, entity, input)
    }

    pub fn show_full_view(&mut self) {
        let side = WORLD_SIZE;
        self.layout = Layout::full(Point(side, side));
        self.full = true;
    }

    pub fn start_rain(&mut self, delta: Point, count: usize) {
        let rainfall = Rainfall {
            ch: Glyph::ray(delta),
            delta,
            drops: VecDeque::with_capacity(count),
            path: LOS(Point::default(), delta),
            lightning: -1,
            thunder: 0,
        };
        self.rainfall = Some(rainfall);
    }

    pub fn update(&mut self, pos: Point, rng: &mut RNG) {
        self.frame += 1;
        self.update_weather(pos, rng);

        for x in self.moves.values_mut() { x.frame += 1; }
        self.moves.retain(|_, v| v.frame < v.limit);
    }

    pub fn update_focus(&mut self, entity: &Entity) {
        let known = &*entity.known;
        let focus = self.focus.and_then(|x| known.entity(x));
        if focus.is_none() { self.focus = None; }

        let target = match &self.target {
            Some(x) => known.get(x.target).entity(),
            None => focus,
        };
        if let Some(target) = target && can_target(target) {
            self.focused.update(entity, target);
            self.focused.active = true;
        } else {
            self.focused.active = false;
        }
    }

    pub fn update_target(&mut self, entity: &Entity) -> bool {
        let Some(target) = &mut self.target else { return false };
        target.frame = (target.frame + 1) % UI_TARGET_FRAMES;
        self.update_focus(entity);
        true
    }

    // Update helpers

    fn update_weather(&mut self, pos: Point, rng: &mut RNG) {
        let Some(rainfall) = &mut self.rainfall else { return; };

        let frame = self.frame;
        while let Some(x) = rainfall.drops.front() && x.frame < frame {
            rainfall.drops.pop_front();
        }
        let total = rainfall.drops.capacity();
        let denom = max(rainfall.delta.1, 1);
        let delta = denom as usize;
        let extra = (frame + 1) * total / delta - (frame * total) / delta;
        for _ in 0..min(extra, total - rainfall.drops.len()) {
            let x = rng.gen_range(0..denom);
            let y = rng.gen_range(0..denom);
            let target_frame = frame + rainfall.path.len() - 1;
            let target_point = Point(x - denom / 2, y - denom / 2) + pos;
            rainfall.drops.push_back(RainDrop { frame: target_frame, point: target_point });
        }

        assert!(rainfall.lightning >= -1);
        if rainfall.lightning == -1 {
            if rng.gen::<f32>() < 0.002 { rainfall.lightning = 10; }
        } else if rainfall.lightning > 0 {
            rainfall.lightning -= 1;
        }

        assert!(rainfall.thunder >= 0);
        if rainfall.thunder == 0 {
            if rainfall.lightning == 0 && rng.gen::<f32>() < 0.02 { rainfall.thunder = 16; }
        } else {
            rainfall.thunder -= 1;
            if rainfall.thunder == 0 { rainfall.lightning = -1; }
        }
    }

    // Rendering the map

    fn render_map(&self, buffer: &mut Buffer, entity: &Entity, frame: Option<&Frame>) {
        let slice = &mut Slice::new(buffer, self.layout.map);
        let offset = self.get_map_offset(entity);

        // Render each tile's base glyph, if it's known.
        let (known, player) = (&*entity.known, entity.player);
        let unseen = Glyph::wide(' ');

        let lookup = |point: Point| -> Glyph {
            let cell = known.get(point);
            let Some(tile) = cell.tile() else { return unseen; };

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
                let turns = if player { cell.turns_since_seen() } else  { 0 };
                let delta = (PLAYER_MAP_MEMORY - turns) as f64;
                let limit = max(PLAYER_MAP_MEMORY, 1) as f64;
                color = Color::white().fade(UI_REMEMBERED * delta / limit);
            } else if shadowed {
                color = color.fade(UI_SHADE_FADE);
            }
            glyph.with_fg(color)
        };

        // Render all currently-visible cells.
        slice.fill(Glyph::wide(' '));
        let size = self.get_map_size();
        for y in 0..size.1 {
            for x in 0..size.0 {
                let glyph = lookup(Point(x, y) + offset);
                let scent = entity.get_scent_at(Point(x, y) + offset);
                let color = Color::from((255, 128, 128)).fade(scent);
                let glyph = glyph.with_bg(color);
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
            for (&k, v) in &self.moves {
                let Point(x, y) = k - offset;
                let p = Point(2 * x, y);
                if v.frame < 0 || !slice.contains(p) { continue; }

                let alpha = 1.0 - (v.frame as f64 / v.limit as f64);
                let color = v.color.fade(UI_MOVE_ALPHA * alpha);
                slice.set(p, slice.get(p).with_bg(color));
            }
        }

        // Helpers used for the map overlay UIs.
        let set = |slice: &mut Slice, point: Point, glyph: Glyph| {
            let point = Point(2 * (point.0 - offset.0), point.1 - offset.1);
            slice.set(point, glyph);
        };
        let highlight = |slice: &mut Slice, point: Point, bg: Color| {
            let point = Point(2 * (point.0 - offset.0), point.1 - offset.1);
            if !slice.contains(point) { return; }

            let glyph = slice.get(point);
            slice.set(point, glyph.with_fg(Color::black()).with_bg(bg));
        };
        let brighten = |slice: &mut Slice, point: Point| {
            let point = Point(2 * (point.0 - offset.0), point.1 - offset.1);
            if !slice.contains(point) { return; }

            let glyph = slice.get(point);
            let glyph = glyph.with_fg(glyph.fg().brighten(UI_FOV_BRIGHTEN));
            let glyph = glyph.with_bg(glyph.bg().brighten(UI_FOV_BRIGHTEN));
            slice.set(point, glyph);
        };

        // Render the focused entity on the map.
        let focused_vision = self.focused.vision.get_points_seen();
        if self.focused.active && let Some(&point) = focused_vision.first() {
            highlight(slice, point, Color::gray(UI_TARGET_SHADE));
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
        if entity.cur_hp > 0 { self.render_arrows(known, offset, slice); }

        // Render the targeting UI on the map.
        if let Some(target) = &self.target {
            let shade = Color::gray(UI_TARGET_SHADE);
            let color = if target.error.is_empty() { 0x440 } else { 0x400 };
            highlight(slice, target.source, shade);
            highlight(slice, target.target, color.into());

            let frame = target.frame >> 1;
            let count = UI_TARGET_FRAMES >> 1;
            let limit = target.path.len() as i32 - 1;
            let ch = Glyph::ray(target.target - target.source);
            for i in 0..limit {
                if !((i + count - frame) % count < 2) { continue; }
                let point = target.path[i as usize];
                let color = if (i as usize) < target.okay_until { 0x440 } else { 0x400 };
                set(slice, point, Glyph::wdfg(ch, color));
            }
        }

        // Render an estimate of the focused entity's FOV on the map.
        if self.focused.active {
            for &point in focused_vision.iter().skip(1) {
                brighten(slice, point);
            }
        }
    }

    fn render_arrows(&self, known: &Knowledge, offset: Point, slice: &mut Slice) {
        let arrow_length = 3;
        let sleep_length = 2;
        let mut arrows = vec![];
        for other in &known.entities {
            if !can_target(other) { continue; }

            let (pos, dir) = (other.pos, other.dir);
            let mut ch = Glyph::ray(dir);
            let mut diff = dir.normalize(arrow_length as f64);
            if other.asleep { (ch, diff) = ('Z', Point(0, -sleep_length)); }
            arrows.push((ch, LOS(pos, pos + diff)));
        }

        for (ch, arrow) in &arrows {
            let glyph = Glyph::wide(*ch);
            let speed = if *ch == 'Z' { 8 } else { 2 };
            let denom = if *ch == 'Z' { 4 * sleep_length } else { 8 * arrow_length };
            let index = (self.frame / speed) % (denom as usize);
            if let Some(x) = arrow.get(index + 1) {
                let point = Point(2 * (x.0 - offset.0), x.1 - offset.1);
                slice.set(point, glyph);
            }
        }
    }

    fn render_debug(&self, buffer: &mut Buffer, entity: &Entity) {
        let slice = &mut Slice::new(buffer, self.layout.debug);
        entity.ai.debug(slice);
    }

    fn render_debug_overlay(&self, buffer: &mut Buffer, entity: &Entity,
                            entities: &[(Point, Glyph)]) {
        let slice = &mut Slice::new(buffer, self.layout.map);
        let offset = self.get_map_offset(entity);

        let path = entity.ai.get_path();
        let slice_point = |p: Point| Point(2 * (p.0 - offset.0), p.1 - offset.1);

        for &p in path.iter().skip(1) {
            let point = slice_point(p);
            let mut glyph = slice.get(point);
            if glyph.ch() == Glyph::wide(' ').ch() { glyph = Glyph::wide('.'); }
            slice.set(point, glyph.with_fg(0x400));
        }
        for &(p, score) in &self.debug.utility {
            let point = slice_point(p);
            let glyph = slice.get(point);
            slice.set(point, glyph.with_bg((score, score, score / 2 + 128)));
        }
        if let Some(&p) = path.first() {
            let point = slice_point(p);
            let glyph = slice.get(point);
            slice.set(point, glyph.with_fg(Color::black()).with_bg(0x400));
        }
        for &(point, glyph) in entities {
            slice.set(slice_point(point), glyph);
        }
        for other in &entity.known.entities {
            let color = if other.visible { 0x040 } else {
                if other.moved { 0x400 } else { 0x440 }
            };
            let glyph = other.glyph.with_fg(Color::black()).with_bg(color);
            let Point(x, y) = other.pos - offset;
            slice.set(Point(2 * x, y), glyph);
        };
    }

    fn render_weather(&self, buffer: &mut Buffer, entity: &Entity) {
        let Some(rainfall) = &self.rainfall else { return };

        let slice = &mut Slice::new(buffer, self.layout.map);
        let offset = self.get_map_offset(entity);

        let size = self.get_map_size();
        let base = Tile::get('~').glyph.fg();
        let known = &*entity.known;

        for drop in &rainfall.drops {
            let index = drop.frame - self.frame;
            let Some(&delta) = rainfall.path.get(index) else { continue; };

            let (ground, point) = (index == 0, drop.point - delta);
            let cell = if ground { known.get(point) } else { known.default() };
            if ground && !cell.visible() { continue; }

            let Point(x, y) = point - offset;
            let ch = if index == 0 { 'o' } else { rainfall.ch };
            let shade = cell.shade();
            let color = if shade { base.fade(UI_SHADE_FADE) } else { base };
            let glyph = Glyph::wdfg(ch, color);
            slice.set(Point(2 * x, y), glyph);
        }

        if rainfall.lightning > 0 {
            let color = Color::from(0x111 * (rainfall.lightning / 2));
            for y in 0..size.1 {
                for x in 0..size.0 {
                    let point = Point(2 * x, y);
                    slice.set(point, slice.get(point).with_bg(color));
                }
            }
        }

        if rainfall.thunder > 0 {
            let shift = (rainfall.thunder - 1) % 4;
            if shift % 2 == 0 {
                let space = Glyph::char(' ');
                let delta = Point(shift - 1, 0);
                let limit = if delta.1 > 0 { -1 } else { 0 };
                for y in 0..size.1 {
                    for x in 0..(size.0 + limit) {
                        let point = Point(2 * x, y);
                        slice.set(point + delta, slice.get(point));
                    }
                    slice.set(Point(0, y), space);
                    slice.set(Point(2 * size.0 - 1, y), space);
                }
            }
        }
    }

    // Rendering each section of the UI

    fn render_log(&self, buffer: &mut Buffer, entity: &Entity) {
        let slice = &mut Slice::new(buffer, self.layout.log);
        slice.write_str(&format!(
            "Scent at {:?}: {:.2}; history = {} items",
            entity.pos, entity.get_scent_at(entity.pos), entity.history.len()
        )).newline();
        for line in &self.log.lines {
            slice.set_fg(Some(line.color)).write_str(&line.text).newline();
        }
    }

    fn render_rivals(&self, buffer: &mut Buffer, entity: &Entity) {
        let slice = &mut Slice::new(buffer, self.layout.rivals);
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

            let targeted = match &self.target {
                Some(x) => x.target == rival.pos,
                None => self.focus == Some(rival.eid),
            };
            if targeted {
                let start = slice.get_cursor() - Point(0, 1);
                let (fg, bg) = (Color::white(), Color::gray(UI_TARGET_FOV_SHADE));
                for x in 0..UI_STATUS_SIZE {
                    let p = start + Point(x, 0);
                    slice.set(p, Glyph::new(slice.get(p).ch(), fg, bg));
                }
            }
        }
    }

    fn render_status(&self, buffer: &mut Buffer, entity: &Entity,
                     menu: Option<&Menu>, summon: Option<&Entity>) {
        assert!(menu.is_some() == summon.is_some());
        let slice = &mut Slice::new(buffer, self.layout.status);
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

    fn render_target(&self, buffer: &mut Buffer, entity: &Entity) {
        let slice = &mut Slice::new(buffer, self.layout.target);
        let known = &*entity.known;
        if self.target.is_none() && self.focus.is_none() {
            let fg = Some(0x111.into());
            slice.newline();
            slice.set_fg(fg).write_str("No target selected.").newline();
            slice.newline();
            slice.set_fg(fg).write_str("[x] examine your surroundings").newline();
            return;
        }

        // TODO: There are several bugs in this code:
        //
        //   - EntityKnowledge for the focused entity can get cleaned up while
        //     the player is still focused on it.
        //
        //   - CellKnowledge for the tile under the focused entity can get
        //     cleaned up while the player is still focused on it.
        //
        //   - If the player hears the targeted entity move, but does not see
        //     it, the knowledge update currently still update the entity's
        //     position, giving the player knowledge of the tile under its
        //     new location. Instead, noise knowledge updates for the player
        //     shouldn't be tied to a specific entity. The player can guess.
        //
        // Unfortunately, the cleanest way to fix all of these bugs is to
        // clone the focused entity's knowledge each frame on which the entity
        // is visible. Is there a better way?
        let (cell, view, header, seen) = match &self.target {
            Some(x) => {
                let cell = known.get(x.target);
                let seen = cell.visible();
                let view = if cell.can_see_entity_at() { cell.entity() } else { None };
                let header = match &x.data {
                    TargetData::FarLook => "Examining...".into(),
                    TargetData::Summon { index: _index, .. } => {
                        //let name = match &entity.data.pokemon[*index] {
                        //    PokemonEdge::In(y) => name(y),
                        //    PokemonEdge::Out(_) => "?",
                        //};
                        let name = "<unimplemented>";
                        format!("Sending out {}...", name)
                    }
                };
                (cell, view, header, seen)
            }
            None => {
                let view = self.focus.and_then(|x| known.entity(x));
                let seen = view.map(|x| x.visible).unwrap_or(false);
                let cell = view.map(|x| known.get(x.pos)).unwrap_or(known.default());
                let header = if seen {
                    "Last target:"
                } else {
                    "Last target: (remembered)"
                }.into();
                (cell, view, header, seen)
            },
        };

        let fg = if self.target.is_some() || seen { None } else { Some(0x111.into()) };
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
    //    self.render_box(buffer, &self.choice);
    //    let slice = &mut Slice::new(buffer, self.choice);
    //    slice.fill(buffer.default);
    //
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

    // High-level private rendering helpers

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

    // Low-level private rendering helpers

    fn get_map_size(&self) -> Point {
        Point(self.layout.map.size.0 / 2, self.layout.map.size.1)
    }

    fn get_map_offset(&self, entity: &Entity) -> Point {
        if self.full { return Point::default(); }
        let size = self.get_map_size();
        entity.pos - Point(size.0 / 2, size.1 / 2)
    }

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

    fn render_layout(&self, buffer: &mut Buffer) {
        let ml = self.layout.map.root.0 - 1;
        let mw = self.layout.map.size.0 + 2;
        let mh = self.layout.map.size.1 + 2;
        let tt = self.layout.target.root.1;
        let th = self.layout.target.size.1;
        let rt = tt + th + UI_ROW_SPACE;
        let uw = self.layout.bounds.0;
        let uh = self.layout.bounds.1;

        self.render_title(buffer, ml, Point(0, 0), "Party");
        self.render_title(buffer, ml, Point(ml + mw, 0), "Target");
        self.render_title(buffer, ml, Point(ml + mw, rt), "Nearby");
        self.render_title(buffer, ml, Point(0, mh - 1), "Log");
        self.render_title(buffer, ml, Point(ml + mw, mh - 1), "");
        self.render_title(buffer, uw, Point(0, uh - 1), "");

        if self.full {
            let dl = self.layout.bounds.0 - mw;
            self.render_title(buffer, dl, Point(ml + mw, 0), "Debug");
        }

        self.render_box(buffer, &self.layout.map);
    }

    fn render_title(&self, buffer: &mut Buffer, width: i32, pos: Point, text: &str) {
        if width <= 0 { return; }

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
