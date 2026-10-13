use std::cmp::{max, min};
use std::collections::VecDeque;

use rand::Rng;

use crate::base::glyph::{Buffer, Color, Glyph, Rect, Slice};
use crate::base::point::{Bound, Delta, DeltaLOS, LOS, Matrix, Point, dirs};
use crate::base::pathing::Status;
use crate::base::util::{HashMap, HashSet, RNG};
use crate::base::vision::{Vision, VisionArgs};

use super::actor::{Actor, AttackTarget, Command, Individual, Teammate};
use super::ai::{ChooseDefenseSquare, CheckFollowerSquare, Follower};
use super::dex::Attack;
use super::effect::{Frame, ParticleData, RenderData};
use super::event::{Call, Location, Sound, UID};
use super::game::{FOV_RADIUS_NPC, FOV_RADIUS_PC_, SUMMON_MOVES, SUMMON_RANGE};
use super::game::{Action, Input, Tile, TileFlags, show_item};
use super::knowledge::{ActorKnowledge, Knowledge, NoiseKnowledge, PointLookup, Trace};

//////////////////////////////////////////////////////////////////////////////

// Constants

const UI_COL_SPACE: i32 = 2;
const UI_ROW_SPACE: i32 = 1;
const UI_KEY_SPACE: i32 = 4;

const UI_LOG_SIZE: i32 = 4;
const UI_CHOICE_SIZE: i32 = 40;
const UI_STATUS_SIZE: i32 = 30;
const UI_COLOR: i32 = 0xffc000;

const UI_MOVE_ALPHA: f64 = 0.75;
const UI_MOVE_FRAMES: i32 = 12;
const UI_TARGET_FRAMES: i32 = 20;

const UI_FOV_BRIGHTEN: f64 = 0.12;
const UI_REMEMBERED: f64 = 0.20;
const UI_SHADOW: f64 = 0.40;
const UI_SUNLIGHT: f64 = 0.80;
const UI_EXTRA_LIGHT: f64 = 0.80;

const UI_TARGET_SHADE: u8 = 192;
const UI_TARGET_FOV_SHADE: u8 = 32;

const UI_LOG_MENU: i32 = 0x80c0ff;
const UI_LOG_FAILURE: i32 = 0xff9090;
const UI_LOG_SUCCESS: i32 = 0xc0ff80;
const UI_GRAY_OPTION: i32 = 0x545454;

const PLAYER_KEY: char = 'a';
const RETURN_KEY: char = 'r';
const SWITCH_KEY: char = 'w';
const SUMMON_KEYS: [char; 3] = ['s', 'd', 'f'];
const ATTACK_KEYS: [char; 4] = ['a', 's', 'd', 'f'];
const ACTION_KEYS: [char; 2] = [SWITCH_KEY, RETURN_KEY];
const PARTY_KEYS: [char; 6] = ['a', 'b', 'c', 'd', 'e', 'f'];

//////////////////////////////////////////////////////////////////////////////

// Helpers

fn get_direction(ch: char) -> Option<Delta> {
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

fn describe_actor(x: &ActorKnowledge) -> &str {
    x.name.as_ref().map_or(x.species.name, |x| &**x)
}

fn describe_noise(x: &NoiseKnowledge) -> &str {
    let Some(sound) = x.sound else { return "an unknown sound" };
    match sound {
        Sound::Attack              => "something attack!",
        Sound::Call(Call::Command) => "someone shout!",
        Sound::Call(Call::Help)    => "a call for help",
        Sound::Call(Call::Warning) => "a warning",
        Sound::Move                => "something move",
        Sound::Sniff               => "something sniff",
    }
}

fn describe_trace<'a>(x: &Trace<'a>) -> &'a str {
    match x {
        Trace::Actor(x) => describe_actor(x),
        Trace::Noise(x) => describe_noise(x),
    }
}

fn rivals(me: &Actor) -> Vec<Trace<'_>> {
    let mut result = vec![];
    for other in &me.known.actors {
        if !other.visible() { break; }
        if !other.friend() { result.push(Trace::Actor(other)); }
    }
    for other in &me.known.noises {
        if can_target_noise(me, other) { result.push(Trace::Noise(other)); }
    }
    let pos = me.pos;
    result.sort_by_cached_key(|x| ((x.pos - pos).len_l2_squared(), x.pos.0, x.pos.1));
    result
}

fn within_map_bounds(ui: &UI, me: &Actor, point: Point) -> bool {
    let size = ui.get_map_size();
    let Delta(x, y) = point - ui.get_map_offset(me);
    0 <= x && x < size.0 && 0 <= y && y < size.1
}

//////////////////////////////////////////////////////////////////////////////

// Layout

struct Layout {
    log: Rect,
    map: Rect,
    choice: Rect,
    rivals: Rect,
    status: Rect,
    target: Rect,
    bounds: Point,
}

impl Default for Layout {
    fn default() -> Self {
        let kl = UI::render_key('a').chars().count() as i32;
        assert!(kl == UI_KEY_SPACE);

        let side = 2 * FOV_RADIUS_PC_.radius + 1;
        let (x, y) = (side, side);

        let ss = UI_STATUS_SIZE;
        let (col, row) = (UI_COL_SPACE, UI_ROW_SPACE);
        let w = 2 * x + 2 + 2 * (ss + kl + 2 * col);
        let h = y + 2 + row + UI_LOG_SIZE + row + 1;
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

        Self { log, map, choice, rivals, status, target, bounds }
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
    delta: Delta,
    drops: VecDeque<RainDrop>,
    path: Vec<Delta>,
    lightning: i32,
    thunder: i32,
}

//////////////////////////////////////////////////////////////////////////////

// Targeting UI

enum TargetData {
    FarLook,
    Attack { summon: usize, attack: &'static Attack },
    Summon { team: usize, range: Bound },
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

fn can_target_actor(actor: &ActorKnowledge) -> bool {
    actor.visible() && !actor.friend()
}

fn can_target_noise(me: &Actor, noise: &NoiseKnowledge) -> bool {
    me.known.get(noise.pos).trace().map(|x| x.uid()) == Some(noise.uid)
}

fn can_target_trace(me: &Actor, trace: &Trace) -> bool {
    match trace {
        Trace::Actor(x) => can_target_actor(x),
        Trace::Noise(x) => can_target_noise(me, x),
    }
}

fn get_summon(me: &Actor, summon: usize) -> Option<&ActorKnowledge> {
    me.summons.get(summon).and_then(|&x| me.known.actor(x))
}

fn get_teammate(me: &Actor, team: usize) -> Option<&Individual> {
    if let Teammate::In(x) = me.team.get(team)? { Some(x) } else { None }
}

fn get_teammate_moves(me: &Actor, team: usize) -> TileFlags {
    get_teammate(me, team).map_or(TileFlags::Empty, |x| x.species.moves)
}

fn init_target(data: TargetData, source: Point, target: Point) -> Box<Target> {
    let (error, frame, okay_until, path) = ("".into(), 0, 0, vec![]);
    Box::new(Target { data, error, frame, okay_until, path, source, target })
}

fn init_summon_target(me: &Actor, team: usize) -> Box<Target> {
    let Actor { pos, dir, .. } = *me;
    let known = &*me.known;
    let moves = get_teammate_moves(me, team);
    let follower = Follower { pos, moves };
    let defender = ChooseDefenseSquare(me, &follower);
    let data = TargetData::Summon { team, range: SUMMON_RANGE };
    let mut target = init_target(data, pos, pos);

    let okay = |p: Point, swap: bool, target: &mut Target| {
        if p == pos || !CheckFollowerSquare(me, &follower, p) { return false; }
        if swap != known.get(p).occupied() { return false; }
        update_target(me, target, p);
        target.error.is_empty()
    };

    let closest = |p: Point, swap: bool, target: &mut Target| {
        let mut best = (None, i64::MAX);
        for dx in -2..=2 {
            for dy in -2..=2 {
                let point = pos + Delta(dx, dy);
                let score = (p - point).len_l2_squared();
                if score >= best.1 || !okay(point, swap, target) { continue; }
                best = (Some(point), score)
            }
        }
        best.0
    };

    let run = |swap: bool, target: &mut Target| {
        if let Some(x) = defender {
            let line = LOS(pos, x);
            for &p in line.iter().skip(1).rev() {
                if okay(p, swap, target) { return Some(p); }
            }
            if let Some(x) = closest(x, swap, target) { return Some(x); }
        }

        let best = pos + dir.scale(2);
        let next = pos + dir.scale(1);
        if okay(best, swap, target) { return Some(best); }
        if okay(next, swap, target) { return Some(next); }
        closest(best, swap, target)
    };

    // Try to choose a spot to summon this teammate without swapping out one
    // that's already out. If that fails, look for a swap.
    let best = run(false, &mut target).or_else(|| run(true, &mut target)).unwrap_or(pos);
    update_target(me, &mut target, best);
    target
}

fn update_target(me: &Actor, target: &mut Target, update: Point) {
    let known = &*me.known;
    let los = LOS(target.source, update);

    target.error.clear();
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
        TargetData::Attack { .. } => {
            let cell = known.get(update);
            if !cell.visible() {
                target.error = "You can't see a clear path there.".into();
            } else if let Some(x) = cell.actor() && x.friend() {
                target.error = "That target is friendly.".into();
            }
            let okay = target.error.is_empty();
            target.okay_until = if okay { target.path.len() } else { 0 };
        }
        &TargetData::Summon { team, range } => {
            let moves = get_teammate_moves(me, team);
            if target.path.is_empty() {
                target.error = "There's something in the way.".into();
            }
            for (i, &x) in target.path.iter().enumerate() {
                let cell = known.get(x);
                let last = i + 1 == target.path.len();
                let moves = if last { moves } else { SUMMON_MOVES };
                let friend = cell.actor().map_or(false, |x| x.friend());

                let free = match cell.status_for(moves) {
                    Status::Blocked => false,
                    Status::Occupied => friend && last,
                    Status::Free | Status::Unknown => true,
                };

                if !free {
                    target.error = "There's something in the way.".into();
                } else if !range.contains(x - target.source) {
                    target.error = "You can't throw that far.".into();
                } else if !cell.visible() {
                    target.error = "You can't see a clear path there.".into();
                } else if last && !friend && !cell.can_see_actor_at() {
                    target.error = "That cell may be occupied.".into();
                }
                if !target.error.is_empty() { break; }
                target.okay_until = i + 1;
            }
        }
    }
}

fn select_valid_target(ui: &mut UI, me: &Actor) -> Option<UID> {
    let target = ui.target.as_ref()?;
    let cell = me.known.get(target.target);
    let (actor, trace) = (cell.actor(), cell.trace());
    let summon = actor.and_then(|x| me.summons.iter().position(|&y| x.eid == y));

    match &target.data {
        TargetData::FarLook => {
            let trace = trace?;
            if can_target_trace(me, &trace) { Some(trace.uid()) } else { None }
        }
        &TargetData::Attack { summon, attack } => {
            let seen = false;
            let eid = actor.map(|x| x.eid);
            let loc = Location { pos: target.target, time: me.known.time() };
            let command = Command::Attack(attack, AttackTarget { eid, loc, seen });
            ui.action = Some(Action::Shout { summon, command });

            let trace = trace?;
            if can_target_trace(me, &trace) { Some(trace.uid()) } else { None }
        }
        &TargetData::Summon { team, .. } => {
            ui.action = Some(match summon {
                Some(summon) => Action::Switch { summon, team, quiet: false, fallback: false },
                None => Action::Summon { team, dir: target.target - me.pos },
            });
            ui.focus
        }
    }
}

//////////////////////////////////////////////////////////////////////////////

// UI inputs

fn process_summon_input(ui: &mut UI, me: &Actor, input: Input) {
    let Some(menu) = ui.summon.as_mut() else { return };

    // Cancel out of the dialog:

    if input == Input::Escape {
        ui.log.log_neutral("Canceled.");
        ui.summon = None;
        return;
    }

    // Use direction keys to scroll up and down:

    let n = me.team.len();
    let dir = if let Input::Char(x) = input { get_direction(x) } else { None };

    if let Some(dir) = dir && dir.0 == 0 {
        menu.choice += dir.1;
        if menu.choice >= n as i32 { menu.choice = 0; }
        if menu.choice < 0 { menu.choice = max(n as i32 - 1, 0); }
        return;
    }

    // Select an entry with either its key or Enter; else, fail.

    let enter = input == Input::Char('\n') || input == Input::Char('.');
    let chosen = if enter {
        Some(menu.choice as usize)
    } else {
        PARTY_KEYS.iter().position(|x| input == Input::Char(*x))
    };
    let Some(chosen) = chosen else { return };

    // If the given index isn't a valid party member, fail.

    let Some(teammate) = me.team.get(chosen) else {
        if n > 0 {
            ui.log.log_failure(format!("You are only carrying {} Pokemon!", n));
        } else {
            ui.log.log_failure("You aren't carrying any Pokemon!");
        };
        return;
    };

    // If the given party member is already out, fail.

    let teammate = match teammate {
        Teammate::In(x) => x,
        Teammate::Out(x) => {
            let other = me.known.actor(*x);
            let error = format!("{} is already out!", other.map_or("?", describe_actor));
            ui.log.log_failure(error);
            return;
        }
    };

    // If the given party member has fainted, fail.

    if teammate.cur_hp == 0 {
        let error = format!("{} has no strength left!", teammate.desc());
        ui.log.log_failure(error);
        return;
    };

    // Choose either a switch or a new summon.

    if let Some(summon) = menu.replacing {
        let command = Command::Switch(chosen);
        ui.action = Some(Action::Shout { summon, command });
        ui.summon = None;
        return;
    }

    let target = init_summon_target(me, chosen);
    let message = format!("Choose where to summon {}:", teammate.desc());
    ui.log.log_neutral(message);
    ui.target = Some(target);
    ui.summon = None;
}

fn process_regular_input(ui: &mut UI, me: &mut Actor, input: Input) -> bool {
    let index = SUMMON_KEYS.iter().position(|&x| input == Input::Char(x));
    if let Some(i) = index && i >= me.summons.len() {
        ui.log.log_neutral("Choose a Pokemon to send out with J/K:");
        ui.summon = Some(SummonMenu { choice: 0, replacing: None });
        return true;
    } else if let Some(i) = index {
        ui.log.log_neutral("Choose a command with J/K:");
        ui.menu = Some(Menu { choice: 0, summon: i });
        return true;
    }

    let Input::Char(ch) = input else { return false; };

    if ch == 'c' {
        me.sneaking = !me.sneaking;
        return true;
    }

    let Some(dir) = get_direction(ch) else { return false };

    if dir == dirs::NONE {
        ui.action = Some(Action::Idle);
        return true;
    }

    let turns = if me.sneaking { 2. } else { 1. };
    ui.action = Some(Action::Move { look: dir, step: dir, turns });
    true
}

fn process_ui_input(ui: &mut UI, me: &mut Actor, input: Input) -> bool {
    let known = &*me.known;
    let tab = input == Input::Char('\t') || input == Input::BackTab;
    let enter = input == Input::Char('\n') || input == Input::Char('.');

    // Mode: selecting a party member to summon:

    if ui.summon.is_some() {
        process_summon_input(ui, me, input);
        return true;
    }

    // Multi-use target selection helpers:

    let apply_tab = |prev: Option<UID>, off: bool| -> Option<UID> {
        let rivals = rivals(me);
        if rivals.is_empty() { return None; }

        let t = input == Input::Char('\t');
        let n = rivals.len() + if off { 1 } else { 0 };

        let next = prev.and_then(|x| rivals.iter().position(|y| y.uid() == x));
        let start = next.or_else(|| if off { Some(n - 1) } else { None });
        let index = start.map(|x| if t { x + n + 1 } else { x + n - 1 } % n)
                         .unwrap_or_else(|| if t { 0 } else { n - 1 });
        if index < rivals.len() { Some(rivals[index].uid()) } else { None }
    };

    let get_initial_target = |source: Point| -> Point {
        let known = &*me.known;
        let trace = ui.focus.and_then(|x| known.trace(x));
        if let Some(trace) = trace && can_target_trace(me, &trace) { return trace.pos; }
        if let Some(rival) = rivals(me).first() { return rival.pos; }
        source
    };

    let get_updated_target = |target: Point| -> Option<Point> {
        if tab {
            let old_uid = known.get(target).trace().map(|x| x.uid());
            let new_uid = apply_tab(old_uid, false);
            return Some(known.trace(new_uid?)?.pos);
        }

        let ch = if let Input::Char(x) = input { Some(x) } else { None }?;
        let dir = get_direction(ch.to_lowercase().next().unwrap_or(ch))?;
        let scale = if ch.is_uppercase() { 4 } else { 1 };

        let mut prev = target;
        for _ in 0..scale {
            let next = prev + dir;
            if !within_map_bounds(ui, me, next) { break; }
            prev = next;
        }
        Some(prev)
    };

    // Mode: selecting a target on the map:

    if let Some(x) = &ui.target {
        let update = get_updated_target(x.target);
        if let Some(update) = update && update != x.target {
            let mut target = ui.target.take();
            target.as_mut().map(|x| update_target(me, x, update));
            ui.target = target;
        } else if enter {
            if x.error.is_empty() {
                ui.focus = select_valid_target(ui, me);
                ui.target = None;
            } else {
                ui.log.log_failure(&x.error);
            }
        } else if input == Input::Escape {
            if let TargetData::FarLook = x.data {
                let valid = x.error.is_empty();
                ui.focus = if valid { select_valid_target(ui, me) } else { None }
            }
            ui.log.log_neutral("Canceled.");
            ui.target = None;
        }
        return true;
    }

    // Mode: giving a command to a party member:

    if let Some(x) = &mut ui.menu {
        let summon = get_summon(me, x.summon).unwrap();
        let attacks = &summon.species.attacks;
        let num_attacks = attacks.len() as i32;
        let max_attacks = ATTACK_KEYS.len() as i32;

        let count = max_attacks + ACTION_KEYS.len() as i32;
        let valid = |x: i32| { 0 <= x && x < num_attacks || x >= max_attacks };

        let ch = if let Input::Char(x) = input { Some(x) } else { None };
        let dir = ch.and_then(get_direction);

        let chosen = if enter { x.choice } else {
            let mut keys = ATTACK_KEYS.iter().chain(&ACTION_KEYS);
            keys.position(|&x| ch == Some(x)).map_or(-1, |x| x as i32)
        };
        let action = ACTION_KEYS.get((chosen - max_attacks) as usize).copied();

        if let Some(dir) = dir && dir.0 == 0 {
            loop {
                x.choice += dir.1;
                if x.choice >= count { x.choice = 0; }
                if x.choice < 0 { x.choice = max(count - 1, 0); }
                if valid(x.choice) { break; }
            }
        } else if chosen >= 0 {
            const _: () = assert!(ACTION_KEYS.len() == 2);
            let name = describe_actor(summon);

            if !valid(chosen) {
                ui.log.log_failure(format!("{} does not have that attack.", name));
            } else if action == Some(RETURN_KEY) {
                let (summon, command) = (x.summon, Command::Return);
                ui.action = Some(Action::Shout { summon, command });
                ui.menu = None;
            } else if action == Some(SWITCH_KEY) {
                ui.log.log_neutral(format!("Choose who will replace {} with J/K:", name));
                ui.summon = Some(SummonMenu { choice: 0, replacing: Some(x.summon) });
                ui.menu = None;
            } else {
                let attack = attacks[chosen as usize];
                let update = get_initial_target(summon.pos);
                let data = TargetData::Attack { summon: x.summon, attack };
                let mut target = init_target(data, summon.pos, update);
                update_target(me, &mut target, update);
                ui.log.log_neutral("Use the movement keys to select a target:");
                ui.target = Some(target);
                ui.menu = None;
            }
        } else if input == Input::Escape {
            ui.log.log_neutral("Canceled.");
            ui.menu = None;
        }
        return true;
    }

    // Regular mode: no modals active:

    if tab {
        ui.focus = apply_tab(ui.focus, true);
        return true;
    } else if input == Input::Escape {
        ui.focus = None;
        return true;
    }

    if input == Input::Char('x') {
        let source = me.pos;
        let update = get_initial_target(source);
        let mut target = init_target(TargetData::FarLook, source, update);
        update_target(me, &mut target, update);
        ui.log.log_neutral("Use the movement keys to examine a location:");
        ui.target = Some(target);
        return true;
    }

    process_regular_input(ui, me, input)
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

    pub fn log_append<S: Into<String>>(&mut self, text: S) {
        if let Some(x) = self.lines.last_mut() {
            x.text.push_str(&format!(" {}", text.into()));
        } else {
            self.log_color(text, Color::white());
        }
    }

    pub fn log_notable<S: Into<String>>(&mut self, text: S) {
        self.log_color(text, UI_LOG_MENU);
    }

    pub fn log_neutral<S: Into<String>>(&mut self, text: S) {
        self.log_menu(text, UI_LOG_MENU);
    }

    pub fn log_failure<S: Into<String>>(&mut self, text: S) {
        self.log_menu(text, UI_LOG_FAILURE);
    }

    pub fn log_success<S: Into<String>>(&mut self, text: S) {
        self.log_menu(text, UI_LOG_SUCCESS);
        self.end_menu_logging();
    }

    pub fn end_menu_logging(&mut self) {
        self.lines.last_mut().map(|x| x.menu = false);
    }

    fn log_color<S: Into<String>, T: Into<Color>>(&mut self, text: S, color: T) {
        let (color, text) = (color.into(), text.into());
        self.lines.push(LogLine { color, menu: false, text });
        if self.lines.len() as i32 > UI_LOG_SIZE { self.lines.remove(0); }
    }

    fn log_menu<S: Into<String>, T: Into<Color>>(&mut self, text: S, color: T) {
        let (color, text) = (color.into(), text.into());
        if self.lines.last().map_or(false, |x| x.menu) { self.lines.pop(); }
        self.lines.push(LogLine { color, menu: true, text });
        if self.lines.len() as i32 > UI_LOG_SIZE { self.lines.remove(0); }
    }
}

//////////////////////////////////////////////////////////////////////////////

// Focus

struct Focused {
    active: bool,
    vision: Vision,
    tile: Option<&'static Tile>,
}

impl Default for Focused {
    fn default() -> Self {
        Self { active: false, vision: Vision::new(FOV_RADIUS_NPC), tile: None }
    }
}

impl Focused {
    fn update(&mut self, me: &Actor, target: &Trace) {
        let known = &*me.known;
        let pos = target.pos;

        if let Trace::Actor(x) = target && !x.asleep() {
            let opacity = |x| known.get(x).tile().map_or(0, |x| x.opacity());
            self.vision.compute(&VisionArgs { pos, dir: x.dir, opacity });
        } else {
            self.vision.clear();
            self.vision.set_center(pos);
        }
        self.tile = known.get(pos).tile();
    }
}

//////////////////////////////////////////////////////////////////////////////

// UI

struct Callout {
    color: Color,
    point: Point,
    text: &'static str,
}

struct Menu {
    choice: i32,
    summon: usize,
}

struct SummonMenu {
    choice: i32,
    replacing: Option<usize>
}

struct SummonOption<'a> {
    out: bool,
    selected: bool,
    name: &'a str,
    hp: f64,
    pp: f64,
}

pub struct Effect<'a> {
    pub frame: &'a Frame,
    pub known: &'a Knowledge,
    pub mask: &'a [bool],
    pub sources: HashSet<Point>,
    pub targets: HashSet<Point>,
}

#[derive(Default)]
pub struct UI {
    frame: usize,
    layout: Layout,

    // Animations:
    moves: HashMap<Point, MoveAnimation>,
    rainfall: Option<Rainfall>,

    // Modal components:
    menu: Option<Menu>,
    focus: Option<UID>,
    summon: Option<SummonMenu>,
    target: Option<Box<Target>>,
    focused: Focused,

    // Used by state updates:
    pub action: Option<Action>,
    pub log: Log,
}

impl UI {
    // Rendering entry point:

    pub fn render(&self, buffer: &mut Buffer, me: &Actor, effect: Option<&Effect>) {
        if buffer.size() == Point::ORIGIN {
            *buffer = Matrix::new(self.layout.bounds, ' '.into());
        }
        buffer.fill(*buffer.default());
        self.render_layout(buffer);

        self.render_log(buffer);
        self.render_rivals(buffer, me);
        self.render_status(buffer, me);
        self.render_target(buffer, me);

        self.render_map(buffer, me, effect);
        self.render_weather(buffer, me, effect);

        if let Some(x) = &self.summon { self.render_summon_menu(buffer, me, x); }
    }

    // Update entry points:

    pub fn animate_move(&mut self, color: Color, point: Point, delta: i32) {
        let frame = -delta * UI_MOVE_FRAMES / 2;
        let manim = MoveAnimation { color, frame, limit: UI_MOVE_FRAMES };
        self.moves.insert(point, manim);
    }

    pub fn process_input(&mut self, me: &mut Actor, input: Input) -> bool {
        process_ui_input(self, me, input)
    }

    pub fn start_rain(&mut self, delta: Delta, count: usize) {
        let rainfall = Rainfall {
            ch: Glyph::ray(delta),
            delta,
            drops: VecDeque::with_capacity(count),
            path: DeltaLOS(delta),
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

    pub fn update_focus(&mut self, me: &Actor) {
        let known = &*me.known;
        let focus = self.focus.and_then(|x| known.trace(x));
        if focus.is_none() { self.focus = None; }

        // TODO: The code below uses a hack to clean up noise targets if an
        // if an entity moves into that cell. Instead, the player's Knowledge
        // update should apply this check.
        //
        // TODO: Attacking a noise source doesn't work correctly; it targets
        // the cell, not the entity.
        //
        // TODO: We should probably be able to attack noise sources even if
        // we don't have a clear line-of-sight to them.
        let focus = focus.filter(|x| match x {
            Trace::Noise(x) => can_target_noise(me, x),
            Trace::Actor(_) => true,
        });
        self.focus = focus.as_ref().map(|x| x.uid());

        let target = match &self.target {
            Some(x) => known.get(x.target).trace(),
            None => focus,
        };
        if let Some(target) = target && can_target_trace(me, &target) {
            self.focused.update(me, &target);
            self.focused.active = true;
        } else {
            self.focused.active = false;
        }
    }

    pub fn update_moves(&mut self, me: &Actor) {
        let known = &*me.known;
        self.moves.retain(|&pos, _| !known.get(pos).occupied())
    }

    pub fn update_target(&mut self, me: &Actor) -> bool {
        let Some(target) = &mut self.target else { return false };
        target.frame = (target.frame + 1) % UI_TARGET_FRAMES;
        self.update_focus(me);
        true
    }

    // Update helpers:

    fn update_weather(&mut self, pos: Point, rng: &mut RNG) {
        let Some(r) = &mut self.rainfall else { return; };

        let frame = self.frame;
        while let Some(x) = r.drops.front() && x.frame < frame {
            r.drops.pop_front();
        }
        let total = r.drops.capacity();
        let denom = max(r.delta.1, 1);
        let delta = denom as usize;
        let extra = (frame + 1) * total / delta - (frame * total) / delta;
        for _ in 0..min(extra, total - r.drops.len()) {
            let x = rng.random_range(0..denom);
            let y = rng.random_range(0..denom);
            let target_frame = frame + r.path.len() - 1;
            let target_point = Delta(x - denom / 2, y - denom / 2) + pos;
            r.drops.push_back(RainDrop { frame: target_frame, point: target_point });
        }

        assert!(r.lightning >= -1);
        if r.lightning == -1 {
            if rng.random::<f32>() < 0.002 { r.lightning = 10; }
        } else if r.lightning > 0 {
            r.lightning -= 1;
        }

        assert!(r.thunder >= 0);
        if r.thunder == 0 {
            if r.lightning == 0 && rng.random::<f32>() < 0.02 { r.thunder = 16; }
        } else {
            r.thunder -= 1;
            if r.thunder == 0 { r.lightning = -1; }
        }
    }

    // Rendering the map

    fn render_map(&self, buffer: &mut Buffer, me: &Actor, effect: Option<&Effect>) {
        let size = self.get_map_size();
        let slice = &mut Slice::new(buffer, self.layout.map);
        let offset = self.get_map_offset(me);
        let player = me.player;

        // Render each tile's base glyph, if it's known.
        slice.fill(Glyph::wide(' '));
        for y in 0..size.1 {
            for x in 0..size.0 {
                let point = Delta(x, y) + offset;
                let glyph = Self::render_tile(me, point, effect);
                slice.set(Point(2 * x, y), glyph);
            }
        }

        // Render moves that we've seen or heard.
        let remap = |point: Point, slice: &Slice| {
            let result = Point(2 * (point.0 - offset.0), point.1 - offset.1);
            if slice.contains(result) { Some(result) } else { None }
        };
        if player {
            for (&k, v) in &self.moves {
                if v.frame < 0 { continue; }
                let Some(p) = remap(k, slice) else { continue };

                let alpha = 1.0 - (v.frame as f64 / v.limit as f64);
                let color = v.color.fade(UI_MOVE_ALPHA * alpha);
                slice.set(p, slice.get(p).with_bg(color));
            }
        }

        // Helpers used for the map overlay UIs.
        let set = |slice: &mut Slice, point: Point, glyph: Glyph| {
            let Some(point) = remap(point, slice) else { return };
            slice.set(point, glyph);
        };
        let highlight = |slice: &mut Slice, point: Point, bg: Color| {
            let Some(point) = remap(point, slice) else { return };
            slice.set(point, slice.get(point).with_fg(Color::black()).with_bg(bg));
        };
        let brighten = |slice: &mut Slice, point: Point| {
            let Some(point) = remap(point, slice) else { return };
            let glyph = slice.get(point);
            let glyph = glyph.with_fg(glyph.fg().brighten(UI_FOV_BRIGHTEN));
            let glyph = glyph.with_bg(glyph.bg().brighten(UI_FOV_BRIGHTEN));
            slice.set(point, glyph);
        };

        // Render the focused actor on the map.
        let mut skip_first = false;
        let focused_vision = self.focused.vision.get_points_seen();
        if self.focused.active &&
           let Some(&point) = focused_vision.first() &&
           !effect.map_or(false, |x| x.sources.contains(&point)) {
            highlight(slice, point, Color::gray(UI_TARGET_SHADE));
            skip_first = true;
        }

        // Render all of the current animation's particles.
        let mut render_particle = |p: Point, r: &RenderData| match r {
            RenderData::Dummy => {},
            &RenderData::Text(text, color) => {
                let callout = Callout { color, point: p, text };
                self.render_callout(me, &callout, slice);
            },
            &RenderData::Flash(color) => {
                let Some(p) = remap(p, slice) else { return };
                slice.set(p, slice.get(p).with_fg(Color::black()).with_bg(color));
            },
            &RenderData::Glyph(glyph) => {
                let Some(p) = remap(p, slice) else { return };
                slice.set(p, glyph);
            },
        };
        if let Some(effect) = effect {
            assert!(effect.frame.len() == effect.mask.len());
            effect.frame.iter().zip(effect.mask).filter(|x| *x.1).for_each(|x| match &x.0.data {
                ParticleData::Light(..) => {},
                ParticleData::Shift(..) => {},
                ParticleData::Sight(r) => render_particle(x.0.point, r),
                ParticleData::Sound(_, r) => render_particle(x.0.point, r),
            });
        }

        // If we're still alive, render arrows showing NPC facing.
        if effect.is_none() && me.cur_hp > 0 {
            self.render_arrows(me, offset, slice);
        }

        // Render the targeting UI on the map.
        if let Some(target) = &self.target {
            // Fade out any cells that are too far to be a valid target.
            if let TargetData::Summon { range, .. } = target.data {
                let bg = Color::black();
                let fg = Color::white().fade(UI_REMEMBERED);
                let mut gray_out = |point: Point| {
                    slice.set(point, slice.get(point).with_fg(fg).with_bg(bg));
                };
                for y in 0..size.1 {
                    for x in 0..size.0 {
                        let point = Delta(x, y) + offset;
                        if range.contains(point - me.pos) { continue; }
                        gray_out(Point(2 * x + 0, y));
                        gray_out(Point(2 * x + 1, y));
                    }
                }
            }

            // Use bright highlights for the source and target cells.
            let shade = Color::gray(UI_TARGET_SHADE);
            let color = if target.error.is_empty() { 0xffff00 } else { 0xff0000 };
            highlight(slice, target.source, shade);
            highlight(slice, target.target, color.into());

            // Draw an animated line from the source to the target.
            let frame = target.frame >> 1;
            let count = UI_TARGET_FRAMES >> 1;
            let limit = target.path.len() as i32 - 1;
            let ch = Glyph::ray(target.target - target.source);
            for i in 0..limit {
                if !((i + count - frame) % count < 2) { continue; }
                let point = target.path[i as usize];
                let valid = (i as usize) < target.okay_until;
                let color = if valid { 0xffff00 } else { 0xff0000 };
                set(slice, point, Glyph::wdfg(ch, color));
            }
        }

        // Render an estimate of the focused actor's FOV on the map.
        if self.focused.active {
            let skip = if skip_first { 1 } else { 0 };
            for &point in focused_vision.iter().skip(skip) {
                brighten(slice, point);
            }
        }
    }

    fn render_arrows(&self, me: &Actor, offset: Point, slice: &mut Slice) {
        let arrow_length = 3;
        let sleep_length = 2;
        let mut arrows = vec![];
        for other in &me.known.actors {
            if !other.visible() { continue; }
            if !other.asleep() && !can_target_actor(other) { continue; }

            let (pos, dir) = (other.pos, other.dir);
            let mut ch = Glyph::ray(dir);
            let mut diff = dir.normalize(arrow_length as f64);
            if other.asleep() { (ch, diff) = ('Z', Delta(0, -sleep_length)); }
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

    fn render_callout(&self, me: &Actor, callout: &Callout, slice: &mut Slice) {
        let len = callout.text.len() as i32;
        let (shift, prefix, suffix) = if callout.point.0 > me.pos.0 {
            (1, Some('-'), None)
        } else {
            (-len / 2 - 1, if len % 2 == 0 { Some(' ') } else { None }, Some('-'))
        };
        let offset = self.get_map_offset(me);
        let Delta(dx, dy) = callout.point + Delta(shift, 0) - offset;

        slice.set_cursor(Point(2 * dx, dy));
        slice.set_fg(Some(callout.color));
        if let Some(c) = prefix { slice.write_chr(c); }
        callout.text.chars().for_each(|x| { slice.write_chr(Glyph::bold(x)); });
        if let Some(c) = suffix { slice.write_chr(c); }
        slice.set_fg(None);
    }

    fn render_weather(&self, buffer: &mut Buffer, me: &Actor, effect: Option<&Effect>) {
        let Some(rainfall) = &self.rainfall else { return };

        let slice = &mut Slice::new(buffer, self.layout.map);
        let offset = self.get_map_offset(me);

        let size = self.get_map_size();
        let base = Tile::get('~').glyph.fg();
        let known = effect.map_or(&*me.known, |x| x.known);

        for drop in &rainfall.drops {
            let index = drop.frame - self.frame;
            let Some(&delta) = rainfall.path.get(index) else { continue; };

            let (ground, point) = (index == 0, drop.point - delta);
            let cell = if ground { known.get(point) } else { known.default() };
            if ground && !cell.visible() { continue; }

            let Delta(x, y) = point - offset;
            let ch = if ground { 'o' } else { rainfall.ch };
            let color = Self::apply_light(&cell, base, /*actor=*/false);
            let glyph = Glyph::wdfg(ch, color);
            slice.set(Point(2 * x, y), glyph);
        }

        if rainfall.lightning > 0 {
            let value = min(255, 84 * rainfall.lightning / 2);
            let color = Color::from((value << 16) | (value << 8) | value);
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
                let delta = Delta(shift - 1, 0);
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

    // Rendering each section of the UI:

    fn render_log(&self, buffer: &mut Buffer) {
        let slice = &mut Slice::new(buffer, self.layout.log);
        for x in &self.log.lines { slice.set_fg(Some(x.color)).write_str(&x.text).newline(); }
    }

    fn render_rivals(&self, buffer: &mut Buffer, me: &Actor) {
        let slice = &mut Slice::new(buffer, self.layout.rivals);

        let mut rivals = rivals(me);
        rivals.truncate(max(slice.size().1, 0) as usize / 2);
        let rivals = rivals;

        for rival in rivals {
            let mut glyph = Glyph::wide('?');
            let mut hp_color = Color::white();
            let mut hp_text = String::default();

            if let Trace::Actor(x) = rival {
                glyph = x.species.glyph;
                hp_color = Self::hp_color(x.hp);
                hp_text = format!("{}%", max((100.0 * x.hp).floor() as i32, 1));
            }

            let name = describe_trace(&rival);
            let (sn, sh) = (name.chars().count(), hp_text.chars().count());
            let ss = max(16 - sn as i32 - sh as i32, 0) as usize;

            slice.newline();
            slice.write_chr(glyph).space().write_str(name);
            slice.spaces(ss).set_fg(Some(hp_color)).write_str(&hp_text).newline();

            let targeted = match &self.target {
                Some(x) => x.target == rival.pos,
                None => self.focus == Some(rival.uid()),
            };
            if targeted {
                let start = slice.get_cursor() - Point(0, 1);
                let (fg, bg) = (Color::white(), Color::gray(UI_TARGET_FOV_SHADE));
                for x in 0..UI_STATUS_SIZE {
                    let p = start + Point(x, 0);
                    slice.set(p, slice.get(p).with_fg(fg).with_bg(bg));
                }
            }
        }
    }

    fn render_status(&self, buffer: &mut Buffer, me: &Actor) {
        let slice = &mut Slice::new(buffer, self.layout.status);
        let known = &*me.known;

        if let Some(actor) = known.actor(me.eid) {
            let key = if self.menu.is_some() { '-' } else { PLAYER_KEY };
            self.render_actor(Some(key), None, actor, slice);
        }

        for (i, &key) in SUMMON_KEYS.iter().enumerate() {
            let key = if self.menu.is_some() { '-' } else { key };
            let eid = me.summons.get(i);
            if let Some(actor) = eid.and_then(|&x| known.actor(x)) {
                self.render_actor(Some(key), None, actor, slice);
                if let Some(x) = &self.menu && x.summon == i {
                    self.render_menu(slice, x.choice, actor);
                }
            } else {
                self.render_empty_option(key, 0, slice);
            }
        }
    }

    fn render_target(&self, buffer: &mut Buffer, me: &Actor) {
        let slice = &mut Slice::new(buffer, self.layout.target);
        let known = &*me.known;

        if self.target.is_none() && self.focus.is_none() {
            let fg = Some(UI_GRAY_OPTION.into());
            slice.newline();
            slice.set_fg(fg).write_str("No target selected.").newline();
            slice.newline();
            slice.set_fg(fg).write_str("[x] examine your surroundings").newline();
            return;
        }

        let (tile, header, actor, noise) = match &self.target {
            Some(x) => {
                let cell = known.get(x.target);
                let tile = cell.tile();
                let noise = cell.noise();
                let actor = cell.actor().filter(|x| x.sensed());
                let header = match &x.data {
                    TargetData::FarLook => "Examining...".into(),
                    &TargetData::Attack { summon, attack } => {
                        let other = get_summon(me, summon).unwrap();
                        format!("Using {}'s {}...", describe_actor(other), attack.name)
                    },
                    &TargetData::Summon { team, .. } => {
                        let other = get_teammate(me, team).unwrap();
                        format!("Sending out {}...", other.desc())
                    }
                };
                (tile, header, actor, noise)
            }
            None => {
                let tile = self.focused.tile;
                let header = "Last target:".into();
                let trace = self.focus.and_then(|x| known.trace(x));
                let actor = trace.as_ref().and_then(|x| x.actor());
                let noise = trace.as_ref().and_then(|x| x.noise());
                (tile, header, actor, noise)
            },
        };

        let gray = actor.map_or(false, |x| !x.sensed());
        let text = if actor.is_some() { "Standing on: " } else { "You see: " };
        let fg = if gray { Some(UI_GRAY_OPTION.into()) } else { None };
        let suffix = if gray { " (remembered)" } else { "" };

        slice.newline();
        slice.set_fg(fg).write_str(&header).write_str(suffix).newline();

        if let Some(actor) = actor {
            self.render_actor(None, fg, actor, slice);
        } else {
            slice.newline();
        }

        slice.set_fg(fg).write_str(text);
        if let Some(tile) = tile {
            slice.write_chr(tile.glyph).space();
            slice.write_chr('(').write_str(tile.description).write_chr(')').newline();
        } else {
            slice.write_str("(unseen location)").newline();
        }

        if actor.is_none() && let Some(noise) = noise {
            let desc = describe_noise(noise);
            slice.newline().write_str("You heard: ").write_str(desc).newline().newline();
        }
    }

    fn render_summon_menu(&self, buffer: &mut Buffer, me: &Actor, menu: &SummonMenu) {
        let default = *buffer.default();
        self.render_box(buffer, &self.layout.choice);
        let slice = &mut Slice::new(buffer, self.layout.choice);
        slice.fill(default);

        let options = &me.team;
        for (i, &key) in PARTY_KEYS.iter().enumerate() {
            let selected = menu.choice == i as i32;
            match options.get(i) {
                Some(Teammate::In(x)) => {
                    let (pp, name) = (1.0, x.desc());
                    let hp = x.cur_hp as f64 / max(x.max_hp, 1) as f64;
                    let option = SummonOption { out: false, selected, name, hp, pp };
                    self.render_option(key, &option, slice)
                }
                Some(&Teammate::Out(x)) => {
                    let x = me.known.actor(x).unwrap();
                    let (hp, pp, name) = (x.hp, x.pp, describe_actor(x));
                    let option = SummonOption { out: true, selected, name, hp, pp };
                    self.render_option(key, &option, slice);
                },
                None => self.render_empty_option(key, UI_COL_SPACE + 1, slice),
            }
        }
    }

    // High-level private rendering helpers

    fn render_menu(&self, slice: &mut Slice, index: i32, summon: &ActorKnowledge) {
        let spaces = UI::render_key('-').chars().count();

        for (i, &key) in ATTACK_KEYS.iter().chain(&ACTION_KEYS).enumerate() {
            let (active, name) = if i < ATTACK_KEYS.len() {
                let attack = summon.species.attacks.get(i);
                let name = attack.map_or("---", |x| x.name);
                (attack.is_some(), name)
            } else {
                const _: () = assert!(ACTION_KEYS.len() == 2);
                let name = if key == SWITCH_KEY { "Switch" } else { "Call back" };
                (true, name)
            };

            let prefix = if index == i as i32 { " > " } else { "  " };
            let fg = if active { None } else { Some(UI_GRAY_OPTION.into()) };

            slice.set_fg(fg).spaces(spaces).write_str(prefix);
            slice.write_str(&Self::render_key(key)).write_str(name);
            slice.newline().newline();
        }
    }

    fn render_empty_option(&self, key: char, space: i32, slice: &mut Slice) {
        let n = space as usize;
        let fg = Some(UI_GRAY_OPTION.into());
        let prefix = UI::render_key(key);

        slice.newline();
        slice.set_fg(fg).spaces(n).write_str(&prefix).write_str("---").newline();
        slice.newline().newline().newline();
    }

    fn render_option(&self, key: char, option: &SummonOption, slice: &mut Slice) {
        let SummonOption { out, selected, name, hp, pp } = *option;
        let (hp_color, pp_color) = (Self::hp_color(hp), Self::pp_color());
        let fg = if !out && hp > 0. { None } else { Some(UI_GRAY_OPTION.into()) };

        let x = if selected { 1 } else { 0 };
        let arrow = if selected { '>' } else { ' ' };

        let prefix = UI::render_key(key);
        let n = prefix.chars().count() + (UI_COL_SPACE + 1) as usize;
        let w = slice.size().0 - (n as i32) - 2 * UI_COL_SPACE - 6;
        let status_bar_line = |p: &str, v: f64, c: Color, s: &mut Slice| {
            s.set_fg(fg).spaces(n + x).write_str(p);
            self.render_bar(v, c, w, s);
            s.newline();
        };

        slice.newline();
        slice.spaces(UI_COL_SPACE as usize).write_chr(arrow).spaces(x);
        slice.set_fg(fg).write_str(&prefix).write_str(name).newline();
        status_bar_line("HP: ", hp, hp_color, slice);
        status_bar_line("PP: ", pp, pp_color, slice);
        slice.newline();
    }

    fn render_actor(&self, key: Option<char>, fg: Option<Color>,
                    actor: &ActorKnowledge, slice: &mut Slice) {
        let prefix = key.map_or_default(UI::render_key);
        let n = prefix.chars().count();
        let w = UI_STATUS_SIZE - 6;
        let status_bar_line = |p: &str, v: f64, c: Color, s: &mut Slice| {
            self.render_bar(v, c, w, s.set_fg(fg).spaces(n).write_str(p));
            s.newline();
        };

        let name = describe_actor(actor);
        let (hp, pp) = (actor.hp, actor.pp);
        let (hp_color, pp_color) = (Self::hp_color(hp), Self::pp_color());

        slice.newline();
        slice.set_fg(fg).write_str(&prefix).write_str(name).newline();
        status_bar_line("HP: ", hp, hp_color, slice);
        if !actor.species.human() {
            status_bar_line("PP: ", pp, pp_color, slice);
        } else {
            slice.spaces(n + 4);
            for &ready in &actor.team {
                slice.set_fg(if ready { None } else { Some(UI_GRAY_OPTION.into()) });
                slice.write_str("* ");
            }
            slice.newline();
        }
        slice.newline();
    }

    // Low-level private rendering helpers

    fn get_map_size(&self) -> Point {
        Point(self.layout.map.size.0 / 2, self.layout.map.size.1)
    }

    fn get_map_offset(&self, me: &Actor) -> Point {
        let size = self.get_map_size();
        me.pos - Delta(size.0 / 2, size.1 / 2)
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
        buffer.set(rect.root + Delta(-1, -1), Glyph::chfg('┌', color));
        buffer.set(rect.root + Delta( w, -1), Glyph::chfg('┐', color));
        buffer.set(rect.root + Delta(-1,  h), Glyph::chfg('└', color));
        buffer.set(rect.root + Delta( w,  h), Glyph::chfg('┘', color));

        let tall = Glyph::chfg('│', color);
        let flat = Glyph::chfg('─', color);
        for x in 0..w {
            buffer.set(rect.root + Delta(x, -1), flat);
            buffer.set(rect.root + Delta(x,  h), flat);
        }
        for y in 0..h {
            buffer.set(rect.root + Delta(-1, y), tall);
            buffer.set(rect.root + Delta( w, y), tall);
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
            buffer.set(pos + Delta(x, 0), dashes);
        }
        for (i, c) in text.chars().enumerate() {
            buffer.set(pos + Delta(i as i32 + shift, 0), Glyph::chfg(c, color));
        }
        for x in prefix_width..width {
            buffer.set(pos + Delta(x, 0), dashes);
        }
    }

    // Static helpers

    fn actor_glyph(actor: &ActorKnowledge) -> Glyph {
        let sneaking = actor.species.human() && actor.sneaking();
        if sneaking { Glyph::wide('e') } else { actor.species.glyph }
    }

    fn noise_glyph(freshness: f64) -> Glyph {
        Glyph::wdfg('?', Color::white().fade(0.25 + 0.5 * freshness))
    }

    fn teammate_glyph(actor: &ActorKnowledge) -> Glyph {
        Glyph::wdfg('?', actor.species.glyph.fg())
    }

    fn combined_glyph(actor: &ActorKnowledge, tile: &Tile) -> Glyph {
        if actor.hp == 0. { return Glyph::wdfg('%', 0xff0000); }

        let glyph = Self::actor_glyph(actor);
        let cover = tile.is_cover() && actor.hidden_by_cover();

        if cover { glyph.with_fg(tile.glyph.fg()) } else { glyph }
    }

    fn hp_color(hp: f64) -> Color {
        if hp <= 0.25 { return 0xff0000.into(); }
        if hp <= 0.50 { return 0xffff00.into(); }
        0x00a800.into()
    }

    fn pp_color() -> Color {
        0x54a8fc.into()
    }

    fn render_key(key: char) -> String {
        format!("[{}] ", key)
    }

    fn apply_light(cell: &PointLookup, color: Color, actor: bool) -> Color {
        let light = cell.light();
        let shade = cell.shade();

        if shade && !light { return color.fade(UI_SHADOW); }

        if actor { return color; }

        let base = if shade { UI_SHADOW } else { UI_SUNLIGHT };
        let extra = base + if light { UI_EXTRA_LIGHT } else { 0. };
        let (r, g, b) = (extra, extra, base);
        color.apply_light(r, g, b)
    }

    pub fn render_tile(me: &Actor, point: Point, effect: Option<&Effect>) -> Glyph {
        let known = effect.map_or(&*me.known, |x| x.known);
        let is_source = effect.map_or(false, |x| x.sources.contains(&point));
        let is_target = effect.map_or(false, |x| x.targets.contains(&point));
        let is_source = is_source && !is_target;

        let cell = known.get(point);
        let noise = me.known.get(point).noise();
        let noise = if is_source { None } else { noise };

        let actor = cell.actor().filter(|x| x.sensed());
        let actor = if is_source { None } else { actor };

        let freshness = noise.map(|x| x.freshness(&me.known));
        let freshness = freshness.unwrap_or(if is_target { 1. } else { -1. });

        // Special cases for non-sight-based actor knowledge:
        if let Some(x) = actor && !x.visible() { return Self::teammate_glyph(x); }
        if actor.is_none() && freshness >= 0. { return Self::noise_glyph(freshness); }

        // Unseen cells are shown as a blank space:
        let Some(tile) = cell.tile() else { return Glyph::wide(' ') };

        // Default case: a visible cell.
        let glyph = if let Some(x) = actor && x.visible() {
            Self::combined_glyph(x, tile)
        } else if let Some(x) = cell.items().last() {
            show_item(x)
        } else {
            tile.glyph
        };

        let mut color = glyph.fg();
        if cell.visible() {
            color = Self::apply_light(&cell, color, actor.is_some());
        } else {
            color = Color::white().fade(UI_REMEMBERED);
        }
        glyph.with_fg(color)
    }
}
