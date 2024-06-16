use std::cmp::{max, min};
use std::collections::VecDeque;
use std::mem::{replace, swap};
use std::num::NonZeroU64;
use std::ops::{Index, IndexMut};

use lazy_static::lazy_static;
use rand::{Rng, SeedableRng};
use slotmap::{DefaultKey, Key, KeyData, SlotMap};

use crate::{or_continue, or_return, static_assert_size};
use crate::base::{Buffer, Color, Glyph};
use crate::base::{HashMap, LOS, Matrix, Point, dirs};
use crate::base::RNG;
use crate::knowledge::{Knowledge, Vision};

//////////////////////////////////////////////////////////////////////////////

// Constants

const FOV_RADIUS_NPC: i32 = 12;
const FOV_RADIUS_PC_: i32 = 21;

const LIGHT: Light = Light::Sun(Point(4, 1));
const WEATHER: Weather = Weather::Rain(Point(0, 64), 96);
const WORLD_SIZE: i32 = 100;

const UI_MAP_SIZE_X: i32 = 2 * FOV_RADIUS_PC_ + 1;
const UI_MAP_SIZE_Y: i32 = 2 * FOV_RADIUS_PC_ + 1;

#[derive(Eq, PartialEq)]
pub enum Input { Escape, BackTab, Char(char) }

//////////////////////////////////////////////////////////////////////////////

// Tile

const FLAG_NONE: u32 = 0;
const FLAG_BLOCKED: u32 = 1 << 0;
const FLAG_OBSCURE: u32 = 1 << 1;

pub struct Tile {
    pub flags: u32,
    pub glyph: Glyph,
    pub description: &'static str,
}
static_assert_size!(Tile, 24);

impl Tile {
    fn get(ch: char) -> &'static Tile { TILES.get(&ch).unwrap() }
    pub fn blocked(&self) -> bool { self.flags & FLAG_BLOCKED != 0 }
    pub fn obscure(&self) -> bool { self.flags & FLAG_OBSCURE != 0 }
}

impl PartialEq for &'static Tile {
    fn eq(&self, next: &&'static Tile) -> bool {
        *self as *const Tile == *next as *const Tile
    }
}

impl Eq for &'static Tile {}

lazy_static! {
    static ref TILES: HashMap<char, Tile> = {
        let items = [
            ('.', (FLAG_NONE,    Glyph::wdfg('.', 0x222), "grass")),
            ('"', (FLAG_OBSCURE, Glyph::wdfg('"', 0x120), "tall grass")),
            ('#', (FLAG_BLOCKED, Glyph::wdfg('#', 0x010), "a tree")),
            ('%', (FLAG_NONE,    Glyph::wdfg('%', 0x200), "flowers")),
            ('~', (FLAG_NONE,    Glyph::wdfg('~', 0x013), "water")),
        ];
        let mut result = HashMap::default();
        for (ch, (flags, glyph, description)) in items {
            result.insert(ch, Tile { flags, glyph, description });
        }
        result
    };
}

//////////////////////////////////////////////////////////////////////////////

// Entity

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
#[repr(transparent)]
pub struct EID(NonZeroU64);
static_assert_size!(Option<EID>, 8);

impl Default for EID {
    fn default() -> Self {
        to_eid(DefaultKey::null())
    }
}

fn to_key(eid: EID) -> DefaultKey {
    KeyData::from_ffi(eid.0.get()).into()
}

fn to_eid(key: DefaultKey) -> EID {
    EID(NonZeroU64::new(key.data().as_ffi()).unwrap())
}

#[derive(Default)]
struct EntityMap {
    map: SlotMap<DefaultKey, Entity>,
}

impl EntityMap {
    fn add(&mut self, args: &EntityArgs) -> EID {
        let key = self.map.insert_with_key(|x| Entity {
            eid: to_eid(x),
            glyph: args.glyph,
            known: Box::default(),
            player: args.player,
            pos: args.pos,
        });
        to_eid(key)
    }

    fn get(&self, eid: EID) -> Option<&Entity> {
        self.map.get(to_key(eid))
    }

    fn get_mut(&mut self, eid: EID) -> Option<&mut Entity> {
        self.map.get_mut(to_key(eid))
    }
}

impl Index<EID> for EntityMap {
    type Output = Entity;
    fn index(&self, eid: EID) -> &Self::Output {
        self.get(eid).unwrap()
    }
}

impl IndexMut<EID> for EntityMap {
    fn index_mut(&mut self, eid: EID) -> &mut Self::Output {
        self.get_mut(eid).unwrap()
    }
}

pub struct Entity {
    pub eid: EID,
    pub glyph: Glyph,
    pub known: Box<Knowledge>,
    pub player: bool,
    pub pos: Point,
}

struct EntityArgs {
    glyph: Glyph,
    player: bool,
    pos: Point,
}

//////////////////////////////////////////////////////////////////////////////

// Board

#[derive(Clone, Copy, Eq, PartialEq)]
pub enum Status { Free, Blocked, Occupied }

#[derive(Clone, Copy)]
pub struct Cell {
    pub eid: Option<EID>,
    pub shadow: i32,
    pub tile: &'static Tile,
}
static_assert_size!(Cell, 24);

enum Light { None, Sun(Point) }

enum Weather { None, Rain(Point, usize) }

struct Drop {
    frame: usize,
    point: Point,
}

struct Rain {
    ch: char,
    diff: Point,
    drops: VecDeque<Drop>,
    path: Vec<Point>,
    lightning: i32,
    thunder: i32,
}

pub struct Board {
    active_entity_index: usize,
    entity_order: Vec<EID>,
    entities: EntityMap,
    map: Matrix<Cell>,
    // Knowledge state
    known: Option<Box<Knowledge>>,
    npc_vision: Vision,
    _pc_vision: Vision,
    // Environmental effects
    light: Light,
    rain: Option<Rain>,
    shadow: Vec<Point>,
}

impl Board {
    fn new(size: Point, light: Light, weather: Weather) -> Self {
        let shadow = match light {
            Light::Sun(x) =>
                LOS(Point::default(), x).into_iter().skip(1).collect(),
            Light::None => vec![],
        };
        let rain = match weather {
            Weather::Rain(x, y) => Some(Rain {
                ch: Glyph::ray(x),
                diff: x,
                drops: VecDeque::with_capacity(y),
                path: LOS(Point::default(), x),
                lightning: -1,
                thunder: 0,
            }),
            Weather::None => None,
        };
        let cell = Cell { eid: None, shadow: 0, tile: Tile::get('#') };

        let mut result = Self {
            active_entity_index: 0,
            entity_order: vec![],
            entities: EntityMap::default(),
            map: Matrix::new(size, cell),
            // Knowledge state
            known: Some(Box::default()),
            npc_vision: Vision::new(FOV_RADIUS_NPC),
            _pc_vision: Vision::new(FOV_RADIUS_PC_),
            // Environmental effects
            light,
            rain,
            shadow,
        };
        result.update_edge_shadows();
        result
    }

    // Getters

    fn get_active_entity(&self) -> EID {
        self.entity_order[self.active_entity_index]
    }

    pub fn get_cell(&self, p: Point) -> &Cell { self.map.entry_ref(p) }

    pub fn get_entity(&self, eid: EID) -> Option<&Entity> { self.entities.get(eid) }

    fn get_size(&self) -> Point { self.map.size }

    fn get_status(&self, p: Point) -> Status {
        let Cell { eid, tile, .. } = self.get_cell(p);
        if eid.is_some() { return Status::Occupied; }
        if tile.blocked() { Status::Blocked } else { Status::Free }
    }

    fn get_tile(&self, p: Point) -> &'static Tile { self.get_cell(p).tile }

    // Setters

    fn add_entity(&mut self, args: &EntityArgs) -> EID {
        let pos = args.pos;
        let eid = self.entities.add(args);
        let cell = self.map.entry_mut(pos).unwrap();
        let prev = replace(&mut cell.eid, Some(eid));
        assert!(prev.is_none());
        self.entity_order.push(eid);
        self.update_known(eid);
        eid
    }

    fn advance_entity(&mut self) {
        self.active_entity_index += 1;
        if self.active_entity_index == self.entity_order.len() {
            self.active_entity_index = 0;
        }
    }

    fn move_entity(&mut self, eid: EID, target: Point) {
        let entity = &mut self.entities[eid];
        let source = replace(&mut entity.pos, target);
        let old = replace(&mut self.map.entry_mut(source).unwrap().eid, None);
        assert!(old == Some(eid));
        let new = replace(&mut self.map.entry_mut(target).unwrap().eid, old);
        assert!(new.is_none());
    }

    fn reset(&mut self, tile: &'static Tile) {
        self.map.fill(Cell { eid: None, shadow: 0, tile });
        self.update_edge_shadows();
        self.entity_order.clear();
        self.active_entity_index = 0;
    }

    fn set_tile(&mut self, point: Point, tile: &'static Tile) {
        let cell = or_return!(self.map.entry_mut(point));
        let old_shadow = if cell.tile.blocked() { 1 } else { 0 };
        let new_shadow = if tile.blocked() { 1 } else { 0 };
        cell.tile = tile;
        self.update_shadow(point, new_shadow - old_shadow);
    }

    fn update_known(&mut self, eid: EID) {
        let mut known = self.known.take().unwrap_or_default();
        swap(&mut known, &mut self.entities[eid].known);

        let me = &self.entities[eid];
        let (player, pos) = (me.player, me.pos);
        let vision = if player { &mut self._pc_vision } else { &mut self.npc_vision };
        vision.compute(pos, |x| self.map.get(x).tile);
        let vision = if player { &self._pc_vision } else { &self.npc_vision };
        known.update(me, &self, vision);

        swap(&mut known, &mut self.entities[eid].known);
        self.known = Some(known);
    }

    fn update_env(&mut self, frame: usize, pos: Point, rng: &mut RNG) {
        let rain = or_return!(&mut self.rain);

        while let Some(x) = rain.drops.front() && x.frame < frame {
            rain.drops.pop_front();
        }
        let total = rain.drops.capacity();
        let denom = max(rain.diff.1, 1);
        let delta = max(rain.diff.1, 1) as usize;
        let extra = (frame + 1) * total / delta - (frame * total) / delta;
        for _ in 0..min(extra, total - rain.drops.len()) {
            let x = rng.gen::<i32>().rem_euclid(denom);
            let y = rng.gen::<i32>().rem_euclid(denom);
            let target_frame = frame + rain.path.len() - 1;
            let target_point = Point(x - denom / 2, y - denom / 2) + pos;
            rain.drops.push_back(Drop { frame: target_frame, point: target_point });
        }

        assert!(rain.lightning >= -1);
        if rain.lightning == -1 {
            if rng.gen::<f32>() < 0.002 { rain.lightning = 10; }
        } else if rain.lightning > 0 {
            rain.lightning -= 1;
        }

        assert!(rain.thunder >= 0);
        if rain.thunder == 0 {
            if rain.lightning == 0 && rng.gen::<f32>() < 0.02 { rain.thunder = 16; }
        } else {
            rain.thunder -= 1;
            if rain.thunder == 0 { rain.lightning = -1; }
        }
    }

    fn update_shadow(&mut self, point: Point, delta: i32) {
        if delta == 0 { return; }
        for shift in &self.shadow {
            let cell = or_continue!(self.map.entry_mut(point + *shift));
            cell.shadow += delta;
            assert!(cell.shadow >= 0);
        }
    }

    fn update_edge_shadows(&mut self) {
        let delta = if self.map.default.tile.blocked() { 1 } else { 0 };
        if delta == 0 || self.shadow.is_empty() { return; }

        for x in -1..(self.map.size.0 + 1) {
            self.update_shadow(Point(x, -1), delta);
            self.update_shadow(Point(x, self.map.size.1), delta);
        }
        for y in 0..self.map.size.1 {
            self.update_shadow(Point(-1, y), delta);
            self.update_shadow(Point(self.map.size.0, y), delta);
        }
    }
}

//////////////////////////////////////////////////////////////////////////////

// Map generation

fn mapgen(board: &mut Board, rng: &mut RNG) {
    let ft = Tile::get('.');
    let wt = Tile::get('#');
    let gt = Tile::get('"');
    let fl = Tile::get('%');
    let wa = Tile::get('~');

    board.reset(ft);
    let size = board.get_size();

    let automata = |rng: &mut RNG| -> Matrix<bool> {
        let mut d100 = || rng.gen::<u32>() % 100;
        let mut result = Matrix::new(size, false);
        for x in 0..size.0 {
            result.set(Point(x, 0), true);
            result.set(Point(x, size.1 - 1), true);
        }
        for y in 0..size.1 {
            result.set(Point(0, y), true);
            result.set(Point(size.0 - 1, y), true);
        }

        for y in 0..size.1 {
            for x in 0..size.0 {
                if d100() < 45 { result.set(Point(x, y),  true); }
            }
        }

        for i in 0..3 {
            let mut next = result.clone();
            for y in 1..size.1 - 1 {
                for x in 1..size.0 - 1 {
                    let point = Point(x, y);
                    let (mut adj1, mut adj2) = (0, 0);
                    for dy in -2_i32..=2 {
                        for dx in -2_i32..=2 {
                            if dx == 0 && dy == 0 { continue; }
                            if min(dx.abs(), dy.abs()) == 2 { continue; }
                            let next = point + Point(dx, dy);
                            if !result.get(next) { continue; }
                            let distance = max(dx.abs(), dy.abs());
                            if distance <= 1 { adj1 += 1; }
                            if distance <= 2 { adj2 += 1; }
                        }
                    }
                    let blocked = adj1 >= 5 || (i < 2 && adj2 <= 1);
                    next.set(point, blocked);
                }
            }
            std::mem::swap(&mut result, &mut next);
        }
        result
    };

    let walls = automata(rng);
    let grass = automata(rng);
    for y in 0..size.1 {
        for x in 0..size.0 {
            let point = Point(x, y);
            if walls.get(point) {
                board.set_tile(point, wt);
            } else if grass.get(point) {
                board.set_tile(point, gt);
            }
        }
    }

    let die = |n: i32, rng: &mut RNG| rng.gen::<i32>().rem_euclid(n);
    let mut river = vec![Point::default()];
    for i in 1..size.1 {
        let last = river.iter().last().unwrap().0;
        let next = last + die(3, rng) - 1;
        river.push(Point(next, i));
    }
    let target = river[0] + *river.iter().last().unwrap();
    let offset = Point((size - target).0 / 2, 0);
    for x in &river { board.set_tile(*x + offset, wa); }

    let pos = |board: &Board, rng: &mut RNG| {
        for _ in 0..100 {
            let p = Point(die(size.0, rng), die(size.1, rng));
            if let Status::Free = board.get_status(p) { return Some(p); }
        }
        None
    };
    for _ in 0..5 {
        if let Some(p) = pos(board, rng) { board.set_tile(p, fl); }
    }
}

//////////////////////////////////////////////////////////////////////////////

// Action

enum Action {
    Move(Point),
    WaitForInput,
}

struct ActionResult {
    success: bool,
}

impl ActionResult {
    fn failure() -> Self { Self { success: false } }
    fn success() -> Self { Self { success: true } }
}

fn plan(board: &Board, eid: EID, input: &mut Action, _: &mut RNG) -> Action {
    let entity = &board.entities[eid];
    if entity.player { return replace(input, Action::WaitForInput); }
    Action::WaitForInput
}

fn act(state: &mut State, eid: EID, action: Action) -> ActionResult {
    match action {
        Action::Move(dir) => {
            if dir == Point::default() { return ActionResult::success(); }
            let entity = &state.board.entities[eid];
            let target = entity.pos + dir;
            match state.board.get_status(target) {
                Status::Blocked => ActionResult::failure(),
                Status::Occupied => ActionResult::failure(),
                Status::Free => {
                    state.board.move_entity(eid, target);
                    ActionResult::success()
                }
            }
        }
        Action::WaitForInput => ActionResult::failure(),
    }
}

//////////////////////////////////////////////////////////////////////////////

// Update

fn get_direction(ch: char) -> Option<Point> {
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

fn process_input(state: &mut State, input: Input) {
    let dir = if let Input::Char(x) = input { get_direction(x) } else { None };
    if let Some(x) = dir { state.input = Action::Move(x); }
}

fn update_state(state: &mut State) {
    state.frame += 1;
    let pos = state.get_player().pos;
    state.board.update_env(state.frame, pos, &mut state.rng);

    let needs_input = |state: &State| {
        if !matches!(state.input, Action::WaitForInput) { return false; }
        state.board.get_active_entity() == state.player
    };

    while !state.inputs.is_empty() && needs_input(state) {
        let input = state.inputs.remove(0);
        process_input(state, input);
    }

    let mut update = false;

    loop {
        let eid = state.board.get_active_entity();
        let player = eid == state.player;
        if player && needs_input(state) { break; }

        state.board.update_known(eid);

        update = true;
        let action = plan(&state.board, eid, &mut state.input, &mut state.rng);
        let result = act(state, eid, action);
        if player && !result.success { break; }

        state.board.advance_entity();
    }

    if update {
        state.board.update_known(state.player);
    }
}

//////////////////////////////////////////////////////////////////////////////

// State

pub struct State {
    board: Board,
    frame: usize,
    input: Action,
    inputs: Vec<Input>,
    player: EID,
    rng: RNG,
}

impl State {
    pub fn new(seed: Option<u64>) -> Self {
        let size = Point(WORLD_SIZE, WORLD_SIZE);
        let pos = Point(size.0 / 2, size.1 / 2);
        let rng = seed.map(|x| RNG::seed_from_u64(x));
        let mut rng = rng.unwrap_or_else(|| RNG::from_entropy());
        let mut board = Board::new(size, LIGHT, WEATHER);

        loop {
            mapgen(&mut board, &mut rng);
            if !board.get_tile(pos).blocked() { break; }
        }
        let input = Action::WaitForInput;
        let glyph = Glyph::wdfg('@', 0x222);
        let player = board.add_entity(&EntityArgs { glyph, player: true, pos });

        let die = |n: i32, rng: &mut RNG| rng.gen::<i32>().rem_euclid(n);
        let pos = |board: &Board, rng: &mut RNG| {
            for _ in 0..100 {
                let p = Point(die(size.0, rng), die(size.1, rng));
                if let Status::Free = board.get_status(p) { return Some(p); }
            }
            None
        };
        for _ in 0..20 {
            if let Some(_) = pos(&board, &mut rng) {}
        }

        Self { board, frame: 0, input, inputs: vec![], player, rng }
    }

    fn get_player(&self) -> &Entity { &self.board.entities[self.player] }

    pub fn add_input(&mut self, input: Input) { self.inputs.push(input) }

    pub fn update(&mut self) { update_state(self); }

    pub fn render(&self, buffer: &mut Buffer) {
        if buffer.data.is_empty() {
            let size = Point(2 * UI_MAP_SIZE_X, UI_MAP_SIZE_Y);
            let mut overwrite = Matrix::new(size, ' '.into());
            std::mem::swap(buffer, &mut overwrite);
        }

        let entity = self.get_player();
        let offset = entity.pos - Point(UI_MAP_SIZE_X / 2, UI_MAP_SIZE_Y / 2);
        let (known, pos) = (&*entity.known, entity.pos);

        let (sun, dark) = match self.board.light {
            Light::None => (Point::default(), true),
            Light::Sun(x) => (x, false),
        };

        let lookup = |point: Point| -> Glyph {
            let unseen = Glyph::wide(' ');
            let cell = known.get(point);
            if !cell.visible() { return unseen; }

            let entity = cell.entity();
            let tile = cell.tile().unwrap();
            // TODO: We shouldn't access the board directly here.
            let shade = self.board.get_cell(point).shadow > 0;
            let delta = point - pos;

            if (dark || (shade && !(tile.blocked() && sun.dot(delta) < 0))) &&
               delta.len_l1() > 1 {
                return unseen;
            }

            let result = entity.map(|x| x.glyph).unwrap_or(tile.glyph);
            if dark || shade { result.with_fg(Color::gray()) } else { result }
        };

        buffer.fill(Glyph::wide(' '));
        for y in 0..UI_MAP_SIZE_Y {
            for x in 0..UI_MAP_SIZE_X {
                let glyph = lookup(Point(x, y) + offset);
                buffer.set(Point(2 * x, y), glyph);
            }
        }

        if let Some(rain) = &self.board.rain {
            for drop in &rain.drops {
                let index = drop.frame - self.frame;
                let point = drop.point - *or_continue!(rain.path.get(index));
                if index == 0 && !known.get(point).visible() { continue; }

                let Point(x, y) = point - offset;
                let ch = if index == 0 { 'o' } else { rain.ch };
                let glyph = Glyph::wdfg(ch, 0x013);
                buffer.set(Point(2 * x, y), glyph);
            }

            if rain.lightning > 0 {
                let color = Color::from(0x111 * (rain.lightning / 2));
                for y in 0..UI_MAP_SIZE_Y {
                    for x in 0..UI_MAP_SIZE_X {
                        let point = Point(2 * x, y);
                        buffer.set(point, buffer.get(point).with_bg(color));
                    }
                }
            }

            if rain.thunder > 0 {
                let shift = (rain.thunder - 1) % 4;
                if shift % 2 == 0 {
                    let space = Glyph::char(' ');
                    let delta = Point(shift - 1, 0);
                    let limit = if delta.1 > 0 { -1 } else { 0 };
                    for y in 0..UI_MAP_SIZE_Y {
                        for x in 0..(UI_MAP_SIZE_X + limit) {
                            let point = Point(2 * x, y);
                            buffer.set(point + delta, buffer.get(point));
                        }
                        buffer.set(Point(0, y), space);
                        buffer.set(Point(2 * UI_MAP_SIZE_X - 1, y), space);
                    }
                }
            }
        }
    }
}
