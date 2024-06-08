use std::cmp::{max, min};
use std::cell::RefCell;
use std::mem::replace;
use std::rc::Rc;

use lazy_static::lazy_static;
use rand::{Rng, SeedableRng};

use crate::static_assert_size;
use crate::base::{Buffer, Glyph};
use crate::base::{HashMap, FOV, Matrix, Point, dirs};
use crate::base::RNG;

//////////////////////////////////////////////////////////////////////////////

// Constants

const WORLD_SIZE: i32 = 100;
const FOV_RADIUS: i32 = 21;
const UI_MAP_SIZE_X: i32 = 2 * FOV_RADIUS + 1;
const UI_MAP_SIZE_Y: i32 = 2 * FOV_RADIUS + 1;
const OBSCURED_VISION: i32 = 3;

#[derive(Eq, PartialEq)]
pub enum Input { Escape, BackTab, Char(char) }

//////////////////////////////////////////////////////////////////////////////

// Tile

const FLAG_NONE: u32 = 0;
const FLAG_BLOCKED: u32 = 1 << 0;
const FLAG_OBSCURE: u32 = 1 << 1;

struct Tile {
    flags: u32,
    glyph: Glyph,
    description: &'static str,
}
static_assert_size!(Tile, 24);

impl Tile {
    fn get(ch: char) -> &'static Tile { TILES.get(&ch).unwrap() }
    fn blocked(&self) -> bool { self.flags & FLAG_BLOCKED != 0 }
    fn obscure(&self) -> bool { self.flags & FLAG_OBSCURE != 0 }
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
            ('.', (FLAG_NONE,    Glyph::wide('.'),        "grass")),
            ('"', (FLAG_OBSCURE, Glyph::wdfg('"', 0x231), "tall grass")),
            ('#', (FLAG_BLOCKED, Glyph::wdfg('#', 0x010), "a tree")),
            ('%', (FLAG_NONE,    Glyph::wdfg('%', 0x400), "flowers")),
            ('~', (FLAG_NONE,    Glyph::wdfg('~', 0x015), "water")),
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

type EntityRef = Rc<RefCell<Entity>>;

struct Entity {
    glyph: Glyph,
    player: bool,
    pos: Point,
}

//////////////////////////////////////////////////////////////////////////////

// Board

#[derive(Clone, Copy, Eq, PartialEq)]
enum Status { Free, Blocked, Occupied }

#[derive(Clone)]
struct Cell {
    entity: Option<EntityRef>,
    tile: &'static Tile,
}
static_assert_size!(Cell, 16);

struct Board {
    active_entity_index: usize,
    entity_order: Vec<EntityRef>,
    map: Matrix<Cell>,
}

impl Board {
    fn new(size: Point) -> Self {
        let map = Matrix::new(size, Cell { entity: None, tile: Tile::get('#') });
        Self { active_entity_index: 0, entity_order: vec![], map }
    }

    // Getters

    fn get_active_entity(&self) -> &EntityRef {
        &self.entity_order[self.active_entity_index]
    }

    fn get_cell(&self, p: Point) -> &Cell { self.map.entry_ref(p) }

    fn get_size(&self) -> Point { self.map.size }

    fn get_status(&self, p: Point) -> Status {
        let Cell { entity, tile } = self.get_cell(p);
        if entity.is_some() { return Status::Occupied; }
        if tile.blocked() { Status::Blocked } else { Status::Free }
    }

    fn get_tile(&self, p: Point) -> &'static Tile { self.get_cell(p).tile }

    // Setters

    fn add_entity(&mut self, entity: &EntityRef) {
        let cell = self.map.entry_mut(entity.borrow().pos).unwrap();
        assert!(cell.entity.is_none());
        cell.entity = Some(entity.clone());
        self.entity_order.push(entity.clone());
    }

    fn advance_entity(&mut self) {
        self.active_entity_index += 1;
        if self.active_entity_index == self.entity_order.len() {
            self.active_entity_index = 0;
        }
    }

    fn move_entity(&mut self, entity: &EntityRef, target: Point) {
        let source = replace(&mut entity.borrow_mut().pos, target);
        let old = replace(&mut self.map.entry_mut(source).unwrap().entity, None);
        assert!(old.as_ref().unwrap().as_ptr() == entity.as_ptr());
        let new = replace(&mut self.map.entry_mut(target).unwrap().entity, old);
        assert!(new.is_none());
    }

    fn reset(&mut self, tile: &'static Tile) {
        self.map.fill(Cell { entity: None, tile });
        self.entity_order.clear();
        self.active_entity_index = 0;
    }

    fn set_tile(&mut self, point: Point, tile: &'static Tile) {
        if let Some(x) = self.map.entry_mut(point) { x.tile = tile; }
    }
}

//////////////////////////////////////////////////////////////////////////////

struct Vision {
    fov: FOV,
    center: Point,
    offset: Point,
    visibility: Matrix<i32>,
    points_seen: Vec<Point>,
}

impl Vision {
    fn new(radius: i32) -> Self {
        let vision_side = 2 * radius + 1;
        let vision_size = Point(vision_side, vision_side);
        Self {
            fov: FOV::new(radius),
            center: Point(radius, radius),
            offset: Point::default(),
            visibility: Matrix::new(vision_size, -1),
            points_seen: vec![],
        }
    }

    fn can_see_now(&self, p: Point) -> bool {
        self.get_visibility_at(p) >= 0
    }

    fn get_visibility_at(&self, p: Point) -> i32 {
        self.visibility.get(p + self.offset)
    }

    fn compute<F: Fn(Point) -> &'static Tile>(&mut self, pos: Point, f: F) {
        self.offset = self.center - pos;
        self.visibility.fill(-1);
        self.points_seen.clear();

        let blocked = |p: Point, prev: Option<&Point>| {
            let lookup = p + self.center;
            let cached = self.visibility.get(lookup);

            let visibility = (|| {
                // These constant values come from Point.distanceNethack.
                // They are chosen such that, in a field of tall grass, we'll
                // only see cells at a distanceNethack <= kVisionRadius.
                if prev.is_none() { return 100 * (OBSCURED_VISION + 1) - 95 - 46 - 25; }

                let tile = f(p + pos);
                if tile.blocked() { return 0; }

                let parent = prev.unwrap();
                let obscure = tile.obscure();
                let diagonal = p.0 != parent.0 && p.1 != parent.1;
                let loss = if obscure { 95 + if diagonal { 46 } else { 0 } } else { 0 };
                let prev = self.visibility.get(*parent + self.center);
                max(prev - loss, 0)
            })();

            if visibility > cached {
                self.visibility.set(lookup, visibility);
                if cached < 0 && 0 <= visibility {
                    self.points_seen.push(p + pos);
                }
            }
            visibility <= 0
        };
        self.fov.apply(blocked);
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

fn plan(_: &Board, entity: &EntityRef, input: &mut Action, _: &mut RNG) -> Action {
    let player = entity.borrow().player;
    if !player { return Action::WaitForInput; }
    replace(input, Action::WaitForInput)
}

fn act(state: &mut State, entity: EntityRef, action: Action) -> ActionResult {
    match action {
        Action::Move(dir) => {
            if dir == Point::default() { return ActionResult::success(); }
            let target = entity.borrow().pos + dir;
            match state.board.get_status(target) {
                Status::Blocked => ActionResult::failure(),
                Status::Occupied => ActionResult::failure(),
                Status::Free => {
                    state.board.move_entity(&entity, target);
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
    let needs_input = |state: &State| {
        if !matches!(state.input, Action::WaitForInput) { return false; }
        state.board.get_active_entity().borrow().player
    };

    while !state.inputs.is_empty() && needs_input(state) {
        let input = state.inputs.remove(0);
        process_input(state, input);
    }

    let mut update = false;

    loop {
        let entity = state.board.get_active_entity();
        let player = entity.borrow().player;
        if player && needs_input(state) { break; }

        update = true;
        let (input, rng) = (&mut state.input, &mut state.rng);
        let action = plan(&state.board, &entity, input, rng);
        let result = act(state, entity.clone(), action);
        if player && !result.success { break; }

        state.board.advance_entity();
    }

    if update {
        let pos = state.player.borrow().pos;
        state.vision.compute(pos, |x| state.board.get_tile(x));
    }
}

//////////////////////////////////////////////////////////////////////////////

// State

pub struct State {
    board: Board,
    input: Action,
    inputs: Vec<Input>,
    player: EntityRef,
    vision: Vision,
    rng: RNG,
}

impl State {
    pub fn new(seed: Option<u64>) -> Self {
        let size = Point(WORLD_SIZE, WORLD_SIZE);
        let pos = Point(size.0 / 2, size.1 / 2);
        let rng = seed.map(|x| RNG::seed_from_u64(x));
        let mut rng = rng.unwrap_or_else(|| RNG::from_entropy());
        let mut board = Board::new(size);

        loop {
            mapgen(&mut board, &mut rng);
            if !board.get_tile(pos).blocked() { break; }
        }
        let glyph = Glyph::wide('@');
        let player = Rc::new((Entity { glyph, player: true, pos}).into());
        board.add_entity(&player);

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

        let input = Action::WaitForInput;
        let inputs = vec![];
        let mut vision = Vision::new(FOV_RADIUS);

        let pos = player.borrow().pos;
        vision.compute(pos, |x| board.get_tile(x));

        Self { board, input, inputs, player, vision, rng }
    }

    pub fn add_input(&mut self, input: Input) { self.inputs.push(input) }

    pub fn update(&mut self) { update_state(self); }

    pub fn render(&self, buffer: &mut Buffer) {
        if buffer.data.is_empty() {
            let size = Point(2 * UI_MAP_SIZE_X, UI_MAP_SIZE_Y);
            let mut overwrite = Matrix::new(size, ' '.into());
            std::mem::swap(buffer, &mut overwrite);
        }

        let pos = self.player.borrow().pos;
        let offset = pos - Point(UI_MAP_SIZE_X / 2, UI_MAP_SIZE_Y / 2);
        let unseen = Glyph::wide(' ');

        for y in 0..UI_MAP_SIZE_Y {
            for x in 0..UI_MAP_SIZE_X {
                let point = Point(x, y) + offset;
                let known = self.vision.can_see_now(point);
                let glyph = if known {
                    let Cell { entity, tile } = self.board.get_cell(point);
                    entity.as_ref().map(|x| x.borrow().glyph).unwrap_or(tile.glyph)
                } else {
                    unseen
                };
                buffer.set(Point(2 * x, y), glyph);
            }
        }
    }
}
