use crate::base::{HashSet, Glyph, LOS, Point, RNG, dirs, sample};
use crate::game::{Board, UpdateEnv};

//////////////////////////////////////////////////////////////////////////////

// Types

#[derive(Copy, Clone, Eq, PartialEq)]
pub enum FT { Fire, Ice, Hit, Summon, Withdraw }

pub type CB = Box<dyn Fn(&mut Board, &mut UpdateEnv)>;

pub enum Event {
    Callback { frame: i32, callback: CB },
    Other { frame: i32, point: Point, what: FT },
}

#[derive(Copy, Clone)]
pub struct Particle {
    pub point: Point,
    pub glyph: Glyph,
}

impl Particle {
    pub fn new(point: Point, glyph: Glyph) -> Self {
        Self { point, glyph }
    }
}

pub type Frame = Vec<Particle>;

#[derive(Default)]
pub struct Effect {
    pub events: Vec<Event>,
    pub frames: Vec<Frame>,
}

impl Effect {
    pub fn new(frames: Vec<Frame>) -> Self {
        Self { events: Vec::new(), frames }
    }

    pub fn and(self, other: Effect) -> Effect {
        Effect::parallel(vec![self, other])
    }

    pub fn then(self, other: Effect) -> Effect {
        Effect::serial(vec![self, other])
    }

    pub fn delay(self, n: i32) -> Effect {
        Effect::pause(n).then(self)
    }

    pub fn scale(self, s: f64) -> Effect {
        let mut result = Effect::default();
        let scale = |f: i32| (s * f as f64).round() as i32;

        for mut event in self.events.into_iter() {
            event.update_frame(scale);
            result.events.push(event);
        }
        for i in 0..(self.frames.len() as i32) {
            let (start, limit) = (scale(i), scale(i + 1));
            let frame: &Frame = &self.frames[i as usize];
            for _ in start..limit {
                result.frames.push(frame.clone());
            }
        }
        result
    }

    // Mutate the Effect in-place

    pub fn add_event(&mut self, event: Event) {
        self.events.push(event);
        self.events.sort_by_key(|e| e.frame());
    }

    pub fn add_particle(&mut self, frame: i32, particle: Particle) {
        while (self.frames.len() as i32) <= frame {
            self.frames.push(Vec::new());
        }
        self.frames[frame as usize].push(particle);
    }

    pub fn sub_on_finished(&mut self, callback: CB) {
        let frame = self.frames.len() as i32;
        self.add_event(Event::Callback { frame, callback });
    }

    // Implementations for Effect::Constant, Effect::Pause, etc.

    pub fn constant(particle: Particle, n: i32) -> Effect {
        if n <= 0 { return Effect::default(); }
        Effect::new(vec![vec![particle]; n as usize])
    }

    pub fn pause(n: i32) -> Effect {
        if n <= 0 { return Effect::default(); }
        Effect::new(vec![vec![]; n as usize])
    }

    pub fn parallel(effects: Vec<Effect>) -> Effect {
        let mut result = Effect::default();
        for effect in effects.into_iter() {
            result.events.extend(effect.events);
            for (i, frame) in effect.frames.iter().enumerate() {
                if i >= result.frames.len() {
                    result.frames.push(Vec::new());
                }
                result.frames[i].extend(frame.clone());
            }
        }
        result.events.sort_by_key(|e| e.frame());
        result
    }

    pub fn serial(effects: Vec<Effect>) -> Effect {
        let mut offset = 0;
        let mut result = Effect::default();
        for effect in effects.into_iter() {
            for mut event in effect.events.into_iter() {
                event.update_frame(|x| x + offset);
                result.events.push(event);
            }
            result.frames.extend(effect.frames.clone());
            offset += effect.frames.len() as i32;
        }
        result
    }
}

impl Event {
    pub fn frame(&self) -> i32 {
        match self {
            Event::Callback { frame, .. } => *frame,
            Event::Other { frame, .. } => *frame,
        }
    }

    pub fn what(&self) -> Option<FT> {
        match self {
            Event::Callback { .. } => None,
            Event::Other { what, .. } => Some(*what),
        }
    }

    pub fn update_frame<F: FnOnce(i32) -> i32>(&mut self, update: F) {
        match self {
            Event::Callback { frame, .. } => *frame = update(*frame),
            Event::Other { frame, .. } => *frame = update(*frame),
        }
    }
}

//////////////////////////////////////////////////////////////////////////////

type Sparkle<'a> = (i32, &'a str, i32);

fn add_sparkle(effect: &mut Effect, sparkle: &[Sparkle], mut frame: i32, point: Point) -> i32 {
    for &(delay, chars, color) in sparkle {
        for _ in 0..delay {
            let index = rand::random_range(0..chars.chars().count());
            let glyph = Glyph::wdfg(chars.chars().nth(index).unwrap(), color);
            effect.add_particle(frame, Particle::new(point, glyph));
            frame += 1;
        }
    }
    frame
}

fn get_glyph_at(board: &Board, p: Point) -> Glyph {
    let cell = board.get_cell(p);
    let entity = cell.eid.and_then(|x| board.get_entity(x));
    entity.map(|x| x.species.glyph).unwrap_or(cell.tile.glyph)
}

fn get_underlying_glyph_at(board: &Board, p: Point) -> Glyph {
    board.get_tile(p).glyph
}

fn random_delay(n: i32) -> i32 {
    let mut count = 1;
    let limit = (1.5 * n as f64).floor() as i32;
    while count < limit && (rand::random::<i32>() % n) != 0 { count += 1; }
    count
}

pub fn ray_character(delta: Point) -> char {
    let Point(x, y) = delta;
    let (ax, ay) = (x.abs(), y.abs());
    if ax > 2 * ay { return '-'; }
    if ay > 2 * ax { return '|'; }
    if (x > 0) == (y > 0) { '\\' } else { '/' }
}

#[allow(non_snake_case)]
fn OverlayEffect(effect: Effect, particle: Particle) -> Effect {
    let constant = Effect::constant(particle, effect.frames.len() as i32);
    Effect::parallel(vec![effect, constant])
}

#[allow(non_snake_case)]
fn UnderlayEffect(effect: Effect, particle: Particle) -> Effect {
    let constant = Effect::constant(particle, effect.frames.len() as i32);
    Effect::parallel(vec![constant, effect])
}

#[allow(non_snake_case)]
fn ExplosionEffect(point: Point) -> Effect {
    let glyph = |ch: char| Glyph::wdfg(ch, 0xff0000);
    let base = vec![
        vec![Particle::new(point, glyph('*'))],
        vec![
            Particle::new(point, glyph('+')),
            Particle::new(point + dirs::N, glyph('|')),
            Particle::new(point + dirs::S, glyph('|')),
            Particle::new(point + dirs::E, glyph('-')),
            Particle::new(point + dirs::W, glyph('-')),
        ],
        vec![
            Particle::new(point + dirs::N,  glyph('-')),
            Particle::new(point + dirs::S,  glyph('-')),
            Particle::new(point + dirs::E,  glyph('|')),
            Particle::new(point + dirs::W,  glyph('|')),
            Particle::new(point + dirs::NE, glyph('\\')),
            Particle::new(point + dirs::SW, glyph('\\')),
            Particle::new(point + dirs::NW, glyph('/')),
            Particle::new(point + dirs::SE, glyph('/')),
        ],
    ];
    Effect::new(base).scale(4.0)
}

#[allow(non_snake_case)]
fn ImplosionEffect(point: Point) -> Effect {
    let glyph = |ch: char| Glyph::wdfg(ch, 0xff0000);
    let base = vec![
        vec![
            Particle::new(point, glyph('*')),
            Particle::new(point + dirs::NE, glyph('/')),
            Particle::new(point + dirs::SW, glyph('/')),
            Particle::new(point + dirs::NW, glyph('\\')),
            Particle::new(point + dirs::SE, glyph('\\')),
        ],
        vec![
            Particle::new(point, glyph('#')),
            Particle::new(point + dirs::NE, glyph('/')),
            Particle::new(point + dirs::SW, glyph('/')),
            Particle::new(point + dirs::NW, glyph('\\')),
            Particle::new(point + dirs::SE, glyph('\\')),
        ],
        vec![Particle::new(point, glyph('*'))],
        vec![Particle::new(point, glyph('#'))],
    ];
    Effect::new(base).scale(3.0)
}

#[allow(non_snake_case)]
fn RayEffect(source: Point, target: Point, speed: i32) -> Effect {
    let line = LOS(source, target);
    if line.len() <= 2 { return Effect::default(); }

    let mut result = Vec::new();
    let glyph = Glyph::wdfg(ray_character(target - source), 0xff0000);
    let denom = ((line.len() - 2 + speed as usize) % speed as usize) as i32;
    let start = if denom == 0 { speed } else { denom } as usize;
    for i in (start..line.len() - 1).step_by(speed as usize) {
        result.push((0..i).map(|j| Particle::new(line[j + 1], glyph)).collect());
    }
    Effect::new(result)
}

#[allow(non_snake_case)]
pub fn SummonEffect(source: Point, target: Point) -> Effect {
    let mut effect = Effect::default();
    let line = LOS(source, target);
    let ball = Glyph::wdfg('*', 0xff0000);
    for i in 1..line.len() - 1 {
        effect.frames.push(vec![Particle::new(line[i], ball)]);
    }
    let frame = effect.frames.len() as i32 + 8;
    effect.add_event(Event::Other { frame, point: target, what: FT::Summon });
    effect.then(ExplosionEffect(target))
}

#[allow(non_snake_case)]
pub fn WithdrawEffect(source: Point, target: Point) -> Effect {
    let mut implode  = ImplosionEffect(target);
    let frame = std::cmp::max(implode.frames.len() as i32 - 6, 0);
    implode.add_event(Event::Other { frame, point: target, what: FT::Withdraw });

    let base = RayEffect(source, target, 4);
    let undo = base.frames.clone().into_iter().rev().collect();
    let last = match base.frames.last() {
        Some(x) => x.clone(),
        None => return implode,
    };

    Effect::serial(vec![
        base,
        Effect::new(vec![last]).scale(implode.frames.len() as f64).and(implode),
        Effect::new(undo),
    ])
}

#[allow(non_snake_case)]
fn SwitchEffect(source: Point, target: Point) -> Effect {
    Effect::serial(vec![
        WithdrawEffect(source, target),
        Effect::pause(4),
        SummonEffect(source, target),
    ])
}

#[allow(non_snake_case)]
pub fn EmberEffect(_: &Board, _: &mut RNG, source: Point, target: Point) -> Effect {
    let mut effect = Effect::default();
    let line = LOS(source, target);

    let trail = || [
        (random_delay(0), "*^^",   0xff0000),
        (random_delay(1), "*^",    0xffa800),
        (random_delay(2), "**^",   0xffff00),
        (random_delay(3), "**^#%", 0xffa800),
        (random_delay(4), "#%",    0xff0000),
    ];

    let flame = || [
        (random_delay(0), "*^^",   0xff0000),
        (random_delay(1), "*^",    0xffa800),
        (random_delay(2), "**^#%", 0xffff00),
        (random_delay(3), "*^#%",  0xffa800),
        (random_delay(4), "*^#%",  0xff0000),
    ];

    for i in 1..line.len() - 1 {
        let frame = (i - 1) / 2;
        add_sparkle(&mut effect, &trail(), frame as i32, line[i]);
    }

    let mut hit: i32 = 0;
    for &dir in [dirs::NONE].iter().chain(&dirs::ALL) {
        let norm = dir.len_taxicab();
        let frame = 2 * norm + (line.len() as i32 - 1) / 2;
        let finish = add_sparkle(&mut effect, &flame(), frame, target + dir);
        if norm == 0 { hit = finish; }
    }
    effect.add_event(Event::Other { frame: hit, point: target, what: FT::Hit });
    effect
}

#[allow(non_snake_case)]
pub fn IceBeamEffect(_: &Board, _: &mut RNG, source: Point, target: Point) -> Effect {
    let mut effect = Effect::default();
    let line = LOS(source, target);
    let ray = ray_character(target - source).to_string();
    let ray = ray.as_str();

    let trail = [
        (2, ray, 0xffffff),
        (2, ray, 0x00ffff),
        (2, ray, 0x0000ff),
    ];

    let flame = [
        (2, "*", 0xffffff),
        (2, "*", 0x00ffff),
        (2, "*", 0x0000ff),
        (2, "*", 0xffffff),
        (2, "*", 0x00ffff),
        (2, "*", 0x0000ff),
    ];

    for i in 1..line.len() {
        let frame = ((i - 1) / 2) as i32;
        add_sparkle(&mut effect, &trail, frame, line[i]);
    }

    let frame = ((line.len() - 1) / 2) as i32;
    let hit = add_sparkle(&mut effect, &flame, frame, target);
    effect.add_event(Event::Other { frame: hit, point: target, what: FT::Hit, });
    effect
}

#[allow(non_snake_case)]
pub fn BlizzardEffect(_: &Board, rng: &mut RNG, source: Point, target: Point) -> Effect {
    let mut effect = Effect::default();
    let ray = ray_character(target - source).to_string();
    let ray = ray.as_str();

    let mut points = vec![target];
    let mut used = HashSet::default();
    while points.len() < 3 {
        let alt = target + *sample(&dirs::ALL, rng);
        if !used.insert(alt) { continue; }
        points.push(alt);
    }

    let trail = [
        (1, ray, 0xffffff),
        (1, ray, 0x00ffff),
        (1, ray, 0x0000ff),
    ];

    let flame = [
        (1, "*", 0xffffff),
        (1, "*", 0x00ffff),
        (1, "*", 0x0000ff),
        (1, "*", 0xffffff),
        (1, "*", 0x00ffff),
        (1, "*", 0x0000ff),
    ];

    let mut hit: i32 = 0;
    for (p, next) in points.iter().enumerate() {
        let d = 9 * p as i32;
        let line = LOS(source, *next);
        let size = line.len() as i32;
        for i in 1..size - 1 {
            add_sparkle(&mut effect, &trail, d + i, line[i as usize]);
        }
        let finish = add_sparkle(&mut effect, &flame, d + size - 1, *next);
        if p == 0 { hit = finish; }
    }
    effect.add_event(Event::Other { frame: hit, point: target, what: FT::Hit });
    effect.scale(2.0 / 3.0)
}

#[allow(non_snake_case)]
pub fn HeadbuttEffect(board: &Board, _: &mut RNG, source: Point, target: Point) -> Effect {
    let glyph = get_glyph_at(board, source);
    let underlying = get_underlying_glyph_at(board, source);

    let trail = [
        (2, "#", 0xffffff),
        (2, "#", 0xc0c0c0),
        (2, "#", 0x808080),
        (2, "#", 0x404040),
    ];

    let move_along_line = |line: &[Point], glyph: Glyph| {
        let mut effect = Effect::default();
        for i in 1..line.len() {
            let tick = i as i32 - 1;
            let (prev, next) = (line[i - 1], line[i]);
            effect.add_particle(tick, Particle::new(next, glyph));
            add_sparkle(&mut effect, &trail, tick, prev);
        }
        effect
    };

    // Move to the "neighbor" cell next to the target, but not onto it.
    let mut line = LOS(source, target);
    line.pop();
    let line_to = line.clone();
    line.reverse();
    let line_from = line;

    // Once we reach the neighbor, pause for a second to do the attack.
    let neighbor    = line_from.first().cloned().unwrap_or(source);
    let hold_effect = Effect::constant(Particle::new(neighbor, glyph), 32);
    let move_length = std::cmp::max(line_to.len() as i32 - 1, 0);

    let to   = move_along_line(&line_to, glyph);
    let hold = hold_effect.delay(move_length);
    let from = move_along_line(&line_from, glyph);

    let back_delays = hold.frames.len() as i32 + move_length;
    let back_length = from.frames.len() as i32 - move_length;
    let back_effect = Effect::constant(Particle::new(source, glyph), back_length);
    let back = back_effect.delay(back_delays);

    let hit  = move_length;
    let mut effect = UnderlayEffect(to.and(hold.then(from).and(back)),
                                    Particle::new(source, underlying));
    effect.add_event(Event::Other { frame: hit, point: target, what: FT::Hit });
    effect.scale(0.5)
}
