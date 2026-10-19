#![cfg_attr(target_family = "wasm", allow(unused))]

use std::cmp::max;
use std::io::{Result, Write};

#[cfg(not(target_family = "wasm"))]
use super::log::{Logger, LogFile};

use crate::base::glyph::{Color, Glyph};
use crate::base::point::{LOS, Matrix, Point};
use crate::base::pathing::Neighborhood;

use super::actor::{Actor, EID};
use super::effect::{Frame, ParticleData, RenderData};
use super::game::{WORLD_SIZE, Action, Board, Cell, show_item};
use super::knowledge::ActorKnowledge;
use super::ui::UI;

//////////////////////////////////////////////////////////////////////////////

fn show(eid: EID) -> u64 {
    unsafe { std::mem::transmute::<EID, u64>(eid) }
}

//////////////////////////////////////////////////////////////////////////////

// DebugLog

pub struct DebugLine {
    pub color: Color,
    pub depth: i32,
    pub text: String,
}

pub struct DebugLog {
    pub depth: usize,
    pub lines: Vec<DebugLine>,
    pub verbose: bool,
}

impl DebugLog {
    pub fn append(&mut self, t: impl std::fmt::Display) {
        let color = Color::white();
        let depth = self.depth as i32;
        self.lines.push(DebugLine { color, depth, text: format!("{}", t) });
    }

    pub fn indent(&mut self, n: usize, f: impl Fn(&mut DebugLog)) {
        self.depth += n;
        f(self);
        self.depth -= n;
    }

    pub fn newline(&mut self) { self.append(""); }
}

//////////////////////////////////////////////////////////////////////////////

// DebugFile wasm stub

#[cfg(target_family = "wasm")]
#[derive(Default)]
pub struct DebugFile {}

#[cfg(target_family = "wasm")]
impl DebugFile {
    pub fn record(&mut self, _: &Action, _: &Board, _: &Actor) {}
    pub fn record_frame(&mut self, _: &Board, _: &Frame) {}
    pub fn record_neighborhood(&mut self, _: &Neighborhood) {}
    pub fn record_utility(&mut self, _: &[(i32, Point)]) {}
}

//////////////////////////////////////////////////////////////////////////////

// DebugFile

#[derive(Default)]
struct DebugDetail {
    animation: Vec<Vec<(Point, Glyph)>>,
    utilities: Vec<(i32, Point)>,
    neighborhood: Vec<Point>,
}

#[cfg(not(target_family = "wasm"))]
pub struct DebugFile {
    detail: DebugDetail,
    map: Matrix<Glyph>,
    tick: usize,
    turn: usize,

    file: LogFile,
    logger: Logger,
}

#[cfg(not(target_family = "wasm"))]
impl Default for DebugFile {
    fn default() -> Self {
        let dir = "wasm/debug".into();
        std::fs::remove_dir_all(&dir).unwrap();
        std::fs::create_dir_all(&dir).unwrap();

        let logger = Logger::new(dir);
        let file = logger.open_text("ticks.txt");

        let detail = DebugDetail::default();
        let map = Matrix::new(Point(WORLD_SIZE, WORLD_SIZE), Glyph::wide(' '));
        Self { detail, map, tick: 0, turn: 0, file, logger }
    }
}

#[cfg(not(target_family = "wasm"))]
impl DebugFile {
    pub fn record(&mut self, action: &Action, board: &Board, me: &Actor) {
        self.try_record(action, board, me).unwrap();
    }

    pub fn record_frame(&mut self, board: &Board, frame: &Frame) {
        self.try_record_frame(board, frame).unwrap();
    }

    pub fn record_neighborhood(&mut self, neighborhood: &Neighborhood) {
        self.detail.neighborhood = neighborhood.visited.iter().map(|x| x.0).collect();
    }

    pub fn record_utility(&mut self, utilities: &[(i32, Point)]) {
        self.detail.utilities = utilities.into()
    }

    fn try_record(&mut self, action: &Action, board: &Board, me: &Actor) -> Result<()> {
        self.record_tick(action, board, me)?;

        self.detail = Default::default();
        self.turn += if me.player { 1 } else { 0 };
        self.tick += 1;

        write!(self.file, "{{")?;
        write!(self.file, r#""index":{},"#, self.tick - 1)?;
        write!(self.file, r#""type":"tick","#)?;
        write!(self.file, r#""time":"{}","#, board.time.nsec())?;
        write!(self.file, r#""eid":"{}""#, show(me.eid))?;
        write!(self.file, "}}\n")?;
        self.file.flush()
    }

    fn try_record_frame(&mut self, board: &Board, frame: &Frame) -> Result<()> {
        let render_move = |source: Point, target: Point| {
            let cell = board.get_cell(source);
            let Some(eid) = cell.eid else { return vec![] };

            let s = Self::underlying_glyph(cell);
            let t = Self::actor_glyph(&board.actors[eid]);
            vec![(source, s), (target, t)]
        };
        let render_particle = |p: Point, r: &RenderData| match r {
            RenderData::Dummy => None,
            RenderData::Text(..) => None,
            &RenderData::Glyph(g) => Some((p, g)),
            &RenderData::Flash(c) => {
                let cell = board.get_cell(p);
                let glyph = if let Some(eid) = cell.eid {
                    Self::actor_glyph(&board.actors[eid])
                } else {
                    Self::underlying_glyph(cell)
                };
                Some((p, glyph.with_fg(Color::black()).with_bg(c)))
            },
        };
        let mut xs = Vec::with_capacity(frame.len());
        frame.iter().for_each(|x| match &x.data {
            ParticleData::Light(_) => {},
            ParticleData::Shift(s) => xs.append(&mut render_move(*s, x.point)),
            ParticleData::Sight(r) => { render_particle(x.point, r).map(|x| xs.push(x)); }
            ParticleData::Sound(_, r) => { render_particle(x.point, r).map(|x| xs.push(x)); }
        });
        self.detail.animation.push(xs);

        write!(self.file, "{{")?;
        write!(self.file, r#""index":{},"#, self.tick)?;
        write!(self.file, r#""type":"animation","#)?;
        write!(self.file, r#""time":"{}","#, board.time.nsec())?;
        write!(self.file, r#""frame":{}"#, self.detail.animation.len() - 1)?;
        write!(self.file, "}}\n")?;
        self.file.flush()
    }

    fn record_tick(&mut self, action: &Action, board: &Board, me: &Actor) -> Result<()> {
        // Dump binary data for all animations happening prior to this tick.
        if !self.detail.animation.is_empty() {
            let filename = format!("animation-{}.bin.gz", self.tick);
            let mut file = self.logger.open_compressed(&filename);

            Self::write_bin(&mut file, &(self.detail.animation.len() as i32))?;
            for frame in &self.detail.animation {
                Self::write_bin(&mut file, &(frame.len() as i32))?;
                Self::write_array(&mut file, frame)?;
            }
            file.flush()?;
        }

        // Dump binary data for the tick itself.
        let filename = format!("tick-{}.bin.gz", self.tick);
        let mut file = self.logger.open_compressed(&filename);

        // Dump binary data revealing all of the actors.
        let actors: Vec<_> = board.actors.iter().collect();
        Self::write_bin(&mut file, &(actors.len() as i32))?;

        for &(eid, actor) in &actors {
            let Actor { cur_hp, max_hp, .. } = *actor;
            let hp_fraction = cur_hp as f64 / max(max_hp, 1) as f64;
            let glyph = Self::actor_glyph(actor);

            Self::write_str(&mut file, &format!("{}", show(eid)))?;
            Self::write_str(&mut file, actor.species.name)?;
            Self::write_bin(&mut file, &hp_fraction)?;
            Self::write_bin(&mut file, &actor.pos)?;
            Self::write_bin(&mut file, &glyph)?;
        };

        // Dump info about our view of other actors.
        let mut sightings = vec![];
        for other in &me.known.scents {
            let color = 0xff8000;
            let glyph = other.species.glyph.with_fg(Color::black()).with_bg(color);
            sightings.push((other.pos, glyph));
        }
        for other in &me.known.noises {
            let color = {
                let current = me.known.get(other.pos).noise();
                let moved = current.map(|x| x.uid) != Some(other.uid);
                if moved { 0xff0000 } else { 0xffff00 }
            };
            let glyph = Glyph::wdfg('?', Color::black()).with_bg(color);
            sightings.push((other.pos, glyph));
        }
        for other in &me.known.actors {
            let color = if other.visible() { 0x00ff00 } else {
                let current = me.known.get(other.pos).actor();
                let moved = current.map(|x| x.eid) != Some(other.eid);
                if moved { 0xff0000 } else { 0xffff00 }
            };
            let glyph = Self::knowledge_glyph(other).with_fg(Color::black()).with_bg(color);
            sightings.push((other.pos, glyph));
        }
        Self::write_bin(&mut file, &(sightings.len() as i32))?;
        Self::write_array(&mut file, sightings.as_slice())?;

        // Dump text debug output from behavior trees.
        let color = Color::white();
        let lines = me.ai.get_trace(&me.known);
        let extra = [
            DebugLine { color, depth: 0, text: format!("Turn {}: {:?}", self.turn, action) },
            DebugLine { color, depth: 0, text: "".into() },
        ];
        Self::write_bin(&mut file, &((extra.len() + lines.len()) as i32))?;

        for line in extra.iter().chain(&lines) {
            Self::write_bin(&mut file, &line.color)?;
            Self::write_bin(&mut file, &line.depth)?;
            Self::write_str(&mut file, &line.text)?;
        }

        // Dump the neighborhood of cells that we searched this turn.
        Self::write_bin(&mut file, &(self.detail.neighborhood.len() as i32))?;
        Self::write_array(&mut file, self.detail.neighborhood.as_slice())?;

        // Dump a utility map, if this behavior tree tick used one.
        Self::write_bin(&mut file, &(self.detail.utilities.len() as i32))?;
        Self::write_array(&mut file, self.detail.utilities.as_slice())?;

        // Render action plans by highlighting each of the plan's steps.
        let color = |map: &mut Matrix<Glyph>, p: Point, c: i32| {
            if p == me.pos { return; }
            let mut glyph = map.get(p);
            if glyph.ch().is_wide_space() { glyph = Glyph::wide('.'); }
            map.set(p, glyph.with_fg(c));
        };
        let highlight = |map: &mut Matrix<Glyph>, p: Point, c: i32| {
            if p == me.pos { return; }
            let mut glyph = map.get(p);
            if glyph.ch().is_wide_space() { glyph = Glyph::wide('.'); }
            map.set(p, glyph.with_fg(Color::black()).with_bg(c));
        };

        // Anything we scribble on the map shows up in the debug UI.
        for (p, x) in self.map.iter_mut() {
            *x = UI::render_tile(me, p, None);
        }
        if let Some(target) = me.ai.get_target() {
            let source = me.ai.get_path().last().cloned().unwrap_or(me.pos);
            for p in LOS(source, target).into_iter().skip(1) {
                color(&mut self.map, p, 0x0000ff);
            }
            highlight(&mut self.map, target, 0x0000ff);
        }
        for &p in me.ai.get_path() {
            color(&mut self.map, p, 0xff0000);
        }
        if let Some(&p) = me.ai.get_path().last() {
            highlight(&mut self.map, p, 0xff0000);
        }
        Self::write_bin(&mut file, &self.map.size())?;
        Self::write_array(&mut file, self.map.raw_data())?;
        file.flush()?;

        Result::Ok(())
    }

    fn write_array<W: Write, T>(f: &mut W, t: &[T]) -> Result<()> {
        let ptr = t.as_ptr() as *const u8;
        let len = std::mem::size_of_val(t);
        f.write_all(unsafe { std::slice::from_raw_parts(ptr, len) })
    }

    fn write_bin<W: Write, T>(f: &mut W, t: &T) -> Result<()> {
        Self::write_array(f, std::slice::from_ref(t))
    }

    fn write_str<W: Write>(f: &mut W, s: &str) -> Result<()> {
        f.write_all(s.as_bytes())?;
        f.write_all(b"\0")
    }

    fn actor_glyph(x: &Actor) -> Glyph {
        let sneaking = x.species.human() && x.sneaking;
        if sneaking { Glyph::wide('e') } else { x.species.glyph }
    }

    fn knowledge_glyph(x: &ActorKnowledge) -> Glyph {
        let sneaking = x.species.human() && x.sneaking();
        if sneaking { Glyph::wide('e') } else { x.species.glyph }
    }

    fn underlying_glyph(cell: &Cell) -> Glyph {
        if let Some(x) = cell.items.last() { show_item(x) } else { cell.tile.glyph }
    }
}

//////////////////////////////////////////////////////////////////////////////
