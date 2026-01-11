use std::cmp::max;
use std::fs::File;
use std::io::{BufWriter, Result, Write};

use flate2::Compression;
use flate2::write::GzEncoder;

use crate::base::{Color, Glyph, Matrix, Point};
use crate::effect::Frame;
use crate::entity::{EID, Entity};
use crate::game::{WORLD_SIZE, Action, Board};
use crate::knowledge::{EntityKnowledge, Timestamp};
use crate::ui::UI;

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

    pub fn indent(&mut self, n: usize, f: impl Fn(&mut DebugLog) -> ()) {
        self.depth += n;
        f(self);
        self.depth -= n;
    }

    pub fn newline(&mut self) { self.append(""); }
}

//////////////////////////////////////////////////////////////////////////////

// DebugFile

pub struct DebugFile {
    animation: Vec<Frame>,
    utilities: Vec<(i32, Point)>,
    dir: &'static str,
    file: BufWriter<File>,
    map: Matrix<Glyph>,
    next_tick: usize,
}

impl Default for DebugFile {
    fn default() -> Self {
        let dir = "wasm/debug";
        std::fs::remove_dir_all(dir).unwrap();
        std::fs::create_dir_all(dir).unwrap();
        let file = BufWriter::new(File::create(format!("{}/ticks.txt", dir)).unwrap());
        let map = Matrix::new(Point(WORLD_SIZE, WORLD_SIZE), Glyph::wide(' '));
        Self { animation: vec![], utilities: vec![], dir, file, map, next_tick: 0 }
    }
}

impl DebugFile {
    pub fn record(&mut self, action: &Action, board: &Board, me: &Entity) {
        self.try_record(action, board, me).unwrap();
    }

    pub fn record_frame(&mut self, time: Timestamp, frame: &Frame) {
        self.try_record_frame(time, frame).unwrap();
    }

    pub fn record_utility(&mut self, utilities: &[(i32, Point)]) {
        self.utilities = utilities.into()
    }

    fn try_record(&mut self, action: &Action, board: &Board, me: &Entity) -> Result<()> {
        self.record_tick(action, board, me)?;

        self.animation.clear();
        self.utilities.clear();
        self.next_tick += 1;

        write!(self.file, "{{")?;
        write!(self.file, r#""index":{},"#, self.next_tick - 1)?;
        write!(self.file, r#""type":"tick","#)?;
        write!(self.file, r#""time":"{}","#, board.time.nsec())?;
        write!(self.file, r#""eid":"{}""#, show(me.eid))?;
        write!(self.file, "}}\n")?;
        self.file.flush()
    }

    fn try_record_frame(&mut self, time: Timestamp, frame: &Frame) -> Result<()> {
        self.animation.push(frame.clone());

        write!(self.file, "{{")?;
        write!(self.file, r#""index":{},"#, self.next_tick)?;
        write!(self.file, r#""type":"animation","#)?;
        write!(self.file, r#""time":"{}","#, time.nsec())?;
        write!(self.file, r#""frame":{}"#, self.animation.len() - 1)?;
        write!(self.file, "}}\n")?;
        self.file.flush()
    }

    fn record_tick(&mut self, action: &Action, board: &Board, me: &Entity) -> Result<()> {
        // Dump binary data for all animations happening prior to this tick.
        if !self.animation.is_empty() {
            let filename = format!("{}/animation-{}.bin.gz", self.dir, self.next_tick);
            let file = BufWriter::new(File::create(&filename).unwrap());
            let mut file = GzEncoder::new(file, Compression::default());

            Self::write_bin(&mut file, &(self.animation.len() as i32))?;
            for frame in &self.animation {
                Self::write_bin(&mut file, &(frame.len() as i32))?;
                Self::write_array(&mut file, frame.as_slice())?;
            }
            file.flush()?;
        }

        // Dump binary data for the tick itself.
        let filename = format!("{}/tick-{}.bin.gz", self.dir, self.next_tick);
        let file = BufWriter::new(File::create(&filename).unwrap());
        let mut file = GzEncoder::new(file, Compression::default());

        // Dump binary data revealing all of the entities.
        let entities: Vec<_> = board.entities.iter().collect();
        Self::write_bin(&mut file, &(entities.len() as i32))?;

        for &(eid, entity) in &entities {
            let Entity { cur_hp, max_hp, .. } = *entity;
            let hp_fraction = cur_hp as f64 / max(max_hp, 1) as f64;
            let glyph = Self::entity_glyph(entity);

            Self::write_str(&mut file, &format!("{}", show(eid)))?;
            Self::write_str(&mut file, entity.species.name)?;
            Self::write_bin(&mut file, &hp_fraction)?;
            Self::write_bin(&mut file, &entity.pos)?;
            Self::write_bin(&mut file, &glyph)?;
        };

        // Dump info about our view of other entities.
        let mut sightings = vec![];
        for other in &me.known.sources {
            let color = {
                let current = me.known.get(other.pos).source();
                let moved = current.map(|x| x.uid) != Some(other.uid);
                if moved { 0xff0000 } else { 0xffff00 }
            };
            let glyph = Glyph::wdfg('?', Color::black()).with_bg(color);
            sightings.push((other.pos, glyph));
        }
        for other in &me.known.entities {
            let color = if other.visible { 0x00ff00 } else {
                let current = me.known.get(other.pos).entity();
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
        let mut lines = me.ai.get_trace(&me.known);
        lines.insert(0, DebugLine { color, depth: 0, text: "".into() });
        lines.insert(0, DebugLine { color, depth: 0, text: format!("{:?}", action) });
        Self::write_bin(&mut file, &(lines.len() as i32))?;

        for line in &lines {
            Self::write_bin(&mut file, &line.color)?;
            Self::write_bin(&mut file, &line.depth)?;
            Self::write_str(&mut file, &line.text)?;
        }

        // Dump a utility map, if this behavior tree tick used one.
        Self::write_bin(&mut file, &(self.utilities.len() as i32))?;
        Self::write_array(&mut file, self.utilities.as_slice())?;

        // Anything we scribble on the map shows up in the debug UI.
        for y in 0..self.map.size.1 {
            for x in 0..self.map.size.0 {
                let glyph = UI::render_tile(&me.known, Point(x, y));
                self.map.set(Point(x, y), glyph);
            }
        }
        for &p in me.ai.get_path() {
            if p == me.pos { continue; }
            let mut glyph = self.map.get(p);
            if glyph.ch() == Glyph::wide(' ').ch() { glyph = Glyph::wide('.'); }
            self.map.set(p, glyph.with_fg(0xff0000));
        }
        if let Some(&p) = me.ai.get_path().last() {
            self.map.set(p, self.map.get(p).with_fg(Color::black()).with_bg(0xff0000));
        }
        Self::write_bin(&mut file, &self.map.size)?;
        Self::write_array(&mut file, self.map.data.as_slice())?;
        file.flush()?;

        Result::Ok(())
    }

    fn write_array<W: Write, T>(f: &mut W, t: &[T]) -> Result<()> {
        let ptr = t.as_ptr() as *const u8;
        let len = t.len() * std::mem::size_of::<T>();
        f.write_all(unsafe { std::slice::from_raw_parts(ptr, len) })
    }

    fn write_bin<W: Write, T>(f: &mut W, t: &T) -> Result<()> {
        Self::write_array(f, std::slice::from_ref(t))
    }

    fn write_str<W: Write>(f: &mut W, s: &str) -> Result<()> {
        f.write_all(s.as_bytes())?;
        f.write_all(&[b'\0'])
    }

    fn entity_glyph(entity: &Entity) -> Glyph {
        let sneaking = entity.species.human && entity.sneaking;
        if sneaking { Glyph::wide('e') } else { entity.species.glyph }
    }

    fn knowledge_glyph(entity: &EntityKnowledge) -> Glyph {
        let sneaking = entity.species.human && entity.sneaking;
        if sneaking { Glyph::wide('e') } else { entity.species.glyph }
    }
}

//////////////////////////////////////////////////////////////////////////////
