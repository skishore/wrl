use std::cmp::max;
use std::fs::File;
use std::io::{BufWriter, Result, Write};
use std::process::Command;

use crate::base::{Glyph, Matrix, Point};
use crate::entity::{EID, Entity};
use crate::game::{WORLD_SIZE, Action, Board};
use crate::ui::UI;

//////////////////////////////////////////////////////////////////////////////

fn show(eid: EID) -> u64 {
    unsafe { std::mem::transmute::<EID, u64>(eid) }
}

//////////////////////////////////////////////////////////////////////////////

pub struct DebugTrace {
    dir: &'static str,
    file: BufWriter<File>,
    map: Matrix<Glyph>,
    next_tick_index: usize,
}

impl Default for DebugTrace {
    fn default() -> Self {
        let dir = "wasm/debug";
        std::fs::remove_dir_all(dir).unwrap();
        std::fs::create_dir_all(dir).unwrap();
        let file = BufWriter::new(File::create(format!("{}/ticks.txt", dir)).unwrap());
        let map = Matrix::new(Point(WORLD_SIZE, WORLD_SIZE), Glyph::wide(' '));
        Self { dir, file, map, next_tick_index: 0 }
    }
}

impl DebugTrace {
    pub fn record(&mut self, action: &Action, board: &Board, me: &Entity) {
        self.record_one(action, board, me).unwrap();
    }

    fn record_one(&mut self, action: &Action, board: &Board, me: &Entity) -> Result<()> {
        self.record_tick(action, board, me)?;

        write!(self.file, "{{")?;
        write!(self.file, r#""time":"{}","#, board.time.nsec())?;
        write!(self.file, r#""eid":"{}""#, show(me.eid))?;
        write!(self.file, "}}\n")?;
        self.file.flush()
    }

    fn record_tick(&mut self, _: &Action, board: &Board, me: &Entity) -> Result<()> {
        let filename = format!("{}/tick-{}.bin", self.dir, self.next_tick_index);
        let mut file = BufWriter::new(File::create(&filename).unwrap());
        self.next_tick_index += 1;

        let entities: Vec<_> = board.entities.iter().collect();
        file.write_all(&(entities.len() as i32).to_le_bytes())?;

        for &(eid, entity) in &entities {
            let Entity { cur_hp, max_hp, .. } = *entity;
            let hp_fraction = cur_hp as f64 / max(max_hp, 1) as f64;

            let sneak = entity.species.human && entity.sneaking;
            let glyph = if sneak { Glyph::wide('e') } else { entity.species.glyph };

            Self::write_str(&mut file, &format!("{}", show(eid)))?;
            Self::write_str(&mut file, entity.species.name)?;
            Self::write_bin(&mut file, &hp_fraction)?;
            Self::write_bin(&mut file, &entity.pos)?;
            Self::write_bin(&mut file, &glyph)?;
        };

        for y in 0..self.map.size.1 {
            for x in 0..self.map.size.0 {
                let glyph = UI::render_tile(&me.known, Point(x, y));
                self.map.set(Point(x, y), glyph);
            }
        }
        Self::write_bin(&mut file, &self.map.size)?;
        Self::write_array(&mut file, self.map.data.as_slice())?;
        file.flush()?;

        std::mem::drop(file);
        Command::new("gzip").arg(filename).spawn()?;

        Result::Ok(())
    }

    fn write_array<T>(f: &mut BufWriter<File>, t: &[T]) -> Result<()> {
        let ptr = &t[0] as *const T as *const u8;
        let len = t.len() * std::mem::size_of::<T>();
        f.write_all(unsafe { std::slice::from_raw_parts(ptr, len) })
    }

    fn write_bin<T>(f: &mut BufWriter<File>, t: &T) -> Result<()> {
        Self::write_array(f, std::slice::from_ref(t))
    }

    fn write_str(f: &mut BufWriter<File>, s: &str) -> Result<()> {
        f.write_all(s.as_bytes())?;
        f.write_all(&[b'\0'])
    }
}

//////////////////////////////////////////////////////////////////////////////
