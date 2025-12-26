use std::cmp::max;
use std::fs::File;
use std::io::{BufWriter, Result, Write};

use crate::base::{Glyph, Point};
use crate::entity::{EID, Entity};
use crate::game::{WORLD_SIZE, Action, Board};
use crate::ui::UI;

//////////////////////////////////////////////////////////////////////////////

pub struct DebugTrace {
    dir: &'static str,
    file: BufWriter<File>,
    next_tick_index: usize,
}

impl Default for DebugTrace {
    fn default() -> Self {
        let dir = "wasm/debug";
        std::fs::create_dir_all(dir).unwrap();
        let file = BufWriter::new(File::create(format!("{}/ticks.txt", dir)).unwrap());
        Self { dir, file, next_tick_index: 0 }
    }
}

impl DebugTrace {
    pub fn record(&mut self, action: &Action, board: &Board, entity: &Entity) {
        self.try_record(action, board, entity).unwrap();
    }

    fn try_record(&mut self, action: &Action, board: &Board, entity: &Entity) -> Result<()> {
        self.try_record_tick(entity)?;

        let Entity { cur_hp, max_hp, .. } = *entity;
        let hp_fraction = cur_hp as f64 / max(max_hp, 1) as f64;
        let eid = unsafe { std::mem::transmute::<EID, u64>(entity.eid) };

        write!(self.file, "{{")?;
        write!(self.file, r#""time":"{}","#, board.time.nsec())?;
        write!(self.file, r#""name":"{}","#, entity.species.name)?;
        write!(self.file, r#""eid":"{}","#, eid)?;
        write!(self.file, r#""health":{:.4},"#, hp_fraction)?;
        write!(self.file, r#""action":"{:?}""#, action)?;
        write!(self.file, "}}\n")?;
        self.file.flush()
    }

    fn try_record_tick(&mut self, entity: &Entity) -> Result<()> {
        let filename = format!("{}/tick-{}.bin", self.dir, self.next_tick_index);
        let mut file = BufWriter::new(File::create(filename).unwrap());
        self.next_tick_index += 1;

        file.write_all(&WORLD_SIZE.to_le_bytes())?;
        file.write_all(&WORLD_SIZE.to_le_bytes())?;

        let known = &entity.known;
        for y in 0..WORLD_SIZE {
            for x in 0..WORLD_SIZE {
                let glyph = UI::render_tile(known, Point(x, y));
                let glyph = unsafe { std::mem::transmute::<Glyph, u64>(glyph) };
                file.write_all(&glyph.to_le_bytes())?;
            }
        }
        file.flush()
    }
}
