use std::num::NonZeroU64;
use std::ops::{Index, IndexMut};

use slotmap::{DefaultKey, Key, KeyData, SlotMap};

use crate::static_assert_size;
use crate::ai::AIState;
use crate::base::{dirs, sample, Glyph, Point, RNG};
use crate::knowledge::Knowledge;

//////////////////////////////////////////////////////////////////////////////

const ATTACK_RANGE: i32 = 8;
const MAX_HP: i32 = 3;

//////////////////////////////////////////////////////////////////////////////

// Entity

pub struct EntityArgs {
    pub glyph: Glyph,
    pub player: bool,
    pub predator: bool,
    pub pos: Point,
    pub speed: f64,
}

pub struct Entity {
    pub eid: EID,
    pub glyph: Glyph,
    pub known: Box<Knowledge>,
    pub ai: Box<AIState>,
    pub cur_hp: i32,
    pub max_hp: i32,
    pub move_timer: i32,
    pub turn_timer: i32,
    pub range: i32,
    pub speed: f64,
    pub pos: Point,
    pub dir: Point,

    // Flags and other conditions
    pub asleep: bool,
    pub player: bool,
    pub predator: bool,
    pub sneaking: bool,
    pub tracking: i32,
}

//////////////////////////////////////////////////////////////////////////////

// EID

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

//////////////////////////////////////////////////////////////////////////////

// EntityMap

#[derive(Default)]
pub struct EntityMap {
    map: SlotMap<DefaultKey, Entity>,
}

impl EntityMap {
    pub fn add(&mut self, args: &EntityArgs, rng: &mut RNG) -> EID {
        let key = self.map.insert_with_key(|x| Entity {
            eid: to_eid(x),
            glyph: args.glyph,
            known: Default::default(),
            ai: Box::new(AIState::new(args.predator, rng)),
            cur_hp: MAX_HP,
            max_hp: MAX_HP,
            move_timer: 0,
            turn_timer: 0,
            range: ATTACK_RANGE,
            speed: args.speed,
            pos: args.pos,
            dir: *sample(&dirs::ALL, rng),

            // Flags and other conditions
            asleep: false,
            player: args.player,
            predator: args.predator,
            sneaking: false,
            tracking: 0,
        });
        to_eid(key)
    }

    pub fn clear(&mut self) { self.map.clear(); }

    pub fn get(&self, eid: EID) -> Option<&Entity> { self.map.get(to_key(eid)) }

    pub fn get_mut(&mut self, eid: EID) -> Option<&mut Entity> { self.map.get_mut(to_key(eid)) }

    pub fn has(&self, eid: EID) -> bool { self.map.contains_key(to_key(eid)) }

    pub fn remove(&mut self, eid: EID) -> Option<Entity> { self.map.remove(to_key(eid)) }
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
