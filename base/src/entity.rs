use std::collections::VecDeque;
use std::iter::FusedIterator;
use std::num::NonZeroU64;
use std::ops::{Index, IndexMut};

use slotmap::{DefaultKey, Key, KeyData};
use slotmap::hop::HopSlotMap;

use crate::static_assert_size;
use crate::ai::AIState;
use crate::base::{dirs, sample, Glyph, Point, RNG};
use crate::knowledge::Knowledge;

//////////////////////////////////////////////////////////////////////////////

const ATTACK_RANGE: i32 = 8;
const HISTORY_SIZE: usize = 64;
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

    // Location:
    pub pos: Point,
    pub dir: Point,
    pub history: VecDeque<Point>,

    // Flags:
    pub asleep: bool,
    pub player: bool,
    pub predator: bool,
    pub sneaking: bool,
}

impl Entity {
    fn new(eid: EID, args: &EntityArgs, rng: &mut RNG) -> Self {
        Self {
            eid,
            glyph: args.glyph,
            known: Default::default(),
            ai: Box::new(AIState::new(args.predator, rng)),
            cur_hp: MAX_HP,
            max_hp: MAX_HP,
            move_timer: 0,
            turn_timer: 0,
            range: ATTACK_RANGE,
            speed: args.speed,

            // Location:
            pos: args.pos,
            dir: *sample(&dirs::ALL, rng),
            history: VecDeque::with_capacity(HISTORY_SIZE),

            // Flags:
            asleep: false,
            player: args.player,
            predator: args.predator,
            sneaking: false,
        }
    }

    pub fn get_scent_at(&self, p: Point) -> f64 {
        let mut total = 0.;
        for age in 0..self.history.capacity() {
            total += self.get_historical_scent_at(p, age);
        }
        if total > 1. { 1. } else { total }
    }

    pub fn get_historical_scent_at(&self, p: Point, age: usize) -> f64 {
        let Some(&pos) = self.history.get(age) else { return 0. };

        let base = 0.2;
        let dropoff = 1. - 1. / (HISTORY_SIZE as f64);
        let variance = 1. + 1. * age as f64;

        let l2_squared = (pos - p).len_l2_squared() as f64;
        let num = (-l2_squared / (2. * variance)).exp();
        let den = (std::f64::consts::TAU * variance).sqrt();
        base * num / den * dropoff.powi(age as i32)
    }
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

type BaseMap = HopSlotMap<DefaultKey, Entity>;

#[derive(Default)]
pub struct EntityMap(BaseMap);

impl EntityMap {
    pub fn add(&mut self, args: &EntityArgs, rng: &mut RNG) -> EID {
        to_eid(self.0.insert_with_key(|x| Entity::new(to_eid(x), args, rng)))
    }

    pub fn clear(&mut self) { self.0.clear(); }

    pub fn get(&self, eid: EID) -> Option<&Entity> { self.0.get(to_key(eid)) }

    pub fn get_mut(&mut self, eid: EID) -> Option<&mut Entity> { self.0.get_mut(to_key(eid)) }

    pub fn has(&self, eid: EID) -> bool { self.0.contains_key(to_key(eid)) }

    pub fn remove(&mut self, eid: EID) -> Option<Entity> { self.0.remove(to_key(eid)) }

    pub fn iter(&self) -> Iter<'_> { Iter(self.0.iter()) }

    pub fn iter_mut(&mut self) -> IterMut<'_> { IterMut(self.0.iter_mut()) }
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

impl<'a> IntoIterator for &'a EntityMap {
    type Item = (EID, &'a Entity);
    type IntoIter = Iter<'a>;
    fn into_iter(self) -> Self::IntoIter { self.iter() }
}


impl<'a> IntoIterator for &'a mut EntityMap {
    type Item = (EID, &'a mut Entity);
    type IntoIter = IterMut<'a>;
    fn into_iter(self) -> Self::IntoIter { self.iter_mut() }
}

//////////////////////////////////////////////////////////////////////////////

// EntityMap iterators

pub struct Iter<'a>(slotmap::hop::Iter<'a, DefaultKey, Entity>);

pub struct IterMut<'a>(slotmap::hop::IterMut<'a, DefaultKey, Entity>);

impl<'a> FusedIterator for Iter<'a> {}

impl<'a> FusedIterator for IterMut<'a> {}

impl<'a> Iterator for Iter<'a> {
    type Item = (EID, &'a Entity);
    fn next(&mut self) -> Option<Self::Item> {
        self.0.next().map(|(k, v)| (to_eid(k), v))
    }
}

impl<'a> Iterator for IterMut<'a> {
    type Item = (EID, &'a mut Entity);
    fn next(&mut self) -> Option<Self::Item> {
        self.0.next().map(|(k, v)| (to_eid(k), v))
    }
}
