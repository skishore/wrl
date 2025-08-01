use std::collections::VecDeque;
use std::iter::FusedIterator;
use std::num::NonZeroU64;
use std::ops::{Index, IndexMut};

use slotmap::{DefaultKey, Key, KeyData};
use slotmap::hop::HopSlotMap;

use crate::static_assert_size;
use crate::ai::AIState;
use crate::base::{dirs, sample, Glyph, Point, RNG};
use crate::knowledge::{Knowledge, Scent};

//////////////////////////////////////////////////////////////////////////////

const ATTACK_RANGE: i32 = 8;
const MAX_HP: i32 = 300;

const SCENT_TRAIL_SIZE: usize = 64;
const SCENT_SPREAD: f64 = 1.;
const SCENT_DECAY: f64 = 1.;
const SCENT_BASE: f64 = 0.25;

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
    pub trail: VecDeque<Scent>,

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
            trail: VecDeque::with_capacity(SCENT_TRAIL_SIZE),

            // Flags:
            asleep: false,
            player: args.player,
            predator: args.predator,
            sneaking: false,
        }
    }

    pub fn desc(&self) -> String {
        if self.player { "you".into() } else { format!("the wild {}", self.name()) }
    }

    pub fn name(&self) -> &'static str {
        if self.player { return "skishore"; }
        if self.predator { "Rattata" } else { "Pidgey" }
    }

    pub fn get_scent_at(&self, p: Point) -> f64 {
        let mut total = 0.;
        for age in 0..self.trail.capacity() {
            total += self.get_historical_scent_at(p, age);
        }
        if total > 1. { 1. } else { total }
    }

    pub fn get_historical_scent_at(&self, p: Point, age: usize) -> f64 {
        let Some(&Scent { pos, .. }) = self.trail.get(age) else { return 0. };

        let base = SCENT_BASE;
        let dropoff = 1. - SCENT_DECAY / (SCENT_TRAIL_SIZE as f64);
        let variance = SCENT_SPREAD * (1. + 1. * age as f64);

        let l2_squared = (pos - p).len_l2_squared() as f64;
        let num = (-l2_squared / (2. * variance)).exp();
        let den = (std::f64::consts::TAU * variance).sqrt();
        base * num / den * dropoff.powi(age as i32)
    }
}

//////////////////////////////////////////////////////////////////////////////

// EID

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct EID(NonZeroU64);
static_assert_size!(Option<EID>, 8);

impl From<DefaultKey> for EID {
    fn from(k: DefaultKey) -> Self {
        Self(NonZeroU64::new(k.data().as_ffi()).unwrap())
    }
}

impl EID {
    fn key(&self) -> DefaultKey {
        KeyData::from_ffi(self.0.get()).into()
    }
}

//////////////////////////////////////////////////////////////////////////////

// EntityMap

type BaseMap = HopSlotMap<DefaultKey, Entity>;

#[derive(Default)]
pub struct EntityMap(BaseMap);

impl EntityMap {
    pub fn add(&mut self, args: &EntityArgs, rng: &mut RNG) -> EID {
        self.0.insert_with_key(|x| Entity::new(x.into(), args, rng)).into()
    }

    pub fn clear(&mut self) { self.0.clear(); }

    pub fn get(&self, eid: EID) -> Option<&Entity> { self.0.get(eid.key()) }

    pub fn get_mut(&mut self, eid: EID) -> Option<&mut Entity> { self.0.get_mut(eid.key()) }

    pub fn has(&self, eid: EID) -> bool { self.0.contains_key(eid.key()) }

    pub fn remove(&mut self, eid: EID) -> Option<Entity> { self.0.remove(eid.key()) }

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
        self.0.next().map(|(k, v)| (k.into(), v))
    }
}

impl<'a> Iterator for IterMut<'a> {
    type Item = (EID, &'a mut Entity);
    fn next(&mut self) -> Option<Self::Item> {
        self.0.next().map(|(k, v)| (k.into(), v))
    }
}
