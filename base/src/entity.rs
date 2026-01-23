use std::collections::VecDeque;
use std::iter::FusedIterator;
use std::num::NonZeroU64;
use std::ops::{Index, IndexMut};

use slotmap::{DefaultKey, Key, KeyData};
use slotmap::dense::DenseSlotMap;

use crate::static_assert_size;
use crate::ai::AIState;
use crate::base::{dirs, sample, Point, RNG};
use crate::dex::Species;
use crate::knowledge::{Knowledge, Scent};

//////////////////////////////////////////////////////////////////////////////

const SCENT_TRAIL_SIZE: usize = 64;
const SCENT_SPREAD: f64 = 1.;
const SCENT_DECAY: f64 = 1.;
const SCENT_BASE: f64 = 0.25;

//////////////////////////////////////////////////////////////////////////////

// Entity

pub struct EntityArgs {
    pub pos: Point,
    pub player: bool,
    pub predator: bool,
    pub species: &'static Species,
}

pub struct Entity {
    pub eid: EID,
    pub species: &'static Species,
    pub known: Box<Knowledge>,
    pub ai: Box<AIState>,
    pub cur_hp: i32,
    pub max_hp: i32,
    pub speed: f64,
    pub move_timer: i32,
    pub turn_timer: i32,

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
            species: args.species,
            known: Default::default(),
            ai: Box::new(AIState::new(args.predator, rng)),
            cur_hp: args.species.hp,
            speed: args.species.speed,
            max_hp: args.species.hp,
            move_timer: 0,
            turn_timer: 0,

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

    // Getters

    pub fn desc(&self) -> String {
        let name = self.species.name;
        if self.player { "you".into() } else { format!("the wild {}", name) }
    }

    pub fn hp_fraction(&self) -> f64 {
        self.cur_hp as f64 / std::cmp::max(self.max_hp, 1) as f64
    }

    pub fn get_scent_at(&self, p: Point) -> f64 {
        let mut total = 0.;
        for (_, scent) in self.get_scent_trail(p) { total += scent; }
        if total > 1. { 1. } else { total }
    }

    pub fn get_scent_trail(&self, p: Point) -> impl Iterator<Item = (&Scent, f64)> {
        let base = SCENT_BASE;
        let dropoff = 1. - SCENT_DECAY / (SCENT_TRAIL_SIZE as f64);
        let mut scale = 1.;

        self.trail.iter().enumerate().map(move |(i, scent)| {
            let variance = SCENT_SPREAD * (1. + 1. * i as f64);
            let l2_squared = (scent.pos - p).len_l2_squared() as f64;
            let num = (-l2_squared / (2. * variance)).exp();
            let den = (std::f64::consts::TAU * variance).sqrt();
            let value = base * num / den * scale;

            scale *= dropoff;

            (scent, value)
        })
    }

    pub fn too_big_to_hide(&self) -> bool {
        self.player && !self.sneaking
    }

    // Mutators

    pub fn face_direction(&mut self, dir: Point) {
        if dir != dirs::NONE { self.dir = dir; }
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

type BaseMap = DenseSlotMap<DefaultKey, Entity>;

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

pub struct Iter<'a>(slotmap::dense::Iter<'a, DefaultKey, Entity>);

pub struct IterMut<'a>(slotmap::dense::IterMut<'a, DefaultKey, Entity>);

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
