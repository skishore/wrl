use std::cell::Cell;
use std::collections::VecDeque;
use std::iter::FusedIterator;
use std::num::NonZeroU64;
use std::ops::{Index, IndexMut};
use std::rc::Rc;

use crate::static_assert_size;
use crate::base::point::{Delta, Point, dirs};
use crate::base::util::{HashMap, RNG, sample};

use super::ai::AIState;
use super::dex::{Attack, Species};
use super::event::Location;
use super::game::MOVE_TIMER;
use super::knowledge::Knowledge;

//////////////////////////////////////////////////////////////////////////////

const SCENT_TRAIL_SIZE: usize = 64;
const SCENT_SPREAD: f64 = 1.;
const SCENT_DECAY: f64 = 1.;
const SCENT_BASE: f64 = 0.25;

//////////////////////////////////////////////////////////////////////////////

// Command

#[derive(Clone, Copy, Debug)]
pub struct AttackTarget { pub eid: Option<EID>, pub loc: Location, pub seen: bool }

#[derive(Clone, Copy, Debug)]
pub enum Command {
    Attack(&'static Attack, AttackTarget),
    Switch(usize),
    Return,
}

//////////////////////////////////////////////////////////////////////////////

// Entity

pub struct EntityArgs {
    pub name: Option<Rc<str>>,
    pub pos: Point,
    pub player: bool,
    pub leader: Option<EID>,
    pub species: &'static Species,
}

pub struct Individual {
    pub species: &'static Species,
    pub cur_hp: i32,
}

pub enum Teammate {
    Out(EID),
    In(Individual),
}

pub struct Entity {
    pub eid: EID,
    pub name: Option<Rc<str>>,
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
    pub dir: Delta,
    pub trail: VecDeque<Location>,

    // Team:
    pub leader: Option<EID>,
    pub command: Cell<Option<Command>>,
    pub summons: Vec<EID>,
    pub team: Vec<Teammate>,

    // Flags:
    pub asleep: bool,
    pub player: bool,
    pub sneaking: bool,
}

impl Entity {
    fn new(eid: EID, args: &EntityArgs, rng: &mut RNG) -> Self {
        Self {
            eid,
            name: args.name.clone(),
            species: args.species,
            known: Default::default(),
            ai: Box::new(AIState::new(rng)),
            cur_hp: args.species.hp,
            speed: args.species.speed,
            max_hp: args.species.hp,
            move_timer: if args.leader.is_some() { MOVE_TIMER } else { 0 },
            turn_timer: 0,

            // Location:
            pos: args.pos,
            dir: *sample(&dirs::ALL, rng),
            trail: VecDeque::with_capacity(SCENT_TRAIL_SIZE),

            // Team:
            leader: args.leader,
            command: None.into(),
            summons: vec![],
            team: vec![],

            // Flags:
            asleep: false,
            player: args.player,
            sneaking: false,
        }
    }

    // Text formatting:

    pub fn lower(&self) -> String {
        if self.player { return "you".into() };
        if let Some(x) = &self.name { return x.as_ref().into(); }

        let name = self.species.name;
        if self.leader.is_some() { name.into() } else { format!("the wild {}", name) }
    }

    pub fn upper(&self) -> String {
        if self.player { return "You".into() };
        if let Some(x) = &self.name { return x.as_ref().into(); }

        let name = self.species.name;
        if self.leader.is_some() { name.into() } else { format!("The wild {}", name) }
    }

    // Getters:

    pub fn hp_fraction(&self) -> f64 {
        self.cur_hp as f64 / std::cmp::max(self.max_hp, 1) as f64
    }

    pub fn get_scent_trail(&self, p: Point) -> impl Iterator<Item = (&Location, f64)> {
        let base = SCENT_BASE;
        let dropoff = 1. - SCENT_DECAY / (SCENT_TRAIL_SIZE as f64);
        let mut scale = 1.;

        self.trail.iter().enumerate().map(move |(i, loc)| {
            let variance = SCENT_SPREAD * (1. + 1. * i as f64);
            let l2_squared = (loc.pos - p).len_l2_squared() as f64;
            let num = (-l2_squared / (2. * variance)).exp();
            let den = (std::f64::consts::TAU * variance).sqrt();
            let scent = base * num / den * scale;

            scale *= dropoff;

            (loc, scent)
        })
    }

    pub fn to_individual(&self) -> Individual {
        let Self { species, cur_hp, .. } = *self;
        Individual { species, cur_hp }
    }

    pub fn too_big_to_hide(&self) -> bool {
        self.species.human() && !self.sneaking
    }

    // Mutators:

    pub fn face_direction(&mut self, dir: Delta) {
        if dir != dirs::NONE { self.dir = dir; }
    }
}

//////////////////////////////////////////////////////////////////////////////

// EID

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct EID(NonZeroU64);
static_assert_size!(Option<EID>, 8);

//////////////////////////////////////////////////////////////////////////////

// EntityMap

#[derive(Default)]
pub struct EntityMap {
    map: HashMap<EID, Entity>,
    next: Option<EID>,
    order: Vec<EID>,
}

impl EntityMap {
    pub fn add(&mut self, args: &EntityArgs, rng: &mut RNG) -> EID {
        let eid = self.next.unwrap_or(EID(NonZeroU64::MIN));
        self.next = Some(EID(eid.0.checked_add(1).unwrap()));
        let prev = self.map.insert(eid, Entity::new(eid, args, rng));
        assert!(prev.is_none());
        self.order.push(eid);
        eid
    }

    pub fn remove(&mut self, eid: EID) -> Option<Entity> {
        self.order.retain(|&x| x != eid);
        self.map.remove(&eid)
    }

    pub fn clear(&mut self) { *self = Default::default() }

    pub fn get(&self, eid: EID) -> Option<&Entity> { self.map.get(&eid) }

    pub fn get_mut(&mut self, eid: EID) -> Option<&mut Entity> { self.map.get_mut(&eid) }

    pub fn has(&self, eid: EID) -> bool { self.map.contains_key(&eid) }

    pub fn iter(&self) -> Iter<'_> {
        Iter(&self.map, self.order.iter())
    }

    pub fn iter_mut(&mut self) -> IterMut<'_> {
        IterMut(&mut self.map, self.order.iter())
    }
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

pub struct Iter<'a>(&'a HashMap<EID, Entity>, std::slice::Iter<'a, EID>);

pub struct IterMut<'a>(*mut HashMap<EID, Entity>, std::slice::Iter<'a, EID>);

impl<'a> FusedIterator for Iter<'a> {}

impl<'a> FusedIterator for IterMut<'a> {}

impl<'a> Iterator for Iter<'a> {
    type Item = (EID, &'a Entity);
    fn next(&mut self) -> Option<Self::Item> {
        self.1.next().map(|&x| (x, &self.0[&x]))
    }
}

// SAFETY: add and remove ensure that EIDs in EntityMap.order are unique.
impl<'a> Iterator for IterMut<'a> {
    type Item = (EID, &'a mut Entity);
    fn next(&mut self) -> Option<Self::Item> {
        self.1.next().map(|&x| (x, unsafe { &mut *self.0 }.get_mut(&x).unwrap()))
    }
}
