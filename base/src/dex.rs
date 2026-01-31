use lazy_static::lazy_static;

use crate::base::{Bound, Glyph, HashMap, Point, RNG};
use crate::effect::*;

//////////////////////////////////////////////////////////////////////////////

// Attack

type AttackEffect = fn(&mut RNG, Point, Point) -> Effect;

pub struct Attack {
    pub name: &'static str,
    pub range: Bound,
    pub damage: i32,
    pub effect: AttackEffect,
}

impl Attack {
    pub fn get(name: &str) -> &'static Attack {
        ATTACKS.get(name).unwrap_or_else(|| panic!("Unknown attack: {}", name))
    }
}

impl std::fmt::Debug for Attack {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        f.write_str(self.name)
    }
}

lazy_static! {
    static ref ATTACKS: HashMap<&'static str, Attack> = {
        let items: Vec<(&'static str, i32, i32, AttackEffect)> = vec![
            ("Blizzard", 12, 120, BlizzardEffect),
            ("Ember",    12, 40,  EmberEffect),
            ("Headbutt", 6,  70,  HeadbuttEffect),
            ("Ice Beam", 12, 60,  IceBeamEffect),
            ("Tackle",   6,  40,  HeadbuttEffect),
        ];
        let mut result = HashMap::default();
        for (name, range, damage, effect) in items {
            let range = Bound::new(range);
            result.insert(name, Attack { name, range, damage, effect });
        }
        result
    };
}

//////////////////////////////////////////////////////////////////////////////

// Species

const FLAGS_NONE: u32 = 0;
const FLAGS_HUMAN: u32 = 1 << 0;
const FLAGS_PREDATOR: u32 = 1 << 1;

pub struct Species {
    pub name: &'static str,
    pub attacks: Vec<&'static Attack>,
    pub flags: u32,
    pub glyph: Glyph,
    pub light: Bound,
    pub speed: f64,
    pub hp: i32,
}

impl Species {
    pub fn get(name: &str) -> &'static Species {
        SPECIES.get(name).unwrap_or_else(|| panic!("Unknown species: {}", name))
    }

    // Raw flags-based predicates.
    pub fn human(&self) -> bool { self.flags & FLAGS_HUMAN != 0 }
    pub fn predator(&self) -> bool { self.flags & FLAGS_PREDATOR != 0 }
}

impl std::fmt::Debug for Species {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        f.write_str(self.name)
    }
}

impl Eq for &'static Species {}

impl PartialEq for &'static Species {
    fn eq(&self, next: &&'static Species) -> bool {
        *self as *const Species == *next as *const Species
    }
}

lazy_static! {
    static ref SPECIES: HashMap<&'static str, Species> = {
        let items = vec![
            ("Human",      ('@', 0xffffff), 0, 0, 0.9, 3,   vec![]),
            ("Pidgey",     ('P', 0xd0a070), 0, 0, 1.0, 200, vec!["Tackle"]),
            ("Rattata",    ('R', 0xa060ff), 1, 0, 1.0, 200, vec!["Tackle", "Headbutt"]),
            ("Charmander", ('C', 0xea8b24), 1, 4, 1.0, 200, vec!["Tackle", "Ember"]),
        ];
        let mut result = HashMap::default();
        for (name, glyph, predator, light, speed, hp, attacks) in items {
            let attacks = attacks.into_iter().map(&Attack::get).collect();
            let f0 = if name == "Human" { FLAGS_HUMAN } else { FLAGS_NONE };
            let f1 = if predator != 0 { FLAGS_PREDATOR } else { FLAGS_NONE };
            let flags = f0 | f1;
            let glyph = Glyph::wdfg(glyph.0, glyph.1);
            let light = Bound::new(if light == 0 { -1 } else { light });
            result.insert(name, Species { name, attacks, flags, glyph, light, speed, hp });
        }
        result
    };
}
