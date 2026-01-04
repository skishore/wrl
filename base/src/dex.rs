use lazy_static::lazy_static;

use crate::base::{Glyph, HashMap, Point, RNG};
use crate::effect::*;
use crate::game::Board;

//////////////////////////////////////////////////////////////////////////////

// Attack

type AttackEffect = fn(&Board, &mut RNG, Point, Point) -> Effect;

pub struct Attack {
    pub name: &'static str,
    pub range: i32,
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
            result.insert(name, Attack { name, range, damage, effect });
        }
        result
    };
}

//////////////////////////////////////////////////////////////////////////////

// Species

pub struct Species {
    pub name: &'static str,
    pub attacks: Vec<&'static Attack>,
    pub glyph: Glyph,
    pub human: bool,
    pub speed: f64,
    pub hp: i32,
}

impl Species {
    pub fn get(name: &str) -> &'static Species {
        SPECIES.get(name).unwrap_or_else(|| panic!("Unknown species: {}", name))
    }
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
            ("Human",   ('@', 0xffffff), 0.9, 30,  vec![]),
            ("Pidgey",  ('P', 0xd0a070), 1.0, 200, vec!["Tackle"]),
            ("Rattata", ('R', 0xa060ff), 1.0, 200, vec!["Tackle", "Headbutt"]),
        ];
        let mut result = HashMap::default();
        for (name, glyph, speed, hp, attacks) in items {
            let human = name == "Human";
            let glyph = Glyph::wdfg(glyph.0, glyph.1);
            let attacks = attacks.into_iter().map(&Attack::get).collect();
            result.insert(name, Species { name, attacks, glyph, human, speed, hp });
        }
        result
    };
}
