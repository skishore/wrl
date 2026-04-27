use std::sync::LazyLock;

use crate::flags;
use crate::base::glyph::Glyph;
use crate::base::point::{Bound, Point};
use crate::base::util::{HashMap, RNG};

use super::effect::{Effect, self};

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

static ATTACKS: LazyLock<HashMap<&'static str, Attack>> = LazyLock::new(|| {
    let items: Vec<(&'static str, i32, i32, AttackEffect)> = vec![
        ("Blizzard", 12, 120, effect::BlizzardEffect),
        ("Ember",    12, 40,  effect::EmberEffect),
        ("Headbutt", 6,  70,  effect::HeadbuttEffect),
        ("Ice Beam", 12, 60,  effect::IceBeamEffect),
        ("Tackle",   6,  40,  effect::HeadbuttEffect),
    ];
    let mut result = HashMap::default();
    for (name, range, damage, effect) in items {
        let range = Bound::new(range);
        result.insert(name, Attack { name, range, damage, effect });
    }
    result
});

//////////////////////////////////////////////////////////////////////////////

// Species

flags! { pub SpeciesFlags(u32) { Human, Predator } }

type SF = SpeciesFlags;

pub struct Species {
    pub name: &'static str,
    pub attacks: Vec<&'static Attack>,
    pub flags: SpeciesFlags,
    pub glyph: Glyph,
    pub light: Bound,
    pub scent: f64,
    pub speed: f64,
    pub hp: i32,
}

impl Species {
    pub fn get(name: &str) -> &'static Species {
        SPECIES.get(name).unwrap_or_else(|| panic!("Unknown species: {}", name))
    }

    // Raw flags-based predicates:

    pub fn human(&self) -> bool { self.flags.any(SF::Human) }
    pub fn predator(&self) -> bool { self.flags.any(SF::Predator) }
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

static SPECIES: LazyLock<HashMap<&'static str, Species>> = LazyLock::new(|| {
    let items = vec![
        ("Human",      0xffffff, 0, 0, 0.0,  0.9, 3,   vec![]),
        ("Pidgey",     0xd0a070, 0, 0, 0.25, 1.0, 200, vec!["Tackle"]),
        ("Rattata",    0xa060ff, 1, 0, 1.0,  1.0, 200, vec!["Tackle", "Headbutt"]),
        ("Bulbasaur",  0x408020, 0, 0, 0.5,  1.0, 300, vec!["Tackle"]),
        ("Charmander", 0xea8b24, 1, 4, 1.0,  1.0, 200, vec!["Tackle", "Ember"]),
        ("Squirtle",   0x80c0ff, 0, 0, 0.5,  1.0, 200, vec!["Tackle", "Ice Beam"]),
        ("Pikachu",    0xffff00, 0, 4, 1.0,  1.1, 200, vec!["Tackle"]),
        ("Eevee",      0xd0a070, 0, 0, 1.0,  1.0, 200, vec!["Tackle", "Headbutt"]),
    ];
    let mut result = HashMap::default();
    for (name, color, predator, light, scent, speed, hp, attacks) in items {
        let attacks = attacks.into_iter().map(&Attack::get).collect();
        let ch = if name == "Human" { '@' } else { name.chars().next().unwrap() };
        let f0 = if name == "Human" { SF::Human } else { SF::Empty };
        let f1 = if predator != 0 { SF::Predator } else { SF::Empty };
        let flags = f0 | f1;
        let glyph = Glyph::wdfg(ch, color);
        let light = Bound::new(if light == 0 { -1 } else { light });
        result.insert(name, Species {
            name, attacks, flags, glyph, light, scent, speed, hp });
    }
    result
});
