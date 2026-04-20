use std::num::NonZeroU64;

use crate::base::point::Point;

use super::dex::Species;
use super::entity::EID;
use super::time::Timestamp;

//////////////////////////////////////////////////////////////////////////////

// Events:

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct UID(pub NonZeroU64);

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Sense { Sight, Sound, Smell }

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Call { Command, Help, Warning }

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Sound { Attack, Call(Call), Move, Sniff }

#[derive(Clone, Debug)]
pub struct AttackEvent { pub combat: bool, pub target: Option<EID> }

#[derive(Clone, Debug)]
pub struct CallEvent { pub call: Call, pub species: &'static Species }

#[derive(Clone, Debug)]
pub struct MoveEvent { pub from: Point }

#[derive(Clone, Debug)]
pub enum EventData {
    Attack(AttackEvent),
    Call(CallEvent),
    Move(MoveEvent),
    Forget,
    Sniff,
    Spot,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct Location {
    pub pos: Point,
    pub time: Timestamp,
}

#[derive(Clone, Debug)]
pub struct Event {
    pub eid: Option<EID>,
    pub uid: Option<UID>,
    pub loc: Location,
    pub data: EventData,
    pub sense: Sense,
}

impl std::ops::Deref for Event {
    type Target = Location;
    fn deref(&self) -> &Self::Target { &self.loc }
}

impl Event {
    pub fn sound(&self) -> Option<Sound> {
        if self.sense != Sense::Sound { return None; }
        match &self.data {
            EventData::Attack(_) => Some(Sound::Attack),
            EventData::Call(x)   => Some(Sound::Call(x.call)),
            EventData::Move(_)   => Some(Sound::Move),
            EventData::Sniff     => Some(Sound::Sniff),
            EventData::Forget    => None,
            EventData::Spot      => None,
        }
    }
}
