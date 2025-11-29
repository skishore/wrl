use std::cmp::{max, min};
use std::fmt::Debug;

use crate::base::{HashMap, Point, Slice};
use crate::entity::{Entity, EID};
use crate::knowledge::{EntityKnowledge, Event, EventData, Sense, Timedelta, Timestamp, UID};
use crate::list::{Handle, List};

//////////////////////////////////////////////////////////////////////////////

const ACTIVE_THREAT_TIME: Timedelta = Timedelta::from_seconds(96.);

const CALL_FOR_HELP_LIMIT: Timedelta = Timedelta::from_seconds(4.);
const CALL_FOR_HELP_RETRY: Timedelta = Timedelta::from_seconds(24.);

//////////////////////////////////////////////////////////////////////////////

// Threat state

pub type ThreatHandle = Handle<Threat>;

#[derive(Copy, Clone, Eq, PartialEq, Hash)]
pub enum TID { EID(EID), UID(UID) }

#[derive(Copy, Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum ThreatStatus { Hostile, Friendly, Neutral, Scanned, Unknown }

#[derive(Copy, Clone, Debug, Default, Eq, PartialEq)]
pub enum FightOrFlight { Fight, Flight, #[default] Safe }

#[derive(Clone)]
pub struct Threat {
    pub pos: Point,
    pub time: Timestamp,
    pub sense: Sense,
    pub status: ThreatStatus,
    pub combat: Timestamp,

    // Stats:
    pub hp: f64,
    pub delta: i32,

    // Flags:
    pub asleep: bool,
    pub seen: bool,
}

#[derive(Default)]
pub struct ThreatState {
    pub threats: List<Threat>,
    pub threat_index: HashMap<TID, ThreatHandle>,

    // Summaries used for flight pathing.
    pub hostile: Vec<Threat>,
    pub unknown: Vec<Threat>,

    // Fight-or-flight.
    pub state: FightOrFlight,

    // Calling for help.
    pub last_call: Timestamp,
    pub call_for_help: bool,
}

impl Threat {
    fn prior(me: &Entity) -> Self {
        Self {
            pos: Default::default(),
            time: Default::default(),
            sense: Sense::Sight,
            status: ThreatStatus::Unknown,
            combat: Default::default(),

            // Stats:
            hp: 1.,
            delta: if me.predator { -1 } else { 1 },

            // Flags:
            asleep: false,
            seen: false,
        }
    }
}

impl Threat {
    fn debug(&self, slice: &mut Slice, time: Timestamp) {
        let mut flags = vec![];
        if self.asleep { flags.push("Asleep"); }
        if self.seen { flags.push("Seen "); }
        if flags.is_empty() { flags.push("None"); }

        slice.write_str("Threat:").newline();
        slice.write_str(&format!("  age: {:?}", time - self.time)).newline();
        slice.write_str(&format!("  pos: {:?}", self.pos)).newline();
        slice.write_str(&format!("  sense: {:?}", self.sense)).newline();
        slice.write_str(&format!("  status: {:?}", self.status)).newline();
        slice.write_str(&format!("  combat: {:?}", time - self.combat)).newline();
        slice.write_str(&format!("  hp: {:.2}", self.hp)).newline();
        slice.write_str(&format!("  delta: {}", self.delta)).newline();
        slice.write_str(&format!("  flags: {}", flags.join(" | "))).newline();
    }

    fn merge_from(&mut self, other: &Threat) {
        // No need to update any fields that we unconditionally update in
        // update_for_event, since we merge right before processing an event.
        self.sense = other.sense;
        self.seen |= other.seen;
        self.update_status(other.status);
    }

    fn update_status(&mut self, status: ThreatStatus) {
        self.status = min(self.status, status);
    }

    fn update_for_event(&mut self, me: &Entity, event: &Event) {
        self.pos = event.point;
        self.time = event.time;
        self.sense = event.sense;
        self.asleep = false;

        match &event.data {
            EventData::Attack(x) => {
                let attacked = x.target == Some(me.eid);
                if attacked { self.update_status(ThreatStatus::Hostile); }
                self.combat = event.time;
            },
            EventData::CallForHelp(_) => {
                if !me.predator { self.update_status(ThreatStatus::Friendly); }
                self.combat = event.time;
            },
            EventData::Forget(_) => {},
            EventData::Move(_) => {},
            EventData::Spot(_) => {},
        }
    }

    fn update_for_sighting(&mut self, me: &Entity, other: &EntityKnowledge) {
        self.pos = other.pos;
        self.time = other.time;
        self.sense = other.sense;

        self.hp = other.hp;
        self.delta = other.delta;

        self.asleep = other.asleep;
        self.seen = true;

        let status = if other.player {
            ThreatStatus::Neutral
        } else if other.delta > 0 {
            ThreatStatus::Hostile
        } else if !me.predator && other.delta == 0 {
            ThreatStatus::Friendly
        } else {
            ThreatStatus::Neutral
        };
        self.update_status(status);
    }
}

impl ThreatState {
    pub fn debug(&self, slice: &mut Slice, time: Timestamp) {
        slice.write_str("ThreatState:").newline();
        slice.write_str(&format!("  state: {:?}", self.state)).newline();
        slice.write_str(&format!("  call_for_help: {}", self.call_for_help)).newline();
        slice.write_str(&format!("  last_call: {:?}", time - self.last_call)).newline();

        //slice.newline();
        //for x in &self.threats { x.debug(slice, time) }
    }

    pub fn update(&mut self, me: &Entity, prev_time: Timestamp) {
        for event in &me.known.events {
            // TODO(shaunak): Maybe only if it's a friendly call?
            if let EventData::CallForHelp(_) = event.data { self.last_call = event.time; }
            let Some(threat) = self.get_by_event(me, event) else { continue };
            threat.update_for_event(me, event);
        }
        for other in &me.known.entities {
            if !other.visible { break; }
            let Some(threat) = self.get_by_entity(me, other.eid) else { continue };
            threat.update_for_sighting(me, other);
        }

        self.hostile.clear();
        self.unknown.clear();

        let was_active = self.state != FightOrFlight::Safe;
        let mut active = was_active;
        let mut hidden_hostile = 0;
        let mut seen_hostile = 0;

        // List known enemies ("hostile") and potential enemies ("unknown").
        for x in &self.threats {
            if me.known.time - x.time > ACTIVE_THREAT_TIME { break; }

            if x.status == ThreatStatus::Hostile { self.hostile.push(x.clone()); }
            if x.status == ThreatStatus::Unknown { self.unknown.push(x.clone()); }

            if x.status == ThreatStatus::Hostile && !x.seen { hidden_hostile += 1; }
            if x.status == ThreatStatus::Hostile && x.seen { seen_hostile += 1; }
        }

        // Start fight-or-flight if we have an active known enemy. Stop when
        // we no longer have any known enemies. We also stop it with known
        // enemies if we escape from them (see: UpdateFlightState).
        if let Some(x) = self.hostile.first() && x.time > prev_time {
            active = true;
        } else if self.hostile.is_empty() {
            active = false;
        }

        // While active, also attack / flee from potential enemies.
        if active && !self.hostile.is_empty() {
            self.hostile.extend_from_slice(&self.unknown);
            self.hostile.sort_by_key(|x| me.known.time - x.time);
        }

        let strength = |x: &Threat| { 1.5f64.powi(x.delta.signum()) * x.hp };
        let mut hidden_count = max(hidden_hostile - seen_hostile, 0);
        let mut team_strength = me.hp_fraction();
        let mut call_strength = team_strength;
        let mut foes_strength = 0.;

        for x in &self.threats {
            if me.known.time - x.time > ACTIVE_THREAT_TIME { break; }

            if x.status == ThreatStatus::Hostile {
                if !x.seen && hidden_count == 0 { continue; }
                if !x.seen { hidden_count -= 1; }
                foes_strength += strength(x);
            } else if x.status == ThreatStatus::Friendly {
                let base = strength(x);
                let delay = me.known.time - x.combat;
                let ratio = delay.nsec() as f64 / ACTIVE_THREAT_TIME.nsec() as f64;
                let decay = if ratio > 1. { 0. } else { 1. - ratio };
                team_strength += base * decay;

                let can_call = me.known.time - x.time <= CALL_FOR_HELP_LIMIT;
                call_strength += if can_call { base } else { base * decay };
            }
        }

        let p = team_strength / (team_strength + foes_strength);
        let q = call_strength / (call_strength + foes_strength);

        if active && !was_active {
            self.state = if p > 0.5 { FightOrFlight::Fight } else { FightOrFlight::Flight };
        } else if active {
            if p > 0.6 { self.state = FightOrFlight::Fight; }
            if p < 0.4 { self.state = FightOrFlight::Flight; }
        } else {
            self.state = FightOrFlight::Safe;
        }

        self.call_for_help = false;
        if self.state == FightOrFlight::Flight && q > 0.6 &&
           (me.known.time - self.last_call) > CALL_FOR_HELP_RETRY {
            self.state = FightOrFlight::Fight;
            self.call_for_help = true;
        }

        debug_assert!(self.check_invariants());
    }

    fn get_by_entity(&mut self, me: &Entity, eid: EID) -> Option<&mut Threat> {
        let handle = self.get_by_tid(me, TID::EID(eid))?;
        Some(&mut self.threats[handle])
    }

    fn get_by_event(&mut self, me: &Entity, event: &Event) -> Option<&mut Threat> {
        let tid = event.eid.map(|x| TID::EID(x)).or(event.uid.map(|x| TID::UID(x)))?;

        if let EventData::Forget(_) = &event.data {
            if let Some(x) = self.threat_index.remove(&tid) { self.threats.remove(x); }
            return None;
        }

        let handle = self.get_by_tid(me, tid)?;

        if event.eid.is_some() && let Some(x) = event.uid &&
           let Some(x) = self.threat_index.remove(&TID::UID(x)) {
            let existing = self.threats.remove(x);
            self.threats[handle].merge_from(&existing);
        }

        Some(&mut self.threats[handle])
    }

    fn get_by_tid(&mut self, me: &Entity, tid: TID) -> Option<ThreatHandle> {
        if self.known_good(me, tid) { return None; }

        Some(*self.threat_index.entry(tid).and_modify(|&mut x| {
            self.threats.move_to_front(x);
        }).or_insert_with(|| {
            self.threats.push_front(Threat::prior(me))
        }))
    }

    fn known_good(&self, me: &Entity, tid: TID) -> bool {
        let TID::EID(x) = tid else { return false };
        x == me.eid || me.known.entity(x).map(|x| x.friend).unwrap_or(false)
    }

    fn check_invariants(&self) -> bool {
        // Check that threats are sorted by time.
        let check_sorted = |xs: Vec<Timestamp>| {
            let mut last = Timestamp::default();
            xs.into_iter().rev().for_each(|x| { assert!(x >= last); last = x; });
        };
        check_sorted(self.threats.iter().map(|x| x.time).collect());

        // Check that every threat is indexed in the index.
        assert!(self.threats.len() == self.threat_index.len());
        let mut handles = HashMap::default();
        for (&tid, &handle) in self.threat_index.iter() {
            assert!(handles.insert(handle, tid).is_none());
            let _ = &self.threats[handle];
        }
        true
    }
}
