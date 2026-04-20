use std::cmp::max;
use std::collections::VecDeque;

//////////////////////////////////////////////////////////////////////////////

// Timedelta

#[derive(Clone, Copy, Default, Eq, Ord, PartialEq, PartialOrd)]
pub struct Timedelta(i64);

impl Timedelta {
    const MSEC_PER_SEC: i64 = 1_000;
    const NSEC_PER_SEC: i64 = 1_000_000_000;
    const NSEC_PER_MSEC: i64 = 1_000_000;

    pub const fn nsec(&self) -> i64 { self.0 }

    pub const fn from_nsec(nsec: i64) -> Self { Self(nsec) }

    pub const fn seconds(&self) -> f64 {
        let factor = Self::NSEC_PER_SEC as f64;
        (1. / factor) * self.nsec() as f64
    }

    pub const fn from_seconds(seconds: f64) -> Self {
        let factor = Self::NSEC_PER_SEC as f64;
        Self::from_nsec((factor * seconds) as i64)
    }
}

impl std::ops::Add<Timedelta> for Timedelta {
    type Output = Timedelta;
    fn add(self, other: Timedelta) -> Self::Output {
        Self(self.0 + other.0)
    }
}

impl std::ops::Sub for Timedelta {
    type Output = Timedelta;
    fn sub(self, other: Timedelta) -> Self::Output {
        Self(self.0 - other.0)
    }
}

impl std::fmt::Debug for Timedelta {
    fn fmt(&self, fmt: &mut std::fmt::Formatter) -> Result<(), std::fmt::Error> {
        let mu = Timedelta::MSEC_PER_SEC;
        let ms = self.nsec() / Timedelta::NSEC_PER_MSEC;
        write!(fmt, "{}.{:0>3}s", ms / mu, ms % mu)
    }
}

//////////////////////////////////////////////////////////////////////////////

// Timestamp

#[derive(Clone, Copy, Default, Eq, Ord, PartialEq, PartialOrd)]
pub struct Timestamp(u64);

impl Timestamp {
    pub fn nsec(&self) -> u64 { self.0 }

    pub fn bump(&self) -> Self { self.latch(Timedelta::default()) }

    pub fn latch(&self, other: Timedelta) -> Self {
        let other = max(other, Timedelta(1));
        let value = self.0 + (other.0 as u64);
        let latch = value - (value % Timedelta::NSEC_PER_MSEC as u64);
        Self(if latch > self.0 { latch } else { value })
    }
}

impl std::ops::Sub for Timestamp {
    type Output = Timedelta;
    fn sub(self, other: Timestamp) -> Self::Output {
        Timedelta(self.0.wrapping_sub(other.0) as i64)
    }
}

impl std::ops::Sub<Timedelta> for Timestamp {
    type Output = Timestamp;
    fn sub(self, other: Timedelta) -> Self::Output {
        Timestamp((self.0 as i64 - other.0 as i64) as u64)
    }
}

impl std::fmt::Debug for Timestamp {
    fn fmt(&self, fmt: &mut std::fmt::Formatter) -> Result<(), std::fmt::Error> {
        let nu = Timedelta::NSEC_PER_MSEC as u64;
        let mu = Timedelta::MSEC_PER_SEC as u64;
        let (left, nsec) = (self.nsec() / nu, self.nsec() % nu);
        let (left, msec) = (left / mu, left % mu);
        let (left, sec) = (left / 60, left % 60);
        let (left, min) = (left / 60, left % 60);
        let (left, hrs) = (left / 24, left % 24);
        write!(fmt, "{}d {:0>2}:{:0>2}:{:0>2}.{:0>3} (+{})",
               left, hrs, min, sec, msec, nsec)
    }
}

//////////////////////////////////////////////////////////////////////////////

// Turn timing

#[derive(Default)]
pub struct TurnTimer {
    pub time: Timestamp,
    pub turn_times: VecDeque<Timestamp>,
}

impl TurnTimer {
    // Reads:

    pub fn debug_time(&self, time: Timestamp) -> String {
        if time == Timestamp::default() { return "<never>".into(); }

        let turns = self.time_to_turn(time);
        let age = self.time - time;

        let count = turns.ceil() as i32;
        let suffix = if count == 1 { "turn" } else { "turns" };
        let prefix = if turns == count as f64 { "" } else { "<" };
        format!("{:?} - {}{} {} ago", age, prefix, count, suffix)
    }

    pub fn time_at_turn(&self, turn: i32) -> Timestamp {
        if turn <= 0 { return self.time; }
        self.turn_times.get((turn - 1) as usize).map(|&x| x).unwrap_or_default()
    }

    pub fn time_to_turn(&self, time: Timestamp) -> f64 {
        let (mut prev, mut next) = (self.time, self.time);
        if time >= next { return 0.; }

        for (i, &n) in self.turn_times.iter().enumerate() {
            (prev, next) = (next, n);
            if time < next { continue; }

            let base = (i + 1) as f64;
            if time == next { return base; }

            let (a, b) = (prev - time, prev - next);
            return base - 1. + a.nsec() as f64 / max(b.nsec(), 1) as f64;
        }

        let base = self.turn_times.len() as f64;
        let (a, b) = (next - time, prev - next);
        base + a.nsec() as f64 / max(b.nsec(), 1) as f64
    }

    // Writes:

    pub fn end_turn(&mut self, limit: usize, speed: f64, time: Timestamp) {
        if self.turn_times.len() < limit {
            let min = 1e-2;
            let speed = if speed < min { min } else { speed };
            let seconds_per_turn = 1. / speed;

            self.turn_times.reserve_exact(limit);
            for i in 1..=limit {
                let age = Timedelta::from_seconds(i as f64 * seconds_per_turn);
                self.turn_times.push_back(time - age);
            }
        }

        assert!(self.turn_times[0] < time);
        assert!(self.turn_times.len() == limit);
        assert!(self.turn_times.capacity() == limit);
        self.turn_times.pop_back();
        self.turn_times.push_front(time);
    }

    pub fn update(&mut self, time: Timestamp) {
        assert!(time >= self.time);
        self.time = time;
    }
}
