use std::marker::PhantomData;

use crate::ai::Ctx;
use crate::game::Action;

//////////////////////////////////////////////////////////////////////////////

// Debug

#[derive(Default)]
pub struct Debug {
    pub depth: usize,
    pub lines: Vec<String>,
}

impl Debug {
    fn append(&mut self, t: impl std::fmt::Display) {
        self.lines.push(format!("{}{}", "  ".repeat(self.depth), t));
    }

    fn indent(&mut self, n: usize, f: impl Fn(&mut Debug) -> ()) {
        self.depth += n;
        f(self);
        self.depth -= n;
    }
}

//////////////////////////////////////////////////////////////////////////////

// Bhv

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum Result { Failed, Running, Success, #[default] Unused }

pub trait Bhv {
    fn debug(&self, debug: &mut Debug);
    fn tick(&mut self, ctx: &mut Ctx) -> Result;
}

//////////////////////////////////////////////////////////////////////////////

// Act

pub struct Act<T> {
    name: &'static str,
    step: T,
}

impl<T: Fn(&mut Ctx) -> Option<Action>> Act<T> {
    pub fn new(name: &'static str, step: T) -> Self {
        Self { name, step }
    }
}

impl<T: Fn(&mut Ctx) -> Option<Action>> Bhv for Act<T> {
    fn debug(&self, debug: &mut Debug) {
        debug.append(self.name);
    }

    fn tick(&mut self, ctx: &mut Ctx) -> Result {
        ctx.action = (self.step)(ctx);
        if ctx.action.is_some() { Result::Running } else { Result::Failed }
    }
}

//////////////////////////////////////////////////////////////////////////////

// Cond

pub struct Cond<T> {
    name: &'static str,
    test: T,
}

impl<T: Fn(&Ctx) -> bool> Cond<T> {
    pub fn new(name: &'static str, test: T) -> Self {
        Self { name, test }
    }
}

impl<T: Fn(&Ctx) -> bool> Bhv for Cond<T> {
    fn debug(&self, debug: &mut Debug) {
        debug.append(self.name);
    }

    fn tick(&mut self, ctx: &mut Ctx) -> Result {
        if (self.test)(ctx) { Result::Success } else { Result::Failed }
    }
}

//////////////////////////////////////////////////////////////////////////////

// Lambda

pub struct Lambda<T> {
    name: &'static str,
    tick: T,
}

impl<T: Fn(&mut Ctx) -> Result> Lambda<T> {
    pub fn new(name: &'static str, tick: T) -> Self {
        Self { name, tick }
    }
}

impl<T: Fn(&mut Ctx) -> Result> Bhv for Lambda<T> {
    fn debug(&self, debug: &mut Debug) {
        debug.append(self.name);
    }

    fn tick(&mut self, ctx: &mut Ctx) -> Result {
        (self.tick)(ctx)
    }
}

//////////////////////////////////////////////////////////////////////////////

// Composite

trait Label {
    fn show(&self) -> Option<&'static str>;
}

impl Label for () {
    fn show(&self) -> Option<&'static str> { None }
}

impl Label for &'static str {
    fn show(&self) -> Option<&'static str> { Some(self) }
}

trait Policy { const OKAY: Result; }

pub struct PriPolicy {}

impl Policy for PriPolicy {
    const OKAY: Result = Result::Failed;
}

pub struct SeqPolicy {}

impl Policy for SeqPolicy {
    const OKAY: Result = Result::Success;
}

pub struct Composite<R, S, T, U> {
    policy: PhantomData<R>,
    label: S,
    lhs: T,
    rhs: U,
}

impl<R, S, T, U> Composite<R, S, T, U> {
    pub fn new(label: S, policy: PhantomData<R>, lhs: T, rhs: U) -> Self {
        Self { label, policy, lhs, rhs }
    }
}

impl<R: Policy, S: Label, T: Bhv, U: Bhv> Bhv for Composite<R, S, T, U> {
    fn debug(&self, debug: &mut Debug) {
        let label = self.label.show();
        if let Some(x) = label { debug.append(x); }
        let n = if label.is_some() { 1 } else { 0 };
        debug.indent(n, |debug: &mut Debug| {
            self.lhs.debug(debug);
            self.rhs.debug(debug);
        });
    }

    fn tick(&mut self, ctx: &mut Ctx) -> Result {
        let result = self.lhs.tick(ctx);
        if result != R::OKAY && result != Result::Unused { return result; }
        self.rhs.tick(ctx)
    }
}

#[macro_export]
macro_rules! act {
    ($name:literal, $x:expr) => {
        crate::bhv::Act::new($name, $x)
    };
}

#[macro_export]
macro_rules! cb {
    ($name:literal, $x:expr) => {
        crate::bhv::Lambda::new($name, $x)
    };
}

#[macro_export]
macro_rules! cond {
    ($name:literal, $x:expr) => {
        crate::bhv::Cond::new($name, $x)
    };
}

#[macro_export]
macro_rules! pri {
    ($label:expr, $x:expr $(,)?) => { $x };
    ($label:expr, $x:expr $(,$xs:expr)+ $(,)?) => {{
        let policy = std::marker::PhantomData::<crate::bhv::PriPolicy>::default();
        crate::bhv::Composite::new($label, policy, $x, pri![() $(,$xs)+])
    }};
}

#[macro_export]
macro_rules! seq {
    ($label:expr, $x:expr $(,)?) => { $x };
    ($label:expr, $x:expr $(,$xs:expr)+ $(,)?) => {{
        let policy = std::marker::PhantomData::<crate::bhv::SeqPolicy>::default();
        crate::bhv::Composite::new($label, policy, $x, seq![() $(,$xs)+])
    }};
}
