use crate::ai::Ctx;
use crate::base::Slice;
use crate::game::Action;

//////////////////////////////////////////////////////////////////////////////

// Debug

pub struct Debug<'a, 'b> {
    pub depth: usize,
    pub slice: &'a mut Slice<'b>,
}

impl<'a, 'b> Debug<'a, 'b> {
    fn append(&mut self, t: impl std::fmt::Display) {
        self.slice.spaces(2 * self.depth).write_str(&format!("{}", t)).newline();
    }

    fn indent(&mut self, n: usize, f: impl Fn(&mut Debug) -> ()) {
        self.depth += n;
        f(self);
        self.depth -= n;
    }
}

//////////////////////////////////////////////////////////////////////////////

// Bhv

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Result { Failed, Running, Success }

pub trait Bhv {
    fn debug(&self, _: &mut Debug) {}
    fn reset(&mut self) {}
    fn tick(&mut self, _: &mut Ctx) -> Result;
}

//////////////////////////////////////////////////////////////////////////////

// Node

pub trait Label {
    fn show(&self) -> Option<&'static str>;
}

impl Label for () {
    fn show(&self) -> Option<&'static str> { None }
}

impl Label for &'static str {
    fn show(&self) -> Option<&'static str> { Some(self) }
}

pub struct Node<S, T> {
    last: Option<Result>,
    label: S,
    tree: T,
}

impl<S: Label, T: Bhv> Node<S, T> {
    pub fn new(label: S, tree: T) -> Self {
        Self { last: None, label, tree }
    }
}

impl<S: Label, T: Bhv> Bhv for Node<S, T> {
    fn debug(&self, debug: &mut Debug) {
        let Some(x) = self.label.show() else {
            return self.tree.debug(debug);
        };
        let color = match self.last {
            Some(Result::Failed)  => 0xff9090,
            Some(Result::Running) => 0x80c0ff,
            Some(Result::Success) => 0xe0ffc0,
            None                  => 0x545454,
        };
        debug.slice.set_fg(Some(color.into()));
        debug.append(x);
        debug.indent(1, |x| self.tree.debug(x));
    }

    fn reset(&mut self) {
        if self.last == None { return; }
        self.tree.reset();
        self.last = None;
    }

    fn tick(&mut self, ctx: &mut Ctx) -> Result {
        let last = self.tree.tick(ctx);
        self.last = Some(last);
        last
    }
}

//////////////////////////////////////////////////////////////////////////////

// Act

pub struct Act<T>(T);

impl<T: Fn(&mut Ctx) -> Option<Action>> Act<T> {
    pub fn new(t: T) -> Self { Self(t) }
}

impl<T: Fn(&mut Ctx) -> Option<Action>> Bhv for Act<T> {
    fn tick(&mut self, ctx: &mut Ctx) -> Result {
        ctx.action = (self.0)(ctx);
        if ctx.action.is_some() { Result::Running } else { Result::Failed }
    }
}

//////////////////////////////////////////////////////////////////////////////

// Cond

pub struct Cond<T>(T);

impl<T: Fn(&Ctx) -> bool> Cond<T> {
    pub fn new(t: T) -> Self { Self(t) }
}

impl<T: Fn(&Ctx) -> bool> Bhv for Cond<T> {
    fn tick(&mut self, ctx: &mut Ctx) -> Result {
        if (self.0)(ctx) { Result::Success } else { Result::Failed }
    }
}

//////////////////////////////////////////////////////////////////////////////

// Lambda

pub struct Lambda<T>(T);

impl<T: Fn(&mut Ctx) -> Result> Lambda<T> {
    pub fn new(t: T) -> Self { Self(t) }
}

impl<T: Fn(&mut Ctx) -> Result> Bhv for Lambda<T> {
    fn tick(&mut self, ctx: &mut Ctx) -> Result { (self.0)(ctx) }
}

//////////////////////////////////////////////////////////////////////////////

// Composite

trait Policy { const OKAY: Result; }

pub struct PriPolicy {}

impl Policy for PriPolicy {
    const OKAY: Result = Result::Failed;
}

pub struct SeqPolicy {}

impl Policy for SeqPolicy {
    const OKAY: Result = Result::Success;
}

pub struct Composite<S, T, U> {
    _policy: S,
    lhs: T,
    rhs: U,
}

impl<S, T, U> Composite<S, T, U> {
    pub fn new(policy: S, lhs: T, rhs: U) -> Self {
        Self { _policy: policy, lhs, rhs }
    }
}

impl<S: Policy, T: Bhv, U: Bhv> Bhv for Composite<S, T, U> {
    fn debug(&self, debug: &mut Debug) {
        self.lhs.debug(debug);
        self.rhs.debug(debug);
    }

    fn reset(&mut self) {
        self.lhs.reset();
        self.rhs.reset();
    }

    fn tick(&mut self, ctx: &mut Ctx) -> Result {
        let result = self.lhs.tick(ctx);
        if result == S::OKAY { return self.rhs.tick(ctx); }
        self.rhs.reset();
        result
    }
}

#[macro_export]
macro_rules! act {
    ($name:literal, $x:expr) => {
        crate::bhv::Node::new(concat!("! ", $name), crate::bhv::Act::new($x))
    };
}

#[macro_export]
macro_rules! cb {
    ($name:literal, $x:expr) => {
        crate::bhv::Node::new(concat!("\\ ", $name), crate::bhv::Lambda::new($x))
    };
}

#[macro_export]
macro_rules! cond {
    ($name:literal, $x:expr) => {
        crate::bhv::Node::new(concat!("\\ ", $name), crate::bhv::Cond::new($x))
    };
}

#[macro_export]
macro_rules! pri {
    (@go $name:expr, $x:expr $(,)?) => { $x };
    (@go $name:expr, $x:expr $(,$xs:expr)+ $(,)?) => {{
        let policy = crate::bhv::PriPolicy {};
        let tree = crate::bhv::Composite::new(policy, $x, pri![@go () $(,$xs)+]);
        crate::bhv::Node::new($name, tree)
    }};
    ($name:literal $(,$xs:expr)+ $(,)?) => { pri![@go concat!("? ", $name) $(,$xs)+] };
}

#[macro_export]
macro_rules! seq {
    (@go $name:expr, $x:expr $(,)?) => { $x };
    (@go $name:expr, $x:expr $(,$xs:expr)+ $(,)?) => {{
        let policy = crate::bhv::SeqPolicy {};
        let tree = crate::bhv::Composite::new(policy, $x, seq![@go () $(,$xs)+]);
        crate::bhv::Node::new($name, tree)
    }};
    ($name:literal $(,$xs:expr)+ $(,)?) => { seq![@go concat!("> ", $name) $(,$xs)+] };
}
