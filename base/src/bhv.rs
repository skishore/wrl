use crate::ai::Ctx;
use crate::debug::DebugLog;
use crate::game::Action;

//////////////////////////////////////////////////////////////////////////////

// Bhv

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Result { Failed, Running, Success }

pub trait Bhv {
    fn debug(&self, _: &mut DebugLog) {}
    fn reset(&mut self, _: &mut Ctx) {}
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

    pub fn on_exit<F: Fn(&mut Ctx) -> ()>(self, f: F) -> Node<S, OnExit<F, T>> {
        Node::new(self.label, OnExit(f, self.tree))
    }

    pub fn on_tick<F: Fn(&mut Ctx) -> ()>(self, f: F) -> Node<S, OnTick<F, T>> {
        Node::new(self.label, OnTick(f, self.tree))
    }

    pub fn post_tick<F: Fn(&mut Ctx) -> ()>(self, f: F) -> Node<S, PostTick<F, T>> {
        Node::new(self.label, PostTick(f, self.tree))
    }
}

impl<S: Label, T: Bhv> Bhv for Node<S, T> {
    fn debug(&self, debug: &mut DebugLog) {
        let Some(x) = self.label.show() else {
            return self.tree.debug(debug);
        };
        let color = match self.last {
            Some(Result::Failed)  => 0xff9090,
            Some(Result::Running) => 0x80c0ff,
            Some(Result::Success) => 0xe0ffc0,
            None                  => 0x545454,
        };
        debug.append(x);
        if let Some(x) = debug.lines.last_mut() { x.color = color.into(); };

        if !debug.verbose && self.last.is_none() { return; }

        debug.indent(1, |x| self.tree.debug(x));
    }

    fn reset(&mut self, ctx: &mut Ctx) {
        if self.last == None { return; }
        self.tree.reset(ctx);
        self.last = None;
    }

    fn tick(&mut self, ctx: &mut Ctx) -> Result {
        let last = self.tree.tick(ctx);
        self.last = Some(last);
        last
    }
}

//////////////////////////////////////////////////////////////////////////////

// Exit / Tick

pub struct OnExit<S, T>(S, T);

impl<S: Fn(&mut Ctx) -> (), T: Bhv> Bhv for OnExit<S, T> {
    fn debug(&self, debug: &mut DebugLog) { self.1.debug(debug) }
    fn reset(&mut self, ctx: &mut Ctx) { self.1.reset(ctx); (self.0)(ctx) }
    fn tick(&mut self, ctx: &mut Ctx) -> Result {
        let result = self.1.tick(ctx);
        if result == Result::Failed { (self.0)(ctx); }
        result
    }
}

pub struct OnTick<S, T>(S, T);

impl<S: Fn(&mut Ctx) -> (), T: Bhv> Bhv for OnTick<S, T> {
    fn debug(&self, debug: &mut DebugLog) { self.1.debug(debug) }
    fn reset(&mut self, ctx: &mut Ctx) { self.1.reset(ctx) }
    fn tick(&mut self, ctx: &mut Ctx) -> Result { (self.0)(ctx); self.1.tick(ctx) }
}

pub struct PostTick<S, T>(S, T);

impl<S: Fn(&mut Ctx) -> (), T: Bhv> Bhv for PostTick<S, T> {
    fn debug(&self, debug: &mut DebugLog) { self.1.debug(debug) }
    fn reset(&mut self, ctx: &mut Ctx) { self.1.reset(ctx) }
    fn tick(&mut self, ctx: &mut Ctx) -> Result { let x = self.1.tick(ctx); (self.0)(ctx); x }
}

//////////////////////////////////////////////////////////////////////////////

// Act

pub struct Act<T>(T);

impl<T: Fn(&mut Ctx) -> Option<Action>> Act<T> {
    pub fn new(t: T) -> Self { Self(t) }
}

impl<T: Fn(&mut Ctx) -> Option<Action>> Bhv for Act<T> {
    fn tick(&mut self, ctx: &mut Ctx) -> Result {
        let Some(action) = (self.0)(ctx) else { return Result::Failed };
        ctx.choose_action(action)
    }
}

//////////////////////////////////////////////////////////////////////////////

// Cond

pub struct Cond<T>(T);

impl<T: Fn(&mut Ctx) -> bool> Cond<T> {
    pub fn new(t: T) -> Self { Self(t) }
}

impl<T: Fn(&mut Ctx) -> bool> Bhv for Cond<T> {
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

// Utility

pub trait BhvPlusUtility : Bhv {
    fn util(&mut self, ctx: &mut Ctx) -> i64;
}

pub struct UtilityNode<S, T>(S, T);

impl<S: Fn(&mut Ctx) -> i64, T: Bhv> UtilityNode<S, T> {
    pub fn new(s: S, t: T) -> Self { Self(s, t) }
}

impl<S: Fn(&mut Ctx) -> i64, T: Bhv> Bhv for UtilityNode<S, T> {
    fn debug(&self, debug: &mut DebugLog) { self.1.debug(debug) }
    fn reset(&mut self, ctx: &mut Ctx) { self.1.reset(ctx) }
    fn tick(&mut self, ctx: &mut Ctx) -> Result { self.1.tick(ctx) }
}

impl<S: Fn(&mut Ctx) -> i64, T: Bhv> BhvPlusUtility for UtilityNode<S, T> {
    fn util(&mut self, ctx: &mut Ctx) -> i64 { (self.0)(ctx) }
}

pub struct Utility(Box<[Box<dyn BhvPlusUtility>]>, Box<[(i64, usize)]>);

impl Utility {
    pub fn new(xs: Vec<Box<dyn BhvPlusUtility>>) -> Self {
        let mut utilities = vec![];
        utilities.resize(xs.len(), (0, 0));
        Self(xs.into_boxed_slice(), utilities.into_boxed_slice())
    }
}

impl Bhv for Utility {
    fn debug(&self, debug: &mut DebugLog) {
        if !debug.verbose && self.1.iter().all(|x| x.0 < 0) { return; }

        for (i, x) in self.0.iter().enumerate() {
            let index = debug.lines.len();
            x.debug(debug);

            let mut value = -1;
            for &(utility, j) in &self.1 { if j == i { value = utility; } }

            if let Some(x) = debug.lines.get_mut(index) {
                x.text = format!("{} [{}]", x.text, value);
            }
        }
    }

    fn reset(&mut self, ctx: &mut Ctx) {
        for x in &mut self.0 { x.reset(ctx); }
        for x in &mut self.1 { x.1 = std::usize::MAX }
    }

    fn tick(&mut self, ctx: &mut Ctx) -> Result {
        for (i, x) in self.0.iter_mut().enumerate() {
            self.1[i] = (x.util(ctx), i);
        }
        self.1.sort_unstable_by_key(|&(utility, i)| (-utility, i));

        let mut result = Result::Failed;
        for &(utility, i) in &self.1 {
            if result == Result::Failed && utility >= 0 {
                result = self.0[i].tick(ctx);
            } else {
                self.0[i].reset(ctx);
            }
        }
        result
    }
}

//////////////////////////////////////////////////////////////////////////////

// Composite

trait Policy {
    fn okay(result: Result) -> bool;
}

pub struct PriPolicy {}

impl Policy for PriPolicy {
    fn okay(result: Result) -> bool { result == Result::Failed }
}

pub struct RunPolicy {}

impl Policy for RunPolicy {
    fn okay(_result: Result) -> bool { true }
}

pub struct SeqPolicy {}

impl Policy for SeqPolicy {
    fn okay(result: Result) -> bool { result == Result::Success }
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
    fn debug(&self, debug: &mut DebugLog) {
        self.lhs.debug(debug);
        self.rhs.debug(debug);
    }

    fn reset(&mut self, ctx: &mut Ctx) {
        self.lhs.reset(ctx);
        self.rhs.reset(ctx);
    }

    fn tick(&mut self, ctx: &mut Ctx) -> Result {
        let result = self.lhs.tick(ctx);
        if S::okay(result) { return self.rhs.tick(ctx); }
        self.rhs.reset(ctx);
        result
    }
}

#[macro_export]
macro_rules! act {
    ($name:literal, $x:expr) => {{
        $crate::bhv::Node::new(concat!("! ", $name), $crate::bhv::Act::new($x))
    }};
}

#[macro_export]
macro_rules! cb {
    ($name:literal, $x:expr) => {{
        $crate::bhv::Node::new(concat!("\\ ", $name), $crate::bhv::Lambda::new($x))
    }};
}

#[macro_export]
macro_rules! cond {
    ($name:literal, $x:expr) => {{
        $crate::bhv::Node::new(concat!("= ", $name), $crate::bhv::Cond::new($x))
    }};
}

#[macro_export]
macro_rules! util {
    ($name:literal $(,($x:expr, $y:expr))+ $(,)?) => {{
        use $crate::bhv::{Node, Utility, UtilityNode};
        let utility = Utility::new(vec![$(Box::new(UtilityNode::new($x, $y)),)+]);
        Node::new(concat!("# ", $name), utility)
    }};
}

#[macro_export]
macro_rules! pri {
    (@go $name:expr, $x:expr) => { $x };
    (@go $name:expr, $x:expr $(,$xs:expr)+) => {{
        use $crate::bhv::{Composite, Node, PriPolicy};
        Node::new($name, Composite::new(PriPolicy {}, $x, pri![@go () $(,$xs)+]))
    }};
    ($name:literal $(,$xs:expr)+ $(,)?) => { pri![@go concat!("? ", $name) $(,$xs)+] };
}

#[macro_export]
macro_rules! run {
    (@go $name:expr, $x:expr) => { $x };
    (@go $name:expr, $x:expr $(,$xs:expr)+) => {{
        use $crate::bhv::{Composite, Node, RunPolicy};
        Node::new($name, Composite::new(RunPolicy {}, $x, run![@go () $(,$xs)+]))
    }};
    ($name:literal $(,$xs:expr)+ $(,)?) => { run![@go concat!("* ", $name) $(,$xs)+] };
}

#[macro_export]
macro_rules! seq {
    (@go $name:expr, $x:expr) => { $x };
    (@go $name:expr, $x:expr $(,$xs:expr)+) => {{
        use $crate::bhv::{Composite, Node, SeqPolicy};
        Node::new($name, Composite::new(SeqPolicy {}, $x, seq![@go () $(,$xs)+]))
    }};
    ($name:literal $(,$xs:expr)+ $(,)?) => { seq![@go concat!("> ", $name) $(,$xs)+] };
}
