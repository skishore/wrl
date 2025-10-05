use std::marker::PhantomData;

// Debug helper:

#[derive(Default)]
struct Debug {
    depth: usize,
    lines: Vec<String>,
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

    fn render(&self) -> String {
        self.lines.join("\n")
    }
}

// Bhv trait:

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Result { Failed, Running, Success }

struct Ctx<'a> {
    rng: &'a mut i32,
}

trait Bhv {
    fn debug(&self, debug: &mut Debug);
    fn tick(&mut self, ctx: &mut Ctx) -> Result;
}

// Bhv implementations:

struct Cond<T> {
    name: &'static str,
    test: T,
}

impl<T: Fn(&Ctx) -> bool> Cond<T> {
    fn new(name: &'static str, test: T) -> Self {
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

// Helpers used to implement Sel and Seq

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

struct PriPolicy {}
impl Policy for PriPolicy { const OKAY: Result = Result::Failed; }

struct SeqPolicy {}
impl Policy for SeqPolicy { const OKAY: Result = Result::Success; }

struct Composite<R, S, T, U> {
    policy: PhantomData<R>,
    label: S,
    lhs: T,
    rhs: U,
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
        if result != R::OKAY { return result; }
        self.rhs.tick(ctx)
    }
}

macro_rules! composite {
    ($policy:ty, $label:expr, $x:expr) => { $x };
    ($policy:ty, $label:expr, $x:expr $(,$xs:expr)+) => {{
        let (lhs, rhs) = ($x, composite!($policy, () $(,$xs)+));
        Composite { label: $label, policy: PhantomData::<$policy>::default(), lhs, rhs }
    }};
}

macro_rules! pri {
    ($name:literal $(,$xs:expr)+ $(,)?) => { composite!(PriPolicy, $name $(,$xs)+) };
}

macro_rules! seq {
    ($name:literal $(,$xs:expr)+ $(,)?) => { composite!(SeqPolicy, $name $(,$xs)+) };
}

// Debugging helpers:

struct Test<T>(T);

impl<T: Bhv> Test<T> {
    #[inline(never)]
    fn tick(&mut self, ctx: &mut Ctx) -> Result {
        self.0.tick(ctx)
    }

    #[inline(never)]
    fn debug(&self) -> String {
        let mut debug = Debug::default();
        self.0.debug(&mut debug);
        debug.render()
    }
}

#[unsafe(no_mangle)]
fn make_tree() -> impl Bhv {
    seq![
        "TestSeq",
        Cond::new("CheckGt17", |x| *x.rng > 17),
        pri![
            "TestPri",
            Cond::new("CheckMod3", |x| *x.rng % 3 == 0),
            Cond::new("CheckMod5", |x| *x.rng % 5 == 0),
        ]
    ]
}

#[unsafe(no_mangle)]
fn entry(mut rng: i32) {
    let mut ctx = Ctx { rng: &mut rng };
    let mut tree = Test(make_tree());
    let result = tree.tick(&mut ctx);
    println!("{} -> {:?}", rng, result);
    println!("{}", tree.debug());
}

fn main() {
    entry(26);
}