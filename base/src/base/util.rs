use std::fmt::Debug;

use rand::Rng;

//////////////////////////////////////////////////////////////////////////////

// Macros:

#[macro_export]
macro_rules! gene {
    ($x:expr) => { $x }
}

#[macro_export]
macro_rules! static_assert_size {
    ($x:ty, $y:expr) => {
        const _: fn() = || { let _ = std::mem::transmute::<$x, [u8; $y]>; };
    }
}

#[macro_export]
macro_rules! flags {
    (@go $i:expr; $x:ident) => {
        #[allow(non_upper_case_globals)]
        pub const $x: Self = Self($i);
    };
    (@go $i:expr; $x:ident $($l:tt)+) => {
        crate::flags!(@go $i; $x);
        crate::flags!(@go 2 * $i; $($l)+);
    };
    (@derive $x:ident = $z:ident $(| $zs:ident)*,) => {
        #[allow(non_upper_case_globals)]
        pub const $x: Self = Self(Self::$z.0 $(| Self::$zs.0)*);
    };
    (@derive $x:ident = $z:ident $(| $zs:ident)*, $($l:tt)+) => {
        #[allow(non_upper_case_globals)]
        crate::flags!(@derive $x = $z $(| $zs)*,);
        crate::flags!(@derive $($l)+);
    };
    ($v:vis $n:ident($t:ty) { $($x:ident $(,)?)+
     $(#[$_:meta] $($y:ident = $z:ident $(| $zs:ident)* $(,)?)+)? }) => {
        #[derive(Clone, Copy, Default, Eq, PartialEq)]
        $v struct $n($t);
        impl $n {
            #[allow(dead_code,non_upper_case_globals)]
            pub const Empty: Self = Self(0);
            crate::flags!(@go 1; $($x)+);
            $($(crate::flags!(@derive $y = $z $(| $zs)*,);)+)?
            fn any(self: Self, r: Self) -> bool { self.0 & r.0 != 0 }
        }
        impl std::ops::Not for $n {
            type Output = Self;
            fn not(self: Self) -> Self::Output { Self(!self.0) }
        }
        impl std::ops::BitOr for $n {
            type Output = Self;
            fn bitor(self: Self, r: Self) -> Self::Output { Self(self.0 | r.0) }
        }
        impl std::ops::BitAnd for $n {
            type Output = Self;
            fn bitand(self: Self, r: Self) -> Self::Output { Self(self.0 & r.0) }
        }
        impl std::ops::BitOrAssign for $n {
            fn bitor_assign(self: &mut Self, r: Self) { self.0 |= r.0; }
        }
        impl std::ops::BitAndAssign for $n {
            fn bitand_assign(self: &mut Self, r: Self) { self.0 &= r.0; }
        }
    };
}

//////////////////////////////////////////////////////////////////////////////

// Utilities:

pub type HashSet<K> = fxhash::FxHashSet<K>;
pub type HashMap<K, V> = fxhash::FxHashMap<K, V>;

pub fn clamp<T: PartialOrd>(x: T, min: T, max: T) -> T {
    if x < min { min } else if x > max { max } else { x }
}

pub fn sortable(x: f64) -> u64 {
    let sign = 1 << 63;
    let bits = x.to_bits();
    if bits & sign == 0 { bits | sign } else { !bits }
}

//////////////////////////////////////////////////////////////////////////////

// RNG helpers:

pub type RNG = rand::rngs::StdRng;

pub fn sample<'a, T>(xs: &'a [T], rng: &mut RNG) -> &'a T {
    assert!(!xs.is_empty());
    &xs[rng.random_range(0..xs.len())]
}

pub fn weighted<'a, T: Debug>(xs: &'a [(i32, T)], rng: &mut RNG) -> &'a T {
    let total = xs.iter().fold(0, |acc, x| acc + x.0);
    assert!(total > 0, "Total: {}; values: {:?}", total, xs);
    let mut value = rng.random_range(0..total);
    for (weight, choice) in xs {
        value -= weight;
        if value <= 0 { return choice; }
    }
    assert!(false);
    &xs[xs.len() - 1].1
}
