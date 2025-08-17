use std::iter::FusedIterator;
use std::marker::PhantomData;
use std::mem::MaybeUninit;
use std::num::NonZeroU32;

use crate::static_assert_size;

//////////////////////////////////////////////////////////////////////////////

// LinkedList

pub struct Handle<T>(NonZeroU32, PhantomData<T>);
static_assert_size!(Handle<u64>, 4);
static_assert_size!(Option<Handle<u64>>, 4);

// Low bit: 0 for items in the used list; 1 for items in the free list
// High bits: index + 1 (if the node is in the vec); 0, if it's a dummy node
//
// Maximum size of the list: i32::MAX - 2
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct Index(u32);

#[derive(Copy, Clone, Debug)]
struct Link {
    prev: Index,
    next: Index,
}

#[derive(Debug)]
struct Node<T> {
    link: Link,
    data: MaybeUninit<T>,
}

#[derive(Debug)]
pub struct List<T> {
    heads: [Link; 2],
    nodes: Vec<Node<T>>,
    length: usize,
}

// Iterators
//
// Implementation note: for Iter and IterMut, the range of elements remaining
// in the iterator is [next, prev); i.e. next is exclusive, prev is exclusive.
//
// We chose this encoding so that iteration is complete iff next == prev.

pub struct IntoIter<T> {
    list: List<T>,
}

pub struct Iter<'a, T> {
    list: &'a List<T>,
    link: Link,
}

pub struct IterMut<'a, T> {
    list: &'a mut List<T>,
    link: Link,
}

// Handle

impl<T> Copy for Handle<T> {}

impl<T> Clone for Handle<T> {
    fn clone(&self) -> Self { Self(self.0, PhantomData) }
}

impl<T> Eq for Handle<T> {}

impl<T> PartialEq for Handle<T> {
    fn eq(&self, other: &Self) -> bool { self.0 == other.0 }
}

impl<T> std::hash::Hash for Handle<T> {
    fn hash<H: std::hash::Hasher>(&self, h: &mut H) { self.0.hash(h) }
}

impl<T> std::ops::Index<Handle<T>> for List<T> {
    type Output = T;
    fn index(&self, h: Handle<T>) -> &Self::Output {
        let node = self.node(h);
        assert!(node.link.next.is_used());
        unsafe { node.data.assume_init_ref() }
    }
}

impl<T> std::ops::IndexMut<Handle<T>> for List<T> {
    fn index_mut(&mut self, h: Handle<T>) -> &mut Self::Output {
        let node = self.node_mut(h);
        assert!(node.link.next.is_used());
        unsafe { node.data.assume_init_mut() }
    }
}

impl<T> Handle<T> {
    unsafe fn unchecked(x: Index) -> Self {
        debug_assert!(!x.is_dummy());
        Self(unsafe { NonZeroU32::new_unchecked(x.0) }, PhantomData)
    }

    fn index(self) -> Index { Index(self.0.get()) }
}

// Index

impl Index {
    fn is_dummy(&self) -> bool { self.0 < 2 }

    fn is_used(&self) -> bool { self.0 & 1 == 0 }

    fn to_free_index(&self) -> Index { Index(self.0 | 1) }

    fn to_used_index(&self) -> Index { Index(self.0 & !1) }
}

// Iterators

impl<T> FusedIterator for IntoIter<T> {}

impl<T> Iterator for IntoIter<T> {
    type Item = T;
    fn next(&mut self) -> Option<Self::Item> { self.list.pop_front() }
}

impl<T> DoubleEndedIterator for IntoIter<T> {
    fn next_back(&mut self) -> Option<Self::Item> { self.list.pop_back() }
}

impl<'a, T> FusedIterator for Iter<'a, T> {}

impl<'a, T> Iterator for Iter<'a, T> {
    type Item = &'a T;
    fn next(&mut self) -> Option<Self::Item> {
        if self.link.prev == self.link.next { return None; }
        let handle = unsafe { Handle::unchecked(self.link.next) };
        self.link.next = self.list.link(self.link.next).next;
        Some(unsafe { self.list.node(handle).data.assume_init_ref() })
    }
}

impl<'a, T> DoubleEndedIterator for Iter<'a, T> {
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.link.prev == self.link.next { return None; }
        self.link.prev = self.list.link(self.link.prev).prev;
        let handle = unsafe { Handle::unchecked(self.link.prev) };
        Some(unsafe { self.list.node(handle).data.assume_init_ref() })
    }
}

impl<'a, T> FusedIterator for IterMut<'a, T> {}

impl<'a, T> Iterator for IterMut<'a, T> {
    type Item = &'a mut T;
    fn next(&mut self) -> Option<&'a mut T> {
        if self.link.prev == self.link.next { return None; }
        let handle = unsafe { Handle::unchecked(self.link.next) };
        self.link.next = self.list.link(self.link.next).next;
        let x = unsafe { self.list.node_mut(handle).data.assume_init_mut() };
        Some(unsafe { &mut *(x as *mut T) })
    }
}

impl<'a, T> DoubleEndedIterator for IterMut<'a, T> {
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.link.prev == self.link.next { return None; }
        self.link.prev = self.list.link(self.link.prev).prev;
        let handle = unsafe { Handle::unchecked(self.link.prev) };
        let x = unsafe { self.list.node_mut(handle).data.assume_init_mut() };
        Some(unsafe { &mut *(x as *mut T) })
    }
}

// List

impl<T> Default for List<T> {
    fn default() -> Self {
        let used = Link { prev: Index(0), next: Index(0) };
        let free = Link { prev: Index(1), next: Index(1) };
        Self { heads: [used, free], nodes: vec![], length: 0 }
    }
}

impl<T> Drop for List<T> {
    fn drop(&mut self) {
        while !self.is_empty() { self.pop_front(); }
    }
}

impl<T> IntoIterator for List<T> {
    type Item = T;
    type IntoIter = IntoIter<T>;
    fn into_iter(self) -> Self::IntoIter { self.into_iter() }
}

impl<'a, T> IntoIterator for &'a List<T> {
    type Item = &'a T;
    type IntoIter = Iter<'a, T>;
    fn into_iter(self) -> Self::IntoIter { self.iter() }
}

impl<'a, T> IntoIterator for &'a mut List<T> {
    type Item = &'a mut T;
    type IntoIter = IterMut<'a, T>;
    fn into_iter(self) -> Self::IntoIter { self.iter_mut() }
}

impl<T> List<T> {
    pub fn back(&self) -> Option<&T> {
        if self.is_empty() { return None; }
        Some(&self[unsafe { Handle::unchecked(self.heads[0].prev) }])
    }

    pub fn back_handle(&self) -> Option<Handle<T>> {
        if self.is_empty() { return None; }
        Some(unsafe { Handle::unchecked(self.heads[0].prev) })
    }

    pub fn front(&self) -> Option<&T> {
        if self.is_empty() { return None; }
        Some(&self[unsafe { Handle::unchecked(self.heads[0].next) }])
    }

    pub fn front_handle(&self) -> Option<Handle<T>> {
        if self.is_empty() { return None; }
        Some(unsafe { Handle::unchecked(self.heads[0].next) })
    }

    pub fn len(&self) -> usize { self.length }

    pub fn is_empty(&self) -> bool { self.heads[0].next == Index(0) }

    pub fn into_iter(self) -> IntoIter<T> { IntoIter { list: self } }

    pub fn iter(&self) -> Iter<'_, T> {
        let link = Link { prev: Index(0), next: self.heads[0].next };
        Iter { list: self, link }
    }

    pub fn iter_mut(&mut self) -> IterMut<'_, T> {
        let link = Link { prev: Index(0), next: self.heads[0].next };
        IterMut { list: self, link }
    }

    pub fn move_to_back(&mut self, h: Handle<T>) {
        let node = self.node_mut(h);
        assert!(node.link.next.is_used());

        let Link { prev, next } = node.link;
        self.link_mut(prev).next = next;
        self.link_mut(next).prev = prev;

        let tail = self.heads[0].prev;
        self.node_mut(h).link = Link { prev: tail, next: Index(0) };
        self.link_mut(tail).next = h.index();
        self.heads[0].prev = h.index();
    }

    pub fn move_to_front(&mut self, h: Handle<T>) {
        let node = self.node_mut(h);
        assert!(node.link.next.is_used());

        let Link { prev, next } = node.link;
        self.link_mut(prev).next = next;
        self.link_mut(next).prev = prev;

        let head = self.heads[0].next;
        self.node_mut(h).link = Link { prev: Index(0), next: head };
        self.link_mut(head).prev = h.index();
        self.heads[0].next = h.index();
    }

    pub fn pop_back(&mut self) -> Option<T> {
        if self.is_empty() { return None; }
        let tail = unsafe { Handle::unchecked(self.heads[0].prev) };
        let prev = self.node(tail).link.prev;

        self.link_mut(prev).next = Index(0);
        self.heads[0].prev = prev;
        self.length -= 1;
        Some(self.push_front_free(tail))
    }

    pub fn pop_front(&mut self) -> Option<T> {
        if self.is_empty() { return None; }
        let head = unsafe { Handle::unchecked(self.heads[0].next) };
        let next = self.node(head).link.next;

        self.link_mut(next).prev = Index(0);
        self.heads[0].next = next;
        self.length -= 1;
        Some(self.push_front_free(head))
    }

    pub fn push_back(&mut self, value: T) -> Handle<T> {
        let tail = self.pop_front_free();
        let prev = self.heads[0].prev;
        let node = self.node_mut(tail);
        node.link = Link { prev, next: Index(0) };
        node.data.write(value);

        self.link_mut(prev).next = tail.index();
        self.heads[0].prev = tail.index();
        self.length += 1;
        tail
    }

    pub fn push_front(&mut self, value: T) -> Handle<T> {
        let head = self.pop_front_free();
        let next = self.heads[0].next;
        let node = self.node_mut(head);
        node.link = Link { prev: Index(0), next };
        node.data.write(value);

        self.link_mut(next).prev = head.index();
        self.heads[0].next = head.index();
        self.length += 1;
        head
    }

    pub fn remove(&mut self, h: Handle<T>) -> T {
        let node = self.node_mut(h);
        assert!(node.link.next.is_used());

        let Link { prev, next } = node.link;
        self.link_mut(prev).next = next;
        self.link_mut(next).prev = prev;
        self.length -= 1;
        self.push_front_free(h)
    }

    pub fn reserve(&mut self, additional: usize) {
        self.nodes.reserve(additional);
        for _ in 0..additional {
            let tail = Index(((self.nodes.len() + 1) << 1) as u32);
            let link = Link { prev: self.heads[1].prev, next: Index(1) };
            self.nodes.push(Node { link, data: MaybeUninit::uninit() });
            self.link_mut(link.prev).next = tail;
            self.heads[1].prev = tail;
        }
    }

    fn link(&self, x: Index) -> &Link {
        if x.0 < 2 { return &self.heads[x.0 as usize & 1]; }
        &self.node(unsafe { Handle::unchecked(x) }).link
    }

    fn link_mut(&mut self, x: Index) -> &mut Link {
        if x.0 < 2 { return &mut self.heads[x.0 as usize & 1]; }
        &mut self.node_mut(unsafe { Handle::unchecked(x) }).link
    }

    fn node(&self, x: Handle<T>) -> &Node<T> {
        &self.nodes[(x.0.get() as usize >> 1) - 1]
    }

    fn node_mut(&mut self, x: Handle<T>) -> &mut Node<T> {
        &mut self.nodes[(x.0.get() as usize >> 1) - 1]
    }

    fn pop_front_free(&mut self) -> Handle<T> {
        let index = if self.heads[1].next == Index(1) {
            self.nodes.push(unsafe { std::mem::zeroed() });
            Index((self.nodes.len() << 1) as u32)
        } else {
            let result = self.heads[1].next;
            let Link { prev, next } = *self.link(result);
            self.link_mut(prev).next = next;
            self.link_mut(next).prev = prev;
            result.to_used_index()
        };
        unsafe { Handle::unchecked(index) }
    }

    fn push_front_free(&mut self, h: Handle<T>) -> T {
        let next = self.heads[1].next;
        let head = h.index().to_free_index();
        self.link_mut(next).prev = head;
        self.heads[1].next = head;

        let node = self.node_mut(h);
        node.link = Link { prev: Index(1), next };
        unsafe { node.data.assume_init_read() }
    }

    #[cfg(test)]
    fn check_invariants(&self) -> bool {
        if self.is_empty() {
            assert!(self.len() == 0);
        } else {
            assert!(self.len() > 0);
            let (b, bh) = (self.back().unwrap(), self.back_handle().unwrap());
            let (f, fh) = (self.front().unwrap(), self.front_handle().unwrap());
            assert!(b as *const T == &self[bh] as *const T);
            assert!(f as *const T == &self[fh] as *const T);
        }

        let mut used = 0;
        let mut total = 0;
        for (i, _) in self.heads.iter().enumerate() {
            let mut source = Index(i as u32);
            let target = source;
            loop {
                let next = self.link(source).next;
                assert!(next.0 & 1 == target.0);
                assert!(self.link(next).prev == source);
                if next == target { break; }
                source = next;
                if i == 0 { used += 1; }
                total += 1;
            }
        }
        assert!(used == self.len());
        assert!(total == self.nodes.len());
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    extern crate test;

    #[test]
    fn test_basic() {
        let mut list = List::default();
        assert!(linearize(&list) == vec![]);

        let i0 = list.push_back(0);
        assert!(linearize(&list) == vec![0]);

        let i1 = list.push_back(1);
        assert!(linearize(&list) == vec![0, 1]);

        let i2 = list.push_front(2);
        assert!(linearize(&list) == vec![2, 0, 1]);

        let i3 = list.push_back(3);
        assert!(linearize(&list) == vec![2, 0, 1, 3]);

        let i4 = list.push_front(4);
        assert!(linearize(&list) == vec![4, 2, 0, 1, 3]);

        assert!(list[i0] == 0);
        assert!(list[i1] == 1);
        assert!(list[i2] == 2);
        assert!(list[i3] == 3);
        assert!(list[i4] == 4);

        assert!(list.remove(i2) == 2);
        assert!(linearize(&list) == vec![4, 0, 1, 3]);

        assert!(list.remove(i3) == 3);
        assert!(linearize(&list) == vec![4, 0, 1]);
    }

    #[test]
    fn test_remove() {
        let mut list: List<Box<_>> = List::default();
        let _ = list.push_back(0.into());
        let _ = list.push_back(1.into());
        let i = list.push_front(2.into());
        let j = list.push_back(3.into());
        let _ = list.push_front(4.into());
        assert!(linearize_and_deref(&list) == vec![4, 2, 0, 1, 3]);

        list.remove(i);
        list.remove(j);
        assert!(linearize_and_deref(&list) == vec![4, 0, 1]);
    }

    #[test]
    fn test_pop_back() {
        let mut list = List::default();
        list.push_back(0);
        list.push_front(1);
        list.push_back(2);
        list.push_front(3);
        assert!(linearize(&list) == vec![3, 1, 0, 2]);

        assert!(list.pop_back() == Some(2));
        assert!(linearize(&list) == vec![3, 1, 0]);

        assert!(list.pop_back() == Some(0));
        assert!(linearize(&list) == vec![3, 1]);

        assert!(list.pop_back() == Some(1));
        assert!(linearize(&list) == vec![3]);

        assert!(list.pop_back() == Some(3));
        assert!(linearize(&list) == vec![]);

        assert!(list.pop_back() == None);
        assert!(linearize(&list) == vec![]);
    }

    #[test]
    fn test_pop_front() {
        let mut list = List::default();
        list.push_back(0);
        list.push_front(1);
        list.push_back(2);
        list.push_front(3);
        assert!(linearize(&list) == vec![3, 1, 0, 2]);

        assert!(list.pop_front() == Some(3));
        assert!(linearize(&list) == vec![1, 0, 2]);

        assert!(list.pop_front() == Some(1));
        assert!(linearize(&list) == vec![0, 2]);

        assert!(list.pop_front() == Some(0));
        assert!(linearize(&list) == vec![2]);

        assert!(list.pop_front() == Some(2));
        assert!(linearize(&list) == vec![]);

        assert!(list.pop_front() == None);
        assert!(linearize(&list) == vec![]);
    }

    #[test]
    fn test_move_to_back() {
        let mut list = List::default();
        let h0 = list.push_back(0);
        let _1 = list.push_front(1);
        let _2 = list.push_back(2);
        let h3 = list.push_front(3);
        assert!(linearize(&list) == vec![3, 1, 0, 2]);

        list.move_to_back(h3);
        assert!(linearize(&list) == vec![1, 0, 2, 3]);

        list.move_to_back(h0);
        assert!(linearize(&list) == vec![1, 2, 3, 0]);

        list.move_to_back(h3);
        assert!(linearize(&list) == vec![1, 2, 0, 3]);

        list.move_to_back(h3);
        assert!(linearize(&list) == vec![1, 2, 0, 3]);
    }

    #[test]
    fn test_move_to_front() {
        let mut list = List::default();
        let h0 = list.push_back(0);
        let h1 = list.push_front(1);
        let h2 = list.push_back(2);
        let h3 = list.push_front(3);
        assert!(linearize(&list) == vec![3, 1, 0, 2]);

        list.move_to_front(h3);
        assert!(linearize(&list) == vec![3, 1, 0, 2]);

        list.move_to_front(h1);
        assert!(linearize(&list) == vec![1, 3, 0, 2]);

        list.move_to_front(h0);
        assert!(linearize(&list) == vec![0, 1, 3, 2]);

        list.move_to_front(h2);
        assert!(linearize(&list) == vec![2, 0, 1, 3]);
    }

    fn linearize<T: Clone + Eq>(list: &List<T>) -> Vec<T> {
        assert!(list.check_invariants());
        let forwards: Vec<_> = list.iter().map(|x| x.clone()).collect();
        let backward: Vec<_> = list.iter().rev().map(|x| x.clone()).collect();
        assert!(list.back() == backward.iter().next());
        assert!(list.front() == forwards.iter().next());
        assert!(forwards == backward.into_iter().rev().collect::<Vec<_>>());
        forwards
    }

    fn linearize_and_deref<T: Clone + Eq>(list: &List<Box<T>>) -> Vec<T> {
        linearize(list).into_iter().map(|x| *x).collect()
    }
}
