use std::cell::RefCell;
use std::cmp::{max, min};
use std::num::NonZeroI32;
use std::ops::{Index, IndexMut};

use crate::static_assert_size;
use crate::base::{HashMap, LOS, Matrix, Point, dirs};

//////////////////////////////////////////////////////////////////////////////

// BFS (breadth-first search)

#[derive(Clone, Copy, Eq, PartialEq)]
pub enum Status { Free, Blocked, Occupied, Unknown }

#[derive(Default)]
pub struct BFSResult {
    pub dirs: Vec<Point>,
    pub targets: Vec<Point>,
}

#[allow(non_snake_case)]
pub fn BFS<F: Fn(Point) -> bool, G: Fn(Point) -> Status>(
        source: Point, target: F, limit: i32, check: G) -> Option<BFSResult> {
    let kUnknown = -1;
    let kBlocked = -2;

    let n = 2 * limit + 1;
    let initial = Point(limit, limit);
    let offset = source - initial;
    let mut distances = Matrix::new(Point(n, n), kUnknown);
    distances.set(initial, 0);

    let mut i = 1;
    let mut prev: Vec<Point> = vec![initial];
    let mut next: Vec<Point> = vec![];
    let mut targets: Vec<Point> = vec![];

    while i <= limit {
        for pp in &prev {
            for dir in &dirs::ALL {
                let np = *pp + *dir;
                let distance = distances.get(np);
                if distance != kUnknown { continue; }

                let point = np + offset;
                let free = check(point) == Status::Free;
                let done = free && target(point);

                distances.set(np, if free { i } else { kBlocked });
                if done { targets.push(np); }
                if free { next.push(np); }
            }
        }
        if next.is_empty() || !targets.is_empty() { break; }
        std::mem::swap(&mut next, &mut prev);
        next.clear();
        i += 1;
    }

    if targets.is_empty() { return None; }

    let mut result = BFSResult { dirs: vec![], targets: vec![] };
    result.targets = targets.iter().map(|x| *x + offset).collect();
    prev = targets;
    next.clear();
    i -= 1;

    while i > 0 {
        for pp in &prev {
            for dir in &dirs::ALL {
                let np = *pp + *dir;
                let distance = distances.get(np);
                if distance != i { continue; }

                distances.set(np, kUnknown);
                next.push(np);
            }
        }
        std::mem::swap(&mut next, &mut prev);
        next.clear();
        i -= 1;
    }

    assert!(!prev.is_empty());
    result.dirs = prev.iter().map(|x| *x - initial).collect();
    Some(result)
}

//////////////////////////////////////////////////////////////////////////////

// Heap, used for Dijkstra and A*

#[derive(Clone, Copy, Eq, PartialEq)] struct AStarHeapIndex(i32);
#[derive(Clone, Copy, Eq, PartialEq)] struct AStarNodeIndex(i32);

const NOT_IN_HEAP: AStarHeapIndex = AStarHeapIndex(-1);
const SOURCE_NODE: AStarNodeIndex = AStarNodeIndex(-1);

#[derive(Default)]
struct AStarState {
    heap: AStarHeap,
    map: HashMap<Point, AStarNodeIndex>,
}

struct AStarNode {
    distance: i32,
    index: AStarHeapIndex,
    parent: AStarNodeIndex,
    pos: Point,
    score: i32,
}

impl AStarNode {
    fn new(pos: Point, parent: AStarNodeIndex, distance: i32, score: i32) -> Self {
        Self { distance, index: NOT_IN_HEAP, parent, pos, score }
    }
}

#[derive(Default)]
struct AStarHeap {
    nodes: Vec<AStarNode>,
    heap: Vec<AStarNodeIndex>,
}

impl AStarHeap {
    // Heap operations

    fn is_empty(&self) -> bool { self.heap.is_empty() }

    fn extract_min(&mut self) -> AStarNodeIndex {
        let mut index = AStarHeapIndex(0);
        let result = self.get_heap(index);
        self.mut_node(result).index = NOT_IN_HEAP;

        let node = self.heap.pop().unwrap();
        if self.is_empty() { return result; }

        let limit = self.heap.len() as i32;
        let score = self.get_node(node).score;
        let (mut c0, mut c1) = Self::children(index);

        while c0.0 < limit {
            let mut child_index = c0;
            let mut child_score = self.heap_score(c0);
            if c1.0 < limit {
                let c1_score = self.heap_score(c1);
                if c1_score < child_score {
                    (child_index, child_score) = (c1, c1_score);
                }
            }
            if score <= child_score { break; }

            self.heap_move(child_index, index);
            (c0, c1) = Self::children(child_index);
            index = child_index;
        }

        self.mut_node(node).index = index;
        self.set_heap(index, node);
        result
    }

    fn heapify(&mut self, n: AStarNodeIndex) {
        let score = self.get_node(n).score;
        let mut index = self.get_node(n).index;

        while index.0 > 0 {
            let parent_index = Self::parent(index);
            let parent_score = self.heap_score(parent_index);
            if parent_score <= score { break; }

            self.heap_move(parent_index, index);
            index = parent_index;
        }

        self.mut_node(n).index = index;
        self.set_heap(index, n);
    }

    fn push(&mut self, mut node: AStarNode) -> AStarNodeIndex {
        assert!(node.index.0 == -1);
        node.index = AStarHeapIndex(self.heap.len() as i32);
        let result = AStarNodeIndex(self.nodes.len() as i32);
        self.nodes.push(node);
        self.heap.push(result);
        self.heapify(result);
        result
    }

    // Lower-level helpers

    fn heap_score(&self, h: AStarHeapIndex) -> i32 {
        self.get_node(self.get_heap(h)).score
    }

    fn heap_move(&mut self, from: AStarHeapIndex, to: AStarHeapIndex) {
        let node = self.get_heap(from);
        self.mut_node(node).index = to;
        self.set_heap(to, node);
    }

    fn get_heap(&self, h: AStarHeapIndex) -> AStarNodeIndex {
        self.heap[h.0 as usize]
    }

    fn set_heap(&mut self, h: AStarHeapIndex, n: AStarNodeIndex) {
        self.heap[h.0 as usize] = n;
    }

    fn get_node(&self, n: AStarNodeIndex) -> &AStarNode {
        &self.nodes[n.0 as usize]
    }

    fn mut_node(&mut self, n: AStarNodeIndex) -> &mut AStarNode {
        &mut self.nodes[n.0 as usize]
    }

    fn parent(h: AStarHeapIndex) -> AStarHeapIndex {
        AStarHeapIndex((h.0 - 1) / 2)
    }

    fn children(h: AStarHeapIndex) -> (AStarHeapIndex, AStarHeapIndex) {
        (AStarHeapIndex(2 * h.0 + 1), AStarHeapIndex(2 * h.0 + 2))
    }
}

//////////////////////////////////////////////////////////////////////////////

// A* for pathfinding to a known target

const ASTAR_UNIT_COST: i32 = 16;
const ASTAR_DIAGONAL_PENALTY: i32 = 6;
const ASTAR_LOS_DIFF_PENALTY: i32 = 1;
const ASTAR_OCCUPIED_PENALTY: i32 = 64;

#[allow(non_snake_case)]
fn AStarLength(p: Point) -> i32 {
    let (x, y) = (p.0.abs(), p.1.abs());
    ASTAR_UNIT_COST * max(x, y) + ASTAR_DIAGONAL_PENALTY * min(x, y)
}

// "diff" penalizes paths that travel far from the direct line-of-sight
// from the source to the target. In order to compute it, we figure out if
// this line is "more horizontal" or "more vertical", then compute the the
// distance from the point to this line orthogonal to this main direction.
//
// Adding this term to our heuristic means that it's no longer admissible,
// but it provides two benefits that are enough for us to use it anyway:
//
//   1. By breaking score ties, we expand the fronter towards T faster than
//      we would with a consistent heuristic. We complete the search sooner
//      at the cost of not always finding an optimal path.
//
//   2. By biasing towards line-of-sight, we select paths that are visually
//      more appealing than alternatives (e.g. that interleave cardinal and
//      diagonal steps, rather than doing all the diagonal steps first).
//
#[allow(non_snake_case)]
fn AStarHeuristic(p: Point, los: &Vec<Point>) -> i32 {
    let Point(px, py) = p;
    let Point(sx, sy) = los[0];
    let Point(tx, ty) = *los.last().unwrap();

    let diff = (|| {
        let dx = tx - sx;
        let dy = ty - sy;
        let l = (los.len() - 1) as i32;
        if dx.abs() > dy.abs() {
            let index = if dx > 0 { px - sx } else { sx - px };
            if index < 0 { return (px - sx).abs() + (py - sy).abs() };
            if index > l { return (px - tx).abs() + (py - ty).abs() };
            (py - los[index as usize].1).abs()
        } else {
            let index = if dy > 0 { py - sy } else { sy - py };
            if index < 0 { return (px - sx).abs() + (py - sy).abs(); }
            if index > l { return (px - tx).abs() + (py - ty).abs(); }
            (px - los[index as usize].0).abs()
        }
    })();

   ASTAR_LOS_DIFF_PENALTY * diff + AStarLength(p - Point(tx, ty))
}

#[allow(non_snake_case)]
pub fn AStar<F: Fn(Point) -> Status>(
        source: Point, target: Point, limit: i32, check: F) -> Option<Vec<Point>> {
    // Try line-of-sight - if that path is clear, then we don't need to search.
    // As with the full search below, we don't check if source is blocked here.
    let los = LOS(source, target);
    let free = (1..los.len() - 1).all(|i| check(los[i]) == Status::Free);
    if free { return Some(los.into_iter().skip(1).collect()) }

    Dijkstra(source, |x| x == target, limit, check, |x| AStarHeuristic(x, &los))
}

//////////////////////////////////////////////////////////////////////////////

// Dijkstra

// TODO: This search algorithm is non-isotropic. It prefers to move northwest.
// Fix it by sampling all nodes at `score` matching `target`.
//
// TODO: If it's AStar, and we haven't found a target, return a path that gets
// us as close as possible to the target.
#[allow(non_snake_case)]
pub fn Dijkstra<F: Fn(Point) -> bool, G: Fn(Point) -> Status, H: Fn(Point) -> i32>(
        source: Point, target: F, limit: i32, check: G, heuristic: H) -> Option<Vec<Point>> {
    CACHE.with_borrow_mut(|cache|{
        // Reuse AStar allocations via the cached AStarState.
        let state = &mut cache.0;
        let result = CachedDijkstra(state, source, target, limit, check, heuristic);

        // Clean up the updates done to the cache state.
        state.heap.heap.clear();
        state.heap.nodes.clear();
        state.map.clear();

        result
    })
}

#[allow(non_snake_case)]
fn CachedDijkstra<F: Fn(Point) -> bool, G: Fn(Point) -> Status, H: Fn(Point) -> i32>(
        state: &mut AStarState, source: Point, target: F,
        limit: i32, check: G, heuristic: H) -> Option<Vec<Point>> {
    let map = &mut state.map;
    let heap = &mut state.heap;

    let score = heuristic(source);
    let node = AStarNode::new(source, SOURCE_NODE, 0, score);
    map.insert(source, heap.push(node));

    for _ in 0..limit {
        if heap.is_empty() { break; }
        let prev = heap.extract_min();
        let prev_pos = heap.get_node(prev).pos;
        let prev_distance = heap.get_node(prev).distance;
        if target(prev_pos) {
            let mut result = vec![];
            let mut current = heap.get_node(prev);
            while current.pos != source {
                result.push(current.pos);
                current = heap.get_node(current.parent);
            }
            result.reverse();
            return Some(result);
        }

        for dir in &dirs::ALL {
            let next = prev_pos + *dir;
            let status = if target(next) { Status::Free } else { check(next) };
            if status == Status::Blocked { continue; }

            let occupied = status == Status::Occupied;
            let diagonal = dir.0 != 0 && dir.1 != 0;
            let distance = prev_distance + ASTAR_UNIT_COST +
                           if diagonal { ASTAR_DIAGONAL_PENALTY } else { 0 } +
                           if occupied { ASTAR_OCCUPIED_PENALTY } else { 0 };

            map.entry(next).and_modify(|x| {
                // index != NOT_IN_HEAP checks if we've already extracted next
                // from heap. We need it since our heuristic is inadmissible.
                //
                // Using such a heuristic speeds up search in easy cases, with
                // the downside that we don't always find an optimal path.
                let existing = heap.mut_node(*x);
                if existing.index != NOT_IN_HEAP && existing.distance > distance {
                    existing.score += distance - existing.distance;
                    existing.distance = distance;
                    existing.parent = prev;
                    heap.heapify(*x);
                }
            }).or_insert_with(|| {
                let score = distance + heuristic(next);
                let node = AStarNode::new(next, prev, distance, score);
                heap.push(node)
            });
        }
    }
    None
}

//////////////////////////////////////////////////////////////////////////////

// DijkstraMap

const DIJKSTRA_COST: i32 = 5;
const DIJKSTRA_DIAGONAL_PENALTY: i32 = 2;
const DIJKSTRA_OCCUPIED_PENALTY: i32 = 20;

#[derive(Clone, Copy, Eq, PartialEq)]
struct DijkstraNodeIndex(NonZeroI32);
static_assert_size!(Option<DijkstraNodeIndex>, 4);

#[derive(Clone, Default)]
struct DijkstraLink {
    next: Option<DijkstraNodeIndex>,
    prev: Option<DijkstraNodeIndex>,
}

#[derive(Clone, Default)]
struct DijkstraNode {
    link: DijkstraLink,
    point: Point,
    score: i32,
    status: Option<Status>,
}

#[derive(Default)]
struct DijkstraState {
    dirty: Vec<DijkstraNodeIndex>,
    lists: Vec<DijkstraLink>,
    nodes: Vec<DijkstraNode>,
}

impl DijkstraState {
    fn link(&mut self, index: Option<DijkstraNodeIndex>, score: i32) -> &mut DijkstraLink {
        if let Some(x) = index { return &mut self.nodes[x].link; }
        &mut self.lists[score as usize]
    }
}

impl Index<DijkstraNodeIndex> for Vec<DijkstraNode> {
    type Output = DijkstraNode;
    fn index(&self, index: DijkstraNodeIndex) -> &Self::Output {
        &self[index.0.get() as usize - 1]
    }
}

impl IndexMut<DijkstraNodeIndex> for Vec<DijkstraNode> {
    fn index_mut(&mut self, index: DijkstraNodeIndex) -> &mut Self::Output {
        &mut self[index.0.get() as usize - 1]
    }
}

thread_local! {
    static CACHE: RefCell<(AStarState, DijkstraState)> = Default::default();
}

#[derive(Default)]
pub struct Neighborhood {
    pub blocked: Vec<(Point, i32)>,
    pub visited: Vec<(Point, i32)>,
}

// Expose a distance function for use in other heuristics.
#[allow(non_snake_case)]
pub fn DijkstraLength(p: Point) -> i32 {
    let (x, y) = (p.0.abs(), p.1.abs());
    DIJKSTRA_COST * max(x, y) + DIJKSTRA_DIAGONAL_PENALTY * min(x, y)
}

#[allow(non_snake_case)]
pub fn DijkstraMap<F: Fn(Point) -> Status>(
        source: Point, check: F, cells: i32, limit: i32) -> Neighborhood {
    CACHE.with_borrow_mut(|cache|{
        // Make sure the Matrix has enough space for the search.
        let state = &mut cache.1;
        let n = ((2 * limit + 1) as usize).pow(2);
        if state.nodes.len() < n { state.nodes.resize_with(n, Default::default); }
        let result = CachedDijkstraMap(state, source, check, cells, limit);

        // Restore the cached state to a clean condition.
        let state = &mut cache.1;
        for &p in &state.dirty { state.nodes[p] = Default::default(); }
        state.dirty.clear();
        state.lists.clear();

        result
    })
}

#[allow(non_snake_case)]
fn CachedDijkstraMap<F: Fn(Point) -> Status>(
        state: &mut DijkstraState, source: Point,
        check: F, cells: i32, limit: i32) -> Neighborhood {
    let cells = cells as usize;
    let mut result = Neighborhood::default();
    result.blocked.reserve(cells);
    result.visited.reserve(cells);

    let initial = Point(limit, limit);
    let offset = source - initial;
    let extent = 2 * limit + 1;

    // We only search points within an L1 distance of `limit` from `source`.
    let get_index = |Point(x, y): Point| {
        if !(0 <= x && x < extent && 0 <= y && y < extent) { return None; }
        Some(DijkstraNodeIndex(unsafe { NonZeroI32::new_unchecked(x + y * extent + 1) }))
    };

    // Add the node at `index` to the tail of the list of nodes at `score`,
    // and set its point and status (both of which may already be set).
    let init = |state: &mut DijkstraState,
                index: DijkstraNodeIndex, point: Point, score: i32, status: Status| {
        while state.lists.len() <= score as usize {
            state.lists.push(DijkstraLink::default())
        }

        let head = &mut state.lists[score as usize];
        let prev = head.prev;
        head.prev = Some(index);
        let tail = state.link(prev, score);
        tail.next = Some(index);

        let entry = &mut state.nodes[index];
        entry.link.prev = prev;
        entry.link.next = None;
        entry.point = point;
        entry.score = score;
        entry.status = Some(status);
    };

    // Relax the edge from `prev_point` (at `prev_score`) to `prev_point + dir`.
    let step = |state: &mut DijkstraState,
                dir: Point, prev_point: Point, prev_score: i32| {
        let point = prev_point + dir;
        let Some(index) = get_index(point) else { return };

        let entry = &mut state.nodes[index];
        let visited = entry.status.is_some();
        let status = entry.status.unwrap_or_else(|| check(point + offset));

        entry.status = Some(status);
        if !visited { state.dirty.push(index); }

        let diagonal = dir.0 != 0 && dir.1 != 0;
        let occupied = status == Status::Occupied;
        let score = prev_score + DIJKSTRA_COST +
                    if diagonal { DIJKSTRA_DIAGONAL_PENALTY } else { 0 } +
                    if occupied { DIJKSTRA_OCCUPIED_PENALTY } else { 0 };
        if visited && score >= entry.score { return; }

        if visited {
            let score = entry.score;
            let DijkstraLink { next, prev } = entry.link;
            state.link(next, score).prev = prev;
            state.link(prev, score).next = next;
        }
        init(state, index, point, score, status);
    };

    let index = get_index(initial).unwrap();
    init(state, index, initial, 0, Status::Free);

    let mut current = 0;
    loop {
        let lists = &state.lists;
        while current < lists.len() && lists[current].next.is_none() { current += 1; }
        if current == lists.len() { break; }

        // Remove the entry at the head of the selected list.
        let head = &mut state.lists[current];
        let prev = head.next.unwrap();
        let next = state.nodes[prev].link.next;
        head.next = next;
        let next = state.link(next, current as i32);
        next.prev = None;

        let node = &state.nodes[prev];
        let DijkstraNode { point, score, .. } = *node;
        let value = (point + offset, score);

        let status = node.status.unwrap();
        if status == Status::Blocked {
            result.blocked.push(value);
        } else {
            result.visited.push(value);
            if result.visited.len() >= (cells as usize) { break; }
        }

        let expand = match status {
            Status::Free | Status::Occupied  => true,
            Status::Blocked | Status::Unknown => false,
        };
        if expand {
            for &dir in &dirs::ALL {
                step(state, dir, point, score);
            }
        }
    }
    result
}

//////////////////////////////////////////////////////////////////////////////

#[allow(soft_unstable)]
#[cfg(test)]
mod tests {
    use super::*;
    use rand::{Rng, SeedableRng};
    use crate::base::RNG;
    extern crate test;

    const BFS_LIMIT: i32 = 32;
    const DIJKSTRA_CELLS: i32 = 1024;
    const DIJKSTRA_LIMIT: i32 = 64;

    #[bench]
    fn bench_bfs(b: &mut test::Bencher) {
        let map = generate_map(2 * BFS_LIMIT);
        b.iter(|| {
            let done = |_: Point| { false };
            let check = |p: Point| { map.get(&p).copied().unwrap_or(Status::Free) };
            BFS(Point::default(), done, BFS_LIMIT, check);
        });
    }

    #[bench]
    fn bench_dijkstra(b: &mut test::Bencher) {
        let map = generate_map(2 * BFS_LIMIT);
        b.iter(|| {
            let done = |_: Point| { false };
            let check = |p: Point| { map.get(&p).copied().unwrap_or(Status::Free) };
            Dijkstra(Point::default(), done, DIJKSTRA_CELLS, check, |_| 0);
        });
    }

    #[bench]
    fn bench_dijkstra_map(b: &mut test::Bencher) {
        let map = generate_map(2 * BFS_LIMIT);
        b.iter(|| {
            let mut result = HashMap::default();
            result.insert(Point::default(), 0);
            let check = |p: Point| { map.get(&p).copied().unwrap_or(Status::Free) };
            DijkstraMap(Point::default(), check, DIJKSTRA_CELLS, DIJKSTRA_LIMIT);
        });
    }

    fn generate_map(n: i32) -> HashMap<Point, Status> {
        let mut result = HashMap::default();
        let mut rng = RNG::seed_from_u64(17);
        for x in -n..n + 1 {
            for y in -n..n + 1 {
                let f = rng.random::<i32>().rem_euclid(100);
                let s = if f < 20 { Status::Blocked } else { Status::Free };
                result.insert(Point(x, y), s);
            }
        }
        result
    }
}
