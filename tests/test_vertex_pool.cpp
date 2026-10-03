// VertexPointerManager under random sequences of line additions, exchanges and removals.
//
// The operations are the ones the updates perform: addVertexPointers (add_*), exchangeLines plus the
// diagram-level conj_vertex rewiring (swp_ph), and removal through findPointer + .conjugated
// (rm_*). After every step the manager must satisfy all its invariants:
//   - used slots are exactly [0, current_length), with position == slot index == vertex->index;
//   - .conjugated is mutual, and every line sits in one aligned slot pair (2m, 2m+1);
//   - .conjugated agrees with Vertex::conj_vertex;
//   - the registered vertices are exactly the live ones (no vertex lost, none removed but still there).
// Includes the specific sequence that corrupted the pool before exchangeLines existed: three lines,
// an exchange between the second and third, then removal of a line that no longer sits in the last pair.
#include <cstdio>
#include <random>
#include <set>
#include <string>
#include <vector>
#include "test_common.hpp"

static std::string violation(const VertexPointerManager & m, const std::set<Vertex *> & live){
    std::set<Vertex *> registered;
    for (int i {0}; i < m.current_length; ++i) {
        const VertexPointer & s {m.ptr_vertex_pool[i]};
        if (!s.used || s.linked_vertex == nullptr) { return "slot " + std::to_string(i) + " empty inside [0, current_length)"; }
        if (s.position != i || s.linked_vertex->index != i) { return "slot " + std::to_string(i) + " position/index mismatch"; }
        if (s.conjugated == nullptr || s.conjugated->conjugated != &s) { return "slot " + std::to_string(i) + " conjugated not mutual"; }
        if (s.conjugated - m.ptr_vertex_pool != (i ^ 1)) { return "slot " + std::to_string(i) + " not in an aligned pair"; }
        if (s.conjugated->linked_vertex != s.linked_vertex->conj_vertex) { return "slot " + std::to_string(i) + " disagrees with conj_vertex"; }
        registered.insert(s.linked_vertex);
    }
    for (int i {m.current_length}; i < m.max_length; ++i) {
        if (m.ptr_vertex_pool[i].used) { return "slot " + std::to_string(i) + " used beyond current_length"; }
    }
    if (registered != live) { return "registered vertices differ from the live ones"; }
    return "";
}

// what swp_ph::accept() does to the pool: diagram-level conj_vertex rewiring, then the manager's relink
static void exchange(VertexPointerManager & m, Vertex * p1, Vertex * p2){
    Vertex * d1 {p1->conj_vertex};
    Vertex * d2 {p2->conj_vertex};
    p1->conj_vertex = d2; p2->conj_vertex = d1; d1->conj_vertex = p2; d2->conj_vertex = p1;
    m.exchangeLines(p1, p2);
}

// what rm_*::accept() does to the pool
static void remove_line(VertexPointerManager & m, Vertex * v, std::set<Vertex *> & live){
    VertexPointer * a {m.findPointer(v)};
    VertexPointer * b {a->conjugated};
    live.erase(a->linked_vertex);
    live.erase(b->linked_vertex);
    m.removeVertexPointers(*a, *b);
}

int main(){
    test::Checks check {"VertexPointerManager invariants under add / exchange / remove"};

    {   // the sequence that used to corrupt the pool
        simplemc::xoshiro256ss rng {1};
        VertexPointerManager m {6, &rng};
        std::vector<Vertex> v(6);
        std::set<Vertex *> live;
        for (int l {0}; l < 3; ++l) {
            v[2*l].conj_vertex = &v[2*l + 1]; v[2*l + 1].conj_vertex = &v[2*l];
            m.addVertexPointers(&v[2*l], &v[2*l + 1]);
            live.insert(&v[2*l]); live.insert(&v[2*l + 1]);
        }
        exchange(m, &v[3], &v[4]);            // lines become (v0,v1) (v2,v4) (v3,v5)
        const std::string after_exchange {violation(m, live)};
        remove_line(m, &v[3], live);          // removes (v3,v5)
        const std::string after_remove {violation(m, live)};
        check(after_exchange.empty() && after_remove.empty(),
              "3 lines, exchange between lines 2 and 3, remove line (v3,v5): %s",
              after_exchange.empty() && after_remove.empty() ? "all invariants hold"
              : (after_exchange.empty() ? after_remove : after_exchange).c_str());
    }

    const int RUNS {2000}, STEPS {300};
    long broken_runs {0}, adds {0}, exchanges {0}, removes {0};
    std::string first;
    for (int run {0}; run < RUNS; ++run) {
        std::mt19937 g {static_cast<unsigned>(run)};
        simplemc::xoshiro256ss rng {static_cast<std::uint64_t>(run) + 1};
        VertexPointerManager m {40, &rng};
        std::vector<Vertex> store(2*STEPS + 2);
        std::size_t next {0};
        std::set<Vertex *> live;
        for (int step {0}; step < STEPS; ++step) {
            const int op {static_cast<int>(g() % 3)};
            if (op == 0 && m.current_length + 2 <= m.max_length) {
                Vertex * a {&store[next++]}; Vertex * b {&store[next++]};
                a->conj_vertex = b; b->conj_vertex = a;
                m.addVertexPointers(a, b);
                live.insert(a); live.insert(b);
                ++adds;
            } else if (op == 1 && m.current_length >= 4) {
                Vertex * p1 {m.ptr_vertex_pool[g() % static_cast<unsigned>(m.current_length)].linked_vertex};
                Vertex * p2 {m.ptr_vertex_pool[g() % static_cast<unsigned>(m.current_length)].linked_vertex};
                if (p1 != p2 && p1->conj_vertex != p2) { exchange(m, p1, p2); ++exchanges; }
            } else if (op == 2 && m.current_length >= 2) {
                remove_line(m, m.ptr_vertex_pool[g() % static_cast<unsigned>(m.current_length)].linked_vertex, live);
                ++removes;
            }
            const std::string bad {violation(m, live)};
            if (!bad.empty()) {
                if (first.empty()) { first = "run " + std::to_string(run) + ", step " + std::to_string(step) + ": " + bad; }
                ++broken_runs;
                break;
            }
        }
    }
    check(broken_runs == 0, "%d random runs x %d steps (%ld adds, %ld exchanges, %ld removals): %ld broken%s%s",
          RUNS, STEPS, adds, exchanges, removes, broken_runs, first.empty() ? "" : " - first: ", first.c_str());
    check(exchanges > 100000 && removes > 100000, "fuzz actually exercised exchanges and removals");
    return check.exit_code();
}
