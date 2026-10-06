#ifndef VERTEX_MANAGER_HPP
#define VERTEX_MANAGER_HPP

#include <random>
#include <cassert>
#include <simplemc/random/xoshiro256.hpp>
#include "diagram/vertex.hpp"

struct VertexPointer {
    Vertex * linked_vertex {nullptr};
    VertexPointer * conjugated {nullptr};
    int position {-1};
    bool used {false};
};

// Layout invariant: the used slots are [0, current_length), and every line occupies one ALIGNED
// pair of slots (2m, 2m+1), the two being each other's .conjugated. addVertexPointers creates it,
// removeVertexPointers relies on it (it fills a hole by moving the last pair down as a unit), and
// exchangeLines preserves it. Code outside this struct must not write .conjugated or
// .linked_vertex directly - relink lines through exchangeLines instead.
//
// Within a pair, slot 2m holds the line's creation vertex (type > 0: +1 internal, +2 external) and
// slot 2m+1 its annihilation vertex (type < 0). addVertexPointers takes them in that order,
// removeVertexPointers moves pairs as a unit, and exchangeLines expects the caller to have swapped
// the two vertices' types already (as swp_ph does), so each slot keeps its type. Code may therefore
// visit one vertex of each line, of a known kind, as the even (or odd) slots of [0, current_length).
struct VertexPointerManager {
    const int max_length {0};
    int current_length {0};
    VertexPointer * ptr_vertex_pool {nullptr}; // fixed-size pool, allocated once: addresses into it
                                                // (via .conjugated) must stay stable for the whole
                                                // lifetime of the manager, hence a raw array rather
                                                // than a std::vector that could reallocate.
    mutable simplemc::xoshiro256ss * rng;

    VertexPointerManager(int max_length, simplemc::xoshiro256ss * rng);
    ~VertexPointerManager();

    // ptr_vertex_pool is uniquely owned (delete[]'d in the destructor), so copying would
    // double-free; only moving it out is allowed. max_length is const, so a move-assignment
    // operator can't reinitialize it - only move-construction is meaningful here.
    VertexPointerManager(const VertexPointerManager&) = delete;
    VertexPointerManager& operator=(const VertexPointerManager&) = delete;
    VertexPointerManager& operator=(VertexPointerManager&&) = delete;

    VertexPointerManager(VertexPointerManager&& other) noexcept;

    void addVertexPointers(Vertex * vertex_one, Vertex * vertex_two){
        assert(vertex_one != nullptr);
        assert(vertex_two != nullptr);
        assert(current_length % 2 == 0);
        assert(current_length < max_length + 1);
        assert(vertex_one->type > 0 && vertex_two->type < 0); // creation first: see the layout comment

        if(current_length + 2 > max_length){
            return;
        }

        const int position_one {current_length};
        const int position_two {current_length + 1};

        ptr_vertex_pool[position_one] = VertexPointer{.linked_vertex = vertex_one, .position = position_one, .used = true};
        ptr_vertex_pool[position_two] = VertexPointer{.linked_vertex = vertex_two, .position = position_two, .used = true};

        // conjugated must point at the pool's own (stable) slots, not at the temporaries above
        ptr_vertex_pool[position_one].conjugated = &ptr_vertex_pool[position_two];
        ptr_vertex_pool[position_two].conjugated = &ptr_vertex_pool[position_one];

        // Vertex::index mirrors .position on the other side, so a bare Vertex* can find its own
        // slot in O(1) via findPointer() without needing to already hold a VertexPointer*.
        vertex_one->index = position_one;
        vertex_two->index = position_two;

        current_length += 2;
    }

    // Exchange which line vertex_a and vertex_b belong to: afterwards vertex_a is paired with
    // vertex_b's former partner and vice versa. Done by swapping the two vertices' slots, so
    // .conjugated never changes and every line stays in its aligned slot pair - unlike
    // re-pointing .conjugated, which would leave lines straddling pairs and break
    // removeVertexPointers. Which slot a vertex sits in carries no meaning for the samplers
    // (chooseAnyVertex/chooseOutgoingVertex/selectVertex are uniform over slots), so this changes
    // no proposal probability. Vertex::conj_vertex is diagram state and is the caller's to update.
    void exchangeLines(Vertex * vertex_a, Vertex * vertex_b){
        assert(vertex_a != nullptr);
        assert(vertex_b != nullptr);
        assert(vertex_a != vertex_b);

        VertexPointer * slot_a {findPointer(vertex_a)};
        VertexPointer * slot_b {findPointer(vertex_b)};
        assert(slot_a->conjugated != slot_b); // already partners: exchanging would self-pair them

        slot_a->linked_vertex = vertex_b;
        slot_b->linked_vertex = vertex_a;
        vertex_b->index = slot_a->position;
        vertex_a->index = slot_b->position;

        // the caller swaps the two vertices' types before relinking, so each slot keeps its kind
        assert((vertex_a->type > 0) == (vertex_a->index % 2 == 0));
        assert((vertex_b->type > 0) == (vertex_b->index % 2 == 0));
    }

    void removeVertexPointers(VertexPointer& pointer_one, VertexPointer& pointer_two){
        assert(current_length % 2 == 0);
        assert(current_length > -1);

        if(current_length < 2){
            return;
        }

        // the layout invariant this function depends on (see the struct comment): the line being
        // removed and the last line are each an aligned conjugate pair.
        assert(pointer_one.conjugated == &pointer_two && pointer_two.conjugated == &pointer_one);
        assert(pointer_one.position / 2 == pointer_two.position / 2);
        assert(ptr_vertex_pool[current_length - 2].conjugated == &ptr_vertex_pool[current_length - 1]);

        // fill the hole by the pair's own slots, even (creation) first, whichever of the two the
        // caller passed as pointer_one - so the last pair moves down with its order intact
        const int position_one {pointer_one.position - pointer_one.position % 2};
        const int position_two {position_one + 1};

        if(position_one != current_length - 2){
            ptr_vertex_pool[position_one].linked_vertex = ptr_vertex_pool[current_length-2].linked_vertex;
            ptr_vertex_pool[position_two].linked_vertex = ptr_vertex_pool[current_length-1].linked_vertex;

            ptr_vertex_pool[position_one].conjugated = &ptr_vertex_pool[position_two];
            ptr_vertex_pool[position_two].conjugated = &ptr_vertex_pool[position_one];

            // the two vertices just swapped into position_one/position_two need their own
            // index updated to match, same as .conjugated above.
            ptr_vertex_pool[position_one].linked_vertex->index = position_one;
            ptr_vertex_pool[position_two].linked_vertex->index = position_two;
        }

        ptr_vertex_pool[current_length - 1].linked_vertex = nullptr;
        ptr_vertex_pool[current_length - 1].conjugated = nullptr;
        ptr_vertex_pool[current_length - 1].used = false;
        ptr_vertex_pool[current_length - 1].position = -1;

        ptr_vertex_pool[current_length - 2].linked_vertex = nullptr;
        ptr_vertex_pool[current_length - 2].conjugated = nullptr;
        ptr_vertex_pool[current_length - 2].used = false;
        ptr_vertex_pool[current_length - 2].position = -1;

        current_length -= 2;
    }

    Vertex * selectVertex(int position) const {
        assert(position < current_length);
        return ptr_vertex_pool[position].linked_vertex;
    }

    // O(1) reverse lookup: given a Vertex known to be registered in *this* manager, find its
    // own VertexPointer slot via the index addVertexPointers/removeVertexPointers maintain on
    // it. If you already hold a VertexPointer* for the pair, prefer its .conjugated instead -
    // this is for when only a bare Vertex* is available (e.g. from ->conj_vertex or a diagram
    // walk).
    VertexPointer * findPointer(Vertex * vertex) const {
        assert(vertex != nullptr);
        assert(vertex->index >= 0 && vertex->index < current_length);
        assert(ptr_vertex_pool[vertex->index].linked_vertex == vertex);
        return &ptr_vertex_pool[vertex->index];
    }

    VertexPointer * chooseAnyVertex() {
        assert(current_length > 0);
        std::uniform_int_distribution<int> select {0, current_length - 1};
        return &ptr_vertex_pool[select(*rng)];
    }

    // A uniformly chosen line's creation vertex (type > 0): by the layout rule, the even slot of its
    // pair - so no rejection on the type is needed, and each line is returned with probability
    // 1/(current_length/2).
    VertexPointer * chooseOutgoingVertex() {
        assert(current_length >= 2 && current_length % 2 == 0);
        assert(current_length <= max_length);
        std::uniform_int_distribution<int> select {0, current_length/2 - 1};

        VertexPointer * slot {&ptr_vertex_pool[2*select(*rng)]};
        assert(slot->linked_vertex != nullptr && slot->linked_vertex->type > 0);

        return slot;
    }
};

#endif // !VERTEX_MANAGER_HPP
