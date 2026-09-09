#include "../include/vertex_manager.hpp"
#include <simplemc/random/xoshiro256.hpp>

VertexPointerManager::VertexPointerManager(int max_length, simplemc::xoshiro256ss * rng) :
    max_length(max_length + (max_length % 2)),
    rng(rng)
{
    assert(rng != nullptr);
    // single fixed-size allocation: every slot's address is stable for the lifetime of
    // this object, so .conjugated pointers into it never dangle or get invalidated.
    ptr_vertex_pool = new VertexPointer[this->max_length];
}

VertexPointerManager::~VertexPointerManager(){
    delete[] ptr_vertex_pool;
}

VertexPointerManager::VertexPointerManager(VertexPointerManager&& other) noexcept :
    max_length(other.max_length),
    current_length(other.current_length),
    ptr_vertex_pool(other.ptr_vertex_pool),
    rng(other.rng)
{
    other.ptr_vertex_pool = nullptr;
    other.current_length = 0;
    other.rng = nullptr;
}
