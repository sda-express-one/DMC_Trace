#include "../include/vertex_manager.hpp"
#include <simplemc/random/xoshiro256.hpp>

VertexPointerManager::VertexPointerManager(int max_length, simplemc::xoshiro256ss * rng) : 
    max_length(max_length + (max_length % 2)), 
    rng(rng) 
{
    assert(rng != nullptr);
    ptr_vertex_vector.reserve(this->max_length);
}
