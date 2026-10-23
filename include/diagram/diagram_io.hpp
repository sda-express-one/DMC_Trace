#ifndef DIAGRAM_IO_HPP
#define DIAGRAM_IO_HPP

#include <array>
#include <vector>
#include <simplemc/serialize/concepts.hpp>
#include "diagram/diagram_config.hpp"
#include "utils/diagram_sanitizer.hpp"

// The state that defines a diagram, for checkpoint/restart: its length, its phonon lines in the managers' slot
// order (line m = slots 2m, 2m+1: the slot order decides which line an update picks, so it is kept) and every
// segment's electron momentum verbatim, head to tail. Everything else on the vertices (bases, masses, overlaps,
// actions, phonon propagators, the cached products, the sign) is a function of these and is rebuilt by the sanitizer
// on restore. The rebuilt caches can differ from the running chain's in the last bits (the updates copy some of them
// from neighbours instead of recomputing): a restart is statistically, not bitwise, the same chain.
struct LineState {
    double t_cre {0.};
    double t_ann {0.};
    std::array<double, 3> w {0., 0., 0.};
    int mode {0};               // index into the PhononModeManager pool
};

struct DiagramState {
    double tau_D {0.};
    std::vector<LineState> internal;
    std::vector<LineState> external;
    std::vector<std::array<double, 3>> k;      // segment momenta, head first
};

// Throws std::runtime_error if a line's (ph_energy, diel_response) matches no phonon mode of the pool.
DiagramState captureDiagram(const diagram_cfg & cfg);

// Rebuilds the diagram in cfg (whose managers and mode pool must be those of the run: capacities, modes) from state:
// reset, the lines in slot order, the saved momenta, then diagram_builder::finalize. Throws std::runtime_error if a
// line cannot be placed or the number of segments does not match; the caller checks the returned report's clean().
numerical::SanitizeReport restoreDiagram(diagram_cfg & cfg, const DiagramState & state);

// simplemc serialization hooks (checkpoint channel): a line is stored as [t_cre, t_ann, wx, wy, wz, mode], so any
// serializer that handles std::vector<std::array<double, N>> can hold it; doubles are exact in binary formats.
template <simplemc::serializer S>
void simplemc_save(S s, const DiagramState & d){
    const auto pack {[](const std::vector<LineState> & lines){
        std::vector<std::array<double, 6>> out;
        out.reserve(lines.size());
        for (const LineState & l : lines) { out.push_back({l.t_cre, l.t_ann, l.w[0], l.w[1], l.w[2], static_cast<double>(l.mode)}); }
        return out;
    }};
    s.save_at("tau_D", d.tau_D);
    s.save_at("internal", pack(d.internal));
    s.save_at("external", pack(d.external));
    s.save_at("k", d.k);
}

template <simplemc::serializer S>
void simplemc_load(const S & s, DiagramState & d){
    const auto unpack {[](const std::vector<std::array<double, 6>> & in){
        std::vector<LineState> lines;
        lines.reserve(in.size());
        for (const auto & a : in) { lines.push_back({a[0], a[1], {a[2], a[3], a[4]}, static_cast<int>(a[5])}); }
        return lines;
    }};
    std::vector<std::array<double, 6>> internal, external;
    s.load_at("tau_D", d.tau_D);
    s.load_at("internal", internal);
    s.load_at("external", external);
    s.load_at("k", d.k);
    d.internal = unpack(internal);
    d.external = unpack(external);
}

#endif // !DIAGRAM_IO_HPP
