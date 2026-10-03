#ifndef TEST_COMMON_HPP
#define TEST_COMMON_HPP

// Shared helpers for the regression tests: a check/report counter and a fixture that builds an
// initialised diagram (order 0, tail at current_tau_length, cached state built by the sanitizer).

#include <array>
#include <cstdarg>
#include <cstdio>
#include <vector>
#include <simplemc/random/xoshiro256.hpp>
#include "diagram/diagram_config.hpp"
#include "diagram/phonon_manager.hpp"
#include "diagram/vertex_manager.hpp"
#include "comp_method/weight_computation.hpp"
#include "utils/diagram_sanitizer.hpp"

namespace test {

// AlAs |Luttinger| parameters (A > B), used as the default band structure throughout the tests
inline constexpr std::array<double, 3> AlAs {4.681, 1.019, 5.498};

inline void setLK(const std::array<double, 3> & abc){ weight::LKMatrix::setLKParameters(abc[0], abc[1], abc[2]); }

// Counts checks and prints one line per check; exit_code() is what main() should return.
class Checks {
public:
    explicit Checks(const char * name) : name_(name) { std::printf("=== %s\n", name_); }

    [[gnu::format(printf, 3, 4)]]
    bool operator()(bool ok, const char * fmt, ...) {
        ++total_;
        if (!ok) { ++failed_; }
        std::printf("  [%s] ", ok ? "PASS" : "FAIL");
        va_list args;
        va_start(args, fmt);
        std::vprintf(fmt, args);
        va_end(args);
        std::printf("\n");
        return ok;
    }

    int exit_code() const {
        std::printf("--- %s: %d/%d checks passed%s\n", name_, total_ - failed_, total_, failed_ ? "  ** FAILED **" : "");
        return failed_ ? 1 : 0;
    }

private:
    const char * name_;
    int total_ {0};
    int failed_ {0};
};

// An initialised diagram with its own rng, managers and one phonon mode. Members are declared in
// construction order (the managers and the config point at the members before them).
struct Diagram {
    simplemc::xoshiro256ss rng;
    VertexPointerManager internal;
    VertexPointerManager external;
    PhononModeManager modes;
    diagram_cfg cfg;

    Diagram(unsigned long long seed, std::array<double, 3> p, int max_internal_vertices, int max_external_vertices,
            double tau_max = 10.0, double chem_pot = -1.0, double ph_energy = 0.5, double diel_response = 2.0)
        : rng(seed),
          internal(max_internal_vertices, &rng),
          external(max_external_vertices, &rng),
          modes(1, std::vector<PhononMode>{PhononMode{ph_energy, diel_response}}, &rng),
          cfg(p, tau_max, chem_pot, &internal, &external, &modes)
    {
        // the diagram_cfg constructor leaves these unset; the sanitizer then builds every cached quantity
        cfg.diagram_tail->tau = cfg.current_tau_length;
        cfg.diagram_head->tau_next = cfg.current_tau_length;
        numerical::sanitizeDiagram(&cfg);
    }

    Diagram(const Diagram &) = delete;
    Diagram & operator=(const Diagram &) = delete;
};

} // namespace test

#endif // !TEST_COMMON_HPP
