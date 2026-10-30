#ifndef TEST_COMMON_HPP
#define TEST_COMMON_HPP

// Shared helpers for the regression tests: a check/report counter and a fixture that builds an
// initialised diagram (order 0, tail at current_tau_length, cached state built by the sanitizer).

#include <array>
#include <cstdarg>
#include <cstdio>
#include <initializer_list>
#include <string>
#include <utility>
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
// CaO valence band (Guster et al.; cluster_outputs/CaO_input.json)
inline constexpr std::array<double, 3> CaO {1.407, 0.174, 0.653};

// Band structures for the diagonalizer tests (test_lk_sphere, test_lk_gauge): AlAs, AlAs with A and B swapped, CaO, then a
// scan of C for both orderings over the allowed range -(A/2 + B) < C < A + B (H_LK positive definite). C = +-(A - B) is
// avoided: there (110) is degenerate, and for C = A - B every k (spherical bands). C = 0 is covered separately.
struct LKCase { std::string label; std::array<double, 3> abc; };

inline std::vector<LKCase> lkCases(){
    const double A {AlAs[0]}, B {AlAs[1]};
    std::vector<LKCase> cases {{"AlAs (A > B)", AlAs}, {"A < B", {B, A, AlAs[2]}}, {"CaO", CaO}};
    const auto add {[&](double a, double b, std::initializer_list<double> cs){
        for (const double c : cs) {
            char label[48];
            std::snprintf(label, sizeof label, "%s, C = %+.2f", a > b ? "A > B" : "A < B", c);
            cases.push_back({label, {a, b, c}});
        }
    }};
    add(A, B, {-3.0, -1.5, -0.3, 0.3, 1.5, 3.0, 4.5, 5.4});      // range (-3.36, 5.70); A - B = 3.66
    add(B, A, {-4.8, -2.5, -0.3, 0.3, 1.5, 3.0, 4.5, 5.4});      // range (-5.19, 5.70); +-(A - B) = +-3.66
    return cases;
}

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

    // the same, with several phonon modes (for the updates that draw or change a line's mode)
    Diagram(unsigned long long seed, std::array<double, 3> p, int max_internal_vertices, int max_external_vertices,
            std::vector<PhononMode> phonon_modes, double tau_max = 10.0, double chem_pot = -1.0)
        : rng(seed),
          internal(max_internal_vertices, &rng),
          external(max_external_vertices, &rng),
          modes(static_cast<int>(phonon_modes.size()), phonon_modes, &rng),
          cfg(p, tau_max, chem_pot, &internal, &external, &modes)
    {
        cfg.diagram_tail->tau = cfg.current_tau_length;
        cfg.diagram_head->tau_next = cfg.current_tau_length;
        numerical::sanitizeDiagram(&cfg);
    }

    Diagram(const Diagram &) = delete;
    Diagram & operator=(const Diagram &) = delete;
};

} // namespace test

#endif // !TEST_COMMON_HPP
