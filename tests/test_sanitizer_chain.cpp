// A Metropolis chain with all eleven updates (two phonon modes, so chg_ph_energy has something to change), sanitized after EVERY accepted move.
//
// numerical::sanitizeDiagram checks the structure (links, tau ordering, tau_next == next->tau, tail at
// current_tau_length), the lines and their manager registration, momentum conservation at every
// vertex, and every cached basis, mass, action and overlap against what k and tau imply, plus the
// cached full-diagram trace against a rebuild. Checking after every accept - not periodically -
// matters: add_ext_ph/rm_ext_ph rewrite cached state across the whole diagram, so a later accept
// can overwrite an inconsistency before a periodic check sees it.
//
// Also checks that no update ever returns a non-finite ratio, and that every update was actually
// exercised (accepted often enough, with internal and external lines both present at some point).
//
// Two external momenta with two seeds each: a generic one, and one on a symmetry plane, p = (0.1, 0, -0.05). There a
// rounding residual in an end segment's k flips the LK gauge and the trace closes on mismatched bases; the sanitizer's
// invariant (both ends bitwise equal, and exactly p without external lines: diagram_cfg::p_ext) must hold after every
// move. Then one shorter chain each on a diagonal plane (0.1, 0.1, -0.05), on (110) and on (111), where the wedge
// sort has a tie and (111) a degenerate pair (weight::LKMatrix::canonicalDirection). In all of them the cached trace
// must also match the gauge-free Tr[prod U A U^T] (SanitizeReport::reference_trace_dev).
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <random>
#include "test_common.hpp"
#include "updates/add_internal_ph.hpp"
#include "updates/rm_internal_ph.hpp"
#include "updates/add_external_ph.hpp"
#include "updates/rm_external_ph.hpp"
#include "updates/swp_ph.hpp"
#include "updates/mv_vertex.hpp"
#include "updates/chg_tau.hpp"
#include "updates/chg_ph_momentum.hpp"
#include "updates/chg_ph_energy.hpp"
#include "updates/str_diagram.hpp"
#include "updates/scl_diagram.hpp"

int main(){
    test::Checks check {"sanitizer after every accepted move, all updates"};
    test::setLK(test::AlAs);

    const char * name[11] {"add_int", "rm_int", "add_ext", "rm_ext", "swp", "mv_tau", "chg_tau", "chg_w", "chg_ph_e", "str_diag", "scl_diag"};
    const long min_accepted[11] {500, 500, 300, 300, 30, 1000, 10000, 1000, 1000, 1000, 1000};
    long accepted[11] {}, nonfinite[11] {}, dirty[11] {};
    long sanitized {0}, with_external {0}, max_internal {0}, max_external {0}, on_plane_line_free {0};
    long on_lines {0};
    double worst_overlap {0.}, worst_trace {0.}, worst_reference {0.};

    struct Run { std::array<double, 3> p; unsigned long long seed; long steps; };
    const Run runs[7] {{{0.11, -0.07, 0.05}, 12345ULL, 300000}, {{0.11, -0.07, 0.05}, 777ULL, 300000},
                       {{0.1, 0., -0.05}, 12345ULL, 300000}, {{0.1, 0., -0.05}, 777ULL, 300000},
                       {{0.1, 0.1, -0.05}, 4242ULL, 100000}, {{0.08, 0.08, 0.}, 4243ULL, 100000},
                       {{0.06, 0.06, 0.06}, 4244ULL, 100000}};
    for (int run {0}; run < 7; ++run) {
        const unsigned long long seed {runs[run].seed};
        const std::array<double, 3> & p_ext {runs[run].p};
        test::Diagram d {seed, p_ext, 24, 8, {PhononMode{0.5, 2.0}, PhononMode{0.8, 3.0}}};
        diagram_cfg * cfg {&d.cfg};
        add_int_ph_update addi {cfg, &d.rng};  rm_int_ph_update rmi {cfg, &d.rng};
        add_ext_ph_update adde {cfg, &d.rng};  rm_ext_ph_update rme {cfg, &d.rng};
        swp_ph_update swp {cfg, &d.rng};       mv_tau_update mv {cfg, &d.rng};     chg_tau_update ch {cfg, &d.rng};
        chg_ph_momentum chw {cfg, &d.rng};     chg_ph_energy che {cfg, &d.rng};
        str_diagram_update strd {cfg, &d.rng};     scl_diagram_update scld {cfg, &d.rng};
        std::uniform_real_distribution<double> u {0., 1.};
        std::uniform_int_distribution<int> pick {0, 10};

        for (long step {0}; step < runs[run].steps; ++step) {
            const int w {pick(d.rng)};
            double r {-1.};
            switch (w) { case 0: r = addi.attempt(); break; case 1: r = rmi.attempt(); break; case 2: r = adde.attempt(); break;
                         case 3: r = rme.attempt(); break; case 4: r = swp.attempt(); break; case 5: r = mv.attempt(); break;
                         case 6: r = ch.attempt(); break; case 7: r = chw.attempt(); break; case 8: r = che.attempt(); break; case 9: r = strd.attempt(); break; default: r = scld.attempt(); }
            if (!std::isfinite(r)) { ++nonfinite[w]; }
            if (!(r > 0. && u(d.rng) < r)) { continue; }
            switch (w) { case 0: addi.accept(); break; case 1: rmi.accept(); break; case 2: adde.accept(); break;
                         case 3: rme.accept(); break; case 4: swp.accept(); break; case 5: mv.accept(); break;
                         case 6: ch.accept(); break; case 7: chw.accept(); break; case 8: che.accept(); break; case 9: strd.accept(); break; default: scld.accept(); }
            ++accepted[w];

            const numerical::SanitizeReport rep {numerical::sanitizeDiagram(cfg)};
            ++sanitized;
            if (!rep.clean()) { ++dirty[w]; }
            worst_overlap = std::max(worst_overlap, rep.max_overlap_dev);
            worst_trace = std::max(worst_trace, rep.relativeTraceChange());
            worst_reference = std::max(worst_reference, rep.reference_trace_dev);
            if (run >= 4) { ++on_lines; }
            if (d.external.current_length > 0) { ++with_external; }
            else if (run == 2 || run == 3) { ++on_plane_line_free; }
            max_internal = std::max<long>(max_internal, d.internal.current_length / 2);
            max_external = std::max<long>(max_external, d.external.current_length / 2);
        }
    }

    for (int i {0}; i < 11; ++i) {
        check(dirty[i] == 0 && nonfinite[i] == 0 && accepted[i] >= min_accepted[i],
              "%-8s accepted %6ld (need >= %ld)   dirty after it %ld   non-finite ratios %ld",
              name[i], accepted[i], min_accepted[i], dirty[i], nonfinite[i]);
    }
    check(max_internal >= 3 && max_external >= 2 && with_external > 1000 && on_plane_line_free > 10000,
          "configurations exercised: up to %ld internal / %ld external lines, %ld sanitized states with external lines, "
          "%ld without external lines at p = (0.1, 0, -0.05) on a symmetry plane", max_internal, max_external, with_external, on_plane_line_free);
    check(worst_overlap < 1e-12 && worst_trace < 1e-12 && worst_reference < 1e-10 && on_lines > 10000,
          "cached state exact across %ld sanitizer runs, %ld of them at p on a diagonal plane, (110) or (111) (worst overlap "
          "deviation %.1e, worst relative trace change %.1e, worst deviation from the gauge-free trace %.1e)",
          sanitized, on_lines, worst_overlap, worst_trace, worst_reference);
    return check.exit_code();
}
