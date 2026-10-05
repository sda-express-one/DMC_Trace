// A Metropolis chain with all eight updates, sanitized after EVERY accepted move.
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
#include <algorithm>
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

int main(){
    test::Checks check {"sanitizer after every accepted move, all updates"};
    test::setLK(test::AlAs);

    const char * name[8] {"add_int", "rm_int", "add_ext", "rm_ext", "swp", "mv_tau", "chg_tau", "chg_w"};
    const long min_accepted[8] {500, 500, 300, 300, 30, 1000, 10000, 1000};
    long accepted[8] {}, nonfinite[8] {}, dirty[8] {};
    long sanitized {0}, with_external {0}, max_internal {0}, max_external {0};
    double worst_overlap {0.}, worst_trace {0.};

    for (unsigned long long seed : {12345ULL, 777ULL}) {
        test::Diagram d {seed, {0.11, -0.07, 0.05}, 24, 8};
        diagram_cfg * cfg {&d.cfg};
        add_int_ph_update addi {cfg, &d.rng};  rm_int_ph_update rmi {cfg, &d.rng};
        add_ext_ph_update adde {cfg, &d.rng};  rm_ext_ph_update rme {cfg, &d.rng};
        swp_ph_update swp {cfg, &d.rng};       mv_tau_update mv {cfg, &d.rng};     chg_tau_update ch {cfg, &d.rng};
        chg_ph_momentum chw {cfg, &d.rng};
        std::uniform_real_distribution<double> u {0., 1.};
        std::uniform_int_distribution<int> pick {0, 7};

        for (long step {0}; step < 300000; ++step) {
            const int w {pick(d.rng)};
            double r {-1.};
            switch (w) { case 0: r = addi.attempt(); break; case 1: r = rmi.attempt(); break; case 2: r = adde.attempt(); break;
                         case 3: r = rme.attempt(); break; case 4: r = swp.attempt(); break; case 5: r = mv.attempt(); break;
                         case 6: r = ch.attempt(); break; default: r = chw.attempt(); }
            if (!std::isfinite(r)) { ++nonfinite[w]; }
            if (!(r > 0. && u(d.rng) < r)) { continue; }
            switch (w) { case 0: addi.accept(); break; case 1: rmi.accept(); break; case 2: adde.accept(); break;
                         case 3: rme.accept(); break; case 4: swp.accept(); break; case 5: mv.accept(); break;
                         case 6: ch.accept(); break; default: chw.accept(); }
            ++accepted[w];

            const numerical::SanitizeReport rep {numerical::sanitizeDiagram(cfg)};
            ++sanitized;
            if (!rep.clean()) { ++dirty[w]; }
            worst_overlap = std::max(worst_overlap, rep.max_overlap_dev);
            worst_trace = std::max(worst_trace, rep.relativeTraceChange());
            if (d.external.current_length > 0) { ++with_external; }
            max_internal = std::max<long>(max_internal, d.internal.current_length / 2);
            max_external = std::max<long>(max_external, d.external.current_length / 2);
        }
    }

    for (int i {0}; i < 8; ++i) {
        check(dirty[i] == 0 && nonfinite[i] == 0 && accepted[i] >= min_accepted[i],
              "%-8s accepted %6ld (need >= %ld)   dirty after it %ld   non-finite ratios %ld",
              name[i], accepted[i], min_accepted[i], dirty[i], nonfinite[i]);
    }
    check(max_internal >= 3 && max_external >= 2 && with_external > 1000,
          "configurations exercised: up to %ld internal / %ld external lines, %ld sanitized states with external lines",
          max_internal, max_external, with_external);
    check(worst_overlap < 1e-12 && worst_trace < 1e-12,
          "cached state exact across %ld sanitizer runs (worst overlap deviation %.1e, worst relative trace change %.1e)",
          sanitized, worst_overlap, worst_trace);
    return check.exit_code();
}
