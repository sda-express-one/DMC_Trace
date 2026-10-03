// add_int_ph / rm_int_ph are exact inverses: for X and Y = X + one internal line, the raw acceptance
// ratios satisfy R_add(X -> Y) * R_rm(Y -> X) = 1. That checks the line-counting factors (p_A, p_B:
// lines and propagators, internal and external), the tau and w proposal densities, (2 pi)^3 and
// V_unit_cell, and that both updates evaluate the same weights W(X), W(Y).
//
// rm_int_ph picks one of Y's internal lines uniformly, so it is re-attempted until it proposes the
// line just added (attempt() only stages a proposal); its ratio already contains the 1/L selection
// probability that the product needs. X can therefore carry any number of internal and external lines,
// and the counting factors are tested at every count - an off-by-one in either p_A, like the one once
// in rm_int_ph, gives a product of n/(n+1) and fails.
//
// X is taken from a chain running all updates. Configurations that must be exercised: the new line with
// and without vertices inside it, and X with and without other internal lines and external lines.
// After each accepted add and remove the diagram must also pass the sanitizer.
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

int main(){
    test::Checks check {"add_int_ph x rm_int_ph ratio product"};
    test::setLK(test::AlAs);

    test::Diagram d {4242ULL, {0.11, -0.07, 0.05}, 24, 8};
    diagram_cfg * cfg {&d.cfg};
    add_int_ph_update addi {cfg, &d.rng};  rm_int_ph_update rmi {cfg, &d.rng};
    add_ext_ph_update adde {cfg, &d.rng};  rm_ext_ph_update rme {cfg, &d.rng};
    swp_ph_update swp {cfg, &d.rng};       mv_tau_update mv {cfg, &d.rng};     chg_tau_update ch {cfg, &d.rng};
    std::uniform_real_distribution<double> u {0., 1.};
    std::uniform_int_distribution<int> pick {0, 6};

    // classes: a new line spanning other vertices needs X to have lines, so there are three
    const char * name[3] {"new line adjacent, X bare      ", "new line adjacent, X with lines",
                          "new line spanning other vertices"};
    const long PAIRS {12000};
    long n[3] {}, pairs {0}, dirty {0}, gave_up {0}, x_with_external {0}, x_with_internal {0}, max_lines {0};
    double worst[3] {};

    for (long step {0}; pairs < PAIRS && step < 40000000; ++step) {
        {   // evolve X with every update
            const int w {pick(d.rng)}; double r {-1.};
            switch (w) { case 0: r = addi.attempt(); break; case 1: r = rmi.attempt(); break; case 2: r = adde.attempt(); break;
                         case 3: r = rme.attempt(); break; case 4: r = swp.attempt(); break; case 5: r = mv.attempt(); break;
                         default: r = ch.attempt(); }
            if (r > 0. && u(d.rng) < r) {
                switch (w) { case 0: addi.accept(); break; case 1: rmi.accept(); break; case 2: adde.accept(); break;
                             case 3: rme.accept(); break; case 4: swp.accept(); break; case 5: mv.accept(); break;
                             default: ch.accept(); }
            }
        }
        if (step % 5 != 0) { continue; }

        const int lines_in_X {d.internal.current_length / 2};
        const bool X_has_external {d.external.current_length > 0};
        const double r_add {addi.attempt()};
        if (!(r_add > 0.) || !std::isfinite(r_add)) { continue; }
        const bool adjacent {addi.ptr_one == addi.ptr_two};     // both new vertices inside one segment
        const double tau_creation {addi.tau_one};
        addi.accept();                                          // X -> Y
        if (!numerical::sanitizeDiagram(cfg).clean()) { ++dirty; }

        // the creation vertex of the line just added sits exactly at tau_one
        Vertex * created {nullptr};
        for (Vertex * v {cfg->diagram_head->next}; v != cfg->diagram_tail; v = v->next) {
            if (v->type == +1 && v->tau == tau_creation) { created = v; }
        }

        double r_rm {-1.};
        int tries {0};
        do { r_rm = rmi.attempt(); ++tries; } while (rmi.ptr_one != created && tries < 100000);
        if (rmi.ptr_one != created) { ++gave_up; rmi.attempt(); }   // leave nothing half-staged

        if (rmi.ptr_one == created) {
            const int k {!adjacent ? 2 : ((lines_in_X > 0 || X_has_external) ? 1 : 0)};
            worst[k] = std::max(worst[k], std::abs(r_add * r_rm - 1.));
            ++n[k]; ++pairs;
            if (X_has_external) { ++x_with_external; }
            if (lines_in_X > 0) { ++x_with_internal; }
            max_lines = std::max<long>(max_lines, lines_in_X);
            rmi.accept();                                       // Y -> X
        }
        else {
            // could not reach the new line: remove it some other way is not possible here, so stop
            break;
        }
        if (!numerical::sanitizeDiagram(cfg).clean()) { ++dirty; }
    }

    check(pairs == PAIRS && gave_up == 0, "collected %ld add/remove pairs (rm_int_ph always reached the new line)", pairs);
    // tolerance: rounding, amplified when the 3x3 trace cancels (up to ~1e-12 seen, at a cancellation
    // |R|_max / |tr R| ~ 300); an off-by-one in a counting factor gives at least 1/2
    for (int k {0}; k < 3; ++k) {
        check(n[k] >= 50 && worst[k] < 1e-10, "%s %6ld pairs   max |R_add * R_rm - 1| = %.1e", name[k], n[k], worst[k]);
    }
    check(x_with_internal >= 500 && x_with_external >= 200,
          "X varied: %ld with other internal lines (up to %ld), %ld with external lines", x_with_internal, max_lines, x_with_external);
    check(dirty == 0, "sanitizer clean after every accepted add and remove (%ld dirty)", dirty);
    return check.exit_code();
}
