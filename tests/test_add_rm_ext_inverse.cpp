// add_ext_ph / rm_ext_ph are exact inverses: for the same pair of diagrams X (no external line) and
// Y = X + one external line, the raw acceptance ratios satisfy R_add(X -> Y) * R_rm(Y -> X) = 1.
// That checks, all at once, that the context factors (p_A, p_B, tau and w proposal densities,
// (2 pi)^3, V_unit_cell) invert each other and that both updates evaluate the same weights W(X), W(Y).
//
// Diagrams X are taken from a chain of internal-only moves, so the external line is added onto varied
// diagrams with internal lines. Y holds exactly one external line, so rm_ext_ph necessarily proposes
// removing the line just added. Every configuration class must be exercised:
//   case 1 (annihilation first) / case 2 (creation first), with and without vertices inside the line.
// After each accepted add and remove the diagram must also pass the sanitizer.
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

int main(){
    test::Checks check {"add_ext_ph x rm_ext_ph ratio product"};
    test::setLK(test::AlAs);

    test::Diagram d {2024ULL, {0.11, -0.07, 0.05}, 24, 2};    // room for exactly one external line
    diagram_cfg * cfg {&d.cfg};
    add_int_ph_update addi {cfg, &d.rng};  rm_int_ph_update rmi {cfg, &d.rng};  swp_ph_update swp {cfg, &d.rng};
    mv_tau_update mv {cfg, &d.rng};         chg_tau_update ch {cfg, &d.rng};
    add_ext_ph_update adde {cfg, &d.rng};  rm_ext_ph_update rme {cfg, &d.rng};
    std::uniform_real_distribution<double> u {0., 1.};
    std::uniform_int_distribution<int> pick {0, 4};

    const char * name[4] {"case 1, no interior vertices", "case 1, interior vertices", "case 2, no interior vertices", "case 2, interior vertices"};
    const long PAIRS {12000};
    long n[4] {}, dirty {0}, pairs {0}, max_internal {0};
    double worst[4] {};
    for (long step {0}; pairs < PAIRS && step < 20000000; ++step) {
        {   // evolve X with internal-only moves
            const int w {pick(d.rng)}; double r {-1.};
            switch (w) { case 0: r = addi.attempt(); break; case 1: r = rmi.attempt(); break; case 2: r = swp.attempt(); break;
                         case 3: r = mv.attempt(); break; default: r = ch.attempt(); }
            if (r > 0. && u(d.rng) < r) {
                switch (w) { case 0: addi.accept(); break; case 1: rmi.accept(); break; case 2: swp.accept(); break;
                             case 3: mv.accept(); break; default: ch.accept(); }
            }
        }
        if (step % 5 != 0) { continue; }

        const double r_add {adde.attempt()};
        if (!(r_add > 0.) || !std::isfinite(r_add)) { continue; }
        const bool case1 {adde.incoming_before_outgoing};
        adde.accept();                                                       // X -> Y
        if (!numerical::sanitizeDiagram(cfg).clean()) { ++dirty; }

        Vertex * va {d.external.ptr_vertex_pool[0].linked_vertex};
        if (va->type != -2) { va = va->conj_vertex; }                        // the annihilation vertex
        Vertex * vc {va->conj_vertex};
        const bool interior {case1 ? va->next != vc : vc->next != va};
        const int k {(case1 ? 0 : 2) + (interior ? 1 : 0)};

        const double r_rm {rme.attempt()};                                    // the only line in Y
        worst[k] = std::max(worst[k], std::abs(r_add * r_rm - 1.));
        ++n[k]; ++pairs;
        max_internal = std::max<long>(max_internal, d.internal.current_length / 2);

        rme.accept();                                                        // Y -> X
        if (!numerical::sanitizeDiagram(cfg).clean()) { ++dirty; }
    }

    check(pairs == PAIRS, "collected %ld add/remove pairs (X with up to %ld internal lines)", pairs, max_internal);
    for (int k {0}; k < 4; ++k) {
        check(n[k] >= 50 && worst[k] < 1e-10, "%-30s %6ld pairs   max |R_add * R_rm - 1| = %.1e", name[k], n[k], worst[k]);
    }
    check(dirty == 0, "sanitizer clean after every accepted add and remove (%ld dirty)", dirty);
    return check.exit_code();
}
