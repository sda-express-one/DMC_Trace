// green_func_measurement in the orbital basis: every sample is rotated with the head's own eigenvectors,
// U P U^T / |tr P| (U = U(k_end), k_end = p - sum(w)), and the band-basis result is U_p^T G_orb U_p.
//   1. Vacuum sector, generic p: k_end = p, so the band-basis result must equal the old estimator, which
//      accumulated the unrotated P / |tr P|, kept here in a reference accumulator fed with the same samples
//      (same sparse indices, same batches): means and errors to rounding.
//   2. p = 0, one boundary phonon and all sectors (AlAs, so the bands at k_end are anisotropic): cubic symmetry
//      makes the orbital matrix proportional to the identity. Pulls of the off-diagonal elements and of the
//      diagonal differences, from a jackknife of those combinations, must be compatible with zero. The old
//      unrotated components, whose bases change with k_end from diagram to diagram, are reported as a contrast.
//   3. Symmetry G_ij = G_ji of the orbital matrix (generic p, vacuum and all sectors), by the same pulls.
//   4. Consistency of the two bases: G_orb = U_p G_band U_p^T and equal traces, to rounding.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <memory>
#include <span>
#include <random>
#include <vector>
#include "test_common.hpp"
#include "updates/add_internal_ph.hpp"
#include "updates/rm_internal_ph.hpp"
#include "updates/add_external_ph.hpp"
#include "updates/rm_external_ph.hpp"
#include "updates/swp_ph.hpp"
#include "updates/mv_vertex.hpp"
#include "updates/chg_tau.hpp"
#include "updates/chg_ph_momentum.hpp"
#include "measurements/green_func.hpp"
#include "measurements/measurement_utils.hpp"

namespace {
    constexpr int NB {30};

    // The chain of test_green_trace (without the negated samples). Next to G, a reference accumulator gets the
    // old, unrotated sample: P_nm / |tr P| in the band basis at k_end, with the same indices and normalisation.
    // (the diagram is owned here: the estimator keeps a pointer to its configuration)
    struct Run {
        std::unique_ptr<test::Diagram> d;
        green_func_measurement G;
        green_func_measurement::acc_type ref;
        double z0 {0.};

        Run(std::array<double, 3> p, int boundary_phonons, unsigned long long seed, long steps);
    };

    Run::Run(std::array<double, 3> p, int boundary_phonons, unsigned long long seed, long steps)
        : d(std::make_unique<test::Diagram>(seed, p, 24, 8)),
          G(&d->cfg, p, NB, boundary_phonons),
          ref(9*NB + 1, 256),
          z0(static_cast<double>(measurement::order0Integral(p, d->cfg.chem_pot, d->cfg.tau_max)))
    {
        test::Diagram & d {*this->d};
        diagram_cfg * cfg {&d.cfg};
        add_int_ph_update addi {cfg, &d.rng};  rm_int_ph_update rmi {cfg, &d.rng};
        add_ext_ph_update adde {cfg, &d.rng};  rm_ext_ph_update rme {cfg, &d.rng};
        swp_ph_update swp {cfg, &d.rng};       mv_tau_update mv {cfg, &d.rng};
        chg_tau_update ch {cfg, &d.rng};       chg_ph_momentum chw {cfg, &d.rng};
        std::uniform_real_distribution<double> u {0., 1.};
        std::uniform_int_distribution<int> pick {0, 7};

        for (long step {0}; step < steps; ++step) {
            const int w {pick(d.rng)};
            double a {-1.};
            switch (w) { case 0: a = addi.attempt(); break; case 1: a = rmi.attempt(); break; case 2: a = adde.attempt(); break;
                         case 3: a = rme.attempt(); break; case 4: a = swp.attempt(); break; case 5: a = mv.attempt(); break;
                         case 6: a = ch.attempt(); break; default: a = chw.attempt(); }
            if (a > 0. && u(d.rng) < a) {
                switch (w) { case 0: addi.accept(); break; case 1: rmi.accept(); break; case 2: adde.accept(); break;
                             case 3: rme.accept(); break; case 4: swp.accept(); break; case 5: mv.accept(); break;
                             case 6: ch.accept(); break; default: chw.accept(); }
            }
            G.measure();

            // the reference: green_func's measure() before the rotation
            std::array<double, 10> vals {};
            std::array<green_func_measurement::acc_type::size_type, 10> idxs {};
            std::size_t k {0};
            const int n_int {cfg->internal_ph_manager->current_length / 2};
            const int n_ext {cfg->external_ph_manager->current_length / 2};
            const Eigen::Matrix3d & P {cfg->diagram_head->right_component};
            if ((boundary_phonons < 0 || n_ext == boundary_phonons) && P.trace() != 0.) {
                const double tau {cfg->current_tau_length};
                const int bin {std::clamp(static_cast<int>(tau / G.bin_width), 0, NB - 1)};
                const double factor {std::exp(-cfg->chem_pot * (tau - bin * G.bin_width)) / std::abs(P.trace())};
                for (int n {0}; n < 3; ++n) {
                    for (int m {0}; m < 3; ++m) { vals[k] = P(n, m) * factor; idxs[k] = 9*bin + 3*n + m; ++k; }
                }
            }
            if (n_int == 0 && n_ext == 0) { vals[k] = 1.; idxs[k] = 9*NB; ++k; }
            ref.accumulate(std::span(vals.data(), k), std::span(idxs.data(), k));
        }
    }

    // per bin, the matrix [3*i + j] of an estimate vector laid out as green_func's x (norm applied)
    Eigen::Matrix3d binMatrix(const Eigen::VectorXd & v, int offset, int b){
        Eigen::Matrix3d M;
        for (int i {0}; i < 3; ++i) { for (int j {0}; j < 3; ++j) { M(i, j) = v(offset + 9*b + 3*i + j); } }
        return M;
    }

    // Pulls of the combinations that vanish by symmetry, per bin, from a jackknife of those combinations of the
    // matrices given by `matrices` (a function of the batch means returning the per-bin matrices back to back):
    // cubic = true: the 3 symmetrised off-diagonals and 2 diagonal differences; always: the 3 antisymmetric parts.
    struct Pulls { double chi2 {0.}; int dof {0}; double worst {0.}; };

    template <class F, class A>
    Pulls pulls(const F & matrices, const A & acc, bool cubic, int bins){
        const auto combos {[&](const Eigen::VectorXd & x){
            const Eigen::VectorXd m {matrices(x)};
            Eigen::VectorXd out((cubic ? 8 : 3) * bins);
            int k {0};
            for (int b {0}; b < bins; ++b) {
                const Eigen::Matrix3d M {binMatrix(m, 0, b)};
                out(k++) = M(0, 1) - M(1, 0); out(k++) = M(0, 2) - M(2, 0); out(k++) = M(1, 2) - M(2, 1);
                if (cubic) {
                    out(k++) = M(0, 1) + M(1, 0); out(k++) = M(0, 2) + M(2, 0); out(k++) = M(1, 2) + M(2, 1);
                    out(k++) = M(0, 0) - M(1, 1); out(k++) = M(1, 1) - M(2, 2);
                }
            }
            return out;
        }};
        const auto jk {simplemc::jackknife(combos, acc)};
        const Eigen::VectorXd mean {jk.naive_mean()}, err {jk.stderror()};
        Pulls p;
        for (Eigen::Index i {0}; i < mean.size(); ++i) {
            if (!(err(i) > 0.)) { continue; }
            const double z {mean(i) / err(i)};
            p.chi2 += z*z; ++p.dof; p.worst = std::max(p.worst, std::abs(z));
        }
        return p;
    }
}

int main(){
    test::Checks check {"green_func_measurement: orbital basis (per-sample rotation) and band basis at p"};
    test::setLK(test::AlAs);
    const int CHECKED {20};     // the tail bins hold too few samples for meaningful pulls

    // 1. and 4.: vacuum sector at a generic p
    {
        const std::array<double, 3> p {0.11, -0.07, 0.05};
        Run r(p, 0, 31337ULL, 600000);
        const green_func_measurement::result R {r.G.normalised()};
        const double bw {r.G.bin_width};
        const auto ref_fn {[&](const Eigen::VectorXd & x){ return Eigen::VectorXd(r.z0 / bw / x(9*NB) * x.head(9*NB)); }};
        const auto jk {simplemc::jackknife(ref_fn, r.ref)};
        const Eigen::VectorXd ref_mean {jk.naive_mean()}, ref_err {jk.stderror()};

        double scale {0.}, dev_mean {0.}, dev_err {0.}, dev_basis {0.}, dev_trace {0.};
        for (int b {0}; b < NB; ++b) { scale = std::max(scale, static_cast<double>(std::abs(R.trace[b]))); }
        for (int b {0}; b < NB; ++b) {
            const double s {static_cast<double>(measurement::binScale(b, bw, r.d->cfg.chem_pot))};
            Eigen::Matrix3d band, orb;
            for (int i {0}; i < 9; ++i) {
                dev_mean = std::max(dev_mean, std::abs(static_cast<double>(R.mean[b][i]) - s * ref_mean(9*b + i)) / scale);
                dev_err = std::max(dev_err, std::abs(static_cast<double>(R.error[b][i]) - s * ref_err(9*b + i)) / scale);
                band(i / 3, i % 3) = static_cast<double>(R.mean[b][i]);
                orb(i / 3, i % 3) = static_cast<double>(R.orbital[b][i]);
            }
            dev_basis = std::max(dev_basis, (orb - r.G.U_p * band * r.G.U_p.transpose()).cwiseAbs().maxCoeff() / scale);
            dev_trace = std::max(dev_trace, std::abs(orb.trace() - static_cast<double>(R.trace[b])) / scale);
        }
        check(dev_mean < 1e-12 && dev_err < 1e-10,
              "vacuum sector: band basis = the old unrotated estimator on the same samples: worst deviation mean %.1e, error %.1e "
              "(relative to max |tr G|)", dev_mean, dev_err);
        check(dev_basis < 1e-12 && dev_trace < 1e-12, "G_orb = U_p G_band U_p^T (%.1e) and tr G_orb = trace (%.1e)", dev_basis, dev_trace);

        const Pulls sym {pulls([&](const Eigen::VectorXd & x){ return Eigen::VectorXd(r.G.unscaled(x).segment(10*NB, 9*NB)); },
                               r.G.acc, false, CHECKED)};
        check(sym.chi2 / sym.dof < 2. && sym.worst < 5., "vacuum sector, generic p: G_ij - G_ji compatible with 0: chi^2/dof = %.1f/%d, "
              "largest |pull| %.2f", sym.chi2, sym.dof, sym.worst);
    }

    // 3.: all sectors at the same generic p
    {
        const std::array<double, 3> p {0.11, -0.07, 0.05};
        Run r(p, -1, 4711ULL, 600000);
        const Pulls sym {pulls([&](const Eigen::VectorXd & x){ return Eigen::VectorXd(r.G.unscaled(x).segment(10*NB, 9*NB)); },
                               r.G.acc, false, CHECKED)};
        check(sym.chi2 / sym.dof < 2. && sym.worst < 5., "all sectors, generic p: G_ij - G_ji compatible with 0: chi^2/dof = %.1f/%d, "
              "largest |pull| %.2f", sym.chi2, sym.dof, sym.worst);
    }

    // 2.: p = 0, outside the vacuum sector
    for (const int sector : {1, -1}) {
        const std::array<double, 3> p {0., 0., 0.};
        Run r(p, sector, 2718ULL + static_cast<unsigned long long>(sector + 1), 600000);
        const Pulls orb {pulls([&](const Eigen::VectorXd & x){ return Eigen::VectorXd(r.G.unscaled(x).segment(10*NB, 9*NB)); },
                               r.G.acc, true, CHECKED)};
        const double bw {r.G.bin_width};
        const Pulls old {pulls([&](const Eigen::VectorXd & x){ return Eigen::VectorXd(r.z0 / bw / x(9*NB) * x.head(9*NB)); },
                               r.ref, true, CHECKED)};
        check(orb.chi2 / orb.dof < 2. && orb.worst < 5.,
              "p = 0, %s: orbital G proportional to the identity (off-diagonals, diagonal differences, antisymmetric parts): "
              "chi^2/dof = %.1f/%d, largest |pull| %.2f; the old unrotated components: chi^2/dof = %.1f/%d, largest |pull| %.1f",
              sector < 0 ? "all sectors" : "1 boundary phonon", orb.chi2, orb.dof, orb.worst, old.chi2, old.dof, old.worst);
    }
    return check.exit_code();
}
