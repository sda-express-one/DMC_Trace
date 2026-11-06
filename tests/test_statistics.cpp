// statistics_measurement, the average number of internal and external phonon lines:
//   1. C = 0 (no sign problem), split on: no negative samples, signed = plain, plus = plain, minus NaN; the plain
//      means agree with the running means of all samples within errors;
//   2. AlAs parameters (C != 0) at fixed tau_D, split on: every quantity equals the same function evaluated on a
//      reference batch accumulator fed with independently computed samples (same batches: same jackknife), the
//      decomposition identities plain = f+ n_plus + (1 - f+) n_minus and signed = (f+ n_plus - f- n_minus)/(f+ - f-)
//      hold, and the negative count equals avg_sign's on the same samples;
//   3. tau cutoff, variable length (chg_tau): only diagrams with tau_D >= cutoff are counted; a cutoff above tau_max
//      measures nothing and summary() stays NaN;
//   4. summary(): NaNs before 2 full batches, then exactly normalised(), in summaryNames() order, with and without
//      the split.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>
#include <Eigen/Core>
#include <simplemc/accs/batch_acc.hpp>
#include <simplemc/accs/jackknife.hpp>
#include "test_common.hpp"
#include "updates/add_internal_ph.hpp"
#include "updates/rm_internal_ph.hpp"
#include "updates/add_external_ph.hpp"
#include "updates/rm_external_ph.hpp"
#include "updates/swp_ph.hpp"
#include "updates/mv_vertex.hpp"
#include "updates/chg_ph_momentum.hpp"
#include "updates/chg_tau.hpp"
#include "measurements/avg_sign.hpp"
#include "measurements/statistics.hpp"

// every length-preserving update, plus chg_tau for variable length (mu < 0)
struct Chain {
    test::Diagram d;
    add_int_ph_update addi;  rm_int_ph_update rmi;
    add_ext_ph_update adde;  rm_ext_ph_update rme;
    mv_tau_update mv;        swp_ph_update swp;
    chg_ph_momentum chw;     chg_tau_update cht;
    std::uniform_real_distribution<double> u {0., 1.};
    std::uniform_int_distribution<int> pick;

    Chain(unsigned long long seed, double L, double mu, double omega, double eps)
        : d(seed, {0., 0., 0.}, 400, 16, L, mu, omega, eps),
          addi(&d.cfg, &d.rng), rmi(&d.cfg, &d.rng), adde(&d.cfg, &d.rng), rme(&d.cfg, &d.rng),
          mv(&d.cfg, &d.rng), swp(&d.cfg, &d.rng), chw(&d.cfg, &d.rng), cht(&d.cfg, &d.rng),
          pick(0, mu < 0. ? 7 : 6) {}

    void step(){
        const int w {pick(d.rng)};
        const double x {u(d.rng)};
        const auto go {[x](auto & up){ const double r {up.attempt()}; (r > 0. && x < r) ? up.accept() : up.reject(); }};
        switch (w) { case 0: go(addi); break; case 1: go(rmi); break; case 2: go(adde); break; case 3: go(rme); break;
                     case 4: go(mv); break; case 5: go(swp); break; case 6: go(chw); break; default: go(cht); }
    }

    double nInt() const { return static_cast<double>(d.cfg.internal_ph_manager->current_length) / 2.; }
    double nExt() const { return static_cast<double>(d.cfg.external_ph_manager->current_length) / 2.; }
    double sign() const { return d.cfg.diagram_head->right_component.trace() < 0. ? -1. : 1.; }
};

namespace {
    // the quantities of statistics_measurement::summaryNames() from the batch means, written independently
    Eigen::VectorXd reference(const Eigen::VectorXd & x){
        const double fp {x(5)}, fm {1. - x(5)};
        Eigen::VectorXd q(13);
        q << x(0), x(1), x(0) + x(1),
             x(3) / x(2), x(4) / x(2), (x(3) + x(4)) / x(2),
             x(6) / fp, x(7) / fp, (x(6) + x(7)) / fp,
             (x(0) - x(6)) / fm, (x(1) - x(7)) / fm, (x(0) + x(1) - x(6) - x(7)) / fm,
             fp;
        return q;
    }

    bool summaryMatches(const statistics_measurement & m){
        const auto r {m.normalised()};
        const std::vector<double> s {m.summary()};
        if (s.size() != 2 * m.summaryNames().size() || r.value.size() != m.summaryNames().size()) { return false; }
        for (std::size_t i {0}; i < r.value.size(); ++i) {
            const auto same {[](double a, double b){ return a == b || (std::isnan(a) && std::isnan(b)); }};
            if (!same(s[2*i], r.value[i]) || !same(s[2*i + 1], r.error[i])) { return false; }
        }
        return true;
    }

    bool allNaN(const std::vector<double> & v){
        for (const double x : v) { if (!std::isnan(x)) { return false; } }
        return !v.empty();
    }
}

int main(){
    test::Checks check {"statistics_measurement (average number of phonon lines)"};

    // 1. C = 0
    {
        test::setLK({1., 1., 0.});
        Chain c {7ULL, 10., 0., 1., 1.};
        statistics_measurement S {&c.d.cfg, 0., true, 16};
        const bool nan_before {allNaN(S.summary())};
        double sum_i {0.}, sum_e {0.};
        std::size_t n {0};
        for (int i {0}; i < 200000; ++i) {
            c.step();
            if (i % 10 == 0) { S.measure(); sum_i += c.nInt(); sum_e += c.nExt(); ++n; }
        }
        const auto r {S.normalised()};
        const double run_i {sum_i / static_cast<double>(n)}, run_e {sum_e / static_cast<double>(n)};
        check(S.n_negative == 0 && S.n_measured == n && r.value[3] == r.value[0] && r.value[4] == r.value[1]
              && r.value[6] == r.value[0] && r.value[7] == r.value[1] && std::isnan(r.value[9]) && r.value[12] == 1.,
              "C = 0: no negative samples; signed = plain (%.5f, %.5f), plus = plain, minus NaN, fraction_plus = 1",
              r.value[0], r.value[1]);
        check(std::abs(r.value[0] - run_i) < 4. * r.error[0] + 1e-12 && std::abs(r.value[1] - run_e) < 4. * r.error[1] + 1e-12
              && r.value[0] > 0.,
              "C = 0: <n_int> = %.5f +- %.5f, <n_ext> = %.5f +- %.5f; running means over all samples %.5f, %.5f",
              r.value[0], r.error[0], r.value[1], r.error[1], run_i, run_e);
        check(nan_before && summaryMatches(S), "C = 0: summary() NaN before 2 batches, then normalised() (split on)");
    }

    // 2. C != 0, fixed length, split on
    {
        test::setLK(test::AlAs);
        Chain c {11ULL, 60., 0., 1., 0.5};
        statistics_measurement S {&c.d.cfg, 0., true, 16};
        avg_sign_measurement A {&c.d.cfg, 0., true, 16};
        simplemc::batch_acc_dynamic<double> ref(8, 16);
        for (int i {0}; i < 2000000; ++i) {
            c.step();
            if (i % 10 == 0) {
                S.measure();
                A.measure();
                const double s {c.sign()}, ni {c.nInt()}, ne {c.nExt()}, p {s > 0. ? 1. : 0.};
                Eigen::VectorXd x(8);
                x << ni, ne, s, s * ni, s * ne, p, p * ni, p * ne;
                ref.accumulate(x);
            }
        }
        const auto r {S.normalised()};
        const auto jk {simplemc::jackknife([](const Eigen::VectorXd & x){ return reference(x); }, ref)};
        const Eigen::VectorXd v {jk.naive_mean()}, e {jk.stderror()};
        double worst {0.};
        for (int i {0}; i < 13; ++i) {
            worst = std::max(worst, std::abs(r.value[static_cast<std::size_t>(i)] - v(i)) / std::max(1., std::abs(v(i))));
            worst = std::max(worst, std::abs(r.error[static_cast<std::size_t>(i)] - e(i)) / std::max(1., std::abs(e(i))));
        }
        check(worst < 1e-12 && S.n_negative > 0,
              "AlAs, tau_D = 60: all 13 quantities and errors equal the reference accumulator's (worst relative diff %.1e); "
              "%llu negative of %llu", worst, static_cast<unsigned long long>(S.n_negative),
              static_cast<unsigned long long>(S.n_measured));
        const double fp {r.value[12]}, fm {1. - fp};
        double id_plain {0.}, id_signed {0.};
        for (std::size_t q {0}; q < 3; ++q) {
            id_plain = std::max(id_plain, std::abs(fp * r.value[6 + q] + fm * r.value[9 + q] - r.value[q]));
            id_signed = std::max(id_signed, std::abs((fp * r.value[6 + q] - fm * r.value[9 + q]) / (fp - fm) - r.value[3 + q]));
        }
        check(id_plain < 1e-12 && id_signed < 1e-10,
              "AlAs: plain = f+ n_plus + f- n_minus (max diff %.1e), signed = (f+ n_plus - f- n_minus)/(f+ - f-) (max diff %.1e)",
              id_plain, id_signed);
        check(S.n_negative == A.n_negative && S.n_measured == A.n_measured,
              "AlAs: the same %llu negative samples as avg_sign", static_cast<unsigned long long>(S.n_negative));
        check(summaryMatches(S), "AlAs: summary() equals normalised() for all %zu quantities", S.summaryNames().size());
        std::printf("    n_int: plain %.4f +- %.4f  signed %.4f +- %.4f  plus %.4f +- %.4f  minus %.4f +- %.4f  (f+ = %.4f)\n",
                    r.value[0], r.error[0], r.value[3], r.error[3], r.value[6], r.error[6], r.value[9], r.error[9], fp);
    }

    // 3. tau cutoff, variable length
    {
        test::setLK({1., 1., 0.});
        Chain c {23ULL, 20., -1., 1., 1.};
        const double cutoff {2.};
        statistics_measurement S {&c.d.cfg, cutoff, false, 16};
        statistics_measurement all {&c.d.cfg, 0., false, 16};
        statistics_measurement none {&c.d.cfg, 25., false, 16};
        std::uint64_t above {0}, total {0};
        for (int i {0}; i < 400000; ++i) {
            c.step();
            if (i % 10 == 0) {
                S.measure(); all.measure(); none.measure();
                ++total;
                if (c.d.cfg.current_tau_length >= cutoff) { ++above; }
            }
        }
        check(S.n_measured == above && all.n_measured == total && above > 0 && above < total,
              "cutoff %.1f: %llu of %llu diagrams counted (tau_D >= cutoff), cutoff 0 counts all", cutoff,
              static_cast<unsigned long long>(S.n_measured), static_cast<unsigned long long>(all.n_measured));
        check(none.n_measured == 0 && allNaN(none.summary()), "cutoff above tau_max: nothing measured, summary() NaN");
        check(summaryMatches(S) && S.summaryNames().size() == 6, "cutoff: summary() equals normalised() without the split (6 quantities)");
        const auto r {S.normalised()}, ra {all.normalised()};
        std::printf("    <n_int>: tau_D >= %.1f %.4f +- %.4f, all %.4f +- %.4f\n", cutoff, r.value[0], r.error[0], ra.value[0], ra.error[0]);
    }
    return check.exit_code();
}
