// measurement::diagnose, the rank-normalised split-R-hat / folded R-hat / bulk- and tail-ESS of Vehtari et al. (2021),
// on synthetic chains with known behaviour - the cases of the paper's Figure 2 and Appendix A:
//   - Phi^-1 against reference values;
//   - 4 chains of iid normals: R-hat < 1.01, bulk-ESS ~ S;
//   - AR(1) chains, rho = 0.5: bulk-ESS ~ S (1 - rho)/(1 + rho);
//   - one chain shifted by one standard deviation: R-hat > 1.01;
//   - one chain with 1/3 of the variance: the bulk R-hat misses it, the folded R-hat catches it;
//   - a linear trend shared by all chains: missed without splitting, caught by split-R-hat;
//   - Cauchy chains (infinite variance): equal ones pass, one shifted chain is caught;
//   - invariance under a monotone transformation (exp); constant chains; too short chains.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>
#include "test_common.hpp"
#include "measurements/convergence.hpp"

using Chains = std::vector<std::vector<double>>;

static Chains normal_chains(unsigned long long seed, int M, int N, double rho = 0.){
    std::mt19937_64 rng {seed};
    std::normal_distribution<double> g {0., 1.};
    Chains c(static_cast<std::size_t>(M));
    for (auto & ch : c) {
        double x {g(rng)};
        for (int i {0}; i < N; ++i) {
            ch.push_back(x);
            x = rho * x + std::sqrt(1. - rho * rho) * g(rng);      // stationary AR(1) with unit variance
        }
    }
    return c;
}

static Chains cauchy_chains(unsigned long long seed, int M, int N){
    std::mt19937_64 rng {seed};
    std::normal_distribution<double> g {0., 1.};
    Chains c(static_cast<std::size_t>(M));
    for (auto & ch : c) { for (int i {0}; i < N; ++i) { ch.push_back(g(rng) / g(rng)); } }
    return c;
}

int main(){
    test::Checks check {"convergence diagnostics (rank-normalised split-R-hat, folding, ESS)"};

    // Phi^-1 against reference values (Newton on Phi(x) = p in 80-bit long double with erfcl), to full double
    // precision; the upper tail close to 1 is where a refinement step on Phi(x) - p loses its digits to cancellation
    struct Ref { double p, x; };
    const Ref refs[] {{0.5, 0.}, {0.975, 1.959963984540053856}, {0.025, -1.959963984540054212}, {0.3, -0.524400512708040816},
                      {1e-6, -4.753424308822898957}, {1. - 1e-6, 4.753424308817087766}, {1. - 1e-10, 6.361340889697421864},
                      {1e-20, -9.262340089798407579}, {1e-300, -37.047096299361199235}};
    double worst_ref {0.};
    for (const Ref & r : refs) {
        const double x {measurement::inverseNormalCdf(r.p)};
        worst_ref = std::max(worst_ref, std::abs(x - r.x) / std::max(1., std::abs(r.x)));
    }
    double worst_sym {0.};
    for (const double p : {1e-12, 1e-4, 0.01, 0.1, 0.25, 0.4}) {
        // u = 1 - p rounds, but t = 1 - u is exact (u >= 1/2): t and u sum to exactly 1, so Phi^-1(t) = -Phi^-1(u)
        const double u {1. - p}, t {1. - u};
        const double xt {measurement::inverseNormalCdf(t)};
        worst_sym = std::max(worst_sym, std::abs(xt + measurement::inverseNormalCdf(u)) / std::max(1., std::abs(xt)));
    }
    check(worst_ref < 1e-14 && worst_sym < 1e-14,
          "Phi^-1 (Wichura AS 241): worst relative deviation from the reference %.1e over p = 1e-300 .. 1 - 1e-10; "
          "Phi^-1(p) + Phi^-1(1 - p) = %.1e", worst_ref, worst_sym);

    const int M {4}, N {1000};
    const double S {M * N};

    const measurement::ConvergenceResult iid {measurement::diagnose(normal_chains(1, M, N))};
    check(iid.rhat < 1.01 && std::abs(iid.ess_bulk / S - 1.) < 0.15 && iid.ess_tail > 0.5 * S && iid.chains == 2*M && iid.draws == N/2,
          "iid normal: R-hat %.4f (bulk %.4f, fold %.4f), bulk-ESS %.0f / S = %.0f, tail-ESS %.0f",
          iid.rhat, iid.rhat_bulk, iid.rhat_fold, iid.ess_bulk, S, iid.ess_tail);

    const double rho {0.5};
    const measurement::ConvergenceResult ar {measurement::diagnose(normal_chains(2, M, 4 * N, rho))};
    const double ess_exact {4. * S * (1. - rho) / (1. + rho)};
    check(ar.rhat < 1.01 && std::abs(ar.ess_bulk / ess_exact - 1.) < 0.15,
          "AR(1) rho = 0.5: R-hat %.4f, bulk-ESS %.0f vs S (1 - rho)/(1 + rho) = %.0f", ar.rhat, ar.ess_bulk, ess_exact);

    Chains shifted {normal_chains(3, M, N)};
    for (double & x : shifted[0]) { x += 1.; }
    const measurement::ConvergenceResult sh {measurement::diagnose(shifted)};
    check(sh.rhat > 1.01, "one chain shifted by 1 sigma: R-hat %.4f > 1.01", sh.rhat);

    Chains narrow {normal_chains(4, M, N)};
    for (double & x : narrow[0]) { x *= std::sqrt(1. / 3.); }
    const measurement::ConvergenceResult nr {measurement::diagnose(narrow)};
    check(nr.rhat_bulk < 1.01 && nr.rhat_fold > 1.01,
          "one chain with 1/3 of the variance: bulk R-hat %.4f misses it, folded R-hat %.4f catches it", nr.rhat_bulk, nr.rhat_fold);

    Chains trend {normal_chains(5, M, N)};
    for (auto & ch : trend) { for (int i {0}; i < N; ++i) { ch[static_cast<std::size_t>(i)] += 1.5 * i / N; } }
    const double unsplit {measurement::splitRhat(measurement::rankNormalize(trend))};
    const measurement::ConvergenceResult tr {measurement::diagnose(trend)};
    check(unsplit < 1.01 && tr.rhat > 1.01,
          "common linear trend: unsplit R-hat %.4f misses it, split-R-hat %.4f catches it", unsplit, tr.rhat);

    const measurement::ConvergenceResult ca {measurement::diagnose(cauchy_chains(6, M, N))};
    Chains cauchy_shifted {cauchy_chains(7, M, N)};
    for (double & x : cauchy_shifted[0]) { x += 2.; }
    const measurement::ConvergenceResult cs {measurement::diagnose(cauchy_shifted)};
    check(ca.rhat < 1.01 && cs.rhat > 1.01,
          "Cauchy chains: equal ones R-hat %.4f, one shifted by 2: R-hat %.4f", ca.rhat, cs.rhat);

    Chains expd {normal_chains(8, M, N)};
    const measurement::ConvergenceResult before {measurement::diagnose(expd)};
    for (auto & ch : expd) { for (double & x : ch) { x = std::exp(x); } }
    const measurement::ConvergenceResult after {measurement::diagnose(expd)};
    check(std::abs(before.rhat_bulk - after.rhat_bulk) < 1e-12 && std::abs(before.ess_bulk - after.ess_bulk) < 1e-9
          && std::abs(before.ess_tail - after.ess_tail) < 1e-9,
          "bulk R-hat and ESS invariant under exp (differences %.1e, %.1e)",
          std::abs(before.rhat_bulk - after.rhat_bulk), std::abs(before.ess_bulk - after.ess_bulk));

    const Chains same(4, std::vector<double>(100, 2.5));
    Chains apart {same};
    apart[1].assign(100, 3.5);
    const measurement::ConvergenceResult cst {measurement::diagnose(same)}, ap {measurement::diagnose(apart)};
    const measurement::ConvergenceResult shortc {measurement::diagnose(Chains(4, std::vector<double>(7, 1.)))};
    check(cst.rhat == 1. && cst.ess_bulk == 400. && std::isinf(ap.rhat) && std::isnan(shortc.rhat),
          "constant chains: R-hat 1, ESS S; constant at different values: R-hat inf; fewer than 8 draws: NaN");

    const measurement::ConvergenceResult one {measurement::diagnose(normal_chains(9, 1, N))};
    check(one.rhat < 1.02 && one.chains == 2, "a single chain (split in two halves): R-hat %.4f", one.rhat);
    return check.exit_code();
}
