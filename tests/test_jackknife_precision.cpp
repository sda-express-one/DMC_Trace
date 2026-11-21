// Precision of simplemc's delete-one jackknife error for the two variance algorithms, at the precisions of real runs.
//
// simplemc::jackknife accumulates the N delete-one replicates f_[b] in a covariance accumulator. With the default
// varalg::standard this is the one-pass sum C = sum f^2 - (sum f)^2 / N without a shift. The replicates lie within
// ~ err / sqrt(N) of their mean m, so C ~ err^2 cancels against sums ~ N m^2: the relative error of the variance is
// ~ eps N m^2 / err^2, i.e. the method fails once err / m drops below ~ sqrt(eps N) ~ 1e-6 (N ~ 1e4 batches).
// varalg::welford accumulates deviations from the running mean and stays accurate.
//
// Synthetic batch means x_b = m + sigma xi_b (xi standard normal, N = 15000 batches), for a grid of means m (Z_0 ~ 1,
// R_ii ~ 1/3, energies 1e-3 .. 1e-2 Ha) and relative errors err / m = 1e-3 .. 1e-8. The reference is a two-pass
// jackknife in long double. Checks: welford agrees with the reference to 1e-6 everywhere. The table of the
// standard algorithm's relative error is printed as a diagnostic (realistic precisions: gs_energy err/E ~ 1e-6 ..
// 1e-4, Z_0 1e-7 .. 1e-5, <s> ~ 1e-7, all with ~ 1.5e4 - 2.3e4 batches).
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>
#include <Eigen/Core>
#include <simplemc/accs/jackknife.hpp>
#include <simplemc/accs/varalg.hpp>
#include "test_common.hpp"
#include "measurements/qp_weights.hpp"

namespace {

// delete-one jackknife error of the mean, two-pass in long double
long double referenceError(const std::vector<double> & x){
    const auto n {static_cast<long double>(x.size())};
    long double sum {0.L};
    for (const double v : x) { sum += v; }
    std::vector<long double> f(x.size());
    long double fbar {0.L};
    for (std::size_t b {0}; b < x.size(); ++b) { f[b] = (sum - x[b]) / (n - 1.L); fbar += f[b]; }
    fbar /= n;
    long double ss {0.L};
    for (const long double v : f) { ss += (v - fbar) * (v - fbar); }
    return std::sqrt((n - 1.L) / n * ss);
}

template <simplemc::varalg A>
double simplemcError(const std::vector<Eigen::VectorXd> & samples){
    const auto jk {simplemc::jackknife<A>([](const Eigen::VectorXd & v){ return Eigen::VectorXd(v); }, samples)};
    return jk.stderror()(0);
}

}

int main(){
    test::Checks check {"jackknife precision: varalg::standard against welford"};

    constexpr std::size_t n_batches {15000};
    const std::vector<double> means {1., 0.3, 1e-2, 1e-3};
    const std::vector<double> rel_errors {1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8};

    std::mt19937_64 rng {20261121ULL};
    std::normal_distribution<double> xi {0., 1.};

    double worst_welford {0.};
    std::printf("  N = %zu batches; relative deviation of the jackknife error from the two-pass reference\n", n_batches);
    std::printf("  %8s %8s %14s %14s\n", "m", "err/m", "standard", "welford");
    for (const double m : means) {
        for (const double rel : rel_errors) {
            const double sigma {rel * m * std::sqrt(static_cast<double>(n_batches))};   // err of the mean = rel * m
            std::vector<double> x(n_batches);
            std::vector<Eigen::VectorXd> samples(n_batches, Eigen::VectorXd(1));
            for (std::size_t b {0}; b < n_batches; ++b) { x[b] = m + sigma * xi(rng); samples[b](0) = x[b]; }

            const long double ref {referenceError(x)};
            const double e_std {simplemcError<simplemc::varalg::standard>(samples)};
            const double e_wel {simplemcError<simplemc::varalg::welford>(samples)};
            const double d_std {static_cast<double>(std::abs(static_cast<long double>(e_std) - ref) / ref)};
            const double d_wel {static_cast<double>(std::abs(static_cast<long double>(e_wel) - ref) / ref)};
            worst_welford = std::max(worst_welford, std::isfinite(d_wel) ? d_wel : 1e300);
            std::printf("  %8.0e %8.0e %14.3e %14.3e%s\n", m, rel, d_std, d_wel, std::isfinite(e_std) ? "" : "   (standard: nan)");
        }
    }
    check(worst_welford < 1e-6, "welford error within 1e-6 of the two-pass reference on the whole grid (worst %.2e)", worst_welford);

    // qp_weights regression (the -nan Z_0 errors of the LK runs): every sample in the N = 0 sector with sign +1, so
    // Z_0 = Z_0_sf = 1 exactly with zero variance, and a residue matrix 1/3 + 1e-6 noise (relative error ~ 1e-8, the Z_0 precision of the runs).
    // With varalg::standard these errors came out as -nan or rounding noise.
    {
        test::setLK(test::AlAs);
        test::Diagram d {5ULL, {0., 0., 0.}, 40, 16};
        qp_weights_measurement Q {&d.cfg, {0., 0., 0.}, 0., true, 64, true};
        const int n_max {Q.n_max};
        const int i_res {n_max + 2}, i_sign {n_max + 11}, i_sf {n_max + 12};
        std::normal_distribution<double> noise {0., 1e-6};
        Eigen::VectorXd v {Eigen::VectorXd::Zero(static_cast<Eigen::Index>(Q.acc.size()))};
        for (int k {0}; k < (1 << 16); ++k) {
            Eigen::Matrix3d M {Eigen::Matrix3d::Identity() / 3.};
            for (int i {0}; i < 3; ++i) { for (int j {0}; j <= i; ++j) { M(i, j) += noise(rng); M(j, i) = M(i, j); } }
            v.setZero();
            v(0) = 1.;                  // N = 0, s = +1
            v(i_sign) = 1.;
            v(i_sf) = 1.;
            for (int i {0}; i < 3; ++i) {
                for (int j {0}; j < 3; ++j) { v(i_res + 3*i + j) = M(i, j); v(i_sf + n_max + 2 + 3*i + j) = M(i, j); }
            }
            Q.acc.accumulate(v);
            ++Q.n_measured;
            ++Q.hits[0];
        }
        const auto r {Q.normalised()};

        // reference: two-pass jackknife of the batch means of each diagonal residue entry (s = 1, so plain means)
        const auto batches {Q.acc.batches()};
        double worst {0.};
        bool finite {std::isfinite(static_cast<double>(r.Z_error[0])) && std::isfinite(static_cast<double>(r.Z_sf_error[0]))};
        for (int i {0}; i < 3; ++i) {
            std::vector<double> x;
            for (const auto & b : batches) { x.push_back(b.mean()(i_res + 4*i)); }
            const long double ref {referenceError(x)};
            for (const double e : {static_cast<double>(r.R_orb_error(i, i)), static_cast<double>(r.R_orb_sf_error(i, i))}) {
                finite = finite && std::isfinite(e);
                worst = std::max(worst, static_cast<double>(std::abs(static_cast<long double>(e) - ref) / ref));
            }
        }
        check(batches.size() >= 32, "qp_weights: %zu full batches", batches.size());
        check(finite, "qp_weights: Z_0, Z_0_sf and residue-diagonal errors are finite");
        check(r.Z_error[0] == 0.L && r.Z_sf_error[0] == 0.L, "qp_weights: Z_0 = %.17Lg with error %Lg (exactly 1, error 0)",
              r.Z[0], r.Z_error[0]);
        check(worst < 1e-6, "qp_weights: residue-diagonal errors within 1e-6 of the two-pass reference (worst %.2e)", worst);

        // values: naive_mean does not depend on varalg
        const auto jk_std {simplemc::jackknife([&Q](const Eigen::VectorXd & x){ return Q.ratios(x); }, Q.acc)};
        const auto jk_wel {Q.jackknifeRatios()};
        const Eigen::VectorXd e_std {jk_std.stderror()};
        std::printf("  qp_weights with varalg::standard (old): err Z_0 = %.3e, err R_xx = %.3e (welford %.3e)\n",
                    e_std(0), e_std(n_max + 2), static_cast<double>(r.R_orb_error(0, 0)));
        check((jk_std.naive_mean().array() == jk_wel.naive_mean().array()).all(),
              "qp_weights: values bitwise identical with standard and welford");
    }
    return check.exit_code();
}
