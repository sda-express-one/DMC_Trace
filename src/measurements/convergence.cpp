#include "measurements/convergence.hpp"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <limits>
#include <numeric>
#include <utility>

namespace measurement {

    namespace {
        constexpr double nan {std::numeric_limits<double>::quiet_NaN()};
        constexpr double inf {std::numeric_limits<double>::infinity()};

        std::vector<double> pooled(const std::vector<std::vector<double>> & chains){
            std::vector<double> all;
            for (const auto & c : chains) { all.insert(all.end(), c.begin(), c.end()); }
            return all;
        }

        // empirical quantile with linear interpolation between order statistics (R's default, type 7)
        double quantile(std::vector<double> v, double q){
            std::sort(v.begin(), v.end());
            const double h {(static_cast<double>(v.size()) - 1.) * q};
            const auto lo {static_cast<std::size_t>(std::floor(h))};
            const std::size_t hi {std::min(lo + 1, v.size() - 1)};
            return v[lo] + (h - static_cast<double>(lo)) * (v[hi] - v[lo]);
        }

        // length of the shortest chain (0 without chains): chains of different lengths are cut to it
        std::size_t shortestLength(const std::vector<std::vector<double>> & chains){
            if (chains.empty()) { return 0; }
            std::size_t n {std::numeric_limits<std::size_t>::max()};
            for (const auto & c : chains) { n = std::min(n, c.size()); }
            return n;
        }

        // exact for a constant vector (sum/n can be off by an ulp, which would make an exactly zero variance tiny)
        double mean(const std::vector<double> & v){
            if (std::all_of(v.begin(), v.end(), [&](double x){ return x == v.front(); })) { return v.front(); }
            return std::accumulate(v.begin(), v.end(), 0.) / static_cast<double>(v.size());
        }

        // autocovariances with divisor n (Geyer's biased estimate), lags 0 .. n - 1
        std::vector<double> autocovariance(const std::vector<double> & x){
            const std::size_t n {x.size()};
            const double m {mean(x)};
            std::vector<double> acov(n, 0.);
            for (std::size_t t {0}; t < n; ++t) {
                double s {0.};
                for (std::size_t i {0}; i + t < n; ++i) { s += (x[i] - m) * (x[i + t] - m); }
                acov[t] = s / static_cast<double>(n);
            }
            return acov;
        }
    }

    double inverseNormalCdf(double p){
        // M. J. Wichura, Algorithm AS 241 (PPND16), Applied Statistics 37, 477 (1988): piecewise rational
        // approximations, relative error ~1e-16 over the whole double range (the algorithm of R's qnorm). Central
        // region |p - 1/2| <= 0.425 in r = 0.180625 - q^2; tails in r = sqrt(-ln(min(p, 1 - p))), split at r = 5,
        // the upper tail by symmetry (1 - p is exact for p > 1/2).
        assert(p > 0. && p < 1.);
        const double q {p - 0.5};
        if (std::abs(q) <= 0.425) {
            const double r {0.180625 - q * q};
            return q * (((((((r * 2509.0809287301226727 + 33430.575583588128105) * r + 67265.770927008700853) * r
                             + 45921.953931549871457) * r + 13731.693765509461125) * r + 1971.5909503065514427) * r
                             + 133.14166789178437745) * r + 3.387132872796366608)
                     / (((((((r * 5226.495278852545925 + 28729.085735721942674) * r + 39307.89580009271061) * r
                             + 21213.794301586595867) * r + 5394.1960214247511077) * r + 687.1870074920579083) * r
                             + 42.313330701600911252) * r + 1.);
        }
        double r {std::sqrt(-std::log(q < 0. ? p : 1. - p))};
        double x {0.};
        if (r <= 5.) {
            r -= 1.6;
            x = (((((((r * 7.7454501427834140764e-4 + .0227238449892691845833) * r + .24178072517745061177) * r
                      + 1.27045825245236838258) * r + 3.64784832476320460504) * r + 5.7694972214606914055) * r
                      + 4.6303378461565452959) * r + 1.42343711074968357734)
              / (((((((r * 1.05075007164441684324e-9 + 5.475938084995344946e-4) * r + .0151986665636164571966) * r
                      + .14810397642748007459) * r + .68976733498510000455) * r + 1.6763848301838038494) * r
                      + 2.05319162663775882187) * r + 1.);
        }
        else {
            r -= 5.;
            x = (((((((r * 2.01033439929228813265e-7 + 2.71155556874348757815e-5) * r + .0012426609473880784386) * r
                      + .026532189526576123093) * r + .29656057182850489123) * r + 1.7848265399172913358) * r
                      + 5.4637849111641143699) * r + 6.6579046435011037772)
              / (((((((r * 2.04426310338993978564e-15 + 1.4215117583164458887e-7) * r + 1.8463183175100546818e-5) * r
                      + 7.868691311456132591e-4) * r + .0148753612908506148525) * r + .13692988092273580531) * r
                      + .59983220655588793769) * r + 1.);
        }
        return q < 0. ? -x : x;
    }

    std::vector<std::vector<double>> splitChains(const std::vector<std::vector<double>> & chains){
        const std::size_t n {shortestLength(chains)};
        const std::size_t half {n / 2};
        std::vector<std::vector<double>> split;
        for (const auto & c : chains) {
            split.emplace_back(c.begin(), c.begin() + static_cast<std::ptrdiff_t>(half));
            // the second half ends at n; an odd middle draw (index half) is dropped
            split.emplace_back(c.begin() + static_cast<std::ptrdiff_t>(n - half), c.begin() + static_cast<std::ptrdiff_t>(n));
        }
        return split;
    }

    std::vector<std::vector<double>> rankNormalize(const std::vector<std::vector<double>> & chains){
        // (value, chain, position), sorted by value; equal values share their average rank
        std::vector<std::pair<double, std::pair<std::size_t, std::size_t>>> all;
        for (std::size_t m {0}; m < chains.size(); ++m) {
            for (std::size_t i {0}; i < chains[m].size(); ++i) { all.push_back({chains[m][i], {m, i}}); }
        }
        std::sort(all.begin(), all.end(), [](const auto & x, const auto & y){ return x.first < y.first; });
        const double S {static_cast<double>(all.size())};
        std::vector<std::vector<double>> z(chains.size());
        for (std::size_t m {0}; m < chains.size(); ++m) { z[m].resize(chains[m].size()); }
        for (std::size_t i {0}; i < all.size();) {
            std::size_t j {i};
            while (j < all.size() && all[j].first == all[i].first) { ++j; }
            const double rank {0.5 * (static_cast<double>(i + 1) + static_cast<double>(j))};     // average of i+1 .. j
            const double score {inverseNormalCdf((rank - 0.375) / (S + 0.25))};
            for (std::size_t k {i}; k < j; ++k) { z[all[k].second.first][all[k].second.second] = score; }
            i = j;
        }
        return z;
    }

    double splitRhat(const std::vector<std::vector<double>> & split){
        const std::size_t M {split.size()}, n {split[0].size()};
        std::vector<double> means(M), vars(M);
        for (std::size_t m {0}; m < M; ++m) {
            means[m] = mean(split[m]);
            double s {0.};
            for (const double x : split[m]) { s += (x - means[m]) * (x - means[m]); }
            vars[m] = s / static_cast<double>(n - 1);
        }
        const double grand {mean(means)};
        double B {0.};
        for (const double mu : means) { B += (mu - grand) * (mu - grand); }
        B *= static_cast<double>(n) / static_cast<double>(M - 1);
        const double W {mean(vars)};
        if (W == 0.) { return B == 0. ? 1. : inf; }
        const double var_plus {(static_cast<double>(n) - 1.) / static_cast<double>(n) * W + B / static_cast<double>(n)};
        return std::sqrt(var_plus / W);
    }

    double essMean(const std::vector<std::vector<double>> & split){
        // multi-chain autocorrelation (Eq. 10) with Geyer's initial positive and monotone sequences (Eqs. 12-13),
        // as in Stan / the posterior package
        const std::size_t M {split.size()}, n {split[0].size()};
        const double S {static_cast<double>(M * n)};
        std::vector<std::vector<double>> acov;
        std::vector<double> means;
        for (const auto & c : split) { acov.push_back(autocovariance(c)); means.push_back(mean(c)); }
        const auto mean_acov {[&](std::size_t t){ double s {0.}; for (const auto & a : acov) { s += a[t]; } return s / static_cast<double>(M); }};

        const double mean_var {mean_acov(0) * static_cast<double>(n) / (static_cast<double>(n) - 1.)};
        double var_plus {mean_var * (static_cast<double>(n) - 1.) / static_cast<double>(n)};
        if (M > 1) {
            const double grand {mean(means)};
            double v {0.};
            for (const double mu : means) { v += (mu - grand) * (mu - grand); }
            var_plus += v / static_cast<double>(M - 1);
        }
        if (var_plus == 0.) { return S; }       // constant everywhere: fully determined

        std::vector<double> rho(n, 0.);
        rho[0] = 1.;
        double rho_even {1.}, rho_odd {1. - (mean_var - mean_acov(1)) / var_plus};
        rho[1] = rho_odd;
        std::size_t t {1};
        while (t + 2 < n - 4 && rho_even + rho_odd > 0.) {
            rho_even = 1. - (mean_var - mean_acov(t + 1)) / var_plus;
            rho_odd = 1. - (mean_var - mean_acov(t + 2)) / var_plus;
            if (rho_even + rho_odd >= 0.) {
                rho[t + 1] = rho_even;
                rho[t + 2] = rho_odd;
            }
            t += 2;
        }
        const std::size_t max_t {t};
        if (rho_even > 0. && max_t + 1 < n) { rho[max_t + 1] = rho_even; }     // antithetic chains: improves the estimate

        // initial monotone sequence
        for (std::size_t u {1}; u + 2 <= max_t; u += 2) {
            if (rho[u + 1] + rho[u + 2] > rho[u - 1] + rho[u]) {
                rho[u + 1] = 0.5 * (rho[u - 1] + rho[u]);
                rho[u + 2] = rho[u + 1];
            }
        }
        double tau {-1.};
        for (std::size_t u {0}; u <= max_t; ++u) { tau += 2. * rho[u]; }
        if (max_t + 1 < n) { tau += rho[max_t + 1]; }
        tau = std::max(tau, 1. / std::log10(S));
        return S / tau;
    }

    ConvergenceResult diagnose(const std::vector<std::vector<double>> & chains){
        ConvergenceResult r;
        if (shortestLength(chains) < 8) {          // also no chains at all (length 0)
            r.rhat_bulk = r.rhat_fold = r.rhat = r.ess_bulk = r.ess_tail = nan;
            return r;
        }
        const std::vector<std::vector<double>> split {splitChains(chains)};
        r.chains = static_cast<int>(split.size());
        r.draws = static_cast<int>(split[0].size());

        // bulk: rank-normalised draws
        const std::vector<std::vector<double>> z {rankNormalize(split)};
        r.rhat_bulk = splitRhat(z);
        r.ess_bulk = essMean(z);

        // fold: rank-normalised |theta - median|
        const std::vector<double> all {pooled(split)};
        const double med {quantile(all, 0.5)};
        std::vector<std::vector<double>> folded {split};
        for (auto & c : folded) { for (double & x : c) { x = std::abs(x - med); } }
        r.rhat_fold = splitRhat(rankNormalize(folded));
        r.rhat = std::max(r.rhat_bulk, r.rhat_fold);

        // tail: ESS of the 5% and 95% quantile indicators
        double tail {inf};
        for (const double q : {0.05, 0.95}) {
            const double cut {quantile(all, q)};
            std::vector<std::vector<double>> ind {split};
            for (auto & c : ind) { for (double & x : c) { x = x <= cut ? 1. : 0.; } }
            tail = std::min(tail, essMean(ind));
        }
        r.ess_tail = tail;
        return r;
    }
}
