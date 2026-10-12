#include "measurements/qp_weights.hpp"
#include <array>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <span>
#include <stdexcept>
#include "comp_method/weight_computation.hpp"

qp_weights_measurement::qp_weights_measurement(diagram_cfg * cfg, std::array<double, 3> p, double tau_min, bool tau_fixed,
                                               std::size_t n_batches)
    : cfg(cfg),
      p(p),
      tau_min(tau_fixed ? -1. : tau_min),
      tau_fixed(tau_fixed),
      n_max(cfg->max_order_ext / 2),
      U_p(weight::LKMatrix::diagonalizeLKHamiltonian(p).block<3,3>(1,0)),
      acc(static_cast<acc_type::size_type>(cfg->max_order_ext / 2 + 12), n_batches),
      hits(static_cast<std::size_t>(cfg->max_order_ext / 2 + 1), 0)
{
    assert(cfg != nullptr);
}

void qp_weights_measurement::measure(){
    if (!tau_fixed && cfg->current_tau_length < tau_min) {
        return;
    }
    const Eigen::Matrix3d & P {cfg->diagram_head->right_component};
    const double trace {P.trace()};
    if (!std::isfinite(trace) || trace == 0.) {     // a zero trace has probability zero
        return;
    }
    const double s {trace < 0. ? -1. : 1.};
    const int N {cfg->external_ph_manager->current_length / 2};
    assert(N >= 0 && N <= n_max);

    // one sparse sample, indices increasing: the N slot, s N, the residue matrix when N = 0, s
    std::array<double, 12> vals {};
    std::array<acc_type::size_type, 12> idxs {};
    std::size_t k {0};

    vals[k] = s;
    idxs[k] = static_cast<acc_type::size_type>(N);
    ++k;

    if (N == 0) {
        // P / |tr P| carries the sign; rotated to the orbital basis and symmetrised
        const Eigen::Matrix3d & U {cfg->diagram_head->baseWF};
        const Eigen::Matrix3d M {U * P * U.transpose() / std::abs(trace)};
        for (int i {0}; i < 3; ++i) {
            for (int j {0}; j < 3; ++j) {
                vals[k] = 0.5 * (M(i, j) + M(j, i));
                idxs[k] = static_cast<acc_type::size_type>(indexResidue() + 3*i + j);
                ++k;
            }
        }
    }
    else {
        vals[k] = s * N;
        idxs[k] = static_cast<acc_type::size_type>(indexMeanN());
        ++k;
    }

    vals[k] = s;
    idxs[k] = static_cast<acc_type::size_type>(indexSign());
    ++k;

    acc.accumulate(std::span(vals.data(), k), std::span(idxs.data(), k));
    ++hits[static_cast<std::size_t>(N)];
    ++n_measured;
    if (s < 0.) { ++n_negative; }
}

Eigen::VectorXd qp_weights_measurement::ratios(const Eigen::VectorXd & x) const {
    const double sign {x(indexSign())};
    Eigen::VectorXd out(n_max + 1 + 1 + 9 + 9);
    out.head(n_max + 1) = x.head(n_max + 1) / sign;
    out(n_max + 1) = x(indexMeanN()) / sign;

    Eigen::Matrix3d R_orb;
    for (int i {0}; i < 3; ++i) {
        for (int j {0}; j < 3; ++j) {
            R_orb(i, j) = x(indexResidue() + 3*i + j) / sign;
        }
    }
    const Eigen::Matrix3d R_band {U_p.transpose() * R_orb * U_p};
    for (int i {0}; i < 3; ++i) {
        for (int j {0}; j < 3; ++j) {
            out(n_max + 2 + 3*i + j) = R_orb(i, j);
            out(n_max + 11 + 3*i + j) = R_band(i, j);
        }
    }
    return out;
}

qp_weights_measurement::result qp_weights_measurement::normalised() const {
    const auto jk {jackknifeRatios()};
    const Eigen::VectorXd mean {jk.naive_mean()};    // the estimator applied to the full-run means
    const Eigen::VectorXd err {jk.stderror()};

    result r;
    r.Z.resize(static_cast<std::size_t>(n_max + 1));
    r.Z_error.resize(static_cast<std::size_t>(n_max + 1));
    for (int N {0}; N <= n_max; ++N) {
        r.Z[static_cast<std::size_t>(N)] = mean(N);
        r.Z_error[static_cast<std::size_t>(N)] = err(N);
    }
    r.mean_N = mean(n_max + 1);
    r.mean_N_error = err(n_max + 1);
    for (int i {0}; i < 3; ++i) {
        for (int j {0}; j < 3; ++j) {
            r.R_orb(i, j) = mean(n_max + 2 + 3*i + j);
            r.R_orb_error(i, j) = err(n_max + 2 + 3*i + j);
            r.R_band(i, j) = mean(n_max + 11 + 3*i + j);
            r.R_band_error(i, j) = err(n_max + 11 + 3*i + j);
        }
    }
    return r;
}

void qp_weights_measurement::write(const std::string & path) const {
    const result r {normalised()};
    FILE * f {std::fopen(path.c_str(), "w")};
    if (f == nullptr) { throw std::runtime_error("qp_weights_measurement::write: cannot open " + path); }

    std::fprintf(f, "# quasiparticle residue and N-phonon weights (trace over all boundary states)\n");
    std::fprintf(f, "# p = %.10f %.10f %.10f   %s", p[0], p[1], p[2], tau_fixed ? "tau_fixed\n" : "");
    if (!tau_fixed) { std::fprintf(f, "tau_min = %.10f\n", tau_min); }
    std::fprintf(f, "# n_samples = %llu   n_measured = %llu   n_negative = %llu   batches = %zu x %llu samples\n",
                 static_cast<unsigned long long>(acc.count()), static_cast<unsigned long long>(n_measured),
                 static_cast<unsigned long long>(n_negative), acc.batches().size(),
                 static_cast<unsigned long long>(acc.batch_count()));
    std::fprintf(f, "# <N> = %.15Le +- %.6Le\n", r.mean_N, r.mean_N_error);

    const auto matrix {[f](const char * name, const Eigen::Matrix3d & m, const Eigen::Matrix3d & e){
        std::fprintf(f, "# %s (rows: value err value err value err)\n", name);
        for (int i {0}; i < 3; ++i) {
            std::fprintf(f, "#   % .15e %.6e   % .15e %.6e   % .15e %.6e\n",
                         m(i, 0), e(i, 0), m(i, 1), e(i, 1), m(i, 2), e(i, 2));
        }
    }};
    matrix("residue matrix R, orbital basis x y z (tr R = Z_0)", r.R_orb, r.R_orb_error);
    matrix("residue matrix R, band basis at p", r.R_band, r.R_band_error);

    std::fprintf(f, "# columns: N hits Z_N err\n");
    for (int N {0}; N <= n_max; ++N) {
        const auto n {static_cast<std::size_t>(N)};
        std::fprintf(f, "%d %llu %.15Le %.6Le\n", N, static_cast<unsigned long long>(hits[n]), r.Z[n], r.Z_error[n]);
    }
    std::fclose(f);
}
