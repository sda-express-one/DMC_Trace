#include "measurements/qp_weights.hpp"
#include <mpi.h>
#include <simplemc/mpi/all_reduce.hpp>
#include <simplemc/mpi/communicator.hpp>
#include <array>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <span>
#include <stdexcept>
#include "comp_method/weight_computation.hpp"

qp_weights_measurement::qp_weights_measurement(diagram_cfg * cfg, std::array<double, 3> p, double tau_min, bool tau_fixed,
                                               std::size_t n_batches, bool sign_free)
    : cfg(cfg),
      p(p),
      tau_min(tau_fixed ? -1. : tau_min),
      tau_fixed(tau_fixed),
      n_max(cfg->max_order_ext / 2),
      U_p(weight::LKMatrix::diagonalizeLKHamiltonian(p).block<3,3>(1,0)),
      sign_free(sign_free),
      acc(static_cast<acc_type::size_type>((cfg->max_order_ext / 2 + 12) + (sign_free ? cfg->max_order_ext / 2 + 11 : 0)), n_batches),
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

    // one sparse sample, indices increasing: the N slot, s N, the residue matrix when N = 0, s, then the same
    // three without the sign
    std::array<double, 23> vals {};
    std::array<acc_type::size_type, 23> idxs {};
    std::size_t k {0};
    Eigen::Matrix3d M_sym {Eigen::Matrix3d::Zero()};

    vals[k] = s;
    idxs[k] = static_cast<acc_type::size_type>(N);
    ++k;

    if (N == 0) {
        // P / |tr P| carries the sign; rotated to the orbital basis and symmetrised
        const Eigen::Matrix3d & U {cfg->diagram_head->baseWF};
        const Eigen::Matrix3d M {U * P * U.transpose() / std::abs(trace)};
        M_sym = 0.5 * (M + M.transpose());
        for (int i {0}; i < 3; ++i) {
            for (int j {0}; j < 3; ++j) {
                vals[k] = M_sym(i, j);
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

    if (sign_free) {
        // U P U^T / tr P = s (U P U^T / |tr P|)
        const int base {indexSignFree()};
        vals[k] = 1.;
        idxs[k] = static_cast<acc_type::size_type>(base + N);
        ++k;
        if (N == 0) {
            for (int i {0}; i < 3; ++i) {
                for (int j {0}; j < 3; ++j) {
                    vals[k] = s * M_sym(i, j);
                    idxs[k] = static_cast<acc_type::size_type>(base + n_max + 2 + 3*i + j);
                    ++k;
                }
            }
        }
        else {
            vals[k] = N;
            idxs[k] = static_cast<acc_type::size_type>(base + n_max + 1);
            ++k;
        }
    }

    acc.accumulate(std::span(vals.data(), k), std::span(idxs.data(), k));
    ++hits[static_cast<std::size_t>(N)];
    ++n_measured;
    if (s < 0.) { ++n_negative; }
}

Eigen::VectorXd qp_weights_measurement::ratios(const Eigen::VectorXd & x) const {
    Eigen::VectorXd out(sign_free ? 2 * outputs() : outputs());
    // one block of estimates [Z_0..Z_{n_max}, <N>, R_orb, R_band] from the means starting at `in`, divided by `norm`
    const auto block {[&](int in, double norm, int at){
        out.segment(at, n_max + 1) = x.segment(in, n_max + 1) / norm;
        out(at + n_max + 1) = x(in + n_max + 1) / norm;
        Eigen::Matrix3d R_orb;
        for (int i {0}; i < 3; ++i) {
            for (int j {0}; j < 3; ++j) {
                R_orb(i, j) = x(in + n_max + 2 + 3*i + j) / norm;
            }
        }
        const Eigen::Matrix3d R_band {U_p.transpose() * R_orb * U_p};
        for (int i {0}; i < 3; ++i) {
            for (int j {0}; j < 3; ++j) {
                out(at + n_max + 2 + 3*i + j) = R_orb(i, j);
                out(at + n_max + 11 + 3*i + j) = R_band(i, j);
            }
        }
    }};
    block(0, x(indexSign()), 0);
    if (sign_free) { block(indexSignFree(), 1., outputs()); }     // plain means: the |W| ensemble
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
    if (sign_free) {
        const int o {outputs()};
        r.Z_sf.resize(static_cast<std::size_t>(n_max + 1));
        r.Z_sf_error.resize(static_cast<std::size_t>(n_max + 1));
        for (int N {0}; N <= n_max; ++N) {
            r.Z_sf[static_cast<std::size_t>(N)] = mean(o + N);
            r.Z_sf_error[static_cast<std::size_t>(N)] = err(o + N);
        }
        r.mean_N_sf = mean(o + n_max + 1);
        r.mean_N_sf_error = err(o + n_max + 1);
        for (int i {0}; i < 3; ++i) {
            for (int j {0}; j < 3; ++j) {
                r.R_orb_sf(i, j) = mean(o + n_max + 2 + 3*i + j);
                r.R_orb_sf_error(i, j) = err(o + n_max + 2 + 3*i + j);
                r.R_band_sf(i, j) = mean(o + n_max + 11 + 3*i + j);
                r.R_band_sf_error(i, j) = err(o + n_max + 11 + 3*i + j);
            }
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
    if (sign_free) {
        std::fprintf(f, "# sign-free values (*_sf): plain means over the sampled |W| ensemble, no ratio to <s>\n");
        std::fprintf(f, "# <N>_sf = %.15Le +- %.6Le\n", r.mean_N_sf, r.mean_N_sf_error);
    }

    const auto matrix {[f](const char * name, const Eigen::Matrix3d & m, const Eigen::Matrix3d & e){
        std::fprintf(f, "# %s (rows: value err value err value err)\n", name);
        for (int i {0}; i < 3; ++i) {
            std::fprintf(f, "#   % .15e %.6e   % .15e %.6e   % .15e %.6e\n",
                         m(i, 0), e(i, 0), m(i, 1), e(i, 1), m(i, 2), e(i, 2));
        }
    }};
    matrix("residue matrix R, orbital basis x y z (tr R = Z_0)", r.R_orb, r.R_orb_error);
    matrix("residue matrix R, band basis at p", r.R_band, r.R_band_error);
    if (sign_free) {
        matrix("sign-free residue matrix R_sf, orbital basis x y z (tr R_sf = Z_0_sf)", r.R_orb_sf, r.R_orb_sf_error);
        matrix("sign-free residue matrix R_sf, band basis at p", r.R_band_sf, r.R_band_sf_error);
    }

    std::fprintf(f, sign_free ? "# columns: N hits Z_N err Z_N_sf err\n" : "# columns: N hits Z_N err\n");
    for (int N {0}; N <= n_max; ++N) {
        const auto n {static_cast<std::size_t>(N)};
        std::fprintf(f, "%d %llu %.15Le %.6Le", N, static_cast<unsigned long long>(hits[n]), r.Z[n], r.Z_error[n]);
        if (sign_free) { std::fprintf(f, " %.15Le %.6Le", r.Z_sf[n], r.Z_sf_error[n]); }
        std::fprintf(f, "\n");
    }
    std::fclose(f);
}

void simplemc_mpi_collect(const simplemc::mpi::communicator & comm, qp_weights_measurement & m){
    simplemc_mpi_collect(comm, m.acc);
    simplemc::mpi::all_reduce_in_place(m.hits, MPI_SUM, comm);
    simplemc::mpi::all_reduce_in_place(m.n_measured, MPI_SUM, comm);
    simplemc::mpi::all_reduce_in_place(m.n_negative, MPI_SUM, comm);
}
