#include "measurements/green_laguerre.hpp"
#include <mpi.h>
#include <simplemc/mpi/all_reduce.hpp>
#include <simplemc/mpi/communicator.hpp>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <span>
#include <Eigen/Eigenvalues>
#include <stdexcept>
#include "measurements/measurement_utils.hpp"

namespace {
    // 2 min_n (E_n(p) - mu); 1 where that is not positive (mu = 0 at p = 0: a fixed-length run, which builds every
    // estimator but where run_input rejects this one as active)
    double autoScale(const std::array<double, 3> & p, double chem_pot){
        const double s {2. * measurement::order0MinRate(p, chem_pot)};
        return s > 0. ? s : 1.;
    }
}

green_laguerre_measurement::green_laguerre_measurement(diagram_cfg * cfg, std::array<double, 3> p, int l_max, double scale,
                                                       int n_grid, int boundary_phonons, std::size_t n_batches)
    : cfg(cfg),
      p(p),
      boundary_phonons(boundary_phonons),
      l_max(l_max),
      scale(scale > 0. ? scale : autoScale(p, cfg->chem_pot)),
      n_grid(n_grid),
      tau_max(cfg->tau_max),
      acc(static_cast<acc_type::size_type>(l_max + 2), n_batches),
      psi(static_cast<std::size_t>(l_max + 1)),
      sample(static_cast<std::size_t>(l_max + 2)),
      basis(n_grid, l_max + 1)
{
    assert(cfg != nullptr);
    assert(l_max >= 0 && n_grid > 0 && this->scale > 0.);

    // phi_l at the grid points, once
    const double width {tau_max / n_grid};
    const double root_s {std::sqrt(this->scale)};
    for (int j {0}; j < n_grid; ++j) {
        measurement::laguerreFunctions(this->scale * static_cast<double>(measurement::binCentre(j, width)), psi);
        for (int l {0}; l <= l_max; ++l) { basis(j, l) = root_s * psi[static_cast<std::size_t>(l)]; }
    }
}

void green_laguerre_measurement::measure(){
    std::fill(sample.begin(), sample.end(), 0.);

    const int n_internal {cfg->internal_ph_manager->current_length / 2};
    const int n_external {cfg->external_ph_manager->current_length / 2};

    if (boundary_phonons < 0 || n_external == boundary_phonons) {
        const double trace {cfg->diagram_head->right_component.trace()};
        if (std::isfinite(trace) && trace != 0.) {        // a zero trace has probability zero
            const bool negative {trace < 0.};
            const double factor {negative ? -std::sqrt(scale) : std::sqrt(scale)};
            measurement::laguerreFunctions(scale * cfg->current_tau_length, psi);
            for (std::size_t l {0}; l < psi.size(); ++l) { sample[l] = factor * psi[l]; }
            ++n_selected;
            if (negative) { ++n_negative; }
        }
    }

    // counted whatever the selection: the normalisation always refers to the order-0 diagram
    if (n_internal == 0 && n_external == 0) {
        sample.back() = 1.;
        ++n_order0;
    }

    // every measure() is one sample, possibly contributing zero to all entries
    acc.accumulate(sample);
}

Eigen::VectorXd green_laguerre_measurement::unscaled(const Eigen::VectorXd & x) const {
    const double norm {static_cast<double>(measurement::order0Integral(p, cfg->chem_pot, tau_max)) / x(l_max + 1)};
    Eigen::VectorXd out(l_max + 1 + n_grid);
    out.head(l_max + 1) = norm * x.head(l_max + 1);
    out.tail(n_grid) = basis * out.head(l_max + 1);
    return out;
}

green_laguerre_measurement::result green_laguerre_measurement::normalised() const {
    const auto jk {jackknifeUnscaled()};
    const Eigen::VectorXd mean {jk.naive_mean()};    // the estimator applied to the full-run means
    const Eigen::VectorXd err {jk.stderror()};

    result r;
    for (int l {0}; l <= l_max; ++l) {
        r.coeff.push_back(mean(l));
        r.coeff_error.push_back(err(l));
    }
    // the coefficients are unscaled: no e^{-mu tau} factor
    r.coeff_cov = jk.covariance().topLeftCorner(l_max + 1, l_max + 1);
    const double width {tau_max / n_grid};
    for (int j {0}; j < n_grid; ++j) {
        const long double tau {measurement::binCentre(j, width)};
        const long double s {std::exp(-static_cast<long double>(cfg->chem_pot) * tau)};
        r.tau.push_back(tau);
        r.trace.push_back(s * mean(l_max + 1 + j));
        r.trace_error.push_back(s * err(l_max + 1 + j));
    }
    return r;
}

void green_laguerre_measurement::write(const std::string & path) const {
    const result r {normalised()};
    FILE * f {std::fopen(path.c_str(), "w")};
    if (f == nullptr) { throw std::runtime_error("green_laguerre_measurement::write: cannot open " + path); }

    std::fprintf(f, "# Green function trace estimator, Laguerre coefficients of g(tau) = tr G(p, tau) e^{mu tau}\n");
    std::fprintf(f, "# phi_l(tau) = sqrt(s) e^{-s tau/2} L_l(s tau);  tr G(tau) = e^{-mu tau} sum_l g_l phi_l(tau)\n");
    std::fprintf(f, "# p = %.10f %.10f %.10f   boundary_phonons = %d%s\n", p[0], p[1], p[2], boundary_phonons,
                 boundary_phonons == 0 ? " (vacuum sector: tr G(p,tau))" : boundary_phonons < 0 ? " (all: P function)" : "");
    std::fprintf(f, "# tau_max = %.10f   mu = %.10f   s = %.10e   l_max = %d   n_grid = %d   Delta_0 tau_max = %.6e\n", tau_max,
                 cfg->chem_pot, scale, l_max, n_grid, measurement::order0MinRate(p, cfg->chem_pot) * tau_max);
    std::fprintf(f, "# n_samples = %llu   n_selected = %llu   n_negative = %llu   n_order0 = %llu   Z_0 = %.18Le   batches = %zu x %llu samples\n",
                 static_cast<unsigned long long>(acc.count()), static_cast<unsigned long long>(n_selected),
                 static_cast<unsigned long long>(n_negative), static_cast<unsigned long long>(n_order0),
                 measurement::order0Integral(p, cfg->chem_pot, tau_max), acc.batches().size(),
                 static_cast<unsigned long long>(acc.batch_count()));
    std::fprintf(f, "# block 0 columns: l g_l err\n");
    for (std::size_t l {0}; l < r.coeff.size(); ++l) {
        std::fprintf(f, "%zu %.15Le %.6Le\n", l, r.coeff[l], r.coeff_error[l]);
    }
    std::fprintf(f, "\n\n# block 1 columns: tau trace err   (rebuilt from l <= l_max)\n");
    for (std::size_t j {0}; j < r.tau.size(); ++j) {
        std::fprintf(f, "%.10Le %.15Le %.6Le\n", r.tau[j], r.trace[j], r.trace_error[j]);
    }
    std::fclose(f);

    const std::string cov_path {covariancePath(path)};
    FILE * c {std::fopen(cov_path.c_str(), "w")};
    if (c == nullptr) { throw std::runtime_error("green_laguerre_measurement::write: cannot open " + cov_path); }
    const std::size_t n_b {acc.batches().size()};
    const int p_coeff {l_max + 1};
    std::fprintf(c, "# jackknife covariance of the Laguerre coefficients g_l of g(tau) = tr G(p, tau) e^{mu tau}"
                    " (normalisation as in the main file)\n");
    std::fprintf(c, "# rows and columns l = 0 .. %d; the diagonal is the square of the main file's err column\n", l_max);
    std::fprintf(c, "# p = %.10f %.10f %.10f   boundary_phonons = %d   s = %.10e   l_max = %d\n", p[0], p[1], p[2],
                 boundary_phonons, scale, l_max);
    std::fprintf(c, "# batches = %zu x %llu samples\n", n_b, static_cast<unsigned long long>(acc.batch_count()));
    std::fprintf(c, "# sample estimate with N_b - 1 degrees of freedom: for chi^2 with the first q coefficients use"
                    " C_q^{-1} (N_b - q - 2)/(N_b - 1) (Hartlap)\n");
    if (static_cast<std::size_t>(p_coeff) + 2 >= n_b) {
        std::fprintf(c, "# warning: l_max + 1 = %d coefficients vs %zu batches: the full matrix is singular or nearly so"
                        " (rank <= N_b - 1); invert only a block of q << N_b coefficients, or use more batches\n",
                     p_coeff, n_b);
    }
    std::fprintf(c, "# numerical rank = %d of %d (eigenvalues > 1e-12 x the largest): every sample lies on the curve"
                    " sign sqrt(s) psi(s tau_D), so the high coefficients are nearly linear combinations of the low ones;"
                    " invert a leading block or use a truncated SVD\n", numericalRank(r.coeff_cov), p_coeff);
    for (int l {0}; l < p_coeff; ++l) {
        for (int m {0}; m < p_coeff; ++m) { std::fprintf(c, m == 0 ? "%.15e" : " %.15e", r.coeff_cov(l, m)); }
        std::fprintf(c, "\n");
    }
    std::fclose(c);
}

int green_laguerre_measurement::numericalRank(const Eigen::MatrixXd & C){
    if (C.size() == 0) { return 0; }
    const Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es {C, Eigen::EigenvaluesOnly};
    const Eigen::VectorXd & ev {es.eigenvalues()};
    const double top {ev(ev.size() - 1)};
    if (!(top > 0.)) { return 0; }
    return static_cast<int>((ev.array() > 1e-12 * top).count());
}

std::string green_laguerre_measurement::covariancePath(const std::string & path){
    const std::string ext {".dat"};
    if (path.size() >= ext.size() && path.compare(path.size() - ext.size(), ext.size(), ext) == 0) {
        return path.substr(0, path.size() - ext.size()) + "_cov" + ext;
    }
    return path + "_cov";
}

void simplemc_mpi_collect(const simplemc::mpi::communicator & comm, green_laguerre_measurement & m){
    simplemc_mpi_collect(comm, m.acc);
    simplemc::mpi::all_reduce_in_place(m.n_selected, MPI_SUM, comm);
    simplemc::mpi::all_reduce_in_place(m.n_negative, MPI_SUM, comm);
    simplemc::mpi::all_reduce_in_place(m.n_order0, MPI_SUM, comm);
}
