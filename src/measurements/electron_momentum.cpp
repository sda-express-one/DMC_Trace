#include "measurements/electron_momentum.hpp"
#include <mpi.h>
#include <simplemc/mpi/all_reduce.hpp>
#include <simplemc/mpi/communicator.hpp>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <limits>
#include <numbers>
#include <stdexcept>
#include "diagram/vertex.hpp"
#include "diagram/vertex_manager.hpp"

// ---------------------------------------------------------------------------------------------------------------------
// bounded map

int electron_momentum_measurement::bounded_map::index(double x) const {
    const double u {symmetric ? x / (std::abs(x) + scale) : x / (x + scale)};
    const double t {symmetric ? 0.5 * (u + 1.) : u};          // in [0, 1)
    const int i {static_cast<int>(std::floor(t * n))};
    return std::clamp(i, 0, n - 1);
}

double electron_momentum_measurement::bounded_map::edge(int i) const {
    const double u {symmetric ? -1. + 2. * i / n : static_cast<double>(i) / n};
    if (u >= 1.) { return std::numeric_limits<double>::infinity(); }
    if (u <= -1.) { return -std::numeric_limits<double>::infinity(); }
    return scale * u / (1. - std::abs(u));
}

double electron_momentum_measurement::bounded_map::center(int i) const {
    const double u {symmetric ? -1. + (2. * i + 1.) / n : (i + 0.5) / n};
    return scale * u / (1. - std::abs(u));
}

// ---------------------------------------------------------------------------------------------------------------------

electron_momentum_measurement::electron_momentum_measurement(diagram_cfg * cfg, std::array<double, 3> p, double tau_min,
                                                             bool tau_fixed, std::size_t n_batches, double k_scale,
                                                             int n_bins_3d, int n_bins_r, int n_max, std::size_t n_batches_3d)
    : cfg(cfg),
      p(p),
      tau_min(tau_fixed ? -1. : tau_min),
      tau_fixed(tau_fixed),
      n_max(n_max),
      n_bins_r(n_bins_r),
      n_bins_3d(n_bins_3d),
      k_scale(k_scale),
      acc(static_cast<size_type>((n_max + 1) * (10 + n_bins_r) + 1), n_batches),
      acc3d(static_cast<size_type>(n_bins_3d * n_bins_3d * n_bins_3d + 1), n_batches_3d)
{
    assert(cfg != nullptr);
    if (!(k_scale > 0.) || !std::isfinite(k_scale)) {
        throw std::invalid_argument("electron_momentum_measurement: k_scale must be finite and > 0");
    }
    if (n_max < 1 || n_bins_r < 2 || n_bins_3d < 2) {
        throw std::invalid_argument("electron_momentum_measurement: n_max >= 1, n_bins_r >= 2, n_bins_3d >= 2");
    }
}

std::vector<int> electron_momentum_measurement::segmentSectors() const {
    std::vector<int> out;
    int N {cfg->external_ph_manager->current_length / 2};
    for (const Vertex * v {cfg->diagram_head}; v != cfg->diagram_tail; v = v->next) {
        if (v != cfg->diagram_head) { N += v->type > 0 ? 1 : -1; }
        out.push_back(N);
    }
    return out;
}

namespace {
    // sort by index and merge equal indices
    void compact(std::vector<std::pair<electron_momentum_measurement::size_type, double>> & v){
        std::sort(v.begin(), v.end(), [](const auto & a, const auto & b){ return a.first < b.first; });
        std::size_t w {0};
        for (std::size_t r {0}; r < v.size(); ++r) {
            if (w > 0 && v[w - 1].first == v[r].first) { v[w - 1].second += v[r].second; }
            else { v[w++] = v[r]; }
        }
        v.resize(w);
    }
}

void electron_momentum_measurement::sample(std::vector<std::pair<size_type, double>> & out,
                                           std::vector<std::pair<size_type, double>> & out3d) const {
    out.clear();
    out3d.clear();
    const double trace {cfg->diagram_head->right_component.trace()};
    const double s {trace < 0. ? -1. : 1.};
    const double tau {cfg->current_tau_length};
    const int n_ext {cfg->external_ph_manager->current_length / 2};
    const bounded_map rmap {radialMap()}, amap {axisMap()};

    int N {n_ext};
    for (const Vertex * v {cfg->diagram_head}; v != cfg->diagram_tail; v = v->next) {
        if (v != cfg->diagram_head) { N += v->type > 0 ? 1 : -1; }
        assert(N >= 0);
        const double w {s * (v->tau_next - v->tau) / tau};
        const int sec {std::min(N, n_max)};
        const auto base {static_cast<size_type>(indexSector(sec))};
        const std::array<double, 3> & k {v->k};
        out.emplace_back(base, w);
        out.emplace_back(base + 1, w * k[0]);
        out.emplace_back(base + 2, w * k[1]);
        out.emplace_back(base + 3, w * k[2]);
        out.emplace_back(base + 4, w * k[0] * k[0]);
        out.emplace_back(base + 5, w * k[1] * k[1]);
        out.emplace_back(base + 6, w * k[2] * k[2]);
        out.emplace_back(base + 7, w * k[0] * k[1]);
        out.emplace_back(base + 8, w * k[0] * k[2]);
        out.emplace_back(base + 9, w * k[1] * k[2]);
        if (N > 0) {
            const double q[3] {k[0] - p[0], k[1] - p[1], k[2] - p[2]};
            const double r {std::sqrt(q[0]*q[0] + q[1]*q[1] + q[2]*q[2])};
            out.emplace_back(base + 10 + rmap.index(r), w);
            const int ix {amap.index(q[0])}, iy {amap.index(q[1])}, iz {amap.index(q[2])};
            out3d.emplace_back(static_cast<size_type>((ix * n_bins_3d + iy) * n_bins_3d + iz), w);
        }
    }
    assert(N == n_ext);
    out.emplace_back(static_cast<size_type>(indexSign()), s);
    out3d.emplace_back(static_cast<size_type>(nCells()), s);
    compact(out);
    compact(out3d);
}

void electron_momentum_measurement::measure(){
    if (!tau_fixed && cfg->current_tau_length < tau_min) {
        return;
    }
    const double trace {cfg->diagram_head->right_component.trace()};
    if (!std::isfinite(trace) || trace == 0.) {     // a zero trace has probability zero
        return;
    }
    static thread_local std::vector<std::pair<size_type, double>> pairs, pairs3d;
    static thread_local std::vector<size_type> idx;
    static thread_local std::vector<double> val;
    sample(pairs, pairs3d);

    const auto push {[](acc_type & a, const std::vector<std::pair<size_type, double>> & v){
        idx.resize(v.size());
        val.resize(v.size());
        for (std::size_t i {0}; i < v.size(); ++i) { idx[i] = v[i].first; val[i] = v[i].second; }
        a.accumulate(val, idx);
    }};
    push(acc, pairs);
    push(acc3d, pairs3d);
    ++n_measured;
    if (trace < 0.) { ++n_negative; }
}

electron_momentum_measurement::result electron_momentum_measurement::normalised() const {
    const auto jk {simplemc::jackknife<simplemc::varalg::welford>([this](const Eigen::VectorXd & x){ return ratios(x); }, acc)};
    const Eigen::VectorXd mean {jk.naive_mean()};
    const Eigen::VectorXd err {jk.stderror()};

    result r;
    const auto n {static_cast<std::size_t>(n_max + 1)};
    r.Z.resize(n); r.Z_error.resize(n);
    r.k.resize(n); r.k_error.resize(n);
    r.kk.resize(n); r.kk_error.resize(n);
    r.radial.assign(n, std::vector<double>(static_cast<std::size_t>(n_bins_r)));
    r.radial_error = r.radial;
    for (int N {0}; N <= n_max; ++N) {
        const auto i {static_cast<std::size_t>(N)};
        const int b {indexSector(N)};
        r.Z[i] = mean(b);
        r.Z_error[i] = err(b);
        for (int a {0}; a < 3; ++a) { r.k[i][static_cast<std::size_t>(a)] = mean(b + 1 + a); r.k_error[i][static_cast<std::size_t>(a)] = err(b + 1 + a); }
        for (int a {0}; a < 6; ++a) { r.kk[i][static_cast<std::size_t>(a)] = mean(b + 4 + a); r.kk_error[i][static_cast<std::size_t>(a)] = err(b + 4 + a); }
        for (int j {0}; j < n_bins_r; ++j) {
            r.radial[i][static_cast<std::size_t>(j)] = mean(b + 10 + j);
            r.radial_error[i][static_cast<std::size_t>(j)] = err(b + 10 + j);
        }
    }
    return r;
}

electron_momentum_measurement::result3d electron_momentum_measurement::jackknife3d(MPI_Comm comm) const {
    const bool mpi {comm != MPI_COMM_NULL};
    const auto nc {static_cast<Eigen::Index>(nCells())};
    const auto sum_ranks {[&](Eigen::VectorXd & v){
        if (mpi) { simplemc::mpi::all_reduce_in_place(v.data(), static_cast<int>(v.size()), MPI_SUM, comm); }
    }};

    // this rank's full batches; with several ranks equal counts, as simplemc's gather_batches: merge up to the largest
    // count, drop the batches of a rank that cannot reach it
    auto bs {acc3d.batches()};
    std::uint64_t count {bs.empty() ? 0 : acc3d.batch_count()};
    if (mpi) {
        std::uint64_t max_count {count};
        simplemc::mpi::all_reduce_in_place(max_count, MPI_MAX, comm);
        if (count > 0) { simplemc::merge_batches(static_cast<std::size_t>(max_count / count), bs); }
        if (bs.empty() || bs[0].count() != max_count) { bs.clear(); }
        count = max_count;
    }
    std::uint64_t B_total {bs.size()};
    if (mpi) { simplemc::mpi::all_reduce_in_place(B_total, MPI_SUM, comm); }
    if (B_total < 2) { throw std::runtime_error("electron_momentum_measurement: fewer than 2 full batches in the 3D accumulator"); }
    const double B {static_cast<double>(B_total)};

    // pass 1: the full-run means (equal batch counts)
    Eigen::VectorXd M {Eigen::VectorXd::Zero(nc + 1)};
    for (const auto & b : bs) { M += b.mean(); }
    sum_ranks(M);
    M /= B;

    // pass 2: the mean of the delete-one replicates f_b = x_b / s_b, x_b = (B M - m_b) / (B - 1); pass 3: their spread
    const auto replicate {[&](const Eigen::VectorXd & mb){
        const Eigen::VectorXd x {(B * M - mb) / (B - 1.)};
        return Eigen::VectorXd(x.head(nc) / x(nc));
    }};
    Eigen::VectorXd fbar {Eigen::VectorXd::Zero(nc)};
    for (const auto & b : bs) { fbar += replicate(b.mean()); }
    sum_ranks(fbar);
    fbar /= B;
    Eigen::VectorXd ss {Eigen::VectorXd::Zero(nc)};
    for (const auto & b : bs) { ss += (replicate(b.mean()) - fbar).array().square().matrix(); }
    sum_ranks(ss);

    result3d r;
    r.weight.resize(static_cast<std::size_t>(nc));
    r.error.resize(static_cast<std::size_t>(nc));
    for (Eigen::Index c {0}; c < nc; ++c) {
        r.weight[static_cast<std::size_t>(c)] = M(c) / M(nc);
        r.error[static_cast<std::size_t>(c)] = std::sqrt((B - 1.) / B * ss(c));
    }
    r.batches = static_cast<std::size_t>(B_total);
    r.batch_count = count;
    return r;
}

void electron_momentum_measurement::write(const std::string & path) const {
    const result r {normalised()};

    // totals over the sectors with errors: <k_a> and <|k - p|^2>
    const auto totals {simplemc::jackknife<simplemc::varalg::welford>([this](const Eigen::VectorXd & x){
        Eigen::Vector4d t {Eigen::Vector4d::Zero()};
        double Z {0.}, k2 {0.};
        for (int N {0}; N <= n_max; ++N) {
            const int b {indexSector(N)};
            Z += x(b);
            for (int a {0}; a < 3; ++a) { t(a) += x(b + 1 + a); }
            k2 += x(b + 4) + x(b + 5) + x(b + 6);
        }
        const double pk {p[0] * t(0) + p[1] * t(1) + p[2] * t(2)};
        const double p2 {p[0] * p[0] + p[1] * p[1] + p[2] * p[2]};
        t(3) = k2 - 2. * pk + p2 * Z;
        return Eigen::VectorXd(t / x(indexSign()));
    }, acc)};
    const Eigen::VectorXd tm {totals.naive_mean()}, te {totals.stderror()};

    FILE * f {std::fopen(path.c_str(), "w")};
    if (f == nullptr) { throw std::runtime_error("electron_momentum_measurement::write: cannot open " + path); }
    std::fprintf(f, "# electron momentum distribution, time-averaged over the diagram, per phonon sector N (signed)\n");
    std::fprintf(f, "# p = %.10f %.10f %.10f   %s", p[0], p[1], p[2], tau_fixed ? "tau_fixed\n" : "");
    if (!tau_fixed) { std::fprintf(f, "tau_min = %.10f\n", tau_min); }
    std::fprintf(f, "# k_s = %.10e   n_max = %d (last sector: N >= n_max)   n_bins_r = %d\n", k_scale, n_max, n_bins_r);
    std::fprintf(f, "# n_samples = %llu   n_measured = %llu   n_negative = %llu   batches = %zu x %llu samples\n",
                 static_cast<unsigned long long>(acc.count()), static_cast<unsigned long long>(n_measured),
                 static_cast<unsigned long long>(n_negative), acc.batches().size(),
                 static_cast<unsigned long long>(acc.batch_count()));
    std::fprintf(f, "# <k> = % .15e % .15e % .15e   err %.6e %.6e %.6e\n", tm(0), tm(1), tm(2), te(0), te(1), te(2));
    std::fprintf(f, "# <|k - p|^2> = %.15e +- %.6e\n", tm(3), te(3));
    std::fprintf(f, "# sector table: Z_N = time fraction with N phonons; <k_a>_N, <k_a k_b>_N not divided by Z_N\n");
    std::fprintf(f, "# columns: N Z err kx err ky err kz err kxx err kyy err kzz err kxy err kxz err kyz err\n");
    for (int N {0}; N <= n_max; ++N) {
        const auto i {static_cast<std::size_t>(N)};
        std::fprintf(f, "%d %.15e %.6e", N, r.Z[i], r.Z_error[i]);
        for (std::size_t a {0}; a < 3; ++a) { std::fprintf(f, " % .15e %.6e", r.k[i][a], r.k_error[i][a]); }
        for (std::size_t a {0}; a < 6; ++a) { std::fprintf(f, " % .15e %.6e", r.kk[i][a], r.kk_error[i][a]); }
        std::fprintf(f, "\n");
    }

    const bounded_map rm {radialMap()};
    std::fprintf(f, "\n# radial histogram of r = |k - p|, map u = r / (r + k_s), %d bins in u over [0, 1); N = 0 omitted "
                    "(it is Z_0 delta(k - p))\n", n_bins_r);
    std::fprintf(f, "# weight = time fraction in the shell; density = weight / shell volume (nan for the last, unbounded bin)\n");
    std::fprintf(f, "# columns: N bin r_lo r_hi weight err density density_err\n");
    for (int N {1}; N <= n_max; ++N) {
        const auto i {static_cast<std::size_t>(N)};
        for (int j {0}; j < n_bins_r; ++j) {
            const double lo {rm.edge(j)}, hi {rm.edge(j + 1)};
            const double vol {4. * std::numbers::pi / 3. * (hi * hi * hi - lo * lo * lo)};
            const double w {r.radial[i][static_cast<std::size_t>(j)]}, e {r.radial_error[i][static_cast<std::size_t>(j)]};
            const double nan {std::numeric_limits<double>::quiet_NaN()};
            std::fprintf(f, "%d %d %.10e %.10e %.15e %.6e %.10e %.6e\n", N, j, lo, hi, w, e,
                         std::isfinite(vol) ? w / vol : nan, std::isfinite(vol) ? e / vol : nan);
        }
    }
    std::fclose(f);

    const std::string ext {".dat"};
    const bool dat {path.size() > ext.size() && path.compare(path.size() - ext.size(), ext.size(), ext) == 0};
    write3d((dat ? path.substr(0, path.size() - ext.size()) : path) + "_3d.dat");
}

void electron_momentum_measurement::write3d(const std::string & path) const {
    const result3d r {normalised3d()};
    const bounded_map am {axisMap()};
    FILE * f {std::fopen(path.c_str(), "w")};
    if (f == nullptr) { throw std::runtime_error("electron_momentum_measurement::write3d: cannot open " + path); }
    std::fprintf(f, "# electron momentum distribution n(k), 3D, summed over N >= 1 (N = 0 is Z_0 delta(k - p)), time-averaged, signed\n");
    std::fprintf(f, "# p = %.10f %.10f %.10f   k_s = %.10e   n_bins_3d = %d per axis\n", p[0], p[1], p[2], k_scale, n_bins_3d);
    std::fprintf(f, "# axes: q_a = k_a - p_a, map u_a = q_a / (|q_a| + k_s), equal bins in u over (-1, 1); the outermost cells reach +-inf\n");
    std::fprintf(f, "# batches = %zu x %llu samples (over all ranks)\n", r.batches, static_cast<unsigned long long>(r.batch_count));
    std::fprintf(f, "# edges:");
    for (int i {0}; i <= n_bins_3d; ++i) { std::fprintf(f, " %.10e", am.edge(i)); }
    std::fprintf(f, "\n# centers:");
    for (int i {0}; i < n_bins_3d; ++i) { std::fprintf(f, " %.10e", am.center(i)); }
    std::fprintf(f, "\n# weight = time fraction in the cell (divide by the cell volume for the density)\n");
    std::fprintf(f, "# columns: ix iy iz weight err\n");
    for (int ix {0}; ix < n_bins_3d; ++ix) {
        for (int iy {0}; iy < n_bins_3d; ++iy) {
            for (int iz {0}; iz < n_bins_3d; ++iz) {
                const auto c {static_cast<std::size_t>((ix * n_bins_3d + iy) * n_bins_3d + iz)};
                std::fprintf(f, "%d %d %d %.12e %.4e\n", ix, iy, iz, r.weight[c], r.error[c]);
            }
        }
    }
    std::fclose(f);
}

void simplemc_mpi_collect(const simplemc::mpi::communicator & comm, electron_momentum_measurement & m){
    simplemc_mpi_collect(comm, m.acc);
    m.collected3d = m.jackknife3d(comm);        // collective; acc3d itself stays per rank
    simplemc::mpi::all_reduce_in_place(m.n_measured, MPI_SUM, comm);
    simplemc::mpi::all_reduce_in_place(m.n_negative, MPI_SUM, comm);
}
