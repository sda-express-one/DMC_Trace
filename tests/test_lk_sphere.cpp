// diagonalizeLKHamiltonian over the unit k sphere.
//
// For A > B (AlAs) and A < B (swapped), on a theta x phi grid:
//   - the returned columns diagonalise H_LK(k_hat) built in the ORIGINAL frame (so the IBZ fold and
//     unfold are right), each paired with the eigenvalue returned next to it, and are orthonormal;
//   - eigenvalues come back descending for A > B and ascending for A <= B, at every point;
//   - the basis is continuous inside every IBZ sector: any jump between neighbouring grid points
//     (sign flip, reordering or rotation) lies across a sector boundary or within a few grid steps of
//     a fold plane (k_i = 0, |k_i| = |k_j|), where the degeneracy points (100) and (111) also lie.
// And exactly:
//   - the (100), (110), (111) spectra of Guster et al. Eq. (81), in every sign/permutation variant;
//   - the k = 0 sentinel: degenerate {1, 1, 1} with identity eigenvectors.
//
// usage: test_lk_sphere [--write FILE]   (FILE: per-point data for tests/tools/lk_sphere.gp)
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <numbers>
#include <vector>
#include "test_common.hpp"

using M3 = Eigen::Matrix3d;

static M3 H_original_frame(const std::array<double, 3> & k){
    const double n {std::sqrt(k[0]*k[0] + k[1]*k[1] + k[2]*k[2])};
    const double x {k[0]/n}, y {k[1]/n}, z {k[2]/n};
    const double A {weight::LKMatrix::A_LK}, B {weight::LKMatrix::B_LK}, C {weight::LKMatrix::C_LK};
    M3 h;
    h << A*x*x + B*(y*y + z*z), C*x*y,                 C*x*z,
         C*x*y,                 A*y*y + B*(x*x + z*z), C*y*z,
         C*x*z,                 C*y*z,                 A*z*z + B*(x*x + y*y);
    return h;
}

static int sector_id(const std::array<double, 3> & k){
    const weight::LKMatrix::choice c {weight::LKMatrix::selectionRulesLK(k)};
    static const std::array<std::array<int, 3>, 6> perms {{{0,1,2},{0,2,1},{1,0,2},{1,2,0},{2,0,1},{2,1,0}}};
    int p {0};
    for (int i {0}; i < 6; ++i) { if (perms[i] == c.position) { p = i; } }
    return 8*p + (c.sign[0] < 0 ? 1 : 0) + (c.sign[1] < 0 ? 2 : 0) + (c.sign[2] < 0 ? 4 : 0);
}

static double dist_to_fold_planes(const std::array<double, 3> & k){
    const double a {std::abs(k[0])}, b {std::abs(k[1])}, c {std::abs(k[2])};
    return std::min({a, b, c, std::abs(a - b), std::abs(b - c), std::abs(a - c)});
}

// true if O = U_prev^T U is close to the identity, i.e. the basis did not jump
static bool smooth(const M3 & U_prev, const M3 & U){
    return (U_prev.transpose() * U - M3::Identity()).cwiseAbs().maxCoeff() < 0.25;
}

static void scan(test::Checks & check, const char * label, const std::array<double, 3> & abc, FILE * out){
    test::setLK(abc);
    const bool a_gt_b {abc[0] > abc[1]};
    const int Nt {90}, Np {180};
    const double step {std::numbers::pi / Nt};
    std::vector<M3> U(static_cast<std::size_t>(Nt*Np));
    std::vector<std::array<double, 3>> kk(U.size());
    std::vector<int> sector(U.size());
    double worst_resid {0.}, worst_orth {0.};
    long bad_order {0};
    for (int i {0}; i < Nt; ++i) {
        const double th {(i + 0.5) * step};
        for (int j {0}; j < Np; ++j) {
            const double ph {j * 2. * std::numbers::pi / Np};
            const std::array<double, 3> k {std::sin(th)*std::cos(ph), std::sin(th)*std::sin(ph), std::cos(th)};
            const Eigen::Matrix<double, 4, 3> r {weight::LKMatrix::diagonalizeLKHamiltonian(k)};
            const std::size_t n {static_cast<std::size_t>(i*Np + j)};
            kk[n] = k; U[n] = r.block<3,3>(1,0); sector[n] = sector_id(k);
            const M3 D {U[n].transpose() * H_original_frame(k) * U[n]};
            for (int a {0}; a < 3; ++a) {
                worst_resid = std::max(worst_resid, std::abs(D(a,a) - r(0,a)));
                for (int b {0}; b < 3; ++b) { if (a != b) { worst_resid = std::max(worst_resid, std::abs(D(a,b))); } }
            }
            worst_orth = std::max(worst_orth, (U[n].transpose()*U[n] - M3::Identity()).cwiseAbs().maxCoeff());
            const double e {1e-12};
            const bool ok {a_gt_b ? (r(0,0) >= r(0,1) - e && r(0,1) >= r(0,2) - e)
                                  : (r(0,0) <= r(0,1) + e && r(0,1) <= r(0,2) + e)};
            if (!ok) { ++bad_order; }
            if (out != nullptr) {
                std::fprintf(out, "%.4f %.6f %.10f %.10f %.10f %d\n", j*360./Np, std::cos(th), r(0,0), r(0,1), r(0,2), sector[n]);
            }
        }
        if (out != nullptr) { std::fprintf(out, "\n"); }
    }
    check(worst_resid < 1e-12, "%s: columns diagonalise H in the original frame, paired with their eigenvalues (max residual %.1e)", label, worst_resid);
    check(worst_orth < 1e-12, "%s: eigenvector matrices orthonormal (max |U^T U - I| %.1e)", label, worst_orth);
    check(bad_order == 0, "%s: eigenvalues %s at all %zu points (%ld violations)", label, a_gt_b ? "descending" : "ascending", U.size(), bad_order);

    long jumps {0}, at_boundary {0}, in_bulk {0};
    for (int i {0}; i < Nt; ++i) {
        for (int j {0}; j < Np; ++j) {
            const std::size_t n {static_cast<std::size_t>(i*Np + j)};
            std::vector<std::size_t> nbs {static_cast<std::size_t>(i*Np + (j + Np - 1) % Np)};
            if (i > 0) { nbs.push_back(static_cast<std::size_t>((i-1)*Np + j)); }
            for (const std::size_t m : nbs) {
                if (smooth(U[m], U[n])) { continue; }
                ++jumps;
                const bool boundary {sector[m] != sector[n]
                    || std::min(dist_to_fold_planes(kk[m]), dist_to_fold_planes(kk[n])) < 2.5*step};
                (boundary ? at_boundary : in_bulk) += 1;
            }
        }
    }
    check(in_bulk == 0, "%s: basis continuous inside every IBZ sector (%ld jumps, %ld at fold planes, %ld in the bulk)",
          label, jumps, at_boundary, in_bulk);
}

static void high_symmetry(test::Checks & check, const char * label, const std::array<double, 3> & abc){
    test::setLK(abc);
    const double A {abc[0]}, B {abc[1]}, C {abc[2]};
    struct Dir { const char * name; std::array<double, 3> v; std::array<double, 3> spectrum; };
    const double s2 {1./std::sqrt(2.)}, s3 {1./std::sqrt(3.)};
    const std::array<Dir, 3> dirs {{
        {"(100)", {1., 0., 0.}, {A, B, B}},
        {"(110)", {s2, s2, 0.}, {(A + B + C)/2., (A + B - C)/2., B}},
        {"(111)", {s3, s3, s3}, {(A + 2.*B + 2.*C)/3., (A + 2.*B - C)/3., (A + 2.*B - C)/3.}},
    }};
    for (const Dir & d : dirs) {
        std::array<double, 3> expect {d.spectrum};
        std::sort(expect.begin(), expect.end());
        double worst {0.};
        int variants {0};
        std::array<int, 3> perm {0, 1, 2};
        do {
            for (int s {0}; s < 8; ++s) {
                std::array<double, 3> k {};
                for (int c {0}; c < 3; ++c) { k[perm[c]] = ((s >> c) & 1 ? -1. : 1.) * d.v[c]; }
                const Eigen::Matrix<double, 4, 3> r {weight::LKMatrix::diagonalizeLKHamiltonian(k)};
                std::array<double, 3> got {r(0,0), r(0,1), r(0,2)};
                std::sort(got.begin(), got.end());
                for (int c {0}; c < 3; ++c) { worst = std::max(worst, std::abs(got[c] - expect[c])); }
                ++variants;
            }
        } while (std::next_permutation(perm.begin(), perm.end()));
        check(worst < 1e-12, "%s: %s spectrum matches Eq. (81) in all %d sign/permutation variants (max error %.1e)",
              label, d.name, variants, worst);
    }

    const Eigen::Matrix<double, 4, 3> r0 {weight::LKMatrix::diagonalizeLKHamiltonian({0., 0., 0.})};
    const bool ok0 {r0(0,0) == 1. && r0(0,1) == 1. && r0(0,2) == 1. && M3(r0.block<3,3>(1,0)) == M3::Identity()};
    check(ok0, "%s: k = 0 sentinel is the degenerate {1, 1, 1} with identity eigenvectors", label);
}

int main(int argc, char ** argv){
    FILE * out {nullptr};
    if (argc == 3 && std::strcmp(argv[1], "--write") == 0) { out = std::fopen(argv[2], "w"); }

    test::Checks check {"diagonalizeLKHamiltonian on the unit k sphere"};
    const std::array<double, 3> swapped {test::AlAs[1], test::AlAs[0], test::AlAs[2]};
    scan(check, "A > B", test::AlAs, out);
    scan(check, "A < B", swapped, nullptr);
    high_symmetry(check, "A > B", test::AlAs);
    high_symmetry(check, "A < B", swapped);

    if (out != nullptr) { std::fclose(out); std::printf("  wrote %s (A > B scan)\n", argv[2]); }
    return check.exit_code();
}
