// Gauge stability of diagonalizeLKHamiltonian next to the symmetry planes and lines (canonicalDirection).
//
// Two momenta that are equal physically but differ by rounding must get the same basis: the trace closes without an
// overlap, and several updates copy a basis or reuse a cached overlap across momenta computed separately. For every
// sign/permutation variant of the axes (100), the lines (110) and (111), the axis planes (a, 0, b) and the diagonal
// planes (a, a, b), at |k| = 1e-6, 0.1 and 10, k is perturbed by rounding-sized and up to 1e-12 * |k| amounts in the
// constrained components (the zeros and the tied magnitudes), and by one ulp in the free ones:
//   - on the lines the eigenvalues and eigenvectors are bitwise those of the unperturbed point (the degenerate pair
//     on (100) and (111) leaves the solver free to rotate, so only identical input is safe);
//   - on the planes the basis is continuous, U0^T U = 1 to 1e-12 (no sign flip, no reordering), and +0 / -0 in a
//     zero component give bitwise the same result;
//   - every returned basis still diagonalises H(k_hat) of the unsnapped k to 1e-8.
// For AlAs (A > B), A < B, a scan of C for both orderings (test::lkCases), C = 0 with A > B, and C = 0 with A = B.
// Also: a generic direction is not touched (canonicalDirection is the plain k / |k|, bitwise), and the snap threshold
// sits at gauge_tol.
// Sign gauge (sign_pattern): for AlAs and CaO (A, B), both orders, at ~100 C over the allowed range
// -(A + 2B)/2 < C < A + B and close to C = A - B, every boundary basis (axis plane, both diagonal planes, (110)) is
// the limit from the wedge interior, along two inward approaches; inside the wedge every component has its
// sign_pattern sign and the basis is bitwise the reference-axis rule's; in the original frame every boundary band
// matches at least one side.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <random>
#include <vector>
#include "test_common.hpp"

using M3 = Eigen::Matrix3d;
using K = std::array<double, 3>;
using Eig = Eigen::Matrix<double, 4, 3>;

static M3 H_original_frame(const K & k){
    const double n {std::sqrt(k[0]*k[0] + k[1]*k[1] + k[2]*k[2])};
    const double x {k[0]/n}, y {k[1]/n}, z {k[2]/n};
    const double A {weight::LKMatrix::A_LK}, B {weight::LKMatrix::B_LK}, C {weight::LKMatrix::C_LK};
    M3 h;
    h << A*x*x + B*(y*y + z*z), C*x*y,                 C*x*z,
         C*x*y,                 A*y*y + B*(x*x + z*z), C*y*z,
         C*x*z,                 C*y*z,                 A*z*z + B*(x*x + y*y);
    return h;
}

static bool bitwise(const Eig & a, const Eig & b){ return std::memcmp(a.data(), b.data(), sizeof(double) * 12) == 0; }

// largest |U^T H(k) U - diag(eigenvalues)|
static double residual(const K & k, const Eig & r){
    const M3 U {r.block<3,3>(1,0)};
    const M3 D {U.transpose() * H_original_frame(k) * U};
    double w {0.};
    for (int a {0}; a < 3; ++a) {
        for (int b {0}; b < 3; ++b) { w = std::max(w, std::abs(D(a,b) - (a == b ? r(0,a) : 0.))); }
    }
    return w;
}

struct Tally {
    long points {0}, perturbed {0}, line_not_bitwise {0}, plane_jumps {0}, signed_zero_diff {0};
    double worst_plane {0.}, worst_resid {0.};
};

// the perturbations of one point k0 (its zero components and tied magnitudes are the constrained ones)
static std::vector<K> perturbations(const K & k0){
    const double n {std::sqrt(k0[0]*k0[0] + k0[1]*k0[1] + k0[2]*k0[2])};
    std::vector<K> out;
    const auto push {[&](int i, double v){ K k {k0}; k[i] = v; out.push_back(k); }};
    for (int i {0}; i < 3; ++i) {
        const double inf {std::numeric_limits<double>::infinity()};
        if (k0[i] == 0.) {
            for (const double e : {1e-17, -1e-17, 1e-12, -1e-12}) { push(i, e * n); }
            push(i, std::nextafter(0., 1.));
            push(i, std::nextafter(0., -1.));
        }
        else {
            push(i, std::nextafter(k0[i], inf));
            push(i, std::nextafter(k0[i], -inf));
            push(i, k0[i] * (1. + 1e-12));
            push(i, k0[i] * (1. - 1e-12));
        }
    }
    // all three components at once, rounding-sized, as from a different summation order
    K k {k0};
    for (int i {0}; i < 3; ++i) { k[i] = k0[i] == 0. ? 3e-17 * n * (i % 2 ? 1. : -1.) : k0[i] * (1. + (i - 1) * 2.2e-16); }
    out.push_back(k);
    return out;
}

static void scan(test::Checks & check, const char * label, const std::array<double, 3> & abc){
    test::setLK(abc);
    // representatives; is_line: (100), (110), (111), whose canonical input is exact
    const std::array<std::pair<K, bool>, 7> reps {{
        {{1., 0., 0.}, true}, {{1., 1., 0.}, true}, {{1., 1., 1.}, true},
        {{0.8, 0., 0.35}, false}, {{0.6, 0.6, 0.25}, false}, {{0.6, 0.6, 0.9}, false}, {{0.3, 0.7, 0.}, false}}};
    static const std::array<std::array<int, 3>, 6> perms {{{0,1,2},{0,2,1},{1,0,2},{1,2,0},{2,0,1},{2,1,0}}};
    Tally t;
    for (const auto & [dir, is_line] : reps) {
        for (const double scale : {1e-6, 0.1, 10.}) {
            for (const auto & pm : perms) {
                for (int s {0}; s < 8; ++s) {
                    K k0;
                    for (int i {0}; i < 3; ++i) { k0[i] = scale * dir[pm[i]] * ((s >> i) & 1 ? -1. : 1.); }
                    for (double & x : k0) { if (x == 0.) { x = 0.; } }     // +0 for the reference point
                    const Eig r0 {weight::LKMatrix::diagonalizeLKHamiltonian(k0)};
                    ++t.points;
                    t.worst_resid = std::max(t.worst_resid, residual(k0, r0));
                    for (const K & k : perturbations(k0)) {
                        const Eig r {weight::LKMatrix::diagonalizeLKHamiltonian(k)};
                        ++t.perturbed;
                        t.worst_resid = std::max(t.worst_resid, residual(k, r));
                        if (is_line) {
                            if (!bitwise(r, r0)) { ++t.line_not_bitwise; }
                        }
                        else {
                            const M3 O {r0.block<3,3>(1,0).transpose() * r.block<3,3>(1,0)};
                            const double dev {(O - M3::Identity()).cwiseAbs().maxCoeff()};
                            t.worst_plane = std::max(t.worst_plane, dev);
                            if (dev > 1e-12) { ++t.plane_jumps; }
                        }
                    }
                    // -0 in a zero component
                    for (int i {0}; i < 3; ++i) {
                        if (k0[i] != 0.) { continue; }
                        K k {k0};
                        k[i] = -0.;
                        if (!bitwise(weight::LKMatrix::diagonalizeLKHamiltonian(k), r0)) { ++t.signed_zero_diff; }
                    }
                }
            }
        }
    }
    check(t.line_not_bitwise == 0 && t.plane_jumps == 0 && t.signed_zero_diff == 0 && t.worst_resid < 1e-8,
          "%-14s %ld points, %ld perturbed: lines not bitwise %ld, plane bases off %ld (worst |U0^T U - 1| %.1e), "
          "+0/-0 differ %ld, worst residual vs H(unsnapped k) %.1e",
          label, t.points, t.perturbed, t.line_not_bitwise, t.plane_jumps, t.worst_plane, t.signed_zero_diff, t.worst_resid);
}

static K unit(const K & k){ const double n {std::sqrt(k[0]*k[0] + k[1]*k[1] + k[2]*k[2])}; return {k[0]/n, k[1]/n, k[2]/n}; }
static M3 basisOf(const K & k){ return weight::LKMatrix::diagonalizeLKHamiltonian(k).block<3,3>(1,0); }

struct Limits { long boundary_off {0}, neither_side {0}, interior_sign {0}, interior_not_ref {0}, boundaries {0}, interior {0}; };

// one A, B, C: boundaries against the interior limit, interior against sign_pattern and the reference-axis rule
static void limits(Limits & t, std::mt19937_64 & rng){
    using L = weight::LKMatrix;
    struct Bd { K v, in1, in2, normal; };
    std::vector<Bd> bds;
    for (const double r : {0.2, 0.6, 0.95}) {
        bds.push_back({unit({1., r, 0.}), {0., 0., 1.}, {0.3, -0.2, 1.}, {0., 0., 1.}});    // axis plane, inward k_2 > 0
        bds.push_back({unit({1., 1., r}), {1., -1., 0.}, {1., -1., 0.3}, {1., -1., 0.}});   // k_0 = k_1, inward k_0 > k_1
        bds.push_back({unit({1., r, r}), {0., 1., -1.}, {0.2, 1., -1.}, {0., 1., -1.}});    // k_1 = k_2, inward k_1 > k_2
    }
    bds.push_back({unit({1., 1., 0.}), {1., -1., 1.}, {1., -1., 0.2}, {0., 0., 1.}});       // (110), a corner
    for (const Bd & b : bds) {
        ++t.boundaries;
        // wedge-frame input: the reduction is the identity, so these are the wedge bases themselves
        const M3 U0 {basisOf(b.v)};
        for (const K & in : {b.in1, b.in2}) {
            const M3 O {U0.transpose() * basisOf(unit({b.v[0] + 1e-6*in[0], b.v[1] + 1e-6*in[1], b.v[2] + 1e-6*in[2]}))};
            for (int n {0}; n < 3; ++n) { if (O(n,n) < 0.99) { ++t.boundary_off; } }
        }
        for (int n {0}; n < 3; ++n) {
            int match {0};
            for (const double e : {1e-6, -1e-6}) {
                const M3 O {U0.transpose() * basisOf({b.v[0] + e*b.normal[0], b.v[1] + e*b.normal[1], b.v[2] + e*b.normal[2]})};
                if (O(n,n) > 0.99) { ++match; }
            }
            if (match == 0) { ++t.neither_side; }
        }
    }
    std::uniform_real_distribution<double> u {0., 1.};
    for (int i {0}; i < 200; ++i) {
        K v {u(rng), u(rng), u(rng)};
        std::sort(v.begin(), v.end(), [](double a, double b){ return a > b; });
        v = unit(v);
        if (v[2] < 1e-3 || v[0] - v[1] < 1e-3 || v[1] - v[2] < 1e-3) { continue; }
        ++t.interior;
        const M3 U {basisOf(v)};
        // the reference-axis rule (positive component on axis n, or 2 - n for A > B) on the same solver output
        M3 R;
        Eigen::RowVector3d e;
        const K kh {L::canonicalDirection(v)};      // the direction diagonalizeLKHamiltonian builds H from
        const double kv[3] {kh[0], kh[1], kh[2]};
        L::wedgeEigen(kv, R, e);
        for (int n {0}; n < 3; ++n) {
            if (R(L::A_LK > L::B_LK ? 2 - n : n, n) < 0.) { R.col(n) *= -1; }
            for (int c {0}; c < 3; ++c) {
                if (L::sign_pattern(c, n) != 0. && U(c, n) * L::sign_pattern(c, n) <= 0.) { ++t.interior_sign; }
            }
        }
        if (std::memcmp(U.data(), R.data(), sizeof(double) * 9) != 0) { ++t.interior_not_ref; }
    }
}

int main(){
    test::Checks check {"LK gauge stability on symmetry planes and lines (canonicalDirection)"};

    for (const test::LKCase & c : test::lkCases()) { scan(check, c.label.c_str(), c.abc); }
    scan(check, "C = 0, A > B", {1.5, 1.0, 0.});
    scan(check, "C = 0, A = B", {0.5, 0.5, 0.});

    // the sign gauge: boundaries = limit from the wedge interior, over the allowed C range
    {
        std::mt19937_64 rng {11};
        struct AB { const char * label; double A, B; };
        for (const AB ab : {AB{"AlAs A, B", test::AlAs[0], test::AlAs[1]}, AB{"CaO A, B", test::CaO[0], test::CaO[1]},
                            AB{"AlAs B, A", test::AlAs[1], test::AlAs[0]}, AB{"CaO B, A", test::CaO[1], test::CaO[0]}}) {
            const double lo {-(ab.A + 2.*ab.B)/2.}, hi {ab.A + ab.B}, sph {ab.A - ab.B};
            std::vector<double> Cs;
            for (int i {1}; i < 100; ++i) {
                const double C {lo + (hi - lo) * i / 100.};
                if (std::abs(C - sph) > 1e-3 * std::abs(sph)) { Cs.push_back(C); }
            }
            for (const double e : {1e-4, -1e-4, 1e-6, -1e-6}) { Cs.push_back(sph * (1. + e)); }
            Cs.push_back(0.);
            Limits t;
            for (const double C : Cs) {
                test::setLK({ab.A, ab.B, C});
                limits(t, rng);
            }
            check(t.boundary_off == 0 && t.neither_side == 0 && t.interior_sign == 0 && t.interior_not_ref == 0,
                  "%-10s %zu C values, %ld boundary points: off the interior limit %ld, bands matching neither side %ld; "
                  "%ld interior points: sign off sign_pattern %ld, not the reference-axis rule %ld",
                  ab.label, Cs.size(), t.boundaries, t.boundary_off, t.neither_side, t.interior, t.interior_sign,
                  t.interior_not_ref);
        }
    }

    // a generic direction is returned as the plain k / |k|, bitwise
    {
        std::mt19937_64 rng {2024};
        std::normal_distribution<double> g {0., 1.};
        long touched {0}, tested {0};
        for (int i {0}; i < 10000; ++i) {
            const K k {g(rng), g(rng), g(rng)};
            const double n {std::sqrt(k[0]*k[0] + k[1]*k[1] + k[2]*k[2])};
            const K plain {k[0]/n, k[1]/n, k[2]/n};
            const double a {std::abs(plain[0])}, b {std::abs(plain[1])}, c {std::abs(plain[2])};
            if (std::min({a, b, c, std::abs(a - b), std::abs(b - c), std::abs(a - c)}) < 1e-6) { continue; }
            ++tested;
            if (weight::LKMatrix::canonicalDirection(k) != plain) { ++touched; }
        }
        check(touched == 0 && tested > 9900, "generic directions untouched: %ld of %ld changed", touched, tested);
    }

    // the threshold: a component at 2 gauge_tol is kept, at gauge_tol / 2 snapped; the same for a tie
    {
        const double tol {weight::LKMatrix::gauge_tol};
        const K kept {weight::LKMatrix::canonicalDirection({0.8, 2. * tol, 0.6})};
        const K snapped {weight::LKMatrix::canonicalDirection({0.8, 0.5 * tol, 0.6})};
        const K tie_kept {weight::LKMatrix::canonicalDirection({0.6, 0.6 + 2. * tol, 0.5})};
        const K tie_snapped {weight::LKMatrix::canonicalDirection({0.6, 0.6 + 0.5 * tol, 0.5})};
        check(kept[1] > 0. && snapped[1] == 0. && !std::signbit(snapped[1]) && tie_kept[0] != tie_kept[1]
              && tie_snapped[0] == tie_snapped[1],
              "threshold at gauge_tol = %.0e: component 2 tol kept (%.2e), tol/2 snapped to +0; magnitudes 2 tol apart "
              "kept, tol/2 apart tied", tol, kept[1]);
    }
    return check.exit_code();
}
