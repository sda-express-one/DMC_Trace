#ifndef LITTLE_GROUP_HPP
#define LITTLE_GROUP_HPP

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <vector>
#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include "comp_method/weight_computation.hpp"

// Symmetry channels of the external momentum p in the orbital space of the Luttinger-Kohn model.
//
// The model (LK matrix, orbital-diagonal Froehlich coupling depending on |q| only) is invariant under the cubic group
// O_h, the 48 signed permutation matrices R acting on k and, as R itself, on the orbital vector (x, y, z). The little
// group G_p = {R : R p = p} commutes with H at momentum p, so the vacuum-sector Green function G(p, tau) in the orbital
// basis commutes with every R in G_p (latex_notes/symmetry_channel_excited_band.tex, Secs. 1-3). Its isotypic
// components on R^3 - the sums of all copies of one irreducible representation - are the channels: with Q_c their
// orthogonal projectors, tr(Q_c G) contains only the states of channel c, and decays with the lowest of them.
//
//     [100]: x | (y, z)        [110]: x+y | x-y | z        [111]: x+y+z | the plane orthogonal to it
//     mirror plane only: the plane (2, two copies of the trivial irrep) | its normal     generic p, p = 0: R^3 (trace)
//
// The isotypic components are found without character tables: they are the joint eigenspaces of the class sums
// C_K = sum_{R in K} R, which act on component Gamma as the scalar |K| chi_Gamma(K) / d_Gamma. A combination
// M = sum_K c_K C_K with incommensurate c_K has one eigenvalue per component. The point groups here have real
// characters (every class is closed under inversion), so C_K is symmetric.
namespace symmetry {

struct Channel {
    Eigen::Matrix3d Q {Eigen::Matrix3d::Zero()};    // orthogonal projector onto the channel (orbital basis)
    Eigen::MatrixXd basis;                          // 3 x dim, orthonormal columns spanning the channel
    int dim {0};
    std::vector<double> bare;                       // eigenvalues of H_LK(p_hat) on the channel, ascending (bare lambda)
};

// the 48 signed permutation matrices of O_h
inline std::vector<Eigen::Matrix3d> cubicGroup(){
    std::vector<Eigen::Matrix3d> G;
    std::array<int, 3> perm {0, 1, 2};
    do {
        for (int signs {0}; signs < 8; ++signs) {
            Eigen::Matrix3d R {Eigen::Matrix3d::Zero()};
            for (int i {0}; i < 3; ++i) { R(perm[i], i) = (signs >> i) & 1 ? -1. : 1.; }
            G.push_back(R);
        }
    } while (std::next_permutation(perm.begin(), perm.end()));
    return G;
}

// the direction used for the symmetry: the code's own canonical direction (weight::LKMatrix::canonicalDirection snaps
// ties and zeros of p_hat to exact values, as the diagonaliser does), or (0, 0, 0) for p = 0
inline std::array<double, 3> symmetryDirection(const std::array<double, 3> & p){
    if (p[0] == 0. && p[1] == 0. && p[2] == 0.) { return {0., 0., 0.}; }
    return weight::LKMatrix::canonicalDirection(p);
}

// G_p: the elements of O_h that fix p (all 48 for p = 0)
inline std::vector<Eigen::Matrix3d> littleGroup(const std::array<double, 3> & p){
    const std::array<double, 3> d {symmetryDirection(p)};
    const Eigen::Vector3d v {d[0], d[1], d[2]};
    std::vector<Eigen::Matrix3d> G;
    for (const Eigen::Matrix3d & R : cubicGroup()) {
        if ((R * v - v).cwiseAbs().maxCoeff() < 1e-12) { G.push_back(R); }
    }
    return G;
}

// H_LK at the unit direction d (orbital basis), with the current LK parameters; zero for d = 0
inline Eigen::Matrix3d lkMatrix(const std::array<double, 3> & d){
    const double A {weight::LKMatrix::A_LK}, B {weight::LKMatrix::B_LK}, C {weight::LKMatrix::C_LK};
    Eigen::Matrix3d H;
    H << A*d[0]*d[0] + B*(d[1]*d[1] + d[2]*d[2]), C*d[0]*d[1], C*d[0]*d[2],
         C*d[0]*d[1], A*d[1]*d[1] + B*(d[0]*d[0] + d[2]*d[2]), C*d[1]*d[2],
         C*d[0]*d[2], C*d[1]*d[2], A*d[2]*d[2] + B*(d[0]*d[0] + d[1]*d[1]);
    return H;
}

// The channels (isotypic components) of the group G acting on R^3 through its own matrices, sorted by their lowest
// bare eigenvalue of H (then by dimension). Their projectors sum to the identity and commute with every element of G.
inline std::vector<Channel> orbitalChannels(const std::vector<Eigen::Matrix3d> & G, const Eigen::Matrix3d & H){
    assert(!G.empty());
    // conjugacy classes, by brute force (|G| <= 48); the matrices have entries 0, +-1, so equality is exact
    const auto same {[](const Eigen::Matrix3d & a, const Eigen::Matrix3d & b){ return (a - b).cwiseAbs().maxCoeff() < 0.5; }};
    std::vector<int> cls(G.size(), -1);
    int n_cls {0};
    for (std::size_t i {0}; i < G.size(); ++i) {
        if (cls[i] >= 0) { continue; }
        for (const Eigen::Matrix3d & h : G) {
            const Eigen::Matrix3d c {h * G[i] * h.transpose()};
            for (std::size_t j {0}; j < G.size(); ++j) {
                if (same(G[j], c)) { cls[j] = n_cls; }
            }
        }
        ++n_cls;
    }
    // M = sum_K c_K (C_K + C_K^T)/2 with incommensurate coefficients
    static constexpr std::array<double, 10> primes {2., 3., 5., 7., 11., 13., 17., 19., 23., 29.};
    Eigen::Matrix3d M {Eigen::Matrix3d::Zero()};
    for (std::size_t j {0}; j < G.size(); ++j) {
        const double c {std::sqrt(primes[static_cast<std::size_t>(cls[j]) % primes.size()]) + 0.1 * cls[j]};
        M += c * 0.5 * (G[j] + G[j].transpose());
    }
    const Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> es {M};
    const Eigen::Vector3d ev {es.eigenvalues()};
    const double tol {1e-8 * std::max(1., ev.cwiseAbs().maxCoeff())};

    std::vector<Channel> out;
    int i {0};
    while (i < 3) {
        int j {i + 1};
        while (j < 3 && std::abs(ev(j) - ev(i)) < tol) { ++j; }
        Channel ch;
        ch.dim = j - i;
        ch.basis = es.eigenvectors().middleCols(i, ch.dim);
        ch.Q = ch.basis * ch.basis.transpose();
        const Eigen::MatrixXd Hc {ch.basis.transpose() * H * ch.basis};
        const Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eh {Hc};
        for (int k {0}; k < ch.dim; ++k) { ch.bare.push_back(eh.eigenvalues()(k)); }
        out.push_back(ch);
        i = j;
    }
    std::stable_sort(out.begin(), out.end(), [](const Channel & a, const Channel & b){
        if (std::abs(a.bare.front() - b.bare.front()) > 1e-12) { return a.bare.front() < b.bare.front(); }
        return a.dim < b.dim;
    });
    return out;
}

// the channels of the external momentum p, with H_LK(p_hat) for the bare values
inline std::vector<Channel> channelsOf(const std::array<double, 3> & p){
    return orbitalChannels(littleGroup(p), lkMatrix(symmetryDirection(p)));
}

} // namespace symmetry

#endif // !LITTLE_GROUP_HPP
