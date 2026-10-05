#ifndef PROP_DISTRIBS_HPP
#define PROP_DISTRIBS_HPP

#include <array>
#include <cmath>
#include <numbers>
#include <random>

// Proposal distributions used by the updates: each one provides draw() and its density, so that the
// acceptance ratios are written in terms of the same object that generated the move.
namespace proposal {

    // Phonon momentum w = r n^ for a line of length l: n^ uniform on the sphere and r = |w| from the
    // half-normal
    //     h(r) = sqrt(2 s l/pi) exp(-s l r^2/2),   r >= 0,
    // s = shape (1: the free-electron width, m = 1). The density is given w.r.t. dr dOmega, i.e.
    // h(r)/(4 pi) = |w|^2 * (density per d^3w): in that measure both this proposal and the target
    // T |g(w)|^2 |w|^2 are finite at w = 0, so no ratio has to cancel two 1/|w|^2 by hand (with a plain
    // 3D Gaussian the importance ratio target/proposal grows as 1/|w|^2, and a line drawn with a small
    // w is then almost never moved off it).
    struct PhononMomentum {
        const double shape {1.};
        // kept between calls: the generator produces normals in pairs and caches the second one
        mutable std::normal_distribution<double> std_norm {0., 1.};

        template <class RNG>
        std::array<double, 3> draw(RNG & rng, double l) const {
            const double r {std::abs(std_norm(rng)) / std::sqrt(shape * l)};
            // a 3D standard normal vector is isotropic: normalised, it is a uniform direction
            const double d[3] {std_norm(rng), std_norm(rng), std_norm(rng)};
            const double d_norm {std::sqrt(d[0]*d[0] + d[1]*d[1] + d[2]*d[2])};
            return {r*d[0]/d_norm, r*d[1]/d_norm, r*d[2]/d_norm};
        }

        // h(r)/(4 pi): the density per dr dOmega
        double sphericalDensity(const std::array<double, 3> & w, double l) const {
            const double r2 {w[0]*w[0] + w[1]*w[1] + w[2]*w[2]};
            return std::sqrt(2. * shape * l / std::numbers::pi) * std::exp(-shape * l * r2 / 2.) / (4. * std::numbers::pi);
        }
    };
}

#endif // !PROP_DISTRIBS_HPP
