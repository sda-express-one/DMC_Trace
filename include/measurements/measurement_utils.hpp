#ifndef MEASUREMENT_UTILS_HPP
#define MEASUREMENT_UTILS_HPP

#include <array>
#include <span>
#include <Eigen/Core>
#include "diagram/diagram_config.hpp"

// Helpers shared by the estimators: the binned Green function ones (green_func_measurement,
// green_trace_measurement) and the effective-mass ones (eff_mass_measurement,
// eff_mass_uniaxial_measurement).
namespace measurement {

    // Z_0 = int_0^tau_max tr G0(p, tau) e^{mu tau} dtau = sum_n (1 - e^{-(E_n - mu) tau_max}) / (E_n - mu):
    // the exact integral of the order-0 diagram's sampling weight, which normalises the estimators.
    // Requires E_n - mu > 0 for every band (mu < 0).
    long double order0Integral(const std::array<double, 3> & p, double chem_pot, double tau_max);

    // min_n (E_n(p) - mu): the slowest decay rate of the order-0 diagram's sampling weight (> 0 for mu < 0)
    double order0MinRate(const std::array<double, 3> & p, double chem_pot);

    // out[l] = e^{-x/2} L_l(x) for l = 0 .. out.size() - 1 (x >= 0), by the forward three-term recurrence
    // (l+1) psi_{l+1} = (2l+1-x) psi_l - l psi_{l-1} with the exponential folded in, so nothing overflows.
    // |psi_l(x)| <= 1; for x above ~1400 psi_0 underflows to 0 and so do all of them (true values < e^{-300}).
    // sqrt(s) psi_l(s tau) are the Laguerre functions, orthonormal on [0, infinity).
    void laguerreFunctions(double x, std::span<double> out);

    // e^{-mu tau_b}, tau_b = bin * bin_width the left edge of the bin: the per-bin constant part of the
    // e^{-mu tau_D} that removes the sampling factor e^{mu tau_D}, applied after the jackknife
    long double binScale(int bin, double bin_width, double chem_pot);

    // (bin + 1/2) * bin_width: the centre of the bin, in imaginary time
    long double binCentre(int bin, double bin_width);

    // Y = sum over segments of k dtau, the imaginary-time integral of the electron momentum of the current
    // diagram. Every segment starts at exactly one vertex - the head or a phonon vertex (the tail starts
    // none) - and every phonon vertex sits in exactly one slot of the two managers, so head + all slots
    // visit each segment once. The sum does not depend on the order; reaching the vertices through the
    // managers' contiguous slot arrays avoids the dependent pointer chase of walking head -> next -> ... .
    Eigen::Vector3d pathMomentum(const diagram_cfg & cfg);
}

#endif // !MEASUREMENT_UTILS_HPP
