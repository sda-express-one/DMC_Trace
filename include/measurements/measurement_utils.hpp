#ifndef MEASUREMENT_UTILS_HPP
#define MEASUREMENT_UTILS_HPP

#include <array>

// Helpers shared by the binned Green function estimators (green_func_measurement,
// green_trace_measurement).
namespace measurement {

    // Z_0 = int_0^tau_max tr G0(p, tau) e^{mu tau} dtau = sum_n (1 - e^{-(E_n - mu) tau_max}) / (E_n - mu):
    // the exact integral of the order-0 diagram's sampling weight, which normalises the estimators.
    // Requires E_n - mu > 0 for every band (mu < 0).
    long double order0Integral(const std::array<double, 3> & p, double chem_pot, double tau_max);

    // e^{-mu tau_b}, tau_b = bin * bin_width the left edge of the bin: the per-bin constant part of the
    // e^{-mu tau_D} that removes the sampling factor e^{mu tau_D}, applied after the jackknife
    long double binScale(int bin, double bin_width, double chem_pot);

    // (bin + 1/2) * bin_width: the centre of the bin, in imaginary time
    long double binCentre(int bin, double bin_width);
}

#endif // !MEASUREMENT_UTILS_HPP
