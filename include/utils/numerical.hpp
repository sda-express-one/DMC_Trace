#ifndef NUMERICAL_HPP
#define NUMERICAL_HPP

#include <cmath>

namespace numerical {
    // evaluates equality between two double precision values
    inline bool isEqual(long double a, long double b, long double epsilon = 1e-9L) {return std::fabs(a - b) < epsilon;};
}

#endif // !NUMERICAL_HPP
