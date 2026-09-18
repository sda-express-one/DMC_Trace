#ifndef GREEN_TRACE_HPP
#define GREEN_TRACE_HPP

#include <cassert>
#include <simplemc/accs/var_acc.hpp>
#include "../diagram/diagram_config.hpp"
#include "../diagram/vertex.hpp"

struct green_trace_measurement {
    diagram_cfg * const cfg {nullptr};

    // accumulates the (unnormalized) diagram weight trace sampled once per MC cycle;
    // var_acc (not mean_acc) so variance()/stderror() are available for error bars.
    simplemc::var_acc<double> acc;

    explicit green_trace_measurement(diagram_cfg * cfg) : cfg(cfg) {
        assert(cfg != nullptr);
    }

    void measure() {
        // diagram_head->right_component is refreshed by every update's accept() via
        // weight::LKMatrix::computeRightSide/computeLeftSide, so this is O(1) here -
        // no diagram walk needed.
        acc << cfg->diagram_head->right_component.trace();
    }
};

#endif // !GREEN_TRACE_HPP
