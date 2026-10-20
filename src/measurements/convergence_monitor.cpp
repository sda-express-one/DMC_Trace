#include "measurements/convergence_monitor.hpp"
#include <cassert>
#include <cmath>
#include <Eigen/Core>
#include "diagram/vertex_manager.hpp"

convergence_monitor::convergence_monitor(diagram_cfg * cfg, Options opt)
    : cfg(cfg),
      opt(opt),
      scalar_names([&]{
          std::vector<std::string> n {"n_int", "n_ext", "Y2_tau", "rms_w", "rms_k"};
          if (opt.tau_D) { n.push_back("tau_D"); }
          if (opt.sign) { n.push_back("sign"); }
          if (opt.energy) { n.push_back("energy"); }
          if (opt.orbital) { n.insert(n.end(), {"M_xx", "M_yy", "M_zz"}); }
          return n;
      }()),
      acc(static_cast<acc_type::size_type>(scalar_names.size()), opt.n_batches)
{
    assert(cfg != nullptr);
    if (opt.energy) {
        energy_estimator = std::make_shared<gs_energy_measurement>(cfg, std::array<double, 3>{0., 0., 0.}, 0., true);
    }
}

std::vector<double> convergence_monitor::scalars() const {
    std::vector<double> out;
    out.reserve(size());
    const double tau {cfg->current_tau_length};

    // one pass over the segments (head + every manager slot: each segment once) and the lines (even slots)
    Eigen::Vector3d Y {Eigen::Vector3d::Zero()};
    double k2 {0.}, w2 {0.};
    const auto segment {[&](const Vertex * v){
        const double d {v->tau_next - v->tau};
        const Eigen::Vector3d k (v->k[0], v->k[1], v->k[2]);
        Y += d * k;
        k2 += d * k.squaredNorm();
    }};
    segment(cfg->diagram_head);
    int lines {0};
    for (const VertexPointerManager * m : {cfg->internal_ph_manager, cfg->external_ph_manager}) {
        for (int i {0}; i < m->current_length; ++i) {
            const Vertex * v {m->selectVertex(i)};
            segment(v);
            if (i % 2 == 0) {
                w2 += v->w[0]*v->w[0] + v->w[1]*v->w[1] + v->w[2]*v->w[2];
                ++lines;
            }
        }
    }

    out.push_back(static_cast<double>(cfg->internal_ph_manager->current_length) / 2);
    out.push_back(static_cast<double>(cfg->external_ph_manager->current_length) / 2);
    out.push_back(Y.squaredNorm() / tau);
    out.push_back(lines > 0 ? std::sqrt(w2 / lines) : 0.);
    out.push_back(std::sqrt(k2 / tau));
    if (opt.tau_D) { out.push_back(tau); }

    const Eigen::Matrix3d & P {cfg->diagram_head->right_component};
    const double trace {P.trace()};
    const bool usable {std::isfinite(trace) && trace != 0.};     // a zero trace has probability zero
    if (opt.sign) { out.push_back(usable ? (trace < 0. ? -1. : 1.) : 0.); }
    if (opt.energy) { out.push_back(usable ? energy_estimator->estimate() : 0.); }
    if (opt.orbital) {
        const Eigen::Matrix3d & U {cfg->diagram_head->baseWF};
        const Eigen::Matrix3d M {usable ? Eigen::Matrix3d(U * P * U.transpose() / std::abs(trace)) : Eigen::Matrix3d::Zero()};
        out.insert(out.end(), {M(0, 0), M(1, 1), M(2, 2)});
    }
    return out;
}

void convergence_monitor::measure(){
    const std::vector<double> s {scalars()};
    acc.accumulate(Eigen::Map<const Eigen::VectorXd>(s.data(), static_cast<Eigen::Index>(s.size())));
}

std::vector<double> convergence_monitor::window(std::size_t & batches) const {
    const auto bs {acc.batches()};
    const std::size_t total {bs.size()}, start {total / 2};
    batches = total - start;
    std::vector<double> out(size() * batches);
    for (std::size_t b {start}; b < total; ++b) {
        const auto m {bs[b].mean()};
        for (std::size_t k {0}; k < size(); ++k) { out[k * batches + (b - start)] = m(static_cast<Eigen::Index>(k)); }
    }
    return out;
}
