#ifndef RUN_INPUT_HPP
#define RUN_INPUT_HPP

#include <array>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <map>
#include <string>
#include <string_view>
#include <vector>
#include <simplemc/mc/simulation_params.hpp>
#include <simplemc/random/splitmix64.hpp>
#include <simplemc/serialize/concepts.hpp>
#include "diagram/diagram_builder.hpp"
#include "diagram/phonon_manager.hpp"
#include "measurements/convergence_monitor.hpp"
#include "measurements/thermalization_control.hpp"

// Every user-set parameter of a run, read from one JSON file through simplemc's input-config channel. The file
// has the layout of simplemc::save_json_input_config:
//
//   {
//     "config":       { "model": {...}, "diagram": {...}, "start": {...}, "rng": {...},
//                       "measurements": {...}, "thermalization": {...}, "output": {...} },   <- run_input
//     "params":       { simplemc::simulation_params of the production run; defaults: input::defaultParams() },
//     "updates":      { "add_int_ph": {"weight": 1.0}, ... },        <- run_input::update_weights
//     "measurements": { "gs_energy": {"is_active": true}, ... }      <- run_input::measurements_active
//   }
//
// Every key is optional: a missing one keeps its default below (input::writeDefault writes the complete file).
// A key that is not known is an error (input::load), so a typo cannot silently leave a default in place.
// Every rank reads the same file: nothing is broadcast.
//
// The "updates" and "measurements" blocks have simplemc's own entry format ({"weight": w}, {"is_active": b}), but
// are read here, tolerantly (simplemc's loader requires every registered name to be present).
//
// Order of use: input::load -> input::applyGlobals (before any measurement is constructed: eff_mass* capture
// A_LK, B_LK at construction) -> build the diagram -> makeUpdateSet (updates/update_set.hpp) with the weights.
// the production estimators, in the order of main's measurement set: the keys of the "measurements" block
inline constexpr std::array<std::string_view, 7> measurement_names {
    "gs_energy", "qp_weights", "eff_mass", "eff_mass_uniaxial", "green_func", "green_trace", "avg_sign"};

struct run_input {
    struct Model {
        double A_LK {1.};                       // A > 0, B > 0, -A/2 - B < C < A + B
        double B_LK {1.};
        double C_LK {0.};
        double V_unit_cell {1.};                // Coupling::Parameters
        double V_BvK {1.};
        std::vector<double> ph_energy {1.};     // one entry per phonon mode (PhononMode::phonon_energy)
        std::vector<double> diel_response {1.}; // same length (PhononMode::diel_response, epsilon*)
    };

    struct Diagram {
        std::array<double, 3> p {0., 0., 0.};   // external momentum
        double mu {-1.};                        // chemical potential: < 0 for variable length, 0 for fixed length
        double tau_max {50.};                   // largest diagram length (= tau_D_start for fixed length)
        bool tau_fixed {false};                 // fixed tau_D: no chg_tau, str_diagram, scl_diagram
        double tau_D_start {1.};                // starting length
        double tau_min {0.};                    // measurements: smallest tau_D that is sampled (variable length)
        int max_internal_lines {200};           // capacity of the managers (2 vertices per line)
        int max_external_lines {8};
    };

    struct Start {
        bool random_start {false};              // diagram_builder::randomDiagram instead of the bare propagator
        int n_internal_min {0};                 // diagram_builder::RandomDiagramSpec
        int n_internal_max {0};
        int n_external_min {0};
        int n_external_max {0};
        double w_spread {1.};                   // momentum spread: std of |w| = w_spread / sqrt(span), 1 = the add proposal
    };

    struct Rng {
        std::uint64_t seed {simplemc::splitmix64::default_seed};   // simplemc::seed_rng(rng, rank, seed)
    };

    struct Measurements {
        std::size_t n_batches {256};            // batches of the estimators' accumulators
        int green_n_bins {100};                 // green_func / green_trace: tau bins over [0, tau_max]
        int boundary_phonons {0};               // green_*: n >= 0 exactly n external lines (0: vacuum sector), < 0 all
        bool split_sign {false};
    };

    struct Thermalization {
        bool enabled {true};
        bool monitor_tau_D {false};             // convergence_monitor::Options
        bool monitor_sign {true};
        bool monitor_energy {true};
        bool monitor_orbital {false};
        std::size_t monitor_n_batches {256};
        double rhat_max {1.01};                 // thermalization_control::Options
        double min_bulk_ess_per_split_chain {50.};
        double min_tail_ess_per_split_chain {50.};
        int consecutive {3};
        int root {0};
        std::uint64_t max_steps {1'000'000'000};        // cap of the thermalization run
        std::uint64_t checkpoint_interval {1'000'000};  // steps between R-hat checkpoints
        std::uint64_t steps_per_cycle {10};             // steps between two monitor samples
        std::uint64_t cycles_per_check {1000};
    };

    struct Output {
        std::string prefix {"dmc"};             // directory/stem of the files written by the estimators
    };

    struct Checkpoint {
        bool enabled {true};                    // write <prefix>_checkpoint_rank<r>.msgpack (one file per rank)
        double interval {3600.};                // seconds between production checkpoints (per rank)
        bool restart {false};                   // continue from the checkpoint files of output.prefix
    };

    Model model;
    Diagram diagram;
    Start start;
    Rng rng;
    Measurements measurements;
    Thermalization thermalization;
    Output output;
    Checkpoint checkpoint;
    std::map<std::string, double> update_weights;   // every name of update_names, set by the constructor
    std::map<std::string, bool> measurements_active; // every name of measurement_names: gs_energy and qp_weights on

    run_input();

    // the existing option types, built from the fields above
    std::vector<PhononMode> phononModes() const;
    diagram_builder::RandomDiagramSpec randomDiagramSpec() const;
    convergence_monitor::Options monitorOptions() const;
    thermalization_control::Options controlOptions() const;
    simplemc::simulation_params thermalizationParams() const;   // step-based only: no time limit
};

namespace input {
    // The project defaults of the "params" block (simplemc's own suit its tutorials, not long runs):
    // max_time = 84600 s (23 h 30 min) is the wall-time budget of the WHOLE job - main stops the thermalization at
    // it and gives the production only what is left; steps_per_cycle = 10, cycles_per_check = 1000 (stop checks
    // every 1e4 steps); max_steps unlimited (set it, or let the budget decide); no checkpoints.
    simplemc::simulation_params defaultParams();

    // Throws simplemc::simplemc_exception listing every violated condition.
    void validate(const run_input & in);

    // Sets the global LKMatrix and Coupling::Parameters. Once, before any measurement is constructed.
    void applyGlobals(const run_input & in);

    // Reads "config", "params", "updates" (weights) and "measurements" (is_active) from the file and validates.
    // Unknown keys throw.
    void load(const std::filesystem::path & path, simplemc::simulation_params & params, run_input & in);

    // Writes the complete file (config, params, updates, measurements) with the given values; writeDefault with
    // defaultParams() and run_input {}.
    void write(const std::filesystem::path & path, const simplemc::simulation_params & params, const run_input & in);
    void writeDefault(const std::filesystem::path & path);

    void print(const run_input & in, std::FILE * fp = stdout);

    // What a checkpoint's saved state depends on - the model, the diagram settings and the estimators' settings - as
    // a canonical JSON string; a restart requires it to be unchanged (params, thermalization, update weights, output,
    // seed, checkpoint and the estimators' is_active may change between jobs).
    std::string fingerprint(const run_input & in);
}

// simplemc input-config hooks (the "config" block); every key optional on load.
template <simplemc::serializer S>
void simplemc_save_input_config(S s, const run_input & in){
    auto m {s["model"]};
    m.save_at("A_LK", in.model.A_LK);
    m.save_at("B_LK", in.model.B_LK);
    m.save_at("C_LK", in.model.C_LK);
    m.save_at("V_unit_cell", in.model.V_unit_cell);
    m.save_at("V_BvK", in.model.V_BvK);
    m.save_at("ph_energy", in.model.ph_energy);
    m.save_at("diel_response", in.model.diel_response);

    auto d {s["diagram"]};
    d.save_at("p", in.diagram.p);
    d.save_at("mu", in.diagram.mu);
    d.save_at("tau_max", in.diagram.tau_max);
    d.save_at("tau_fixed", in.diagram.tau_fixed);
    d.save_at("tau_D_start", in.diagram.tau_D_start);
    d.save_at("tau_min", in.diagram.tau_min);
    d.save_at("max_internal_lines", in.diagram.max_internal_lines);
    d.save_at("max_external_lines", in.diagram.max_external_lines);

    auto st {s["start"]};
    st.save_at("random_start", in.start.random_start);
    st.save_at("n_internal_min", in.start.n_internal_min);
    st.save_at("n_internal_max", in.start.n_internal_max);
    st.save_at("n_external_min", in.start.n_external_min);
    st.save_at("n_external_max", in.start.n_external_max);
    st.save_at("w_spread", in.start.w_spread);

    s["rng"].save_at("seed", in.rng.seed);

    auto ms {s["measurements"]};
    ms.save_at("n_batches", in.measurements.n_batches);
    ms.save_at("green_n_bins", in.measurements.green_n_bins);
    ms.save_at("boundary_phonons", in.measurements.boundary_phonons);
    ms.save_at("split_sign", in.measurements.split_sign);

    auto t {s["thermalization"]};
    t.save_at("enabled", in.thermalization.enabled);
    t.save_at("monitor_tau_D", in.thermalization.monitor_tau_D);
    t.save_at("monitor_sign", in.thermalization.monitor_sign);
    t.save_at("monitor_energy", in.thermalization.monitor_energy);
    t.save_at("monitor_orbital", in.thermalization.monitor_orbital);
    t.save_at("monitor_n_batches", in.thermalization.monitor_n_batches);
    t.save_at("rhat_max", in.thermalization.rhat_max);
    t.save_at("min_bulk_ess_per_split_chain", in.thermalization.min_bulk_ess_per_split_chain);
    t.save_at("min_tail_ess_per_split_chain", in.thermalization.min_tail_ess_per_split_chain);
    t.save_at("consecutive", in.thermalization.consecutive);
    t.save_at("root", in.thermalization.root);
    t.save_at("max_steps", in.thermalization.max_steps);
    t.save_at("checkpoint_interval", in.thermalization.checkpoint_interval);
    t.save_at("steps_per_cycle", in.thermalization.steps_per_cycle);
    t.save_at("cycles_per_check", in.thermalization.cycles_per_check);

    s["output"].save_at("prefix", in.output.prefix);

    auto ck {s["checkpoint"]};
    ck.save_at("enabled", in.checkpoint.enabled);
    ck.save_at("interval", in.checkpoint.interval);
    ck.save_at("restart", in.checkpoint.restart);
}

template <simplemc::serializer S>
void simplemc_load_input_config(const S & s, run_input & in){
    if (s.has("model")) {
        const auto m {s["model"]};
        m.try_load_at("A_LK", in.model.A_LK);
        m.try_load_at("B_LK", in.model.B_LK);
        m.try_load_at("C_LK", in.model.C_LK);
        m.try_load_at("V_unit_cell", in.model.V_unit_cell);
        m.try_load_at("V_BvK", in.model.V_BvK);
        m.try_load_at("ph_energy", in.model.ph_energy);
        m.try_load_at("diel_response", in.model.diel_response);
    }
    if (s.has("diagram")) {
        const auto d {s["diagram"]};
        d.try_load_at("p", in.diagram.p);
        d.try_load_at("mu", in.diagram.mu);
        d.try_load_at("tau_max", in.diagram.tau_max);
        d.try_load_at("tau_fixed", in.diagram.tau_fixed);
        d.try_load_at("tau_D_start", in.diagram.tau_D_start);
        d.try_load_at("tau_min", in.diagram.tau_min);
        d.try_load_at("max_internal_lines", in.diagram.max_internal_lines);
        d.try_load_at("max_external_lines", in.diagram.max_external_lines);
    }
    if (s.has("start")) {
        const auto st {s["start"]};
        st.try_load_at("random_start", in.start.random_start);
        st.try_load_at("n_internal_min", in.start.n_internal_min);
        st.try_load_at("n_internal_max", in.start.n_internal_max);
        st.try_load_at("n_external_min", in.start.n_external_min);
        st.try_load_at("n_external_max", in.start.n_external_max);
        st.try_load_at("w_spread", in.start.w_spread);
    }
    if (s.has("rng")) { s["rng"].try_load_at("seed", in.rng.seed); }
    if (s.has("measurements")) {
        const auto ms {s["measurements"]};
        ms.try_load_at("n_batches", in.measurements.n_batches);
        ms.try_load_at("green_n_bins", in.measurements.green_n_bins);
        ms.try_load_at("boundary_phonons", in.measurements.boundary_phonons);
        ms.try_load_at("split_sign", in.measurements.split_sign);
    }
    if (s.has("thermalization")) {
        const auto t {s["thermalization"]};
        t.try_load_at("enabled", in.thermalization.enabled);
        t.try_load_at("monitor_tau_D", in.thermalization.monitor_tau_D);
        t.try_load_at("monitor_sign", in.thermalization.monitor_sign);
        t.try_load_at("monitor_energy", in.thermalization.monitor_energy);
        t.try_load_at("monitor_orbital", in.thermalization.monitor_orbital);
        t.try_load_at("monitor_n_batches", in.thermalization.monitor_n_batches);
        t.try_load_at("rhat_max", in.thermalization.rhat_max);
        t.try_load_at("min_bulk_ess_per_split_chain", in.thermalization.min_bulk_ess_per_split_chain);
        t.try_load_at("min_tail_ess_per_split_chain", in.thermalization.min_tail_ess_per_split_chain);
        t.try_load_at("consecutive", in.thermalization.consecutive);
        t.try_load_at("root", in.thermalization.root);
        t.try_load_at("max_steps", in.thermalization.max_steps);
        t.try_load_at("checkpoint_interval", in.thermalization.checkpoint_interval);
        t.try_load_at("steps_per_cycle", in.thermalization.steps_per_cycle);
        t.try_load_at("cycles_per_check", in.thermalization.cycles_per_check);
    }
    if (s.has("output")) { s["output"].try_load_at("prefix", in.output.prefix); }
    if (s.has("checkpoint")) {
        const auto ck {s["checkpoint"]};
        ck.try_load_at("enabled", in.checkpoint.enabled);
        ck.try_load_at("interval", in.checkpoint.interval);
        ck.try_load_at("restart", in.checkpoint.restart);
    }
}

#endif // !RUN_INPUT_HPP
