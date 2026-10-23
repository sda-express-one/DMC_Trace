#include "io/run_input.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <nlohmann/json.hpp>
#include <simplemc/mc/json.hpp>
#include <simplemc/utils/simplemc_exception.hpp>
#include "comp_method/vertex_coupling.hpp"
#include "comp_method/weight_computation.hpp"
#include "updates/dmc_update_set.hpp"
#include "utils/numerical.hpp"

run_input::run_input(){
    for (const std::string_view name : update_names) { update_weights[std::string {name}] = 1.; }
    for (const std::string_view name : measurement_names) {
        measurements_active[std::string {name}] = name == "gs_energy" || name == "qp_weights";
    }
}

std::vector<PhononMode> run_input::phononModes() const {
    std::vector<PhononMode> modes;
    for (std::size_t i {0}; i < model.ph_energy.size(); ++i) { modes.push_back(PhononMode {model.ph_energy[i], model.diel_response[i]}); }
    return modes;
}

diagram_builder::RandomDiagramSpec run_input::randomDiagramSpec() const {
    return {.n_internal_min = start.n_internal_min, .n_internal_max = start.n_internal_max,
            .n_external_min = start.n_external_min, .n_external_max = start.n_external_max, .w_spread = start.w_spread};
}

convergence_monitor::Options run_input::monitorOptions() const {
    return {.tau_D = thermalization.monitor_tau_D, .sign = thermalization.monitor_sign, .energy = thermalization.monitor_energy,
            .orbital = thermalization.monitor_orbital, .n_batches = thermalization.monitor_n_batches};
}

thermalization_control::Options run_input::controlOptions() const {
    return {.rhat_max = thermalization.rhat_max, .min_bulk_ess_per_split_chain = thermalization.min_bulk_ess_per_split_chain,
            .min_tail_ess_per_split_chain = thermalization.min_tail_ess_per_split_chain,
            .consecutive = thermalization.consecutive, .root = thermalization.root};
}

simplemc::simulation_params run_input::thermalizationParams() const {
    simplemc::simulation_params p;
    p.max_steps = thermalization.max_steps;
    p.max_time = std::numeric_limits<double>::max();
    p.steps_per_cycle = thermalization.steps_per_cycle;
    p.cycles_per_check = thermalization.cycles_per_check;
    p.checkpoint_after_steps = thermalization.checkpoint_interval;
    return p;
}

namespace {
    bool contains(const auto & names, std::string_view name){
        return std::find(names.begin(), names.end(), name) != names.end();
    }

    // every object key of `doc` must exist in `ref` (the complete default file); arrays and values are leaves
    void unknownKeys(const nlohmann::json & doc, const nlohmann::json & ref, const std::string & where, std::vector<std::string> & out){
        for (const auto & [key, value] : doc.items()) {
            if (!ref.contains(key)) { out.push_back(where + "/" + key); continue; }
            if (value.is_object() && ref.at(key).is_object()) { unknownKeys(value, ref.at(key), where + "/" + key, out); }
        }
    }

    // the complete file, as written by input::write
    void save(simplemc::json_serializer & s, const simplemc::simulation_params & params, const run_input & in){
        simplemc_save_input_config(s["config"], in);
        simplemc::simplemc_save_input_config(s["params"], params);
        for (const auto & [name, w] : in.update_weights) { s["updates"][name].save_at("weight", w); }
        for (const auto & [name, a] : in.measurements_active) { s["measurements"][name].save_at("is_active", a); }
    }
}

namespace input {
    simplemc::simulation_params defaultParams(){
        simplemc::simulation_params p;
        p.max_time = 23.5 * 3600.;      // 84600 s: a 24 h cluster slot minus 30 min for startup and output
        p.steps_per_cycle = 10;
        p.cycles_per_check = 1000;
        return p;
    }

    void validate(const run_input & in){
        std::vector<std::string> err;
        const auto require {[&](bool ok, const std::string & what){ if (!ok) { err.push_back(what); } }};
        const auto finite {[](double x){ return std::isfinite(x); }};

        const auto & m {in.model};
        require(m.A_LK > 0. && m.B_LK > 0., "model: A_LK > 0 and B_LK > 0");
        require(m.C_LK < m.A_LK + m.B_LK && m.C_LK > -0.5 * m.A_LK - m.B_LK, "model: -A_LK/2 - B_LK < C_LK < A_LK + B_LK");
        require(m.V_unit_cell > 0. && m.V_BvK > 0., "model: V_unit_cell > 0 and V_BvK > 0");
        require(!m.ph_energy.empty() && m.ph_energy.size() == m.diel_response.size(),
                "model: ph_energy and diel_response non-empty and of equal length (one entry per phonon mode)");
        require(std::all_of(m.ph_energy.begin(), m.ph_energy.end(), [&](double x){ return finite(x) && x > 0.; }), "model: ph_energy > 0");
        require(std::all_of(m.diel_response.begin(), m.diel_response.end(), [&](double x){ return finite(x) && x > 0.; }),
                "model: diel_response > 0");

        const auto & d {in.diagram};
        require(std::all_of(d.p.begin(), d.p.end(), finite), "diagram: p finite");
        require(finite(d.tau_max) && d.tau_max > 0., "diagram: tau_max > 0");
        if (d.tau_fixed) {
            require(d.tau_D_start == d.tau_max, "diagram: tau_fixed requires tau_D_start == tau_max");
            require(d.mu == 0., "diagram: tau_fixed requires mu == 0 (e^{mu tau_D} is a constant there)");
        }
        else {
            require(d.mu < 0., "diagram: variable length requires mu < 0");
            require(d.tau_D_start > 0. && d.tau_D_start <= d.tau_max, "diagram: 0 < tau_D_start <= tau_max");
            require(d.tau_min >= 0. && d.tau_min < d.tau_max, "diagram: 0 <= tau_min < tau_max");
        }
        require(d.max_internal_lines >= 1 && d.max_external_lines >= 0, "diagram: max_internal_lines >= 1, max_external_lines >= 0");

        const auto & st {in.start};
        if (st.random_start) {
            require(0 <= st.n_internal_min && st.n_internal_min <= st.n_internal_max && st.n_internal_max <= d.max_internal_lines,
                    "start: 0 <= n_internal_min <= n_internal_max <= max_internal_lines");
            require(0 <= st.n_external_min && st.n_external_min <= st.n_external_max && st.n_external_max <= d.max_external_lines,
                    "start: 0 <= n_external_min <= n_external_max <= max_external_lines");
            require(st.w_spread > 0., "start: w_spread > 0");
        }

        const auto & ms {in.measurements};
        require(ms.n_batches >= 2, "measurements: n_batches >= 2");
        require(ms.green_n_bins >= 1, "measurements: green_n_bins >= 1");
        require(ms.boundary_phonons >= 0 && ms.boundary_phonons <= d.max_external_lines, "measurements: 0 <= boundary_phonons <= max_external_lines");

        const auto & t {in.thermalization};
        if (t.enabled) {
            require(t.min_bulk_ess_per_split_chain > 0. && t.min_tail_ess_per_split_chain > 0., "thermalization: ESS thresholds > 0");
            // the thermalization_control guard: right after a merge the window has monitor_n_batches/4 draws per split chain
            const double min_draws {static_cast<double>(t.monitor_n_batches) / 4.};
            require(min_draws >= t.min_bulk_ess_per_split_chain && min_draws >= t.min_tail_ess_per_split_chain,
                    "thermalization: monitor_n_batches >= 4 x the ESS thresholds (otherwise the test can never pass)");
            require(t.rhat_max > 1., "thermalization: rhat_max > 1");
            require(t.consecutive >= 1 && t.root >= 0, "thermalization: consecutive >= 1, root >= 0");
            require(!t.monitor_orbital || m.C_LK == 0., "thermalization: monitor_orbital requires C_LK == 0");
            require(t.max_steps > 0 && t.steps_per_cycle > 0 && t.cycles_per_check > 0,
                    "thermalization: max_steps, steps_per_cycle, cycles_per_check > 0");
            require(t.checkpoint_interval > 0 && t.steps_per_cycle * t.cycles_per_check > 0
                        && t.checkpoint_interval % (t.steps_per_cycle * t.cycles_per_check) == 0,
                    "thermalization: checkpoint_interval a positive multiple of steps_per_cycle x cycles_per_check");
        }

        require(!in.output.prefix.empty(), "output: prefix non-empty");

        // estimators: the conditions their constructors and measure() only assert
        const auto active {[&](std::string_view name){
            const auto it {in.measurements_active.find(std::string {name})};
            return it != in.measurements_active.end() && it->second;
        }};
        for (const auto & [name, a] : in.measurements_active) {
            require(contains(measurement_names, name), "measurements: unknown estimator '" + name + "'");
        }
        const bool p_zero {d.p[0] == 0. && d.p[1] == 0. && d.p[2] == 0.};
        require(!active("eff_mass") || (numerical::isEqual(m.A_LK, m.B_LK) && numerical::isEqual(m.C_LK, 0.)),
                "measurements: eff_mass requires A_LK == B_LK and C_LK == 0 (use eff_mass_uniaxial for C_LK == 0)");
        require(!active("eff_mass_uniaxial") || m.C_LK == 0., "measurements: eff_mass_uniaxial requires C_LK == 0");
        require(!(active("eff_mass") || active("eff_mass_uniaxial")) || p_zero || d.tau_fixed,
                "measurements: eff_mass at p != 0 requires tau_fixed (variable length: valid at p = 0 only)");
        require(!(active("green_func") || active("green_trace")) || !d.tau_fixed,
                "measurements: green_func and green_trace require variable length (no tau binning at fixed tau_D)");

        // update weights
        bool any {false}, any_length {false};
        for (const auto & [name, w] : in.update_weights) {
            require(contains(update_names, name), "updates: unknown update '" + name + "'");
            require(finite(w) && w >= 0., "updates: weight of '" + name + "' finite and >= 0");
            any = any || w > 0.;
            if (contains(length_changing_updates, name)) {
                any_length = any_length || w > 0.;
                require(!d.tau_fixed || w == 0., "updates: tau_fixed requires weight 0 for '" + name + "' (it changes tau_D)");
            }
        }
        require(any, "updates: at least one weight > 0");
        require(d.tau_fixed || any_length, "updates: variable length needs chg_tau, str_diagram or scl_diagram with weight > 0");
        for (const auto & [a, b] : update_inverse_pairs) {
            const auto wa {in.update_weights.find(std::string {a})}, wb {in.update_weights.find(std::string {b})};
            const double va {wa == in.update_weights.end() ? 1. : wa->second}, vb {wb == in.update_weights.end() ? 1. : wb->second};
            require((va == 0.) == (vb == 0.), "updates: '" + std::string {a} + "' and '" + std::string {b} + "' both zero or both non-zero");
        }

        if (!err.empty()) {
            std::string msg {"invalid input:"};
            for (const auto & e : err) { msg += "\n  - " + e; }
            throw simplemc::simplemc_exception(msg);
        }
    }

    void applyGlobals(const run_input & in){
        weight::LKMatrix::setLKParameters(in.model.A_LK, in.model.B_LK, in.model.C_LK);
        Coupling::Parameters::configParameters(in.model.V_unit_cell, in.model.V_BvK);
    }

    void load(const std::filesystem::path & path, simplemc::simulation_params & params, run_input & in){
        simplemc::load_json_file(path, [&](const simplemc::json_serializer & s){
            // unknown keys: compared with the complete default file
            simplemc::json_serializer ref;
            save(ref, defaultParams(), run_input {});
            std::vector<std::string> unknown;
            unknownKeys(s.root(), ref.root(), "", unknown);
            if (!unknown.empty()) {
                std::string msg {"unknown keys in " + path.string() + ":"};
                for (const auto & k : unknown) { msg += " " + k; }
                throw simplemc::simplemc_exception(msg);
            }

            if (s.has("config")) { simplemc_load_input_config(s["config"], in); }
            if (s.has("params")) { simplemc::simplemc_load_input_config(s["params"], params); }
            if (s.has("updates")) {
                const auto u {s["updates"]};
                for (const std::string_view name : update_names) {
                    if (u.has(name)) { u[name].try_load_at("weight", in.update_weights[std::string {name}]); }
                }
            }
            if (s.has("measurements")) {
                const auto ms {s["measurements"]};
                for (const std::string_view name : measurement_names) {
                    if (ms.has(name)) { ms[name].try_load_at("is_active", in.measurements_active[std::string {name}]); }
                }
            }
        });
        validate(in);
    }

    void write(const std::filesystem::path & path, const simplemc::simulation_params & params, const run_input & in){
        simplemc::save_json_file(path, [&](simplemc::json_serializer & s){ save(s, params, in); }, {.indent = 4});
    }

    void writeDefault(const std::filesystem::path & path){
        write(path, defaultParams(), run_input {});
    }

    void print(const run_input & in, std::FILE * fp){
        const auto & m {in.model};
        const auto & d {in.diagram};
        const auto & t {in.thermalization};
        std::fprintf(fp, "# model: A_LK = %.10g  B_LK = %.10g  C_LK = %.10g  V_unit_cell = %.10g  V_BvK = %.10g\n",
                     m.A_LK, m.B_LK, m.C_LK, m.V_unit_cell, m.V_BvK);
        for (std::size_t i {0}; i < m.ph_energy.size(); ++i) {
            std::fprintf(fp, "# phonon mode %zu: omega = %.10g  diel_response = %.10g\n", i, m.ph_energy[i], m.diel_response[i]);
        }
        std::fprintf(fp, "# diagram: p = (%.10g, %.10g, %.10g)  mu = %.10g  tau_max = %.10g  tau_fixed = %d  tau_D_start = %.10g  "
                         "tau_min = %.10g  max lines int/ext = %d/%d\n",
                     d.p[0], d.p[1], d.p[2], d.mu, d.tau_max, d.tau_fixed ? 1 : 0, d.tau_D_start, d.tau_min,
                     d.max_internal_lines, d.max_external_lines);
        std::fprintf(fp, "# start: random = %d  internal %d..%d  external %d..%d  w_spread = %.10g\n", in.start.random_start ? 1 : 0,
                     in.start.n_internal_min, in.start.n_internal_max, in.start.n_external_min, in.start.n_external_max, in.start.w_spread);
        std::fprintf(fp, "# rng: seed = %llu\n", static_cast<unsigned long long>(in.rng.seed));
        std::fprintf(fp, "# measurements: n_batches = %zu  green_n_bins = %d  boundary_phonons = %d  split_sign = %d\n",
                     in.measurements.n_batches, in.measurements.green_n_bins, in.measurements.boundary_phonons,
                     in.measurements.split_sign ? 1 : 0);
        std::fprintf(fp, "# thermalization: enabled = %d  monitor tau_D/sign/energy/orbital = %d/%d/%d/%d  monitor_n_batches = %zu  rhat_max = %.4f  "
                         "ESS bulk/tail per split chain = %.1f/%.1f  consecutive = %d  root = %d\n",
                     t.enabled ? 1 : 0, t.monitor_tau_D ? 1 : 0, t.monitor_sign ? 1 : 0, t.monitor_energy ? 1 : 0, t.monitor_orbital ? 1 : 0, t.monitor_n_batches,
                     t.rhat_max, t.min_bulk_ess_per_split_chain, t.min_tail_ess_per_split_chain, t.consecutive, t.root);
        std::fprintf(fp, "# thermalization: max_steps = %llu  checkpoint_interval = %llu  steps_per_cycle = %llu  cycles_per_check = %llu\n",
                     static_cast<unsigned long long>(t.max_steps), static_cast<unsigned long long>(t.checkpoint_interval),
                     static_cast<unsigned long long>(t.steps_per_cycle), static_cast<unsigned long long>(t.cycles_per_check));
        std::fprintf(fp, "# update weights:");
        for (const std::string_view name : update_names) {
            const auto it {in.update_weights.find(std::string {name})};
            std::fprintf(fp, " %.*s = %g", static_cast<int>(name.size()), name.data(), it == in.update_weights.end() ? 1. : it->second);
        }
        std::fprintf(fp, "\n# active estimators:");
        for (const std::string_view name : measurement_names) {
            const auto it {in.measurements_active.find(std::string {name})};
            if (it != in.measurements_active.end() && it->second) { std::fprintf(fp, " %.*s", static_cast<int>(name.size()), name.data()); }
        }
        std::fprintf(fp, "\n# output: prefix = %s\n", in.output.prefix.c_str());
    }
}
