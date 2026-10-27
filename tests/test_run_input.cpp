// The run input (io/run_input.hpp) and the weighted update set (updates/dmc_update_set.hpp):
//   - the default file round-trips (config, params, update weights);
//   - a partial file changes only the keys it names; an unknown key is an error;
//   - validate rejects every inconsistent setting it is meant to catch;
//   - applyGlobals sets LKMatrix and Coupling::Parameters;
//   - makeUpdateSet links the add/rm pairs with the detailed-balance factor w_inverse/w;
//   - the repository's inputs/*.json load and validate, and inputs/default.json matches the current defaults;
//   - a simplemc::metropolis_kernel run with unequal pair weights samples the same distribution as with equal
//     weights (<n_int>, <n_ext> at alpha = 0.5, fixed tau_D, mu = 0).
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <functional>
#include <string>
#include <vector>
#include <simplemc/mc/json.hpp>
#include <simplemc/mc/kernels.hpp>
#include "test_common.hpp"
#include "io/run_input.hpp"
#include "updates/dmc_update_set.hpp"
#include "comp_method/vertex_coupling.hpp"
#include "diagram/diagram_builder.hpp"

namespace fs = std::filesystem;

// set by tests/makefile; the fallback (the repository seen from tests/, where make -C tests runs) keeps editors happy
#ifndef DMC_ROOT
#define DMC_ROOT ".."
#endif

static nlohmann::json document(const simplemc::simulation_params & params, const run_input & in){
    const fs::path tmp {fs::temp_directory_path() / "dmc_test_run_input_doc.json"};
    input::write(tmp, params, in);
    nlohmann::json j;
    std::ifstream(tmp) >> j;
    fs::remove(tmp);
    return j;
}

static void writeText(const fs::path & path, const std::string & text){ std::ofstream(path) << text; }

static bool throws(const std::function<void()> & f){
    try { f(); }
    catch (const std::exception &) { return true; }
    return false;
}

// <n_int>, <n_ext> and their errors (64 batch means) from a metropolis_kernel run with the given weights
struct Orders { double n_int, e_int, n_ext, e_ext; };

static Orders sample(unsigned long long seed, const std::map<std::string, double> & weights, long steps){
    const double L {10.};
    test::Diagram d(seed, {0., 0., 0.}, 200, 16, L, 0., 1., 1.);     // alpha = 0.5 at A = B = 1, mu = 0
    diagram_builder::reset(d.cfg, L);
    diagram_builder::assignMomenta(d.cfg, {0., 0., 0.});
    diagram_builder::finalize(d.cfg);
    dmc_update_set us {makeUpdateSet(&d.cfg, &d.rng, weights)};
    simplemc::metropolis_kernel kernel {us};
    kernel.prepare();

    for (long i {0}; i < steps / 10; ++i) { kernel.step(d.rng); }      // burn-in
    const int B {64};
    const long per_batch {steps / B};
    std::vector<double> bi, be;
    for (int b {0}; b < B; ++b) {
        double si {0.}, se {0.};
        for (long i {0}; i < per_batch; ++i) {
            kernel.step(d.rng);
            si += static_cast<double>(d.cfg.internal_ph_manager->current_length) / 2;
            se += static_cast<double>(d.cfg.external_ph_manager->current_length) / 2;
        }
        bi.push_back(si / static_cast<double>(per_batch));
        be.push_back(se / static_cast<double>(per_batch));
    }
    const auto stats {[&](const std::vector<double> & x, double & mean, double & err){
        mean = 0.;
        for (const double v : x) { mean += v; }
        mean /= B;
        double var {0.};
        for (const double v : x) { var += (v - mean) * (v - mean); }
        err = std::sqrt(var / (B - 1.) / B);
    }};
    Orders o {};
    stats(bi, o.n_int, o.e_int);
    stats(be, o.n_ext, o.e_ext);
    return o;
}

int main(){
    test::Checks check {"run input (simplemc JSON input-config) and weighted update set"};
    const fs::path dir {fs::temp_directory_path()};

    // 1. defaults round trip
    {
        const fs::path f {dir / "dmc_test_run_input_default.json"};
        input::writeDefault(f);
        simplemc::simulation_params params;
        params.max_time = 1.;           // overwritten by the file
        run_input in;
        in.diagram.mu = -7.;
        input::load(f, params, in);
        const bool same {document(params, in) == document(input::defaultParams(), run_input {})};
        check(same && params.max_time == input::defaultParams().max_time && in.update_weights.size() == update_names.size(),
              "writeDefault -> load reproduces the defaults (config, params, %zu update weights)", in.update_weights.size());
        fs::remove(f);
    }

    // 2. partial file and unknown keys
    {
        const fs::path f {dir / "dmc_test_run_input_partial.json"};
        writeText(f, R"({"config": {"model": {"A_LK": 4.681, "B_LK": 1.019, "C_LK": 0.0, "ph_energy": [0.5, 0.25], "diel_response": [2.0, 3.0]},
                                    "diagram": {"tau_fixed": true, "tau_max": 100.0, "tau_D_start": 100.0, "mu": 0.0}},
                         "params": {"max_steps": 12345},
                         "updates": {"chg_tau": {"weight": 0}, "str_diagram": {"weight": 0}, "scl_diagram": {"weight": 0},
                                     "swp_ph": {"weight": 0.25}}})");
        simplemc::simulation_params params {input::defaultParams()};     // as main starts
        run_input in;
        input::load(f, params, in);
        const run_input def;
        const bool set {in.model.A_LK == 4.681 && in.model.ph_energy.size() == 2 && in.model.diel_response[1] == 3.
                        && in.diagram.tau_fixed && in.diagram.tau_max == 100. && params.max_steps == 12345
                        && in.update_weights.at("swp_ph") == 0.25 && in.update_weights.at("chg_tau") == 0.};
        const bool kept {in.diagram.max_internal_lines == def.diagram.max_internal_lines && in.thermalization.rhat_max == def.thermalization.rhat_max
                         && in.update_weights.at("add_int_ph") == 1. && params.steps_per_cycle == input::defaultParams().steps_per_cycle
                         && in.output.prefix == def.output.prefix};
        check(set && kept, "a partial file sets the keys it names (2 phonon modes, fixed tau, weights) and keeps every other default");

        writeText(f, R"({"config": {"diagram": {"tau_fixd": true}}})");
        const bool typo {throws([&]{ run_input x; simplemc::simulation_params p; input::load(f, p, x); })};
        writeText(f, R"({"updates": {"add_internal": {"weight": 2}}})");
        const bool bad_update {throws([&]{ run_input x; simplemc::simulation_params p; input::load(f, p, x); })};
        writeText(f, R"({"measurements": {"gs_energy_x": {"is_active": true}}})");
        const bool bad_meas {throws([&]{ run_input x; simplemc::simulation_params p; input::load(f, p, x); })};
        writeText(f, R"({"measurements": {"qp_weights": {"is_active": false}, "eff_mass": {"is_active": true}}, "params": {"max_time": 5}})");
        run_input meas;
        simplemc::simulation_params mp;
        const bool meas_ok {!throws([&]{ input::load(f, mp, meas); }) && !meas.measurements_active.at("qp_weights")
                            && meas.measurements_active.at("eff_mass") && meas.measurements_active.at("gs_energy")
                            && !meas.measurements_active.at("green_func")};
        check(typo && bad_update && bad_meas && meas_ok,
              "unknown keys are rejected (config/diagram/tau_fixd, updates/add_internal, measurements/gs_energy_x); "
              "is_active flags are read, the others keep their defaults");
        fs::remove(f);
    }

    // 3. validate
    {
        const auto rejected {[](const std::function<void(run_input &)> & change){
            run_input in;
            change(in);
            return throws([&]{ input::validate(in); });
        }};
        int ok {0}, total {0};
        const auto expect {[&](bool rej, const char * what){ ++total; if (rej) { ++ok; } else { std::printf("    not rejected: %s\n", what); } }};
        expect(!rejected([](run_input &){}), "defaults (must pass)");
        expect(rejected([](run_input & in){ in.model.B_LK = -1.; }), "B_LK < 0");
        expect(rejected([](run_input & in){ in.model.C_LK = 3.; }), "C_LK >= A + B");
        expect(rejected([](run_input & in){ in.model.C_LK = -1.6; }), "C_LK <= -A/2 - B");
        expect(rejected([](run_input & in){ in.model.diel_response = {1., 2.}; }), "mode arrays of different length");
        expect(rejected([](run_input & in){ in.model.ph_energy = {0.}; }), "omega = 0");
        expect(rejected([](run_input & in){ in.diagram.mu = 0.; }), "variable length with mu = 0");
        expect(rejected([](run_input & in){ in.diagram.tau_D_start = 60.; }), "tau_D_start > tau_max");
        expect(rejected([](run_input & in){
            in.diagram.tau_fixed = true; in.diagram.tau_D_start = in.diagram.tau_max; in.diagram.mu = -1.;
            for (const auto n : length_changing_updates) { in.update_weights[std::string {n}] = 0.; } }), "fixed tau with mu != 0");
        expect(rejected([](run_input & in){ in.diagram.tau_fixed = true; in.diagram.tau_D_start = in.diagram.tau_max; in.diagram.mu = 0.; }),
               "fixed tau with chg_tau weight > 0");
        expect(!rejected([](run_input & in){
            in.diagram.tau_fixed = true; in.diagram.tau_D_start = in.diagram.tau_max; in.diagram.mu = 0.;
            for (const auto n : length_changing_updates) { in.update_weights[std::string {n}] = 0.; } }), "consistent fixed tau (must pass)");
        expect(rejected([](run_input & in){ for (const auto n : length_changing_updates) { in.update_weights[std::string {n}] = 0.; } }),
               "variable length without a length-changing update");
        expect(rejected([](run_input & in){ in.update_weights["rm_ext_ph"] = 0.; }), "one of the add/rm pair zero");
        expect(rejected([](run_input & in){ in.update_weights["mv_tau"] = -1.; }), "negative weight");
        expect(rejected([](run_input & in){ in.thermalization.monitor_n_batches = 32; }), "monitor_n_batches below 4 x ESS per split chain");
        expect(rejected([](run_input & in){ in.thermalization.min_tail_ess = 0.; }), "total ESS threshold <= 0");
        expect(!rejected([](run_input & in){ in.thermalization.monitor_n_batches = 256; }),
               "monitor_n_batches = 256 with the default floor: above max_ess_fraction, only warned about (must pass)");
        expect(rejected([](run_input & in){ in.thermalization.max_ess_fraction = 0.; }), "max_ess_fraction = 0");
        expect(rejected([](run_input & in){ in.thermalization.max_ess_fraction = 1.5; }), "max_ess_fraction > 1");
        expect(rejected([](run_input & in){ in.thermalization.checkpoint_interval = 15000; }), "checkpoint interval not a multiple of a check");
        expect(rejected([](run_input & in){ in.model.C_LK = 0.5; in.thermalization.monitor_orbital = true; }), "orbital monitor with C != 0");
        expect(rejected([](run_input & in){ in.start.random_start = true; in.start.n_internal_max = 500; }), "random start beyond the capacity");
        expect(!rejected([](run_input & in){ in.measurements.boundary_phonons = -1; }), "boundary_phonons < 0, all sectors (must pass)");
        expect(rejected([](run_input & in){ in.measurements.boundary_phonons = in.diagram.max_external_lines + 1; }),
               "boundary_phonons beyond max_external_lines");
        expect(rejected([](run_input & in){ in.model.A_LK = 2.; in.measurements_active["eff_mass"] = true; }), "eff_mass with A != B");
        expect(!rejected([](run_input & in){ in.model.A_LK = 2.; in.measurements_active["eff_mass_uniaxial"] = true; }),
               "eff_mass_uniaxial with A != B, C = 0 (must pass)");
        expect(rejected([](run_input & in){ in.model.C_LK = 0.5; in.measurements_active["eff_mass_uniaxial"] = true; }),
               "eff_mass_uniaxial with C != 0");
        expect(rejected([](run_input & in){ in.diagram.p = {0.1, 0., 0.}; in.measurements_active["eff_mass"] = true; }),
               "eff_mass at p != 0 with variable length");
        expect(rejected([](run_input & in){
            in.diagram.tau_fixed = true; in.diagram.tau_D_start = in.diagram.tau_max; in.diagram.mu = 0.;
            for (const auto n : length_changing_updates) { in.update_weights[std::string {n}] = 0.; }
            in.measurements_active["green_trace"] = true; }), "green_trace at fixed tau");
        check(ok == total, "validate: %d/%d settings classified correctly (4 valid, the rest rejected)", ok, total);
    }

    // 4. globals
    {
        run_input in;
        in.model.A_LK = 4.681; in.model.B_LK = 1.019; in.model.C_LK = 5.498; in.model.V_unit_cell = 2.5; in.model.V_BvK = 7.;
        input::applyGlobals(in);
        const bool set {weight::LKMatrix::A_LK == 4.681 && weight::LKMatrix::B_LK == 1.019 && weight::LKMatrix::C_LK == 5.498
                        && Coupling::Parameters::V_unit_cell == 2.5 && Coupling::Parameters::V_BvK == 7.};
        input::applyGlobals(run_input {});
        check(set && weight::LKMatrix::A_LK == 1. && Coupling::Parameters::V_unit_cell == 1., "applyGlobals sets LKMatrix and Coupling::Parameters");
    }

    // 5. update set: pairs and detailed-balance factors
    test::setLK({1., 1., 0.});
    {
        test::Diagram d(1, {0., 0., 0.}, 20, 4);
        const dmc_update_set us {makeUpdateSet(&d.cfg, &d.rng, {{"add_int_ph", 3.}, {"rm_int_ph", 1.}, {"add_ext_ph", 0.5}})};
        double r_add {0.}, r_rm {0.}, r_adde {0.}, r_rme {0.}, r_swp {0.};
        std::string inv_add;
        for (const auto & s : us.stats()) {
            if (s.name == "add_int_ph") { r_add = s.ratio; inv_add = s.inv_name; }
            if (s.name == "rm_int_ph") { r_rm = s.ratio; }
            if (s.name == "add_ext_ph") { r_adde = s.ratio; }
            if (s.name == "rm_ext_ph") { r_rme = s.ratio; }
            if (s.name == "swp_ph") { r_swp = s.ratio; }
        }
        const bool pair_zero {throws([&]{ (void) makeUpdateSet(&d.cfg, &d.rng, {{"rm_int_ph", 0.}}); })};
        check(inv_add == "rm_int_ph" && r_add == 1. / 3. && r_rm == 3. && r_adde == 2. && r_rme == 0.5 && r_swp == 1. && pair_zero,
              "makeUpdateSet: add/rm linked, ratios w_inv/w = %.4f, %.4f (int), %.4f, %.4f (ext), self-inverse 1; a half-zero pair throws",
              r_add, r_rm, r_adde, r_rme);
    }

    // 6. weighted kernel samples the same distribution
    {
        std::map<std::string, double> equal, skewed;
        for (const auto n : update_names) { equal[std::string {n}] = 1.; }
        for (const auto n : length_changing_updates) { equal[std::string {n}] = 0.; }
        skewed = equal;
        skewed["add_int_ph"] = 3.;
        skewed["rm_ext_ph"] = 2.5;
        skewed["chg_ph_energy"] = 0.2;
        const long steps {4'000'000};
        const Orders a {sample(11, equal, steps)}, b {sample(12, skewed, steps)};
        const double zi {(a.n_int - b.n_int) / std::hypot(a.e_int, b.e_int)}, ze {(a.n_ext - b.n_ext) / std::hypot(a.e_ext, b.e_ext)};
        check(std::abs(zi) < 4. && std::abs(ze) < 4.,
              "metropolis_kernel, alpha = 0.5, tau_D = 10: <n_int> %.4f +- %.4f (equal weights) vs %.4f +- %.4f (add_int x3) [%.1f sigma]; "
              "<n_ext> %.4f +- %.4f vs %.4f +- %.4f (rm_ext x2.5) [%.1f sigma]",
              a.n_int, a.e_int, b.n_int, b.e_int, zi, a.n_ext, a.e_ext, b.n_ext, b.e_ext, ze);
    }

    // 7. the repository's input files (inputs/**/*.json): every one loads and validates (no stale or unknown key),
    //    and inputs/default.json is exactly what writeDefault produces now (no key missing, no default changed)
    {
        const fs::path inputs {fs::path(DMC_ROOT) / "inputs"};
        int files {0};
        std::vector<std::string> failed;
        if (fs::is_directory(inputs)) {
            for (const auto & e : fs::recursive_directory_iterator(inputs)) {
                if (!e.is_regular_file() || e.path().extension() != ".json") { continue; }
                ++files;
                try { run_input x; simplemc::simulation_params p; input::load(e.path(), p, x); }
                catch (const std::exception & ex) { failed.push_back(e.path().filename().string() + ": " + ex.what()); }
            }
        }
        for (const auto & f : failed) { std::printf("    %s\n", f.c_str()); }
        nlohmann::json def;
        const fs::path def_path {inputs / "default.json"};
        if (fs::exists(def_path)) { std::ifstream(def_path) >> def; }
        const bool current {def == document(input::defaultParams(), run_input {})};
        check(files > 0 && failed.empty() && current,
              "inputs/: %d file(s) load and validate (%zu failed); default.json %s (regenerate with bin/diagramc inputs/default.json)",
              files, failed.size(), current ? "matches the current defaults" : "is OUT OF DATE or missing");
    }
    return check.exit_code();
}
