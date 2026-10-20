// diagramc <input.json>
//
// One Markov chain per MPI rank: the input is read (io/run_input.hpp), each rank builds its own diagram, the
// chains are thermalized under R-hat control (the ranks are the chains: measurements/thermalization_control.hpp),
// then the production run measures the active estimators, the ranks are combined and the root writes
//     <prefix>_<estimator>.dat   one per active estimator (its write())
//     <prefix>_thermalization.dat  the R-hat checkpoints
//     <prefix>_run.dat           input, run parameters, update statistics, sign, thermalization outcome
//     <prefix>_input.json        the input actually used (every key, defaults filled in)
// Without an argument, or with a file that does not exist, the root writes the default input to that path
// (input.json without an argument) and the program exits.
//
// Wall time: params.max_time is the budget of the WHOLE job (default 84600 s = 23 h 30 min, input::defaultParams).
// The thermalization is ended collectively at the first R-hat checkpoint past it; the production gets what is left
// (skipped, exit code 3, if less than a minute). <prefix>_run.dat records the budget and why each phase ended.
//
// Starts: even ranks from the bare propagator, odd ranks from a random diagram (start.random_start), so that the
// chains approach the target from both sides. If R-hat does not pass before the thermalization cap, production
// runs anyway, with a warning and "thermalized: no" in <prefix>_run.dat.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <filesystem>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
#include <mpi.h>
#include <simplemc/mc/json.hpp>
#include <simplemc/mc/kernels.hpp>
#include <simplemc/mc/measurement_set.hpp>
#include <simplemc/mc/progress_printer.hpp>
#include <simplemc/mc/run.hpp>
#include <simplemc/mc/simulation_stats.hpp>
#include <simplemc/mc/utils.hpp>
#include <simplemc/mpi/all_reduce.hpp>
#include <simplemc/mpi/broadcast.hpp>
#include <simplemc/mpi/communicator.hpp>
#include <simplemc/mpi/environment.hpp>
#include <simplemc/random/seed_rng.hpp>
#include <simplemc/random/xoshiro256.hpp>
#include "diagram/diagram_builder.hpp"
#include "diagram/diagram_config.hpp"
#include "diagram/phonon_manager.hpp"
#include "diagram/vertex_manager.hpp"
#include "io/run_input.hpp"
#include "measurements/convergence_monitor.hpp"
#include "measurements/eff_mass.hpp"
#include "measurements/eff_mass_uniaxial.hpp"
#include "measurements/green_func.hpp"
#include "measurements/green_trace.hpp"
#include "measurements/gs_energy.hpp"
#include "measurements/qp_weights.hpp"
#include "measurements/thermalization_control.hpp"
#include "updates/dmc_update_set.hpp"

namespace {
    // The chain's state. Members in construction order (the managers and the config point at the members before
    // them); never moved, since the updates and the estimators keep pointers to cfg.
    struct Chain {
        simplemc::xoshiro256ss rng;
        VertexPointerManager internal;
        VertexPointerManager external;
        PhononModeManager modes;
        diagram_cfg cfg;

        Chain(const run_input & in, int rank)
            : rng(),
              internal(2 * in.diagram.max_internal_lines, &rng),
              external(2 * in.diagram.max_external_lines, &rng),
              modes(static_cast<int>(in.model.ph_energy.size()), in.phononModes(), &rng),
              cfg(in.diagram.p, in.diagram.tau_max, in.diagram.mu, &internal, &external, &modes)
        {
            simplemc::seed_rng(rng, rank, in.rng.seed);
        }

        Chain(const Chain &) = delete;
        Chain & operator=(const Chain &) = delete;
    };

    // the starting diagram: even ranks the bare propagator, odd ranks a random one (if start.random_start)
    void start(Chain & c, const run_input & in, int rank){
        const double tau_D {in.diagram.tau_D_start};
        if (in.start.random_start && rank % 2 == 1) {
            simplemc::xoshiro256ss builder;
            simplemc::seed_rng(builder, rank, in.rng.seed ^ 0x5bd1e9955bd1e995ULL);
            const numerical::SanitizeReport rep {diagram_builder::randomDiagram(c.cfg, builder, in.diagram.p, tau_D,
                                                                                in.randomDiagramSpec())};
            const double trace {c.cfg.diagram_head->right_component.trace()};
            if (!rep.clean() || !std::isfinite(trace) || trace == 0.) {
                throw std::runtime_error("rank " + std::to_string(rank) + ": no usable random start diagram (narrow the start ranges)");
            }
        }
        else {
            diagram_builder::reset(c.cfg, tau_D);
            diagram_builder::assignMomenta(c.cfg, in.diagram.p);
            if (!diagram_builder::finalize(c.cfg).clean()) {
                throw std::runtime_error("rank " + std::to_string(rank) + ": the bare propagator is not a clean diagram");
            }
        }
    }

    simplemc::progress_printer progress(const char * label, std::uint64_t max_steps, double max_time, bool root){
        // budgets: 0 disables one. The step budget is shown when there is one (production runs are capped by steps;
        // a time limit is then only a safeguard), else the time budget. One line per 5%, no '\r' redraws: the output
        // usually goes to a log file (and under mpirun stdout is a pty even then, so isatty cannot tell).
        const std::uint64_t steps {max_steps == std::numeric_limits<std::uint64_t>::max() ? 0 : max_steps};
        const double time {steps > 0 ? 0. : max_time};
        return simplemc::progress_printer {simplemc::progress_line {.prefix = label, .max_steps = steps, .max_time = time,
                                                                     .show_bar = steps > 0 || time > 0.},
                                           false, root, 0.05};
    }

    std::string outPath(const run_input & in, const std::string & what){ return in.output.prefix + "_" + what; }

    int run(int argc, char ** argv, const simplemc::mpi::communicator & comm){
        const double t0 {MPI_Wtime()};      // start of the job: params.max_time is the budget of the whole job
        const int rank {comm.rank()};
        const bool root {rank == 0};
        const std::filesystem::path path {argc > 1 ? argv[1] : "input.json"};

        // 1. input: written with the defaults if missing
        int have_input {root && std::filesystem::exists(path) ? 1 : 0};
        simplemc::mpi::broadcast(have_input, 0, comm);
        if (have_input == 0) {
            if (root) {
                input::writeDefault(path);
                std::printf("diagramc: wrote the default input to %s - edit it and run again\n", path.string().c_str());
            }
            return 0;
        }
        simplemc::simulation_params params {input::defaultParams()};     // keys missing from the file keep these
        run_input in;
        input::load(path, params, in);
        input::applyGlobals(in);
        if (root) {
            const std::filesystem::path dir {std::filesystem::path(in.output.prefix).parent_path()};
            if (!dir.empty()) { std::filesystem::create_directories(dir); }
            input::write(outPath(in, "input.json"), params, in);
            std::printf("diagramc: %d chain(s), input %s\n", comm.size(), path.string().c_str());
            input::print(in, stdout);
        }

        // 2. chain, start, updates
        Chain chain(in, rank);
        diagram_cfg & cfg {chain.cfg};
        start(chain, in, rank);
        dmc_update_set us {makeUpdateSet(&cfg, &chain.rng, in.update_weights)};
        simplemc::metropolis_kernel kernel {us};

        // 3. thermalization under R-hat control, ended also (collectively, by the root's clock) by the wall-time budget
        const double budget {params.max_time};
        bool thermalized {false}, therm_timed_out {false};
        std::uint64_t therm_steps {0};
        double therm_time {0.};
        if (in.thermalization.enabled) {
            simplemc::measurement_set therm {simplemc::measurement {convergence_monitor {&cfg, in.monitorOptions()}, "convergence"}};
            convergence_monitor & mon {therm.get<0>().value()};
            thermalization_control ctl {comm, mon.names(), mon.maxWindowBatches(), in.controlOptions()};
            ctl.setDeadline(t0 + budget);
            const simplemc::simulation_params tp {in.thermalizationParams()};
            simplemc::progress_printer printer {progress("[thermalization]", tp.max_steps, 0., root)};
            const simplemc::simulation_ctx ctx {simplemc::run(chain.rng, kernel, therm, tp, simplemc::run_callbacks {
                .on_cycle = [&](const simplemc::simulation_ctx & c){ printer(c); },
                .on_checkpoint = [&](const simplemc::simulation_ctx & c){ ctl.checkpoint(mon, c.steps_done); },
                .stop_when = [&](const simplemc::simulation_ctx &){ return ctl.stopped(); }})};
            thermalized = ctl.thermalized();
            therm_timed_out = ctl.timedOut();
            therm_steps = ctx.steps_done;
            therm_time = ctx.runtime;
            if (root) {
                ctl.write(outPath(in, "thermalization.dat"));
                if (thermalized) {
                    std::printf("\ndiagramc: thermalized after %llu steps per chain\n", static_cast<unsigned long long>(therm_steps));
                }
                else {
                    std::printf("\ndiagramc: WARNING - R-hat did not pass within %llu steps per chain (%s); production runs anyway "
                                "if time is left (see %s)\n", static_cast<unsigned long long>(therm_steps),
                                therm_timed_out ? "wall-time budget reached" : "step cap reached", outPath(in, "thermalization.dat").c_str());
                }
            }
            us.reset_counters();
            cfg.n_negative = 0;
        }

        // 4. production: every estimator is built (fixed types), the input selects the active ones. Built after the
        //    start, so that eff_mass* capture the final tau_D of fixed-length runs.
        const auto & d {in.diagram};
        const std::size_t nb {in.measurements.n_batches};
        simplemc::measurement_set ms {
            simplemc::measurement {gs_energy_measurement {&cfg, d.p, d.tau_min, d.tau_fixed, nb}, "gs_energy"},
            simplemc::measurement {qp_weights_measurement {&cfg, d.p, d.tau_min, d.tau_fixed, nb}, "qp_weights"},
            simplemc::measurement {eff_mass_measurement {&cfg, d.p, d.tau_min, d.tau_fixed, nb}, "eff_mass"},
            simplemc::measurement {eff_mass_uniaxial_measurement {&cfg, d.p, d.tau_min, d.tau_fixed, nb}, "eff_mass_uniaxial"},
            simplemc::measurement {green_func_measurement {&cfg, d.p, in.measurements.green_n_bins, in.measurements.boundary_phonons, nb},
                                   "green_func"},
            simplemc::measurement {green_trace_measurement {&cfg, d.p, in.measurements.green_n_bins, in.measurements.boundary_phonons,
                                                            in.measurements.split_sign, nb}, "green_trace"}};
        for (const auto & [name, active] : in.measurements_active) { ms.set_active(name, active); }

        // the production gets what is left of the budget, the same value on every rank (the root's clock); it has no
        // collectives until the end, so ranks may stop at different step counts (the final collect merges them)
        double remaining {budget - (MPI_Wtime() - t0)};
        simplemc::mpi::broadcast(remaining, 0, comm);
        constexpr double min_production_time {60.};
        const bool produce {remaining >= min_production_time};
        simplemc::simulation_params prod {params};
        prod.max_time = std::max(remaining, 0.);
        simplemc::simulation_stats stats;
        std::uint64_t prod_steps {0};
        if (produce) {
            simplemc::progress_printer printer {progress("[production]", prod.max_steps, prod.max_time, root)};
            const simplemc::simulation_ctx ctx {simplemc::run(chain.rng, kernel, ms, prod, simplemc::run_callbacks {
                .on_step = [&](const simplemc::simulation_ctx &){ cfg.countSign(); },
                .on_cycle = [&](const simplemc::simulation_ctx & c){ printer(c); }})};
            prod_steps = ctx.steps_done;
            stats += ctx;
            if (root) { std::printf("\n"); }
        }
        else if (root) {
            std::printf("diagramc: WARNING - only %.0f s of the %.0f s wall-time budget left after thermalization: production "
                        "skipped\n", std::max(remaining, 0.), budget);
        }
        const char * prod_end {!produce ? "skipped (budget used up)"
                               : prod_steps >= prod.max_steps ? "max_steps reached" : "wall-time budget reached"};

        // 5. combine the ranks (collective: every rank, the same active estimators) and write
        std::uint64_t n_negative {cfg.n_negative};
        simplemc::mpi::all_reduce_in_place(n_negative, MPI_SUM, comm);
        simplemc_mpi_collect(comm, us);
        simplemc_mpi_collect(comm, stats);
        int failed {0};
        ms.for_each([&](auto & m){
            if (!produce || !m.is_active()) { return; }
            try {
                simplemc_mpi_collect(comm, m.value());
                if (root) { m.value().write(outPath(in, m.name() + ".dat")); }
            }
            catch (const std::exception & e) {     // e.g. fewer than 2 batches: the others are still written
                ++failed;
                if (root) { std::fprintf(stderr, "diagramc: %s not written: %s\n", m.name().c_str(), e.what()); }
            }
        });

        if (root) {
            const std::string run_path {outPath(in, "run.dat")};
            std::FILE * f {std::fopen(run_path.c_str(), "w")};
            if (f == nullptr) { throw std::runtime_error("cannot open " + run_path); }
            std::fprintf(f, "# diagramc run: %d chain(s), input %s\n", comm.size(), path.string().c_str());
            input::print(in, f);
            std::fprintf(f, "\n# thermalized: %s, %llu steps per chain, %.1f s\n",
                         !in.thermalization.enabled ? "not checked (thermalization disabled)"
                         : thermalized ? "yes (R-hat passed)" : therm_timed_out ? "NO (wall-time budget reached)" : "NO (step cap reached)",
                         static_cast<unsigned long long>(therm_steps), therm_time);
            std::fprintf(f, "# wall-time budget = %.0f s for the whole job; production limit = %.0f s; production: %s\n",
                         budget, prod.max_time, prod_end);
            std::fprintf(f, "\n# production parameters (max_time: what was left of the budget)\n");
            simplemc::print(prod, f);
            std::fprintf(f, "\n# production, summed over the chains\n");
            simplemc::print(stats, f);
            std::fprintf(f, "Negative-sign steps = %llu (fraction %.6g)\n", static_cast<unsigned long long>(n_negative),
                         stats.cumulative_steps > 0 ? static_cast<double>(n_negative) / static_cast<double>(stats.cumulative_steps) : 0.);
            std::fprintf(f, "\n# updates, summed over the chains\n");
            simplemc::print(us.stats(), f);
            std::fclose(f);

            // headline results
            if (in.measurements_active.at("gs_energy")) {
                if (const auto * e {ms.get<gs_energy_measurement>("gs_energy")}; e != nullptr && e->acc.batches().size() >= 2) {
                    const auto r {e->normalised()};
                    std::printf("E0 = %.10Lg +- %.3Lg\n", r.energy, r.energy_error);
                }
            }
            if (in.measurements_active.at("eff_mass")) {
                if (const auto * e {ms.get<eff_mass_measurement>("eff_mass")}; e != nullptr && e->acc.batches().size() >= 2) {
                    const auto r {e->normalised()};
                    std::printf("m*/m = %.8Lg +- %.3Lg\n", r.mass_ratio, r.mass_ratio_error);
                }
            }
            if (in.measurements_active.at("eff_mass_uniaxial")) {
                if (const auto * e {ms.get<eff_mass_uniaxial_measurement>("eff_mass_uniaxial")}; e != nullptr && e->acc.batches().size() >= 2) {
                    const auto r {e->normalised()};
                    std::printf("m*_par/m_par = %.8Lg +- %.3Lg   m*_perp/m_perp = %.8Lg +- %.3Lg\n",
                                r.mass_ratio_par, r.mass_ratio_par_error, r.mass_ratio_perp, r.mass_ratio_perp_error);
                }
            }
            std::printf("diagramc: %llu steps in %.1f s (summed over the chains); results in %s_*\n",
                        static_cast<unsigned long long>(stats.cumulative_steps), stats.cumulative_time, in.output.prefix.c_str());
        }
        return !produce ? 3 : failed == 0 ? 0 : 2;
    }
}

int main(int argc, char ** argv){
    simplemc::mpi::environment env {argc, argv};
    const simplemc::mpi::communicator comm;
    try {
        return run(argc, argv, comm);
    }
    catch (const std::exception & e) {
        // input errors are raised identically on every rank; the root's message is enough there, but an error on
        // a single rank (e.g. its start diagram) must be visible too
        std::fprintf(stderr, "diagramc (rank %d): %s\n", comm.rank(), e.what());
        MPI_Abort(comm, 1);
    }
    return 1;
}
