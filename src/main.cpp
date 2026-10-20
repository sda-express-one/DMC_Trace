// diagramc <input.json>
//
// One Markov chain per MPI rank: the input is read (io/run_input.hpp), each rank builds its own diagram, the
// chains are thermalized under R-hat control (the ranks are the chains: measurements/thermalization_control.hpp),
// then the production run measures the active estimators, the ranks are combined and the root writes
//     <prefix>_<estimator>.dat   one per active estimator (its write())
//     <prefix>_<estimator>_chains.dat  gs_energy, eff_mass, eff_mass_uniaxial, avg_sign: every chain's own value and error
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
#include "measurements/avg_sign.hpp"
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

    // Per-chain results (collective, before the ranks are merged): every rank's own summary() - value and jackknife
    // error of each quantity in M::summaryNames() - and its sample counters are gathered on the root, which writes
    // <prefix>_<name>_chains.dat: one line per rank, preceded by the mean over the chains with its standard error
    // and the chi^2/dof of the chains around their error-weighted mean (~1 if they agree, >> 1 for a stuck chain).
    template <typename M>
    void writeChains(const simplemc::mpi::communicator & comm, const M & m, const std::string & name, const run_input & in){
        std::vector<double> mine {m.summary()};
        mine.push_back(static_cast<double>(m.n_measured));
        mine.push_back(static_cast<double>(m.n_negative));
        const int n {static_cast<int>(mine.size())};
        const bool root {comm.rank() == 0};
        std::vector<double> all(root ? static_cast<std::size_t>(n * comm.size()) : 0);
        MPI_Gather(mine.data(), n, MPI_DOUBLE, root ? all.data() : nullptr, n, MPI_DOUBLE, 0, comm);
        if (!root) { return; }

        const std::vector<std::string> names {M::summaryNames()};
        const auto R {static_cast<std::size_t>(comm.size())};
        const auto row {[&](std::size_t r){ return all.data() + r * static_cast<std::size_t>(n); }};
        const std::string path {outPath(in, name + "_chains.dat")};
        std::FILE * f {std::fopen(path.c_str(), "w")};
        if (f == nullptr) { throw std::runtime_error("cannot open " + path); }
        std::fprintf(f, "# %s, per chain (one chain per MPI rank; each with its own delete-one-batch jackknife)\n", name.c_str());
        std::fprintf(f, "# p = %.10f %.10f %.10f   tau_fixed = %d   chains = %zu\n", in.diagram.p[0], in.diagram.p[1],
                     in.diagram.p[2], in.diagram.tau_fixed ? 1 : 0, R);
        for (std::size_t q {0}; q < names.size(); ++q) {
            // chains with a finite value and a positive error only
            double s {0.}, s2 {0.}, sw {0.}, swv {0.};
            std::size_t k {0};
            for (std::size_t r {0}; r < R; ++r) {
                const double v {row(r)[2*q]}, e {row(r)[2*q + 1]};
                if (!std::isfinite(v) || !std::isfinite(e) || e <= 0.) { continue; }
                s += v; s2 += v * v; sw += 1. / (e * e); swv += v / (e * e); ++k;
            }
            if (k < 2) {
                std::fprintf(f, "# %s: fewer than 2 chains with a result\n", names[q].c_str());
                continue;
            }
            const double kk {static_cast<double>(k)};
            const double mean {s / kk};
            const double sem {std::sqrt(std::max(s2 / kk - mean * mean, 0.) * kk / (kk - 1.) / kk)};
            const double wmean {swv / sw};
            double chi2 {0.};
            for (std::size_t r {0}; r < R; ++r) {
                const double v {row(r)[2*q]}, e {row(r)[2*q + 1]};
                if (std::isfinite(v) && std::isfinite(e) && e > 0.) { chi2 += (v - wmean) * (v - wmean) / (e * e); }
            }
            std::fprintf(f, "# %s: mean over %zu chains = %.10e +- %.3e (spread / sqrt(chains))   weighted mean = %.10e   "
                            "chi2/dof = %.3f\n", names[q].c_str(), k, mean, sem, wmean, chi2 / (kk - 1.));
        }
        std::fprintf(f, "# columns: rank");
        for (const auto & nm : names) { std::fprintf(f, " %s err", nm.c_str()); }
        std::fprintf(f, " n_measured n_negative\n");
        for (std::size_t r {0}; r < R; ++r) {
            std::fprintf(f, "%zu", r);
            for (std::size_t q {0}; q < names.size(); ++q) { std::fprintf(f, " %.10e %.3e", row(r)[2*q], row(r)[2*q + 1]); }
            std::fprintf(f, " %.0f %.0f\n", row(r)[2*names.size()], row(r)[2*names.size() + 1]);
        }
        std::fclose(f);
    }

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
                                                            in.measurements.split_sign, nb}, "green_trace"},
            simplemc::measurement {avg_sign_measurement {&cfg, d.tau_min, d.tau_fixed, nb}, "avg_sign"}};
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
        // per-chain files first: the collect below merges the ranks' batches in place
        if (produce) {
            if (in.measurements_active.at("gs_energy")) {
                writeChains(comm, *ms.get<gs_energy_measurement>("gs_energy"), "gs_energy", in);
            }
            if (in.measurements_active.at("eff_mass")) {
                writeChains(comm, *ms.get<eff_mass_measurement>("eff_mass"), "eff_mass", in);
            }
            if (in.measurements_active.at("eff_mass_uniaxial")) {
                writeChains(comm, *ms.get<eff_mass_uniaxial_measurement>("eff_mass_uniaxial"), "eff_mass_uniaxial", in);
            }
            if (in.measurements_active.at("avg_sign")) {
                writeChains(comm, *ms.get<avg_sign_measurement>("avg_sign"), "avg_sign", in);
            }
        }
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
            if (in.measurements_active.at("avg_sign")) {
                if (const auto * e {ms.get<avg_sign_measurement>("avg_sign")}; e != nullptr && e->acc.batches().size() >= 2) {
                    const auto r {e->normalised()};
                    std::printf("<s> = %.8Lg +- %.3Lg\n", r.sign, r.sign_error);
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
