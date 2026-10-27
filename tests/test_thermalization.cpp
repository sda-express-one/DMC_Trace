// R-hat-controlled thermalization: convergence_monitor + thermalization_control.
//
// 1. Several chains in one process (the decision logic, evaluate(), without MPI): isotropic alpha = 0.5, fixed tau_D,
//    half the chains from order 0 and half from random high-order diagrams (diagram_builder), stepped in lockstep
//    with a checkpoint every fixed number of steps. The early evaluations must fail (the high starts are still
//    relaxing: R-hat of the line count well above 1.01), thermalization must then be declared after `consecutive`
//    passing checkpoints, and a short production run afterwards must give the ground-state energy.
// 2. The MPI path inside simplemc::run (one rank in the regression suite; run with mpirun -n 4 by hand): a kernel
//    with all length-preserving updates, the monitor in a measurement_set, checkpoint() from on_checkpoint and
//    thermalized() from stop_when. The run must stop thermalized before the step cap, at a multiple of the
//    checkpoint interval, with every rank at the same step count.
// 3. The wall-time deadline: with a deadline already past, a chain that cannot pass is stopped at the checkpoint
//    after the first evaluation, as timedOut() and not thermalized(); without a deadline it keeps going.
// 4. The ESS criterion on iid windows, default thresholds (total 400, 50 per split chain): the total decides for
//    4 chains, the per-split-chain floor for 48; windows of 64 draws per chain (the old 256-batch monitor) fail in
//    both, 256 draws (the 1024-batch monitor) pass in both. requiredEssFraction / batchesForFraction: the required
//    ESS as a fraction of the smallest window, and the batch count that brings it under 0.5.
// 5. Live output: openLog appends each evaluation to the per-scalar log and to the raw numeric file (flushed, the
//    status line only at closeLog); printRecord shows R-hat bulk/fold, ESS and the failed tests per scalar.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <memory>
#include <random>
#include <vector>
#include <mpi.h>
#include <simplemc/mc/measurement.hpp>
#include <simplemc/mc/measurement_set.hpp>
#include <simplemc/mc/run.hpp>
#include <simplemc/mc/run_callbacks.hpp>
#include <simplemc/mc/simulation_params.hpp>
#include "test_common.hpp"
#include "diagram/diagram_builder.hpp"
#include "updates/add_internal_ph.hpp"
#include "updates/rm_internal_ph.hpp"
#include "updates/add_external_ph.hpp"
#include "updates/rm_external_ph.hpp"
#include "updates/swp_ph.hpp"
#include "updates/mv_vertex.hpp"
#include "updates/chg_ph_momentum.hpp"
#include "measurements/convergence_monitor.hpp"
#include "measurements/gs_energy.hpp"
#include "measurements/thermalization_control.hpp"

// one chain at fixed tau_D (alpha = 0.5: A = B = 1, omega = 1, eps = 1) with every length-preserving update
struct Chain {
    test::Diagram d;
    add_int_ph_update addi;  rm_int_ph_update rmi;
    add_ext_ph_update adde;  rm_ext_ph_update rme;
    mv_tau_update mv;        swp_ph_update swp;
    chg_ph_momentum chw;
    std::uniform_real_distribution<double> u {0., 1.};
    std::uniform_int_distribution<int> pick {0, 6};

    Chain(unsigned long long seed, double L, bool high_start)
        : d(seed, {0., 0., 0.}, 400, 16, L + 1., -1., 1., 1.),
          addi(&d.cfg, &d.rng), rmi(&d.cfg, &d.rng), adde(&d.cfg, &d.rng), rme(&d.cfg, &d.rng),
          mv(&d.cfg, &d.rng), swp(&d.cfg, &d.rng), chw(&d.cfg, &d.rng)
    {
        simplemc::xoshiro256ss init {seed + 7777ULL};
        if (high_start) {
            // well above the target (alpha omega L = 6 lines): 40-60 internal, 0-3 external lines, momenta 1.5 x the physical spread
            diagram_builder::randomDiagram(d.cfg, init, {0., 0., 0.}, L,
                                           {.n_internal_min = 40, .n_internal_max = 60, .n_external_min = 0, .n_external_max = 3, .w_spread = 1.5});
        }
        else {
            diagram_builder::reset(d.cfg, L);
            diagram_builder::assignMomenta(d.cfg, {0., 0., 0.});
            diagram_builder::finalize(d.cfg);
        }
    }

    template <typename RNG>
    void step(RNG &){        // simplemc kernel interface; the chain's own rng drives the updates
        const int w {pick(d.rng)};
        double r {-1.};
        switch (w) { case 0: r = addi.attempt(); break; case 1: r = rmi.attempt(); break; case 2: r = adde.attempt(); break;
                     case 3: r = rme.attempt(); break; case 4: r = mv.attempt(); break; case 5: r = swp.attempt(); break;
                     default: r = chw.attempt(); }
        if (r > 0. && u(d.rng) < r) {
            switch (w) { case 0: addi.accept(); break; case 1: rmi.accept(); break; case 2: adde.accept(); break;
                         case 3: rme.accept(); break; case 4: mv.accept(); break; case 5: swp.accept(); break;
                         default: chw.accept(); }
        }
    }
};

static void in_process(test::Checks & check){
    test::setLK({1., 1., 0.});
    const double L {12.};
    const int n_chains {6};
    const long steps_per_cycle {10}, cycles_per_checkpoint {1000}, max_checkpoints {600};

    std::vector<std::unique_ptr<Chain>> chains;
    std::vector<convergence_monitor> monitors;
    for (int c {0}; c < n_chains; ++c) {
        chains.push_back(std::make_unique<Chain>(100ULL + static_cast<unsigned long long>(c), L, c % 2 == 1));
        monitors.emplace_back(&chains.back()->d.cfg, convergence_monitor::Options{.n_batches = 256});    // 1000 samples per checkpoint: 1024 would leave no window (2048 before the first merge)
    }
    thermalization_control ctl {MPI_COMM_SELF, monitors[0].names(), monitors[0].maxWindowBatches()};
    const int need {thermalization_control::Options {}.consecutive};
    simplemc::xoshiro256ss unused {1ULL};

    int declared_at {0};
    for (int cp {1}; cp <= max_checkpoints && declared_at == 0; ++cp) {
        for (long cy {0}; cy < cycles_per_checkpoint; ++cy) {
            for (int c {0}; c < n_chains; ++c) {
                for (long s {0}; s < steps_per_cycle; ++s) { chains[static_cast<std::size_t>(c)]->step(unused); }
                monitors[static_cast<std::size_t>(c)].measure();
            }
        }
        std::vector<std::vector<double>> windows;
        std::size_t batches {0};
        for (const auto & m : monitors) { windows.push_back(m.window(batches)); }
        const thermalization_control::Record rec {ctl.evaluate(windows, batches, static_cast<std::uint64_t>(cp) * cycles_per_checkpoint * steps_per_cycle)};
        if (rec.consecutive >= need) { declared_at = cp; }
    }

    const auto & h {ctl.history()};
    // the first evaluation must fail (the high starts are still relaxing); report its worst scalar
    std::size_t worst {0};
    for (std::size_t k {1}; k < h.front().results.size(); ++k) { if (h.front().results[k].rhat > h.front().results[worst].rhat) { worst = k; } }
    const bool early_fail {!h.front().pass};
    bool tail_passes {declared_at > 0 && h.size() >= static_cast<std::size_t>(need)};
    for (int i {1}; tail_passes && i <= need; ++i) { tail_passes = h[h.size() - static_cast<std::size_t>(i)].pass; }
    check(early_fail && tail_passes,
          "%d chains (3 from order 0, 3 from 40-60-line starts): first checkpoint fails (worst R-hat: %s %.3f), thermalized at checkpoint %d "
          "(%ld steps per chain) after %d consecutive passes",
          n_chains, ctl.names()[worst].c_str(), h.front().results[worst].rhat, declared_at, declared_at * cycles_per_checkpoint * steps_per_cycle, need);

    // production: the energy from every chain after thermalization
    double E {0.}, e2 {0.};
    for (auto & ch : chains) {
        gs_energy_measurement G {&ch->d.cfg, {0., 0., 0.}, 0., true};
        for (long s {0}; s < 600000; ++s) { ch->step(unused); if (s % 10 == 0) { G.measure(); } }
        const double e {static_cast<double>(G.normalised().energy)};
        E += e; e2 += e * e;
    }
    E /= n_chains;
    const double err {std::sqrt((e2 / n_chains - E * E) / (n_chains - 1))};
    check(std::abs(E + 0.504) < 4. * err + 0.01,
          "after thermalization: E0 = %.4f +- %.4f over the 6 chains (exact -0.504)", E, err);
}

static void mpi_run(test::Checks & check){
    test::setLK({1., 1., 0.});
    int rank {0}, size {1};
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    Chain chain {500ULL + static_cast<unsigned long long>(rank), 12., rank % 2 == 0};
    simplemc::measurement_set therm {simplemc::measurement{convergence_monitor{&chain.d.cfg, {}}, "convergence"}};
    convergence_monitor & mon {therm.get<0>().value()};
    // the suite runs on one rank (2 split chains, 512 draws right after a merge): a total of 400 would ask for
    // ESS/draws ~ 0.8, so the totals are scaled to the chains present, 50 per split chain (the floor) as for many ranks
    thermalization_control ctl {MPI_COMM_WORLD, mon.names(), mon.maxWindowBatches(),
                                {.min_bulk_ess = 100. * size, .min_tail_ess = 100. * size}};

    const std::uint64_t interval {100000}, cap {20000000};
    simplemc::simulation_params tp;
    tp.max_steps = cap;
    tp.max_time = 1e9;                       // capped by steps only: every rank stops at the same checkpoint
    tp.steps_per_cycle = 10;
    tp.cycles_per_check = 1000;
    tp.checkpoint_after_steps = interval;
    simplemc::xoshiro256ss rng {9ULL + static_cast<unsigned long long>(rank)};
    const simplemc::simulation_ctx ctx {simplemc::run(rng, chain, therm, tp, simplemc::run_callbacks{
        .on_checkpoint = [&](const simplemc::simulation_ctx & c){ ctl.checkpoint(mon, c.steps_done); },
        .stop_when = [&](const simplemc::simulation_ctx &){ return ctl.thermalized(); }})};

    // every rank must have stopped at the same step count
    unsigned long long mine {ctx.steps_done}, lo {0}, hi {0};
    MPI_Allreduce(&mine, &lo, 1, MPI_UNSIGNED_LONG_LONG, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(&mine, &hi, 1, MPI_UNSIGNED_LONG_LONG, MPI_MAX, MPI_COMM_WORLD);
    const bool ok {ctl.thermalized() && ctx.steps_done < cap && ctx.steps_done % interval == 0 && lo == hi};
    if (rank == 0) {
        check(ok, "simplemc::run on %d rank(s): thermalized at %llu steps (cap %llu), a multiple of the checkpoint interval, "
                  "all ranks at the same step count; %zu evaluations, the last %d passing",
              size, static_cast<unsigned long long>(ctx.steps_done), static_cast<unsigned long long>(cap), ctl.history().size(),
              ctl.history().empty() ? 0 : ctl.history().back().consecutive);
    }
}

static void options_and_guard(test::Checks & check){
    test::setLK({1., 1., 0.});
    Chain chain {3ULL, 12., true};
    convergence_monitor all {&chain.d.cfg, {.tau_D = true, .energy = true, .orbital = true}};
    const std::vector<double> s {all.scalars()};
    bool finite {s.size() == 11 && all.names().size() == 11};
    for (const double x : s) { finite = finite && std::isfinite(x); }
    // the sign (index 6, after tau_D) is +1 at C = 0; orbital fractions of A = B: 1/3 each
    finite = finite && all.names()[6] == "sign" && s[6] == 1.
          && std::abs(s[8] - 1./3.) < 1e-12 && std::abs(s[9] - 1./3.) < 1e-12 && std::abs(s[10] - 1./3.) < 1e-12;

    bool threw {false};
    try { thermalization_control bad {MPI_COMM_SELF, all.names(), 32}; } catch (const std::invalid_argument &) { threw = true; }
    check(finite && threw, "all monitor options: 11 finite scalars (sign +1, orbital fractions 1/3 for A = B); n_batches = 32 refused "
                           "(the window could never reach 50 ESS per split chain)");
}

static void deadline(test::Checks & check){
    // one scalar, one chain, 16 batch means of iid noise per checkpoint: 16 draws in total, below the 400 total ESS,
    // so the test can never pass and only the deadline can end the phase
    std::mt19937_64 rng {77ULL};
    std::normal_distribution<double> g {0., 1.};
    const auto window {[&]{ std::vector<double> w(16); for (double & x : w) { x = g(rng); } return w; }};

    thermalization_control timed {MPI_COMM_SELF, {"x"}, 256};
    timed.setDeadline(MPI_Wtime() - 1.);
    thermalization_control open {MPI_COMM_SELF, {"x"}, 256};
    int timed_at {0};
    for (int cp {1}; cp <= 5; ++cp) {
        timed.checkpoint(window(), 16, static_cast<std::uint64_t>(cp));
        open.checkpoint(window(), 16, static_cast<std::uint64_t>(cp));
        if (timed.stopped() && timed_at == 0) { timed_at = cp; }
    }
    // the decision taken at checkpoint 1 is read at checkpoint 2 (one interval late, as for thermalization)
    check(timed_at == 2 && timed.timedOut() && !timed.thermalized() && !open.stopped(),
          "past deadline: stopped at checkpoint %d (expected 2), timedOut %d, thermalized %d; without deadline still running: %d",
          timed_at, timed.timedOut() ? 1 : 0, timed.thermalized() ? 1 : 0, open.stopped() ? 0 : 1);
}

static void ess_criterion(test::Checks & check){
    // one scalar, iid windows of `batches` draws per chain: R-hat ~ 1, total ESS ~ chains x batches
    std::mt19937_64 rng {91ULL};
    std::normal_distribution<double> g {0., 1.};
    const auto windows {[&](int chains, std::size_t batches){
        std::vector<std::vector<double>> w(static_cast<std::size_t>(chains), std::vector<double>(batches));
        for (auto & c : w) { for (double & x : c) { x = g(rng); } }
        return w;
    }};
    thermalization_control::ScalarCheck last {};
    const auto passes {[&](thermalization_control::Options opt, int chains, std::size_t batches, double & ess){
        thermalization_control ctl {MPI_COMM_SELF, {"x"}, 1024, opt};
        const thermalization_control::Record rec {ctl.evaluate(windows(chains, batches), batches, 1)};
        ess = std::min(rec.results[0].ess_bulk, rec.results[0].ess_tail);
        last = rec.checks[0];
        return rec.pass;
    }};

    // 4 chains (8 split): total 400 = floor 8 x 50; 48 chains (96 split): floor 4800 > total 400
    double e4s {0.}, e4l {0.}, e48s {0.}, e48l {0.};
    const bool p4s {passes({}, 4, 64, e4s)};       // ESS ~ 256 < 400
    const bool p4l {passes({}, 4, 256, e4l)};      // ESS ~ 1000
    const thermalization_control::ScalarCheck c4l {last};
    const bool p48s {passes({}, 48, 64, e48s)};    // ESS ~ 3000: above the total, below the floor
    const thermalization_control::ScalarCheck c48s {last};
    const bool p48l {passes({}, 48, 256, e48l)};   // ESS ~ 12000
    check(!p4s && p4l && !p48s && p48l,
          "ESS criterion (total 400, 50 per split chain): 4 chains x 64 draws, ESS %.0f < 400 fails (%d), x 256, ESS %.0f passes (%d); "
          "48 chains x 64 draws, ESS %.0f < floor 4800 fails (%d), x 256, ESS %.0f passes (%d)",
          e4s, p4s ? 1 : 0, e4l, p4l ? 1 : 0, e48s, p48s ? 1 : 0, e48l, p48l ? 1 : 0);
    check(c4l.ok() && c4l.bulk_required == 400. && c4l.tail_required == 400.
              && c48s.rhat_ok && !c48s.bulk_ok && !c48s.tail_ok && c48s.bulk_required == 4800. && c48s.tail_required == 4800.,
          "per-scalar flags: 4 chains x 256 all ok, required %.0f/%.0f (expected 400); 48 chains x 64 R-hat ok %d, bulk/tail ok %d/%d "
          "(expected 1, 0/0), required %.0f/%.0f (expected 4800)",
          c4l.bulk_required, c4l.tail_required, c48s.rhat_ok ? 1 : 0, c48s.bulk_ok ? 1 : 0, c48s.tail_ok ? 1 : 0,
          c48s.bulk_required, c48s.tail_required);

    // the required ESS as a fraction of the window right after a merge (2R split chains of n/4 draws)
    {
        using tc = thermalization_control;
        const tc::Options d {};
        const double r1 {tc::requiredEssFraction(d, 1, 1024)}, r2 {tc::requiredEssFraction(d, 2, 1024)};
        const double r48 {tc::requiredEssFraction(d, 48, 1024)}, r48_256 {tc::requiredEssFraction(d, 48, 256)};
        const std::size_t n1 {tc::batchesForFraction(d, 1, 0.5)}, n2 {tc::batchesForFraction(d, 2, 0.5)};
        const std::size_t n48 {tc::batchesForFraction(d, 48, 0.5)};
        const auto near {[](double a, double b){ return std::abs(a - b) < 1e-12; }};
        check(near(r1, 400. / 512.) && near(r2, 400. / 1024.) && near(r48, 50. / 256.) && near(r48_256, 50. / 64.)
                  && n1 == 2048 && n2 == 1024 && n48 == 512,
              "required ESS fraction: 1 rank %.3f (400/512), 2 ranks %.3f, 48 ranks %.3f (50/256), 48 ranks with 256 batches %.3f "
              "(the coupling-5 run); batches for 0.5: %zu / %zu / %zu (expected 2048 / 1024 / 512)",
              r1, r2, r48, r48_256, n1, n2, n48);
    }

    // the log: two scalars (one that passes, one that fails R-hat: shifted chains), the required-ESS header, 14 columns
    // per row, and the flags consistent with the checkpoint's pass
    {
        thermalization_control ctl {MPI_COMM_SELF, {"good", "shifted"}, 1024};
        std::size_t rows {0}, bad_rows {0};
        bool header {false}, flags_seen {false};
        for (int cp {0}; cp < 2; ++cp) {
            std::vector<std::vector<double>> w(8, std::vector<double>(2 * 256));
            for (std::size_t c {0}; c < w.size(); ++c) {
                for (std::size_t b {0}; b < 256; ++b) { w[c][b] = g(rng); w[c][256 + b] = g(rng) + 0.5 * static_cast<double>(c); }
            }
            ctl.evaluate(w, 256, static_cast<std::uint64_t>(cp + 1));
        }
        const std::filesystem::path f {std::filesystem::temp_directory_path() / "dmc_test_thermalization_log.dat"};
        ctl.write(f.string());
        std::ifstream in {f};
        for (std::string line; std::getline(in, line);) {
            if (line.rfind("# required ESS with 16 split chains: bulk >= 800.0, tail >= 800.0", 0) == 0) { header = true; }
            if (line.empty() || line[0] == '#') { continue; }
            std::istringstream ss {line};
            std::vector<std::string> col;
            for (std::string x; ss >> x;) { col.push_back(x); }
            ++rows;
            if (col.size() != 14) { ++bad_rows; continue; }
            const bool all_ok {col[9] == "1" && col[10] == "1" && col[11] == "1"};
            if (col[12] == "1" && !all_ok) { ++bad_rows; }
            // the good scalar passes everything; the shifted one fails R-hat (its ESS is fine)
            if (col[3] == "good" && !all_ok) { ++bad_rows; }
            if (col[3] == "shifted") { flags_seen = col[9] == "0" && col[12] == "0"; }
        }
        std::filesystem::remove(f);
        check(header && rows == 4 && bad_rows == 0 && flags_seen,
              "thermalization log: required-ESS header %d, %zu rows (expected 4), %zu inconsistent or not 14 columns, "
              "shifted scalar logged with rhat_ok 0 and pass 0: %d", header ? 1 : 0, rows, bad_rows, flags_seen ? 1 : 0);
    }
}

static std::vector<std::string> readLines(const std::filesystem::path & f){
    std::vector<std::string> lines;
    std::ifstream in {f};
    for (std::string line; std::getline(in, line);) { lines.push_back(line); }
    return lines;
}

static std::vector<std::string> fields(const std::string & line){
    std::istringstream ss {line};
    std::vector<std::string> col;
    for (std::string x; ss >> x;) { col.push_back(x); }
    return col;
}

static void live_output(test::Checks & check){
    std::mt19937_64 rng {123ULL};
    std::normal_distribution<double> g {0., 1.};
    // 8 chains, scalars "good" (iid) and "shifted" (chain c offset by 0.5 c: R-hat fails); a first window of 4 batches
    const auto window {[&](std::size_t batches){
        std::vector<std::vector<double>> w(8, std::vector<double>(2 * batches));
        for (std::size_t c {0}; c < w.size(); ++c) {
            for (std::size_t b {0}; b < batches; ++b) { w[c][b] = g(rng); w[c][batches + b] = g(rng) + 0.5 * static_cast<double>(c); }
        }
        return w;
    }};
    const std::filesystem::path dir {std::filesystem::temp_directory_path()};
    const std::filesystem::path log {dir / "dmc_test_live_thermalization.dat"}, raw {dir / "dmc_test_live_rhat.dat"};

    thermalization_control ctl {MPI_COMM_SELF, {"good", "shifted"}, 1024};
    ctl.openLog(log.string(), raw.string());
    ctl.evaluate(window(4), 4, 10);
    ctl.evaluate(window(256), 256, 20);
    ctl.evaluate(window(256), 256, 30);

    // before closeLog: the rows are on disk (flushed), the status line is not
    const std::vector<std::string> mid {readLines(log)};
    std::size_t rows {0};
    bool required {false}, status_early {false};
    for (const auto & l : mid) {
        if (l.rfind("# required ESS with 16 split chains", 0) == 0) { required = true; }
        if (l.rfind("# thermalized:", 0) == 0) { status_early = true; }
        if (!l.empty() && l[0] != '#') { ++rows; }
    }
    // raw: names on the first line, 3 + 4 x 2 + 2 = 13 columns, one numeric row per checkpoint, nan without a window
    const std::vector<std::string> rl {readLines(raw)};
    const std::vector<std::string> names {rl.empty() ? std::vector<std::string> {} : fields(rl.front())};
    std::vector<std::vector<std::string>> data;
    for (const auto & l : rl) { if (!l.empty() && l[0] != '#') { data.push_back(fields(l)); } }
    bool widths {data.size() == 3};
    for (const auto & d : data) { widths = widths && d.size() == 13; }
    const auto & h {ctl.history()};
    bool values {widths && data[0][3] == "nan"};
    for (std::size_t i {1}; values && i < 3; ++i) {
        values = std::abs(std::stod(data[i][3]) - h[i].results[0].rhat_bulk) < 1e-6
              && std::abs(std::stod(data[i][8]) - h[i].results[1].rhat_fold) < 1e-6
              && std::abs(std::stod(data[i][5]) - h[i].results[0].ess_bulk) < 0.1
              && data[i][11] == (h[i].pass ? "1" : "0") && data[i][12] == std::to_string(h[i].consecutive);
    }
    const bool raw_names {names.size() == 14 && names[0] == "#" && names[1] == "checkpoint" && names[4] == "good_rbulk"
                          && names[9] == "shifted_rfold" && names[13] == "consecutive"};

    ctl.closeLog();
    const std::vector<std::string> fin {readLines(log)};
    const bool status_last {!fin.empty() && fin.back().rfind("# thermalized: no", 0) == 0};
    std::filesystem::remove(log);
    std::filesystem::remove(raw);
    check(rows == 6 && required && !status_early && status_last,
          "live log: %zu rows before closeLog (expected 6), required-ESS line %d, status line only after closeLog and last: %d",
          rows, required ? 1 : 0, (!status_early && status_last) ? 1 : 0);
    check(raw_names && widths && values,
          "raw file: column names on the first line %d, 3 rows of 13 numeric columns %d, values match the records (nan before a window) %d",
          raw_names ? 1 : 0, widths ? 1 : 0, values ? 1 : 0);

    // printRecord
    std::FILE * f {std::tmpfile()};
    ctl.printRecord(f, h[2]);
    ctl.printRecord(f, h[0]);
    std::rewind(f);
    std::string text;
    for (int ch; (ch = std::fgetc(f)) != EOF;) { text.push_back(static_cast<char>(ch)); }
    std::fclose(f);
    std::istringstream ts {text};
    std::vector<std::string> pl;
    for (std::string l; std::getline(ts, l);) { pl.push_back(l); }
    const bool block {pl.size() == 4 && pl[0].find("[thermalization] checkpoint 3") == 0 && pl[0].find("consecutive 0/5") != std::string::npos
                      && pl[0].find("required ESS 800/800") != std::string::npos
                      && pl[1].find("good") != std::string::npos && pl[1].find("R_bulk") != std::string::npos
                      && pl[1].find("R_fold") != std::string::npos && pl[1].substr(pl[1].size() - 2) == "ok"
                      && pl[2].find("shifted") != std::string::npos && pl[2].find("FAIL rhat") != std::string::npos
                      && pl[3].find("no window yet (batches 4 < 8)") != std::string::npos};
    check(block, "printRecord: header with consecutive 0/5 and required ESS, 'ok' for the good scalar, 'FAIL rhat' for the shifted one, "
                 "'no window yet' before 8 batches%s%s", block ? "" : "\n", block ? "" : text.c_str());
}

int main(int argc, char ** argv){
    MPI_Init(&argc, &argv);
    int rank {0}, size {1};
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    int code {0};
    if (rank == 0) {
        test::Checks check {"R-hat thermalization (convergence_monitor, thermalization_control)"};
        if (size == 1) { options_and_guard(check); in_process(check); deadline(check); ess_criterion(check); live_output(check); }     // once, not on every rank
        mpi_run(check);
        code = check.exit_code();
    }
    else {
        test::Checks quiet {"(rank > 0)"};
        mpi_run(quiet);
    }
    MPI_Finalize();
    return code;
}
