// Checkpoint/restart (diagram/diagram_io.hpp, io/checkpoint.hpp, the estimators' and updates' simplemc hooks):
//   1. diagram round trip: AlAs (C != 0) chains at a generic p (no zero component: on a symmetry plane, e.g.
//      p = (0.1, 0, -0.05), the running chain itself can end up with a head/tail basis flip from k_y = +-1e-17
//      rounding, independently of any checkpoint) with internal and external lines and two phonon modes, evolved by every
//      update (so the slot order and the momenta carry the chain's history), captured, written to a MessagePack file,
//      read back and restored into a fresh diagram_cfg: sanitizer clean, the same lines in the same manager slots,
//      every segment momentum bitwise equal, the same trace;
//   2. estimator round trip: every production estimator, filled by a chain, saved and loaded into a freshly built one:
//      identical normalised() results and counters;
//   3. continuation: a chain run for N steps, checkpointed (checkpoint::save), restored into new objects
//      (readHeader + restoreDiagram + checkpoint::load) and run for M more, against an uninterrupted N + M run:
//      counters add up, the restored chain stays clean, and the energies agree within their errors.
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>
#include <simplemc/mc/json.hpp>
#include <simplemc/mc/kernels.hpp>
#include <simplemc/mc/measurement_set.hpp>
#include <simplemc/mc/run.hpp>
#include "test_common.hpp"
#include "diagram/diagram_builder.hpp"
#include "diagram/diagram_io.hpp"
#include "io/checkpoint.hpp"
#include "measurements/avg_sign.hpp"
#include "measurements/eff_mass.hpp"
#include "measurements/eff_mass_uniaxial.hpp"
#include "measurements/green_func.hpp"
#include "measurements/green_trace.hpp"
#include "measurements/gs_energy.hpp"
#include "measurements/qp_weights.hpp"
#include "updates/dmc_update_set.hpp"

namespace fs = std::filesystem;

static const fs::path tmp_dir {fs::temp_directory_path()};

// fixed-length weights: every update except the length-changing ones
static std::map<std::string, double> fixed_weights(){
    std::map<std::string, double> w;
    for (const auto n : update_names) { w[std::string {n}] = 1.; }
    for (const auto n : length_changing_updates) { w[std::string {n}] = 0.; }
    return w;
}

static void diagram_round_trip(test::Checks & check){
    test::setLK(test::AlAs);
    const std::vector<PhononMode> modes {PhononMode{0.5, 2.0}, PhononMode{0.8, 3.0}};
    int ok_slots {0}, ok_k {0}, ok_clean {0}, total {0};
    double worst_trace {0.};
    std::size_t lines_int {0}, lines_ext {0};
    for (unsigned long long seed {1}; seed <= 6; ++seed) {
        test::Diagram d {seed, {0.1, 0.07, -0.05}, 120, 16, modes, 20.};
        simplemc::xoshiro256ss b {seed + 50};
        diagram_builder::randomDiagram(d.cfg, b, {0.1, 0.07, -0.05}, 20.,
                                       {.n_internal_min = 10, .n_internal_max = 20, .n_external_min = 1, .n_external_max = 4, .w_spread = 1.});
        dmc_update_set us {makeUpdateSet(&d.cfg, &d.rng, fixed_weights())};
        simplemc::metropolis_kernel k {us};
        for (int i {0}; i < 50000; ++i) { k.step(d.rng); }

        const DiagramState saved {captureDiagram(d.cfg)};
        const fs::path file {tmp_dir / ("dmc_test_ckpt_diagram_" + std::to_string(seed) + ".msgpack")};
        simplemc::save_json_file(file, [&](simplemc::json_serializer & s){ s.save_at("d", saved); }, checkpoint::io);
        DiagramState loaded;
        simplemc::load_json_file(file, [&](const simplemc::json_serializer & s){ s.load_at("d", loaded); }, checkpoint::io);
        fs::remove(file);

        test::Diagram r {seed + 100, {0.1, 0.07, -0.05}, 120, 16, modes, 20.};
        const numerical::SanitizeReport rep {restoreDiagram(r.cfg, loaded)};
        ++total;
        if (rep.clean()) { ++ok_clean; }
        else {
            const numerical::SanitizeReport orig {numerical::sanitizeDiagram(&d.cfg)};
            std::printf("    seed %llu not clean: links %d lines %d momentum %.2e basis %.2e mass %.2e action %.2e shift %.2e ph %.2e "
                        "overlap %.2e flips %d closing %.2e | original diagram: momentum %.2e clean %d\n", seed, rep.broken_links, rep.bad_lines,
                        rep.max_momentum_residual, rep.max_basis_dev, rep.max_mass_dev, rep.max_action_dev, rep.max_shift_dev,
                        rep.max_ph_action_dev, rep.max_overlap_dev, rep.overlap_flips, rep.closing_basis_dev, orig.max_momentum_residual,
                        orig.clean() ? 1 : 0);
        }
        // the same lines in the same slots of both managers
        bool slots {r.cfg.internal_ph_manager->current_length == d.cfg.internal_ph_manager->current_length
                    && r.cfg.external_ph_manager->current_length == d.cfg.external_ph_manager->current_length};
        for (const auto & [a, b2] : {std::pair {d.cfg.internal_ph_manager, r.cfg.internal_ph_manager},
                                     std::pair {d.cfg.external_ph_manager, r.cfg.external_ph_manager}}) {
            for (int i {0}; slots && i < a->current_length; ++i) {
                const Vertex * x {a->selectVertex(i)}, * y {b2->selectVertex(i)};
                slots = x->tau == y->tau && x->type == y->type && x->w == y->w && x->ph_energy == y->ph_energy
                     && x->diel_response == y->diel_response && x->conj_vertex->tau == y->conj_vertex->tau;
            }
        }
        if (slots) { ++ok_slots; }
        bool same_k {true};
        const Vertex * x {d.cfg.diagram_head}, * y {r.cfg.diagram_head};
        for (; x != d.cfg.diagram_tail && y != r.cfg.diagram_tail; x = x->next, y = y->next) { same_k = same_k && x->k == y->k; }
        same_k = same_k && x == d.cfg.diagram_tail && y == r.cfg.diagram_tail;
        if (same_k) { ++ok_k; }
        const double t0 {d.cfg.diagram_head->right_component.trace()}, t1 {r.cfg.diagram_head->right_component.trace()};
        worst_trace = std::max(worst_trace, std::abs(t1 - t0) / std::abs(t0));
        lines_int += saved.internal.size();
        lines_ext += saved.external.size();
    }
    check(ok_clean == total && ok_slots == total && ok_k == total && worst_trace < 1e-10,
          "diagram round trip (AlAs C != 0, 2 modes, %zu internal / %zu external lines after 5e4 steps of every update): "
          "%d/%d clean, %d/%d same slot maps, %d/%d momenta bitwise equal, worst relative trace change %.1e",
          lines_int, lines_ext, ok_clean, total, ok_slots, total, ok_k, total, worst_trace);
}

// one estimator, filled, saved and loaded into a fresh one built by make(): identical normalised() and counters
template <typename M, typename Make, typename Same>
static bool round_trip(const std::string & name, M & m, Make make, Same same){
    const fs::path file {tmp_dir / ("dmc_test_ckpt_" + name + ".msgpack")};
    simplemc::save_json_file(file, [&](simplemc::json_serializer & s){ s.save_at("m", m); }, checkpoint::io);
    M fresh {make()};
    simplemc::load_json_file(file, [&](const simplemc::json_serializer & s){ s.load_at("m", fresh); }, checkpoint::io);
    fs::remove(file);
    const bool ok {fresh.acc.count() == m.acc.count() && fresh.n_negative == m.n_negative && same(m, fresh)};
    if (!ok) { std::printf("    %s: round trip differs\n", name.c_str()); }
    return ok;
}

static void estimator_round_trip(test::Checks & check){
    test::setLK({1., 1., 0.});         // every estimator applicable (A = B, C = 0)
    test::Diagram d {7ULL, {0., 0., 0.}, 120, 16, 10., -1., 1., 1.};
    diagram_builder::reset(d.cfg, 8.);
    diagram_builder::assignMomenta(d.cfg, {0., 0., 0.});
    diagram_builder::finalize(d.cfg);
    dmc_update_set us {makeUpdateSet(&d.cfg, &d.rng, fixed_weights())};
    simplemc::metropolis_kernel k {us};
    const std::array<double, 3> p {0., 0., 0.};
    diagram_cfg * c {&d.cfg};
    const auto mk_gs {[&]{ return gs_energy_measurement {c, p, 0., true, 16}; }};
    const auto mk_qp {[&]{ return qp_weights_measurement {c, p, 0., true, 16}; }};
    const auto mk_em {[&]{ return eff_mass_measurement {c, p, 0., true, 16}; }};
    const auto mk_eu {[&]{ return eff_mass_uniaxial_measurement {c, p, 0., true, 16}; }};
    const auto mk_gf {[&]{ return green_func_measurement {c, p, 20, -1, 16}; }};
    const auto mk_gt {[&]{ return green_trace_measurement {c, p, 20, -1, true, 16}; }};
    const auto mk_as {[&]{ return avg_sign_measurement {c, 0., true, 16}; }};
    auto gs {mk_gs()}; auto qp {mk_qp()}; auto em {mk_em()}; auto eu {mk_eu()}; auto gf {mk_gf()}; auto gt {mk_gt()}; auto as {mk_as()};
    for (int i {0}; i < 100003; ++i) {        // not a multiple of the batch size: partial batches are saved too
        k.step(d.rng);
        if (i % 5 == 0) { gs.measure(); qp.measure(); em.measure(); eu.measure(); gf.measure(); gt.measure(); as.measure(); }
    }
    int ok {0};
    ok += round_trip("gs_energy", gs, mk_gs, [](const auto & a, const auto & b){
        return a.n_measured == b.n_measured && a.normalised().energy == b.normalised().energy && a.normalised().energy_error == b.normalised().energy_error; });
    ok += round_trip("qp_weights", qp, mk_qp, [](const auto & a, const auto & b){
        return a.hits == b.hits && a.n_measured == b.n_measured && a.normalised().Z == b.normalised().Z && a.normalised().Z_error == b.normalised().Z_error; });
    ok += round_trip("eff_mass", em, mk_em, [](const auto & a, const auto & b){
        return a.n_measured == b.n_measured && a.normalised().mass_ratio == b.normalised().mass_ratio
            && a.normalised().inv_mass == b.normalised().inv_mass; });
    ok += round_trip("eff_mass_uniaxial", eu, mk_eu, [](const auto & a, const auto & b){
        return a.n_measured == b.n_measured && a.normalised().mass_ratio_par == b.normalised().mass_ratio_par
            && a.normalised().mass_ratio_perp_error == b.normalised().mass_ratio_perp_error; });
    ok += round_trip("green_func", gf, mk_gf, [](const auto & a, const auto & b){
        return a.hits == b.hits && a.n_selected == b.n_selected && a.n_order0 == b.n_order0 && a.normalised().trace == b.normalised().trace; });
    ok += round_trip("green_trace", gt, mk_gt, [](const auto & a, const auto & b){
        return a.hits == b.hits && a.n_selected == b.n_selected && a.n_order0 == b.n_order0 && a.normalised().trace == b.normalised().trace
            && a.normalised().negative == b.normalised().negative; });
    ok += round_trip("avg_sign", as, mk_as, [](const auto & a, const auto & b){
        return a.n_measured == b.n_measured && a.normalised().sign == b.normalised().sign; });
    check(ok == 7, "estimator round trip through MessagePack: %d/7 estimators with identical results and counters "
                   "(gs_energy, qp_weights, eff_mass, eff_mass_uniaxial, green_func, green_trace, avg_sign)", ok);
}

// one fixed-length chain (alpha = 0.5 at A = B = 1) with gs_energy, run by simplemc::run
struct Run {
    test::Diagram d;
    dmc_update_set us;
    simplemc::measurement_set<gs_energy_measurement> ms;
    simplemc::simulation_stats stats;

    explicit Run(unsigned long long seed)
        : d(seed, {0., 0., 0.}, 200, 16, 12., -1., 1., 1.),
          us(makeUpdateSet(&d.cfg, &d.rng, fixed_weights())),
          ms(simplemc::measurement {gs_energy_measurement {&d.cfg, {0., 0., 0.}, 0., true, 64}, "gs_energy"}) {}

    void steps(std::uint64_t n){
        simplemc::metropolis_kernel k {us};
        const simplemc::simulation_params p {.max_steps = n, .max_time = 1e9, .steps_per_cycle = 10, .cycles_per_check = 1000};
        stats += simplemc::run(d.rng, k, ms, p, simplemc::run_callbacks {.on_step = [&](const simplemc::simulation_ctx &){ d.cfg.countSign(); }});
    }
    gs_energy_measurement & E(){ return ms.get<0>().value(); }
};

static void continuation(test::Checks & check){
    test::setLK({1., 1., 0.});
    const std::uint64_t N {2'000'000}, M {2'000'000};
    const auto start {[](Run & r){
        diagram_builder::reset(r.d.cfg, 12.);
        diagram_builder::assignMomenta(r.d.cfg, {0., 0., 0.});
        diagram_builder::finalize(r.d.cfg);
    }};

    // uninterrupted N + M
    Run full {21ULL};
    start(full);
    full.steps(N + M);

    // N, checkpoint, restore into new objects, M more
    Run first {21ULL};
    start(first);
    first.steps(N);
    const fs::path file {tmp_dir / "dmc_test_ckpt_chain.msgpack"};
    checkpoint::Meta meta;
    meta.n_ranks = 1;
    meta.phase = "production";
    meta.n_negative = first.d.cfg.n_negative;
    checkpoint::save(file, first.d.rng, first.us, first.ms, first.stats, first.d.cfg, meta);

    Run second {999ULL};             // different seed: everything must come from the file
    checkpoint::Meta meta2;
    DiagramState saved;
    checkpoint::readHeader(file, meta2, saved);
    const bool restored_clean {restoreDiagram(second.d.cfg, saved).clean()};
    checkpoint::load(file, second.d.rng, second.us, second.ms, second.stats);
    fs::remove(file);
    const bool rng_same {second.d.rng == first.d.rng};
    second.steps(M);

    std::uint64_t props_full {0}, props_second {0};
    for (const auto & s : full.us.stats()) { props_full += s.nprops; }
    for (const auto & s : second.us.stats()) { props_second += s.nprops; }
    const auto ef {full.E().normalised()}, es {second.E().normalised()};
    const double z {static_cast<double>((ef.energy - es.energy) / std::sqrt(ef.energy_error * ef.energy_error + es.energy_error * es.energy_error))};
    const bool counts {second.E().n_measured == full.E().n_measured && second.stats.cumulative_steps == N + M && props_second == props_full};
    const bool clean_end {numerical::sanitizeDiagram(&second.d.cfg).clean()};
    check(restored_clean && rng_same && counts && clean_end && meta2.phase == "production" && std::abs(z) < 4.,
          "continuation: %llu + %llu steps through a checkpoint vs %llu uninterrupted: rng restored %d, %llu samples and %llu proposals "
          "as in the uninterrupted run, clean %d; E0 = %.5Lf +- %.5Lf vs %.5Lf +- %.5Lf (%.1f sigma)",
          static_cast<unsigned long long>(N), static_cast<unsigned long long>(M), static_cast<unsigned long long>(N + M), rng_same ? 1 : 0,
          static_cast<unsigned long long>(second.E().n_measured), static_cast<unsigned long long>(props_second), clean_end ? 1 : 0,
          es.energy, es.energy_error, ef.energy, ef.energy_error, z);
}

int main(){
    test::Checks check {"checkpoint/restart (diagram state, estimator and update hooks, continuation)"};
    diagram_round_trip(check);
    estimator_round_trip(check);
    continuation(check);
    return check.exit_code();
}
