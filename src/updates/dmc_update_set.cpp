#include "updates/dmc_update_set.hpp"

dmc_update_set makeUpdateSet(diagram_cfg * cfg, simplemc::xoshiro256ss * rng, const std::map<std::string, double> & weights){
    const auto w {[&](std::string_view name){
        const auto it {weights.find(std::string {name})};
        return it == weights.end() ? 1. : it->second;
    }};
    dmc_update_set us {
        simplemc::update {add_int_ph_update {cfg, rng}, std::string {update_names[0]}, w(update_names[0])},
        simplemc::update {rm_int_ph_update {cfg, rng}, std::string {update_names[1]}, w(update_names[1])},
        simplemc::update {add_ext_ph_update {cfg, rng}, std::string {update_names[2]}, w(update_names[2])},
        simplemc::update {rm_ext_ph_update {cfg, rng}, std::string {update_names[3]}, w(update_names[3])},
        simplemc::update {swp_ph_update {cfg, rng}, std::string {update_names[4]}, w(update_names[4])},
        simplemc::update {mv_tau_update {cfg, rng}, std::string {update_names[5]}, w(update_names[5])},
        simplemc::update {chg_tau_update {cfg, rng}, std::string {update_names[6]}, w(update_names[6])},
        simplemc::update {chg_ph_momentum {cfg, rng}, std::string {update_names[7]}, w(update_names[7])},
        simplemc::update {chg_ph_energy {cfg, rng}, std::string {update_names[8]}, w(update_names[8])},
        simplemc::update {str_diagram_update {cfg, rng}, std::string {update_names[9]}, w(update_names[9])},
        simplemc::update {scl_diagram_update {cfg, rng}, std::string {update_names[10]}, w(update_names[10])}};
    for (const auto & [a, b] : update_inverse_pairs) { us.link_pair(a, b); }
    us.rebuild_distribution();
    return us;
}
