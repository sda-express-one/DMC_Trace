#ifndef DMC_UPDATE_SET_HPP
#define DMC_UPDATE_SET_HPP

#include <array>
#include <map>
#include <string>
#include <string_view>
#include <utility>
#include <simplemc/mc/update_set.hpp>
#include <simplemc/random/xoshiro256.hpp>
#include "diagram/diagram_config.hpp"
#include "updates/add_internal_ph.hpp"
#include "updates/rm_internal_ph.hpp"
#include "updates/add_external_ph.hpp"
#include "updates/rm_external_ph.hpp"
#include "updates/swp_ph.hpp"
#include "updates/mv_vertex.hpp"
#include "updates/chg_tau.hpp"
#include "updates/chg_ph_momentum.hpp"
#include "updates/chg_ph_energy.hpp"
#include "updates/str_diagram.hpp"
#include "updates/scl_diagram.hpp"

// All updates in one simplemc::update_set, selected by simplemc::metropolis_kernel with probability proportional
// to their weights. add/rm of internal and of external lines are linked inverse pairs: rebuild_distribution sets
// the detailed-balance factor w_inverse/w, which the kernel multiplies into attempt() (whose ratios contain no
// update-selection probability). Every other update is its own inverse. attempt() returns |R| (the sign goes to
// the SignCounter) or -1 for an impossible proposal, which the kernel rejects.
using dmc_update_set = simplemc::update_set<add_int_ph_update, rm_int_ph_update, add_ext_ph_update, rm_ext_ph_update,
                                            swp_ph_update, mv_tau_update, chg_tau_update, chg_ph_momentum, chg_ph_energy,
                                            str_diagram_update, scl_diagram_update>;

// names in the set's order: the keys of the input file's "updates" block
inline constexpr std::array<std::string_view, 11> update_names {
    "add_int_ph", "rm_int_ph", "add_ext_ph", "rm_ext_ph", "swp_ph", "mv_tau", "chg_tau",
    "chg_ph_momentum", "chg_ph_energy", "str_diagram", "scl_diagram"};

inline constexpr std::array<std::pair<std::string_view, std::string_view>, 2> update_inverse_pairs {{
    {"add_int_ph", "rm_int_ph"}, {"add_ext_ph", "rm_ext_ph"}}};

// the updates that change tau_D: weight 0 in fixed-length runs
inline constexpr std::array<std::string_view, 3> length_changing_updates {"chg_tau", "str_diagram", "scl_diagram"};

// The set bound to one diagram and rng, with the given weights (a name missing from the map keeps weight 1),
// the inverse pairs linked and the selection distribution built. Throws simplemc::simplemc_exception on
// inconsistent weights (negative, one of a pair zero, all zero).
dmc_update_set makeUpdateSet(diagram_cfg * cfg, simplemc::xoshiro256ss * rng, const std::map<std::string, double> & weights);

#endif // !DMC_UPDATE_SET_HPP
