#include "../../include/updates/rm_external_ph.hpp"
#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <Eigen/Core>
#include <Eigen/SVD>
#include "../../include/utils/numerical.hpp"
#include "../../include/diagram/vertex.hpp"
#include "../../include/diagram/vertex_manager.hpp"

double rm_ext_ph_update::attempt(){
    proposed_weights_beginning.clear();
    proposed_weights_end.clear();
    proposed_weights_middle.clear();

    assert(cfg->external_ph_manager->current_length % 2 == 0);
    if(cfg->external_ph_manager->current_length < 2){
        return -1; 
    }
    
    ptr_one = cfg->external_ph_manager->chooseOutgoingVertex()->conjugated->linked_vertex;
    ptr_two = ptr_one->conj_vertex;

    assert(ptr_one != nullptr);
    assert(ptr_two != nullptr);

    const double tau_one {ptr_one->tau};
    const double tau_two {ptr_two->tau};
    assert(!numerical::isEqual(tau_one, tau_two));
    
    const double ph_mode_energy {ptr_one->ph_energy};
    const double ph_mode_diel_response {ptr_one->diel_response};
    const std::array<double, 3> w_to_reject {ptr_one->w};
    std::array<double, 3> p_fin {0., 0., 0.};
    double p_fin_sq {0.};

    incoming_before_outgoing = tau_one < tau_two;

    Eigen::Matrix<double, 4, 3> eigensolution_wrapper;
    std::array<double, 3> eigenvalues {1., 1., 1.};

    Vertex * ptr {cfg->diagram_head};

    if(incoming_before_outgoing){
        // mirrors add_ext_ph_update's case 1, reverted: beginning [diagram_head...ptr_one->prev]
        // and end [ptr_two->next...diagram_tail] both revert from shifted back to unshifted
        // (+w_to_reject); middle [ptr_one->next...ptr_two->prev] is untouched. ptr_one->prev
        // absorbs ptr_one (extending to ptr_one->next); ptr_two->prev absorbs ptr_two (extending
        // to ptr_two->next) - unless ptr_two == ptr_one->next (no middle at all), in which case
        // ptr_one->prev absorbs BOTH in one step, extending straight to ptr_two->next.
        const bool adjacent {ptr_two == ptr_one->next};

        // beginning walk: diagram_head -> ptr_one->prev->prev, pure revert (ptr_one->prev itself
        // is handled separately below, since it also needs the merge treatment, not just revert).
        while (ptr != ptr_one->prev) {
            weight::ProposedVertexWeight current_new_weight;
            p_fin = {ptr->k[0] + w_to_reject[0], ptr->k[1] + w_to_reject[1], ptr->k[2] + w_to_reject[2]};
            current_new_weight.k = p_fin;
            p_fin_sq = p_fin[0]*p_fin[0] + p_fin[1]*p_fin[1] + p_fin[2]*p_fin[2];

            eigensolution_wrapper = weight::LKMatrix::diagonalizeLKHamiltonian(p_fin);
            eigenvalues = {eigensolution_wrapper(0,0), eigensolution_wrapper(0,1), eigensolution_wrapper(0,2)};
            current_new_weight.eff_masses = weight::LKMatrix::computeEffMassfromEigenval(eigenvalues);
            current_new_weight.baseWF = eigensolution_wrapper.block<3,3>(1,0);

            if (ptr == cfg->diagram_head) {
                current_new_weight.vertex_wf_component = Eigen::Matrix3d::Identity();
            }
            else {
                current_new_weight.vertex_wf_component = Coupling::LKOverlap::computeMatrix(proposed_weights_beginning.back().baseWF, current_new_weight.baseWF);
            }

            current_new_weight.el_prop_action(0,0) = std::exp(-p_fin_sq/(2*current_new_weight.eff_masses[0])*(ptr->tau_next - ptr->tau));
            current_new_weight.el_prop_action(1,1) = std::exp(-p_fin_sq/(2*current_new_weight.eff_masses[1])*(ptr->tau_next - ptr->tau));
            current_new_weight.el_prop_action(2,2) = std::exp(-p_fin_sq/(2*current_new_weight.eff_masses[2])*(ptr->tau_next - ptr->tau));

            proposed_weights_beginning.push_back(current_new_weight);
            ptr = ptr->next;
        }

        // ptr_one->prev absorbs ptr_one (and ptr_two too, if adjacent) - reuses ptr_one's OWN
        // cached k/eff_masses/baseWF directly rather than recomputing: crossing an annihilation
        // vertex reverts the shift for whatever follows, so ptr_one->k already equals
        // ptr_one->prev->k + w_to_reject exactly - reusing it avoids a redundant diagonalization
        // and mirrors rm_internal_ph.cpp's merge-point convention (reuse whichever side already
        // holds the target value).
        {
            Vertex * prev {ptr_one->prev};
            weight::ProposedVertexWeight prev_weight;
            prev_weight.k = ptr_one->k;
            prev_weight.eff_masses = ptr_one->eff_masses;
            prev_weight.baseWF = ptr_one->baseWF;

            if (prev == cfg->diagram_head) {
                prev_weight.vertex_wf_component = Eigen::Matrix3d::Identity();
            }
            else {
                prev_weight.vertex_wf_component = Coupling::LKOverlap::computeMatrix(proposed_weights_beginning.back().baseWF, prev_weight.baseWF);
            }

            const double k_sq {ptr_one->k[0]*ptr_one->k[0] + ptr_one->k[1]*ptr_one->k[1] + ptr_one->k[2]*ptr_one->k[2]};
            const double duration_end {adjacent ? ptr_two->tau_next : ptr_one->tau_next};
            prev_weight.el_prop_action(0,0) = std::exp(-k_sq/(2*prev_weight.eff_masses[0])*(duration_end - prev->tau));
            prev_weight.el_prop_action(1,1) = std::exp(-k_sq/(2*prev_weight.eff_masses[1])*(duration_end - prev->tau));
            prev_weight.el_prop_action(2,2) = std::exp(-k_sq/(2*prev_weight.eff_masses[2])*(duration_end - prev->tau));

            proposed_weights_beginning.push_back(prev_weight);
        }

        // middle: untouched - exact via left_component division when well-conditioned (same
        // trick as add_ext_ph_update's case 1), falling back to an explicit walk otherwise. Only
        // meaningful when NOT adjacent; Identity otherwise (no middle at all). Stops at
        // ptr_two->prev EXCLUSIVE (dividing by left_component(ptr_two->prev), not
        // left_component(ptr_two), and the fallback walk stopping at != ptr_two->prev, not
        // != ptr_two) - ptr_two->prev's own contribution is folded into the merge entry pushed
        // below instead, so including it here too would double-count it (once with its stale
        // short duration here, once with its correct extended duration there). This also
        // degrades correctly when there's exactly one interior vertex (ptr_two->prev ==
        // ptr_one->next): both sides of the division become the same matrix, giving Identity.
        Eigen::Matrix3d middle_product {Eigen::Matrix3d::Identity()};
        if (!adjacent) {
            const Eigen::Matrix3d & left_next {ptr_one->next->left_component};
            Eigen::JacobiSVD<Eigen::Matrix3d> svd {left_next, Eigen::ComputeFullU | Eigen::ComputeFullV};
            const double smallest_sv {svd.singularValues()(2)};
            const double condition_number {svd.singularValues()(0) / smallest_sv};

            constexpr double min_singular_value {1e-12};
            constexpr double max_condition_number {1e10};

            if (smallest_sv > min_singular_value && condition_number < max_condition_number) {
                middle_product = svd.solve(ptr_two->prev->left_component);
            }
            else {
                Vertex * ptr_mid {ptr_one->next};
                while (ptr_mid != ptr_two->prev) {
                    middle_product = middle_product * ptr_mid->vertex_wf_component * ptr_mid->el_prop_action;
                    ptr_mid = ptr_mid->next;
                }
            }
        }

        // ptr_two->prev absorbs ptr_two - only when NOT adjacent (already folded into the
        // beginning's merge point above otherwise). Reuses ptr_two->prev's OWN (already
        // unshifted) data - it already holds the value this merge should end up with, so only
        // its duration needs recomputing.
        if (!adjacent) {
            Vertex * prev_two {ptr_two->prev};
            weight::ProposedVertexWeight prev_two_weight;
            prev_two_weight.k = prev_two->k;
            prev_two_weight.eff_masses = prev_two->eff_masses;
            prev_two_weight.baseWF = prev_two->baseWF;
            // prev_two's real overlap is unchanged (middle is untouched) - read the already-
            // cached value directly instead of a placeholder deferred to the final trace, which
            // is exactly what got missed the first time around.
            prev_two_weight.vertex_wf_component = prev_two->vertex_wf_component;

            const double k_sq {prev_two->k[0]*prev_two->k[0] + prev_two->k[1]*prev_two->k[1] + prev_two->k[2]*prev_two->k[2]};
            prev_two_weight.el_prop_action(0,0) = std::exp(-k_sq/(2*prev_two_weight.eff_masses[0])*(ptr_two->tau_next - prev_two->tau));
            prev_two_weight.el_prop_action(1,1) = std::exp(-k_sq/(2*prev_two_weight.eff_masses[1])*(ptr_two->tau_next - prev_two->tau));
            prev_two_weight.el_prop_action(2,2) = std::exp(-k_sq/(2*prev_two_weight.eff_masses[2])*(ptr_two->tau_next - prev_two->tau));

            proposed_weights_end.push_back(prev_two_weight);
        }

        // pure revert: ptr_two->next -> diagram_tail->prev. The first entry here chains off
        // whichever region's last entry actually precedes it: proposed_weights_end's own (if the
        // ptr_two->prev merge above was pushed) or proposed_weights_beginning's (if adjacent,
        // since that's where the merged, ptr_two-absorbing entry landed instead).
        ptr = ptr_two->next;
        while (ptr != cfg->diagram_tail) {
            weight::ProposedVertexWeight current_new_weight;
            p_fin = {ptr->k[0] + w_to_reject[0], ptr->k[1] + w_to_reject[1], ptr->k[2] + w_to_reject[2]};
            current_new_weight.k = p_fin;
            p_fin_sq = p_fin[0]*p_fin[0] + p_fin[1]*p_fin[1] + p_fin[2]*p_fin[2];

            eigensolution_wrapper = weight::LKMatrix::diagonalizeLKHamiltonian(p_fin);
            eigenvalues = {eigensolution_wrapper(0,0), eigensolution_wrapper(0,1), eigensolution_wrapper(0,2)};
            current_new_weight.eff_masses = weight::LKMatrix::computeEffMassfromEigenval(eigenvalues);
            current_new_weight.baseWF = eigensolution_wrapper.block<3,3>(1,0);

            const Eigen::Matrix3d & prev_baseWF {proposed_weights_end.empty() ? proposed_weights_beginning.back().baseWF : proposed_weights_end.back().baseWF};
            current_new_weight.vertex_wf_component = Coupling::LKOverlap::computeMatrix(prev_baseWF, current_new_weight.baseWF);

            current_new_weight.el_prop_action(0,0) = std::exp(-p_fin_sq/(2*current_new_weight.eff_masses[0])*(ptr->tau_next - ptr->tau));
            current_new_weight.el_prop_action(1,1) = std::exp(-p_fin_sq/(2*current_new_weight.eff_masses[1])*(ptr->tau_next - ptr->tau));
            current_new_weight.el_prop_action(2,2) = std::exp(-p_fin_sq/(2*current_new_weight.eff_masses[2])*(ptr->tau_next - ptr->tau));

            proposed_weights_end.push_back(current_new_weight);
            ptr = ptr->next;
        }

        // trace assembly: same shape as add_ext_ph_update's case 1, run in reverse - weight_proposed
        // is the diagram WITHOUT this line (no phonon coupling/decay factors, since there's no
        // phonon); weight_current is the diagram WITH it (right_component.trace() plus the line's
        // own coupling/decay factors, which aren't part of the electron trace at all).
        Eigen::Matrix3d beginning_product {Eigen::Matrix3d::Identity()};
        for (const auto & entry : proposed_weights_beginning) {
            beginning_product = beginning_product * entry.vertex_wf_component * entry.el_prop_action;
        }

        Eigen::Matrix3d end_product {Eigen::Matrix3d::Identity()};
        for (const auto & entry : proposed_weights_end) {
            end_product = end_product * entry.vertex_wf_component * entry.el_prop_action;
        }

        const double weight_proposed {
            (beginning_product * middle_product * end_product).trace()
        };
        const double weight_current {
            this->cfg->diagram_head->right_component.trace() *
            Coupling::Strength::compute(w_to_reject, ph_mode_energy, ph_mode_diel_response) *
            Coupling::Strength::compute(w_to_reject, ph_mode_energy, ph_mode_diel_response) *
            std::exp(-ph_mode_energy*(cfg->current_tau_length - tau_two + tau_one))
        };

        const double p_A {1.*(static_cast<double>(cfg->external_ph_manager->current_length)/2.)};
        const double p_B {1.};

        const double numerator {
            p_A *
            std::pow(2.*std::numbers::pi, 3) *
            weight_proposed *
            ph_mode_energy * ph_mode_energy * std::exp(-ph_mode_energy*(cfg->current_tau_length - tau_two + tau_one)) *
            std::pow((cfg->current_tau_length - tau_two + tau_one)/(2*std::numbers::pi), 1.5) *
            std::exp(-((w_to_reject[0]*w_to_reject[0] + w_to_reject[1]*w_to_reject[1] + w_to_reject[2]*w_to_reject[2])/2.)*(cfg->current_tau_length -tau_two + tau_one))
        };

        const double denominator {
            p_B *
            weight_current *
            Coupling::Parameters::V_unit_cell
        };

        return numerator/denominator;
    }
    else {
        // mirrors add_ext_ph_update's case 2, reverted: beginning [diagram_head...ptr_two->prev]
        // is single-shifted, middle [ptr_two->next...ptr_one->prev] is DOUBLE-shifted, end
        // [ptr_one->next...diagram_tail] is single-shifted again - there is no untouched region
        // at all here (unlike case 1), so every entry (including both merge points) needs a
        // fresh diagonalization; neither side of either merge point already holds the fully-
        // unshifted target value the way ptr_one/ptr_two->prev did in case 1, so there's no reuse
        // shortcut available here. ptr_two->prev absorbs ptr_two (extending to ptr_two->next);
        // ptr_one->prev absorbs ptr_one (extending to ptr_one->next) - unless ptr_one->prev ==
        // ptr_two (no middle at all), in which case ptr_two->prev absorbs BOTH in one step,
        // extending straight to ptr_one->next.
        const bool adjacent {ptr_one->prev == ptr_two};
        const std::array<double, 3> w_double_to_reject {2.*w_to_reject[0], 2.*w_to_reject[1], 2.*w_to_reject[2]};

        // beginning walk: diagram_head -> ptr_two->prev->prev, pure revert (single shift).
        ptr = cfg->diagram_head;
        while (ptr != ptr_two->prev) {
            weight::ProposedVertexWeight current_new_weight;
            p_fin = {ptr->k[0] + w_to_reject[0], ptr->k[1] + w_to_reject[1], ptr->k[2] + w_to_reject[2]};
            current_new_weight.k = p_fin;
            p_fin_sq = p_fin[0]*p_fin[0] + p_fin[1]*p_fin[1] + p_fin[2]*p_fin[2];

            eigensolution_wrapper = weight::LKMatrix::diagonalizeLKHamiltonian(p_fin);
            eigenvalues = {eigensolution_wrapper(0,0), eigensolution_wrapper(0,1), eigensolution_wrapper(0,2)};
            current_new_weight.eff_masses = weight::LKMatrix::computeEffMassfromEigenval(eigenvalues);
            current_new_weight.baseWF = eigensolution_wrapper.block<3,3>(1,0);

            if (ptr == cfg->diagram_head) {
                current_new_weight.vertex_wf_component = Eigen::Matrix3d::Identity();
            }
            else {
                current_new_weight.vertex_wf_component = Coupling::LKOverlap::computeMatrix(proposed_weights_beginning.back().baseWF, current_new_weight.baseWF);
            }

            current_new_weight.el_prop_action(0,0) = std::exp(-p_fin_sq/(2*current_new_weight.eff_masses[0])*(ptr->tau_next - ptr->tau));
            current_new_weight.el_prop_action(1,1) = std::exp(-p_fin_sq/(2*current_new_weight.eff_masses[1])*(ptr->tau_next - ptr->tau));
            current_new_weight.el_prop_action(2,2) = std::exp(-p_fin_sq/(2*current_new_weight.eff_masses[2])*(ptr->tau_next - ptr->tau));

            proposed_weights_beginning.push_back(current_new_weight);
            ptr = ptr->next;
        }

        // ptr_two->prev absorbs ptr_two (and ptr_one too, if adjacent) - fresh diagonalization
        // (single-shift reversion), since ptr_two->prev's own k is still single-shifted and
        // ptr_two's own k is double-shifted; neither already holds the unshifted target.
        {
            Vertex * prev_two {ptr_two->prev};
            weight::ProposedVertexWeight prev_two_weight;
            p_fin = {prev_two->k[0] + w_to_reject[0], prev_two->k[1] + w_to_reject[1], prev_two->k[2] + w_to_reject[2]};
            prev_two_weight.k = p_fin;
            p_fin_sq = p_fin[0]*p_fin[0] + p_fin[1]*p_fin[1] + p_fin[2]*p_fin[2];

            eigensolution_wrapper = weight::LKMatrix::diagonalizeLKHamiltonian(p_fin);
            eigenvalues = {eigensolution_wrapper(0,0), eigensolution_wrapper(0,1), eigensolution_wrapper(0,2)};
            prev_two_weight.eff_masses = weight::LKMatrix::computeEffMassfromEigenval(eigenvalues);
            prev_two_weight.baseWF = eigensolution_wrapper.block<3,3>(1,0);

            if (prev_two == cfg->diagram_head) {
                prev_two_weight.vertex_wf_component = Eigen::Matrix3d::Identity();
            }
            else {
                prev_two_weight.vertex_wf_component = Coupling::LKOverlap::computeMatrix(proposed_weights_beginning.back().baseWF, prev_two_weight.baseWF);
            }

            const double duration_end {adjacent ? ptr_one->tau_next : ptr_two->tau_next};
            prev_two_weight.el_prop_action(0,0) = std::exp(-p_fin_sq/(2*prev_two_weight.eff_masses[0])*(duration_end - prev_two->tau));
            prev_two_weight.el_prop_action(1,1) = std::exp(-p_fin_sq/(2*prev_two_weight.eff_masses[1])*(duration_end - prev_two->tau));
            prev_two_weight.el_prop_action(2,2) = std::exp(-p_fin_sq/(2*prev_two_weight.eff_masses[2])*(duration_end - prev_two->tau));

            proposed_weights_beginning.push_back(prev_two_weight);
        }

        // middle: ptr_two->next -> ptr_one->prev->prev, pure revert (double shift) - only when
        // NOT adjacent. The first entry here (and the merge entry below, if the loop runs zero
        // times) chains off whichever of proposed_weights_beginning/_middle actually precedes it.
        if (!adjacent) {
            ptr = ptr_two->next;
            while (ptr != ptr_one->prev) {
                weight::ProposedVertexWeight current_new_weight;
                p_fin = {ptr->k[0] + w_double_to_reject[0], ptr->k[1] + w_double_to_reject[1], ptr->k[2] + w_double_to_reject[2]};
                current_new_weight.k = p_fin;
                p_fin_sq = p_fin[0]*p_fin[0] + p_fin[1]*p_fin[1] + p_fin[2]*p_fin[2];

                eigensolution_wrapper = weight::LKMatrix::diagonalizeLKHamiltonian(p_fin);
                eigenvalues = {eigensolution_wrapper(0,0), eigensolution_wrapper(0,1), eigensolution_wrapper(0,2)};
                current_new_weight.eff_masses = weight::LKMatrix::computeEffMassfromEigenval(eigenvalues);
                current_new_weight.baseWF = eigensolution_wrapper.block<3,3>(1,0);

                const Eigen::Matrix3d & prev_baseWF {proposed_weights_middle.empty() ? proposed_weights_beginning.back().baseWF : proposed_weights_middle.back().baseWF};
                current_new_weight.vertex_wf_component = Coupling::LKOverlap::computeMatrix(prev_baseWF, current_new_weight.baseWF);

                current_new_weight.el_prop_action(0,0) = std::exp(-p_fin_sq/(2*current_new_weight.eff_masses[0])*(ptr->tau_next - ptr->tau));
                current_new_weight.el_prop_action(1,1) = std::exp(-p_fin_sq/(2*current_new_weight.eff_masses[1])*(ptr->tau_next - ptr->tau));
                current_new_weight.el_prop_action(2,2) = std::exp(-p_fin_sq/(2*current_new_weight.eff_masses[2])*(ptr->tau_next - ptr->tau));

                proposed_weights_middle.push_back(current_new_weight);
                ptr = ptr->next;
            }

            // ptr_one->prev absorbs ptr_one - fresh diagonalization (double-shift reversion),
            // since ptr_one->prev's own k is still double-shifted, not unshifted.
            Vertex * prev_one {ptr_one->prev};
            weight::ProposedVertexWeight prev_one_weight;
            p_fin = {prev_one->k[0] + w_double_to_reject[0], prev_one->k[1] + w_double_to_reject[1], prev_one->k[2] + w_double_to_reject[2]};
            prev_one_weight.k = p_fin;
            p_fin_sq = p_fin[0]*p_fin[0] + p_fin[1]*p_fin[1] + p_fin[2]*p_fin[2];

            eigensolution_wrapper = weight::LKMatrix::diagonalizeLKHamiltonian(p_fin);
            eigenvalues = {eigensolution_wrapper(0,0), eigensolution_wrapper(0,1), eigensolution_wrapper(0,2)};
            prev_one_weight.eff_masses = weight::LKMatrix::computeEffMassfromEigenval(eigenvalues);
            prev_one_weight.baseWF = eigensolution_wrapper.block<3,3>(1,0);

            const Eigen::Matrix3d & prev_baseWF {proposed_weights_middle.empty() ? proposed_weights_beginning.back().baseWF : proposed_weights_middle.back().baseWF};
            prev_one_weight.vertex_wf_component = Coupling::LKOverlap::computeMatrix(prev_baseWF, prev_one_weight.baseWF);

            prev_one_weight.el_prop_action(0,0) = std::exp(-p_fin_sq/(2*prev_one_weight.eff_masses[0])*(ptr_one->tau_next - prev_one->tau));
            prev_one_weight.el_prop_action(1,1) = std::exp(-p_fin_sq/(2*prev_one_weight.eff_masses[1])*(ptr_one->tau_next - prev_one->tau));
            prev_one_weight.el_prop_action(2,2) = std::exp(-p_fin_sq/(2*prev_one_weight.eff_masses[2])*(ptr_one->tau_next - prev_one->tau));

            proposed_weights_middle.push_back(prev_one_weight);
        }

        // end walk: ptr_one->next -> diagram_tail->prev, pure revert (single shift). The first
        // entry chains off proposed_weights_middle's last entry, or proposed_weights_beginning's
        // if adjacent (middle is empty then, since both merges landed in beginning together).
        ptr = ptr_one->next;
        while (ptr != cfg->diagram_tail) {
            weight::ProposedVertexWeight current_new_weight;
            p_fin = {ptr->k[0] + w_to_reject[0], ptr->k[1] + w_to_reject[1], ptr->k[2] + w_to_reject[2]};
            current_new_weight.k = p_fin;
            p_fin_sq = p_fin[0]*p_fin[0] + p_fin[1]*p_fin[1] + p_fin[2]*p_fin[2];

            eigensolution_wrapper = weight::LKMatrix::diagonalizeLKHamiltonian(p_fin);
            eigenvalues = {eigensolution_wrapper(0,0), eigensolution_wrapper(0,1), eigensolution_wrapper(0,2)};
            current_new_weight.eff_masses = weight::LKMatrix::computeEffMassfromEigenval(eigenvalues);
            current_new_weight.baseWF = eigensolution_wrapper.block<3,3>(1,0);

            const Eigen::Matrix3d & prev_baseWF {proposed_weights_middle.empty() ? proposed_weights_beginning.back().baseWF : proposed_weights_middle.back().baseWF};
            current_new_weight.vertex_wf_component = Coupling::LKOverlap::computeMatrix(prev_baseWF, current_new_weight.baseWF);

            current_new_weight.el_prop_action(0,0) = std::exp(-p_fin_sq/(2*current_new_weight.eff_masses[0])*(ptr->tau_next - ptr->tau));
            current_new_weight.el_prop_action(1,1) = std::exp(-p_fin_sq/(2*current_new_weight.eff_masses[1])*(ptr->tau_next - ptr->tau));
            current_new_weight.el_prop_action(2,2) = std::exp(-p_fin_sq/(2*current_new_weight.eff_masses[2])*(ptr->tau_next - ptr->tau));

            proposed_weights_end.push_back(current_new_weight);
            ptr = ptr->next;
        }

        // trace assembly: same shape as case 1, but no untouched region - all three folds are
        // freshly computed straight from proposed_weights_*, no cached-quantity shortcut needed.
        Eigen::Matrix3d beginning_product {Eigen::Matrix3d::Identity()};
        for (const auto & entry : proposed_weights_beginning) {
            beginning_product = beginning_product * entry.vertex_wf_component * entry.el_prop_action;
        }

        Eigen::Matrix3d middle_product {Eigen::Matrix3d::Identity()};
        for (const auto & entry : proposed_weights_middle) {
            middle_product = middle_product * entry.vertex_wf_component * entry.el_prop_action;
        }

        Eigen::Matrix3d end_product {Eigen::Matrix3d::Identity()};
        for (const auto & entry : proposed_weights_end) {
            end_product = end_product * entry.vertex_wf_component * entry.el_prop_action;
        }

        const double weight_proposed {
            (beginning_product * middle_product * end_product).trace()
        };
        const double weight_current {
            this->cfg->diagram_head->right_component.trace() *
            Coupling::Strength::compute(w_to_reject, ph_mode_energy, ph_mode_diel_response) *
            Coupling::Strength::compute(w_to_reject, ph_mode_energy, ph_mode_diel_response) *
            std::exp(-ph_mode_energy*(cfg->current_tau_length - tau_two + tau_one))
        };

        const double p_A {1.*(static_cast<double>(cfg->external_ph_manager->current_length)/2.)};
        const double p_B {1.};

        const double numerator {
            p_A *
            std::pow(2.*std::numbers::pi, 3) *
            weight_proposed *
            ph_mode_energy * ph_mode_energy * std::exp(-ph_mode_energy*(cfg->current_tau_length - tau_two + tau_one)) *
            std::pow((cfg->current_tau_length - tau_two + tau_one)/(2*std::numbers::pi), 1.5) *
            std::exp(-((w_to_reject[0]*w_to_reject[0] + w_to_reject[1]*w_to_reject[1] + w_to_reject[2]*w_to_reject[2])/2.)*(cfg->current_tau_length -tau_two + tau_one))
        };

        const double denominator {
            p_B *
            weight_current *
            Coupling::Parameters::V_unit_cell
        };

        return numerator/denominator;
    }
}

void rm_ext_ph_update::accept(){
    if (incoming_before_outgoing) {
        // === case 1: annihilation (tau_one) before creation (tau_two) ===
        const bool adjacent {ptr_two == ptr_one->next};

        // 1. commit reverted state onto diagram_head...ptr_one->prev->prev (pure revert), then
        //    ptr_one->prev itself (the merge entry: absorbs ptr_one, and ptr_two too if adjacent).
        {
            Vertex * ptr {cfg->diagram_head};
            std::size_t idx {0};
            while (ptr != ptr_one->prev) {
                ptr->k = proposed_weights_beginning[idx].k;
                ptr->eff_masses = proposed_weights_beginning[idx].eff_masses;
                ptr->baseWF = proposed_weights_beginning[idx].baseWF;
                ptr->vertex_wf_component = proposed_weights_beginning[idx].vertex_wf_component;
                ptr->el_prop_action = proposed_weights_beginning[idx].el_prop_action;
                ptr = ptr->next;
                ++idx;
            }
            // ptr is now ptr_one->prev; proposed_weights_beginning[idx] is its merged piece.
            ptr_one->prev->k = proposed_weights_beginning[idx].k;
            ptr_one->prev->eff_masses = proposed_weights_beginning[idx].eff_masses;
            ptr_one->prev->baseWF = proposed_weights_beginning[idx].baseWF;
            ptr_one->prev->vertex_wf_component = proposed_weights_beginning[idx].vertex_wf_component;
            ptr_one->prev->el_prop_action = proposed_weights_beginning[idx].el_prop_action;
            ptr_one->prev->tau_next = adjacent ? ptr_two->tau_next : ptr_one->tau_next;
        }

        // 2. commit reverted state onto ptr_two->prev (merge entry: absorbs ptr_two) - only if
        //    NOT adjacent (already folded into step 1 otherwise), then the pure-revert stretch
        //    ptr_two->next...diagram_tail->prev. Both read from proposed_weights_end, whose
        //    layout shifts by one index depending on whether the merge entry was pushed.
        std::size_t end_idx {0};
        if (!adjacent) {
            ptr_two->prev->k = proposed_weights_end[0].k;
            ptr_two->prev->eff_masses = proposed_weights_end[0].eff_masses;
            ptr_two->prev->baseWF = proposed_weights_end[0].baseWF;
            ptr_two->prev->vertex_wf_component = proposed_weights_end[0].vertex_wf_component;
            ptr_two->prev->el_prop_action = proposed_weights_end[0].el_prop_action;
            ptr_two->prev->tau_next = ptr_two->tau_next;
            end_idx = 1;
        }
        {
            Vertex * ptr {ptr_two->next};
            std::size_t idx {end_idx};
            while (ptr != cfg->diagram_tail) {
                ptr->k = proposed_weights_end[idx].k;
                ptr->eff_masses = proposed_weights_end[idx].eff_masses;
                ptr->baseWF = proposed_weights_end[idx].baseWF;
                ptr->vertex_wf_component = proposed_weights_end[idx].vertex_wf_component;
                ptr->el_prop_action = proposed_weights_end[idx].el_prop_action;
                ptr = ptr->next;
                ++idx;
            }
        }
    }
    else {
        // === case 2: creation (tau_two) before annihilation (tau_one) ===
        const bool adjacent {ptr_one->prev == ptr_two};

        // 1. commit reverted state onto diagram_head...ptr_two->prev->prev (pure revert), then
        //    ptr_two->prev itself (the merge entry: absorbs ptr_two, and ptr_one too if adjacent).
        {
            Vertex * ptr {cfg->diagram_head};
            std::size_t idx {0};
            while (ptr != ptr_two->prev) {
                ptr->k = proposed_weights_beginning[idx].k;
                ptr->eff_masses = proposed_weights_beginning[idx].eff_masses;
                ptr->baseWF = proposed_weights_beginning[idx].baseWF;
                ptr->vertex_wf_component = proposed_weights_beginning[idx].vertex_wf_component;
                ptr->el_prop_action = proposed_weights_beginning[idx].el_prop_action;
                ptr = ptr->next;
                ++idx;
            }
            // ptr is now ptr_two->prev; proposed_weights_beginning[idx] is its merged piece.
            ptr_two->prev->k = proposed_weights_beginning[idx].k;
            ptr_two->prev->eff_masses = proposed_weights_beginning[idx].eff_masses;
            ptr_two->prev->baseWF = proposed_weights_beginning[idx].baseWF;
            ptr_two->prev->vertex_wf_component = proposed_weights_beginning[idx].vertex_wf_component;
            ptr_two->prev->el_prop_action = proposed_weights_beginning[idx].el_prop_action;
            ptr_two->prev->tau_next = adjacent ? ptr_one->tau_next : ptr_two->tau_next;
        }

        // 2. commit reverted state onto ptr_two->next...ptr_one->prev->prev (pure revert), then
        //    ptr_one->prev itself (the merge entry: absorbs ptr_one) - only when NOT adjacent
        //    (already folded into step 1 otherwise).
        if (!adjacent) {
            Vertex * ptr {ptr_two->next};
            std::size_t idx {0};
            while (ptr != ptr_one->prev) {
                ptr->k = proposed_weights_middle[idx].k;
                ptr->eff_masses = proposed_weights_middle[idx].eff_masses;
                ptr->baseWF = proposed_weights_middle[idx].baseWF;
                ptr->vertex_wf_component = proposed_weights_middle[idx].vertex_wf_component;
                ptr->el_prop_action = proposed_weights_middle[idx].el_prop_action;
                ptr = ptr->next;
                ++idx;
            }
            // ptr is now ptr_one->prev; proposed_weights_middle[idx] is its merged piece.
            ptr_one->prev->k = proposed_weights_middle[idx].k;
            ptr_one->prev->eff_masses = proposed_weights_middle[idx].eff_masses;
            ptr_one->prev->baseWF = proposed_weights_middle[idx].baseWF;
            ptr_one->prev->vertex_wf_component = proposed_weights_middle[idx].vertex_wf_component;
            ptr_one->prev->el_prop_action = proposed_weights_middle[idx].el_prop_action;
            ptr_one->prev->tau_next = ptr_one->tau_next;
        }

        // 3. commit reverted state onto ptr_one->next...diagram_tail->prev (pure revert only -
        //    no merge entry here, both merges land in beginning/middle above).
        {
            Vertex * ptr {ptr_one->next};
            std::size_t idx {0};
            while (ptr != cfg->diagram_tail) {
                ptr->k = proposed_weights_end[idx].k;
                ptr->eff_masses = proposed_weights_end[idx].eff_masses;
                ptr->baseWF = proposed_weights_end[idx].baseWF;
                ptr->vertex_wf_component = proposed_weights_end[idx].vertex_wf_component;
                ptr->el_prop_action = proposed_weights_end[idx].el_prop_action;
                ptr = ptr->next;
                ++idx;
            }
        }
    }

    // shared: unregister the line (findPointer() gives ptr_one's own slot in O(1) from its
    // maintained index, .conjugated gives ptr_two's slot from there), splice both vertices out of
    // the diagram and back to the free pool, then refresh the whole diagram's cache - same as
    // add_ext_ph_update, neither case here leaves any untouched prefix/suffix to bound the
    // recomputation to.
    VertexPointer * ptr_one_slot {cfg->external_ph_manager->findPointer(ptr_one)};
    VertexPointer * ptr_two_slot {ptr_one_slot->conjugated};
    assert(ptr_two_slot != nullptr);
    cfg->external_ph_manager->removeVertexPointers(*ptr_one_slot, *ptr_two_slot);

    cfg->addVertexToPool(cfg->removeVertex(ptr_one));
    cfg->addVertexToPool(cfg->removeVertex(ptr_two));

    weight::LKMatrix::computeRightSide(cfg->diagram_head, cfg->diagram_tail);
    weight::LKMatrix::computeLeftSide(cfg->diagram_tail, cfg->diagram_head);
}
