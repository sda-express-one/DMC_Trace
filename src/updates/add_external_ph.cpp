#include "../../include/updates/add_external_ph.hpp"
#include <Eigen/src/Core/Matrix.h>
#include <Eigen/SVD>
#include <array>
#include <cassert>
#include <cmath>
#include <random>
#include "../../include/utils/numerical.hpp"
#include "../../include/vertex.hpp"

double add_ext_ph_update::attempt(){
    proposed_weights_beginning.clear();
    proposed_weights_end.clear();
    proposed_weights_middle.clear();

    if(cfg->external_ph_manager->current_length == cfg->external_ph_manager->max_length){
        return -1.;
    }

    const int current_total_order {cfg->internal_ph_manager->current_length + cfg->external_ph_manager->current_length};

    ph_index = this->cfg->phonon_mode_manager->drawPhononMode();
    const double ph_mode_energy {this->cfg->phonon_mode_manager->phonon_mode_pool[ph_index].phonon_energy};
    const double ph_mode_diel_response {this->cfg->phonon_mode_manager->phonon_mode_pool[ph_index].diel_response};

    tau_one = 0. - std::log(1-this->std_unif(*rng))/ph_mode_energy;
    tau_two = cfg->current_tau_length + std::log(1-this->std_unif(*rng))/ph_mode_energy;

    if(tau_one < 0 || tau_two < 0){
        return -1.;
    }
    else if(numerical::isEqual(tau_one, tau_two)){
        return -1.;
    }
    else if(tau_one > cfg->current_tau_length || tau_two > cfg->current_tau_length){
        return -1;
    }

    std::normal_distribution<double> distrib_norm(0, std::sqrt(1/(cfg->current_tau_length-tau_two+tau_one)));
    w_proposed = {distrib_norm(*this->rng), distrib_norm(*this->rng), distrib_norm(*this->rng)};

    if (tau_one < tau_two){
        incoming_before_outgoing = true;

        ptr_one = this->cfg->findPositionFromLeft(this->cfg->diagram_head, tau_one);
        ptr_two = this->cfg->findPositionFromRight(this->cfg->diagram_tail->prev, tau_two);

        assert(ptr_one != nullptr);
        assert(ptr_two != nullptr);

        std::array<double, 3> p_fin {0., 0., 0.};
        double p_fin_sq {0.};

        Eigen::Matrix<double, 4, 3> eigensolution_wrapper;
        std::array<double, 3> eigenvalues {1., 1., 1.};

        // beginning walk: diagram_head -> ptr_one, forward. This whole stretch sits inside the
        // external phonon's (wrapped) existence span - same shift as an internal line's interior.
        Vertex * ptr {this->cfg->diagram_head};

        do {
            weight::ProposedVertexWeight current_new_weight;
            p_fin = {ptr->k[0] - w_proposed[0], ptr->k[1] - w_proposed[1], ptr->k[2] - w_proposed[2]};
            current_new_weight.k = p_fin;
            p_fin_sq = p_fin[0]*p_fin[0] + p_fin[1]*p_fin[1] + p_fin[2]*p_fin[2];

            eigensolution_wrapper = weight::LKMatrix::diagonalizeLKHamiltonian(p_fin);
            eigenvalues = {eigensolution_wrapper(0,0), eigensolution_wrapper(0,1), eigensolution_wrapper(0,2)};
            current_new_weight.eff_masses = weight::LKMatrix::computeEffMassfromEigenval(eigenvalues);
            current_new_weight.baseWF = eigensolution_wrapper.block<3,3>(1,0);

            // diagram_head has no real predecessor - its own overlap stays Identity, matching its
            // existing convention, rather than chaining off a (nonexistent) previous entry.
            if(ptr == this->cfg->diagram_head){
                current_new_weight.vertex_wf_component = Eigen::Matrix3d::Identity();
            }
            else {
                current_new_weight.vertex_wf_component = Coupling::LKOverlap::computeMatrix(proposed_weights_beginning.back().baseWF, current_new_weight.baseWF);
            }

            if(ptr == ptr_one){
                // ptr_one's own segment shrinks to [ptr_one.tau, tau_one], still shifted
                current_new_weight.el_prop_action(0,0) = std::exp(-p_fin_sq/(2*current_new_weight.eff_masses[0])*(tau_one - ptr->tau));
                current_new_weight.el_prop_action(1,1) = std::exp(-p_fin_sq/(2*current_new_weight.eff_masses[1])*(tau_one - ptr->tau));
                current_new_weight.el_prop_action(2,2) = std::exp(-p_fin_sq/(2*current_new_weight.eff_masses[2])*(tau_one - ptr->tau));

                proposed_weights_beginning.push_back(current_new_weight);

                // the new annihilation vertex at tau_one reverts to ptr_one's ORIGINAL, unshifted
                // momentum - this is where the "beginning" region ends and the untouched middle begins.
                weight::ProposedVertexWeight ann_weight;
                ann_weight.k = ptr->k;
                ann_weight.eff_masses = ptr->eff_masses;
                ann_weight.baseWF = ptr->baseWF;
                ann_weight.vertex_wf_component = Coupling::LKOverlap::computeMatrix(current_new_weight.baseWF, ann_weight.baseWF);

                const double k_ann_sq {ptr->k[0]*ptr->k[0] + ptr->k[1]*ptr->k[1] + ptr->k[2]*ptr->k[2]};
                if(ptr_one != ptr_two){
                    ann_weight.el_prop_action(0,0) = std::exp(-k_ann_sq/(2*ann_weight.eff_masses[0])*(ptr->tau_next - tau_one));
                    ann_weight.el_prop_action(1,1) = std::exp(-k_ann_sq/(2*ann_weight.eff_masses[1])*(ptr->tau_next - tau_one));
                    ann_weight.el_prop_action(2,2) = std::exp(-k_ann_sq/(2*ann_weight.eff_masses[2])*(ptr->tau_next - tau_one));
                }
                else {
                    ann_weight.el_prop_action(0,0) = std::exp(-k_ann_sq/(2*ann_weight.eff_masses[0])*(tau_two - tau_one));
                    ann_weight.el_prop_action(1,1) = std::exp(-k_ann_sq/(2*ann_weight.eff_masses[1])*(tau_two - tau_one));
                    ann_weight.el_prop_action(2,2) = std::exp(-k_ann_sq/(2*ann_weight.eff_masses[2])*(tau_two - tau_one));
                }

                proposed_weights_beginning.push_back(ann_weight);
            }
            else {
                current_new_weight.el_prop_action(0,0) = std::exp(-p_fin_sq/(2*current_new_weight.eff_masses[0])*(ptr->tau_next - ptr->tau));
                current_new_weight.el_prop_action(1,1) = std::exp(-p_fin_sq/(2*current_new_weight.eff_masses[1])*(ptr->tau_next - ptr->tau));
                current_new_weight.el_prop_action(2,2) = std::exp(-p_fin_sq/(2*current_new_weight.eff_masses[2])*(ptr->tau_next - ptr->tau));

                proposed_weights_beginning.push_back(current_new_weight);
            }

            ptr = ptr->next;
        } while (ptr != ptr_one->next);

        // end walk: ptr_two -> diagram_tail, forward. ptr_two's own (pre-tau_two) segment is
        // still outside the shift, only shortened; from tau_two onward everything through
        // diagram_tail is shifted, same as the beginning walk's interior. Chaining uses .back()
        // rather than an index, since (unlike the beginning walk) the double push happens on the
        // FIRST iteration here, so a naive [i-1] would be wrong for every iteration after it.
        ptr = ptr_two;

        do {
            if (ptr == ptr_two) {
                weight::ProposedVertexWeight ptr_two_weight;
                ptr_two_weight.k = ptr_two->k;
                ptr_two_weight.eff_masses = ptr_two->eff_masses;
                ptr_two_weight.baseWF = ptr_two->baseWF;
                const double k_sq {ptr_two->k[0]*ptr_two->k[0] + ptr_two->k[1]*ptr_two->k[1] + ptr_two->k[2]*ptr_two->k[2]};

                if(ptr_two != ptr_one){
                    // ptr_two's real predecessor and overlap haven't changed - read the already-
                    // cached value directly instead of recomputing it.
                    ptr_two_weight.vertex_wf_component = ptr_two->vertex_wf_component;

                    ptr_two_weight.el_prop_action(0,0) = std::exp(-k_sq/(2*ptr_two_weight.eff_masses[0])*(tau_two - ptr_two->tau));
                    ptr_two_weight.el_prop_action(1,1) = std::exp(-k_sq/(2*ptr_two_weight.eff_masses[1])*(tau_two - ptr_two->tau));
                    ptr_two_weight.el_prop_action(2,2) = std::exp(-k_sq/(2*ptr_two_weight.eff_masses[2])*(tau_two - ptr_two->tau));

                    proposed_weights_end.push_back(ptr_two_weight);
                }
                // if ptr_two == ptr_one, the [tau_one, tau_two] interval this piece would cover
                // was already pushed as ann_weight in the beginning walk above - pushing it again
                // here would double-count that stretch in the trace product. ptr_two_weight.baseWF
                // (== ptr_two->baseWF, unshifted and unchanged either way) is still valid to chain
                // creation_weight's overlap off below, whether or not it was actually pushed.

                // the new creation vertex at tau_two picks up the shift, starting the "end" region.
                weight::ProposedVertexWeight creation_weight;
                p_fin = {ptr_two->k[0] - w_proposed[0], ptr_two->k[1] - w_proposed[1], ptr_two->k[2] - w_proposed[2]};
                creation_weight.k = p_fin;
                p_fin_sq = p_fin[0]*p_fin[0] + p_fin[1]*p_fin[1] + p_fin[2]*p_fin[2];

                eigensolution_wrapper = weight::LKMatrix::diagonalizeLKHamiltonian(p_fin);
                eigenvalues = {eigensolution_wrapper(0,0), eigensolution_wrapper(0,1), eigensolution_wrapper(0,2)};
                creation_weight.eff_masses = weight::LKMatrix::computeEffMassfromEigenval(eigenvalues);
                creation_weight.baseWF = eigensolution_wrapper.block<3,3>(1,0);
                creation_weight.vertex_wf_component = Coupling::LKOverlap::computeMatrix(ptr_two_weight.baseWF, creation_weight.baseWF);

                creation_weight.el_prop_action(0,0) = std::exp(-p_fin_sq/(2*creation_weight.eff_masses[0])*(ptr_two->tau_next - tau_two));
                creation_weight.el_prop_action(1,1) = std::exp(-p_fin_sq/(2*creation_weight.eff_masses[1])*(ptr_two->tau_next - tau_two));
                creation_weight.el_prop_action(2,2) = std::exp(-p_fin_sq/(2*creation_weight.eff_masses[2])*(ptr_two->tau_next - tau_two));

                proposed_weights_end.push_back(creation_weight);
            }
            else {
                weight::ProposedVertexWeight current_new_weight;
                p_fin = {ptr->k[0] - w_proposed[0], ptr->k[1] - w_proposed[1], ptr->k[2] - w_proposed[2]};
                current_new_weight.k = p_fin;
                p_fin_sq = p_fin[0]*p_fin[0] + p_fin[1]*p_fin[1] + p_fin[2]*p_fin[2];

                eigensolution_wrapper = weight::LKMatrix::diagonalizeLKHamiltonian(p_fin);
                eigenvalues = {eigensolution_wrapper(0,0), eigensolution_wrapper(0,1), eigensolution_wrapper(0,2)};
                current_new_weight.eff_masses = weight::LKMatrix::computeEffMassfromEigenval(eigenvalues);
                current_new_weight.baseWF = eigensolution_wrapper.block<3,3>(1,0);
                current_new_weight.vertex_wf_component = Coupling::LKOverlap::computeMatrix(proposed_weights_end.back().baseWF, current_new_weight.baseWF);

                current_new_weight.el_prop_action(0,0) = std::exp(-p_fin_sq/(2*current_new_weight.eff_masses[0])*(ptr->tau_next - ptr->tau));
                current_new_weight.el_prop_action(1,1) = std::exp(-p_fin_sq/(2*current_new_weight.eff_masses[1])*(ptr->tau_next - ptr->tau));
                current_new_weight.el_prop_action(2,2) = std::exp(-p_fin_sq/(2*current_new_weight.eff_masses[2])*(ptr->tau_next - ptr->tau));

                proposed_weights_end.push_back(current_new_weight);
            }

            ptr = ptr->next;
        } while (ptr != this->cfg->diagram_tail);

        // trace assembly: FULL_trace = trace(L(X)*X.vertex_wf_component*R(X)) holds for any vertex
        // X (verified against computeLeftSide/computeRightSide's recursive definitions), but here
        // the fresh region sits at BOTH ends (beginning, end) with the untouched region bounded in
        // the MIDDLE - the mirror of add_internal_ph's topology, where the fresh region is bounded
        // and the untouched region is unbounded-to-an-end. That means no single cached
        // left_component/right_component isolates "just the middle": it's folded explicitly here
        // from each untouched vertex's already-cached vertex_wf_component/el_prop_action - no
        // physics recomputation, just reusing cached matrices.
        Eigen::Matrix3d beginning_product {Eigen::Matrix3d::Identity()};
        for (const auto & entry : proposed_weights_beginning) {
            beginning_product = beginning_product * entry.vertex_wf_component * entry.el_prop_action;
        }

        // middle_product: exact via left_component division when well-conditioned - unrolling
        // left_component's recursive definition gives left_component(ptr_one->next) =
        // left_component(ptr_one) * ptr_one.wf * ptr_one.action, and left_component(ptr_two) =
        // that same prefix * middle_product, so middle_product =
        // left_component(ptr_one->next)^-1 * left_component(ptr_two) - no walk needed, and this
        // degrades correctly to Identity when ptr_one->next == ptr_two (adjacent, empty middle).
        // Falls back to the explicit walk (numerically robust, no division) when
        // left_component(ptr_one->next) is too ill-conditioned to invert reliably - its diagonal
        // action entries shrink like exp(-energy*duration), so for a long stretch before ptr_one
        // the inversion can amplify floating-point error even though it's exact in principle.
        Eigen::Matrix3d middle_product {Eigen::Matrix3d::Identity()};
        if (ptr_one != ptr_two) {
            const Eigen::Matrix3d & left_next {ptr_one->next->left_component};
            Eigen::JacobiSVD<Eigen::Matrix3d> svd {left_next, Eigen::ComputeFullU | Eigen::ComputeFullV};
            const double smallest_sv {svd.singularValues()(2)};
            const double condition_number {svd.singularValues()(0) / smallest_sv};

            constexpr double min_singular_value {1e-12};
            constexpr double max_condition_number {1e10};

            if (smallest_sv > min_singular_value && condition_number < max_condition_number) {
                middle_product = svd.solve(ptr_two->left_component);
            }
            else {
                Vertex * ptr_mid {ptr_one->next};
                while (ptr_mid != ptr_two) {
                    middle_product = middle_product * ptr_mid->vertex_wf_component * ptr_mid->el_prop_action;
                    ptr_mid = ptr_mid->next;
                }
            }
        }

        Eigen::Matrix3d end_product {Eigen::Matrix3d::Identity()};
        for (const auto & entry : proposed_weights_end) {
            end_product = end_product * entry.vertex_wf_component * entry.el_prop_action;
        }

        // the phonon's own existence span (per computeExternalPhPropAction's wrap convention) is
        // current_tau_length - tau_two + tau_one, matching the variance used to sample w_proposed
        // above.
        const double weight_proposed {
            (beginning_product * middle_product * end_product).trace() *
            Coupling::Strength::compute(w_proposed, ph_mode_energy, ph_mode_diel_response) *
            Coupling::Strength::compute(w_proposed, ph_mode_energy, ph_mode_diel_response) *
            std::exp(-ph_mode_energy*(cfg->current_tau_length - tau_two + tau_one))
        };
        const double weight_current {this->cfg->diagram_head->right_component.trace()};

        // TODO: numerator/denominator combinatorial (p_A/p_B-style) normalization still needs to
        // be derived against the (not yet written) rm_ext_ph_update's selection probability, and
        // the tau_one/tau_two sampling density's own normalization (see the open question raised
        // separately) - not filled in here to avoid guessing at either.
        
        const double p_B {1.};
        const double p_A {1.*(static_cast<double>(cfg->external_ph_manager->current_length)/2.) + 1.};

        const double numerator {
            p_B *
            weight_proposed *
            Coupling::Parameters::V_unit_cell
        };

        const double denominator {
            p_A *
            std::pow(2*std::numbers::pi, 3) *
            weight_current *
            ph_mode_energy *ph_mode_energy * std::exp(-ph_mode_energy*(cfg->current_tau_length - tau_two + tau_one)) *
            std::pow((cfg->current_tau_length - tau_two + tau_one)/(2*std::numbers::pi), 1.5) *
            std::exp(-((w_proposed[0]*w_proposed[0] + w_proposed[1]*w_proposed[1] + w_proposed[2]*w_proposed[2])/2.)*(cfg->current_tau_length - tau_two + tau_one))
        };

        return numerator/denominator;

    }
    else {
        incoming_before_outgoing = false;

        // roles reverse relative to case 1: tau_two (creation) is now the smaller value, so it's
        // found forward from diagram_head; tau_one (annihilation) is now the larger value, found
        // backward from diagram_tail.
        ptr_two = this->cfg->findPositionFromLeft(this->cfg->diagram_head, tau_two);
        ptr_one = this->cfg->findPositionFromRight(this->cfg->diagram_tail->prev, tau_one);

        assert(ptr_one != nullptr);
        assert(ptr_two != nullptr);

        std::array<double, 3> p_fin {0., 0., 0.};
        double p_fin_sq {0.};

        Eigen::Matrix<double, 4, 3> eigensolution_wrapper;
        std::array<double, 3> eigenvalues {1., 1., 1.};

        const std::array<double, 3> w_double {2.*w_proposed[0], 2.*w_proposed[1], 2.*w_proposed[2]};

        // beginning walk: diagram_head -> ptr_two, forward, single shift. Unlike case 1's
        // beginning walk, the new vertex here (creation) doesn't revert the shift - it starts the
        // double-shifted middle instead - so it's pushed into proposed_weights_middle below, not
        // appended here.
        Vertex * ptr {this->cfg->diagram_head};

        do {
            weight::ProposedVertexWeight current_new_weight;
            p_fin = {ptr->k[0] - w_proposed[0], ptr->k[1] - w_proposed[1], ptr->k[2] - w_proposed[2]};
            current_new_weight.k = p_fin;
            p_fin_sq = p_fin[0]*p_fin[0] + p_fin[1]*p_fin[1] + p_fin[2]*p_fin[2];

            eigensolution_wrapper = weight::LKMatrix::diagonalizeLKHamiltonian(p_fin);
            eigenvalues = {eigensolution_wrapper(0,0), eigensolution_wrapper(0,1), eigensolution_wrapper(0,2)};
            current_new_weight.eff_masses = weight::LKMatrix::computeEffMassfromEigenval(eigenvalues);
            current_new_weight.baseWF = eigensolution_wrapper.block<3,3>(1,0);

            if(ptr == this->cfg->diagram_head){
                current_new_weight.vertex_wf_component = Eigen::Matrix3d::Identity();
            }
            else {
                current_new_weight.vertex_wf_component = Coupling::LKOverlap::computeMatrix(proposed_weights_beginning.back().baseWF, current_new_weight.baseWF);
            }

            if(ptr == ptr_two){
                // ptr_two's own segment shrinks to [ptr_two.tau, tau_two], still single-shifted -
                // this is where the "beginning" region ends and the double-shifted middle begins.
                current_new_weight.el_prop_action(0,0) = std::exp(-p_fin_sq/(2*current_new_weight.eff_masses[0])*(tau_two - ptr->tau));
                current_new_weight.el_prop_action(1,1) = std::exp(-p_fin_sq/(2*current_new_weight.eff_masses[1])*(tau_two - ptr->tau));
                current_new_weight.el_prop_action(2,2) = std::exp(-p_fin_sq/(2*current_new_weight.eff_masses[2])*(tau_two - ptr->tau));
            }
            else {
                current_new_weight.el_prop_action(0,0) = std::exp(-p_fin_sq/(2*current_new_weight.eff_masses[0])*(ptr->tau_next - ptr->tau));
                current_new_weight.el_prop_action(1,1) = std::exp(-p_fin_sq/(2*current_new_weight.eff_masses[1])*(ptr->tau_next - ptr->tau));
                current_new_weight.el_prop_action(2,2) = std::exp(-p_fin_sq/(2*current_new_weight.eff_masses[2])*(ptr->tau_next - ptr->tau));
            }

            proposed_weights_beginning.push_back(current_new_weight);

            ptr = ptr->next;
        } while (ptr != ptr_two->next);

        // middle walk: the new creation vertex starts the double-shifted stretch running through
        // to ptr_one's own (shortened) segment - the region directly between the two new
        // vertices, where the wrap-path's two halves overlap and the phonon shift applies twice.
        // If ptr_one == ptr_two, tau_two and tau_one both fall inside the same original segment:
        // there's no interior stretch and no separate ptr_one piece at all - creation_weight IS
        // the whole (shortened) middle region, ending directly at tau_one.
        const bool unshared_segment {ptr_one != ptr_two};

        weight::ProposedVertexWeight creation_weight;
        p_fin = {ptr_two->k[0] - w_double[0], ptr_two->k[1] - w_double[1], ptr_two->k[2] - w_double[2]};
        creation_weight.k = p_fin;
        p_fin_sq = p_fin[0]*p_fin[0] + p_fin[1]*p_fin[1] + p_fin[2]*p_fin[2];

        eigensolution_wrapper = weight::LKMatrix::diagonalizeLKHamiltonian(p_fin);
        eigenvalues = {eigensolution_wrapper(0,0), eigensolution_wrapper(0,1), eigensolution_wrapper(0,2)};
        creation_weight.eff_masses = weight::LKMatrix::computeEffMassfromEigenval(eigenvalues);
        creation_weight.baseWF = eigensolution_wrapper.block<3,3>(1,0);
        creation_weight.vertex_wf_component = Coupling::LKOverlap::computeMatrix(proposed_weights_beginning.back().baseWF, creation_weight.baseWF);

        const double creation_duration_end {unshared_segment ? ptr_two->tau_next : tau_one};
        creation_weight.el_prop_action(0,0) = std::exp(-p_fin_sq/(2*creation_weight.eff_masses[0])*(creation_duration_end - tau_two));
        creation_weight.el_prop_action(1,1) = std::exp(-p_fin_sq/(2*creation_weight.eff_masses[1])*(creation_duration_end - tau_two));
        creation_weight.el_prop_action(2,2) = std::exp(-p_fin_sq/(2*creation_weight.eff_masses[2])*(creation_duration_end - tau_two));

        proposed_weights_middle.push_back(creation_weight);

        if (unshared_segment) {
            ptr = ptr_two->next;
            while (ptr != ptr_one) {
                weight::ProposedVertexWeight current_new_weight;
                p_fin = {ptr->k[0] - w_double[0], ptr->k[1] - w_double[1], ptr->k[2] - w_double[2]};
                current_new_weight.k = p_fin;
                p_fin_sq = p_fin[0]*p_fin[0] + p_fin[1]*p_fin[1] + p_fin[2]*p_fin[2];

                eigensolution_wrapper = weight::LKMatrix::diagonalizeLKHamiltonian(p_fin);
                eigenvalues = {eigensolution_wrapper(0,0), eigensolution_wrapper(0,1), eigensolution_wrapper(0,2)};
                current_new_weight.eff_masses = weight::LKMatrix::computeEffMassfromEigenval(eigenvalues);
                current_new_weight.baseWF = eigensolution_wrapper.block<3,3>(1,0);
                current_new_weight.vertex_wf_component = Coupling::LKOverlap::computeMatrix(proposed_weights_middle.back().baseWF, current_new_weight.baseWF);

                current_new_weight.el_prop_action(0,0) = std::exp(-p_fin_sq/(2*current_new_weight.eff_masses[0])*(ptr->tau_next - ptr->tau));
                current_new_weight.el_prop_action(1,1) = std::exp(-p_fin_sq/(2*current_new_weight.eff_masses[1])*(ptr->tau_next - ptr->tau));
                current_new_weight.el_prop_action(2,2) = std::exp(-p_fin_sq/(2*current_new_weight.eff_masses[2])*(ptr->tau_next - ptr->tau));

                proposed_weights_middle.push_back(current_new_weight);

                ptr = ptr->next;
            }

            // ptr_one's own segment shrinks to [ptr_one.tau, tau_one], still double-shifted - the
            // last entry of the middle region.
            weight::ProposedVertexWeight ptr_one_weight;
            p_fin = {ptr_one->k[0] - w_double[0], ptr_one->k[1] - w_double[1], ptr_one->k[2] - w_double[2]};
            ptr_one_weight.k = p_fin;
            p_fin_sq = p_fin[0]*p_fin[0] + p_fin[1]*p_fin[1] + p_fin[2]*p_fin[2];

            eigensolution_wrapper = weight::LKMatrix::diagonalizeLKHamiltonian(p_fin);
            eigenvalues = {eigensolution_wrapper(0,0), eigensolution_wrapper(0,1), eigensolution_wrapper(0,2)};
            ptr_one_weight.eff_masses = weight::LKMatrix::computeEffMassfromEigenval(eigenvalues);
            ptr_one_weight.baseWF = eigensolution_wrapper.block<3,3>(1,0);
            ptr_one_weight.vertex_wf_component = Coupling::LKOverlap::computeMatrix(proposed_weights_middle.back().baseWF, ptr_one_weight.baseWF);

            ptr_one_weight.el_prop_action(0,0) = std::exp(-p_fin_sq/(2*ptr_one_weight.eff_masses[0])*(tau_one - ptr_one->tau));
            ptr_one_weight.el_prop_action(1,1) = std::exp(-p_fin_sq/(2*ptr_one_weight.eff_masses[1])*(tau_one - ptr_one->tau));
            ptr_one_weight.el_prop_action(2,2) = std::exp(-p_fin_sq/(2*ptr_one_weight.eff_masses[2])*(tau_one - ptr_one->tau));

            proposed_weights_middle.push_back(ptr_one_weight);
        }

        // end walk: the new annihilation vertex reverts from double back to single shift
        // (removing one copy of w_proposed from ptr_one's ORIGINAL momentum, not from the
        // already-double-shifted value), then diagram_tail. Single-shifted throughout, same as
        // case 1's end walk, just anchored at ptr_one instead of ptr_two.
        weight::ProposedVertexWeight ann_weight;
        p_fin = {ptr_one->k[0] - w_proposed[0], ptr_one->k[1] - w_proposed[1], ptr_one->k[2] - w_proposed[2]};
        ann_weight.k = p_fin;
        p_fin_sq = p_fin[0]*p_fin[0] + p_fin[1]*p_fin[1] + p_fin[2]*p_fin[2];

        eigensolution_wrapper = weight::LKMatrix::diagonalizeLKHamiltonian(p_fin);
        eigenvalues = {eigensolution_wrapper(0,0), eigensolution_wrapper(0,1), eigensolution_wrapper(0,2)};
        ann_weight.eff_masses = weight::LKMatrix::computeEffMassfromEigenval(eigenvalues);
        ann_weight.baseWF = eigensolution_wrapper.block<3,3>(1,0);
        ann_weight.vertex_wf_component = Coupling::LKOverlap::computeMatrix(proposed_weights_middle.back().baseWF, ann_weight.baseWF);

        ann_weight.el_prop_action(0,0) = std::exp(-p_fin_sq/(2*ann_weight.eff_masses[0])*(ptr_one->tau_next - tau_one));
        ann_weight.el_prop_action(1,1) = std::exp(-p_fin_sq/(2*ann_weight.eff_masses[1])*(ptr_one->tau_next - tau_one));
        ann_weight.el_prop_action(2,2) = std::exp(-p_fin_sq/(2*ann_weight.eff_masses[2])*(ptr_one->tau_next - tau_one));

        proposed_weights_end.push_back(ann_weight);

        ptr = ptr_one->next;
        while (ptr != this->cfg->diagram_tail) {
            weight::ProposedVertexWeight current_new_weight;
            p_fin = {ptr->k[0] - w_proposed[0], ptr->k[1] - w_proposed[1], ptr->k[2] - w_proposed[2]};
            current_new_weight.k = p_fin;
            p_fin_sq = p_fin[0]*p_fin[0] + p_fin[1]*p_fin[1] + p_fin[2]*p_fin[2];

            eigensolution_wrapper = weight::LKMatrix::diagonalizeLKHamiltonian(p_fin);
            eigenvalues = {eigensolution_wrapper(0,0), eigensolution_wrapper(0,1), eigensolution_wrapper(0,2)};
            current_new_weight.eff_masses = weight::LKMatrix::computeEffMassfromEigenval(eigenvalues);
            current_new_weight.baseWF = eigensolution_wrapper.block<3,3>(1,0);
            current_new_weight.vertex_wf_component = Coupling::LKOverlap::computeMatrix(proposed_weights_end.back().baseWF, current_new_weight.baseWF);

            current_new_weight.el_prop_action(0,0) = std::exp(-p_fin_sq/(2*current_new_weight.eff_masses[0])*(ptr->tau_next - ptr->tau));
            current_new_weight.el_prop_action(1,1) = std::exp(-p_fin_sq/(2*current_new_weight.eff_masses[1])*(ptr->tau_next - ptr->tau));
            current_new_weight.el_prop_action(2,2) = std::exp(-p_fin_sq/(2*current_new_weight.eff_masses[2])*(ptr->tau_next - ptr->tau));

            proposed_weights_end.push_back(current_new_weight);

            ptr = ptr->next;
        }

        // trace assembly: unlike case 1, there is no untouched "middle" here - the entire diagram
        // is either single-shifted (beginning, end) or double-shifted (middle), so all three folds
        // are freshly computed straight from proposed_weights_*; no cached left_component/
        // right_component reuse (and no inversion trick) is needed, since nothing is being reused
        // unchanged from the current diagram.
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

        // computeExternalPhPropAction's wrap-convention formula (current_tau_length-tau_two+tau_one)
        // is unconditional on case: in case 2 (tau_two < tau_one) it evaluates to MORE than
        // current_tau_length, exactly reflecting the middle region being traversed twice - matches
        // the same variance used to sample w_proposed above.
        const double weight_proposed {
            (beginning_product * middle_product * end_product).trace() *
            Coupling::Strength::compute(w_proposed, ph_mode_energy, ph_mode_diel_response) *
            Coupling::Strength::compute(w_proposed, ph_mode_energy, ph_mode_diel_response) *
            std::exp(-ph_mode_energy*(cfg->current_tau_length - tau_two + tau_one))
        };
        const double weight_current {this->cfg->diagram_head->right_component.trace()};

        // same forward-proposal mechanics as case 1 (mode draw, the two exponentials, the Gaussian
        // w draw) - p_A/p_B don't depend on which case the draw landed in, so they're unchanged.
        const double p_B {1.};
        const double p_A {1.*(static_cast<double>(cfg->external_ph_manager->current_length)/2.) + 1.};

        const double numerator {
            p_B *
            weight_proposed *
            Coupling::Parameters::V_unit_cell
        };

        const double denominator {
            p_A *
            std::pow(2*std::numbers::pi, 3) *
            weight_current *
            ph_mode_energy * ph_mode_energy * std::exp(-ph_mode_energy*(cfg->current_tau_length - tau_two + tau_one)) *
            std::pow((cfg->current_tau_length - tau_two + tau_one)/(2*std::numbers::pi), 1.5) *
            std::exp(-((w_proposed[0]*w_proposed[0] + w_proposed[1]*w_proposed[1] + w_proposed[2]*w_proposed[2])/2.)*(cfg->current_tau_length - tau_two + tau_one))
        };

        return numerator/denominator;
    }
}
