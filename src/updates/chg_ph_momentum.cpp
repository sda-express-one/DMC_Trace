#include "updates/chg_ph_momentum.hpp"
#include <array>
#include <cassert>
#include <cmath>
#include <random>
#include "comp_method/vertex_coupling.hpp"
#include "comp_method/weight_computation.hpp"
#include "diagram/vertex.hpp"

namespace {
    // Acceptance ratio T'|g'|^2|w'|^2 q(w) / (T |g|^2|w|^2 q(w')), everything per dr dOmega. The line
    // keeps its phonon mode, so |g|^2|w|^2 (independent of w) cancels and only the trace ratio and the
    // proposal densities remain.
    double momentumRatio(const proposal::PhononMomentum & prop, double trace_proposed, double trace_current,
                         const std::array<double, 3> & w_current, const std::array<double, 3> & w_proposed, double l){
        return trace_proposed / trace_current * prop.sphericalDensity(w_current, l) / prop.sphericalDensity(w_proposed, l);
    }
}

double chg_ph_momentum::attempt(){
    const int current_order {cfg->internal_ph_manager->current_length + cfg->external_ph_manager->current_length};

    if(current_order < 2){
        return -1.;
    }

    proposed_weights.clear();
        
    std::uniform_int_distribution<int> choose_ph_vertex {0, current_order - 1};
    int chosen_vertex {choose_ph_vertex(*rng)};

    double k_new_sq {0.};
    Eigen::Matrix<double, 4, 3> eigensolution_wrapper;
    std::array<double, 3> eigenvalues {1., 1., 1.};
        
    if (chosen_vertex < cfg->internal_ph_manager->current_length) {
        branch = Branch::internal;
        ptr_one = cfg->internal_ph_manager->selectVertex(chosen_vertex);
            
        if (ptr_one->type != 1) {
            ptr_two = ptr_one;
            ptr_one = ptr_two->conj_vertex;
        }
        else {
            ptr_two = ptr_one->conj_vertex;
        }

        const double tau_one {ptr_one->tau};
        const double tau_two {ptr_two->tau};

        w_proposed = w_proposal.draw(*rng, tau_two - tau_one);

        const std::array<double, 3> w_current {ptr_one->w};

        Vertex * ptr {ptr_one};
        int i {0};

        while(ptr != ptr_two){
            weight::ProposedVertexWeight current_new_weight;
            current_new_weight.k[0] = ptr->k[0] + w_current[0] - w_proposed[0];
            current_new_weight.k[1] = ptr->k[1] + w_current[1] - w_proposed[1];
            current_new_weight.k[2] = ptr->k[2] + w_current[2] - w_proposed[2];

            k_new_sq = current_new_weight.k[0]*current_new_weight.k[0] + current_new_weight.k[1]*current_new_weight.k[1] + current_new_weight.k[2]*current_new_weight.k[2];

            eigensolution_wrapper = weight::LKMatrix::diagonalizeLKHamiltonian(current_new_weight.k);

            eigenvalues = {eigensolution_wrapper(0,0), eigensolution_wrapper(0,1), eigensolution_wrapper(0,2)};
            current_new_weight.eff_masses = weight::LKMatrix::computeEffMassfromEigenval(eigenvalues);
            current_new_weight.baseWF = eigensolution_wrapper.block<3,3>(1,0);
            
            if(ptr == ptr_one){
                current_new_weight.vertex_wf_component = Coupling::LKOverlap::computeMatrix(ptr_one->prev->baseWF, current_new_weight.baseWF);
            }
            else {
                current_new_weight.vertex_wf_component = Coupling::LKOverlap::computeMatrix(proposed_weights.back().baseWF, current_new_weight.baseWF);
            }

            current_new_weight.el_prop_action(0,0) = std::exp(-k_new_sq/(2*current_new_weight.eff_masses[0])*(ptr->tau_next - ptr->tau));
            current_new_weight.el_prop_action(1,1) = std::exp(-k_new_sq/(2*current_new_weight.eff_masses[1])*(ptr->tau_next - ptr->tau));
            current_new_weight.el_prop_action(2,2) = std::exp(-k_new_sq/(2*current_new_weight.eff_masses[2])*(ptr->tau_next - ptr->tau));

            proposed_weights.push_back(current_new_weight);
            
            ++i;
            ptr = ptr->next;
        }

        weight::ProposedVertexWeight ptr_two_weight;
        ptr_two_weight.vertex_wf_component = Coupling::LKOverlap::computeMatrix(proposed_weights.back().baseWF, ptr_two->baseWF);

        Eigen::Matrix3d new_matrix_product {ptr_two_weight.vertex_wf_component};

        proposed_weights.push_back(ptr_two_weight);

        for(int j {i-1}; j > -1; --j){
            new_matrix_product = proposed_weights[j].vertex_wf_component * proposed_weights[j].el_prop_action * new_matrix_product;
        }

        return sign.take(momentumRatio(w_proposal, (new_matrix_product * ptr_two->right_component * ptr_one->left_component).trace(), cfg->diagram_head->right_component.trace(),
                             w_current, w_proposed, tau_two - tau_one));
    }
    else {
        chosen_vertex -= cfg->internal_ph_manager->current_length;
        ptr_one = cfg->external_ph_manager->selectVertex(chosen_vertex);

        if(ptr_one->type != -2){
            ptr_two = ptr_one;
            ptr_one = ptr_two->conj_vertex;
        }
        else {
            ptr_two = ptr_one->conj_vertex;
        }
        
        const double tau_one {ptr_one->tau};
        const double tau_two {ptr_two->tau};
        const double tau_length = {cfg->diagram_tail->tau - tau_two + tau_one};

        w_proposed = w_proposal.draw(*rng, tau_length);

        const std::array<double, 3> w_current {ptr_one->w};

        Vertex * ptr {cfg->diagram_head};
        int i {0};

        if(tau_one < tau_two){
            branch = Branch::external_ann_first;
            proposed_weights_ext_second_term.clear();
            
            while(ptr != ptr_one){
                weight::ProposedVertexWeight current_weight;
                
                current_weight.k[0] = ptr->k[0] + w_current[0] - w_proposed[0];
                current_weight.k[1] = ptr->k[1] + w_current[1] - w_proposed[1];
                current_weight.k[2] = ptr->k[2] + w_current[2] - w_proposed[2];

                k_new_sq = current_weight.k[0]*current_weight.k[0] + current_weight.k[1]*current_weight.k[1] + current_weight.k[2]*current_weight.k[2];

                eigensolution_wrapper = weight::LKMatrix::diagonalizeLKHamiltonian(current_weight.k);
                eigenvalues = {eigensolution_wrapper(0,0), eigensolution_wrapper(0,1), eigensolution_wrapper(0,2)};
                current_weight.eff_masses = weight::LKMatrix::computeEffMassfromEigenval(eigenvalues);
                
                current_weight.baseWF = eigensolution_wrapper.block<3,3>(1,0);
                
                if(ptr != cfg->diagram_head) {
                    current_weight.vertex_wf_component = Coupling::LKOverlap::computeMatrix(proposed_weights.back().baseWF, current_weight.baseWF);
                }

                current_weight.el_prop_action(0,0) = std::exp(-(k_new_sq/(2*current_weight.eff_masses[0]))*(ptr->tau_next - ptr->tau));
                current_weight.el_prop_action(1,1) = std::exp(-(k_new_sq/(2*current_weight.eff_masses[1]))*(ptr->tau_next - ptr->tau));
                current_weight.el_prop_action(2,2) = std::exp(-(k_new_sq/(2*current_weight.eff_masses[2]))*(ptr->tau_next - ptr->tau));

                proposed_weights.push_back(current_weight);

                ++i;
                ptr = ptr->next;
            }

            weight::ProposedVertexWeight ptr_one_weight;
            ptr_one_weight.vertex_wf_component = Coupling::LKOverlap::computeMatrix(proposed_weights.back().baseWF, ptr_one->baseWF);
            
            Eigen::Matrix3d new_matrix_product {Eigen::Matrix3d::Identity()}; // DA RIFARE, PRENDERE DA ptr_two->prev->el_prop_actio
            
            ptr = ptr_two->prev;

            while(ptr != ptr_one){
                    new_matrix_product = ptr->vertex_wf_component * ptr->el_prop_action * new_matrix_product;
                    ptr = ptr->prev;
            }
            
            new_matrix_product = ptr_one_weight.vertex_wf_component * ptr_one->el_prop_action * new_matrix_product;

            proposed_weights.push_back(ptr_one_weight);

            for(int j {i-1}; j > -1; --j){
                new_matrix_product = proposed_weights[j].vertex_wf_component * proposed_weights[j].el_prop_action * new_matrix_product;
            }
            
            ptr = ptr_two;
            i = 0;

            while(ptr != cfg->diagram_tail){
                weight::ProposedVertexWeight current_weight;
                
                current_weight.k[0] = ptr->k[0] + w_current[0] - w_proposed[0];
                current_weight.k[1] = ptr->k[1] + w_current[1] - w_proposed[1];
                current_weight.k[2] = ptr->k[2] + w_current[2] - w_proposed[2];

                k_new_sq = current_weight.k[0]*current_weight.k[0] + current_weight.k[1]*current_weight.k[1] + current_weight.k[2]*current_weight.k[2];

                eigensolution_wrapper = weight::LKMatrix::diagonalizeLKHamiltonian(current_weight.k);
                eigenvalues = {eigensolution_wrapper(0,0), eigensolution_wrapper(0,1), eigensolution_wrapper(0,2)};
                current_weight.eff_masses = weight::LKMatrix::computeEffMassfromEigenval(eigenvalues);
                
                current_weight.baseWF = eigensolution_wrapper.block<3,3>(1,0);

                if(ptr == ptr_two){
                    current_weight.vertex_wf_component = Coupling::LKOverlap::computeMatrix(ptr_two->prev->baseWF, current_weight.baseWF);
                }
                else {
                    current_weight.vertex_wf_component = Coupling::LKOverlap::computeMatrix(proposed_weights_ext_second_term.back().baseWF, current_weight.baseWF);
                }

                current_weight.el_prop_action(0,0) = std::exp(-(k_new_sq/(2*current_weight.eff_masses[0]))*(ptr->tau_next - ptr->tau));
                current_weight.el_prop_action(1,1) = std::exp(-(k_new_sq/(2*current_weight.eff_masses[1]))*(ptr->tau_next - ptr->tau));
                current_weight.el_prop_action(2,2) = std::exp(-(k_new_sq/(2*current_weight.eff_masses[2]))*(ptr->tau_next - ptr->tau));

                proposed_weights_ext_second_term.push_back(current_weight);

                ++i;
                ptr = ptr->next;
            }

            for(int j {i-1}; j > -1; --j){
                new_matrix_product = proposed_weights_ext_second_term[j].vertex_wf_component * proposed_weights_ext_second_term[j].el_prop_action * new_matrix_product;
            }

            
            return sign.take(momentumRatio(w_proposal, new_matrix_product.trace(), cfg->diagram_head->right_component.trace(),
                                 w_current, w_proposed, tau_length));
        }
        else {
            branch = Branch::external_cre_first;

            // here the creation vertex ptr_two comes first: single shift on head … ptr_two->prev,
            // double shift on ptr_two … ptr_one->prev (traversed twice by the line), single shift
            // again on ptr_one … tail->prev
            while(ptr != ptr_two){
                weight::ProposedVertexWeight current_weight;
                
                current_weight.k[0] = ptr->k[0] + w_current[0] - w_proposed[0];
                current_weight.k[1] = ptr->k[1] + w_current[1] - w_proposed[1];
                current_weight.k[2] = ptr->k[2] + w_current[2] - w_proposed[2];

                k_new_sq = current_weight.k[0]*current_weight.k[0] + current_weight.k[1]*current_weight.k[1] + current_weight.k[2]*current_weight.k[2];

                eigensolution_wrapper = weight::LKMatrix::diagonalizeLKHamiltonian(current_weight.k);
                eigenvalues = {eigensolution_wrapper(0,0), eigensolution_wrapper(0,1), eigensolution_wrapper(0,2)};
                current_weight.eff_masses = weight::LKMatrix::computeEffMassfromEigenval(eigenvalues);
                
                current_weight.baseWF = eigensolution_wrapper.block<3,3>(1,0);
                
                if(ptr != cfg->diagram_head) {
                    current_weight.vertex_wf_component = Coupling::LKOverlap::computeMatrix(proposed_weights.back().baseWF, current_weight.baseWF);
                }

                current_weight.el_prop_action(0,0) = std::exp(-(k_new_sq/(2*current_weight.eff_masses[0]))*(ptr->tau_next - ptr->tau));
                current_weight.el_prop_action(1,1) = std::exp(-(k_new_sq/(2*current_weight.eff_masses[1]))*(ptr->tau_next - ptr->tau));
                current_weight.el_prop_action(2,2) = std::exp(-(k_new_sq/(2*current_weight.eff_masses[2]))*(ptr->tau_next - ptr->tau));

                proposed_weights.push_back(current_weight);
                
                ptr = ptr->next;
                ++i;
            }
            
            while (ptr != ptr_one) {
                weight::ProposedVertexWeight current_weight;

                current_weight.k[0] = ptr->k[0] + 2*w_current[0] - 2*w_proposed[0];
                current_weight.k[1] = ptr->k[1] + 2*w_current[1] - 2*w_proposed[1];
                current_weight.k[2] = ptr->k[2] + 2*w_current[2] - 2*w_proposed[2];

                k_new_sq = current_weight.k[0]*current_weight.k[0] + current_weight.k[1]*current_weight.k[1] + current_weight.k[2]*current_weight.k[2];

                eigensolution_wrapper = weight::LKMatrix::diagonalizeLKHamiltonian(current_weight.k);
                eigenvalues = {eigensolution_wrapper(0,0), eigensolution_wrapper(0,1), eigensolution_wrapper(0,2)};
                current_weight.eff_masses = weight::LKMatrix::computeEffMassfromEigenval(eigenvalues);
                
                current_weight.baseWF = eigensolution_wrapper.block<3,3>(1,0);
                
                current_weight.vertex_wf_component = Coupling::LKOverlap::computeMatrix(proposed_weights.back().baseWF, current_weight.baseWF);

                current_weight.el_prop_action(0,0) = std::exp(-(k_new_sq/(2*current_weight.eff_masses[0]))*(ptr->tau_next - ptr->tau));
                current_weight.el_prop_action(1,1) = std::exp(-(k_new_sq/(2*current_weight.eff_masses[1]))*(ptr->tau_next - ptr->tau));
                current_weight.el_prop_action(2,2) = std::exp(-(k_new_sq/(2*current_weight.eff_masses[2]))*(ptr->tau_next - ptr->tau));

                proposed_weights.push_back(current_weight);
                
                ptr = ptr->next;
                ++i;
            }

            while (ptr != cfg->diagram_tail) {
                weight::ProposedVertexWeight current_weight;

                current_weight.k[0] = ptr->k[0] + w_current[0] - w_proposed[0];
                current_weight.k[1] = ptr->k[1] + w_current[1] - w_proposed[1];
                current_weight.k[2] = ptr->k[2] + w_current[2] - w_proposed[2];

                k_new_sq = current_weight.k[0]*current_weight.k[0] + current_weight.k[1]*current_weight.k[1] + current_weight.k[2]*current_weight.k[2];

                eigensolution_wrapper = weight::LKMatrix::diagonalizeLKHamiltonian(current_weight.k);
                eigenvalues = {eigensolution_wrapper(0,0), eigensolution_wrapper(0,1), eigensolution_wrapper(0,2)};
                current_weight.eff_masses = weight::LKMatrix::computeEffMassfromEigenval(eigenvalues);
                
                current_weight.baseWF = eigensolution_wrapper.block<3,3>(1,0);
                
                current_weight.vertex_wf_component = Coupling::LKOverlap::computeMatrix(proposed_weights.back().baseWF, current_weight.baseWF);

                current_weight.el_prop_action(0,0) = std::exp(-(k_new_sq/(2*current_weight.eff_masses[0]))*(ptr->tau_next - ptr->tau));
                current_weight.el_prop_action(1,1) = std::exp(-(k_new_sq/(2*current_weight.eff_masses[1]))*(ptr->tau_next - ptr->tau));
                current_weight.el_prop_action(2,2) = std::exp(-(k_new_sq/(2*current_weight.eff_masses[2]))*(ptr->tau_next - ptr->tau));

                proposed_weights.push_back(current_weight);
                
                ptr = ptr->next;
                ++i;
            }

            Eigen::Matrix3d new_matrix_product {Eigen::Matrix3d::Identity()};

            for(int j {i-1}; j > -1; --j){
                new_matrix_product = proposed_weights[j].vertex_wf_component * proposed_weights[j].el_prop_action * new_matrix_product;
            }

            return sign.take(momentumRatio(w_proposal, new_matrix_product.trace(), cfg->diagram_head->right_component.trace(),
                                 w_current, w_proposed, tau_length));
        }
    }
}

void chg_ph_momentum::accept(){
    // copies one staged segment (momentum, band basis, incoming overlap, action) onto its vertex
    auto commit = [](Vertex * v, const weight::ProposedVertexWeight& staged){
        v->k = staged.k;
        v->eff_masses = staged.eff_masses;
        v->baseWF = staged.baseWF;
        v->vertex_wf_component = staged.vertex_wf_component;
        v->el_prop_action = staged.el_prop_action;
    };

    switch(branch){
    case Branch::internal: {
        assert(std::abs(ptr_one->type) % 2 == 1);
        assert(proposed_weights.size() >= 2);

        // the segments from ptr_one up to ptr_two->prev take the state staged in attempt(), in order
        Vertex * ptr {ptr_one};
        std::size_t j {0};

        while(ptr != ptr_two){
            commit(ptr, proposed_weights[j]);

            ++j;
            ptr = ptr->next;
        }

        assert(j == proposed_weights.size() - 1);

        // ptr_two's own segment is untouched; only its incoming overlap changes
        ptr_two->vertex_wf_component = proposed_weights[j].vertex_wf_component;

        ptr_one->w = w_proposed;
        ptr_two->w = w_proposed;

        // the coupling does not actually depend on the masses: only the two ends, with the new w, change
        ptr_one->vertexStrength();
        ptr_two->vertexStrength();

        // R(ptr_two) is unchanged and wf(ptr_two) is committed, so the right walk can stop there;
        // L(ptr_one) is unchanged and wf(ptr_one) is committed, so the left walk can start there.
        weight::LKMatrix::computeRightSide(cfg->diagram_head, ptr_two);
        weight::LKMatrix::computeLeftSide(cfg->diagram_tail, ptr_one);
        break;
    }
    case Branch::external_ann_first: {
        assert(ptr_one->type == -2 && ptr_two->type == 2);
        assert(ptr_one->tau < ptr_two->tau);
        assert(proposed_weights.size() >= 2);
        assert(!proposed_weights_ext_second_term.empty());

        // beginning: diagram_head up to ptr_one->prev, staged in proposed_weights[0 .. size-2]
        Vertex * ptr {cfg->diagram_head};
        std::size_t j {0};

        while(ptr != ptr_one){
            commit(ptr, proposed_weights[j]);

            ++j;
            ptr = ptr->next;
        }

        assert(j == proposed_weights.size() - 1);

        // ptr_one's own segment opens the untouched middle; only its incoming overlap changes
        ptr_one->vertex_wf_component = proposed_weights[j].vertex_wf_component;

        // end: ptr_two up to diagram_tail->prev, staged in proposed_weights_ext_second_term, the
        // first entry already carrying ptr_two's new overlap off the unchanged ptr_two->prev
        ptr = ptr_two;
        j = 0;

        while(ptr != cfg->diagram_tail){
            commit(ptr, proposed_weights_ext_second_term[j]);

            ++j;
            ptr = ptr->next;
        }

        assert(j == proposed_weights_ext_second_term.size());

        ptr_one->w = w_proposed;
        ptr_two->w = w_proposed;

        // the coupling does not actually depend on the masses: only the two ends, with the new w, change
        ptr_one->vertexStrength();
        ptr_two->vertexStrength();

        // both ends of the diagram changed, so neither side has an untouched stretch to seed from
        weight::LKMatrix::computeRightSide(cfg->diagram_head, cfg->diagram_tail);
        weight::LKMatrix::computeLeftSide(cfg->diagram_tail, cfg->diagram_head);
        break;
    }
    case Branch::external_cre_first: {
        assert(ptr_one->type == -2 && ptr_two->type == 2);
        assert(ptr_two->tau < ptr_one->tau);

        // every segment is re-based (single, double, single shift), staged in time order from
        // diagram_head to diagram_tail->prev, each with its new incoming overlap
        Vertex * ptr {cfg->diagram_head};
        std::size_t j {0};

        while(ptr != cfg->diagram_tail){
            commit(ptr, proposed_weights[j]);

            ++j;
            ptr = ptr->next;
        }

        assert(j == proposed_weights.size());

        ptr_one->w = w_proposed;
        ptr_two->w = w_proposed;

        // the coupling does not actually depend on the masses: only the two ends, with the new w, change
        ptr_one->vertexStrength();
        ptr_two->vertexStrength();

        weight::LKMatrix::computeRightSide(cfg->diagram_head, cfg->diagram_tail);
        weight::LKMatrix::computeLeftSide(cfg->diagram_tail, cfg->diagram_head);
        break;
    }
    }

    sign.accepted(cfg);
}
