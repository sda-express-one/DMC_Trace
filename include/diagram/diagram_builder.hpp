#ifndef DIAGRAM_BUILDER_HPP
#define DIAGRAM_BUILDER_HPP

#include <array>
#include <simplemc/random/xoshiro256.hpp>
#include "diagram/diagram_config.hpp"
#include "utils/diagram_sanitizer.hpp"

// Building diagrams directly, e.g. different, overdispersed initial diagrams for independent chains (an R-hat
// convergence test is only meaningful if the chains do not all start from the same order-0 diagram).
//
// Only the topology is laid out here - vertices and their times, line pairs (type, conj_vertex, manager
// slots), phonon momenta and modes, electron momenta. Every cached quantity (masses, bases, overlaps, actions,
// phonon propagators, left/right products, sign) is then rebuilt by numerical::sanitizeDiagram in finalize(),
// so the primitives may leave the cache stale in between. Typical use:
//     diagram_builder::reset(cfg, tau_D);
//     diagram_builder::addInternalLine(cfg, t_cre, t_ann, w, mode);   // ... any number of lines
//     diagram_builder::assignMomenta(cfg, p);
//     const numerical::SanitizeReport rep {diagram_builder::finalize(cfg)};   // check rep.clean()
// or, in one call, diagram_builder::randomDiagram(...).
namespace diagram_builder {

    // Back to the order-0 diagram at length tau_D: every phonon vertex is unlinked, unregistered from its
    // manager and returned to the pool; head and tail span [0, tau_D] and current_tau_length = tau_D.
    // The electron momenta are left as they are (assignMomenta sets them).
    void reset(diagram_cfg & cfg, double tau_D);

    // An internal line created at t_cre and annihilated at t_ann (0 < t_cre < t_ann < tau_D), with momentum w
    // and phonon mode `mode` (index into cfg.phonon_mode_manager). Returns false, leaving the diagram
    // unchanged, if the times are invalid or tie with an existing vertex, or the internal manager or the
    // vertex pool is full.
    bool addInternalLine(diagram_cfg & cfg, double t_cre, double t_ann, const std::array<double, 3> & w, int mode);

    // An external line (present across the imaginary-time boundary), created at t_cre and annihilated at
    // t_ann, in either time order: t_ann < t_cre is the "annihilation first" line, t_cre < t_ann the
    // "creation first" one (it wraps over the interval twice). Same failure rules as addInternalLine.
    bool addExternalLine(diagram_cfg & cfg, double t_cre, double t_ann, const std::array<double, 3> & w, int mode);

    // Electron momenta from total momentum p: the head segment carries p - sum of the external lines' w (the
    // phonons present at tau = 0), and each vertex changes it by -w (creation) or +w (annihilation) - the
    // convention the sanitizer checks.
    void assignMomenta(diagram_cfg & cfg, const std::array<double, 3> & p);

    // Cached vertex strengths of every phonon vertex (not rebuilt by the sanitizer), then the full rebuild of
    // numerical::sanitizeDiagram, which also sets cfg.current_sign. The sanitizer is run twice: the first pass
    // necessarily reports the stale cache and rebuilds it, the returned report is the second one, on the
    // rebuilt diagram. The caller checks its clean().
    numerical::SanitizeReport finalize(diagram_cfg & cfg);

    // Ranges of a random diagram. Counts are drawn uniformly in [min, max] (clamped to what the managers and
    // the vertex pool can hold). Every line is drawn the way the add updates propose one, so that a start is
    // overdispersed in its number of lines but made of physical lines (a far-out start can be a metastable trap
    // no update leaves: markdown_notes/external_line_traps.md):
    //   - mode uniform; its omega sets the span scale;
    //   - internal line: t_cre uniform in (0, tau_D), t_ann = t_cre + Exp(omega) (redrawn while beyond tau_D);
    //   - external line (as add_ext_ph): t_ann = Exp(omega), t_cre = tau_D - Exp(omega), annihilation first only
    //     (creation-first lines span more than tau_D and weigh <= e^{-omega tau_D});
    //   - momentum from the add proposal proposal::PhononMomentum for the line span l, |w| half-normal with
    //     std w_spread / sqrt(l): w_spread = 1 is exactly that proposal, w_spread = 2 four times its variance.
    // Choose the count ranges wide compared with the target (from 0 up to a few times the expected number of
    // lines, ~ alpha omega tau_D), so independent chains start on both sides of it.
    struct RandomDiagramSpec {
        int n_internal_min {0};
        int n_internal_max {0};
        int n_external_min {0};
        int n_external_max {0};
        double w_spread {1.};
    };

    // One random diagram: reset + random lines (times redrawn on a tie) + assignMomenta + finalize. Its trace
    // may be 0 - see randomDiagram.
    numerical::SanitizeReport randomDiagramOnce(diagram_cfg & cfg, simplemc::xoshiro256ss & rng, const std::array<double, 3> & p,
                                                double tau_D, const RandomDiagramSpec & spec);

    // randomDiagramOnce, redrawn (up to max_attempts times) until the rebuilt diagram is clean with a finite,
    // non-zero trace. A random diagram's weight can be too small to represent: the cached trace stores
    // band-normalised actions, and when different bands are lowest on different segments every band's product
    // can underflow to 0 - an unusable start that a chain would never visit. Returns the last report; the
    // caller should still check clean() and the trace (a spec far from the typical diagrams may never succeed).
    inline constexpr int max_attempts {1000};
    numerical::SanitizeReport randomDiagram(diagram_cfg & cfg, simplemc::xoshiro256ss & rng, const std::array<double, 3> & p,
                                            double tau_D, const RandomDiagramSpec & spec);
}

#endif // !DIAGRAM_BUILDER_HPP
