#ifndef CONVERGENCE_HPP
#define CONVERGENCE_HPP

#include <vector>

// Convergence diagnostics for several Markov chains of one scalar quantity, following Vehtari, Gelman, Simpson,
// Carpenter, Buerkner, "Rank-normalization, folding, and localization: An improved R-hat for assessing convergence
// of MCMC", Bayesian Analysis (2021), arXiv:1903.08008 (literature/R_estimator_1903.08008v5.pdf):
//
//   - every chain is split in half (a middle draw is dropped for odd lengths), so a drift shared by all chains
//     shows up as a difference between first and second halves;
//   - rank normalisation (Eq. 14): each draw is replaced by z = Phi^-1((r - 3/8)/(S + 1/4)), r its rank among all
//     S pooled draws (average ranks for ties), which makes R-hat and ESS valid for heavy tails and invariant under
//     monotone transformations;
//   - split-R-hat (Eqs. 1-4) on the normal scores ("bulk"), and on the normal scores of the folded draws
//     |theta - median| ("fold", Eq. 15), which catches chains with the same centre but a different spread;
//     rhat = max(bulk, fold), to be compared with 1.01;
//   - bulk-ESS: multi-chain effective sample size of the normal scores (Eq. 10), with Geyer's initial positive and
//     monotone sequence truncation (Eqs. 12-13), capped at S log10 S; tail-ESS: the smaller ESS of the indicators
//     I(theta <= q_0.05) and I(theta <= q_0.95).
//
// The draws can be single samples or, as in this code, batch means of consecutive samples (near-independent,
// order preserved). Chains of different lengths are cut to the shortest. A quantity that is constant over all
// draws has R-hat 1 and ESS S; chains constant at different values have R-hat infinite.
namespace measurement {

    struct ConvergenceResult {
        double rhat_bulk {0.};
        double rhat_fold {0.};
        double rhat {0.};           // max(rhat_bulk, rhat_fold)
        double ess_bulk {0.};
        double ess_tail {0.};
        int chains {0};             // split chains (twice the input chains)
        int draws {0};              // draws per split chain
    };

    // M >= 1 chains of N >= 8 draws each; fewer draws give a result with every field NaN.
    ConvergenceResult diagnose(const std::vector<std::vector<double>> & chains);

    // The building blocks, exposed for testing.
    double inverseNormalCdf(double p);                                          // Phi^-1, Wichura AS 241, rel. error ~1e-16
    std::vector<std::vector<double>> splitChains(const std::vector<std::vector<double>> & chains);
    std::vector<std::vector<double>> rankNormalize(const std::vector<std::vector<double>> & chains);
    double splitRhat(const std::vector<std::vector<double>> & split);           // Eqs. 1-4 on already-split chains
    double essMean(const std::vector<std::vector<double>> & split);             // Eqs. 10-13 on already-split chains
}

#endif // !CONVERGENCE_HPP
