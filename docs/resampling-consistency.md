# Nondecreasing latent resample sizes

The TIMSS implementation retains multinomial resampling. With N original
students it draws m_N = max(1, floor(c N)) latent observations, where the
default c is 0.2. This note explains the amendment needed to the manuscript's
strictly increasing integer-valued sampling rule. A nondecreasing rule alone
is insufficient: the resample size must diverge.

## Sufficient conditions

Assume a fixed number of finite-state latent variables, a strictly positive
true latent distribution p*, faithfulness to a DAG G*, and a Stage I estimate
with ||p_hat_N - p*|| = O_p(N^(-1/2)), after resolving the model's labeling
ambiguity. Use BDeu with fixed positive hyperparameters and an unrestricted
GES search. Let m_N be positive integers satisfying m_N -> infinity and
m_N = o(N log N). The sequence may be nondecreasing; strict monotonicity is
unnecessary. Under these conditions, GES on multinomial resamples from
p_hat_N recovers the Markov equivalence class of G* with probability tending
to one. This is an asymptotic result under model assumptions, not a guarantee
for any particular finite TIMSS graph or optimizer run.

## Direct proof avoiding inverse functions

Let p_tilde_N be the empirical distribution of the m_N resamples. Conditional
on p_hat_N, multinomial variance bounds give
||p_tilde_N - p_hat_N|| = O_p(m_N^(-1/2)). Consequently,
||p_tilde_N - p*|| = O_p(N^(-1/2) + m_N^(-1/2)). This argument does not require
an inverse of the function N -> m_N.

For an edge addition i -> j with existing parent set A, the local BDeu score
gain is m_N I_p_tilde(Z_i; Z_j | Z_A) - (Delta d/2) log(m_N) + O_p(1).
The remainder follows from the finite-state Stirling expansion: positivity
keeps all limiting local cell proportions away from zero. Fixed graph-prior
terms are also O(1).

If the conditional independence is true, conditional mutual information is
zero at p* and has zero first derivative along the probability simplex.
A Taylor expansion therefore gives I_p_tilde = O_p(1/N + 1/m_N), including
the cross term by 2ab <= a^2+b^2. The likelihood gain is O_p(1 + m_N/N).
We have m_N/(N log(m_N)) -> 0: for m_N <= N this ratio is at most
1/log(m_N); for m_N > N it is at most m_N/(N log N). The logarithmic penalty
therefore dominates the spurious likelihood gain, and the edge is rejected
with probability tending to one.

If the conditional dependence is true, I_p_tilde converges in probability to
the strictly positive I_p*. Its fit gain is m_N I_p* + o_p(m_N), which dominates
the O(log(m_N)) penalty. The edge is favored with probability tending to one.

There are finitely many local comparisons at fixed dimension, so both signs
hold simultaneously with probability tending to one. BDeu is decomposable
and score equivalent, and the local-consistency argument for GES then gives
recovery of the true equivalence class. See Chickering (2002), Sections 4.1–4.3:
https://www.jmlr.org/papers/volume3/chickering02b/chickering02b.pdf

## Manuscript amendment

Replace the strictly increasing rule with: "Let f(N) be a nondecreasing
positive integer sequence such that f(N) -> infinity and f(N) = o(N log N)."
Replace the inverse-function argument in Section 4.3 and the associated proof
with the direct two-sample-size argument above. Merely changing the adjective
while retaining an ordinary f^(-1) would leave a gap.

Every fixed c > 0 satisfies these conditions with m_N = max(1, floor(c N)),
including c = 0.2 and c = 0.5. The diagnostic m_N = floor(N log N) lies at the excluded
upper boundary and is not covered by this result. Choosing a ratio from a
fixed finite collection of admissible constants also preserves consistency,
because all those candidates recover the same true equivalence class with
probability tending to one.

Reproducibility here comes from the recorded seed, fitted
probabilities, data preprocessing, and numerical/software settings.
