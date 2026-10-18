# Machine health with a Hidden Markov Model

Estimate an unobserved machine condition (`healthy` or `worn`) from discrete
sensor readings (`quiet`, `vibration`, `alarm`). Python 3.11+, standard library
only; no data downloads or training step.

```bash
python projects/11_hidden_markov_model/main.py
python projects/11_hidden_markov_model/main.py quiet vibration alarm
pytest -q projects/11_hidden_markov_model
```

## Model and implementation

`initial[s]` is the probability of starting in state s. `transition[i][j]`
is the probability of moving from i to j. `emission[s][o]` is the probability
of observing o in state s. Every probability row must sum to one.

The example starts healthy with probability 0.9. A healthy machine stays healthy
with probability 0.95 and emits an alarm with probability 0.02. A worn machine
stays worn with probability 0.9 and emits an alarm with probability 0.5.
These numbers are chosen for illustration, not fitted to measurements.

The forward algorithm starts with `alpha[0,s] = initial[s] * emission[s,o[0]]`.
At each subsequent reading it computes
`alpha[t,j] = emission[j,o[t]] * sum_i(alpha[t-1,i] * transition[i,j])`.
Summing the final state scores gives the observation sequence likelihood.

Viterbi replaces the sum with a maximum and stores the winning predecessor
for each state at each time. Following these backpointers in reverse recovers
the most likely complete state sequence. This differs from independently
choosing a likely state at each reading, because transitions constrain the path.

Both methods use logarithms. Products become sums; forward uses log-sum-exp
to add probabilities without numerical underflow. Zero probabilities become
negative infinity. Empty input has log likelihood zero and an empty path;
an impossible sequence has log likelihood negative infinity and no decoded path.
Unknown readings are rejected. Ties follow state declaration order.

For T readings and K states, both algorithms take O(T K²) time. Forward uses
O(K) working memory, while Viterbi stores O(T K) backpointers.

## Default run

```text
Observations: quiet -> quiet -> vibration -> alarm -> alarm
Best path:    healthy -> healthy -> worn -> worn -> worn
Sequence log likelihood: -5.599935
Best path joint log probability: -6.111979
Best path posterior probability: 0.599270
```

The last value is `P(best path, observations) / P(observations)`. Other paths
share the remaining probability. It is not the probability of machine failure
or the accuracy of a trained classifier.

## Verification and limits

Tests enumerate all eight hidden paths for every length-three observation
sequence and compare their sum and maximum against the two algorithms.
They also cover invalid models, impossible sequences, ties, empty input,
unknown observations and numerical stability on 10,000 readings.

The model assumes the next state depends only on the current state, and each
reading depends only on its current hidden state. Probabilities stay constant
through time. No parameter learning, smoothing, continuous sensor values or
held-out accuracy evaluation is implemented. A next step would be estimating
probabilities from labelled sequences and comparing log likelihood on held-out
data. Real maintenance decisions require validated models and measured data.

To explain this project, be able to derive the two recurrences, explain why
Viterbi cannot replace forward for likelihood, and show why raw probability
multiplication fails for long sequences.
