# Logistic regression from scratch

I wanted to understand the gradient behind a linear classifier before relying on a library implementation. I implement stable sigmoid probabilities, binary cross-entropy and full-batch gradient descent in NumPy. I compare the result with unregularized scikit-learn logistic regression on the same synthetic split.

From the repository root, after the main setup:

```bash
python projects/07_logistic_regression/main.py
pytest projects/07_logistic_regression/tests -q
```

Example output:

```text
initial_loss: 0.6931
final_loss: 0.0289
numpy_accuracy: 0.9778
baseline_accuracy: 0.9778
```

I fit scaling on training rows only and reserve thirty percent for evaluation. The fixed-seed result exceeds my 0.90 accuracy threshold and matches the baseline. Tests check gradient finite differences, probability shapes, invalid inputs and repeated fitting. I can change `--seed 13 --iterations 1000`. The data is deliberately well separated, so this score is a correctness check rather than evidence of performance on real clinical or social data. There is no regularization or multiclass support.
