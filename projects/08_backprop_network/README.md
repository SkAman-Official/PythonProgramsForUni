# Neural network with explicit backpropagation

I wanted to check each derivative in a small nonlinear classifier. I use a tanh hidden layer and a sigmoid output, then write the chain-rule updates directly in NumPy. The input consists of generated noisy moons, split before fitting the standard scaler.

From the repository root, after the main setup:

```bash
python projects/08_backprop_network/main.py
pytest projects/08_backprop_network/tests -q
```

Example output:

```text
initial_loss: 0.6449
final_loss: 0.0934
test_accuracy: 0.9333
```

The default run reduces training loss and exceeds my 0.85 held-out accuracy threshold. I check every parameter gradient against central finite differences on a tiny example, and test prediction shapes, invalid inputs and seeded reproducibility. I can rerun with `--seed 19 --iterations 2500`. The sixteen-unit network uses full-batch updates. I do not tune on the test split. This demonstration has no regularization, early stopping, calibration or uncertainty estimates; other seeds may produce different scores.
