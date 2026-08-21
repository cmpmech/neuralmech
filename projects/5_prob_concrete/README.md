# Active Learning

Active learning on real data for **Chapter 5 (Probabilistic Deep Learning)**, where
the epistemic uncertainty is not just reported but used to decide what to measure
next. The running problem is the UCI concrete compressive strength dataset: 8
mixture components (cement, slag, fly ash, water, superplasticizer, coarse and fine
aggregate, age) against the strength in MPa. A Bayesian polynomial regression is fit
to 10 mixtures, one further mixture is acquired, the fit is repeated, and both
drivers plot how the error falls as the training set grows.

The model is `BayesianPolynomialRegression` from `ML.py`: a Gaussian prior on the
polynomial weights and Gaussian observation noise give a closed-form Gaussian
posterior, whose covariance is the epistemic uncertainty and whose noise level is the
aleatoric uncertainty. Both drivers use it, and differ only in how the next mixture
is chosen -- the random baseline simply never queries the covariance, in which case
the posterior mean is an ordinary regularized polynomial fit.

`config_concrete.py` defines the design space shared by both drivers: a box of
admissible component amounts, plus a bound on the total mass, since a cubic metre of
concrete weighs about 2340 kg and the box on its own would admit mixtures of several
tonnes. The dataset is cropped to that box (971 of 1030 mixtures survive), and the
mixtures probed for uncertainty are drawn inside it.

The reported error is an RMSE over the **entire** cropped dataset, acquired mixtures
included. It is therefore not a held-out test error but a measure of how well the fit
covers the design space; with at most 200 of 971 mixtures acquired, the two differ
little.

## Data generation

- `concrete_gen.py` -> `data/concrete.npz`
  reads `external_data/Concrete_Data.xls`, 1030 mixtures

## Drivers

- `active_bayesian_concrete.py` _needs `concrete.npz`_
  acquires the mixture the posterior is least certain about, and tracks both the
  error and the largest epistemic uncertainty anywhere in the design space
- `active_random_concrete.py` _needs `concrete.npz`_
  the same fit acquiring at random, as the baseline the active strategy has to beat

Both are averaged over the same 20 initial training sets, so their error curves are
directly comparable.
