# Code corresponding to paper: "Backpropagation-Free Metropolis-Adjusted Langevin Algorithm"

To run any of the scripts in the library, first `pip install -e .` in the current directory. 

In `./src/fojax/sampler.py` you will find all the algorithms from the paper.

### Examples for running experiments:
* Funnel: To run the full funnel distribution grid search, in `./scripts/funnel` run `./gridsearch.sh make_fmala_step 100 10000 0.01` to run FMALA.
* Logistic Regression: In `./scripts/logreg` run `python logreg.py --sampler make_mala_step --epsilon 0.001 --num_samples 100 --subsample 2 --gpu_index 0` to run MALA.
* Regression: In `./scripts/regression` run `python sample_regression.py --epsilon 0.01 --num_samples 10000`.
* CNN: In `./scripts/cnn` run `./time.sh` to run timing experiments.

### Acknowledgements

We would like to acknowledge the valuable discussions, feedback, and resources provided by our colleagues and external collaborators throughout the process. This material is based upon work supported by the United States Air Force and DARPA under Contract No. FA8750-23-C-0519 and HR0011-24-9-0424, and the U.S. Army Research Laboratory under Cooperative Research Agreement W911NF-17-2-0196. Any opinions, findings and conclusions or recommendations expressed in this material are those of the author(s) and do not necessarily reflect the views of the United States Air Force, DARPA, the U.S. Army Research Laboratory, or the United States Government.