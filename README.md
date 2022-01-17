This is the code for the experiments in the paper [On the privacy risks of model explanations](https://dl.acm.org/doi/abs/10.1145/3461702.3462533)

## System requirements and setup
The code is written in python 3 and was originally run on a machine with two GeForce GTX 1080 Ti 
and 40 Intel(R) Xeon(R) Gold 5115 CPU @ 2.40GHz and uses tensorflow. Given the relatively small size 
of the (most) considered neural networks it is possible to run (most of) the code without GPUs, but that's not 
recommended.
Further, the experiments for the record based explanations have relatively heavy CPU usage.

### Libraries
Running the file `setup.sh` will install all necessary libraries assuming Python 3 and pip are installed. 

### Datasets
The paper uses 5 different datasets running the file `get_datasets.sh` will download the datasets 
and save them in the `data` folder.

To obtain the latent representations of the 'Dog vs. Fish' dataset additionally the file 
`ExampleBased/ExampleTargetTraining\create_latent_representations.sh` needs to be run.

## Recreating the experiments in the paper
The paper contains three very different types of attacks and the code of the attacks is separate (only some dataset utilities are shared).
We have:
- Threshold-based attacks
- Example-based attacks
- Learning-based attacks (for comparison)
The experiments can be rerun by running the respective script files and/or going through the respective notebooks. 

- Figure  1: (Threshold attacks)  Run notebooks in `01 Threshold/Figure 1`
- Figure  2: (Heuristic attacks) Run scripts in `02 Example Based/ExampleTargetTraing` afterwards go through `02 Example Based/Evaluation.ipynb` 
- Table   1: (Recovered by algorithm 1) Run scripts in `02 Example Based/ExampleTargetTraing` and `02 Example Based/Table3.sh`  afterwards go through `02 Example Based/Evaluation.ipynb` 
- Table   2: (Graph statistics) Run scripts in `02 Example Based/ExampleTargetTraing` afterwards go through `02 Example Based/Evaluation.ipynb` 
- Table   3: (Overview of datasets) Nothing to do
- Figure  3: (Other explanations)  Run notebooks in `01 Threshold/Figure 3`
- Figure  4: (Synthetic datasets) Run script `01 Threshold/03 Figure 4-6/Figure4.sh`  aterwards go through `01 Threshold/03 Figure 4-6/Evaluation.ipynb` 
- Figure  5: (Dataset generation) Go through `01 Threshold/03 Figure 4-6/Evaluation.ipynb` 
- Figure  6: (Synthetic dataset small and big) Run script `01 Threshold/03 Figure 4-6/Figure6.sh`  aterwards go through `01 Threshold/03 Figure 4-6/Evaluation.ipynb` 
- Figure  7: (Threshold vs Network) Run scripts in `03 LearningBased` first the target and then the attack training, the results can be found in the attackoutput folder in the respective `result.csv` files
- Figure  8: (Epochs) Run notebooks in `01 Threshold/Figure 8`
- Figure  9: (Lime and Smooth Grad) Run notebooks in `01 Threshold/Figure 9`
- Figure 10: (Self-revealing points) Run scripts in `02 Example Based/ExampleTargetTraing` afterwards go through `02 Example Based/Evaluation.ipynb` 
- Figure 11: (Increasing dataset size self-revealing) Run scripts in `02 Example Based/ExampleTargetTraing` afterwards go through `02 Example Based/Evaluation.ipynb` 
- Figure 12: (Influnence training points) Run scripts in `02 Example Based/ExampleTargetTraing` afterwards go through `02 Example Based/Evaluation.ipynb` 
- Table   4: (Minorities) Run scripts in `02 Example Based/ExampleTargetTraing` afterwards go through `02 Example Based/Evaluation.ipynb` 
- Figure 13: (Illustration reconstruction) Nothing to do
- Figure 14: (Baseline attacks) Run scripts in `02 Example Based/ExampleTargetTraing` afterwards go through `02 Example Based/Evaluation.ipynb` 

The notebooks also contain some code to export the results to be plotted in latex.
## Additional experiments (for Learning based)
The easiest way to modify the experiments is to start from an existing script. However, some functions 
allow for additional inputs (e.g. the neural network training parameters), so at the beginning of most files 
is a parser which specifies all possible inputs. New datasets should be incorporated\ in the `dataset.py` file,
while new network architectures can be added to the `architectures.py` file. 