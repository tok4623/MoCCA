This is the code repository for MoCCA

# MoCCA: A Movable Circle Probability of Collision Approximation


## Run the tests
Tested with Python 3.10.12 and:
- casadi==3.7.2
- matplotlib==3.5.1
- numpy==1.26.4
- tqdm==4.67.1

To run the scripts used to generate the Paper:
```bash
cd /root/of/git/repo
python3 main.py 
# or if you want to run just one sceanrio
python3 tests/scenarioA.py # or tests/scenarioB.py
```


## Citation
if you are using this work please cite:
```bibtext
@article{kern2026moccamovablecircleprobability,
      title={MoCCA: A Movable Circle Probability of Collision Approximation}, 
      author={Tobias Kern and Christian Birkner},
      year={2026},
      eprint={2605.13125},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2605.13125}, 
}
```