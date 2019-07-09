
#!/bin/bash
#PBS -l select=1

cd $PBS_O_WORKDIR
# activate venv
source ~/venv/pytorch/bin/activate

# execute
python main.py --train-ratio 0.7 --val-ratio 0.2 --test-ratio 0.1 data/test-for-bandgap >> output.txt

# deactivate venv
deactivate

