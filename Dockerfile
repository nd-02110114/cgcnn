From python:3.6

SHELL ["/bin/bash", "-c"]

# install dependecies
RUN pip install jupyter numpy scipy pandas scikit-learn matplotlib pep8
RUN pip install jupyterlab pandas_profiling shap seaborn pymatgen ase

# install pytorch for CPU
RUN pip install https://download.pytorch.org/whl/cpu/torch-1.1.0-cp36-cp36m-linux_x86_64.whl
RUN pip install https://download.pytorch.org/whl/cpu/torchvision-0.3.0-cp36-cp36m-linux_x86_64.whl

# create directory for mounting local notebooks
RUN mkdir /opt/notebooks
