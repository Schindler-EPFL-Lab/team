# ARCO learning from demonstrations

[![Build Status](https://dev.azure.com/devsdb/CRD-NT_ARCO/_apis/build/status/arco-learning-by-demonstration)](https://dev.azure.com/devsdb/CRD-NT_ARCO/_apis/build/status/arco-learning-by-demonstration)

This package provides the python code implementation of the learning from demonstrations framework.
The learning algorithm follows the dynamic movement primitives approach, implemented with the work and 
code of A. J. Ijspeert, J. Nakanishi, H. Hoffmann, P. Pastor, and S. Schaal, "Dynamical 
movement primitives: Learning attractor models for motor behaviors," Neural Computationvol. 25, no. 2, pp. 328–373, 2013. doi: 10.1162/NECO_a_00393.


# Demonstration Data Collection

Record and play back robot trajectory demonstration.

## Record data

Data recording is done by running the `recorder.py` script. \
The data is recorded in the json format. \
To stop the recording, press the `q` keyboard button.

## Play recorded data

To play a json file, use the script `player.py`.
