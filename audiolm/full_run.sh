#!/bin/bash

python3 folderMaker.py
python3 ssTrainer.py 
python3 stTrainer.py
python3 ctTrainer.py 
python3 ftTrainer.py 
python3 inference.py