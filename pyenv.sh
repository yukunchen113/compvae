#!/bin/bash
#source pyenv.sh run_train_ladder.py 0
export PYTHONPATH=${PYTHONPATH}:${HOME}/code/

# display
export DISPLAY=:99.0
which Xvfb
Xvfb :99 -screen 0 1024x768x24 > /dev/null 2>&1 &
sleep 3

# command
python "$@"