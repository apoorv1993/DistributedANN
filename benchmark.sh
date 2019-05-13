#!/bin/bash
# vim:set ts=8 sw=4 sts=4 et:

# Copyright (c) 2011 Serban Giuroiu
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

# ------------------------------------------------------------------------------

if [[ "$#" -ne 1 ]]; then
    echo "Illegal number of parameters. Arguments are <suffix>"
    exit 1
fi

# Include MPI runtime and compile time
source ~/.bashrc
mpic++ ann.cpp CycleTimer.cpp -o ann

processorCount=(16 4 1)
suffix=$1

echo "--------------------------------------------------------------------------------"
uptime
echo "--------------------------------------------------------------------------------"

# process for multiple nodes
for (( np=14; np>=1; np-=2 )); do
    echo "Processing np $np"
    for (( updateFrequency = 6; updateFrequency>=1; updateFrequency-=2 )); do
        echo "Processing updateFrequency $updateFrequency"
        command="./submitJob.py -a './ann -d data/train-images-idx3-ubyte -l data/train-labels-idx1-ubyte -t data/t10k-images-idx3-ubyte -o data/t10k-labels-idx1-ubyte -m 60000 -n 10000 -e 5 -f $updateFrequency' -s ${suffix}_${np}_24_${updateFrequency} -n ${np} -p 24"
        echo "Command is $command"
        eval ${command}
    done
done
# Process for single node also
for np in ${processorCount[@]}; do
    echo "Processing np $np"
    command="./submitJob.py -a './ann -d data/train-images-idx3-ubyte -l data/train-labels-idx1-ubyte -t data/t10k-images-idx3-ubyte -o data/t10k-labels-idx1-ubyte -m 60000 -n 10000 -e 5 -f 5' -s ${suffix}_1_${np}_5 -n 1 -p ${np}"
    echo "Command is $command"
    eval ${command}
done