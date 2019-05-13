#!/bin/bash
if [[ "$#" -ne 2 ]]; then
    echo "Illegal number of parameters. Arguments are <suffix> <OUTPUT CSV FILE>"
    exit 1
fi
suffix=$1
filename="$suffix*.sh.o*"
outputfile=$2

columns="clusterSize,Training Size,Accuracy,Update Interval,Communication Time,Computation Time, Total time"
echo ${columns} > ${outputfile}

for file in ${filename};do
    echo $file
    #whatever you need with "$file"
    output=$(cat $file| grep "ANN")
    # Time computation
    totalTime=$(echo "${output}" | grep 'Total time' | awk '{print $3}')
    communicationTime=$(echo "${output}" | grep 'Communication Time' | awk '{print $3}')
    computationTime=$(echo "scale=3; ${totalTime} - ${communicationTime}" | bc)

    clusterSize=$(echo "${output}" | grep 'Cluster Size' | awk '{print $3}')
    trainingSize=$(echo "${output}" | grep 'Training Size' | awk '{print $3}')
    accuracy=$(echo "${output}" | grep 'Final Accuracy' | awk '{print $3}')
    updateInterval=$(echo "${output}" | grep 'Update Interval' | awk '{print $3}')

    testString=${clusterSize}','${trainingSize}','${accuracy}','${updateInterval}','${communicationTime}',
    '${computationTime}','${totalTime}
    echo ${testString}
    echo ${testString} >> ${outputfile}
done