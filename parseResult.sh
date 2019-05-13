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
    output=$(cat $file| grep "DOMP")
    # Only for master
    masterLibTime=$(echo "${output}" | grep 'DOMP Master Library time' | awk '{print $6}')
    masterTotalTime=$(echo "${output}" | grep 'DOMP Master Total time' | awk '{print $6}')
    masterComputationTime=$(echo "scale=3; ${masterTotalTime} - ${masterLibTime}" | bc)

    # Only Slave average
    slaveLibTime=$(echo "${output}" | grep 'DOMP Slave Library time' | awk '{print $6}')
    slaveTotalTime=$(echo "${output}" | grep 'DOMP Slave Total time' | awk '{print $6}')
    slaveComputationTime=$(echo "scale=3; ${slaveTotalTime} - ${slaveLibTime}" | bc)

    # Whole cluster average
    clusterLibTime=$(echo "${output}" | grep 'DOMP Cluster Library time' | awk '{print $6}')
    clusterTotalTime=$(echo "${output}" | grep 'DOMP Cluster Total time' | awk '{print $6}')
    clusterComputationTime=$(echo "scale=3; ${clusterTotalTime} - ${clusterLibTime}" | bc)

    clusterSize=$(echo "${output}" | grep 'DOMP Cluster Size' | awk '{print $5}')

    testString=${clusterSize}','${masterLibTime}','${masterComputationTime}','${masterTotalTime}','${slaveLibTime}',
    '${slaveComputationTime}','${slaveTotalTime}','${clusterLibTime}','${clusterComputationTime}','${clusterTotalTime}
    echo ${testString}
    echo ${testString} >> ${outputfile}
done