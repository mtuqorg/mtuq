#!/bin/bash


#
# Checks whether existing files match code generater output
#

FILENAMES="\
    ../examples/DetailedAnalysis.py\
    ../examples/GridSearch.DoubleCouple.py\
    ../examples/GridSearch.DoubleCouple+Magnitude+Depth.py\
    ../examples/GridSearch.FullMomentTensor.py\
    ../examples/SerialGridSearch.DoubleCouple.py\
    ../examples/Waveforms+Polarities.py\
    ../tests/benchmark_cap_vs_mtuq.py\
    ../tests/test_graphics.py\
    ../tests/test_grid_search_mt.py\
    ../tests/test_grid_search_mt_depth.py\
    ../tests/test_misfit.py\
    "



# navigate to mtuq/tests
cd $(dirname ${BASH_SOURCE[0]})


# if any comparison fails, stop immediately
set -e

for filename in $FILENAMES
do
    cp ${filename}{,~}
done

python ../setup/code_generator.py
for filename in $FILENAMES
do
    echo "Checking $filename..."
    cmp ${filename}{,~}
    rm ${filename}~
done
echo "SUCCESS"
echo ""

