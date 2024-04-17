#!/bin/bash

#
# Unpacks data for tests
#

# navigate to mtuq/data
cd $(dirname ${BASH_SOURCE[0]})
wd=$PWD

for filename in \
    benchmark_cap/20090407201255351.tgz\
    benchmark_cps/greens.tgz;
do
    cd $wd
    cd $(dirname $filename)
    echo "Unpacking $filename"
    tar -xzf $(basename $filename)
done
echo "Done"
echo ""

