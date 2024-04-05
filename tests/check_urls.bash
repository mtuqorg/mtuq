#!/bin/bash


#
# Checks whether web pages still exist
#

URLS="\
    https://raw.githubusercontent.com/geodynamics/axisem/master/MANUAL/manual_axisem1.3.pdf\
    https://www.eas.slu.edu/People/LZhu/home.html\
    https://github.com/geodynamics/axisem\
    https://github.com/Liang-Ding/seisgen\
    https://ds.iris.edu/ds/products/syngine\
    https://ds.iris.edu/ds/products/syngine/#models\
    https://ds.iris.edu/files/sac-manual/manual/file_format.html\
    https://instaseis.net\
    https://docs.obspy.org/tutorial/index.html\
    https://docs.obspy.org/packages/autogen/obspy.core.stream.Stream.html\
    https://docs.obspy.org/packages/autogen/obspy.imaging.mopad_wrapper.beach.html#supported-basis-systems\
    https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html\
    https://docs.xarray.dev/en/stable/generated/xarray.DataArray.html\
    https://github.com/uafgeotools/mtuq/blob/master/docs/user_guide/05/code/gallery_mt.py
    https://github.com/uafgeotools/mtuq/blob/master/docs/user_guide/05/code/gallery_force.py
    https://conda.org/blog/2023-11-06-conda-23-10-0-release\
    https://www.anaconda.com/blog/a-faster-conda-for-a-growing-community\
    "


function check_url {
  if curl --head --silent --fail $1 &> /dev/null; then
    :
  else
    echo
    echo "This page does not exist:"
    echo $1
    echo
    return 1
  fi
}


echo
echo "Checking URLs"
echo

# for broken link, stop immediately
set -e

for url in $URLS
do
    echo $url
    check_url $url
done
echo
echo SUCCESS
echo

