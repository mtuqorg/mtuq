#!/bin/bash


#
# TESTS MTUQ INSTALLATION UNDER CONDA
#

#
# path where conda will be installed
#
_CONDA_PATH_="$HOME/miniforge3"

#
# conda or mamba?
# 
_CONDA_EXE_='conda'

#
# NOTES
#
# - CONDA_EXE and CONDA_PATH are reserved by the conda distribution, so we use
#   _CONDA_EXE_ and _CONDA_PATH_ instead
#
# - it appears that conda and mamba are not completely interchangeable
#
# - for example, conda supports silent installation through `always_yes`
#   and `changeps` settings, but it appers mamba does not
#


function conda_install {
    INSTALL_PATH=$1

    hash -r
    wget -nv $(miniconda_url) -O miniforge.sh
    bash miniforge.sh -b -f -p $INSTALL_PATH
    hash -r
    rm miniforge.sh
}


function conda_update {
    $_CONDA_EXE_ config --set always_yes yes --set changeps1 no 
    $_CONDA_EXE_ update -q conda
    $_CONDA_EXE_ info -a
    $_CONDA_EXE_ config --add channels conda-forge
}


function os_string {
    echo "$(uname -ms)"
}


function miniforge_url {
case "$(os_string)" in
   "Darwin arm64")
     URL="https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-arm64.sh"
     ;;
   "Linux x86_64")
     URL="https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh"
     ;;
   *)
     # OS not supported
     URL=""
     ;;
esac
echo $URL
}


function miniconda_url {
case "$(os_string)" in
   "Darwin arm64")
     URL="https://repo.continuum.io/miniconda/Miniconda3-latest-MacOSX-arm64.sh"
     ;;
   "Linux x86_64")
     URL="https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh"
     ;;
   *)
     # OS not supported
     URL=""
     ;;
esac
echo $URL
}

echo "uname -ms: $(os_string)"


CONDA_URL=$(miniforge_url)
if [[ -z "$CONDA_URL" ]]
then
    echo "installation not supported"
    exit -1
fi


#
# mtuq root directory
#
MTUQ_PATH=$(dirname ${BASH_SOURCE[0]})/..


#
# installation tests begin now
#

echo
echo "See mtuq/tests/ for installation logs"
echo


# from now on, if anything fails, stop immediately
set -e

# verbose mode from now on
set -x

cd $MTUQ_PATH


echo "Installing latest version of conda"
echo
[ -d $_CONDA_PATH_ ] || conda_install $_CONDA_PATH_ > tests/log0
source $_CONDA_PATH_/etc/profile.d/conda.sh
conda_update $_CONDA_PATH_ >> tests/log0
echo SUCCESS
echo


echo "Testing mtuq installation"
$_CONDA_EXE_ env create -q --name env1 --file env.yaml > tests/log1
$_CONDA_EXE_ activate env1
$_CONDA_EXE_ deactivate
echo SUCCESS
echo 

