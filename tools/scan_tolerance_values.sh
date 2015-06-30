#!/bin/bash

MAIN_DIR=/home/dcampora/projects/vp_efficiency
BRUNEL_VERSION=v47r9
BRUNEL_DIR=${MAIN_DIR}/Brunel_${BRUNEL_VERSION}
RESULTS_DIR=results
WORK_DIR=`pwd`

export User_release_area=${MAIN_DIR}

MINX_SLOPE=40
MAXX_SLOPE=81
MINY_SLOPE=30
MAXY_SLOPE=71
MIN_TOL=60
MAX_TOL=81

STEP=5

XSLOPE_SEARCH=${MINX_SLOPE}
while [[ ${XSLOPE_SEARCH} < ${MAXX_SLOPE} ]]; do
  YSLOPE_SEARCH=${MINY_SLOPE}

  sed -ri "s/^#define PARAM_MAXXSLOPE [0-9]\.[0-9]+f/#define PARAM_MAXXSLOPE 0.${XSLOPE_SEARCH}f/" ${BRUNEL_DIR}/Pr/PrPixelCuda/src/Definitions.cuh
  sed -ri "s/^#define PARAM_MAXXSLOPE_CANDIDATES [0-9]\.[0-9]+f/#define PARAM_MAXXSLOPE_CANDIDATES 0.${XSLOPE_SEARCH}f/" ${BRUNEL_DIR}/Pr/PrPixelCuda/src/Definitions.cuh

  while [[ ${YSLOPE_SEARCH} < ${MAXY_SLOPE} ]]; do
    TOL_SEARCH=${MIN_TOL}

    sed -ri "s/^#define PARAM_MAXYSLOPE [0-9]\.[0-9]+f/#define PARAM_MAXYSLOPE 0.${YSLOPE_SEARCH}f/" ${BRUNEL_DIR}/Pr/PrPixelCuda/src/Definitions.cuh

    while [[ ${TOL_SEARCH} < ${MAX_TOL} ]]; do

      # sed
      sed -ri "s/^#define PARAM_TOLERANCE [0-9]\.[0-9]+f/#define PARAM_TOLERANCE 0.${TOL_SEARCH}f/" ${BRUNEL_DIR}/Pr/PrPixelCuda/src/Definitions.cuh
      sed -ri "s/^#define PARAM_TOLERANCE_CANDIDATES [0-9]\.[0-9]+f/#define PARAM_TOLERANCE_CANDIDATES 0.${TOL_SEARCH}f/" ${BRUNEL_DIR}/Pr/PrPixelCuda/src/Definitions.cuh

      # delete old files (cmt should do this)
      rm -rf ${BRUNEL_DIR}/Pr/PrPixelCuda/x86_64*

      # compile
      cd ${BRUNEL_DIR}/Pr/PrPixelCuda/cmt
      cmt make
      cd ${WORK_DIR}

      # Update environment
      SetupProject Brunel ${BRUNEL_VERSION}

      # start server
      gpuserver &
      sleep 0.2

      # start client
      gpuserver --load PrPixelCudaHandler
      sleep 0.1
      gaudirun.py ${MAIN_DIR}/Brunel-Default.py > ${RESULTS_DIR}/${XSLOPE_SEARCH}_${YSLOPE_SEARCH}_${TOL_SEARCH}.out
      gpuserver --exit

      TOL_SEARCH=$((${TOL_SEARCH} + $STEP))
    done
    YSLOPE_SEARCH=$((${YSLOPE_SEARCH} + $STEP))
  done
  XSLOPE_SEARCH=$((${XSLOPE_SEARCH} + $STEP))
done
