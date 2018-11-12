#!/bin/sh
# to set RAINIER_PATH, execute the following in terminal, or save in .bashrc/profile
# export RAINIER_PATH=/path/to/RAINIER

cp ${RAINIER_PATH}/RAINIER.C RAINIER_copy.C
root < ${RAINIER_PATH}/input.RAINIER
