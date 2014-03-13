#!/bin/bash

BINPATH="./bin"
DATAPATH="./data/ml-100k"
# movie-len 100k data without any extra feature
# ./main -train ./data/ml-100k/ratings.train -config ./data/ml-100k/config.conf -nround 40
# ./main -predict ./data/ml-100k/ratings.test -config ./data/ml-100k/config.conf -nround 39

# ${BINPATH}/main -train ./data/ml-100k/ratings.train -cross_validation ./data/ml-100k/ratings.test -config ./data/ml-100k/config.conf -nround 100

${BINPATH}/main -train ${DATAPATH}/ratings.train -cross_validation ${DATAPATH}/ratings.test -config ${DATAPATH}/config.conf -user_feature ${DATAPATH}/extra-feature/user_feature -nround 100


# echo ${BINPATH}/main -train ${DATAPATH}/ratings.train -cross_validation ${DATAPATH}/ratings.test -config ${DATAPATH}/config.conf -user_feature ${DATAPATH}/extra-feature/user_feature -nround 100
