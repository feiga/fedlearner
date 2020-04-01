export PYTHONPATH=/home/gaofei.gf/fl/fedlearner
export INFSEC_HADDOP_ENABLED=1

# read from hdfs
export HADOOP_HOME=${HADOOP_HOME:-/opt/tiger/yarn_deploy/hadoop}
export JAVA_HOME=${JAVA_HOME:-/opt/tiger/jdk/jdk1.8}
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${HADOOP_HOME}/lib/native:${JAVA_HOME}/jre/lib/amd64/server
export CLASSPATH=`${HADOOP_HOME}/bin/hadoop classpath --glob`

ROLE=leader
# LOCAL_DATA_PATH=/data00/dssm/data/${ROLE}
HDFS_DATA_PATH=hdfs://haruna/ad_base/algo/user/luwei/fl_deep_cvr_origin_fid/${ROLE}/
STRAT_DATE=20200201
END_DATE=20200210

doas python leader.py \
    --local-addr=localhost:33009 \
    --peer-addr=localhost:33008 \
    --data-path=${HDFS_DATA_PATH} \
    --sparse-estimator True \
    --use-hdfs True \
    --start-time ${STRAT_DATE} \
    --end-time ${END_DATE} > log_single_leader.txt 2>&1 &

