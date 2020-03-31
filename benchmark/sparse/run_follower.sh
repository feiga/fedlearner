CUR_DIR="$(cd -P "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH=$PYTHONPATH:$CUR_DIR/../..:$CUR_DIR/../../fedlearner/trainer:$CUR_DIR/../../fedlearner/common
export PYTHONPATH=/home/gaofei.gf/fl/fedlearner
# hdfs gdpr
export INFSEC_HADDOP_ENABLED=1

# read from hdfs
export HADOOP_HOME=${HADOOP_HOME:-/opt/tiger/yarn_deploy/hadoop}
export JAVA_HOME=${JAVA_HOME:-/opt/tiger/jdk/jdk1.8}
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${HADOOP_HOME}/lib/native:${JAVA_HOME}/jre/lib/amd64/server
export CLASSPATH=`${HADOOP_HOME}/bin/hadoop classpath --glob`

ROLE=follower
HDFS_DATA_PATH=hdfs://haruna/ad_base/algo/user/luwei/fl_deep_cvr_origin_fid/${ROLE}/
STRAT_DATE=20200201
END_DATE=20200210
# MODEL_DIR=/data00/deep_cvr/model3_${ROLE}
# EXPORT_DIR=/data00/deep_cvr/model3_${ROLE}/saved_model


# read from local disk
# CUDA_VISIBLE_DEVICES="" python follower.py --ps_hosts=localhost:20001,localhost:20002 \
#                                            --tf_addr=localhost:20011 \
#                                            --local_ip=localhost:50011 \
#                                            --remote_ip=localhost:50001 \
#                                            --job_name=worker \
#                                            --task_id=0 \
#                                            --data_sdir=${LOCAL_DATA_PATH} \
#                                            --model_dir=${MODEL_DIR} \
#                                            --export_dir=${EXPORT_DIR} \
#                                            --clean=True &

# read from hdfs
CUDA_VISIBLE_DEVICES="" doas python follower.py --local-addr=localhost:33008 --peer-addr=localhost:33009 --data-path=${HDFS_DATA_PATH} --sparse-estimator True  --use-hdfs True --start-time ${STRAT_DATE} --end-time ${END_DATE} > log_single_follower.txt 2>&1  &


# run from local data
# CUDA_VISIBLE_DEVICES="" doas python follower.py --local-addr=localhost:5008 --peer-addr=localhost:5009 --data-path=/data00/deep_cvr/sparse_data/follower --sparse-estimator True   > log_follower_local.txt 2>&1  &

