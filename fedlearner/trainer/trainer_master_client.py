# Copyright 2020 The FedLearner Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# coding: utf-8

import os
import time
import logging
import datetime
import collections
import subprocess

from fedlearner.common import trainer_master_service_pb2 as tm_pb
from fedlearner.common import trainer_master_service_pb2_grpc as tm_grpc
from fedlearner.proxy.channel import make_insecure_channel, ChannelType
from fedlearner.common import common_pb2 as common_pb
from fedlearner.data_join.data_block_visitor import DataBlockVisitor

DataBlockInfo = collections.namedtuple('DataBlockInfo',
                                       ['block_id', 'data_path'])
ETCD_NAME = os.environ.get('ETCD_NAME', None)
ETCD_ADDR = os.environ.get('ETCD_ADDR', None)
ETCD_BASE_DIR = os.environ.get('ETCD_BASE_DIR', None)


class LocalTrainerMasterClient(object):
    def __init__(self,
                 role,
                 path,
                 files=None,
                 ext='.tfrecord',
                 start_time=None,
                 end_time=None,
                 from_data_source=False):
        self._role = role
        self._path = path
        self._block_queue = []
        self._block_map = {}
        if from_data_source:
            data_block_visitor = DataBlockVisitor(path, ETCD_NAME,
                                                  ETCD_BASE_DIR, ETCD_ADDR)
            # pylint: disable=line-too-long
            for block_id, block_item in data_block_visitor.LoadDataBlockRepByTimeFrame(
                    start_time, end_time).items():
                self._block_queue.append(block_item)
                self._block_map[block_id] = block_item
        else:
            if files is None:
                files = []
                for filename in os.listdir(path):
                    fullname = os.path.join(path, filename)
                    if not os.path.isfile(fullname):
                        continue
                    _, fileext = os.path.splitext(filename)
                    if ext and fileext != ext:
                        continue
                    files.append(filename)
            files.sort()

            for filename in files:
                block_id, _ = os.path.splitext(filename)
                fullname = os.path.join(path, filename)
                block = DataBlockInfo(block_id, fullname)
                self._block_queue.append(block)
                self._block_map[block_id] = block

    def request_data_block(self, block_id=None):
        if self._role == 'leader':
            assert block_id is None, "Must not set block_id for leader"
            if self._block_queue:
                ret = self._block_queue.pop(0)
                logging.debug('Return data block %s', ret)
                return ret
            return None

        assert block_id, "Must set block_id for follower"
        if block_id not in self._block_map:
            return None
        return self._block_map[block_id]


class TrainerMasterClient(object):
    def __init__(self, addr, role, task_id):
        self._addr = addr
        self._role = role
        self._task_id = task_id

        channel = make_insecure_channel(self._addr, ChannelType.INTERNAL)
        self._stub = tm_grpc.TrainerMasterServiceStub(channel)
        self._request = tm_pb.DataBlockRequest()
        if self._role == 'leader':
            self._request.worker_rank = self._task_id

    def request_data_block(self, block_id=None):
        if self._role == 'follower':
            assert block_id, "Must set block_id for follower"
            self._request.block_id = block_id

        while True:
            try:
                result = self._stub.RequestDataBlock(self._request)
                break
            except Exception as e:  # pylint: disable=broad-except
                logging.warning("Get data block failed: %s. " \
                                    "Retry in 1 second...",
                                e.code().name)
                time.sleep(1)

        if result.status.code == common_pb.STATUS_SUCCESS:
            logging.debug("%s:%d failed to get data block %s at %s", self._role,
                          self._task_id, result.data_block_info.block_id,
                          result.data_block_info.data_path)
            return DataBlockInfo(result.data_block_info.block_id,
                                 result.data_block_info.data_path)

        logging.error("%s:%d gets block failed with error[%s].", self._role,
                      self._task_id, result.status.error_message)
        return None

class HDFSTrainerMasterClient(object):
    def __init__(self, role, path, start = None, end = None, shuffle = False, file_ext = 'part-'):
        self._role = role
        self._block_queue = []
        self._block_map = {}
        self._file_ext = file_ext

        # iterate all days
        curr_date = datetime.datetime.strptime(start, '%Y%m%d')
        end_date = datetime.datetime.strptime(end, '%Y%m%d')
        step = datetime.timedelta(days=1)
        while curr_date <= end_date:
            curr_path = os.path.join(path, curr_date.strftime('%Y%m%d'))
            self._add_hdfs_files(curr_date.strftime('%Y%m%d'), curr_path)
            curr_date += step
        if shuffle:
            random.shuffle(self._block_queue)

    def _add_hdfs_files(self, date, path):
        hdfs = "/data00/tiger/yarn_deploy/hadoop-2.6.0-cdh5.4.4/bin/hdfs"
        command = "%s dfs -ls %s |  awk '{print $8}'" %(hdfs, path)
        p = subprocess.Popen(command,
                            shell=True,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT)
        cnt = 0
        used_file_path = []
        for line in p.stdout.readlines():
            file_path = line.strip().decode('utf8') # bytes -> str
            _, file_ext = os.path.splitext(file_path)
            # print('file_path: ', file_path)
            if self._file_ext in file_path:
                block_id = date+os.path.basename(file_path)
                block = DataBlockInfo(block_id, file_path)
                self._block_queue.append(block)
                self._block_map[block_id] = block
                cnt += 1
                used_file_path.append(file_path)
        print('used_file_path: ', used_file_path)
        logging.info("used_file_path: {0}".format(used_file_path))
        logging.info("loading {0} files for {1}".format(cnt, path))

    def request_data_block(self, block_id=None):
        if self._role == 'leader':
            assert block_id is None, "Must not set block_id for leader"
            if self._block_queue:
                ret = self._block_queue.pop(0)
                logging.debug('Return data block %s', ret)
                return ret
            return None
        elif self._role == 'follower':
            assert block_id, "Must set block_id for follower"
            if block_id not in self._block_map:
                return None
            return self._block_map[block_id]
        else:
            print('unknown role {0}'.format(self._role))
            return
