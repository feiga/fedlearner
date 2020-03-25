# -*- encoding=utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import numpy as np

SLOT_BITS = 10
SLOT_BITS_v2 = 15

MAX_SLOTS = 2**SLOT_BITS
MAX_SLOTS_v2 = 2**SLOT_BITS_v2

FEATURE_BITS = 64 - SLOT_BITS
FEATURE_BITS_v2 = 64 - SLOT_BITS_v2 - 1


def get_max_slot(use_fid_v2):
    return MAX_SLOTS_v2 if use_fid_v2 else MAX_SLOTS


def make_fid(slot_id, hash_value):
    return int(np.int64(np.uint64((hash_value & ((1 << FEATURE_BITS) - 1)) |
                                  (slot_id << FEATURE_BITS))))


def make_fid_v2(slot_id, hash_value):
    return int(np.int64(np.uint64((hash_value & ((1 << FEATURE_BITS_v2) - 1)) |
                                  (slot_id << FEATURE_BITS_v2))))


def _compute_slot_config(unsorted_slot_config, groups=None, use_fid_v2=False):
    slot_config = sorted(unsorted_slot_config, key=lambda x: (x[3], x[1]))
    num_slots = len(slot_config)
    num_keys = len({[x[3] for x in slot_config]})

    if groups < num_keys:
        groups = num_keys
    if groups > num_slots:
        groups = num_slots

    MAX_INT = sys.maxsize // 2

    cost = [[MAX_INT for _ in  range(num_slots + 1)] for _ in range(groups + 1)]
    mark = [[-1 for _ in range(num_slots + 1)] for _ in range(groups + 1)]

    for p in range(groups + 1):
        cost[p][0] = 0
    for p in range(1, groups + 1):
        for s in range(1, num_slots + 1):
            size = slot_config[s - 1][1]
            hash_size = slot_config[s - 1][2]
            group_key = slot_config[s - 1][3]
            min_cost = MAX_INT
            for k in reversed(range(s)):
                if slot_config[k - 1 + 1][3] != group_key:
                    break
                c = cost[p - 1][k] + size * hash_size
                if c < min_cost:
                    min_cost = c
                    mark[p][s] = k
                hash_size += slot_config[k - 1][2]
            cost[p][s] = min_cost

    weight_sizes = [0] * groups
    weight_hash_sizes = [0] * groups
    weight_group_keys = [0] * groups
    slot_weight_index = [-1] * get_max_slot(use_fid_v2)
    slot_weight_offset = [0] * get_max_slot(use_fid_v2)
    p = groups
    s = num_slots
    while p > 0:
        m = mark[p][s]
        slot_id, size, hash_size, group_key = slot_config[s - 1]
        weight_sizes[p - 1] = size
        weight_group_keys[p - 1] = group_key
        for i in range(m + 1, s + 1):
            slot_id, size, hash_size, group_key = slot_config[i - 1]
            slot_weight_index[slot_id] = p - 1
            slot_weight_offset[slot_id] = weight_hash_sizes[p - 1]
            weight_hash_sizes[p - 1] += hash_size
        p = p - 1
        s = m

    slot_size = [0] * get_max_slot(use_fid_v2)
    slot_output_offset = [0] * get_max_slot(use_fid_v2)
    slot_hash_size = [0] * get_max_slot(use_fid_v2)

    offset = 0
    for id, size, hash_size, _ in unsorted_slot_config:
        slot_size[id] = size
        slot_hash_size[id] = hash_size
        slot_output_offset[id] = offset
        offset += size

    return {
        'num_groups': groups,
        'weight_sizes': weight_sizes,
        'weight_group_keys': weight_group_keys,
        'weight_hash_sizes': weight_hash_sizes,
        'slot_size': slot_size,
        'slot_weight_index': slot_weight_index,
        'slot_output_offset': slot_output_offset,
        'slot_hash_size': slot_hash_size,
        'slot_weight_offset': slot_weight_offset,
        'output_size': offset
    }
