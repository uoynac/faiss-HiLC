/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c++ -*-

#ifndef FAISS_INDEX_HiLC_H
#define FAISS_INDEX_HiLC_H

#include <vector>
#include <map>

#include "IndexIVF.h"
#include "IndexPQ.h"
#include "IndexHNSW.h"
#include "VectorTransform.h"

namespace faiss {
    struct IndexHiLC;

struct IndexHiLC: IndexIVF {
    typedef __uint16_t int_short;
    std::vector<IndexHNSWPQ *> million_worlds;
    std::vector<std::vector<idx_t>> pegging_codes;
    std::vector<idx_t> pegging_which_world;
    std::vector<idx_t> pegging_world_codes;
    size_t ntotal1;
    std::vector<uint8_t> codes1;

    struct Neighbor {
        idx_t id;
        float distance;

        Neighbor() = default;
        Neighbor(idx_t id, float distance) : id{id}, distance{distance} {}

        inline bool operator < (const Neighbor &other) const {
            return distance < other.distance;
        }
    };

    bool by_residual;

    ProductQuantizer pq;
    int link;

    size_t scan_table_threshold;

    int use_precomputed_table;
    static size_t precomputed_table_max_bytes;

    ProductQuantizer refine_pq;
    std::vector <uint8_t> refine_codes;

    float k_factor;

    void reset() override;
    size_t remove_ids(const IDSelector& sel) override;
    void add_core (idx_t n, const float *x, const idx_t *xids,
                     const idx_t *precomputed_idx = nullptr);

    std::vector <float> precomputed_table;

    IndexHiLC (Index * quantizer, IndexHNSWFlat *clustering_hnsw_index,size_t d, size_t nlist,
            size_t M, size_t nbits_per_idx, size_t M_refine, size_t nbits_per_idx_refine);

    void add_with_ids(idx_t n, const float* x, const idx_t* xids = nullptr)
        override;

    faiss::IndexHNSWFlat* clustering_hnsw_index;
    void train(idx_t n, const float* x) override;

    void search (idx_t n, const float *x, idx_t k,
                float *distances, idx_t *labels) const override;

    void train_q11 (size_t n, const float *x, bool verbose,
                MetricType metric_type);

    void encode_vectors(idx_t n, const float* x,
                        const idx_t *list_nos,
                        uint8_t * codes) const override;

    void add_core_o (idx_t n, const float *x,
                     const idx_t *xids, float *residuals_2,
                     const idx_t *precomputed_idx = nullptr);

    void precompute_table ();                 
    IndexHiLC();

    void train_residual(idx_t n, const float* x) override;

    void train_residual_o (idx_t n, const float *x, float *residuals_2);

    };

} // namespace faiss


#endif
