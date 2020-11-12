/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c++ -*-

#include "IndexHiLC.h"

#include <iostream>

#include <cmath>
#include <cstdio>
#include <cassert>
#include <stdint.h>
#ifdef __SSE__
#include <immintrin.h>
#endif

#include <algorithm>

#include "Heap.h"
#include "utils.h"

#include "Clustering.h"
#include "IndexFlat.h"

#include "hamming.h"

#include "FaissAssert.h"

#include "AuxIndexStructures.h"

extern "C" {

/* declare BLAS functions, see http://www.netlib.org/clapack/cblas/ */

int sgemm_ (const char *transa, const char *transb, FINTEGER *m, FINTEGER *
            n, FINTEGER *k, const float *alpha, const float *a,
            FINTEGER *lda, const float *b, FINTEGER *
            ldb, float *beta, float *c, FINTEGER *ldc);

}

namespace faiss {

IndexHiLC::IndexHiLC (Index * quantizer, IndexHNSWFlat *clustering_hnsw_index, size_t d, size_t nlist,
                        size_t M, size_t nbits_per_idx,
                        size_t M_refine, size_t nbits_per_idx_refine):
    IndexIVF (quantizer, d, nlist, 0, METRIC_L2),
    pq (d, M, nbits_per_idx),
    refine_pq (d, M_refine, nbits_per_idx_refine),
    clustering_hnsw_index(clustering_hnsw_index),
    k_factor (4)
{
    FAISS_THROW_IF_NOT (nbits_per_idx <= 8);
    code_size = pq.code_size;
    invlists->code_size = code_size;
    is_trained = false;
    by_residual = true;
    use_precomputed_table = 0;
    scan_table_threshold = 0;
    ntotal1 = 0;
}

IndexHiLC::IndexHiLC ():
    k_factor (1)
{
    use_precomputed_table = 0;
    scan_table_threshold = 0;
    by_residual = true;
}

void IndexHiLC::reset()
{
    refine_codes.clear();
}


/****************************************************************
 * training                                                     */

void IndexHiLC::train (idx_t n, const float *x)
{
    train_q11 (n, x, verbose, metric_type);

    train_residual (n, x);
    is_trained = true;
}

void IndexHiLC::train_q11 (size_t n, const float *x, bool verbose, MetricType metric_type)
{
    Clustering clus (d, nlist, cp);
    clus.verbose = verbose;
    quantizer->reset();
    if (clustering_index) {
        clus.train (n, x, *clustering_index);
        quantizer->add (nlist, clus.centroids.data());
    } else {
        clus.train (n, x, *quantizer);
        clustering_hnsw_index->hnsw.efConstruction = 40;
        clustering_hnsw_index->hnsw.efSearch = 40;
        clustering_hnsw_index->verbose = verbose;
        clustering_hnsw_index->add(nlist, clus.centroids.data());
    }
}

void IndexHiLC::train_residual (idx_t n, const float *x)
{
    float * residual_2 = new float [n * d];
    ScopeDeleter <float> del(residual_2);
    train_residual_o (n, x, residual_2);
    refine_pq.cp.max_points_per_centroid = 1000;
    refine_pq.cp.verbose = verbose;
    refine_pq.train (n, residual_2);
}

void IndexHiLC::train_residual_o (idx_t n, const float *x, float *residuals_2)
{
    const float * x_in = x;

    x = fvecs_maybe_subsample (
         d, (size_t*)&n, pq.cp.max_points_per_centroid * pq.ksub,
         x, verbose, pq.cp.seed);

    ScopeDeleter<float> del_x (x_in == x ? nullptr : x);

    const float *trainset;
    ScopeDeleter<float> del_residuals;
    if (by_residual) {
        idx_t * assign = new idx_t [n];
        ScopeDeleter<idx_t> del (assign);
        clustering_hnsw_index->assign(n, x, assign);
        
        float *residuals = new float [n * d];
        del_residuals.set (residuals);
        for (idx_t i = 0; i < n; i++)
           quantizer->compute_residual (x + i * d, residuals+i*d, assign[i]);
        trainset = residuals;
    } else {
        trainset = x;
    }

    for(int i = 0; i < nlist; i++) 
        million_worlds.push_back(new IndexHNSWPQ(d, pq.M, 32));

    pegging_codes.resize(nlist);

    million_worlds[0]->train(n,trainset);

    float *sdc_table = (dynamic_cast<IndexPQ*> (million_worlds[0]->storage))->pq.sdc_table.data();
    static std::vector<float> *centroids = &((dynamic_cast<IndexPQ*> (million_worlds[0]->storage))->pq.centroids);
    
    for(int i = 0; i < nlist; i++) {
        million_worlds[i]->is_trained = million_worlds[0]->is_trained;
        (dynamic_cast<IndexPQ*> (million_worlds[i]->storage))->pq.sdc_table1 = sdc_table;
        (dynamic_cast<IndexPQ*> (million_worlds[i]->storage))->pq.centroids = *centroids;
        (dynamic_cast<IndexPQ*> (million_worlds[i]->storage))->is_trained = (dynamic_cast<IndexPQ*> (million_worlds[0]->storage))->is_trained;

        million_worlds[i]->hnsw.efConstruction = 300;
        million_worlds[i]->hnsw.efSearch = 150;
        million_worlds[i]->verbose = false;
        million_worlds[i]->hnsw.set_nb_neighbors(0, link);
    }

    pq.verbose = verbose;

    if (residuals_2) {
        uint8_t *train_codes = new uint8_t [pq.code_size * n];
        ScopeDeleter<uint8_t> del (train_codes);
        (dynamic_cast<IndexPQ*> (million_worlds[0]->storage))->pq.compute_codes (trainset, train_codes, n);

        for (idx_t i = 0; i < n; i++) {
            const float *xx = trainset + i * d;
            float * res = residuals_2 + i * d;
            (dynamic_cast<IndexPQ*> (million_worlds[0]->storage))->pq.decode (train_codes + i * pq.code_size, res);
            for (int j = 0; j < d; j++)
                res[j] = xx[j] - res[j];
        }

    }

    if (by_residual) {
        precompute_table ();
    }

}

void IndexHiLC::add_with_ids (idx_t n, const float * x, const idx_t *xids)
{
    add_core (n, x, xids, nullptr);
}

void IndexHiLC::add_core (idx_t n, const float *x, const idx_t *xids,
                                const idx_t *precomputed_idx) {
    float * residual_2 = new float [n * d];
    ScopeDeleter <float> del(residual_2);

    idx_t n0 = ntotal;
    add_core_o (n, x, xids, residual_2, precomputed_idx);
    refine_codes.resize (ntotal * refine_pq.code_size);

    refine_pq.compute_codes (
        residual_2, &refine_codes[n0 * refine_pq.code_size], n);
}

static float * compute_residuals (
        const Index *quantizer,
        Index::idx_t n, const float* x,
        const Index::idx_t *list_nos)
{
    size_t d = quantizer->d;
    float *residuals = new float [n * d];
    for (size_t i = 0; i < n; i++) {
        if (list_nos[i] < 0)
            memset (residuals + i * d, 0, sizeof(*residuals) * d);
        else
            quantizer->compute_residual (
                 x + i * d, residuals + i * d, list_nos[i]);
    }
    return residuals;
}

void IndexHiLC::encode_vectors(idx_t n, const float* x,
                                const idx_t *list_nos,
                                uint8_t * codes) const
{
    if (by_residual) {
        float *to_encode = compute_residuals (quantizer, n, x, list_nos);
        ScopeDeleter<float> del (to_encode);
        pq.compute_codes (to_encode, codes, n);
    } else {
        pq.compute_codes (x, codes, n);
    }
}

void IndexHiLC::add_core_o (idx_t n, const float * x, const idx_t *xids,
                             float *residuals_2, const idx_t *precomputed_idx)
{
    InterruptCallback::check();

    FAISS_THROW_IF_NOT (is_trained);
    ScopeDeleter<idx_t> del_idx;
    idx_t * idx0 = new idx_t [n];
    del_idx.set (idx0);
    

    if (precomputed_idx) {

    } else {
        clustering_hnsw_index->assign(n, x, idx0);
    }
    uint8_t * xcodes = new uint8_t [n * code_size];
    ScopeDeleter<uint8_t> del_xcodes (xcodes);
    int_short *pegging_code = new int_short[n];
    int_short *pegging_world = new int_short[n];
    pegging_which_world.resize(ntotal+n);
    pegging_world_codes.resize(ntotal+n);
    int prev_display = 0;

    const float *to_encode = nullptr;
    ScopeDeleter<float> del_to_encode;
    if (by_residual) {
        to_encode = compute_residuals (quantizer, n, x, idx0);
        del_to_encode.set (to_encode);
    } else {
        to_encode = x;
    }
    
    std::vector<omp_lock_t> locks(n);
    for(int i = 0; i < n; i++)
        omp_init_lock(&locks[i]);

    #pragma omp parallel for
    for(idx_t w0 = 0; w0 < n; w0++)
    {
        idx_t we = w0 + ntotal1;
        omp_set_lock(&locks[*(idx0 + w0)]);
        idx_t nt = million_worlds[*(idx0 + w0)]->storage->ntotal;
        pegging_codes[*(idx0 + w0)].push_back(we);
        pegging_code[w0] = nt;
        pegging_world[w0] = *(idx0+w0);

        million_worlds[*(idx0 + w0)] -> add(1,to_encode + w0 * d);

        if (residuals_2) {
            float *res2 = residuals_2 + w0 * d;
            const float *xi = to_encode + w0 * d;
            int which_world = *(idx0 + w0);

            const uint8_t *codes = (dynamic_cast<IndexPQ*> (million_worlds[which_world]->storage))->codes.data();
            const uint8_t *code = codes + nt * (dynamic_cast<IndexPQ*> (million_worlds[which_world]->storage))->pq.code_size;

            (dynamic_cast<IndexPQ*> (million_worlds[which_world]->storage))->pq.decode (code, res2);
            
            for (int j = 0; j < d; j++)
                res2[j] = xi[j] - res2[j];
        }
        
        omp_unset_lock(&locks[*(idx0 + w0)]);

    }
        for(int i = 0; i<n ;i++){
            idx_t we = i + ntotal1;
            pegging_which_world[we] = pegging_world[i];
            
            pegging_world_codes[we] = pegging_code[i];
    }
    for(int i = 0; i < n; i++) {
        omp_destroy_lock(&locks[i]);
    }

    size_t n_ignore = 0;
    for (size_t i = 0; i < n; i++) {
        idx_t key = idx0[i];
        if (key < 0) {
            n_ignore ++;
            if (residuals_2)
                memset (residuals_2, 0, sizeof(*residuals_2) * d);
            continue;
        }
        idx_t id = xids ? xids[i] : ntotal + i;
    }

    ntotal += n;
    ntotal1 += n;
}

size_t IndexHiLC::remove_ids(const IDSelector& /*sel*/) {
  FAISS_THROW_MSG("not implemented");
  return 0;
}

size_t IndexHiLC::precomputed_table_max_bytes = ((size_t)1) << 31;

void IndexHiLC::precompute_table ()
{
    if (use_precomputed_table == -1)
        return;

    if (use_precomputed_table == 0) {
        if (quantizer->metric_type == METRIC_INNER_PRODUCT) {
            return;
        }
        const MultiIndexQuantizer *miq =
            dynamic_cast<const MultiIndexQuantizer *> (quantizer);
        if (miq && pq.M % miq->pq.M == 0)
            use_precomputed_table = 2;
        else {
            size_t table_size = pq.M * pq.ksub * nlist * sizeof(float);
            if (table_size > precomputed_table_max_bytes) {
                if (verbose) {
                    use_precomputed_table = 0;
                }
                return;
            }
            use_precomputed_table = 1;
        }
    }

    std::vector<float> r_norms (pq.M * pq.ksub, NAN);
    for (int m = 0; m < pq.M; m++)
        for (int j = 0; j < pq.ksub; j++)
            r_norms [m * pq.ksub + j] =
                fvec_norm_L2sqr (pq.get_centroids (m, j), pq.dsub);

    if (use_precomputed_table == 1) {

        precomputed_table.resize (nlist * pq.M * pq.ksub);
        std::vector<float> centroid (d);

        for (size_t i = 0; i < nlist; i++) {
            quantizer->reconstruct (i, centroid.data());

            float *tab = &precomputed_table[i * pq.M * pq.ksub];
            pq.compute_inner_prod_table (centroid.data(), tab);
            fvec_madd (pq.M * pq.ksub, r_norms.data(), 2.0, tab, tab);
        }
    } else if (use_precomputed_table == 2) {
        const MultiIndexQuantizer *miq =
           dynamic_cast<const MultiIndexQuantizer *> (quantizer);
        FAISS_THROW_IF_NOT (miq);
        const ProductQuantizer &cpq = miq->pq;
        FAISS_THROW_IF_NOT (pq.M % cpq.M == 0);

        precomputed_table.resize(cpq.ksub * pq.M * pq.ksub);

        std::vector<float> centroids (d * cpq.ksub, NAN);

        for (int m = 0; m < cpq.M; m++) {
            for (size_t i = 0; i < cpq.ksub; i++) {
                memcpy (centroids.data() + i * d + m * cpq.dsub,
                        cpq.get_centroids (m, i),
                        sizeof (*centroids.data()) * cpq.dsub);
            }
        }

        pq.compute_inner_prod_tables (cpq.ksub, centroids.data (),
                                      precomputed_table.data ());

        for (size_t i = 0; i < cpq.ksub; i++) {
            float *tab = &precomputed_table[i * pq.M * pq.ksub];
            fvec_madd (pq.M * pq.ksub, r_norms.data(), 2.0, tab, tab);
        }

    }

}

void IndexHiLC::search (idx_t n, const float *x, idx_t k,
                         float *distances, idx_t *labels) const
{
    std::unique_ptr<idx_t[]> idx(new idx_t[n * nprobe]);
    std::unique_ptr<float[]> coarse_dis(new float[n * nprobe]);
    
    clustering_hnsw_index->search(n, x, nprobe, coarse_dis.get(), idx.get());

#pragma omp parallel
    {
#pragma omp for
        for (size_t i = 0; i < n; i++) {
            std::vector<Neighbor> search_result_set;
            for(size_t j = 0; j< nprobe;j++){
                int world_k = 100;
                idx_t idxi_temp1[world_k];
                float simi_temp1[world_k];

                int which_world = idx.get()[i * nprobe + j];
                if(which_world == -1)
                    continue;
                if(million_worlds[which_world] ->storage->ntotal < 2)
                    continue;
                
                float *residual = new float [d];
                ScopeDeleter<float> del (residual);
                quantizer->compute_residual (x + i * d, residual, which_world);

                million_worlds[which_world] -> search(1, residual, world_k, simi_temp1, idxi_temp1);

                const uint8_t *codes = (dynamic_cast<IndexPQ*> (million_worlds[which_world]->storage))->codes.data();
                size_t code_size = (dynamic_cast<IndexPQ*> (million_worlds[which_world]->storage))->pq.code_size;

                for (size_t l = 0; l < world_k; l++){

                    if(idxi_temp1[l] == -1)
                        continue;

                    if(l<10){
                        const uint8_t *code = codes + idxi_temp1[l] * code_size;

                        float *x_vector = new float[d];
                        float *y_vector = new float[d];
                        (dynamic_cast<IndexPQ*> (million_worlds[which_world]->storage))->pq.decode(code,y_vector);
                        
                        for (int l1 = 0; l1 < d; l1++)
                            y_vector[l1] = residual[l1] - y_vector[l1];
                        idx_t id =pegging_codes[which_world][idxi_temp1[l]];
                        assert (0 <= id && id < ntotal);
                        refine_pq.decode(&refine_codes [id * refine_pq.code_size], x_vector);
                        float dis = fvec_L2sqr (x_vector, y_vector, d);

                        idxi_temp1[l]=pegging_codes[which_world][idxi_temp1[l]];

                        Neighbor n1;
                        n1.id = idxi_temp1[l];
                        n1.distance = dis;
                        search_result_set.push_back(n1);
                    }else{
                        idxi_temp1[l]=pegging_codes[which_world][idxi_temp1[l]];

                        Neighbor n1;
                        n1.id = idxi_temp1[l];
                        n1.distance = simi_temp1[l];
                        search_result_set.push_back(n1);
                    }
                }
            }
            std::sort(search_result_set.begin(),search_result_set.end());
            for(idx_t j1 = 0; j1<k; j1++) {
                labels[i*k+j1] = search_result_set[j1].id;
                distances[i*k+j1] = search_result_set[j1].distance;
            }
        }
    }
}

} // namespace faiss
