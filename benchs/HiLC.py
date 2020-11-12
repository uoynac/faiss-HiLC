# -*- coding: utf-8 -*-

import os
import time
import numpy as np
import sys
import faiss
from decimal import Decimal

def ivecs_read(fname):
    a = np.fromfile(fname, dtype='int32')
    d = a[0]
    return a.reshape(-1, d + 1)[:, 1:].copy()

def ivecs_mmap(fname):
    a = np.memmap(fname, dtype='int32', mode='r')
    d = a[0]
    return a.reshape(-1, d + 1)[:, 1:]

def fvecs_mmap(fname):
    return ivecs_mmap(fname).view('float32')

def bvecs_mmap(fname):
    x = np.memmap(fname, dtype='uint8', mode='r')
    d = x[:4].view('int32')[0]
    return x.reshape(-1, d + 4)[:, 4:]
    
def fvecs_read(fname):
    return ivecs_read(fname).view('float32')

def mmap_fvecs(fname):
    x = np.memmap(fname, dtype='int32', mode='r')
    d = x[0]
    return x.view('float32').reshape(-1, d + 1)[:, 1:]

def mmap_bvecs(fname):
    x = np.memmap(fname, dtype='uint8', mode='r')
    d = x[:4].view('int32')[0]
    return x.reshape(-1, d + 4)[:, 4:]

def sanitize(x):
    return np.ascontiguousarray(x.astype('float32'))

basedir = '/mnt/d/github/sift1B/'

dbsize = 1

xb = bvecs_mmap(basedir + '1milliard.p1.siftbin')
xq = bvecs_mmap(basedir + 'queries.bvecs')
xt = bvecs_mmap(basedir + 'learn.bvecs')
xb = xb[:dbsize * 1000 * 1000]
gt = ivecs_read(basedir + 'gnd/idx_%dM.ivecs' % dbsize)
xq = sanitize(xq)
xt = sanitize(xt[:50000])
nq, d = xq.shape
nb,d = xb.shape
k=100
add_bs = 10000000

opq_d = 160
opq = False
#################################
m = 4
link = 10
reorder_m = 8
ncentroids = 100
nprobe_list = [1,5]
#################################
beta_ntrain = 50000
verbose = False

if opq != True:
    opq_d = d
hnswquantizer = faiss.IndexHNSWFlat(opq_d, 16)
quantizer = faiss.IndexFlatL2(opq_d)
index_hilc = faiss.IndexHiLC(quantizer, hnswquantizer, opq_d, ncentroids, m, 8, reorder_m, 8)

index_hilc.verbose = verbose
index_hilc.link = link

if opq:
    if opq_d != d:
        opq_matrix = faiss.OPQMatrix(d, m, opq_d)
    else:
        opq_matrix = faiss.OPQMatrix(d, m)
    index = faiss.IndexPreTransform(opq_matrix, index_hilc)
    vec_transform = index.chain.at(0).apply_py
else:
    index = index_hilc
    vec_transform = lambda x:x

index.verbose = verbose
t0 = time.time()
print("training")
index.train(xt)
print("  train in %.3f s" % (time.time() - t0))



t0 = time.time()
print("adding")
if add_bs == -1:
    index.add(sanitize(xb))
else:
    for i0 in range(0, nb, add_bs):
        i1 = min(nb, i0 + add_bs)
        print("adding %d:%d / %d [%.3f s]\r" % (i0, i1, nb, time.time() - t0))
        index.add(sanitize(xb[i0:i1]))
print("  add in %.3f s" % (time.time() - t0))

def search():
    print("searching")
    for nprobe in nprobe_list:
        index_hilc.nprobe = nprobe
        print(ncentroids, nprobe, end="")
        t0 = time.time()
        D, I = index.search(xq, k)

        t1 = time.time()

        recall_at_1 = (I[:, :1] == gt[:, :1]).sum() / float(nq)
        recall_at_10a = (I[:, :10] == gt[:, :1]).sum() / float(nq) 
        recall_at_100a = (I[:, :100] == gt[:, :1]).sum() / float(nq)

        print("\t%7.3f ms per query, R1@1 %.4f, R1@10 %.4f, R1@100 %.4f" % (
            (t1 - t0) * 1000.0 / nq, recall_at_1, recall_at_10a, recall_at_100a))
search()