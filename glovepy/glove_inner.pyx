#!/usr/bin/env cython
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# coding: utf-8

import cython
import numpy as np
cimport numpy as np

from libc.math cimport exp, log, pow, sqrt, isnan

ctypedef np.float64_t REAL_t
ctypedef np.uint32_t  INT_t

cdef void train_glove_thread(
        REAL_t * W,          REAL_t * W_,
        REAL_t * gradsqW,    REAL_t * gradsqW_,
        REAL_t * bias,       REAL_t * bias_,
        REAL_t * gradsqb,    REAL_t * gradsqb_,
        REAL_t * error,
        INT_t * job_key, INT_t * job_subkey, REAL_t * job_target,
        int vector_size, int batch_size, REAL_t x_max, REAL_t alpha, REAL_t step_size) nogil:

    cdef long long b, l1, l2
    cdef int example_idx = 0
    cdef REAL_t temp1, temp2, diff, fdiff
    cdef REAL_t temp1_, temp2_, diff_, fdiff_

    for example_idx in range(batch_size):
        # Calculate cost, save diff for gradients
        l1 = job_key[example_idx]    * vector_size
        l2 = job_subkey[example_idx] * vector_size

        diff  = 0.0
        diff_ = 0.0
        for b in range(vector_size):
            diff  += W[b + l1]  * W[b + l2] # dot product of word and context word vector
            diff_ += W_[b + l1] * W_[b + l2]

        # add separate bias for each word
        diff   += bias[job_key[example_idx]]  + bias[job_subkey[example_idx]]  - log(job_target[example_idx]) 
        diff_  += bias_[job_key[example_idx]] + bias_[job_subkey[example_idx]] - log(job_target[example_idx])

        # multiply weighting function (f) with diff
        fdiff  = diff  if (job_target[example_idx] > x_max) else pow(job_target[example_idx] / x_max, alpha) * diff 
        fdiff_ = diff_ if (job_target[example_idx] > x_max) else pow(job_target[example_idx] / x_max, alpha) * diff_

        # Check for NaN in the diffs, skip updating if caught NaN in diffs
        if isnan(diff) or isnan(fdiff) or isnan(diff_) or isnan(fdiff_):
            continue

        # weighted squared error, only get the error from the first solution
        error[0] += 0.5 * fdiff * diff

        # Adaptive gradient updates
        fdiff  *= step_size    # for ease in calculating gradient
        fdiff_ *= step_size

        for b in range(vector_size):
            # learning rate times gradient for word vectors
            temp1  = fdiff  * W[b + l2]
            temp2  = fdiff  * W[b + l1]
            temp1_ = fdiff_ * W_[b + l2]
            temp2_ = fdiff_ * W_[b + l1]

            # adaptive updates
            W[b + l1]              -= (temp1  / sqrt(gradsqW[b + l1]))
            W[b + l2]              -= (temp2  / sqrt(gradsqW[b + l2]))
            W_[b + l1]             -= (temp1_ / sqrt(gradsqW_[b + l1]))
            W_[b + l2]             -= (temp2_ / sqrt(gradsqW_[b + l2]))
            gradsqW[b + l1]        += temp1  * temp1
            gradsqW[b + l2]        += temp2  * temp2
            gradsqW_[b + l1]       += temp1_ * temp1_
            gradsqW_[b + l2]       += temp2_ * temp2_

        # updates for bias terms
        bias[job_key[example_idx]]        -= fdiff  / sqrt(gradsqb[job_key[example_idx]])
        bias[job_subkey[example_idx]]     -= fdiff  / sqrt(gradsqb[job_subkey[example_idx]])
        bias_[job_key[example_idx]]       -= fdiff_ / sqrt(gradsqb_[job_key[example_idx]])
        bias_[job_subkey[example_idx]]    -= fdiff_ / sqrt(gradsqb_[job_subkey[example_idx]])

        fdiff  *= fdiff
        fdiff_ *= fdiff_
        gradsqb[job_key[example_idx]]         += fdiff
        gradsqb[job_subkey[example_idx]]      += fdiff
        gradsqb_[job_key[example_idx]]        += fdiff_
        gradsqb_[job_subkey[example_idx]]     += fdiff_


def train_glove(model, jobs, float _step_size, _error):
    cdef REAL_t *W              = <REAL_t *>(np.PyArray_DATA(model.W))
    cdef REAL_t *W_             = <REAL_t *>(np.PyArray_DATA(model.W_))
    cdef REAL_t *gradsqW        = <REAL_t *>(np.PyArray_DATA(model.gradsqW))
    cdef REAL_t *gradsqW_       = <REAL_t *>(np.PyArray_DATA(model.gradsqW_))

    cdef REAL_t *b              = <REAL_t *>(np.PyArray_DATA(model.b))
    cdef REAL_t *b_             = <REAL_t *>(np.PyArray_DATA(model.b_))
    cdef REAL_t *gradsqb        = <REAL_t *>(np.PyArray_DATA(model.gradsqb))
    cdef REAL_t *gradsqb_       = <REAL_t *>(np.PyArray_DATA(model.gradsqb_))

    cdef REAL_t *error          = <REAL_t *>(np.PyArray_DATA(_error))

    cdef INT_t  *job_key        = <INT_t  *>(np.PyArray_DATA(jobs[0]))
    cdef INT_t  *job_subkey     = <INT_t  *>(np.PyArray_DATA(jobs[1]))
    cdef REAL_t *job_target     = <REAL_t *>(np.PyArray_DATA(jobs[2]))

    # configuration and parameters
    cdef REAL_t step_size = _step_size
    cdef int vector_size  = model.d
    cdef int batch_size   = len(jobs[0])
    cdef REAL_t x_max     = model.x_max
    cdef REAL_t alpha     = model.alpha

    # release GIL & train on the sentence
    with nogil:
        train_glove_thread(
            W,\
            W_,\
            gradsqW,\
            gradsqW_,\
            b,\
            b_,\
            gradsqb,\
            gradsqb_,\
            error,\
            job_key,\
            job_subkey,\
            job_target, \
            vector_size,\
            batch_size, \
            x_max, \
            alpha, \
            step_size
        )
