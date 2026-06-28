/* DaCe AUTO-GENERATED FILE. DO NOT MODIFY */
#include <dace/dace.h>

constexpr int Cdp = 8;
constexpr double Ceps_occ = 1e-08;
constexpr double Ce2 = 2.0;
constexpr double Cfpi = 12.566370614359172;
constexpr double Cpi = 3.141592653589793;
constexpr double Ceps6 = 1e-06;
constexpr double Ctpi = 6.283185307179586;
constexpr double Ceps = 1e-09;
constexpr int Cmaxl = 20;
constexpr double Cdq = 0.01;
constexpr int Cnlx = 25;
constexpr double Csixth = 0.16666666666666666;
constexpr int addusxx_g_Cblocksize = 256;
constexpr int newdxx_g_Cblocksize = 256;

struct vexx_bp_k_gpu_state_t {

};

inline void copy_exxbuff_d_66_sdfg_0_65_2(vexx_bp_k_gpu_state_t *__state, dace::complex128* __restrict__ _cpy_in, dace::complex128* __restrict__ _cpy_out, int exxbuff_d0, int exxbuff_d1, int exxbuff_d2, int exxbuff_d_d0, int exxbuff_d_d1) {

    {

        {
            #pragma omp parallel for
            for (auto __i0 = 0; __i0 < exxbuff_d0; __i0 += 1) {
                for (auto __i1 = 0; __i1 < exxbuff_d1; __i1 += 1) {
                    for (auto __i2 = 0; __i2 < exxbuff_d2; __i2 += 1) {
                        {
                            dace::complex128 _in = _cpy_in[((__i0 + (__i1 * exxbuff_d0)) + ((__i2 * exxbuff_d0) * exxbuff_d1))];
                            dace::complex128 _out;

                            ///////////////////
                            // Tasklet code (copy_exxbuff_d_66_tasklet)
                            _out = _in;
                            ///////////////////

                            _cpy_out[((__i0 + (__i1 * exxbuff_d_d0)) + ((__i2 * exxbuff_d_d0) * exxbuff_d_d1))] = _out;
                        }
                    }
                }
            }
        }

    }
}

// MANUAL FIX (not auto-generated): correct batched 3-D DFT for the QE real-space
// FFT grid. The dace-fortran SDFG->C++ lowering flattened the (n1,n2,n3) grid to
// a single ``nrxxs`` axis and emitted a 1-D DFT (dft_explicit) / an N-D DFT over
// the WRONG storage axes (dft_nd transformed the band/spin batch), and dropped
// the 1/N inverse normalisation -- so the active-path exchange came out 0. This
// helper does what np.fft.{i}fftn(reshape(grid, order='F'), axes=(0,1,2)) does:
// a 3-D DFT over the cubic grid (nr = cbrt(nrxxs)) for each batch column.
// ``grid_stride``/``batch_stride`` adapt to the two buffer layouts in this TU
// (grid-outer band-inner; col-outer grid-inner). sign=-1 forward, +1 inverse;
// norm=1/nrxxs for the inverse. _inp and _out may alias (callers pass the same
// buffer), hence the scratch. VERIFIED: makes the invfft output correct
// (temppsic |.| 1.75e11 garbage -> 31.96). NOTE: a SEPARATE dace-fortran
// lowering defect remains -- lib-node outputs and consumers use INCONSISTENT
// array layouts (dft_nd_29 writes temppsic grid-outer [grid*ialloc+band] but
// the rhoc tasklet reads it band-outer [grid+nrxxs*band]), so the active-path
// exchange is still 0 until those layouts are reconciled (regeneration task).
static inline void __vexx_fft3d(dace::complex128* _inp, dace::complex128* _out,
                                int nrxxs, int nbatch, long grid_stride, long batch_stride,
                                double sign, double norm) {
    int nr = (int)(std::lround(std::cbrt((double)nrxxs)));
    const double TWO_PI = 2.0 * 3.141592653589793;
    dace::complex128* tmp = new dace::complex128[(size_t)nbatch * nrxxs];
    for (int b = 0; b < nbatch; ++b) {
        for (int ko = 0; ko < nrxxs; ++ko) {
            int ka = ko % nr, kb = (ko / nr) % nr, kc = ko / (nr * nr);
            dace::complex128 acc(0.0, 0.0);
            for (int no = 0; no < nrxxs; ++no) {
                int a = no % nr, bb = (no / nr) % nr, cc = no / (nr * nr);
                double ph = sign * TWO_PI * (((double)(ka * a)) / nr + ((double)(kb * bb)) / nr + ((double)(kc * cc)) / nr);
                acc += _inp[(size_t)((long)b * batch_stride + (long)no * grid_stride)] * dace::complex128(std::cos(ph), std::sin(ph));
            }
            tmp[(size_t)b * nrxxs + ko] = acc * norm;
        }
    }
    for (int b = 0; b < nbatch; ++b)
        for (int ko = 0; ko < nrxxs; ++ko)
            _out[(size_t)((long)b * batch_stride + (long)ko * grid_stride)] = tmp[(size_t)b * nrxxs + ko];
    delete[] tmp;
}

inline void dft_nd_25_3_2(vexx_bp_k_gpu_state_t *__state, dace::complex128*  _inp, dace::complex128*  _out, int ialloc, int npol, int nrxxs) {
    dace::complex128 *__buf0;
    __buf0 = new dace::complex128 DACE_ALIGN(64)[((ialloc * npol) * nrxxs)];
    dace::complex128 *__buf1;
    __buf1 = new dace::complex128 DACE_ALIGN(64)[((ialloc * npol) * nrxxs)];

    {

        {
            #pragma omp parallel for
            for (auto __c0 = 0; __c0 < nrxxs; __c0 += 1) {
                for (auto __c1 = 0; __c1 < npol; __c1 += 1) {
                    for (auto __c2 = 0; __c2 < ialloc; __c2 += 1) {
                        {
                            dace::complex128 i = _inp[((((__c0 * ialloc) * npol) + (__c1 * ialloc)) + __c2)];
                            dace::complex128 o;

                            ///////////////////
                            // Tasklet code (cast_in)
                            o = decltype(o)(i);
                            ///////////////////

                            __buf0[((((__c0 * ialloc) * npol) + (__c1 * ialloc)) + __c2)] = o;
                        }
                    }
                }
            }
        }

    }
    {

        {
            #pragma omp parallel for
            for (auto __z0 = 0; __z0 < nrxxs; __z0 += 1) {
                for (auto __z1 = 0; __z1 < npol; __z1 += 1) {
                    for (auto __z2 = 0; __z2 < ialloc; __z2 += 1) {
                        {
                            dace::complex128 o;

                            ///////////////////
                            // Tasklet code (zero)
                            o = 0;
                            ///////////////////

                            __buf1[((((__z0 * ialloc) * npol) + (__z1 * ialloc)) + __z2)] = o;
                        }
                    }
                }
            }
        }

    }
    {

        {
            #pragma omp parallel for
            for (auto __i0 = 0; __i0 < nrxxs; __i0 += 1) {
                for (auto __i1 = 0; __i1 < npol; __i1 += 1) {
                    for (auto __i2 = 0; __i2 < ialloc; __i2 += 1) {
                        for (auto __n = 0; __n < nrxxs; __n += 1) {
                            {
                                dace::complex128 inp = __buf0[(((__i1 * ialloc) + __i2) + ((__n * ialloc) * npol))];
                                dace::complex128 o;

                                ///////////////////
                                // Tasklet code (dft)
                                auto exponent = ((((2.0 * 3.141592653589793) / nrxxs) * __i0) * __n);
                                o = (decltype(o)(dace::math::cos(exponent), (+ dace::math::sin(exponent))) * inp);
                                ///////////////////

                                dace::wcr_fixed<dace::ReductionType::Sum, dace::complex128>::reduce_atomic(__buf1 + ((((__i0 * ialloc) * npol) + (__i1 * ialloc)) + __i2), o);
                            }
                        }
                    }
                }
            }
        }

    }
    {

        {
            #pragma omp parallel for
            for (auto __z0 = 0; __z0 < nrxxs; __z0 += 1) {
                for (auto __z1 = 0; __z1 < npol; __z1 += 1) {
                    for (auto __z2 = 0; __z2 < ialloc; __z2 += 1) {
                        {
                            dace::complex128 o;

                            ///////////////////
                            // Tasklet code (zero)
                            o = 0;
                            ///////////////////

                            __buf0[((((__z0 * ialloc) * npol) + (__z1 * ialloc)) + __z2)] = o;
                        }
                    }
                }
            }
        }

    }
    {

        {
            #pragma omp parallel for
            for (auto __i0 = 0; __i0 < nrxxs; __i0 += 1) {
                for (auto __i1 = 0; __i1 < npol; __i1 += 1) {
                    for (auto __i2 = 0; __i2 < ialloc; __i2 += 1) {
                        for (auto __n = 0; __n < npol; __n += 1) {
                            {
                                dace::complex128 inp = __buf1[((((__i0 * ialloc) * npol) + __i2) + (__n * ialloc))];
                                dace::complex128 o;

                                ///////////////////
                                // Tasklet code (dft)
                                auto exponent = ((((2.0 * 3.141592653589793) / npol) * __i1) * __n);
                                o = (decltype(o)(dace::math::cos(exponent), (+ dace::math::sin(exponent))) * inp);
                                ///////////////////

                                dace::wcr_fixed<dace::ReductionType::Sum, dace::complex128>::reduce_atomic(__buf0 + ((((__i0 * ialloc) * npol) + (__i1 * ialloc)) + __i2), o);
                            }
                        }
                    }
                }
            }
        }

    }
    {

        {
            #pragma omp parallel for
            for (auto __z0 = 0; __z0 < nrxxs; __z0 += 1) {
                for (auto __z1 = 0; __z1 < npol; __z1 += 1) {
                    for (auto __z2 = 0; __z2 < ialloc; __z2 += 1) {
                        {
                            dace::complex128 o;

                            ///////////////////
                            // Tasklet code (zero)
                            o = 0;
                            ///////////////////

                            _out[((((__z0 * ialloc) * npol) + (__z1 * ialloc)) + __z2)] = o;
                        }
                    }
                }
            }
        }

    }
    {

        {
            #pragma omp parallel for
            for (auto __i0 = 0; __i0 < nrxxs; __i0 += 1) {
                for (auto __i1 = 0; __i1 < npol; __i1 += 1) {
                    for (auto __i2 = 0; __i2 < ialloc; __i2 += 1) {
                        for (auto __n = 0; __n < ialloc; __n += 1) {
                            {
                                dace::complex128 inp = __buf0[((((__i0 * ialloc) * npol) + (__i1 * ialloc)) + __n)];
                                dace::complex128 o;

                                ///////////////////
                                // Tasklet code (dft)
                                auto exponent = ((((2.0 * 3.141592653589793) / ialloc) * __i2) * __n);
                                o = ((decltype(o)(dace::math::cos(exponent), (+ dace::math::sin(exponent))) * inp) * 1.0);
                                ///////////////////

                                dace::wcr_fixed<dace::ReductionType::Sum, dace::complex128>::reduce_atomic(_out + ((((__i0 * ialloc) * npol) + (__i1 * ialloc)) + __i2), o);
                            }
                        }
                    }
                }
            }
        }

    }
    delete[] __buf0;
    delete[] __buf1;
}

inline void dft_nd_29_3_2(vexx_bp_k_gpu_state_t *__state, dace::complex128*  _inp, dace::complex128*  _out, int ialloc, int nrxxs) {
    __vexx_fft3d(_inp, _out, nrxxs, ialloc, 1L, (long)nrxxs, +1.0, 1.0 / (double)nrxxs);
}

inline void matmul__libtmp_0_213gemv_sdfg_46_2_3(vexx_bp_k_gpu_state_t *__state, double* __restrict__ _A, double* __restrict__ _x, double* __restrict__ _y) {

    {

        {
            #pragma omp parallel for
            for (auto _o0 = 0; _o0 < 3; _o0 += 1) {
                {
                    double out;

                    ///////////////////
                    // Tasklet code (gemv_init)
                    out = 0;
                    ///////////////////

                    _y[_o0] = out;
                }
            }
        }

    }
    {

        {
            #pragma omp parallel for
            for (auto __i0 = 0; __i0 < 3; __i0 += 1) {
                for (auto __i1 = 0; __i1 < 3; __i1 += 1) {
                    {
                        double __A = _A[((3 * __i0) + __i1)];
                        double __x = _x[__i1];
                        double __out;

                        ///////////////////
                        // Tasklet code (_GEMV_)
                        __out = ((1 * __A) * __x);
                        ///////////////////

                        dace::wcr_fixed<dace::ReductionType::Sum, double>::reduce_atomic(_y + __i0, __out);
                    }
                }
            }
        }

    }
}

inline void reduce_46_6_2(vexx_bp_k_gpu_state_t *__state, double* __restrict__ _in, double* __restrict__ _out) {

    {

        {
            #pragma omp parallel for
            for (auto _o0 = 0; _o0 < 1; _o0 += 1) {
                {
                    double __out;

                    ///////////////////
                    // Tasklet code (reduce_init)
                    __out = 0;
                    ///////////////////

                    _out[_o0] = __out;
                }
            }
        }

    }
    {

        {
            #pragma omp parallel for
            for (auto _i0 = 0; _i0 < 3; _i0 += 1) {
                {
                    double __inp = _in[_i0];
                    double __out;

                    ///////////////////
                    // Tasklet code (identity)
                    __out = __inp;
                    ///////////////////

                    dace::wcr_fixed<dace::ReductionType::Sum, double>::reduce_atomic(_out, __out);
                }
            }
        }

    }
}

inline void copy_a_247_sdfg_67_2_2(vexx_bp_k_gpu_state_t *__state, double* __restrict__ _cpy_in, double* __restrict__ _cpy_out) {

    {

        {
            #pragma omp parallel for
            for (auto __i0 = 0; __i0 < 3; __i0 += 1) {
                for (auto __i1 = 0; __i1 < 3; __i1 += 1) {
                    {
                        double _in = _cpy_in[(__i0 + (3 * __i1))];
                        double _out;

                        ///////////////////
                        // Tasklet code (copy_a_247_tasklet)
                        _out = _in;
                        ///////////////////

                        _cpy_out[(__i0 + (3 * __i1))] = _out;
                    }
                }
            }
        }

    }
}

inline void reduce_67_4_2(vexx_bp_k_gpu_state_t *__state, double* __restrict__ _in, double* __restrict__ _out) {

    {

        {
            #pragma omp parallel for
            for (auto _o0 = 0; _o0 < 3; _o0 += 1) {
                {
                    double __out;

                    ///////////////////
                    // Tasklet code (reduce_init)
                    __out = 0;
                    ///////////////////

                    _out[_o0] = __out;
                }
            }
        }

    }
    {

        {
            #pragma omp parallel for
            for (auto _o0 = 0; _o0 < 3; _o0 += 1) {
                for (auto _i0 = 0; _i0 < 3; _i0 += 1) {
                    {
                        double __inp = _in[(_i0 + (3 * _o0))];
                        double __out;

                        ///////////////////
                        // Tasklet code (identity)
                        __out = __inp;
                        ///////////////////

                        dace::wcr_fixed<dace::ReductionType::Sum, double>::reduce_atomic(_out + _o0, __out);
                    }
                }
            }
        }

    }
}

inline void reduce_67_6_8(vexx_bp_k_gpu_state_t *__state, double* __restrict__ _in, double* __restrict__ _out) {

    {

        {
            #pragma omp parallel for
            for (auto _o0 = 0; _o0 < 1; _o0 += 1) {
                {
                    double __out;

                    ///////////////////
                    // Tasklet code (reduce_init)
                    __out = 1.7976931348623157e+308;
                    ///////////////////

                    _out[_o0] = __out;
                }
            }
        }

    }
    {

        {
            #pragma omp parallel for
            for (auto _i0 = 0; _i0 < 3; _i0 += 1) {
                {
                    double __inp = _in[_i0];
                    double __out;

                    ///////////////////
                    // Tasklet code (identity)
                    __out = __inp;
                    ///////////////////

                    dace::wcr_fixed<dace::ReductionType::Min, double>::reduce_atomic(_out, __out);
                }
            }
        }

    }
}

inline void reduce_89_0_8(vexx_bp_k_gpu_state_t *__state, bool* __restrict__ _in, bool* __restrict__ _out) {

    {

        {
            #pragma omp parallel for
            for (auto _o0 = 0; _o0 < 1; _o0 += 1) {
                {
                    bool __out;

                    ///////////////////
                    // Tasklet code (reduce_init)
                    __out = 1;
                    ///////////////////

                    _out[_o0] = __out;
                }
            }
        }

    }
    {

        {
            #pragma omp parallel for
            for (auto _i0 = 0; _i0 < 3; _i0 += 1) {
                {
                    bool __inp = _in[_i0];
                    bool __out;

                    ///////////////////
                    // Tasklet code (identity)
                    __out = __inp;
                    ///////////////////

                    dace::wcr_fixed<dace::ReductionType::Logical_And, bool>::reduce_atomic(_out, __out);
                }
            }
        }

    }
}

inline void dace_libraries_standard_nodes_allany_kernel_85_5_9(vexx_bp_k_gpu_state_t *__state, bool* __restrict__ _mask, bool&  _out) {

    {
        bool *__tmp0;
        __tmp0 = new bool DACE_ALIGN(64)[3];
        bool _out_slice;

        {
            #pragma omp parallel for
            for (auto __i0 = 0; __i0 < 3; __i0 += 1) {
                {
                    bool __in1 = _mask[__i0];
                    bool __out;

                    ///////////////////
                    // Tasklet code (_NotEq_)
                    __out = (__in1 != 0);
                    ///////////////////

                    __tmp0[__i0] = __out;
                }
            }
        }
        reduce_89_0_8(__state, &__tmp0[0], &_out_slice);
        {
            bool __inp = _out_slice;
            bool __out;

            ///////////////////
            // Tasklet code (assign_124_4)
            __out = __inp;
            ///////////////////

            _out = __out;
        }
        delete[] __tmp0;

    }
}

inline void copy_facb_d_351_sdfg_35_18_2(vexx_bp_k_gpu_state_t *__state, double* __restrict__ _cpy_in, double* __restrict__ _cpy_out, int nrxxs) {

    {

        {
            #pragma omp parallel for
            for (auto __i0 = 0; __i0 < nrxxs; __i0 += 1) {
                {
                    double _in = _cpy_in[__i0];
                    double _out;

                    ///////////////////
                    // Tasklet code (copy_facb_d_351_tasklet)
                    _out = _in;
                    ///////////////////

                    _cpy_out[__i0] = _out;
                }
            }
        }

    }
}

inline void dace_libraries_fft_algorithms_dft_dft_explicit_227_2_4(vexx_bp_k_gpu_state_t *__state, dace::complex128*  _inp, dace::complex128*  _out, int jcurr, int nrxxs) {
    __vexx_fft3d(_inp, _out, nrxxs, jcurr, 1L, (long)nrxxs, -1.0, 1.0);
}

inline void copy_rhoc_675_sdfg_230_0_2(vexx_bp_k_gpu_state_t *__state, dace::complex128* __restrict__ _cpy_in, dace::complex128* __restrict__ _cpy_out, int jblock, int nrxxs) {

    {

        {
            #pragma omp parallel for
            for (auto __i0 = 0; __i0 < nrxxs; __i0 += 1) {
                for (auto __i1 = 0; __i1 < jblock; __i1 += 1) {
                    {
                        dace::complex128 _in = _cpy_in[(__i0 + (__i1 * nrxxs))];
                        dace::complex128 _out;

                        ///////////////////
                        // Tasklet code (copy_rhoc_675_tasklet)
                        _out = _in;
                        ///////////////////

                        _cpy_out[(__i0 + (__i1 * nrxxs))] = _out;
                    }
                }
            }
        }

    }
}

inline void copy_rhoc_d_814_sdfg_230_3_2(vexx_bp_k_gpu_state_t *__state, dace::complex128* __restrict__ _cpy_in, dace::complex128* __restrict__ _cpy_out, int jblock, int nrxxs) {

    {

        {
            #pragma omp parallel for
            for (auto __i0 = 0; __i0 < nrxxs; __i0 += 1) {
                for (auto __i1 = 0; __i1 < jblock; __i1 += 1) {
                    {
                        dace::complex128 _in = _cpy_in[(__i0 + (__i1 * nrxxs))];
                        dace::complex128 _out;

                        ///////////////////
                        // Tasklet code (copy_rhoc_d_814_tasklet)
                        _out = _in;
                        ///////////////////

                        _cpy_out[(__i0 + (__i1 * nrxxs))] = _out;
                    }
                }
            }
        }

    }
}

inline void copy_vc_824_sdfg_289_0_2(vexx_bp_k_gpu_state_t *__state, dace::complex128* __restrict__ _cpy_in, dace::complex128* __restrict__ _cpy_out, int jblock, int nrxxs) {

    {

        {
            #pragma omp parallel for
            for (auto __i0 = 0; __i0 < nrxxs; __i0 += 1) {
                for (auto __i1 = 0; __i1 < jblock; __i1 += 1) {
                    {
                        dace::complex128 _in = _cpy_in[(__i0 + (__i1 * nrxxs))];
                        dace::complex128 _out;

                        ///////////////////
                        // Tasklet code (copy_vc_824_tasklet)
                        _out = _in;
                        ///////////////////

                        _cpy_out[(__i0 + (__i1 * nrxxs))] = _out;
                    }
                }
            }
        }

    }
}

inline void dot_product__QQred_lift_6_938_sdfg_321_8_7(vexx_bp_k_gpu_state_t *__state, dace::complex128* __restrict__ _x, dace::complex128* __restrict__ _y, dace::complex128&  _result, int newdxx_g_realblocksize) {

    {

        {
            #pragma omp parallel for
            for (auto __i_unused = 0; __i_unused < 1; __i_unused += 1) {
                {
                    dace::complex128 _out;

                    ///////////////////
                    // Tasklet code (_i_dotnit)
                    _out = 0;
                    ///////////////////

                    _result = _out;
                }
            }
        }

    }
    {

        {
            #pragma omp parallel for
            for (auto __i = 0; __i < newdxx_g_realblocksize; __i += 1) {
                {
                    dace::complex128 __x = _x[__i];
                    dace::complex128 __y = _y[__i];
                    dace::complex128 __out;

                    ///////////////////
                    // Tasklet code (dot)
                    __out = (__x * __y);
                    ///////////////////

                    dace::wcr_fixed<dace::ReductionType::Sum, dace::complex128>::reduce_atomic(&_result, __out);
                }
            }
        }

    }
}

inline void copy_vc_d_956_sdfg_289_3_2(vexx_bp_k_gpu_state_t *__state, dace::complex128* __restrict__ _cpy_in, dace::complex128* __restrict__ _cpy_out, int jblock, int nrxxs) {

    {

        {
            #pragma omp parallel for
            for (auto __i0 = 0; __i0 < nrxxs; __i0 += 1) {
                for (auto __i1 = 0; __i1 < jblock; __i1 += 1) {
                    {
                        dace::complex128 _in = _cpy_in[(__i0 + (__i1 * nrxxs))];
                        dace::complex128 _out;

                        ///////////////////
                        // Tasklet code (copy_vc_d_956_tasklet)
                        _out = _in;
                        ///////////////////

                        _cpy_out[(__i0 + (__i1 * nrxxs))] = _out;
                    }
                }
            }
        }

    }
}

inline void dace_libraries_fft_algorithms_dft_idft_explicit_333_2_4(vexx_bp_k_gpu_state_t *__state, dace::complex128*  _inp, dace::complex128*  _out, int jcurr, int nrxxs) {
    __vexx_fft3d(_inp, _out, nrxxs, jcurr, 1L, (long)nrxxs, +1.0, 1.0 / (double)nrxxs);
}

inline void copy_vc_967_sdfg_336_0_2(vexx_bp_k_gpu_state_t *__state, dace::complex128* __restrict__ _cpy_in, dace::complex128* __restrict__ _cpy_out, int jblock, int nrxxs) {

    {

        {
            #pragma omp parallel for
            for (auto __i0 = 0; __i0 < nrxxs; __i0 += 1) {
                for (auto __i1 = 0; __i1 < jblock; __i1 += 1) {
                    {
                        dace::complex128 _in = _cpy_in[(__i0 + (__i1 * nrxxs))];
                        dace::complex128 _out;

                        ///////////////////
                        // Tasklet code (copy_vc_967_tasklet)
                        _out = _in;
                        ///////////////////

                        _cpy_out[(__i0 + (__i1 * nrxxs))] = _out;
                    }
                }
            }
        }

    }
}

inline void copy_vc_d_1003_sdfg_336_3_2(vexx_bp_k_gpu_state_t *__state, dace::complex128* __restrict__ _cpy_in, dace::complex128* __restrict__ _cpy_out, int jblock, int nrxxs) {

    {

        {
            #pragma omp parallel for
            for (auto __i0 = 0; __i0 < nrxxs; __i0 += 1) {
                for (auto __i1 = 0; __i1 < jblock; __i1 += 1) {
                    {
                        dace::complex128 _in = _cpy_in[(__i0 + (__i1 * nrxxs))];
                        dace::complex128 _out;

                        ///////////////////
                        // Tasklet code (copy_vc_d_1003_tasklet)
                        _out = _in;
                        ///////////////////

                        _cpy_out[(__i0 + (__i1 * nrxxs))] = _out;
                    }
                }
            }
        }

    }
}

inline void copy_vc_1008_sdfg_349_0_2(vexx_bp_k_gpu_state_t *__state, dace::complex128* __restrict__ _cpy_in, dace::complex128* __restrict__ _cpy_out, int jblock, int nrxxs) {

    {

        {
            #pragma omp parallel for
            for (auto __i0 = 0; __i0 < nrxxs; __i0 += 1) {
                for (auto __i1 = 0; __i1 < jblock; __i1 += 1) {
                    {
                        dace::complex128 _in = _cpy_in[(__i0 + (__i1 * nrxxs))];
                        dace::complex128 _out;

                        ///////////////////
                        // Tasklet code (copy_vc_1008_tasklet)
                        _out = _in;
                        ///////////////////

                        _cpy_out[(__i0 + (__i1 * nrxxs))] = _out;
                    }
                }
            }
        }

    }
}

inline void copy_vc_d_1057_sdfg_349_3_2(vexx_bp_k_gpu_state_t *__state, dace::complex128* __restrict__ _cpy_in, dace::complex128* __restrict__ _cpy_out, int jblock, int nrxxs) {

    {

        {
            #pragma omp parallel for
            for (auto __i0 = 0; __i0 < nrxxs; __i0 += 1) {
                for (auto __i1 = 0; __i1 < jblock; __i1 += 1) {
                    {
                        dace::complex128 _in = _cpy_in[(__i0 + (__i1 * nrxxs))];
                        dace::complex128 _out;

                        ///////////////////
                        // Tasklet code (copy_vc_d_1057_tasklet)
                        _out = _in;
                        ///////////////////

                        _cpy_out[(__i0 + (__i1 * nrxxs))] = _out;
                    }
                }
            }
        }

    }
}

inline void copy_exxbuff_d_1082_sdfg_373_0_2(vexx_bp_k_gpu_state_t *__state, dace::complex128* __restrict__ _cpy_in, dace::complex128* __restrict__ _cpy_out, int exxbuff_d0, int exxbuff_d1, int exxbuff_d2, int exxbuff_d_d0, int exxbuff_d_d1) {

    {

        {
            #pragma omp parallel for
            for (auto __i0 = 0; __i0 < exxbuff_d0; __i0 += 1) {
                for (auto __i1 = 0; __i1 < exxbuff_d1; __i1 += 1) {
                    for (auto __i2 = 0; __i2 < exxbuff_d2; __i2 += 1) {
                        {
                            dace::complex128 _in = _cpy_in[((__i0 + (__i1 * exxbuff_d0)) + ((__i2 * exxbuff_d0) * exxbuff_d1))];
                            dace::complex128 _out;

                            ///////////////////
                            // Tasklet code (copy_exxbuff_d_1082_tasklet)
                            _out = _in;
                            ///////////////////

                            _cpy_out[((__i0 + (__i1 * exxbuff_d_d0)) + ((__i2 * exxbuff_d_d0) * exxbuff_d_d1))] = _out;
                        }
                    }
                }
            }
        }

    }
}

inline void dft_nd_383_1_2(vexx_bp_k_gpu_state_t *__state, dace::complex128*  _inp, dace::complex128*  _out, int ialloc, int npol, int nrxxs) {
    dace::complex128 *__buf0;
    __buf0 = new dace::complex128 DACE_ALIGN(64)[((ialloc * npol) * nrxxs)];
    dace::complex128 *__buf1;
    __buf1 = new dace::complex128 DACE_ALIGN(64)[((ialloc * npol) * nrxxs)];

    {

        {
            #pragma omp parallel for
            for (auto __c0 = 0; __c0 < nrxxs; __c0 += 1) {
                for (auto __c1 = 0; __c1 < npol; __c1 += 1) {
                    for (auto __c2 = 0; __c2 < ialloc; __c2 += 1) {
                        {
                            dace::complex128 i = _inp[((((__c0 * ialloc) * npol) + (__c1 * ialloc)) + __c2)];
                            dace::complex128 o;

                            ///////////////////
                            // Tasklet code (cast_in)
                            o = decltype(o)(i);
                            ///////////////////

                            __buf0[((((__c0 * ialloc) * npol) + (__c1 * ialloc)) + __c2)] = o;
                        }
                    }
                }
            }
        }

    }
    {

        {
            #pragma omp parallel for
            for (auto __z0 = 0; __z0 < nrxxs; __z0 += 1) {
                for (auto __z1 = 0; __z1 < npol; __z1 += 1) {
                    for (auto __z2 = 0; __z2 < ialloc; __z2 += 1) {
                        {
                            dace::complex128 o;

                            ///////////////////
                            // Tasklet code (zero)
                            o = 0;
                            ///////////////////

                            __buf1[((((__z0 * ialloc) * npol) + (__z1 * ialloc)) + __z2)] = o;
                        }
                    }
                }
            }
        }

    }
    {

        {
            #pragma omp parallel for
            for (auto __i0 = 0; __i0 < nrxxs; __i0 += 1) {
                for (auto __i1 = 0; __i1 < npol; __i1 += 1) {
                    for (auto __i2 = 0; __i2 < ialloc; __i2 += 1) {
                        for (auto __n = 0; __n < nrxxs; __n += 1) {
                            {
                                dace::complex128 inp = __buf0[(((__i1 * ialloc) + __i2) + ((__n * ialloc) * npol))];
                                dace::complex128 o;

                                ///////////////////
                                // Tasklet code (dft)
                                auto exponent = ((((2.0 * 3.141592653589793) / nrxxs) * __i0) * __n);
                                o = (decltype(o)(dace::math::cos(exponent), (- dace::math::sin(exponent))) * inp);
                                ///////////////////

                                dace::wcr_fixed<dace::ReductionType::Sum, dace::complex128>::reduce_atomic(__buf1 + ((((__i0 * ialloc) * npol) + (__i1 * ialloc)) + __i2), o);
                            }
                        }
                    }
                }
            }
        }

    }
    {

        {
            #pragma omp parallel for
            for (auto __z0 = 0; __z0 < nrxxs; __z0 += 1) {
                for (auto __z1 = 0; __z1 < npol; __z1 += 1) {
                    for (auto __z2 = 0; __z2 < ialloc; __z2 += 1) {
                        {
                            dace::complex128 o;

                            ///////////////////
                            // Tasklet code (zero)
                            o = 0;
                            ///////////////////

                            __buf0[((((__z0 * ialloc) * npol) + (__z1 * ialloc)) + __z2)] = o;
                        }
                    }
                }
            }
        }

    }
    {

        {
            #pragma omp parallel for
            for (auto __i0 = 0; __i0 < nrxxs; __i0 += 1) {
                for (auto __i1 = 0; __i1 < npol; __i1 += 1) {
                    for (auto __i2 = 0; __i2 < ialloc; __i2 += 1) {
                        for (auto __n = 0; __n < npol; __n += 1) {
                            {
                                dace::complex128 inp = __buf1[((((__i0 * ialloc) * npol) + __i2) + (__n * ialloc))];
                                dace::complex128 o;

                                ///////////////////
                                // Tasklet code (dft)
                                auto exponent = ((((2.0 * 3.141592653589793) / npol) * __i1) * __n);
                                o = (decltype(o)(dace::math::cos(exponent), (- dace::math::sin(exponent))) * inp);
                                ///////////////////

                                dace::wcr_fixed<dace::ReductionType::Sum, dace::complex128>::reduce_atomic(__buf0 + ((((__i0 * ialloc) * npol) + (__i1 * ialloc)) + __i2), o);
                            }
                        }
                    }
                }
            }
        }

    }
    {

        {
            #pragma omp parallel for
            for (auto __z0 = 0; __z0 < nrxxs; __z0 += 1) {
                for (auto __z1 = 0; __z1 < npol; __z1 += 1) {
                    for (auto __z2 = 0; __z2 < ialloc; __z2 += 1) {
                        {
                            dace::complex128 o;

                            ///////////////////
                            // Tasklet code (zero)
                            o = 0;
                            ///////////////////

                            _out[((((__z0 * ialloc) * npol) + (__z1 * ialloc)) + __z2)] = o;
                        }
                    }
                }
            }
        }

    }
    {

        {
            #pragma omp parallel for
            for (auto __i0 = 0; __i0 < nrxxs; __i0 += 1) {
                for (auto __i1 = 0; __i1 < npol; __i1 += 1) {
                    for (auto __i2 = 0; __i2 < ialloc; __i2 += 1) {
                        for (auto __n = 0; __n < ialloc; __n += 1) {
                            {
                                dace::complex128 inp = __buf0[((((__i0 * ialloc) * npol) + (__i1 * ialloc)) + __n)];
                                dace::complex128 o;

                                ///////////////////
                                // Tasklet code (dft)
                                auto exponent = ((((2.0 * 3.141592653589793) / ialloc) * __i2) * __n);
                                o = ((decltype(o)(dace::math::cos(exponent), (- dace::math::sin(exponent))) * inp) * 1.0);
                                ///////////////////

                                dace::wcr_fixed<dace::ReductionType::Sum, dace::complex128>::reduce_atomic(_out + ((((__i0 * ialloc) * npol) + (__i1 * ialloc)) + __i2), o);
                            }
                        }
                    }
                }
            }
        }

    }
    delete[] __buf0;
    delete[] __buf1;
}

inline void dft_nd_387_1_2(vexx_bp_k_gpu_state_t *__state, dace::complex128*  _inp, dace::complex128*  _out, int ialloc, int nrxxs) {
    __vexx_fft3d(_inp, _out, nrxxs, ialloc, 1L, (long)nrxxs, -1.0, 1.0);
}

inline void copy_big_result_d_1372_sdfg_0_119_8(vexx_bp_k_gpu_state_t *__state, dace::complex128* __restrict__ _cpy_in, dace::complex128* __restrict__ _cpy_out, int m, int n, int npol) {

    {

        {
            #pragma omp parallel for
            for (auto __i0 = 0; __i0 < (n * npol); __i0 += 1) {
                for (auto __i1 = 0; __i1 < m; __i1 += 1) {
                    {
                        dace::complex128 _in = _cpy_in[(__i0 + ((__i1 * n) * npol))];
                        dace::complex128 _out;

                        ///////////////////
                        // Tasklet code (copy_big_result_d_1372_tasklet)
                        _out = _in;
                        ///////////////////

                        _cpy_out[(__i0 + ((__i1 * n) * npol))] = _out;
                    }
                }
            }
        }

    }
}

inline void copy_hpsi_1404_sdfg_0_122_2(vexx_bp_k_gpu_state_t *__state, dace::complex128* __restrict__ _cpy_in, dace::complex128* __restrict__ _cpy_out, int hpsi_d_d0, int hpsi_d_d1, int lda, int npol) {

    {

        {
            #pragma omp parallel for
            for (auto __i0 = 0; __i0 < hpsi_d_d0; __i0 += 1) {
                for (auto __i1 = 0; __i1 < hpsi_d_d1; __i1 += 1) {
                    {
                        dace::complex128 _in = _cpy_in[(__i0 + (__i1 * hpsi_d_d0))];
                        dace::complex128 _out;

                        ///////////////////
                        // Tasklet code (copy_hpsi_1404_tasklet)
                        _out = _in;
                        ///////////////////

                        _cpy_out[(__i0 + ((__i1 * lda) * npol))] = _out;
                    }
                }
            }
        }

    }
}

void __program_vexx_bp_k_gpu_internal(vexx_bp_k_gpu_state_t*__state, int * __restrict__ all_end, int * __restrict__ all_start, double * __restrict__ ap, double * __restrict__ at, double * __restrict__ becphi_r, dace::complex128 * __restrict__ becpsi_k, int * __restrict__ becpsi_nbnd, dace::complex128 * __restrict__ becpsi_nc, double * __restrict__ becpsi_r, dace::complex128 * __restrict__ becxx_k, bool * __restrict__ coulomb_done, double * __restrict__ coulomb_fac, int * __restrict__ dfftt_nl, int * __restrict__ dfftt_nlm, int * __restrict__ egrp_pairs, dace::complex128 * __restrict__ eigts1, dace::complex128 * __restrict__ eigts2, dace::complex128 * __restrict__ eigts3, double * __restrict__ eps, double * __restrict__ eps_qdiv, double * __restrict__ erf_scrlen, double * __restrict__ erfc_scrlen, double * __restrict__ exxalfa, dace::complex128 * __restrict__ exxbuff, dace::complex128 * __restrict__ exxbuff_d, double * __restrict__ exxdiv, double * __restrict__ g, bool * __restrict__ gamma_only, double * __restrict__ gau_scrlen, double * __restrict__ grid_factor, double * __restrict__ gt, dace::complex128 * __restrict__ hpsi, int * __restrict__ ibands, int * __restrict__ iexx_iend, int * __restrict__ iexx_istart, int * __restrict__ iexx_istart_d, int * __restrict__ igk_exx, int * __restrict__ igk_exx_d, int * __restrict__ ijtoh, int * __restrict__ index_xk, int * __restrict__ index_xkq, int * __restrict__ indv, int * __restrict__ inter_egrp_comm, int * __restrict__ intra_egrp_comm, bool * __restrict__ ionode, int * __restrict__ ityp, double * __restrict__ ke_k, int * __restrict__ kunit, int * __restrict__ lpl, int * __restrict__ lpx, int * __restrict__ many_fft, int * __restrict__ me_egrp, int * __restrict__ mill, int * __restrict__ nh, int * __restrict__ nhtol, int * __restrict__ nhtolm, int * __restrict__ nibands, int * __restrict__ nij_type, int * __restrict__ nkstot, bool * __restrict__ noncolin, int * __restrict__ npool, int * __restrict__ nq1, int * __restrict__ nq2, int * __restrict__ nq3, int * __restrict__ ofsbeta, bool * __restrict__ okpaw, bool * __restrict__ okvan, double * __restrict__ omega, bool * __restrict__ paw_has_init_paw_fockrnl, dace::complex128 * __restrict__ psi, dace::complex128 * __restrict__ qgm, double * __restrict__ tab_beta, double * __restrict__ tab_qrad, int * __restrict__ tabxx_box, int * __restrict__ tabxx_maxbox, double * __restrict__ tabxx_qr, double * __restrict__ tau, double * __restrict__ tpiba, double * __restrict__ tpiba2, bool * __restrict__ tqr, int * __restrict__ upf_nbeta, bool * __restrict__ upf_tpawp, bool * __restrict__ upf_tvanp, bool * __restrict__ use_coulomb_vcut_spheric, bool * __restrict__ use_coulomb_vcut_ws, double * __restrict__ vcut_a, double * __restrict__ vcut_corrected, bool * __restrict__ x_gamma_extrapolation, double * __restrict__ x_occupation, double * __restrict__ x_occupation_d, double * __restrict__ xk, double * __restrict__ xkq_collect, double * __restrict__ yukawa, int64_t becpsi_k_d0, int64_t becpsi_nc_d0, int64_t becpsi_nc_d1, int64_t becxx_k_d0, int64_t becxx_k_d1, int current_k, int64_t dfftt__nl_d0, int dfftt_ngm, int dfftt_nnr, int64_t egrp_pairs_d0, int64_t egrp_pairs_d1, int64_t eigts1_d0, int64_t eigts2_d0, int64_t eigts3_d0, int64_t exxbuff_d0, int64_t exxbuff_d1, int64_t exxbuff_d2, int64_t exxbuff_d_d0, int64_t exxbuff_d_d1, int64_t g_d0, int gstart, int64_t gt_d0, int64_t hpsi_d_d0, int64_t hpsi_d_d1, int64_t ibands_d0, int iexx_start, int64_t igk_exx_d0, int64_t igk_exx_d_d0, int64_t ijtoh_d0, int64_t ijtoh_d1, int64_t index_xkq_d0, int64_t indv_d0, int jblock, int64_t ke_k_d0, int64_t ke_k_d1, int64_t ke_k_d2, int64_t ke_k_d3, int lda, int lmaxkb, int lmaxq, int m, int max_pairs, int64_t mill_d0, int my_egrp_id, int my_pool_id, int n, int nat, int nbetam, int negrp, int nhm, int64_t nhtol_d0, int64_t nhtolm_d0, int nkb, int npol, int npwx, int nqs, int nqx, int nsp, int64_t offset_becpsi_k_d0, int64_t offset_becpsi_k_d1, int64_t psi_d_d0, int64_t psi_d_d1, bool run_on_gpu_, int64_t tab_beta_d0, int64_t tab_beta_d1, int64_t tab_qrad_d0, int64_t tab_qrad_d1, int64_t tab_qrad_d2, int64_t tabxx_box_d0, int64_t tabxx_qr_d0, int64_t tabxx_qr_d1, int64_t tau_d0, int64_t vcut_corrected_d0, int64_t vcut_corrected_d1, int64_t vcut_corrected_d2, int64_t x_occupation_d0, int64_t x_occupation_d_d0, int64_t xkq_collect_d0, int64_t ylm_d1)
{
    dace::complex128 *big_result;
    big_result = new dace::complex128 DACE_ALIGN(64)[(((n * npol) * (m - 1)) + (n * npol))];
    dace::complex128 *big_result_d;
    big_result_d = new dace::complex128 DACE_ALIGN(64)[(((n * npol) * (m - 1)) + (n * npol))];
    dace::complex128 *deexx = nullptr;
    int *dfftt__nl;
    dfftt__nl = new int DACE_ALIGN(64)[dfftt__nl_d0];
    // MANUAL FIX (not auto-generated): the Fortran aliases dfftt__nl => dfftt%nl_d
    // (the device copy of the FFT-grid map, == the host dfftt_nl). The SDFG made
    // it an uninitialized transient, so the setup scatter read garbage indices.
    // Initialize it from the input dfftt_nl (device == host).
    for (int64_t __i = 0; __i < dfftt__nl_d0; __i += 1)
        dfftt__nl[__i] = dfftt_nl[__i];
    double *facb = nullptr;
    double *facb_d = nullptr;
    dace::complex128 *hpsi_d;
    hpsi_d = new dace::complex128 DACE_ALIGN(64)[((hpsi_d_d0 * (hpsi_d_d1 - 1)) + hpsi_d_d0)];
    dace::complex128 *psi_d;
    psi_d = new dace::complex128 DACE_ALIGN(64)[((psi_d_d0 * (psi_d_d1 - 1)) + psi_d_d0)];
    // MANUAL FIX (not auto-generated): the Fortran allocates hpsi_d/psi_d with
    // SOURCE=hpsi / SOURCE=psi -- i.e. as copies of the inputs. The SDFG lowering
    // dropped that initialization (flagged "Use of uninitialized transient"), so
    // the closing ``hpsi = hpsi_d`` and the setup loop read garbage. Capture the
    // SOURCE= copy here. Index convention mirrors copy_hpsi_1404 (input stride
    // lda*npol, transient stride <name>_d0).
    for (int64_t __i1 = 0; __i1 < hpsi_d_d1; __i1 += 1)
        for (int64_t __i0 = 0; __i0 < hpsi_d_d0; __i0 += 1)
            hpsi_d[(__i0 + (__i1 * hpsi_d_d0))] = hpsi[(__i0 + ((__i1 * lda) * npol))];
    for (int64_t __i1 = 0; __i1 < psi_d_d1; __i1 += 1)
        for (int64_t __i0 = 0; __i0 < psi_d_d0; __i0 += 1)
            psi_d[(__i0 + (__i1 * psi_d_d0))] = psi[(__i0 + ((__i1 * lda) * npol))];
    dace::complex128 *result_d = nullptr;
    dace::complex128 *result_nc_d = nullptr;
    dace::complex128 *rhoc = nullptr;
    dace::complex128 *rhoc_d = nullptr;
    dace::complex128 *temppsic_d = nullptr;
    dace::complex128 *temppsic_nc_d = nullptr;
    dace::complex128 *vc = nullptr;
    dace::complex128 *vc_d = nullptr;
    double *xkp;
    xkp = new double DACE_ALIGN(64)[3];
    double *xkq;
    xkq = new double DACE_ALIGN(64)[3];
    double *nqhalf_dble;
    nqhalf_dble = new double DACE_ALIGN(64)[3];
    bool *odg;
    odg = new bool DACE_ALIGN(64)[3];
    double *g2_convolution_q;
    g2_convolution_q = new double DACE_ALIGN(64)[3];
    double *grid_factor_track;
    grid_factor_track = new double DACE_ALIGN(64)[dfftt_ngm];
    double *qq_track;
    qq_track = new double DACE_ALIGN(64)[dfftt_ngm];
    int *i;
    i = new int DACE_ALIGN(64)[3];
    double *i_real;
    i_real = new double DACE_ALIGN(64)[3];
    double *a;
    a = new double DACE_ALIGN(64)[9];
    double *qvan_init_q;
    qvan_init_q = new double DACE_ALIGN(64)[(3 * dfftt_ngm)];
    double *qmod;
    qmod = new double DACE_ALIGN(64)[dfftt_ngm];
    double *qvan_init_qq;
    qvan_init_qq = new double DACE_ALIGN(64)[dfftt_ngm];
    double *ylmk0;
    ylmk0 = new double DACE_ALIGN(64)[((dfftt_ngm * ((lmaxq * lmaxq) - 1)) + dfftt_ngm)];
    dace::complex128 *addusxx_g_aux1;
    addusxx_g_aux1 = new dace::complex128 DACE_ALIGN(64)[256];
    dace::complex128 *addusxx_g_aux2;
    addusxx_g_aux2 = new dace::complex128 DACE_ALIGN(64)[256];
    dace::complex128 *addusxx_g_eigqts;
    addusxx_g_eigqts = new dace::complex128 DACE_ALIGN(64)[nat];
    dace::complex128 *newdxx_g_aux1;
    newdxx_g_aux1 = new dace::complex128 DACE_ALIGN(64)[256];
    dace::complex128 *newdxx_g_aux2;
    newdxx_g_aux2 = new dace::complex128 DACE_ALIGN(64)[256];
    dace::complex128 *auxvc = nullptr;
    dace::complex128 *newdxx_g_eigqts;
    newdxx_g_eigqts = new dace::complex128 DACE_ALIGN(64)[nat];
    dace::complex128 *vkbp;
    vkbp = new dace::complex128 DACE_ALIGN(64)[((npwx * (nkb - 1)) + npwx)];
    double *gk;
    gk = new double DACE_ALIGN(64)[(3 * n)];
    double *init_us_2_acc_qg;
    init_us_2_acc_qg = new double DACE_ALIGN(64)[n];
    dace::complex128 *sk;
    sk = new dace::complex128 DACE_ALIGN(64)[n];
    double *vkb1;
    vkb1 = new double DACE_ALIGN(64)[((n * (nhm - 1)) + n)];
    double *vq;
    vq = new double DACE_ALIGN(64)[((n * (nbetam - 1)) + n)];
    double *ylm;
    ylm = new double DACE_ALIGN(64)[((n * (ylm_d1 - 1)) + n)];
    int dfftt_nr3;
    int dfftt_nr2;
    int dfftt_nr1;
    int jblock_end;
    int jblock_start;
    double nqs_inv;
    double omega_inv;
    int ik_g;
    int nkbl;
    double vcut_cutoff;
    double g2_convolution_qq;
    double x;
    double vcut_get_res;
    double kg2;
    double rcut;
    double vcut_spheric_get_res;
    double c;
    double cost;
    double gmod;
    bool goto_10;
    double phi;
    double sent;
    double dqi;
    double qm;
    double sig;
    bool addusxx_g_add_complex;
    bool addusxx_g_add_imaginary;
    bool addusxx_g_add_real;
    int addusxx_g_ijkb0;
    int addusxx_g_ngms;
    bool newdxx_g_add_complex;
    bool newdxx_g_add_imaginary;
    bool newdxx_g_add_real;
    double fact;
    dace::complex128 fm;
    dace::complex128 fp;
    int newdxx_g_ijkb0;
    dace::complex128 aux;
    double domega;
    int newdxx_r_ijkb0;
    double __assoc_scalar_14;
    int paw_newdxx_ijkb0;
    double __assoc_scalar_16;
    bool run_on_gpu;
    double init_us_2_acc_arg;
    double q1;
    double q2;
    double q3;
    double interp_beta_px;
    double qgr;
    double interp_beta_ux;
    double interp_beta_vx;
    double interp_beta_wx;
    int64_t if_cond_185;
    int64_t if_cond_193;
    int64_t if_cond_203;
    double *_libtmp_0;
    _libtmp_0 = new double DACE_ALIGN(64)[3];
    double *_libsrc_0;
    _libsrc_0 = new double DACE_ALIGN(64)[3];
    double __reduce_cond_1;
    double *_libsrc_2;
    _libsrc_2 = new double DACE_ALIGN(64)[3];
    double __reduce_cond_3;
    double *_mask_4;
    _mask_4 = new double DACE_ALIGN(64)[3];
    double *_libsrc_6;
    _libsrc_6 = new double DACE_ALIGN(64)[9];
    double *_libtmp_1;
    _libtmp_1 = new double DACE_ALIGN(64)[3];
    double *_mask_5;
    _mask_5 = new double DACE_ALIGN(64)[3];
    double *_mask_7;
    _mask_7 = new double DACE_ALIGN(64)[3];
    double *_mask_8;
    _mask_8 = new double DACE_ALIGN(64)[3];
    bool __allany_cond_9;
    double *_mask_10;
    _mask_10 = new double DACE_ALIGN(64)[3];
    double *_mask_11;
    _mask_11 = new double DACE_ALIGN(64)[3];
    int __brkc_3;
    int __sc_0;
    int64_t if_cond_449;
    int64_t if_cond_455;
    int64_t if_cond_596;
    double *_mask_12;
    _mask_12 = new double DACE_ALIGN(64)[3];
    int64_t if_cond_722;
    double *_mask_13;
    _mask_13 = new double DACE_ALIGN(64)[3];
    int64_t if_cond_1023;
    int __brkc_4;
    int __sc_1;
    int64_t if_cond_1239;
    int64_t if_cond_1245;
    int64_t if_cond_1303;
    int64_t if_cond_1340;
    int64_t if_cond_1373;
    bool program_limit;
    int64_t __sym_dfftt_ngm_1;
    int addusxx_g_aux1_allocated;
    int addusxx_g_aux2_allocated;
    int addusxx_g_eigqts_allocated;
    int auxvc_allocated;
    int becpsi_k_allocated;
    int big_result_allocated;
    int big_result_d_allocated;
    int coulomb_done_allocated;
    int coulomb_fac_allocated;
    int deexx_allocated;
    int fac_allocated;
    int facb_allocated;
    int facb_d_allocated;
    int gk_allocated;
    int hpsi_d_allocated;
    int init_us_2_acc_qg_allocated;
    int newdxx_g_aux1_allocated;
    int newdxx_g_aux2_allocated;
    int newdxx_g_eigqts_allocated;
    int nij_type_allocated;
    int psi_d_allocated;
    int qgm_allocated;
    int qmod_allocated;
    int qvan_init_q_allocated;
    int qvan_init_qq_allocated;
    int result_d_allocated;
    int result_nc_d_allocated;
    int rhoc_allocated;
    int rhoc_d_allocated;
    int sk_allocated;
    int temppsic_d_allocated;
    int temppsic_nc_d_allocated;
    int vc_allocated;
    int vc_d_allocated;
    int vkb1_allocated;
    int vkbp_allocated;
    int vq_allocated;
    int ylm_allocated;
    int ylmk0_allocated;
    int __al_0;
    int __al_1;
    int __al_2;
    int __al_3;
    int __al_4;
    int __al_5;
    int __al_6;
    int __al_7;
    int __al_8;
    int __al_9;
    int __al_10;
    int __al_11;
    int __al_12;
    int __al_13;
    int __al_14;
    int __al_15;
    int __al_16;
    int __al_17;
    int ialloc;
    int nrxxs;
    int64_t facb_d0;
    int64_t facb_d_d0;
    int64_t if_cond_67;
    int64_t if_cond_86;
    int nks;
    int rest;
    int64_t if_cond_95;
    int64_t if_cond_100;
    int current_ik;
    int64_t ss_0;
    int64_t big_result_d0;
    int64_t big_result_d1;
    int64_t big_result_d_d0;
    int64_t big_result_d_d1;
    int64_t rhoc_d_d0;
    int64_t rhoc_d_d1;
    int64_t vc_d_d0;
    int64_t vc_d_d1;
    int64_t rhoc_d0;
    int64_t rhoc_d1;
    int64_t vc_d0;
    int64_t vc_d1;
    int64_t loopend_125;
    int ii;
    int64_t if_cond_172;
    int iq;
    int64_t loopend_1092;
    int64_t result_nc_d_d0;
    int64_t result_nc_d_d1;
    int64_t result_nc_d_d2;
    int64_t temppsic_nc_d_d0;
    int64_t temppsic_nc_d_d1;
    int64_t temppsic_nc_d_d2;
    int64_t result_d_d0;
    int64_t result_d_d1;
    int64_t temppsic_d_d0;
    int64_t temppsic_d_d1;
    int64_t deexx_d0;
    int64_t deexx_d1;
    int64_t _loop_it_822;
    int64_t _loop_it_823;
    int ibnd;
    int64_t if_cond_129;
    int64_t if_cond_132;
    int64_t if_cond_138;
    int64_t if_cond_150;
    int64_t loopend_134;
    int64_t as_0;
    int64_t _loop_it_824;
    int64_t loopend_140;
    int64_t _loop_it_825;
    int64_t loopend_143;
    int64_t as_1;
    int64_t _loop_it_826;
    int64_t loopend_146;
    int64_t _loop_it_827;
    int ig;
    int64_t _loop_it_828;
    int64_t igk_exx_d_at20;
    int64_t dfftt__nl_at21;
    int64_t _loop_it_829;
    int64_t igk_exx_d_at22;
    int64_t dfftt__nl_at23;
    int64_t _loop_it_830;
    int ikq;
    int ik;
    int64_t loopend_182;
    int64_t if_cond_352;
    int iegrp;
    int64_t if_cond_1085;
    int64_t _loop_it_831;
    int64_t coulomb_fac_d1;
    int64_t coulomb_fac_d2;
    int64_t coulomb_done_d0;
    int64_t coulomb_done_d1;
    int64_t if_cond_207;
    int g2_convolution_ig;
    int64_t _loop_it_832;
    int64_t ei0;
    int64_t li0;
    int64_t if_cond_218;
    int64_t if_cond_223;
    int64_t _loop_it_833;
    int64_t _loop_it_834;
    int64_t _loop_it_835;
    int64_t _loop_it_836;
    int64_t _loop_it_837;
    int64_t _loop_it_838;
    int64_t if_cond_228;
    int64_t i_at24;
    int64_t i_at25;
    int64_t i_at26;
    int64_t if_cond_241;
    int64_t _loop_it_839;
    int64_t if_cond_255;
    int64_t if_cond_259;
    int64_t _loop_it_840;
    int64_t _loop_it_841;
    int64_t li1;
    int64_t _loop_it_842;
    int64_t _loop_it_843;
    int64_t _loop_it_844;
    int64_t if_cond_273;
    int64_t _loop_it_845;
    int64_t _loop_it_846;
    int64_t _loop_it_847;
    int64_t ab_0;
    int64_t _loop_it_848;
    int64_t _loop_it_849;
    int64_t _loop_it_850;
    int64_t _loop_it_851;
    int64_t _loop_it_852;
    int64_t if_cond_309;
    int64_t if_cond_314;
    int64_t if_cond_317;
    int64_t if_cond_322;
    int64_t if_cond_330;
    int64_t if_cond_335;
    int64_t _loop_it_853;
    int64_t dfftt__nl_at27;
    int __al_18;
    int __al_19;
    int __al_20;
    int __al_21;
    int64_t nij_type_d0;
    int qvan_init_nij;
    int qvan_init_nt;
    int64_t qgm_d1;
    int64_t ylmk0_d1;
    int64_t qvan_init_q_d0;
    int qvan_init_ig;
    int __assoc_scalar_2;
    int64_t if_cond_389;
    int ijh;
    int64_t _loop_it_854;
    int64_t _loop_it_855;
    int64_t loopend_381;
    int64_t _loop_it_856;
    int64_t _loop_it_857;
    int __al_22;
    int lmax;
    int64_t if_cond_414;
    int64_t if_cond_417;
    int64_t if_cond_396;
    int64_t if_cond_410;
    int64_t if_cond_399;
    int64_t _loop_it_858;
    int ylmr2_ig;
    int64_t _loop_it_859;
    int64_t if_cond_423;
    int ylmr2_l;
    int ylmr2_lm;
    int64_t _loop_it_860;
    int64_t loopend_435;
    int ylmr2_m;
    int lm1;
    int lm2;
    int64_t _loop_it_861;
    int64_t _loop_it_862;
    int64_t _loop_it_863;
    int64_t _loop_it_864;
    int64_t loopend_485;
    int qvan_init_ih;
    int64_t _loop_it_865;
    int64_t loopend_488;
    int qvan_init_jh;
    int64_t _loop_it_866;
    int qvan2_nb;
    int mb;
    int64_t if_cond_495;
    int ivl;
    int jvl;
    int64_t if_cond_504;
    int64_t if_cond_508;
    int64_t loopend_514;
    int qvan2_lm;
    int ijv;
    int64_t _loop_it_867;
    int lp;
    int64_t if_cond_518;
    int64_t if_cond_522;
    int qvan2_ig;
    int qvan2_l;
    int ind;
    int64_t if_cond_528;
    int64_t if_cond_534;
    int64_t if_cond_540;
    int64_t if_cond_546;
    int64_t if_cond_552;
    int64_t _loop_it_868;
    int qvan2_i0;
    int qvan2_i1;
    int qvan2_i2;
    int qvan2_i3;
    int64_t _loop_it_869;
    int wegrp;
    int njt;
    int ijt;
    int64_t if_cond_1079;
    int64_t _loop_it_870;
    int64_t loopend_585;
    int64_t _loop_it_871;
    int64_t if_cond_589;
    int jstart;
    int jend;
    int ipair;
    int jcount;
    int64_t if_cond_612;
    int64_t _loop_it_872;
    int64_t if_cond_600;
    int all_start_tmp;
    int jbnd;
    int64_t if_cond_629;
    int64_t if_cond_672;
    int64_t if_cond_821;
    int64_t if_cond_964;
    int64_t if_cond_1005;
    int64_t _loop_it_873;
    int ir;
    int64_t _loop_it_874;
    int64_t if_cond_619;
    int64_t _loop_it_875;
    int64_t if_cond_633;
    int addusxx_r_ia;
    int64_t _loop_it_876;
    int addusxx_r_mbia;
    int64_t if_cond_638;
    int addusxx_r_nt;
    int64_t loopend_643;
    int addusxx_r_ih;
    int64_t _loop_it_877;
    int64_t loopend_646;
    int addusxx_r_jh;
    int64_t _loop_it_878;
    int addusxx_r_ikb;
    int addusxx_r_jkb;
    int addusxx_r_ir;
    int64_t _loop_it_879;
    int64_t ijtoh_at28;
    int irb;
    int _loop_it_880;
    int jcurr;
    int64_t _loop_it_881;
    int __al_23;
    int __al_24;
    int __al_25;
    int64_t if_cond_681;
    int64_t if_cond_684;
    int64_t if_cond_688;
    int64_t if_cond_692;
    int64_t if_cond_696;
    int64_t addusxx_g_eigqts_d0;
    int addusxx_g_na;
    int addusxx_g_numblock;
    int64_t addusxx_g_aux1_d0;
    int64_t addusxx_g_aux2_d0;
    int addusxx_g_nt;
    int64_t _loop_it_882;
    int64_t _loop_it_883;
    int64_t _loop_it_884;
    int addusxx_g_nij;
    int addusxx_g_iblock;
    int64_t _loop_it_885;
    int64_t _loop_it_886;
    int addusxx_g_offset;
    int addusxx_g_realblocksize;
    int64_t loopend_729;
    int64_t loopend_731;
    int addusxx_g_ih;
    int64_t _loop_it_887;
    int64_t _loop_it_888;
    int addusxx_g_ikb;
    int64_t loopend_736;
    int64_t loopend_738;
    int addusxx_g_jh;
    int64_t _loop_it_889;
    int64_t _loop_it_890;
    int addusxx_g_jkb;
    int64_t _loop_it_891;
    int64_t ijtoh_at29;
    int64_t _loop_it_892;
    int64_t ijtoh_at30;
    int64_t _loop_it_893;
    int64_t _loop_it_894;
    int64_t _loop_it_895;
    int64_t mill_at31;
    int64_t mill_at32;
    int64_t mill_at33;
    int64_t loopend_764;
    int64_t _doit_76;
    int64_t _loop_it_896;
    int64_t dfftt__nl_at34;
    int64_t loopend_771;
    int64_t _doit_77;
    int64_t if_cond_777;
    int64_t loopend_781;
    int64_t _doit_78;
    int64_t _loop_it_897;
    int64_t dfftt__nl_at35;
    int64_t _loop_it_898;
    int64_t dfftt_nlm_at36;
    int64_t loopend_788;
    int64_t _doit_79;
    int64_t if_cond_794;
    int64_t loopend_798;
    int64_t _doit_80;
    int64_t _loop_it_899;
    int64_t dfftt__nl_at37;
    int64_t _loop_it_900;
    int64_t dfftt_nlm_at38;
    int64_t _loop_it_901;
    int64_t _loop_it_902;
    int64_t _loop_it_903;
    int __al_26;
    int __al_27;
    int __al_28;
    int __al_29;
    int64_t if_cond_831;
    int newdxx_g_ngms;
    int64_t if_cond_835;
    int64_t if_cond_839;
    int64_t if_cond_843;
    int64_t if_cond_847;
    int64_t auxvc_d0;
    int64_t newdxx_g_eigqts_d0;
    int newdxx_g_na;
    int newdxx_g_numblock;
    int64_t newdxx_g_aux1_d0;
    int64_t newdxx_g_aux2_d0;
    int newdxx_g_iblock;
    int64_t _loop_it_904;
    int64_t _loop_it_905;
    int64_t _loop_it_906;
    int64_t dfftt__nl_at39;
    int newdxx_g_ig;
    int64_t _loop_it_907;
    int64_t dfftt__nl_at40;
    int64_t dfftt_nlm_at41;
    int64_t dfftt__nl_at42;
    int64_t dfftt_nlm_at43;
    int64_t _loop_it_908;
    int64_t dfftt__nl_at44;
    int64_t dfftt_nlm_at45;
    int64_t dfftt__nl_at46;
    int64_t dfftt_nlm_at47;
    int64_t _loop_it_909;
    int newdxx_g_offset;
    int newdxx_g_realblocksize;
    int64_t _loop_it_910;
    int newdxx_g_nt;
    int newdxx_g_nij;
    int64_t loopend_910;
    int64_t loopend_916;
    int newdxx_g_ih;
    int64_t _loop_it_911;
    int64_t mill_at48;
    int64_t mill_at49;
    int64_t mill_at50;
    int64_t _loop_it_912;
    int newdxx_g_ikb;
    int64_t loopend_921;
    int64_t loopend_923;
    int newdxx_g_jh;
    int64_t if_cond_940;
    int64_t _loop_it_913;
    int64_t _loop_it_914;
    int newdxx_g_jkb;
    int64_t if_cond_928;
    int64_t _loop_it_915;
    int64_t ijtoh_at51;
    int64_t _loop_it_916;
    int64_t ijtoh_at52;
    int _loop_it_917;
    int64_t _loop_it_918;
    int newdxx_r_ia;
    int64_t _loop_it_919;
    int newdxx_r_mbia;
    int64_t if_cond_973;
    int newdxx_r_nt;
    int64_t loopend_978;
    int newdxx_r_ih;
    int64_t _loop_it_920;
    int64_t loopend_981;
    int newdxx_r_jh;
    int64_t _loop_it_921;
    int newdxx_r_ikb;
    int newdxx_r_jkb;
    int newdxx_r_ir;
    int64_t _loop_it_922;
    int64_t ijtoh_at53;
    int64_t tabxx_box_at54;
    int64_t _loop_it_923;
    int64_t if_cond_1012;
    int64_t if_cond_1016;
    int paw_newdxx_np;
    int64_t _loop_it_924;
    int paw_newdxx_na;
    int64_t _loop_it_925;
    int64_t loopend_1028;
    int uh;
    int64_t _loop_it_926;
    int ukb;
    int64_t loopend_1032;
    int oh;
    int64_t _loop_it_927;
    int okb;
    int64_t loopend_1036;
    int paw_newdxx_jh;
    int64_t _loop_it_928;
    int paw_newdxx_jkb;
    int64_t loopend_1040;
    int paw_newdxx_ih;
    int64_t _loop_it_929;
    int paw_newdxx_ikb;
    int64_t _loop_it_930;
    int64_t _loop_it_931;
    int64_t if_cond_1063;
    int64_t _loop_it_932;
    int64_t if_cond_1096;
    int64_t if_cond_1099;
    int64_t if_cond_1102;
    int64_t loopend_1125;
    int64_t if_cond_1129;
    int64_t _loop_it_933;
    int64_t igk_exx_d_at55;
    int64_t dfftt__nl_at56;
    int64_t _loop_it_934;
    int64_t igk_exx_d_at57;
    int64_t dfftt__nl_at58;
    int64_t _loop_it_935;
    int __al_30;
    int64_t if_cond_1133;
    int64_t vkbp_d0;
    int64_t vkbp_d1;
    int __al_31;
    int __al_32;
    int __al_33;
    int __al_34;
    int __al_35;
    int __al_36;
    int64_t if_cond_1148;
    int add_nlxx_pot_np;
    int64_t vkb1_d0;
    int64_t vkb1_d1;
    int64_t sk_d0;
    int64_t init_us_2_acc_qg_d0;
    int64_t vq_d0;
    int64_t vq_d1;
    int64_t ylm_d0;
    int64_t gk_d0;
    int64_t gk_d1;
    int init_us_2_acc_ig;
    int __assoc_scalar_17;
    int64_t if_cond_1179;
    int init_us_2_acc_jkb;
    int init_us_2_acc_nt;
    int64_t _loop_it_936;
    int64_t _loop_it_937;
    int64_t _loop_it_938;
    int iv_d;
    int __al_37;
    int64_t if_cond_1204;
    int64_t if_cond_1207;
    int64_t if_cond_1186;
    int64_t if_cond_1200;
    int64_t if_cond_1189;
    int64_t _loop_it_939;
    int64_t _loop_it_940;
    int64_t if_cond_1213;
    int64_t _loop_it_941;
    int64_t loopend_1225;
    int64_t _loop_it_942;
    int64_t _loop_it_943;
    int64_t _loop_it_944;
    int64_t _loop_it_945;
    int64_t _loop_it_946;
    int nbnt;
    int interp_beta_nb;
    int nhnt;
    int init_us_2_acc_ih;
    int init_us_2_acc_na;
    int64_t _loop_it_947;
    int interp_beta_ig;
    int64_t _loop_it_948;
    int interp_beta_i0;
    int interp_beta_i1;
    int interp_beta_i2;
    int interp_beta_i3;
    int64_t if_cond_1283;
    int64_t _loop_it_949;
    int64_t _loop_it_950;
    int init_us_2_acc_nb;
    int init_us_2_acc_lm;
    int64_t _loop_it_951;
    int64_t _loop_it_952;
    int64_t mill_at59;
    int64_t mill_at60;
    int64_t mill_at61;
    int64_t _loop_it_953;
    int64_t _loop_it_954;
    int64_t _loop_it_955;
    int add_nlxx_pot_na;
    int64_t _loop_it_956;
    int64_t loopend_1343;
    int add_nlxx_pot_ih;
    int64_t _loop_it_957;
    int add_nlxx_pot_ikb;
    int64_t if_cond_1348;
    int64_t if_cond_1351;
    int add_nlxx_pot_ig;
    int64_t _loop_it_958;
    int64_t _loop_it_959;
    int64_t if_cond_1377;
    int64_t if_cond_1384;
    int ending_im;
    int program_im;
    int im;
    int64_t _loop_it_960;
    int64_t _loop_it_961;
    int64_t iexx_istart_d_at62;
    int64_t _loop_it_962;
    int64_t _loop_it_963;
    int64_t iexx_istart_d_at63;

    {

        {
            int _out;

            ///////////////////
            // Tasklet code (zinit_dfftt_nr2)
            _out = 0;
            ///////////////////

            dfftt_nr2 = _out;
        }
        {
            int _out;

            ///////////////////
            // Tasklet code (zinit_dfftt_nr1)
            _out = 0;
            ///////////////////

            dfftt_nr1 = _out;
        }
        {
            int _out;

            ///////////////////
            // Tasklet code (zinit_dfftt_nr3)
            _out = 0;
            ///////////////////

            dfftt_nr3 = _out;
        }

    }

    __sym_dfftt_ngm_1 = dfftt_ngm;

    addusxx_g_aux1_allocated = 0;

    addusxx_g_aux2_allocated = 0;

    addusxx_g_eigqts_allocated = 0;

    auxvc_allocated = 0;

    becpsi_k_allocated = 0;

    big_result_allocated = 0;

    big_result_d_allocated = 0;

    coulomb_done_allocated = 0;

    coulomb_fac_allocated = 0;

    deexx_allocated = 0;

    fac_allocated = 0;

    facb_allocated = 0;

    facb_d_allocated = 0;

    gk_allocated = 0;

    hpsi_d_allocated = 0;

    init_us_2_acc_qg_allocated = 0;

    newdxx_g_aux1_allocated = 0;

    newdxx_g_aux2_allocated = 0;

    newdxx_g_eigqts_allocated = 0;

    nij_type_allocated = 0;

    psi_d_allocated = 0;

    qgm_allocated = 0;

    qmod_allocated = 0;

    qvan_init_q_allocated = 0;

    qvan_init_qq_allocated = 0;

    result_d_allocated = 0;

    result_nc_d_allocated = 0;

    rhoc_allocated = 0;

    rhoc_d_allocated = 0;

    sk_allocated = 0;

    temppsic_d_allocated = 0;

    temppsic_nc_d_allocated = 0;

    vc_allocated = 0;

    vc_d_allocated = 0;

    vkb1_allocated = 0;

    vkbp_allocated = 0;

    vq_allocated = 0;

    ylm_allocated = 0;

    ylmk0_allocated = 0;

    __al_0 = 0;

    __al_1 = 0;

    __al_2 = 0;

    __al_3 = 0;

    __al_4 = 0;

    __al_5 = 0;

    __al_6 = 0;

    __al_7 = 0;

    __al_8 = 0;

    __al_9 = 0;

    __al_10 = 0;

    __al_11 = 0;

    __al_12 = 0;

    __al_13 = 0;

    __al_14 = 0;

    __al_15 = 0;

    __al_16 = 0;

    __al_17 = 0;

    ialloc = nibands[my_egrp_id];

    fac_allocated = 1;

    nrxxs = dfftt_nnr;

    facb_d0 = nrxxs;

    facb_allocated = 1;

    facb_d_d0 = nrxxs;

    facb_d_allocated = 1;
    {

        copy_exxbuff_d_66_sdfg_0_65_2(__state, &exxbuff[0], &exxbuff_d[0], exxbuff_d0, exxbuff_d1, exxbuff_d2, exxbuff_d_d0, exxbuff_d_d1);

    }
    if_cond_67 = noncolin[0];


    if (if_cond_67) {

        result_nc_d_d0 = nrxxs;

        result_nc_d_d1 = npol;

        result_nc_d_d2 = ialloc;

        result_nc_d_allocated = 1;

        temppsic_nc_d_d0 = nrxxs;

        temppsic_nc_d_d1 = npol;

        temppsic_nc_d_d2 = ialloc;

        temppsic_nc_d_allocated = 1;

    } else {

        result_d_d0 = nrxxs;

        result_d_d1 = ialloc;

        result_d_allocated = 1;

        temppsic_d_d0 = nrxxs;

        temppsic_d_d1 = ialloc;

        temppsic_d_allocated = 1;

    }


    if_cond_86 = okvan[0];


    if (if_cond_86) {

        deexx_d0 = nkb;

        deexx_d1 = ialloc;

        deexx_allocated = 1;

    }

    {

        {
            int _in_nkstot = nkstot[0];
            int _in_kunit = kunit[0];
            int _out;

            ///////////////////
            // Tasklet code (set_nkbl)
            _out = dace::math::ifloor(_in_nkstot / _in_kunit);
            ///////////////////

            nkbl = _out;
        }

    }
    nks = (kunit[0] * dace::math::ifloor(nkbl / npool[0]));

    rest = dace::math::ifloor((nkstot[0] - (nks * npool[0])) / kunit[0]);

    if_cond_95 = (my_pool_id < rest);


    if (if_cond_95) {

        nks = (nks + kunit[0]);

    }

    {

        {
            int _out;

            ///////////////////
            // Tasklet code (set_ik_g)
            _out = ((nks * my_pool_id) + current_k);
            ///////////////////

            ik_g = _out;
        }

    }
    if_cond_100 = (my_pool_id >= rest);


    if (if_cond_100) {
        {

            {
                int _in_kunit = kunit[0];
                int _in_ik_g = ik_g;
                int _out;

                ///////////////////
                // Tasklet code (set_ik_g)
                _out = (_in_ik_g + (rest * _in_kunit));
                ///////////////////

                ik_g = _out;
            }

        }
    }


    current_ik = ik_g;


    for (_loop_it_822 = 1; (_loop_it_822 < (3 + 1)); _loop_it_822 = (_loop_it_822 + 1)) {
        {

            {
                double _in_xk_0 = xk[((_loop_it_822 + (3 * current_k)) - 4)];
                double _out_xkp;

                ///////////////////
                // Tasklet code (t_0)
                _out_xkp = _in_xk_0;
                ///////////////////

                xkp[(_loop_it_822 - 1)] = _out_xkp;
            }

        }

    }

    ss_0 = 4;


    big_result_d0 = (n * npol);

    big_result_d1 = m;

    big_result_allocated = 1;
    {

        {
            #pragma omp parallel for
            for (auto __i0 = 0; __i0 < (n * npol); __i0 += 1) {
                for (auto __i1 = 0; __i1 < m; __i1 += 1) {
                    {
                        dace::complex128 _out;

                        ///////////////////
                        // Tasklet code (set_big_result)
                        _out = (0.0 + (dace::complex128(0.0, 1.0) * 0.0));
                        ///////////////////

                        big_result[(__i0 + ((__i1 * n) * npol))] = _out;
                    }
                }
            }
        }

    }
    big_result_d_d0 = (n * npol);

    big_result_d_d1 = m;

    big_result_d_allocated = 1;
    {

        {
            #pragma omp parallel for
            for (auto __i0 = 0; __i0 < (n * npol); __i0 += 1) {
                for (auto __i1 = 0; __i1 < m; __i1 += 1) {
                    {
                        dace::complex128 _out;

                        ///////////////////
                        // Tasklet code (set_big_result_d)
                        _out = (0.0 + (dace::complex128(0.0, 1.0) * 0.0));
                        ///////////////////

                        big_result_d[(__i0 + ((__i1 * n) * npol))] = _out;
                    }
                }
            }
        }

    }
    rhoc_d_d0 = nrxxs;

    rhoc_d_d1 = jblock;

    rhoc_d_allocated = 1;

    vc_d_d0 = nrxxs;

    vc_d_d1 = jblock;

    vc_d_allocated = 1;

    rhoc_d0 = nrxxs;

    rhoc_d1 = jblock;

    rhoc_allocated = 1;

    vc_d0 = nrxxs;

    vc_d1 = jblock;

    vc_allocated = 1;

    loopend_125 = nibands[my_egrp_id];

    deexx = new dace::complex128 DACE_ALIGN(64)[((nkb * (ialloc - 1)) + nkb)];
    temppsic_d = new dace::complex128 DACE_ALIGN(64)[((nrxxs * (ialloc - 1)) + nrxxs)];
    temppsic_nc_d = new dace::complex128 DACE_ALIGN(64)[((((npol * nrxxs) * (ialloc - 1)) + (nrxxs * (npol - 1))) + nrxxs)];

    for (_loop_it_823 = 1; (_loop_it_823 < (loopend_125 + 1)); _loop_it_823 = (_loop_it_823 + 1)) {

        ibnd = ibands[((_loop_it_823 + (ibands_d0 * my_egrp_id)) - 1)];

        if_cond_129 = (((ibnd == 0) || (ibnd > m)) != true);


        if (if_cond_129) {

            if_cond_132 = okvan[0];


            if (if_cond_132) {

                loopend_134 = ((1 + nkb) - 1);


                for (_loop_it_824 = 1; (_loop_it_824 < (loopend_134 + 1)); _loop_it_824 = (_loop_it_824 + 1)) {
                    {

                        {
                            dace::complex128 _out_deexx;

                            ///////////////////
                            // Tasklet code (t_0)
                            _out_deexx = (0.0 + (dace::complex128(0.0, 1.0) * 0.0));
                            ///////////////////

                            deexx[((_loop_it_824 + (nkb * (_loop_it_823 - 1))) - 1)] = _out_deexx;
                        }

                    }

                }

                as_0 = (loopend_134 + 1);

            }


            if_cond_138 = noncolin[0];


            if (if_cond_138) {

                loopend_140 = ((1 + nrxxs) - 1);


                for (_loop_it_825 = 1; (_loop_it_825 < (loopend_140 + 1)); _loop_it_825 = (_loop_it_825 + 1)) {

                    loopend_143 = ((1 + npol) - 1);


                    for (_loop_it_826 = 1; (_loop_it_826 < (loopend_143 + 1)); _loop_it_826 = (_loop_it_826 + 1)) {
                        {

                            {
                                dace::complex128 _out_temppsic_nc_d;

                                ///////////////////
                                // Tasklet code (t_0)
                                _out_temppsic_nc_d = (0.0 + (dace::complex128(0.0, 1.0) * 0.0));
                                ///////////////////

                                temppsic_nc_d[(((_loop_it_825 + ((npol * nrxxs) * (_loop_it_823 - 1))) + (nrxxs * (_loop_it_826 - 1))) - 1)] = _out_temppsic_nc_d;
                            }

                        }

                    }

                    as_1 = (loopend_143 + 1);


                }

                as_0 = (loopend_140 + 1);

            } else {

                loopend_146 = ((1 + nrxxs) - 1);


                for (_loop_it_827 = 1; (_loop_it_827 < (loopend_146 + 1)); _loop_it_827 = (_loop_it_827 + 1)) {
                    {

                        {
                            dace::complex128 _out_temppsic_d;

                            ///////////////////
                            // Tasklet code (t_0)
                            _out_temppsic_d = (0.0 + (dace::complex128(0.0, 1.0) * 0.0));
                            ///////////////////

                            temppsic_d[((_loop_it_827 + (nrxxs * (_loop_it_823 - 1))) - 1)] = _out_temppsic_d;
                        }

                    }

                }

                as_0 = (loopend_146 + 1);

            }


            if_cond_150 = noncolin[0];


            if (if_cond_150) {

                for (_loop_it_828 = 1; (_loop_it_828 < (n + 1)); _loop_it_828 = (_loop_it_828 + 1)) {

                    igk_exx_d_at20 = igk_exx_d[((_loop_it_828 + (igk_exx_d_d0 * (current_k - 1))) - 1)];

                    dfftt__nl_at21 = dfftt__nl[(igk_exx_d_at20 - 1)];

                    {

                        {
                            dace::complex128 _in_psi_d_0 = psi_d[((_loop_it_828 + (psi_d_d0 * (_loop_it_823 - 1))) - 1)];
                            dace::complex128 _out_temppsic_nc_d;

                            ///////////////////
                            // Tasklet code (t_0)
                            _out_temppsic_nc_d = _in_psi_d_0;
                            ///////////////////

                            temppsic_nc_d[((dfftt__nl_at21 + ((npol * nrxxs) * (_loop_it_823 - 1))) - 1)] = _out_temppsic_nc_d;
                        }
                        {
                            dace::complex128 _in_psi_d_0 = psi_d[(((_loop_it_828 + npwx) + (psi_d_d0 * (_loop_it_823 - 1))) - 1)];
                            dace::complex128 _out_temppsic_nc_d;

                            ///////////////////
                            // Tasklet code (t_1)
                            _out_temppsic_nc_d = _in_psi_d_0;
                            ///////////////////

                            temppsic_nc_d[(((dfftt__nl_at21 + ((npol * nrxxs) * (_loop_it_823 - 1))) + nrxxs) - 1)] = _out_temppsic_nc_d;
                        }

                    }

                }

                ig = (n + 1);


                ig = ig;

                {

                    dft_nd_25_3_2(__state, &temppsic_nc_d[0], &temppsic_nc_d[0], ialloc, npol, nrxxs);

                }
                {

                    dft_nd_25_3_2(__state, &temppsic_nc_d[0], &temppsic_nc_d[0], ialloc, npol, nrxxs);

                }
            } else {

                for (_loop_it_829 = 1; (_loop_it_829 < (n + 1)); _loop_it_829 = (_loop_it_829 + 1)) {

                    igk_exx_d_at22 = igk_exx_d[((_loop_it_829 + (igk_exx_d_d0 * (current_k - 1))) - 1)];

                    dfftt__nl_at23 = dfftt__nl[(igk_exx_d_at22 - 1)];

                    {

                        {
                            dace::complex128 _in_psi_d_0 = psi_d[((_loop_it_829 + (psi_d_d0 * (_loop_it_823 - 1))) - 1)];
                            dace::complex128 _out_temppsic_d;

                            ///////////////////
                            // Tasklet code (t_0)
                            _out_temppsic_d = _in_psi_d_0;
                            ///////////////////

                            temppsic_d[((dfftt__nl_at23 + (nrxxs * (_loop_it_823 - 1))) - 1)] = _out_temppsic_d;
                        }

                    }

                }

                ig = (n + 1);


                ig = ig;

                {

                    dft_nd_29_3_2(__state, &temppsic_d[0], &temppsic_d[0], ialloc, nrxxs);

                }
            }

        }


    }

    ii = (loopend_125 + 1);


    ii = ii;

    if_cond_172 = noncolin[0];

    result_d = new dace::complex128 DACE_ALIGN(64)[((nrxxs * (ialloc - 1)) + nrxxs)];
    result_nc_d = new dace::complex128 DACE_ALIGN(64)[((((npol * nrxxs) * (ialloc - 1)) + (nrxxs * (npol - 1))) + nrxxs)];

    if (if_cond_172) {
        {

            {
                #pragma omp parallel for
                for (auto __i0 = 0; __i0 < nrxxs; __i0 += 1) {
                    for (auto __i1 = 0; __i1 < npol; __i1 += 1) {
                        for (auto __i2 = 0; __i2 < ialloc; __i2 += 1) {
                            {
                                dace::complex128 _out;

                                ///////////////////
                                // Tasklet code (set_result_nc_d)
                                _out = (0.0 + (dace::complex128(0.0, 1.0) * 0.0));
                                ///////////////////

                                result_nc_d[((__i0 + (__i1 * nrxxs)) + ((__i2 * npol) * nrxxs))] = _out;
                            }
                        }
                    }
                }
            }

        }
    } else {
        {

            {
                #pragma omp parallel for
                for (auto __i0 = 0; __i0 < nrxxs; __i0 += 1) {
                    for (auto __i1 = 0; __i1 < ialloc; __i1 += 1) {
                        {
                            dace::complex128 _out;

                            ///////////////////
                            // Tasklet code (set_result_d)
                            _out = (0.0 + (dace::complex128(0.0, 1.0) * 0.0));
                            ///////////////////

                            result_d[(__i0 + (__i1 * nrxxs))] = _out;
                        }
                    }
                }
            }

        }
    }


    psi_d_allocated = 0;
    {

        {
            #pragma omp parallel for
            for (auto __i0 = 0; __i0 < psi_d_d0; __i0 += 1) {
                for (auto __i1 = 0; __i1 < psi_d_d1; __i1 += 1) {
                    {
                        dace::complex128 _out;

                        ///////////////////
                        // Tasklet code (set_psi_d)
                        _out = 0;
                        ///////////////////

                        psi_d[(__i0 + (__i1 * psi_d_d0))] = _out;
                    }
                }
            }
        }
        {
            double _in_omega = omega[0];
            double _out;

            ///////////////////
            // Tasklet code (set_omega_inv)
            _out = (1.0 / _in_omega);
            ///////////////////

            omega_inv = _out;
        }
        {
            double _out;

            ///////////////////
            // Tasklet code (set_nqs_inv)
            _out = dace::float32((dace::float32(1.0) / dace::float32(nqs)));
            ///////////////////

            nqs_inv = _out;
        }

    }
    facb = new double DACE_ALIGN(64)[nrxxs];
    facb_d = new double DACE_ALIGN(64)[nrxxs];
    rhoc = new dace::complex128 DACE_ALIGN(64)[((nrxxs * (jblock - 1)) + nrxxs)];
    rhoc_d = new dace::complex128 DACE_ALIGN(64)[((nrxxs * (jblock - 1)) + nrxxs)];
    vc = new dace::complex128 DACE_ALIGN(64)[((nrxxs * (jblock - 1)) + nrxxs)];
    vc_d = new dace::complex128 DACE_ALIGN(64)[((nrxxs * (jblock - 1)) + nrxxs)];

    for (_loop_it_830 = 1; (_loop_it_830 < (nqs + 1)); _loop_it_830 = (_loop_it_830 + 1)) {

        ikq = index_xkq[((current_ik + (index_xkq_d0 * (_loop_it_830 - 1))) - 1)];

        ik = index_xk[(ikq - 1)];

        loopend_182 = ((1 + xkq_collect_d0) - 1);


        for (_loop_it_831 = 1; (_loop_it_831 < (loopend_182 + 1)); _loop_it_831 = (_loop_it_831 + 1)) {
            {

                {
                    double _in_xkq_collect_0 = xkq_collect[((_loop_it_831 + (xkq_collect_d0 * (ikq - 1))) - 1)];
                    double _out_xkq;

                    ///////////////////
                    // Tasklet code (t_0)
                    _out_xkq = _in_xkq_collect_0;
                    ///////////////////

                    xkq[(_loop_it_831 - 1)] = _out_xkq;
                }

            }

        }

        ss_0 = (loopend_182 + 1);


        {

            {
                double* _in_coulomb_fac_0 = &coulomb_fac[0];
                int64_t _out_if_cond_185;

                ///////////////////
                // Tasklet code (t_186)
                _out_if_cond_185 = (_in_coulomb_fac_0 == 0);
                ///////////////////

                if_cond_185 = _out_if_cond_185;
            }

        }

        if (if_cond_185) {

            coulomb_fac_d1 = nqs;

            coulomb_fac_d2 = nks;

            coulomb_fac_allocated = 1;

        }


        {

            {
                bool* _in_coulomb_done_0 = &coulomb_done[0];
                int64_t _out_if_cond_193;

                ///////////////////
                // Tasklet code (t_194)
                _out_if_cond_193 = (_in_coulomb_done_0 == 0);
                ///////////////////

                if_cond_193 = _out_if_cond_193;
            }

        }

        if (if_cond_193) {

            coulomb_done_d0 = nqs;

            coulomb_done_d1 = nks;

            coulomb_done_allocated = 1;
            {

                {
                    bool* _mset_out = coulomb_done;

                    ///////////////////
                    memset(_mset_out, 0, (nks * nqs) * sizeof(bool));
                    ///////////////////

                }

            }

        }


        {

            {
                bool _in_coulomb_done_0 = coulomb_done[((_loop_it_830 + (nqs * (current_k - 1))) - 1)];
                int64_t _out_if_cond_203;

                ///////////////////
                // Tasklet code (t_204)
                _out_if_cond_203 = _in_coulomb_done_0;
                ///////////////////

                if_cond_203 = _out_if_cond_203;
            }

        }

        if (if_cond_203) {

        } else {

            if_cond_207 = use_coulomb_vcut_ws[0];


            if (if_cond_207) {

                for (_loop_it_832 = 1; (_loop_it_832 < (__sym_dfftt_ngm_1 + 1)); _loop_it_832 = (_loop_it_832 + 1)) {


                    for (_loop_it_833 = 1; (_loop_it_833 < (3 + 1)); _loop_it_833 = (_loop_it_833 + 1)) {
                        {

                            {
                                double _in_gt_0 = gt[((_loop_it_833 + (gt_d0 * (_loop_it_832 - 1))) - 1)];
                                double _in_xkp_0 = xkp[(_loop_it_833 - 1)];
                                double _in_xkq_0 = xkq[(_loop_it_833 - 1)];
                                double _in_tpiba = tpiba[0];
                                double _out_g2_convolution_q;

                                ///////////////////
                                // Tasklet code (t_0)
                                _out_g2_convolution_q = (((_in_xkp_0 - _in_xkq_0) + _in_gt_0) * _in_tpiba);
                                ///////////////////

                                g2_convolution_q[(_loop_it_833 - 1)] = _out_g2_convolution_q;
                            }

                        }

                    }

                    ei0 = 4;

                    {

                        matmul__libtmp_0_213gemv_sdfg_46_2_3(__state, &vcut_a[0], &g2_convolution_q[0], &_libtmp_0[0]);

                    }

                    for (_loop_it_834 = 1; (_loop_it_834 < (3 + 1)); _loop_it_834 = (_loop_it_834 + 1)) {
                        {

                            {
                                double _in__libtmp_0_0 = _libtmp_0[(_loop_it_834 - 1)];
                                double _out_i_real;

                                ///////////////////
                                // Tasklet code (t_0)
                                _out_i_real = (_in__libtmp_0_0 / 6.283185307179586);
                                ///////////////////

                                i_real[(_loop_it_834 - 1)] = _out_i_real;
                            }

                        }

                    }

                    ei0 = 4;


                    for (_loop_it_835 = 1; (_loop_it_835 < (3 + 1)); _loop_it_835 = (_loop_it_835 + 1)) {
                        {

                            {
                                double _in_i_real_0 = i_real[(_loop_it_835 - 1)];
                                int _out_i;

                                ///////////////////
                                // Tasklet code (t_0)
                                _out_i = dace::int32(round(_in_i_real_0));
                                ///////////////////

                                i[(_loop_it_835 - 1)] = _out_i;
                            }

                        }

                    }

                    ei0 = 4;


                    for (_loop_it_836 = 1; (_loop_it_836 < (3 + 1)); _loop_it_836 = (_loop_it_836 + 1)) {
                        {

                            {
                                int _in_i_0 = i[(_loop_it_836 - 1)];
                                double _in_i_real_0 = i_real[(_loop_it_836 - 1)];
                                double _out__libsrc_0;

                                ///////////////////
                                // Tasklet code (t_0)
                                _out__libsrc_0 = (dace::math::ipow((dace::float64(_in_i_0) - _in_i_real_0), 2));
                                ///////////////////

                                _libsrc_0[(_loop_it_836 - 1)] = _out__libsrc_0;
                            }

                        }

                    }

                    li0 = 4;

                    {

                        reduce_46_6_2(__state, &_libsrc_0[0], &__reduce_cond_1);

                    }
                    if_cond_218 = (__reduce_cond_1 > 1e-06);


                    if (if_cond_218) {
                        {
                            int __assoc_scalar_0;

                            {
                                int _out;

                                ///////////////////
                                // Tasklet code (set___assoc_scalar_0)
                                _out = 10;
                                ///////////////////

                                __assoc_scalar_0 = _out;
                            }

                        }
                    }


                    for (_loop_it_837 = 1; (_loop_it_837 < (3 + 1)); _loop_it_837 = (_loop_it_837 + 1)) {
                        {

                            {
                                double _in_g2_convolution_q_0 = g2_convolution_q[(_loop_it_837 - 1)];
                                double _out__libsrc_2;

                                ///////////////////
                                // Tasklet code (t_0)
                                _out__libsrc_2 = (dace::math::ipow(_in_g2_convolution_q_0, 2));
                                ///////////////////

                                _libsrc_2[(_loop_it_837 - 1)] = _out__libsrc_2;
                            }

                        }

                    }

                    li0 = 4;

                    {

                        reduce_46_6_2(__state, &_libsrc_2[0], &__reduce_cond_3);

                    }
                    if_cond_223 = (__reduce_cond_3 > (dace::math::ipow(vcut_cutoff, 2)));


                    if (if_cond_223) {

                        for (_loop_it_838 = 1; (_loop_it_838 < (3 + 1)); _loop_it_838 = (_loop_it_838 + 1)) {
                            {

                                {
                                    double _in_g2_convolution_q_0 = g2_convolution_q[(_loop_it_838 - 1)];
                                    double _out__mask_4;

                                    ///////////////////
                                    // Tasklet code (t_0)
                                    _out__mask_4 = (dace::math::ipow(_in_g2_convolution_q_0, 2));
                                    ///////////////////

                                    _mask_4[(_loop_it_838 - 1)] = _out__mask_4;
                                }

                            }

                        }

                        ei0 = 4;

                        {
                            double _QQred_lift_0;

                            reduce_46_6_2(__state, &_mask_4[0], &_QQred_lift_0);
                            {
                                double _in__QQred_lift_0 = _QQred_lift_0;
                                double _out;

                                ///////////////////
                                // Tasklet code (set_vcut_get_res)
                                _out = (25.132741228718345 / _in__QQred_lift_0);
                                ///////////////////

                                vcut_get_res = _out;
                            }

                        }
                    } else {

                        if_cond_228 = ((((((i[0] > ((vcut_corrected_d0 + 1) - 1)) || (i[0] < 1)) || (i[1] > ((vcut_corrected_d1 + 1) - 1))) || (i[1] < 1)) || (i[2] > ((vcut_corrected_d2 + 1) - 1))) || (i[2] < 1));


                        if (if_cond_228) {
                            {
                                int __assoc_scalar_1;

                                {
                                    int _out;

                                    ///////////////////
                                    // Tasklet code (set___assoc_scalar_1)
                                    _out = 10;
                                    ///////////////////

                                    __assoc_scalar_1 = _out;
                                }

                            }
                        }


                        i_at24 = i[0];

                        i_at25 = i[1];

                        i_at26 = i[2];
                        {

                            {
                                double _in_vcut_corrected_0 = vcut_corrected[(((i_at24 + ((vcut_corrected_d0 * vcut_corrected_d1) * (i_at26 - 1))) + (vcut_corrected_d0 * (i_at25 - 1))) - 1)];
                                double _out_vcut_get_res;

                                ///////////////////
                                // Tasklet code (t_235)
                                _out_vcut_get_res = _in_vcut_corrected_0;
                                ///////////////////

                                vcut_get_res = _out_vcut_get_res;
                            }

                        }
                    }

                    {

                        {
                            double _in_vcut_get_res = vcut_get_res;
                            double _out_g2_convolution_fac;

                            ///////////////////
                            // Tasklet code (t_237)
                            _out_g2_convolution_fac = _in_vcut_get_res;
                            ///////////////////

                            coulomb_fac[(((_loop_it_832 + ((dfftt_ngm * nqs) * (current_k - 1))) + (dfftt_ngm * (_loop_it_830 - 1))) - 1)] = _out_g2_convolution_fac;
                        }

                    }

                }

                g2_convolution_ig = (__sym_dfftt_ngm_1 + 1);


                g2_convolution_ig = g2_convolution_ig;

            } else {

                if_cond_241 = use_coulomb_vcut_spheric[0];


                if (if_cond_241) {

                    for (_loop_it_839 = 1; (_loop_it_839 < (__sym_dfftt_ngm_1 + 1)); _loop_it_839 = (_loop_it_839 + 1)) {


                        for (_loop_it_840 = 1; (_loop_it_840 < (3 + 1)); _loop_it_840 = (_loop_it_840 + 1)) {
                            {

                                {
                                    double _in_gt_0 = gt[((_loop_it_840 + (gt_d0 * (_loop_it_839 - 1))) - 1)];
                                    double _in_xkp_0 = xkp[(_loop_it_840 - 1)];
                                    double _in_xkq_0 = xkq[(_loop_it_840 - 1)];
                                    double _in_tpiba = tpiba[0];
                                    double _out_g2_convolution_q;

                                    ///////////////////
                                    // Tasklet code (t_0)
                                    _out_g2_convolution_q = (((_in_xkp_0 - _in_xkq_0) + _in_gt_0) * _in_tpiba);
                                    ///////////////////

                                    g2_convolution_q[(_loop_it_840 - 1)] = _out_g2_convolution_q;
                                }

                            }

                        }

                        ei0 = 4;

                        {

                            copy_a_247_sdfg_67_2_2(__state, &vcut_a[0], &a[0]);

                        }

                        for (_loop_it_841 = 1; (_loop_it_841 < (3 + 1)); _loop_it_841 = (_loop_it_841 + 1)) {

                            for (_loop_it_842 = 1; (_loop_it_842 < (3 + 1)); _loop_it_842 = (_loop_it_842 + 1)) {
                                {

                                    {
                                        double _in_a_0 = a[((_loop_it_841 + (3 * _loop_it_842)) - 4)];
                                        double _out__libsrc_6;

                                        ///////////////////
                                        // Tasklet code (t_0)
                                        _out__libsrc_6 = (dace::math::ipow(_in_a_0, 2));
                                        ///////////////////

                                        _libsrc_6[((_loop_it_841 + (3 * _loop_it_842)) - 4)] = _out__libsrc_6;
                                    }

                                }

                            }

                            li1 = 4;


                        }

                        li0 = 4;

                        {

                            reduce_67_4_2(__state, &_libsrc_6[0], &_libtmp_1[0]);

                        }

                        for (_loop_it_843 = 1; (_loop_it_843 < (3 + 1)); _loop_it_843 = (_loop_it_843 + 1)) {
                            {

                                {
                                    double _in__libtmp_1_0 = _libtmp_1[(_loop_it_843 - 1)];
                                    double _out__mask_5;

                                    ///////////////////
                                    // Tasklet code (t_0)
                                    _out__mask_5 = sqrt(_in__libtmp_1_0);
                                    ///////////////////

                                    _mask_5[(_loop_it_843 - 1)] = _out__mask_5;
                                }

                            }

                        }

                        ei0 = 4;

                        {
                            double _QQred_lift_1;

                            reduce_67_6_8(__state, &_mask_5[0], &_QQred_lift_1);
                            {
                                double _in__QQred_lift_1 = _QQred_lift_1;
                                double _out;

                                ///////////////////
                                // Tasklet code (set_rcut)
                                _out = (_in__QQred_lift_1 * 0.5);
                                ///////////////////

                                rcut = _out;
                            }
                            {
                                double _in_rcut = rcut;
                                double _out;

                                ///////////////////
                                // Tasklet code (set_rcut)
                                _out = (_in_rcut - (_in_rcut / 50.0));
                                ///////////////////

                                rcut = _out;
                            }
                            {
                                bool _out;

                                ///////////////////
                                // Tasklet code (set_limit)
                                _out = 0;
                                ///////////////////

                                program_limit = _out;
                            }

                        }

                        for (_loop_it_844 = 1; (_loop_it_844 < (3 + 1)); _loop_it_844 = (_loop_it_844 + 1)) {
                            {

                                {
                                    double _in_g2_convolution_q_0 = g2_convolution_q[(_loop_it_844 - 1)];
                                    double _out__mask_7;

                                    ///////////////////
                                    // Tasklet code (t_0)
                                    _out__mask_7 = (dace::math::ipow(_in_g2_convolution_q_0, 2));
                                    ///////////////////

                                    _mask_7[(_loop_it_844 - 1)] = _out__mask_7;
                                }

                            }

                        }

                        ei0 = 4;

                        {

                            reduce_46_6_2(__state, &_mask_7[0], &kg2);

                        }
                        if_cond_255 = (kg2 < 1e-06);


                        if (if_cond_255) {
                            {

                                {
                                    bool _out;

                                    ///////////////////
                                    // Tasklet code (set_limit)
                                    _out = -1;
                                    ///////////////////

                                    program_limit = _out;
                                }

                            }
                        }


                        if_cond_259 = (program_limit != true);


                        if (if_cond_259) {
                            {

                                {
                                    double _in_rcut = rcut;
                                    double _in_kg2 = kg2;
                                    double _out;

                                    ///////////////////
                                    // Tasklet code (set_vcut_spheric_get_res)
                                    _out = ((25.132741228718345 / _in_kg2) * (1.0 - cos((_in_rcut * sqrt(_in_kg2)))));
                                    ///////////////////

                                    vcut_spheric_get_res = _out;
                                }

                            }
                        } else {
                            {

                                {
                                    double _in_rcut = rcut;
                                    double _out;

                                    ///////////////////
                                    // Tasklet code (set_vcut_spheric_get_res)
                                    _out = (((dace::math::ipow(_in_rcut, 2)) * 25.132741228718345) / 2.0);
                                    ///////////////////

                                    vcut_spheric_get_res = _out;
                                }

                            }
                        }

                        {

                            {
                                double _in_vcut_spheric_get_res = vcut_spheric_get_res;
                                double _out_g2_convolution_fac;

                                ///////////////////
                                // Tasklet code (t_264)
                                _out_g2_convolution_fac = _in_vcut_spheric_get_res;
                                ///////////////////

                                coulomb_fac[(((_loop_it_839 + ((dfftt_ngm * nqs) * (current_k - 1))) + (dfftt_ngm * (_loop_it_830 - 1))) - 1)] = _out_g2_convolution_fac;
                            }

                        }

                    }

                    g2_convolution_ig = (__sym_dfftt_ngm_1 + 1);


                    g2_convolution_ig = g2_convolution_ig;

                } else {
                    {

                        {
                            int _in_nq1 = nq1[0];
                            double _out_nqhalf_dble;

                            ///////////////////
                            // Tasklet code (t_268)
                            _out_nqhalf_dble = (dace::float64(_in_nq1) * 0.5);
                            ///////////////////

                            nqhalf_dble[0] = _out_nqhalf_dble;
                        }

                    }
                    {

                        {
                            int _in_nq2 = nq2[0];
                            double _out_nqhalf_dble;

                            ///////////////////
                            // Tasklet code (t_270)
                            _out_nqhalf_dble = (dace::float64(_in_nq2) * 0.5);
                            ///////////////////

                            nqhalf_dble[1] = _out_nqhalf_dble;
                        }

                    }
                    {

                        {
                            int _in_nq3 = nq3[0];
                            double _out_nqhalf_dble;

                            ///////////////////
                            // Tasklet code (t_272)
                            _out_nqhalf_dble = (dace::float64(_in_nq3) * 0.5);
                            ///////////////////

                            nqhalf_dble[2] = _out_nqhalf_dble;
                        }

                    }
                    if_cond_273 = x_gamma_extrapolation[0];


                    if (if_cond_273) {

                        for (_loop_it_845 = 1; (_loop_it_845 < (__sym_dfftt_ngm_1 + 1)); _loop_it_845 = (_loop_it_845 + 1)) {


                            for (_loop_it_846 = 1; (_loop_it_846 < (3 + 1)); _loop_it_846 = (_loop_it_846 + 1)) {
                                {

                                    {
                                        double _in_gt_0 = gt[((_loop_it_846 + (gt_d0 * (_loop_it_845 - 1))) - 1)];
                                        double _in_xkp_0 = xkp[(_loop_it_846 - 1)];
                                        double _in_xkq_0 = xkq[(_loop_it_846 - 1)];
                                        double _out_g2_convolution_q;

                                        ///////////////////
                                        // Tasklet code (t_0)
                                        _out_g2_convolution_q = ((_in_xkp_0 - _in_xkq_0) + _in_gt_0);
                                        ///////////////////

                                        g2_convolution_q[(_loop_it_846 - 1)] = _out_g2_convolution_q;
                                    }

                                }

                            }

                            ei0 = 4;


                            for (_loop_it_847 = 1; (_loop_it_847 < (3 + 1)); _loop_it_847 = (_loop_it_847 + 1)) {
                                {

                                    {
                                        double _in_g2_convolution_q_0 = g2_convolution_q[(_loop_it_847 - 1)];
                                        double _out__mask_8;

                                        ///////////////////
                                        // Tasklet code (t_0)
                                        _out__mask_8 = (dace::math::ipow(_in_g2_convolution_q_0, 2));
                                        ///////////////////

                                        _mask_8[(_loop_it_847 - 1)] = _out__mask_8;
                                    }

                                }

                            }

                            ei0 = 4;

                            {
                                double _QQred_lift_2;

                                reduce_46_6_2(__state, &_mask_8[0], &_QQred_lift_2);
                                {
                                    double _in__QQred_lift_2 = _QQred_lift_2;
                                    double _in_tpiba2 = tpiba2[0];
                                    double _out_qq_track;

                                    ///////////////////
                                    // Tasklet code (t_280)
                                    _out_qq_track = (_in__QQred_lift_2 * _in_tpiba2);
                                    ///////////////////

                                    qq_track[(_loop_it_845 - 1)] = _out_qq_track;
                                }
                                {
                                    double _in_at_0 = at[0];
                                    double _in_at_1 = at[1];
                                    double _in_at_2 = at[2];
                                    double _in_g2_convolution_q_0 = g2_convolution_q[0];
                                    double _in_g2_convolution_q_1 = g2_convolution_q[1];
                                    double _in_g2_convolution_q_2 = g2_convolution_q[2];
                                    double _in_nqhalf_dble_0 = nqhalf_dble[0];
                                    double _out_x;

                                    ///////////////////
                                    // Tasklet code (t_281)
                                    _out_x = ((((_in_g2_convolution_q_0 * _in_at_0) + (_in_g2_convolution_q_1 * _in_at_1)) + (_in_g2_convolution_q_2 * _in_at_2)) * _in_nqhalf_dble_0);
                                    ///////////////////

                                    x = _out_x;
                                }
                                {
                                    double _in_eps = eps[0];
                                    double _in_x = x;
                                    bool _out_odg;

                                    ///////////////////
                                    // Tasklet code (t_282)
                                    _out_odg = (abs((_in_x - dace::int32(round(_in_x)))) < _in_eps);
                                    ///////////////////

                                    odg[0] = _out_odg;
                                }

                            }
                            {

                                {
                                    double _in_at_0 = at[3];
                                    double _in_at_1 = at[4];
                                    double _in_at_2 = at[5];
                                    double _in_g2_convolution_q_0 = g2_convolution_q[0];
                                    double _in_g2_convolution_q_1 = g2_convolution_q[1];
                                    double _in_g2_convolution_q_2 = g2_convolution_q[2];
                                    double _in_nqhalf_dble_0 = nqhalf_dble[1];
                                    double _out_x;

                                    ///////////////////
                                    // Tasklet code (t_284)
                                    _out_x = ((((_in_g2_convolution_q_0 * _in_at_0) + (_in_g2_convolution_q_1 * _in_at_1)) + (_in_g2_convolution_q_2 * _in_at_2)) * _in_nqhalf_dble_0);
                                    ///////////////////

                                    x = _out_x;
                                }
                                {
                                    double _in_eps = eps[0];
                                    double _in_x = x;
                                    bool _out_odg;

                                    ///////////////////
                                    // Tasklet code (t_285)
                                    _out_odg = (abs((_in_x - dace::int32(round(_in_x)))) < _in_eps);
                                    ///////////////////

                                    odg[1] = _out_odg;
                                }

                            }
                            {

                                {
                                    double _in_at_0 = at[6];
                                    double _in_at_1 = at[7];
                                    double _in_at_2 = at[8];
                                    double _in_g2_convolution_q_0 = g2_convolution_q[0];
                                    double _in_g2_convolution_q_1 = g2_convolution_q[1];
                                    double _in_g2_convolution_q_2 = g2_convolution_q[2];
                                    double _in_nqhalf_dble_0 = nqhalf_dble[2];
                                    double _out_x;

                                    ///////////////////
                                    // Tasklet code (t_287)
                                    _out_x = ((((_in_g2_convolution_q_0 * _in_at_0) + (_in_g2_convolution_q_1 * _in_at_1)) + (_in_g2_convolution_q_2 * _in_at_2)) * _in_nqhalf_dble_0);
                                    ///////////////////

                                    x = _out_x;
                                }
                                {
                                    double _in_eps = eps[0];
                                    double _in_x = x;
                                    bool _out_odg;

                                    ///////////////////
                                    // Tasklet code (t_288)
                                    _out_odg = (abs((_in_x - dace::int32(round(_in_x)))) < _in_eps);
                                    ///////////////////

                                    odg[2] = _out_odg;
                                }
                                dace_libraries_standard_nodes_allany_kernel_85_5_9(__state, &odg[0], __allany_cond_9);

                            }

                            if (__allany_cond_9) {
                                {

                                    {
                                        double _out_grid_factor_track;

                                        ///////////////////
                                        // Tasklet code (t_292)
                                        _out_grid_factor_track = 0.0;
                                        ///////////////////

                                        grid_factor_track[(_loop_it_845 - 1)] = _out_grid_factor_track;
                                    }

                                }
                            } else {
                                {

                                    {
                                        double _in_grid_factor = grid_factor[0];
                                        double _out_grid_factor_track;

                                        ///////////////////
                                        // Tasklet code (t_294)
                                        _out_grid_factor_track = _in_grid_factor;
                                        ///////////////////

                                        grid_factor_track[(_loop_it_845 - 1)] = _out_grid_factor_track;
                                    }

                                }
                            }


                        }

                        g2_convolution_ig = (__sym_dfftt_ngm_1 + 1);


                        g2_convolution_ig = g2_convolution_ig;

                    } else {

                        for (_loop_it_848 = 1; (_loop_it_848 < (__sym_dfftt_ngm_1 + 1)); _loop_it_848 = (_loop_it_848 + 1)) {


                            for (_loop_it_849 = 1; (_loop_it_849 < (3 + 1)); _loop_it_849 = (_loop_it_849 + 1)) {
                                {

                                    {
                                        double _in_gt_0 = gt[((_loop_it_849 + (gt_d0 * (_loop_it_848 - 1))) - 1)];
                                        double _in_xkp_0 = xkp[(_loop_it_849 - 1)];
                                        double _in_xkq_0 = xkq[(_loop_it_849 - 1)];
                                        double _out_g2_convolution_q;

                                        ///////////////////
                                        // Tasklet code (t_0)
                                        _out_g2_convolution_q = ((_in_xkp_0 - _in_xkq_0) + _in_gt_0);
                                        ///////////////////

                                        g2_convolution_q[(_loop_it_849 - 1)] = _out_g2_convolution_q;
                                    }

                                }

                            }

                            ei0 = 4;


                            for (_loop_it_850 = 1; (_loop_it_850 < (3 + 1)); _loop_it_850 = (_loop_it_850 + 1)) {
                                {

                                    {
                                        double _in_g2_convolution_q_0 = g2_convolution_q[(_loop_it_850 - 1)];
                                        double _out__mask_10;

                                        ///////////////////
                                        // Tasklet code (t_0)
                                        _out__mask_10 = (dace::math::ipow(_in_g2_convolution_q_0, 2));
                                        ///////////////////

                                        _mask_10[(_loop_it_850 - 1)] = _out__mask_10;
                                    }

                                }

                            }

                            ei0 = 4;

                            {
                                double _QQred_lift_3;

                                reduce_46_6_2(__state, &_mask_10[0], &_QQred_lift_3);
                                {
                                    double _in__QQred_lift_3 = _QQred_lift_3;
                                    double _in_tpiba2 = tpiba2[0];
                                    double _out_qq_track;

                                    ///////////////////
                                    // Tasklet code (t_302)
                                    _out_qq_track = (_in__QQred_lift_3 * _in_tpiba2);
                                    ///////////////////

                                    qq_track[(_loop_it_848 - 1)] = _out_qq_track;
                                }

                            }

                        }

                        g2_convolution_ig = (__sym_dfftt_ngm_1 + 1);


                        g2_convolution_ig = g2_convolution_ig;


                        for (_loop_it_851 = 1; (_loop_it_851 < (dfftt_ngm + 1)); _loop_it_851 = (_loop_it_851 + 1)) {
                            {

                                {
                                    double _out_grid_factor_track;

                                    ///////////////////
                                    // Tasklet code (t_0)
                                    _out_grid_factor_track = 1.0;
                                    ///////////////////

                                    grid_factor_track[(_loop_it_851 - 1)] = _out_grid_factor_track;
                                }

                            }

                        }

                        ab_0 = (dfftt_ngm + 1);

                    }


                    for (_loop_it_852 = 1; (_loop_it_852 < (__sym_dfftt_ngm_1 + 1)); _loop_it_852 = (_loop_it_852 + 1)) {
                        {

                            {
                                double _in_qq_track_0 = qq_track[(_loop_it_852 - 1)];
                                double _out_g2_convolution_qq;

                                ///////////////////
                                // Tasklet code (t_308)
                                _out_g2_convolution_qq = _in_qq_track_0;
                                ///////////////////

                                g2_convolution_qq = _out_g2_convolution_qq;
                            }

                        }
                        if_cond_309 = (gau_scrlen[0] > 0.0);


                        if (if_cond_309) {
                            {

                                {
                                    double _in_grid_factor_track_0 = grid_factor_track[(_loop_it_852 - 1)];
                                    double _in_g2_convolution_qq = g2_convolution_qq;
                                    double _in_gau_scrlen = gau_scrlen[0];
                                    double _out_g2_convolution_fac;

                                    ///////////////////
                                    // Tasklet code (t_312)
                                    _out_g2_convolution_fac = (((dace::math::pow((3.141592653589793 / _in_gau_scrlen), 1.5) * 2.0) * exp((- ((_in_g2_convolution_qq / 4.0) / _in_gau_scrlen)))) * _in_grid_factor_track_0);
                                    ///////////////////

                                    coulomb_fac[(((_loop_it_852 + ((dfftt_ngm * nqs) * (current_k - 1))) + (dfftt_ngm * (_loop_it_830 - 1))) - 1)] = _out_g2_convolution_fac;
                                }

                            }
                        } else {

                            if_cond_314 = (g2_convolution_qq > eps_qdiv[0]);


                            if (if_cond_314) {

                                if_cond_317 = (erfc_scrlen[0] > 0.0);


                                if (if_cond_317) {
                                    {

                                        {
                                            double _in_grid_factor_track_0 = grid_factor_track[(_loop_it_852 - 1)];
                                            double _in_erfc_scrlen = erfc_scrlen[0];
                                            double _in_g2_convolution_qq = g2_convolution_qq;
                                            double _out_g2_convolution_fac;

                                            ///////////////////
                                            // Tasklet code (t_320)
                                            _out_g2_convolution_fac = (((25.132741228718345 / _in_g2_convolution_qq) * (1.0 - exp((- ((_in_g2_convolution_qq / 4.0) / (dace::math::ipow(_in_erfc_scrlen, 2))))))) * _in_grid_factor_track_0);
                                            ///////////////////

                                            coulomb_fac[(((_loop_it_852 + ((dfftt_ngm * nqs) * (current_k - 1))) + (dfftt_ngm * (_loop_it_830 - 1))) - 1)] = _out_g2_convolution_fac;
                                        }

                                    }
                                } else {

                                    if_cond_322 = (erf_scrlen[0] > 0.0);


                                    if (if_cond_322) {
                                        {

                                            {
                                                double _in_grid_factor_track_0 = grid_factor_track[(_loop_it_852 - 1)];
                                                double _in_erf_scrlen = erf_scrlen[0];
                                                double _in_g2_convolution_qq = g2_convolution_qq;
                                                double _out_g2_convolution_fac;

                                                ///////////////////
                                                // Tasklet code (t_325)
                                                _out_g2_convolution_fac = (((25.132741228718345 / _in_g2_convolution_qq) * exp((- ((_in_g2_convolution_qq / 4.0) / (dace::math::ipow(_in_erf_scrlen, 2)))))) * _in_grid_factor_track_0);
                                                ///////////////////

                                                coulomb_fac[(((_loop_it_852 + ((dfftt_ngm * nqs) * (current_k - 1))) + (dfftt_ngm * (_loop_it_830 - 1))) - 1)] = _out_g2_convolution_fac;
                                            }

                                        }
                                    } else {
                                        {

                                            {
                                                double _in_grid_factor_track_0 = grid_factor_track[(_loop_it_852 - 1)];
                                                double _in_g2_convolution_qq = g2_convolution_qq;
                                                double _in_yukawa = yukawa[0];
                                                double _out_g2_convolution_fac;

                                                ///////////////////
                                                // Tasklet code (t_327)
                                                _out_g2_convolution_fac = ((25.132741228718345 / (_in_g2_convolution_qq + _in_yukawa)) * _in_grid_factor_track_0);
                                                ///////////////////

                                                coulomb_fac[(((_loop_it_852 + ((dfftt_ngm * nqs) * (current_k - 1))) + (dfftt_ngm * (_loop_it_830 - 1))) - 1)] = _out_g2_convolution_fac;
                                            }

                                        }
                                    }

                                }

                            } else {
                                {

                                    {
                                        double _in_exxdiv = exxdiv[0];
                                        double _out_g2_convolution_fac;

                                        ///////////////////
                                        // Tasklet code (t_329)
                                        _out_g2_convolution_fac = (- _in_exxdiv);
                                        ///////////////////

                                        coulomb_fac[(((_loop_it_852 + ((dfftt_ngm * nqs) * (current_k - 1))) + (dfftt_ngm * (_loop_it_830 - 1))) - 1)] = _out_g2_convolution_fac;
                                    }

                                }
                                if_cond_330 = ((yukawa[0] > 0.0) && (x_gamma_extrapolation[0] != true));


                                if (if_cond_330) {
                                    {

                                        {
                                            double _in_g2_convolution_fac_0 = coulomb_fac[(((_loop_it_852 + ((dfftt_ngm * nqs) * (current_k - 1))) + (dfftt_ngm * (_loop_it_830 - 1))) - 1)];
                                            double _in_g2_convolution_qq = g2_convolution_qq;
                                            double _in_yukawa = yukawa[0];
                                            double _out_g2_convolution_fac;

                                            ///////////////////
                                            // Tasklet code (t_333)
                                            _out_g2_convolution_fac = (_in_g2_convolution_fac_0 + (25.132741228718345 / (_in_g2_convolution_qq + _in_yukawa)));
                                            ///////////////////

                                            coulomb_fac[(((_loop_it_852 + ((dfftt_ngm * nqs) * (current_k - 1))) + (dfftt_ngm * (_loop_it_830 - 1))) - 1)] = _out_g2_convolution_fac;
                                        }

                                    }
                                }


                                if_cond_335 = ((erfc_scrlen[0] > 0.0) && (x_gamma_extrapolation[0] != true));


                                if (if_cond_335) {
                                    {

                                        {
                                            double _in_g2_convolution_fac_0 = coulomb_fac[(((_loop_it_852 + ((dfftt_ngm * nqs) * (current_k - 1))) + (dfftt_ngm * (_loop_it_830 - 1))) - 1)];
                                            double _in_erfc_scrlen = erfc_scrlen[0];
                                            double _out_g2_convolution_fac;

                                            ///////////////////
                                            // Tasklet code (t_338)
                                            _out_g2_convolution_fac = (_in_g2_convolution_fac_0 + (6.283185307179586 / (dace::math::ipow(_in_erfc_scrlen, 2))));
                                            ///////////////////

                                            coulomb_fac[(((_loop_it_852 + ((dfftt_ngm * nqs) * (current_k - 1))) + (dfftt_ngm * (_loop_it_830 - 1))) - 1)] = _out_g2_convolution_fac;
                                        }

                                    }
                                }

                            }

                        }


                    }

                    g2_convolution_ig = (__sym_dfftt_ngm_1 + 1);


                    g2_convolution_ig = g2_convolution_ig;

                }

            }

            {

                {
                    bool _out_coulomb_done;

                    ///////////////////
                    // Tasklet code (t_342)
                    _out_coulomb_done = -1;
                    ///////////////////

                    coulomb_done[((_loop_it_830 + (nqs * (current_k - 1))) - 1)] = _out_coulomb_done;
                }

            }
        }

        {

            {
                double* _mset_out = facb;

                ///////////////////
                memset(_mset_out, 0, nrxxs * sizeof(double));
                ///////////////////

            }

        }


        for (_loop_it_853 = 1; (_loop_it_853 < (__sym_dfftt_ngm_1 + 1)); _loop_it_853 = (_loop_it_853 + 1)) {

            dfftt__nl_at27 = dfftt__nl[(_loop_it_853 - 1)];

            {

                {
                    double _in_coulomb_fac_0 = coulomb_fac[(((_loop_it_853 + ((dfftt_ngm * nqs) * (current_k - 1))) + (dfftt_ngm * (_loop_it_830 - 1))) - 1)];
                    double _out_facb;

                    ///////////////////
                    // Tasklet code (t_0)
                    _out_facb = _in_coulomb_fac_0;
                    ///////////////////

                    facb[(dfftt__nl_at27 - 1)] = _out_facb;
                }

            }

        }

        ig = (__sym_dfftt_ngm_1 + 1);


        ig = ig;
        {

            copy_facb_d_351_sdfg_35_18_2(__state, &facb[0], &facb_d[0], nrxxs);

        }
        if_cond_352 = (okvan[0] && (tqr[0] != true));


        if (if_cond_352) {

            __al_18 = 0;

            __al_19 = 0;

            __al_20 = 0;

            __al_21 = 0;

            nij_type_d0 = nsp;

            nij_type_allocated = 1;

            qvan_init_nij = 0;


            for (_loop_it_854 = 1; (_loop_it_854 < (nsp + 1)); _loop_it_854 = (_loop_it_854 + 1)) {
                {

                    {
                        int _out_nij_type;

                        ///////////////////
                        // Tasklet code (t_364)
                        _out_nij_type = qvan_init_nij;
                        ///////////////////

                        nij_type[(_loop_it_854 - 1)] = _out_nij_type;
                    }

                }

                if (upf_tvanp) {

                    qvan_init_nij = (qvan_init_nij + dace::math::ifloor((nh[(_loop_it_854 - 1)] * (nh[(_loop_it_854 - 1)] + 1)) / 2));

                }


            }

            qvan_init_nt = (nsp + 1);


            qvan_init_nt = qvan_init_nt;

            qgm_d1 = qvan_init_nij;

            qgm_allocated = 1;

            ylmk0_d1 = (lmaxq * lmaxq);

            ylmk0_allocated = 1;

            qmod_allocated = 1;

            qvan_init_q_d0 = 3;

            qvan_init_q_allocated = 1;

            qvan_init_qq_allocated = 1;


            for (_loop_it_855 = 1; (_loop_it_855 < (__sym_dfftt_ngm_1 + 1)); _loop_it_855 = (_loop_it_855 + 1)) {


                for (_loop_it_856 = 1; (_loop_it_856 < (3 + 1)); _loop_it_856 = (_loop_it_856 + 1)) {
                    {

                        {
                            double _in_g_0 = g[((_loop_it_856 + (g_d0 * (_loop_it_855 - 1))) - 1)];
                            double _in_xkp_0 = xkp[(_loop_it_856 - 1)];
                            double _in_xkq_0 = xkq[(_loop_it_856 - 1)];
                            double _out_qvan_init_q;

                            ///////////////////
                            // Tasklet code (t_0)
                            _out_qvan_init_q = ((_in_xkp_0 - _in_xkq_0) + _in_g_0);
                            ///////////////////

                            qvan_init_q[(((3 * _loop_it_855) + _loop_it_856) - 4)] = _out_qvan_init_q;
                        }

                    }

                }

                ei0 = 4;


                loopend_381 = ((((1 + 3) - 1) - 1) + 1);


                for (_loop_it_857 = 1; (_loop_it_857 < (loopend_381 + 1)); _loop_it_857 = (_loop_it_857 + 1)) {
                    {

                        {
                            double _in_qvan_init_q_0 = qvan_init_q[(((3 * _loop_it_855) + _loop_it_857) - 4)];
                            double _out__mask_11;

                            ///////////////////
                            // Tasklet code (t_0)
                            _out__mask_11 = (dace::math::ipow(_in_qvan_init_q_0, 2));
                            ///////////////////

                            _mask_11[(_loop_it_857 - 1)] = _out__mask_11;
                        }

                    }

                }

                ei0 = (loopend_381 + 1);

                {

                    reduce_46_6_2(__state, &_mask_11[0], &qvan_init_qq[(_loop_it_855 - 1)]);
                    {
                        double _in_qvan_init_qq_0 = qvan_init_qq[(_loop_it_855 - 1)];
                        double _in_tpiba = tpiba[0];
                        double _out_qmod;

                        ///////////////////
                        // Tasklet code (t_385)
                        _out_qmod = (sqrt(_in_qvan_init_qq_0) * _in_tpiba);
                        ///////////////////

                        qmod[(_loop_it_855 - 1)] = _out_qmod;
                    }

                }

            }

            qvan_init_ig = (__sym_dfftt_ngm_1 + 1);


            qvan_init_ig = qvan_init_ig;

            __assoc_scalar_2 = (lmaxq * lmaxq);
            {

                {
                    bool _out;

                    ///////////////////
                    // Tasklet code (set_goto_10)
                    _out = 0;
                    ///////////////////

                    goto_10 = _out;
                }

            }
            if_cond_389 = ((dfftt_ngm < 1) || (__assoc_scalar_2 < 1));


            if (if_cond_389) {

            } else {

                __al_22 = 21;

                lmax = 0;


                for (; true; ) {
                    {

                        {
                            bool _in_goto_10 = goto_10;
                            int _out;

                            ///////////////////
                            // Tasklet code (set___brkc_3)
                            _out = ((__al_22 > 0) ? (! _in_goto_10) : 0);
                            ///////////////////

                            __brkc_3 = _out;
                        }

                    }
                    if_cond_396 = (__al_22 > 0);


                    if (if_cond_396) {

                        if_cond_399 = ((dace::math::ipow((lmax + 1), 2)) == __assoc_scalar_2);


                        if (if_cond_399) {
                            {

                                {
                                    bool _out;

                                    ///////////////////
                                    // Tasklet code (set_goto_10)
                                    _out = -1;
                                    ///////////////////

                                    goto_10 = _out;
                                }

                            }
                        }



                        if (goto_10) {

                        } else {

                            __al_22 = (__al_22 - 1);

                            lmax = (lmax + 1);

                        }

                        {

                            {
                                bool _in_goto_10 = goto_10;
                                int _out;

                                ///////////////////
                                // Tasklet code (set___sc_0)
                                _out = (_in_goto_10 != true);
                                ///////////////////

                                __sc_0 = _out;
                            }

                        }
                    } else {
                        {

                            {
                                int _out;

                                ///////////////////
                                // Tasklet code (set___sc_0)
                                _out = false;
                                ///////////////////

                                __sc_0 = _out;
                            }

                        }
                    }


                    if_cond_410 = (! __brkc_3);


                    if (if_cond_410) {
                        break;
                    }


                }


                if_cond_414 = (goto_10 != true);


                if (if_cond_414) {

                }

                {

                    {
                        bool _out;

                        ///////////////////
                        // Tasklet code (set_goto_10)
                        _out = 0;
                        ///////////////////

                        goto_10 = _out;
                    }

                }
                if_cond_417 = (lmax == 0);


                if (if_cond_417) {

                    for (_loop_it_858 = 1; (_loop_it_858 < (dfftt_ngm + 1)); _loop_it_858 = (_loop_it_858 + 1)) {
                        {

                            {
                                double _out_ylmk0;

                                ///////////////////
                                // Tasklet code (t_0)
                                _out_ylmk0 = 0.28209479177387814;
                                ///////////////////

                                ylmk0[(_loop_it_858 - 1)] = _out_ylmk0;
                            }

                        }

                    }

                    as_0 = (dfftt_ngm + 1);

                } else {

                    for (_loop_it_859 = 1; (_loop_it_859 < (__sym_dfftt_ngm_1 + 1)); _loop_it_859 = (_loop_it_859 + 1)) {
                        {

                            {
                                double _in_qvan_init_qq_0 = qvan_init_qq[(_loop_it_859 - 1)];
                                double _out_gmod;

                                ///////////////////
                                // Tasklet code (t_422)
                                _out_gmod = sqrt(_in_qvan_init_qq_0);
                                ///////////////////

                                gmod = _out_gmod;
                            }

                        }
                        if_cond_423 = (gmod < 1e-09);


                        if (if_cond_423) {
                            {

                                {
                                    double _out;

                                    ///////////////////
                                    // Tasklet code (set_cost)
                                    _out = 0.0;
                                    ///////////////////

                                    cost = _out;
                                }

                            }
                        } else {
                            {

                                {
                                    double _in_qvan_init_q_0 = qvan_init_q[((3 * _loop_it_859) - 1)];
                                    double _in_gmod = gmod;
                                    double _out_cost;

                                    ///////////////////
                                    // Tasklet code (t_427)
                                    _out_cost = (_in_qvan_init_q_0 / _in_gmod);
                                    ///////////////////

                                    cost = _out_cost;
                                }

                            }
                        }

                        {

                            {
                                double _in_cost = cost;
                                double _out;

                                ///////////////////
                                // Tasklet code (set_sent)
                                _out = sqrt(max(0.0, (1.0 - (_in_cost * _in_cost))));
                                ///////////////////

                                sent = _out;
                            }
                            {
                                double _out_ylmk0;

                                ///////////////////
                                // Tasklet code (t_429)
                                _out_ylmk0 = 1.0;
                                ///////////////////

                                ylmk0[(_loop_it_859 - 1)] = _out_ylmk0;
                            }

                        }
                        {

                            {
                                double _in_cost = cost;
                                double _out_ylmk0;

                                ///////////////////
                                // Tasklet code (t_431)
                                _out_ylmk0 = _in_cost;
                                ///////////////////

                                ylmk0[((_loop_it_859 + dfftt_ngm) - 1)] = _out_ylmk0;
                            }

                        }
                        {

                            {
                                double _in_sent = sent;
                                double _out_ylmk0;

                                ///////////////////
                                // Tasklet code (t_433)
                                _out_ylmk0 = (- (_in_sent / 1.4142135623730951));
                                ///////////////////

                                ylmk0[((_loop_it_859 + (3 * dfftt_ngm)) - 1)] = _out_ylmk0;
                            }

                        }

                        for (_loop_it_860 = 2; (_loop_it_860 < (lmax + 1)); _loop_it_860 = (_loop_it_860 + 1)) {

                            loopend_435 = (_loop_it_860 - 2);


                            for (_loop_it_861 = 0; (_loop_it_861 < (loopend_435 + 1)); _loop_it_861 = (_loop_it_861 + 1)) {

                                ylmr2_lm = (((dace::math::ipow(_loop_it_860, 2)) + 1) + (_loop_it_861 * 2));
                                lm1 = (((dace::math::ipow((_loop_it_860 - 1), 2)) + 1) + (_loop_it_861 * 2));
                                lm2 = (((dace::math::ipow((_loop_it_860 - 2), 2)) + 1) + (_loop_it_861 * 2));
                                {

                                    {
                                        double _in_ylmk0_0 = ylmk0[((_loop_it_859 + (dfftt_ngm * (lm1 - 1))) - 1)];
                                        double _in_ylmk0_1 = ylmk0[((_loop_it_859 + (dfftt_ngm * (lm2 - 1))) - 1)];
                                        double _in_cost = cost;
                                        double _out_ylmk0;

                                        ///////////////////
                                        // Tasklet code (t_0)
                                        _out_ylmk0 = ((((_in_cost * dace::float64(((_loop_it_860 * 2) - 1))) / sqrt(dace::float64(((_loop_it_860 * _loop_it_860) - (_loop_it_861 * _loop_it_861))))) * _in_ylmk0_0) - ((sqrt(dace::float64((((_loop_it_860 - 1) * (_loop_it_860 - 1)) - (_loop_it_861 * _loop_it_861)))) / sqrt(dace::float64(((_loop_it_860 * _loop_it_860) - (_loop_it_861 * _loop_it_861))))) * _in_ylmk0_1));
                                        ///////////////////

                                        ylmk0[((_loop_it_859 + (dfftt_ngm * (ylmr2_lm - 1))) - 1)] = _out_ylmk0;
                                    }

                                }

                            }

                            ylmr2_m = (loopend_435 + 1);


                            ylmr2_m = ylmr2_m;

                            ylmr2_lm = (((dace::math::ipow(_loop_it_860, 2)) + 1) + (_loop_it_860 * 2));

                            lm1 = (((dace::math::ipow(_loop_it_860, 2)) + 1) + ((_loop_it_860 - 1) * 2));

                            lm2 = (((dace::math::ipow((_loop_it_860 - 1), 2)) + 1) + ((_loop_it_860 - 1) * 2));
                            {

                                {
                                    double _in_ylmk0_0 = ylmk0[((_loop_it_859 + (dfftt_ngm * (lm2 - 1))) - 1)];
                                    double _in_cost = cost;
                                    double _out_ylmk0;

                                    ///////////////////
                                    // Tasklet code (t_444)
                                    _out_ylmk0 = ((_in_cost * sqrt(dace::float64(((_loop_it_860 * 2) - 1)))) * _in_ylmk0_0);
                                    ///////////////////

                                    ylmk0[((_loop_it_859 + (dfftt_ngm * (lm1 - 1))) - 1)] = _out_ylmk0;
                                }

                            }
                            {

                                {
                                    double _in_ylmk0_0 = ylmk0[((_loop_it_859 + (dfftt_ngm * (lm2 - 1))) - 1)];
                                    double _in_sent = sent;
                                    double _out_ylmk0;

                                    ///////////////////
                                    // Tasklet code (t_446)
                                    _out_ylmk0 = (- (((sqrt(dace::float64(((_loop_it_860 * 2) - 1))) / sqrt(dace::float64((_loop_it_860 * 2)))) * _in_sent) * _in_ylmk0_0));
                                    ///////////////////

                                    ylmk0[((_loop_it_859 + (dfftt_ngm * (ylmr2_lm - 1))) - 1)] = _out_ylmk0;
                                }

                            }

                        }

                        ylmr2_l = (lmax + 1);


                        ylmr2_l = ylmr2_l;

                        {

                            {
                                double _in_qvan_init_q_0 = qvan_init_q[((3 * _loop_it_859) - 3)];
                                int64_t _out_if_cond_449;

                                ///////////////////
                                // Tasklet code (t_450)
                                _out_if_cond_449 = (_in_qvan_init_q_0 > 1e-09);
                                ///////////////////

                                if_cond_449 = _out_if_cond_449;
                            }

                        }

                        if (if_cond_449) {
                            {

                                {
                                    double _in_qvan_init_q_0 = qvan_init_q[((3 * _loop_it_859) - 2)];
                                    double _in_qvan_init_q_1 = qvan_init_q[((3 * _loop_it_859) - 3)];
                                    double _out_phi;

                                    ///////////////////
                                    // Tasklet code (t_453)
                                    _out_phi = atan((_in_qvan_init_q_0 / _in_qvan_init_q_1));
                                    ///////////////////

                                    phi = _out_phi;
                                }

                            }
                        } else {

                            {

                                {
                                    double _in_qvan_init_q_0 = qvan_init_q[((3 * _loop_it_859) - 3)];
                                    int64_t _out_if_cond_455;

                                    ///////////////////
                                    // Tasklet code (t_456)
                                    _out_if_cond_455 = (_in_qvan_init_q_0 < -1e-09);
                                    ///////////////////

                                    if_cond_455 = _out_if_cond_455;
                                }

                            }

                            if (if_cond_455) {
                                {

                                    {
                                        double _in_qvan_init_q_0 = qvan_init_q[((3 * _loop_it_859) - 2)];
                                        double _in_qvan_init_q_1 = qvan_init_q[((3 * _loop_it_859) - 3)];
                                        double _out_phi;

                                        ///////////////////
                                        // Tasklet code (t_459)
                                        _out_phi = (atan((_in_qvan_init_q_0 / _in_qvan_init_q_1)) + 3.141592653589793);
                                        ///////////////////

                                        phi = _out_phi;
                                    }

                                }
                            } else {
                                {

                                    {
                                        double _in_qvan_init_q_0 = qvan_init_q[((3 * _loop_it_859) - 2)];
                                        double _out_phi;

                                        ///////////////////
                                        // Tasklet code (t_461)
                                        _out_phi = copysign(1.5707963267948966, _in_qvan_init_q_0);
                                        ///////////////////

                                        phi = _out_phi;
                                    }

                                }
                            }

                        }


                        ylmr2_lm = 1;
                        {

                            {
                                double _in_ylmk0_0 = ylmk0[(_loop_it_859 - 1)];
                                double _out_ylmk0;

                                ///////////////////
                                // Tasklet code (t_464)
                                _out_ylmk0 = (_in_ylmk0_0 / 3.5449077018110318);
                                ///////////////////

                                ylmk0[(_loop_it_859 - 1)] = _out_ylmk0;
                            }

                        }

                        for (_loop_it_862 = 1; (_loop_it_862 < (lmax + 1)); _loop_it_862 = (_loop_it_862 + 1)) {
                            {

                                {
                                    double _out;

                                    ///////////////////
                                    // Tasklet code (set_c)
                                    _out = sqrt((dace::float64(((_loop_it_862 * 2) + 1)) / 12.566370614359172));
                                    ///////////////////

                                    c = _out;
                                }

                            }
                            ylmr2_lm = (ylmr2_lm + 1);
                            {

                                {
                                    double _in_ylmk0_0 = ylmk0[((_loop_it_859 + (dfftt_ngm * (ylmr2_lm - 1))) - 1)];
                                    double _in_c = c;
                                    double _out_ylmk0;

                                    ///////////////////
                                    // Tasklet code (t_468)
                                    _out_ylmk0 = (_in_c * _in_ylmk0_0);
                                    ///////////////////

                                    ylmk0[((_loop_it_859 + (dfftt_ngm * (ylmr2_lm - 1))) - 1)] = _out_ylmk0;
                                }

                            }

                            for (_loop_it_863 = 1; (_loop_it_863 < (_loop_it_862 + 1)); _loop_it_863 = (_loop_it_863 + 1)) {

                                ylmr2_lm = (ylmr2_lm + 2);
                                {

                                    {
                                        double _in_ylmk0_0 = ylmk0[((_loop_it_859 + (dfftt_ngm * (ylmr2_lm - 1))) - 1)];
                                        double _in_c = c;
                                        double _in_phi = phi;
                                        double _out_ylmk0;

                                        ///////////////////
                                        // Tasklet code (t_0)
                                        _out_ylmk0 = (((_in_c * 1.4142135623730951) * _in_ylmk0_0) * cos((dace::float64(_loop_it_863) * _in_phi)));
                                        ///////////////////

                                        ylmk0[((_loop_it_859 + (dfftt_ngm * (ylmr2_lm - 2))) - 1)] = _out_ylmk0;
                                    }

                                }
                                {

                                    {
                                        double _in_ylmk0_0 = ylmk0[((_loop_it_859 + (dfftt_ngm * (ylmr2_lm - 1))) - 1)];
                                        double _in_c = c;
                                        double _in_phi = phi;
                                        double _out_ylmk0;

                                        ///////////////////
                                        // Tasklet code (t_1)
                                        _out_ylmk0 = (((_in_c * 1.4142135623730951) * _in_ylmk0_0) * sin((dace::float64(_loop_it_863) * _in_phi)));
                                        ///////////////////

                                        ylmk0[((_loop_it_859 + (dfftt_ngm * (ylmr2_lm - 1))) - 1)] = _out_ylmk0;
                                    }

                                }

                            }

                            ylmr2_m = (_loop_it_862 + 1);


                            ylmr2_m = ylmr2_m;


                        }

                        ylmr2_l = (lmax + 1);


                        ylmr2_l = ylmr2_l;


                    }

                    ylmr2_ig = (__sym_dfftt_ngm_1 + 1);


                    ylmr2_ig = ylmr2_ig;

                }

            }


            qvan_init_qq_allocated = 0;
            {

                {
                    #pragma omp parallel for
                    for (auto __i0 = 0; __i0 < dfftt_ngm; __i0 += 1) {
                        {
                            double _out;

                            ///////////////////
                            // Tasklet code (set_qvan_init_qq)
                            _out = 0;
                            ///////////////////

                            qvan_init_qq[__i0] = _out;
                        }
                    }
                }

            }
            qvan_init_q_allocated = 0;
            {

                {
                    #pragma omp parallel for
                    for (auto __i0 = 0; __i0 < 3; __i0 += 1) {
                        for (auto __i1 = 0; __i1 < dfftt_ngm; __i1 += 1) {
                            {
                                double _out;

                                ///////////////////
                                // Tasklet code (set_qvan_init_q)
                                _out = 0;
                                ///////////////////

                                qvan_init_q[(__i0 + (3 * __i1))] = _out;
                            }
                        }
                    }
                }

            }
            ijh = 0;


            for (_loop_it_864 = 1; (_loop_it_864 < (nsp + 1)); _loop_it_864 = (_loop_it_864 + 1)) {


                if (upf_tvanp) {

                    loopend_485 = nh[(_loop_it_864 - 1)];


                    for (_loop_it_865 = 1; (_loop_it_865 < (loopend_485 + 1)); _loop_it_865 = (_loop_it_865 + 1)) {

                        loopend_488 = nh[(_loop_it_864 - 1)];


                        for (_loop_it_866 = _loop_it_865; (_loop_it_866 < (loopend_488 + 1)); _loop_it_866 = (_loop_it_866 + 1)) {

                            ijh = (ijh + 1);

                            qvan2_nb = indv[((_loop_it_865 + (indv_d0 * (_loop_it_864 - 1))) - 1)];

                            mb = indv[((_loop_it_866 + (indv_d0 * (_loop_it_864 - 1))) - 1)];

                            if_cond_495 = (qvan2_nb >= mb);


                            if (if_cond_495) {

                                ijv = (dace::math::ifloor((qvan2_nb * (qvan2_nb - 1)) / 2) + mb);

                            } else {

                                ijv = (dace::math::ifloor((mb * (mb - 1)) / 2) + qvan2_nb);

                            }


                            ivl = nhtolm[((_loop_it_865 + (nhtolm_d0 * (_loop_it_864 - 1))) - 1)];

                            jvl = nhtolm[((_loop_it_866 + (nhtolm_d0 * (_loop_it_864 - 1))) - 1)];

                            if_cond_504 = ((qvan2_nb > nbetam) || (mb > nbetam));


                            if (if_cond_504) {
                                {
                                    int __assoc_scalar_3;

                                    {
                                        int _out;

                                        ///////////////////
                                        // Tasklet code (set___assoc_scalar_3)
                                        _out = max(qvan2_nb, mb);
                                        ///////////////////

                                        __assoc_scalar_3 = _out;
                                    }

                                }
                            }


                            if_cond_508 = ((ivl > 25) || (jvl > 25));


                            if (if_cond_508) {
                                {
                                    int __assoc_scalar_4;

                                    {
                                        int _out;

                                        ///////////////////
                                        // Tasklet code (set___assoc_scalar_4)
                                        _out = max(ivl, jvl);
                                        ///////////////////

                                        __assoc_scalar_4 = _out;
                                    }

                                }
                            }

                            {
                                dace::complex128* qvan2_qg;
                                qvan2_qg = &qgm[(dfftt_ngm * (ijh - 1))];

                                {
                                    double _out;

                                    ///////////////////
                                    // Tasklet code (set_dqi)
                                    _out = 100.0;
                                    ///////////////////

                                    dqi = _out;
                                }
                                {
                                    dace::complex128* _mset_out = qvan2_qg;

                                    ///////////////////
                                    memset(_mset_out, 0, dfftt_ngm * sizeof(dace::complex128));
                                    ///////////////////

                                }

                            }

                            loopend_514 = lpx[((ivl + (25 * jvl)) - 26)];


                            for (_loop_it_867 = 1; (_loop_it_867 < (loopend_514 + 1)); _loop_it_867 = (_loop_it_867 + 1)) {

                                lp = lpl[((((625 * _loop_it_867) + ivl) + (25 * jvl)) - 651)];

                                if_cond_518 = ((lp < 1) || (lp > 49));


                                if (if_cond_518) {
                                    {
                                        int __assoc_scalar_5;

                                        {
                                            int _out;

                                            ///////////////////
                                            // Tasklet code (set___assoc_scalar_5)
                                            _out = max(lp, 1);
                                            ///////////////////

                                            __assoc_scalar_5 = _out;
                                        }

                                    }
                                }


                                if_cond_522 = (lp == 1);


                                if (if_cond_522) {

                                    qvan2_l = 1;
                                    {

                                        {
                                            double _out;

                                            ///////////////////
                                            // Tasklet code (set_sig)
                                            _out = 1.0;
                                            ///////////////////

                                            sig = _out;
                                        }

                                    }
                                    ind = 1;

                                } else {

                                    if_cond_528 = (lp <= 4);


                                    if (if_cond_528) {

                                        qvan2_l = 2;
                                        {

                                            {
                                                double _out;

                                                ///////////////////
                                                // Tasklet code (set_sig)
                                                _out = -1.0;
                                                ///////////////////

                                                sig = _out;
                                            }

                                        }
                                        ind = 2;

                                    } else {

                                        if_cond_534 = (lp <= 9);


                                        if (if_cond_534) {

                                            qvan2_l = 3;
                                            {

                                                {
                                                    double _out;

                                                    ///////////////////
                                                    // Tasklet code (set_sig)
                                                    _out = -1.0;
                                                    ///////////////////

                                                    sig = _out;
                                                }

                                            }
                                            ind = 1;

                                        } else {

                                            if_cond_540 = (lp <= 16);


                                            if (if_cond_540) {

                                                qvan2_l = 4;
                                                {

                                                    {
                                                        double _out;

                                                        ///////////////////
                                                        // Tasklet code (set_sig)
                                                        _out = 1.0;
                                                        ///////////////////

                                                        sig = _out;
                                                    }

                                                }
                                                ind = 2;

                                            } else {

                                                if_cond_546 = (lp <= 25);


                                                if (if_cond_546) {

                                                    qvan2_l = 5;
                                                    {

                                                        {
                                                            double _out;

                                                            ///////////////////
                                                            // Tasklet code (set_sig)
                                                            _out = 1.0;
                                                            ///////////////////

                                                            sig = _out;
                                                        }

                                                    }
                                                    ind = 1;

                                                } else {

                                                    if_cond_552 = (lp <= 36);


                                                    if (if_cond_552) {

                                                        qvan2_l = 6;
                                                        {

                                                            {
                                                                double _out;

                                                                ///////////////////
                                                                // Tasklet code (set_sig)
                                                                _out = -1.0;
                                                                ///////////////////

                                                                sig = _out;
                                                            }

                                                        }
                                                        ind = 2;

                                                    } else {

                                                        qvan2_l = 7;
                                                        {

                                                            {
                                                                double _out;

                                                                ///////////////////
                                                                // Tasklet code (set_sig)
                                                                _out = -1.0;
                                                                ///////////////////

                                                                sig = _out;
                                                            }

                                                        }
                                                        ind = 1;

                                                    }

                                                }

                                            }

                                        }

                                    }

                                }

                                {

                                    {
                                        double _in_ap_0 = ap[((((81 * ivl) + (2025 * jvl)) + lp) - 2107)];
                                        double _in_sig = sig;
                                        double _out_sig;

                                        ///////////////////
                                        // Tasklet code (t_561)
                                        _out_sig = (_in_sig * _in_ap_0);
                                        ///////////////////

                                        sig = _out_sig;
                                    }

                                }

                                for (_loop_it_868 = 1; (_loop_it_868 < (__sym_dfftt_ngm_1 + 1)); _loop_it_868 = (_loop_it_868 + 1)) {

                                    qvan2_i0 = (dace::int32(qm) + 1);

                                    qvan2_i1 = (qvan2_i0 + 1);
                                    qvan2_i2 = (qvan2_i0 + 2);
                                    qvan2_i3 = (qvan2_i0 + 3);
                                    {
                                        dace::complex128* qvan2_qg;
                                        qvan2_qg = &qgm[(dfftt_ngm * (ijh - 1))];
                                        double pwx;
                                        double qvan2_px;
                                        double uvx;
                                        double qvan2_ux;
                                        double qvan2_vx;
                                        double work;
                                        double qvan2_wx;

                                        {
                                            double _in_qmod_0 = qmod[(_loop_it_868 - 1)];
                                            double _in_dqi = dqi;
                                            double _out_qm;

                                            ///////////////////
                                            // Tasklet code (t_0)
                                            _out_qm = (_in_qmod_0 * _in_dqi);
                                            ///////////////////

                                            qm = _out_qm;
                                        }
                                        {
                                            double _in_qm = qm;
                                            double _out_qvan2_px;

                                            ///////////////////
                                            // Tasklet code (t_1)
                                            _out_qvan2_px = (_in_qm - dace::float64(dace::int32(_in_qm)));
                                            ///////////////////

                                            qvan2_px = _out_qvan2_px;
                                        }
                                        {
                                            double _in_qvan2_px = qvan2_px;
                                            double _out_qvan2_ux;

                                            ///////////////////
                                            // Tasklet code (t_2)
                                            _out_qvan2_ux = (1.0 - _in_qvan2_px);
                                            ///////////////////

                                            qvan2_ux = _out_qvan2_ux;
                                        }
                                        {
                                            double _in_qvan2_px = qvan2_px;
                                            double _out_qvan2_vx;

                                            ///////////////////
                                            // Tasklet code (t_3)
                                            _out_qvan2_vx = (2.0 - _in_qvan2_px);
                                            ///////////////////

                                            qvan2_vx = _out_qvan2_vx;
                                        }
                                        {
                                            double _in_qvan2_ux = qvan2_ux;
                                            double _in_qvan2_vx = qvan2_vx;
                                            double _out_uvx;

                                            ///////////////////
                                            // Tasklet code (t_5)
                                            _out_uvx = ((_in_qvan2_ux * _in_qvan2_vx) * 0.16666666666666666);
                                            ///////////////////

                                            uvx = _out_uvx;
                                        }
                                        {
                                            double _in_qvan2_px = qvan2_px;
                                            double _out_qvan2_wx;

                                            ///////////////////
                                            // Tasklet code (t_4)
                                            _out_qvan2_wx = (3.0 - _in_qvan2_px);
                                            ///////////////////

                                            qvan2_wx = _out_qvan2_wx;
                                        }
                                        {
                                            double _in_qvan2_px = qvan2_px;
                                            double _in_qvan2_wx = qvan2_wx;
                                            double _out_pwx;

                                            ///////////////////
                                            // Tasklet code (t_6)
                                            _out_pwx = ((_in_qvan2_px * _in_qvan2_wx) * 0.5);
                                            ///////////////////

                                            pwx = _out_pwx;
                                        }
                                        {
                                            double _in_tab_qrad_0 = tab_qrad[((((qvan2_i0 + (((tab_qrad_d0 * tab_qrad_d1) * tab_qrad_d2) * (_loop_it_864 - 1))) + ((tab_qrad_d0 * tab_qrad_d1) * (qvan2_l - 1))) + (tab_qrad_d0 * (ijv - 1))) - 1)];
                                            double _in_tab_qrad_1 = tab_qrad[((((qvan2_i1 + (((tab_qrad_d0 * tab_qrad_d1) * tab_qrad_d2) * (_loop_it_864 - 1))) + ((tab_qrad_d0 * tab_qrad_d1) * (qvan2_l - 1))) + (tab_qrad_d0 * (ijv - 1))) - 1)];
                                            double _in_tab_qrad_2 = tab_qrad[((((qvan2_i2 + (((tab_qrad_d0 * tab_qrad_d1) * tab_qrad_d2) * (_loop_it_864 - 1))) + ((tab_qrad_d0 * tab_qrad_d1) * (qvan2_l - 1))) + (tab_qrad_d0 * (ijv - 1))) - 1)];
                                            double _in_tab_qrad_3 = tab_qrad[((((qvan2_i3 + (((tab_qrad_d0 * tab_qrad_d1) * tab_qrad_d2) * (_loop_it_864 - 1))) + ((tab_qrad_d0 * tab_qrad_d1) * (qvan2_l - 1))) + (tab_qrad_d0 * (ijv - 1))) - 1)];
                                            double _in_pwx = pwx;
                                            double _in_qvan2_px = qvan2_px;
                                            double _in_qvan2_ux = qvan2_ux;
                                            double _in_qvan2_vx = qvan2_vx;
                                            double _in_qvan2_wx = qvan2_wx;
                                            double _in_uvx = uvx;
                                            double _out_work;

                                            ///////////////////
                                            // Tasklet code (t_7)
                                            _out_work = (((((_in_tab_qrad_0 * _in_uvx) * _in_qvan2_wx) + ((_in_tab_qrad_1 * _in_pwx) * _in_qvan2_vx)) - ((_in_tab_qrad_2 * _in_pwx) * _in_qvan2_ux)) + ((_in_tab_qrad_3 * _in_qvan2_px) * _in_uvx));
                                            ///////////////////

                                            work = _out_work;
                                        }
                                        {
                                            dace::complex128 _in_z = qvan2_qg[(_loop_it_868 - 1)];
                                            double _in_ylmk0_0 = ylmk0[((_loop_it_868 + (dfftt_ngm * (lp - 1))) - 1)];
                                            double _in_sig = sig;
                                            double _in_work = work;
                                            dace::complex128 _out_z;

                                            ///////////////////
                                            // Tasklet code (cc_qvan2_qg_8)
                                            auto _cur = ((ind == 1) ? _in_z.real() : _in_z.imag());
                                            auto _new = (_cur + ((_in_sig * _in_ylmk0_0) * _in_work));
                                            _out_z = ((ind == 1) ? (_new + (dace::complex128(0.0, 1.0) * _in_z.imag())) : (_in_z.real() + (dace::complex128(0.0, 1.0) * _new)));
                                            ///////////////////

                                            qvan2_qg[(_loop_it_868 - 1)] = _out_z;
                                        }

                                    }

                                }

                                qvan2_ig = (__sym_dfftt_ngm_1 + 1);


                                qvan2_ig = qvan2_ig;


                            }

                            qvan2_lm = (loopend_514 + 1);


                            qvan2_lm = qvan2_lm;


                        }

                        qvan_init_jh = (loopend_488 + 1);


                        qvan_init_jh = qvan_init_jh;


                    }

                    qvan_init_ih = (loopend_485 + 1);


                    qvan_init_ih = qvan_init_ih;

                }


            }

            qvan_init_nt = (nsp + 1);


            qvan_init_nt = qvan_init_nt;

            qmod_allocated = 0;
            {

                {
                    #pragma omp parallel for
                    for (auto __i0 = 0; __i0 < dfftt_ngm; __i0 += 1) {
                        {
                            double _out;

                            ///////////////////
                            // Tasklet code (set_qmod)
                            _out = 0;
                            ///////////////////

                            qmod[__i0] = _out;
                        }
                    }
                }

            }
            ylmk0_allocated = 0;
            {

                {
                    #pragma omp parallel for
                    for (auto __i0 = 0; __i0 < dfftt_ngm; __i0 += 1) {
                        for (auto __i1 = 0; __i1 < (lmaxq * lmaxq); __i1 += 1) {
                            {
                                double _out;

                                ///////////////////
                                // Tasklet code (set_ylmk0)
                                _out = 0;
                                ///////////////////

                                ylmk0[(__i0 + (__i1 * dfftt_ngm))] = _out;
                            }
                        }
                    }
                }

            }
        }


        for (_loop_it_869 = 1; (_loop_it_869 < (negrp + 1)); _loop_it_869 = (_loop_it_869 + 1)) {

            wegrp = ((((_loop_it_869 + my_egrp_id) - 1) % negrp) + 1);

            njt = dace::math::ifloor(((all_end[(wegrp - 1)] - all_start[(wegrp - 1)]) + jblock) / jblock);


            for (_loop_it_870 = 1; (_loop_it_870 < (njt + 1)); _loop_it_870 = (_loop_it_870 + 1)) {
                {

                    {
                        int _in_all_start_0 = all_start[(wegrp - 1)];
                        int _out_jblock_start;

                        ///////////////////
                        // Tasklet code (t_583)
                        _out_jblock_start = (((_loop_it_870 - 1) * jblock) + _in_all_start_0);
                        ///////////////////

                        jblock_start = _out_jblock_start;
                    }
                    {
                        int _in_all_end_0 = all_end[(wegrp - 1)];
                        int _in_jblock_start = jblock_start;
                        int _out_jblock_end;

                        ///////////////////
                        // Tasklet code (t_584)
                        _out_jblock_end = min(((_in_jblock_start + jblock) - 1), _in_all_end_0);
                        ///////////////////

                        jblock_end = _out_jblock_end;
                    }

                }
                loopend_585 = nibands[my_egrp_id];


                for (_loop_it_871 = 1; (_loop_it_871 < (loopend_585 + 1)); _loop_it_871 = (_loop_it_871 + 1)) {

                    ibnd = ibands[((_loop_it_871 + (ibands_d0 * my_egrp_id)) - 1)];

                    if_cond_589 = (((ibnd == 0) || (ibnd > m)) != true);


                    if (if_cond_589) {

                        jstart = 0;

                        jend = 0;


                        for (_loop_it_872 = 1; (_loop_it_872 < (max_pairs + 1)); _loop_it_872 = (_loop_it_872 + 1)) {

                            {

                                {
                                    int _in_egrp_pairs_0 = egrp_pairs[(((egrp_pairs_d0 * egrp_pairs_d1) * my_egrp_id) + (egrp_pairs_d0 * (_loop_it_872 - 1)))];
                                    int64_t _out_if_cond_596;

                                    ///////////////////
                                    // Tasklet code (t_597)
                                    _out_if_cond_596 = (_in_egrp_pairs_0 == ibnd);
                                    ///////////////////

                                    if_cond_596 = _out_if_cond_596;
                                }

                            }

                            if (if_cond_596) {

                                if_cond_600 = (jstart == 0);


                                if (if_cond_600) {

                                    jstart = egrp_pairs[((((egrp_pairs_d0 * egrp_pairs_d1) * my_egrp_id) + (egrp_pairs_d0 * (_loop_it_872 - 1))) + 1)];

                                    jend = jstart;

                                } else {

                                    jend = egrp_pairs[((((egrp_pairs_d0 * egrp_pairs_d1) * my_egrp_id) + (egrp_pairs_d0 * (_loop_it_872 - 1))) + 1)];

                                }

                            }


                        }

                        ipair = (max_pairs + 1);


                        ipair = ipair;

                        jstart = max(jstart, jblock_start);

                        jend = min(jend, jblock_end);

                        jcount = ((jend - jstart) + 1);

                        if_cond_612 = (jcount > 0);


                        if (if_cond_612) {
                            {
                                int nblock;
                                int nrt;

                                {
                                    int _out;

                                    ///////////////////
                                    // Tasklet code (set_nblock)
                                    _out = 2048;
                                    ///////////////////

                                    nblock = _out;
                                }
                                {
                                    int _in_nblock = nblock;
                                    int _out;

                                    ///////////////////
                                    // Tasklet code (set_nrt)
                                    _out = dace::math::ifloor(((nrxxs + _in_nblock) - 1) / _in_nblock);
                                    ///////////////////

                                    nrt = _out;
                                }

                            }
                            all_start_tmp = all_start[(wegrp - 1)];


                            for (_loop_it_873 = jstart; (_loop_it_873 < (jend + 1)); _loop_it_873 = (_loop_it_873 + 1)) {

                                for (_loop_it_874 = 1; (_loop_it_874 < (nrxxs + 1)); _loop_it_874 = (_loop_it_874 + 1)) {

                                    if_cond_619 = noncolin[0];


                                    if (if_cond_619) {
                                        {

                                            {
                                                dace::complex128 _in_exxbuff_d_0 = exxbuff_d[(((_loop_it_874 + ((exxbuff_d_d0 * exxbuff_d_d1) * (ikq - 1))) + (exxbuff_d_d0 * (((_loop_it_873 - all_start_tmp) + iexx_start) - 1))) - 1)];
                                                dace::complex128 _in_exxbuff_d_1 = exxbuff_d[(((_loop_it_874 + ((exxbuff_d_d0 * exxbuff_d_d1) * (ikq - 1))) + (exxbuff_d_d0 * (((_loop_it_873 - all_start_tmp) + iexx_start) - 1))) - 1)];
                                                dace::complex128 _in_exxbuff_d_2 = exxbuff_d[((((_loop_it_874 + ((exxbuff_d_d0 * exxbuff_d_d1) * (ikq - 1))) + (exxbuff_d_d0 * (((_loop_it_873 - all_start_tmp) + iexx_start) - 1))) + nrxxs) - 1)];
                                                dace::complex128 _in_exxbuff_d_3 = exxbuff_d[((((_loop_it_874 + ((exxbuff_d_d0 * exxbuff_d_d1) * (ikq - 1))) + (exxbuff_d_d0 * (((_loop_it_873 - all_start_tmp) + iexx_start) - 1))) + nrxxs) - 1)];
                                                dace::complex128 _in_temppsic_nc_d_0 = temppsic_nc_d[((_loop_it_874 + ((npol * nrxxs) * (_loop_it_871 - 1))) - 1)];
                                                dace::complex128 _in_temppsic_nc_d_1 = temppsic_nc_d[(((_loop_it_874 + ((npol * nrxxs) * (_loop_it_871 - 1))) + nrxxs) - 1)];
                                                double _in_omega_inv = omega_inv;
                                                dace::complex128 _out_rhoc_d;

                                                ///////////////////
                                                // Tasklet code (t_622)
                                                _out_rhoc_d = (((conj(_in_exxbuff_d_0) * _in_temppsic_nc_d_0) + (conj(_in_exxbuff_d_1) * _in_temppsic_nc_d_1)) * (_in_omega_inv + (dace::complex128(0.0, 1.0) * 0.0)));
                                                ///////////////////

                                                rhoc_d[((_loop_it_874 + (nrxxs * (_loop_it_873 - jstart))) - 1)] = _out_rhoc_d;
                                            }

                                        }
                                    } else {
                                        {

                                            {
                                                dace::complex128 _in_exxbuff_d_0 = exxbuff_d[(((_loop_it_874 + ((exxbuff_d_d0 * exxbuff_d_d1) * (ikq - 1))) + (exxbuff_d_d0 * (((_loop_it_873 - all_start_tmp) + iexx_start) - 1))) - 1)];
                                                dace::complex128 _in_exxbuff_d_1 = exxbuff_d[(((_loop_it_874 + ((exxbuff_d_d0 * exxbuff_d_d1) * (ikq - 1))) + (exxbuff_d_d0 * (((_loop_it_873 - all_start_tmp) + iexx_start) - 1))) - 1)];
                                                dace::complex128 _in_temppsic_d_0 = temppsic_d[((_loop_it_874 + (nrxxs * (_loop_it_871 - 1))) - 1)];
                                                double _in_omega_inv = omega_inv;
                                                dace::complex128 _out_rhoc_d;

                                                ///////////////////
                                                // Tasklet code (t_624)
                                                _out_rhoc_d = ((conj(_in_exxbuff_d_0) * _in_temppsic_d_0) * (_in_omega_inv + (dace::complex128(0.0, 1.0) * 0.0)));
                                                ///////////////////

                                                rhoc_d[((_loop_it_874 + (nrxxs * (_loop_it_873 - jstart))) - 1)] = _out_rhoc_d;
                                            }

                                        }
                                    }


                                }

                                ir = (nrxxs + 1);


                                ir = ir;


                            }

                            jbnd = (jend + 1);


                            jbnd = jbnd;

                            if_cond_629 = (okvan[0] && tqr[0]);


                            if (if_cond_629) {

                                for (_loop_it_875 = jstart; (_loop_it_875 < (jend + 1)); _loop_it_875 = (_loop_it_875 + 1)) {

                                    if_cond_633 = okvan[0];


                                    if (if_cond_633) {

                                        for (_loop_it_876 = 1; (_loop_it_876 < (nat + 1)); _loop_it_876 = (_loop_it_876 + 1)) {

                                            addusxx_r_mbia = tabxx_maxbox[(_loop_it_876 - 1)];

                                            if_cond_638 = (addusxx_r_mbia != 0);


                                            if (if_cond_638) {

                                                addusxx_r_nt = ityp[(_loop_it_876 - 1)];


                                                if (upf_tvanp) {

                                                    loopend_643 = nh[(addusxx_r_nt - 1)];


                                                    for (_loop_it_877 = 1; (_loop_it_877 < (loopend_643 + 1)); _loop_it_877 = (_loop_it_877 + 1)) {

                                                        loopend_646 = nh[(addusxx_r_nt - 1)];


                                                        for (_loop_it_878 = 1; (_loop_it_878 < (loopend_646 + 1)); _loop_it_878 = (_loop_it_878 + 1)) {

                                                            addusxx_r_ikb = (ofsbeta[(_loop_it_876 - 1)] + _loop_it_877);

                                                            addusxx_r_jkb = (ofsbeta[(_loop_it_876 - 1)] + _loop_it_878);


                                                            for (_loop_it_879 = 1; (_loop_it_879 < (addusxx_r_mbia + 1)); _loop_it_879 = (_loop_it_879 + 1)) {

                                                                ijtoh_at28 = ijtoh[(((_loop_it_877 + ((ijtoh_d0 * ijtoh_d1) * (addusxx_r_nt - 1))) + (ijtoh_d0 * (_loop_it_878 - 1))) - 1)];

                                                                irb = tabxx_box[((_loop_it_876 + (tabxx_box_d0 * (_loop_it_879 - 1))) - 1)];
                                                                {

                                                                    {
                                                                        dace::complex128 _in_addusxx_r_becphi_0 = becxx_k[(((((becxx_k_d0 * becxx_k_d1) * (_loop_it_875 - 1)) + (becxx_k_d0 * (addusxx_r_ikb - 1))) + ikq) - 1)];
                                                                        dace::complex128 _in_addusxx_r_becphi_1 = becxx_k[(((((becxx_k_d0 * becxx_k_d1) * (_loop_it_875 - 1)) + (becxx_k_d0 * (addusxx_r_ikb - 1))) + ikq) - 1)];
                                                                        dace::complex128 _in_addusxx_r_becpsi_0 = becpsi_k[((addusxx_r_jkb + (becpsi_k_d0 * (ibnd - offset_becpsi_k_d1))) - offset_becpsi_k_d0)];
                                                                        dace::complex128 _in_rhoc_0 = rhoc[((irb + (nrxxs * (_loop_it_875 - jstart))) - 1)];
                                                                        double _in_tabxx_qr_0 = tabxx_qr[(((_loop_it_876 + ((tabxx_qr_d0 * tabxx_qr_d1) * (ijtoh_at28 - 1))) + (tabxx_qr_d0 * (_loop_it_879 - 1))) - 1)];
                                                                        dace::complex128 _out_rhoc;

                                                                        ///////////////////
                                                                        // Tasklet code (t_0)
                                                                        _out_rhoc = (_in_rhoc_0 + (((_in_tabxx_qr_0 + (dace::complex128(0.0, 1.0) * 0.0)) * conj(_in_addusxx_r_becphi_0)) * _in_addusxx_r_becpsi_0));
                                                                        ///////////////////

                                                                        rhoc[((irb + (nrxxs * (_loop_it_875 - jstart))) - 1)] = _out_rhoc;
                                                                    }

                                                                }

                                                            }

                                                            addusxx_r_ir = (addusxx_r_mbia + 1);


                                                            addusxx_r_ir = addusxx_r_ir;


                                                        }

                                                        addusxx_r_jh = (loopend_646 + 1);


                                                        addusxx_r_jh = addusxx_r_jh;


                                                    }

                                                    addusxx_r_ih = (loopend_643 + 1);


                                                    addusxx_r_ih = addusxx_r_ih;

                                                }

                                            }


                                        }

                                        addusxx_r_ia = (nat + 1);


                                        addusxx_r_ia = addusxx_r_ia;

                                    }


                                }

                                jbnd = (jend + 1);


                                jbnd = jbnd;

                            }


                            for (_loop_it_880 = jstart; (_loop_it_880 < (jend + 1)); _loop_it_880 = (_loop_it_880 + many_fft[0])) {

                                jcurr = min(many_fft[0], ((jend - _loop_it_880) + 1));

                                {
                                    // MANUAL FIX: removed a spurious ``rhoc_d = new`` realloc here -- the
                                    // SDFG lowering re-allocated rhoc_d (to a fresh ZERO buffer) between the
                                    // rhoc compute loop and this fwfft, discarding the computed values (so
                                    // the fwfft transformed zeros). prhoc_d must slice the EXISTING rhoc_d.
                                    dace::complex128* prhoc_d;
                                    prhoc_d = &rhoc_d[(nrxxs * (_loop_it_880 - jstart))];

                                    dace_libraries_fft_algorithms_dft_dft_explicit_227_2_4(__state, &prhoc_d[0], &prhoc_d[0], jcurr, nrxxs);

                                }

                            }

                            jbnd = ((jend + many_fft[0]) - ((jend - jstart) % many_fft[0]));


                            jbnd = jbnd;

                            if_cond_672 = (okvan[0] && (tqr[0] != true));


                            if (if_cond_672) {
                                {

                                    copy_rhoc_675_sdfg_230_0_2(__state, &rhoc_d[0], &rhoc[0], jblock, nrxxs);

                                }

                                for (_loop_it_881 = jstart; (_loop_it_881 < (jend + 1)); _loop_it_881 = (_loop_it_881 + 1)) {

                                    __al_23 = 0;

                                    __al_24 = 0;

                                    __al_25 = 0;

                                    if_cond_681 = okvan[0];


                                    if (if_cond_681) {
                                        {

                                            {
                                                int _out;

                                                ///////////////////
                                                // Tasklet code (set_addusxx_g_ngms)
                                                _out = dfftt_ngm;
                                                ///////////////////

                                                addusxx_g_ngms = _out;
                                            }
                                            {
                                                bool _out;

                                                ///////////////////
                                                // Tasklet code (set_addusxx_g_add_complex)
                                                _out = -1;
                                                ///////////////////

                                                addusxx_g_add_complex = _out;
                                            }
                                            {
                                                bool _out;

                                                ///////////////////
                                                // Tasklet code (set_addusxx_g_add_real)
                                                _out = -1;
                                                ///////////////////

                                                addusxx_g_add_real = _out;
                                            }
                                            {
                                                bool _out;

                                                ///////////////////
                                                // Tasklet code (set_addusxx_g_add_imaginary)
                                                _out = -1;
                                                ///////////////////

                                                addusxx_g_add_imaginary = _out;
                                            }

                                        }
                                        if_cond_684 = (((addusxx_g_add_complex || addusxx_g_add_real) || addusxx_g_add_imaginary) != true);


                                        if (if_cond_684) {
                                            {
                                                int __assoc_scalar_6;

                                                {
                                                    int _out;

                                                    ///////////////////
                                                    // Tasklet code (set___assoc_scalar_6)
                                                    _out = 1;
                                                    ///////////////////

                                                    __assoc_scalar_6 = _out;
                                                }

                                            }
                                        }


                                        if_cond_688 = ((gamma_only[0] != true) && (addusxx_g_add_real || addusxx_g_add_imaginary));


                                        if (if_cond_688) {
                                            {
                                                int __assoc_scalar_7;

                                                {
                                                    int _out;

                                                    ///////////////////
                                                    // Tasklet code (set___assoc_scalar_7)
                                                    _out = 2;
                                                    ///////////////////

                                                    __assoc_scalar_7 = _out;
                                                }

                                            }
                                        }


                                        if_cond_692 = (gamma_only[0] && addusxx_g_add_complex);


                                        if (if_cond_692) {
                                            {
                                                int __assoc_scalar_8;

                                                {
                                                    int _out;

                                                    ///////////////////
                                                    // Tasklet code (set___assoc_scalar_8)
                                                    _out = 3;
                                                    ///////////////////

                                                    __assoc_scalar_8 = _out;
                                                }

                                            }
                                        }


                                        if_cond_696 = (((addusxx_g_add_complex && ((1 != true) || (1 != true))) || (addusxx_g_add_real && ((0 != true) || (0 != true)))) || (addusxx_g_add_imaginary && ((0 != true) || (0 != true))));


                                        if (if_cond_696) {
                                            {
                                                int __assoc_scalar_9;

                                                {
                                                    int _out;

                                                    ///////////////////
                                                    // Tasklet code (set___assoc_scalar_9)
                                                    _out = 2;
                                                    ///////////////////

                                                    __assoc_scalar_9 = _out;
                                                }

                                            }
                                        }


                                        addusxx_g_eigqts_d0 = nat;

                                        addusxx_g_eigqts_allocated = 1;


                                        for (_loop_it_882 = 1; (_loop_it_882 < (nat + 1)); _loop_it_882 = (_loop_it_882 + 1)) {


                                            for (_loop_it_883 = 1; (_loop_it_883 < (3 + 1)); _loop_it_883 = (_loop_it_883 + 1)) {
                                                {

                                                    {
                                                        double _in_tau_0 = tau[((_loop_it_883 + (tau_d0 * (_loop_it_882 - 1))) - 1)];
                                                        double _in_xkp_0 = xkp[(_loop_it_883 - 1)];
                                                        double _in_xkq_0 = xkq[(_loop_it_883 - 1)];
                                                        double _out__mask_12;

                                                        ///////////////////
                                                        // Tasklet code (t_0)
                                                        _out__mask_12 = ((_in_xkp_0 - _in_xkq_0) * _in_tau_0);
                                                        ///////////////////

                                                        _mask_12[(_loop_it_883 - 1)] = _out__mask_12;
                                                    }

                                                }

                                            }

                                            ei0 = 4;

                                            {
                                                double _QQred_lift_4;
                                                double addusxx_g_arg;

                                                reduce_46_6_2(__state, &_mask_12[0], &_QQred_lift_4);
                                                {
                                                    double _in__QQred_lift_4 = _QQred_lift_4;
                                                    double _out;

                                                    ///////////////////
                                                    // Tasklet code (set_addusxx_g_arg)
                                                    _out = (_in__QQred_lift_4 * 6.283185307179586);
                                                    ///////////////////

                                                    addusxx_g_arg = _out;
                                                }
                                                {
                                                    double _in_addusxx_g_arg = addusxx_g_arg;
                                                    dace::complex128 _out_addusxx_g_eigqts;

                                                    ///////////////////
                                                    // Tasklet code (t_706)
                                                    _out_addusxx_g_eigqts = (cos(_in_addusxx_g_arg) + (dace::complex128(0.0, 1.0) * (- sin(_in_addusxx_g_arg))));
                                                    ///////////////////

                                                    addusxx_g_eigqts[(_loop_it_882 - 1)] = _out_addusxx_g_eigqts;
                                                }

                                            }

                                        }

                                        addusxx_g_na = (nat + 1);


                                        addusxx_g_na = addusxx_g_na;

                                        addusxx_g_numblock = dace::math::ifloor((addusxx_g_ngms + 255) / 256);

                                        addusxx_g_aux1_d0 = 256;

                                        addusxx_g_aux1_allocated = 1;

                                        addusxx_g_aux2_d0 = 256;

                                        addusxx_g_aux2_allocated = 1;


                                        for (_loop_it_884 = 1; (_loop_it_884 < (nsp + 1)); _loop_it_884 = (_loop_it_884 + 1)) {


                                            if (upf_tvanp) {

                                                addusxx_g_nij = nij_type[(_loop_it_884 - 1)];


                                                for (_loop_it_885 = 1; (_loop_it_885 < (addusxx_g_numblock + 1)); _loop_it_885 = (_loop_it_885 + 1)) {

                                                    for (_loop_it_886 = 1; (_loop_it_886 < (nat + 1)); _loop_it_886 = (_loop_it_886 + 1)) {

                                                        {

                                                            {
                                                                int _in_ityp_0 = ityp[(_loop_it_886 - 1)];
                                                                int64_t _out_if_cond_722;

                                                                ///////////////////
                                                                // Tasklet code (t_723)
                                                                _out_if_cond_722 = (_in_ityp_0 == _loop_it_884);
                                                                ///////////////////

                                                                if_cond_722 = _out_if_cond_722;
                                                            }

                                                        }

                                                        if (if_cond_722) {

                                                            addusxx_g_offset = ((_loop_it_885 - 1) * 256);

                                                            addusxx_g_realblocksize = min((addusxx_g_ngms - addusxx_g_offset), 256);
                                                            {

                                                                {
                                                                    int _in_ofsbeta_0 = ofsbeta[(_loop_it_886 - 1)];
                                                                    int _out_addusxx_g_ijkb0;

                                                                    ///////////////////
                                                                    // Tasklet code (t_728)
                                                                    _out_addusxx_g_ijkb0 = _in_ofsbeta_0;
                                                                    ///////////////////

                                                                    addusxx_g_ijkb0 = _out_addusxx_g_ijkb0;
                                                                }

                                                            }
                                                            loopend_729 = ((1 + 256) - 1);


                                                            for (_loop_it_887 = 1; (_loop_it_887 < (loopend_729 + 1)); _loop_it_887 = (_loop_it_887 + 1)) {
                                                                {

                                                                    {
                                                                        dace::complex128 _out_addusxx_g_aux2;

                                                                        ///////////////////
                                                                        // Tasklet code (t_0)
                                                                        _out_addusxx_g_aux2 = (0.0 + (dace::complex128(0.0, 1.0) * 0.0));
                                                                        ///////////////////

                                                                        addusxx_g_aux2[(_loop_it_887 - 1)] = _out_addusxx_g_aux2;
                                                                    }

                                                                }

                                                            }

                                                            as_0 = (loopend_729 + 1);


                                                            loopend_731 = nh[(_loop_it_884 - 1)];


                                                            for (_loop_it_888 = 1; (_loop_it_888 < (loopend_731 + 1)); _loop_it_888 = (_loop_it_888 + 1)) {

                                                                addusxx_g_ikb = (addusxx_g_ijkb0 + _loop_it_888);

                                                                loopend_736 = ((1 + 256) - 1);


                                                                for (_loop_it_889 = 1; (_loop_it_889 < (loopend_736 + 1)); _loop_it_889 = (_loop_it_889 + 1)) {
                                                                    {

                                                                        {
                                                                            dace::complex128 _out_addusxx_g_aux1;

                                                                            ///////////////////
                                                                            // Tasklet code (t_0)
                                                                            _out_addusxx_g_aux1 = (0.0 + (dace::complex128(0.0, 1.0) * 0.0));
                                                                            ///////////////////

                                                                            addusxx_g_aux1[(_loop_it_889 - 1)] = _out_addusxx_g_aux1;
                                                                        }

                                                                    }

                                                                }

                                                                as_0 = (loopend_736 + 1);


                                                                loopend_738 = nh[(_loop_it_884 - 1)];


                                                                for (_loop_it_890 = 1; (_loop_it_890 < (loopend_738 + 1)); _loop_it_890 = (_loop_it_890 + 1)) {

                                                                    addusxx_g_jkb = (addusxx_g_ijkb0 + _loop_it_890);


                                                                    if (addusxx_g_add_complex) {

                                                                        for (_loop_it_891 = 1; (_loop_it_891 < (addusxx_g_realblocksize + 1)); _loop_it_891 = (_loop_it_891 + 1)) {

                                                                            ijtoh_at29 = ijtoh[(((_loop_it_888 + ((ijtoh_d0 * ijtoh_d1) * (_loop_it_884 - 1))) + (ijtoh_d0 * (_loop_it_890 - 1))) - 1)];

                                                                            {

                                                                                {
                                                                                    dace::complex128 _in_addusxx_g_aux1_0 = addusxx_g_aux1[(_loop_it_891 - 1)];
                                                                                    dace::complex128 _in_becpsi_c_0 = becpsi_k[((addusxx_g_jkb + (becpsi_k_d0 * (ibnd - offset_becpsi_k_d1))) - offset_becpsi_k_d0)];
                                                                                    dace::complex128 _in_qgm_0 = qgm[(((_loop_it_891 + addusxx_g_offset) + (dfftt_ngm * ((addusxx_g_nij + ijtoh_at29) - 1))) - 1)];
                                                                                    dace::complex128 _out_addusxx_g_aux1;

                                                                                    ///////////////////
                                                                                    // Tasklet code (t_0)
                                                                                    _out_addusxx_g_aux1 = (_in_addusxx_g_aux1_0 + (_in_qgm_0 * _in_becpsi_c_0));
                                                                                    ///////////////////

                                                                                    addusxx_g_aux1[(_loop_it_891 - 1)] = _out_addusxx_g_aux1;
                                                                                }

                                                                            }

                                                                        }

                                                                        ei0 = (addusxx_g_realblocksize + 1);

                                                                    } else {

                                                                        for (_loop_it_892 = 1; (_loop_it_892 < (addusxx_g_realblocksize + 1)); _loop_it_892 = (_loop_it_892 + 1)) {

                                                                            ijtoh_at30 = ijtoh[(((_loop_it_888 + ((ijtoh_d0 * ijtoh_d1) * (_loop_it_884 - 1))) + (ijtoh_d0 * (_loop_it_890 - 1))) - 1)];

                                                                            {

                                                                                {
                                                                                    dace::complex128 _in_addusxx_g_aux1_0 = addusxx_g_aux1[(_loop_it_892 - 1)];
                                                                                    double _in_becpsi_r_0 = becpsi_r[(addusxx_g_jkb - 1)];
                                                                                    dace::complex128 _in_qgm_0 = qgm[(((_loop_it_892 + addusxx_g_offset) + (dfftt_ngm * ((addusxx_g_nij + ijtoh_at30) - 1))) - 1)];
                                                                                    dace::complex128 _out_addusxx_g_aux1;

                                                                                    ///////////////////
                                                                                    // Tasklet code (t_0)
                                                                                    _out_addusxx_g_aux1 = (_in_addusxx_g_aux1_0 + (_in_qgm_0 * (_in_becpsi_r_0 + (dace::complex128(0.0, 1.0) * 0.0))));
                                                                                    ///////////////////

                                                                                    addusxx_g_aux1[(_loop_it_892 - 1)] = _out_addusxx_g_aux1;
                                                                                }

                                                                            }

                                                                        }

                                                                        ei0 = (addusxx_g_realblocksize + 1);

                                                                    }


                                                                }

                                                                addusxx_g_jh = (loopend_738 + 1);


                                                                addusxx_g_jh = addusxx_g_jh;


                                                                if (addusxx_g_add_complex) {

                                                                    for (_loop_it_893 = 1; (_loop_it_893 < (addusxx_g_realblocksize + 1)); _loop_it_893 = (_loop_it_893 + 1)) {
                                                                        {

                                                                            {
                                                                                dace::complex128 _in_addusxx_g_aux1_0 = addusxx_g_aux1[(_loop_it_893 - 1)];
                                                                                dace::complex128 _in_addusxx_g_aux2_0 = addusxx_g_aux2[(_loop_it_893 - 1)];
                                                                                dace::complex128 _in_becphi_c_0 = becxx_k[(((((becxx_k_d0 * becxx_k_d1) * (_loop_it_881 - 1)) + (becxx_k_d0 * (addusxx_g_ikb - 1))) + ikq) - 1)];
                                                                                dace::complex128 _in_becphi_c_1 = becxx_k[(((((becxx_k_d0 * becxx_k_d1) * (_loop_it_881 - 1)) + (becxx_k_d0 * (addusxx_g_ikb - 1))) + ikq) - 1)];
                                                                                dace::complex128 _out_addusxx_g_aux2;

                                                                                ///////////////////
                                                                                // Tasklet code (t_0)
                                                                                _out_addusxx_g_aux2 = (_in_addusxx_g_aux2_0 + (_in_addusxx_g_aux1_0 * conj(_in_becphi_c_0)));
                                                                                ///////////////////

                                                                                addusxx_g_aux2[(_loop_it_893 - 1)] = _out_addusxx_g_aux2;
                                                                            }

                                                                        }

                                                                    }

                                                                    ei0 = (addusxx_g_realblocksize + 1);

                                                                } else {

                                                                    for (_loop_it_894 = 1; (_loop_it_894 < (addusxx_g_realblocksize + 1)); _loop_it_894 = (_loop_it_894 + 1)) {
                                                                        {

                                                                            {
                                                                                dace::complex128 _in_addusxx_g_aux1_0 = addusxx_g_aux1[(_loop_it_894 - 1)];
                                                                                dace::complex128 _in_addusxx_g_aux2_0 = addusxx_g_aux2[(_loop_it_894 - 1)];
                                                                                double _in_becphi_r_0 = becphi_r[(addusxx_g_ikb - 1)];
                                                                                dace::complex128 _out_addusxx_g_aux2;

                                                                                ///////////////////
                                                                                // Tasklet code (t_0)
                                                                                _out_addusxx_g_aux2 = (_in_addusxx_g_aux2_0 + (_in_addusxx_g_aux1_0 * (_in_becphi_r_0 + (dace::complex128(0.0, 1.0) * 0.0))));
                                                                                ///////////////////

                                                                                addusxx_g_aux2[(_loop_it_894 - 1)] = _out_addusxx_g_aux2;
                                                                            }

                                                                        }

                                                                    }

                                                                    ei0 = (addusxx_g_realblocksize + 1);

                                                                }


                                                            }

                                                            addusxx_g_ih = (loopend_731 + 1);


                                                            addusxx_g_ih = addusxx_g_ih;


                                                            for (_loop_it_895 = 1; (_loop_it_895 < (addusxx_g_realblocksize + 1)); _loop_it_895 = (_loop_it_895 + 1)) {

                                                                mill_at31 = mill[(mill_d0 * ((_loop_it_895 + addusxx_g_offset) - 1))];

                                                                mill_at32 = mill[((mill_d0 * ((_loop_it_895 + addusxx_g_offset) - 1)) + 1)];

                                                                mill_at33 = mill[((mill_d0 * ((_loop_it_895 + addusxx_g_offset) - 1)) + 2)];

                                                                {

                                                                    {
                                                                        dace::complex128 _in_addusxx_g_aux2_0 = addusxx_g_aux2[(_loop_it_895 - 1)];
                                                                        dace::complex128 _in_addusxx_g_eigqts_0 = addusxx_g_eigqts[(_loop_it_886 - 1)];
                                                                        dace::complex128 _in_eigts1_0 = eigts1[(((eigts1_d0 * (_loop_it_886 - 1)) + mill_at31) - 1)];
                                                                        dace::complex128 _in_eigts2_0 = eigts2[(((eigts2_d0 * (_loop_it_886 - 1)) + mill_at32) - 1)];
                                                                        dace::complex128 _in_eigts3_0 = eigts3[(((eigts3_d0 * (_loop_it_886 - 1)) + mill_at33) - 1)];
                                                                        dace::complex128 _out_addusxx_g_aux2;

                                                                        ///////////////////
                                                                        // Tasklet code (t_0)
                                                                        _out_addusxx_g_aux2 = ((((_in_addusxx_g_aux2_0 * _in_addusxx_g_eigqts_0) * _in_eigts1_0) * _in_eigts2_0) * _in_eigts3_0);
                                                                        ///////////////////

                                                                        addusxx_g_aux2[(_loop_it_895 - 1)] = _out_addusxx_g_aux2;
                                                                    }

                                                                }

                                                            }

                                                            ei0 = (addusxx_g_realblocksize + 1);



                                                            if (addusxx_g_add_complex) {

                                                                loopend_764 = max((((addusxx_g_offset + addusxx_g_realblocksize) - (addusxx_g_offset + 1)) + 1), 0);


                                                                for (_loop_it_896 = 1; (_loop_it_896 < (loopend_764 + 1)); _loop_it_896 = (_loop_it_896 + 1)) {

                                                                    dfftt__nl_at34 = dfftt__nl[((_loop_it_896 + addusxx_g_offset) - 1)];

                                                                    {

                                                                        {
                                                                            dace::complex128 _in_addusxx_g_aux2_0 = addusxx_g_aux2[(_loop_it_896 - 1)];
                                                                            dace::complex128 _in_addusxx_g_rhoc_0 = rhoc[((dfftt__nl_at34 + (nrxxs * (_loop_it_881 - jstart))) - 1)];
                                                                            dace::complex128 _out_addusxx_g_rhoc;

                                                                            ///////////////////
                                                                            // Tasklet code (t_0)
                                                                            _out_addusxx_g_rhoc = (_in_addusxx_g_rhoc_0 + _in_addusxx_g_aux2_0);
                                                                            ///////////////////

                                                                            rhoc[((dfftt__nl_at34 + (nrxxs * (_loop_it_881 - jstart))) - 1)] = _out_addusxx_g_rhoc;
                                                                        }

                                                                    }

                                                                }

                                                                _doit_76 = (loopend_764 + 1);

                                                            } else {


                                                                if (addusxx_g_add_real) {

                                                                    loopend_771 = max((((addusxx_g_offset + addusxx_g_realblocksize) - (addusxx_g_offset + 1)) + 1), 0);


                                                                    for (_loop_it_897 = 1; (_loop_it_897 < (loopend_771 + 1)); _loop_it_897 = (_loop_it_897 + 1)) {

                                                                        dfftt__nl_at35 = dfftt__nl[((_loop_it_897 + addusxx_g_offset) - 1)];

                                                                        {

                                                                            {
                                                                                dace::complex128 _in_addusxx_g_aux2_0 = addusxx_g_aux2[(_loop_it_897 - 1)];
                                                                                dace::complex128 _in_addusxx_g_rhoc_0 = rhoc[((dfftt__nl_at35 + (nrxxs * (_loop_it_881 - jstart))) - 1)];
                                                                                dace::complex128 _out_addusxx_g_rhoc;

                                                                                ///////////////////
                                                                                // Tasklet code (t_0)
                                                                                _out_addusxx_g_rhoc = (_in_addusxx_g_rhoc_0 + _in_addusxx_g_aux2_0);
                                                                                ///////////////////

                                                                                rhoc[((dfftt__nl_at35 + (nrxxs * (_loop_it_881 - jstart))) - 1)] = _out_addusxx_g_rhoc;
                                                                            }

                                                                        }

                                                                    }

                                                                    _doit_77 = (loopend_771 + 1);


                                                                    if_cond_777 = ((gstart == 2) && (_loop_it_885 == 1));


                                                                    if (if_cond_777) {
                                                                        {

                                                                            {
                                                                                dace::complex128 _out_addusxx_g_aux2;

                                                                                ///////////////////
                                                                                // Tasklet code (t_780)
                                                                                _out_addusxx_g_aux2 = (0.0 + (dace::complex128(0.0, 1.0) * 0.0));
                                                                                ///////////////////

                                                                                addusxx_g_aux2[0] = _out_addusxx_g_aux2;
                                                                            }

                                                                        }
                                                                    }


                                                                    loopend_781 = max((((addusxx_g_offset + addusxx_g_realblocksize) - (addusxx_g_offset + 1)) + 1), 0);


                                                                    for (_loop_it_898 = 1; (_loop_it_898 < (loopend_781 + 1)); _loop_it_898 = (_loop_it_898 + 1)) {

                                                                        dfftt_nlm_at36 = dfftt_nlm[((_loop_it_898 + addusxx_g_offset) - 1)];

                                                                        {

                                                                            {
                                                                                dace::complex128 _in_addusxx_g_aux2_0 = addusxx_g_aux2[(_loop_it_898 - 1)];
                                                                                dace::complex128 _in_addusxx_g_aux2_1 = addusxx_g_aux2[(_loop_it_898 - 1)];
                                                                                dace::complex128 _in_addusxx_g_rhoc_0 = rhoc[((dfftt_nlm_at36 + (nrxxs * (_loop_it_881 - jstart))) - 1)];
                                                                                dace::complex128 _out_addusxx_g_rhoc;

                                                                                ///////////////////
                                                                                // Tasklet code (t_0)
                                                                                _out_addusxx_g_rhoc = (_in_addusxx_g_rhoc_0 + conj(_in_addusxx_g_aux2_0));
                                                                                ///////////////////

                                                                                rhoc[((dfftt_nlm_at36 + (nrxxs * (_loop_it_881 - jstart))) - 1)] = _out_addusxx_g_rhoc;
                                                                            }

                                                                        }

                                                                    }

                                                                    _doit_78 = (loopend_781 + 1);

                                                                } else {


                                                                    if (addusxx_g_add_imaginary) {

                                                                        loopend_788 = max((((addusxx_g_offset + addusxx_g_realblocksize) - (addusxx_g_offset + 1)) + 1), 0);


                                                                        for (_loop_it_899 = 1; (_loop_it_899 < (loopend_788 + 1)); _loop_it_899 = (_loop_it_899 + 1)) {

                                                                            dfftt__nl_at37 = dfftt__nl[((_loop_it_899 + addusxx_g_offset) - 1)];

                                                                            {

                                                                                {
                                                                                    dace::complex128 _in_addusxx_g_aux2_0 = addusxx_g_aux2[(_loop_it_899 - 1)];
                                                                                    dace::complex128 _in_addusxx_g_rhoc_0 = rhoc[((dfftt__nl_at37 + (nrxxs * (_loop_it_881 - jstart))) - 1)];
                                                                                    dace::complex128 _out_addusxx_g_rhoc;

                                                                                    ///////////////////
                                                                                    // Tasklet code (t_0)
                                                                                    _out_addusxx_g_rhoc = (_in_addusxx_g_rhoc_0 + ((0.0 + (dace::complex128(0.0, 1.0) * 1.0)) * _in_addusxx_g_aux2_0));
                                                                                    ///////////////////

                                                                                    rhoc[((dfftt__nl_at37 + (nrxxs * (_loop_it_881 - jstart))) - 1)] = _out_addusxx_g_rhoc;
                                                                                }

                                                                            }

                                                                        }

                                                                        _doit_79 = (loopend_788 + 1);


                                                                        if_cond_794 = ((gstart == 2) && (_loop_it_885 == 1));


                                                                        if (if_cond_794) {
                                                                            {

                                                                                {
                                                                                    dace::complex128 _out_addusxx_g_aux2;

                                                                                    ///////////////////
                                                                                    // Tasklet code (t_797)
                                                                                    _out_addusxx_g_aux2 = (0.0 + (dace::complex128(0.0, 1.0) * 0.0));
                                                                                    ///////////////////

                                                                                    addusxx_g_aux2[0] = _out_addusxx_g_aux2;
                                                                                }

                                                                            }
                                                                        }


                                                                        loopend_798 = max((((addusxx_g_offset + addusxx_g_realblocksize) - (addusxx_g_offset + 1)) + 1), 0);


                                                                        for (_loop_it_900 = 1; (_loop_it_900 < (loopend_798 + 1)); _loop_it_900 = (_loop_it_900 + 1)) {

                                                                            dfftt_nlm_at38 = dfftt_nlm[((_loop_it_900 + addusxx_g_offset) - 1)];

                                                                            {

                                                                                {
                                                                                    dace::complex128 _in_addusxx_g_aux2_0 = addusxx_g_aux2[(_loop_it_900 - 1)];
                                                                                    dace::complex128 _in_addusxx_g_aux2_1 = addusxx_g_aux2[(_loop_it_900 - 1)];
                                                                                    dace::complex128 _in_addusxx_g_rhoc_0 = rhoc[((dfftt_nlm_at38 + (nrxxs * (_loop_it_881 - jstart))) - 1)];
                                                                                    dace::complex128 _out_addusxx_g_rhoc;

                                                                                    ///////////////////
                                                                                    // Tasklet code (t_0)
                                                                                    _out_addusxx_g_rhoc = (_in_addusxx_g_rhoc_0 + ((0.0 + (dace::complex128(0.0, 1.0) * 1.0)) * conj(_in_addusxx_g_aux2_0)));
                                                                                    ///////////////////

                                                                                    rhoc[((dfftt_nlm_at38 + (nrxxs * (_loop_it_881 - jstart))) - 1)] = _out_addusxx_g_rhoc;
                                                                                }

                                                                            }

                                                                        }

                                                                        _doit_80 = (loopend_798 + 1);

                                                                    }

                                                                }

                                                            }

                                                        }


                                                    }

                                                    addusxx_g_na = (nat + 1);


                                                    addusxx_g_na = addusxx_g_na;


                                                }

                                                addusxx_g_iblock = (addusxx_g_numblock + 1);


                                                addusxx_g_iblock = addusxx_g_iblock;

                                            }


                                        }

                                        addusxx_g_nt = (nsp + 1);


                                        addusxx_g_nt = addusxx_g_nt;

                                        addusxx_g_aux2_allocated = 0;
                                        {

                                            {
                                                #pragma omp parallel for
                                                for (auto __i0 = 0; __i0 < 256; __i0 += 1) {
                                                    {
                                                        dace::complex128 _out;

                                                        ///////////////////
                                                        // Tasklet code (set_addusxx_g_aux2)
                                                        _out = 0;
                                                        ///////////////////

                                                        addusxx_g_aux2[__i0] = _out;
                                                    }
                                                }
                                            }

                                        }
                                        addusxx_g_aux1_allocated = 0;
                                        {

                                            {
                                                #pragma omp parallel for
                                                for (auto __i0 = 0; __i0 < 256; __i0 += 1) {
                                                    {
                                                        dace::complex128 _out;

                                                        ///////////////////
                                                        // Tasklet code (set_addusxx_g_aux1)
                                                        _out = 0;
                                                        ///////////////////

                                                        addusxx_g_aux1[__i0] = _out;
                                                    }
                                                }
                                            }

                                        }
                                        addusxx_g_eigqts_allocated = 0;
                                        {

                                            {
                                                #pragma omp parallel for
                                                for (auto __i0 = 0; __i0 < nat; __i0 += 1) {
                                                    {
                                                        dace::complex128 _out;

                                                        ///////////////////
                                                        // Tasklet code (set_addusxx_g_eigqts)
                                                        _out = 0;
                                                        ///////////////////

                                                        addusxx_g_eigqts[__i0] = _out;
                                                    }
                                                }
                                            }

                                        }
                                    }


                                }

                                jbnd = (jend + 1);


                                jbnd = jbnd;
                                {

                                    copy_rhoc_d_814_sdfg_230_3_2(__state, &rhoc[0], &rhoc_d[0], jblock, nrxxs);

                                }
                            }


                            for (_loop_it_901 = jstart; (_loop_it_901 < (jend + 1)); _loop_it_901 = (_loop_it_901 + 1)) {

                                for (_loop_it_902 = 1; (_loop_it_902 < (nrxxs + 1)); _loop_it_902 = (_loop_it_902 + 1)) {
                                    {

                                        {
                                            double _in_facb_d_0 = facb_d[(_loop_it_902 - 1)];
                                            dace::complex128 _in_rhoc_d_0 = rhoc_d[((_loop_it_902 + (nrxxs * (_loop_it_901 - jstart))) - 1)];
                                            double _in_x_occupation_d_0 = x_occupation_d[((_loop_it_901 + (x_occupation_d_d0 * (ik - 1))) - 1)];
                                            double _in_nqs_inv = nqs_inv;
                                            dace::complex128 _out_vc_d;

                                            ///////////////////
                                            // Tasklet code (t_0)
                                            _out_vc_d = ((((_in_facb_d_0 + (dace::complex128(0.0, 1.0) * 0.0)) * _in_rhoc_d_0) * (_in_x_occupation_d_0 + (dace::complex128(0.0, 1.0) * 0.0))) * (_in_nqs_inv + (dace::complex128(0.0, 1.0) * 0.0)));
                                            ///////////////////

                                            vc_d[((_loop_it_902 + (nrxxs * (_loop_it_901 - jstart))) - 1)] = _out_vc_d;
                                        }

                                    }

                                }

                                ir = (nrxxs + 1);


                                ir = ir;


                            }

                            jbnd = (jend + 1);


                            jbnd = jbnd;

                            if_cond_821 = (okvan[0] && (tqr[0] != true));


                            if (if_cond_821) {
                                {

                                    copy_vc_824_sdfg_289_0_2(__state, &vc_d[0], &vc[0], jblock, nrxxs);

                                }

                                for (_loop_it_903 = jstart; (_loop_it_903 < (jend + 1)); _loop_it_903 = (_loop_it_903 + 1)) {

                                    __al_26 = 0;

                                    __al_27 = 0;

                                    __al_28 = 0;

                                    __al_29 = 0;

                                    if_cond_831 = okvan[0];


                                    if (if_cond_831) {

                                        newdxx_g_ngms = dfftt_ngm;
                                        {

                                            {
                                                bool _out;

                                                ///////////////////
                                                // Tasklet code (set_newdxx_g_add_complex)
                                                _out = -1;
                                                ///////////////////

                                                newdxx_g_add_complex = _out;
                                            }
                                            {
                                                bool _out;

                                                ///////////////////
                                                // Tasklet code (set_newdxx_g_add_real)
                                                _out = -1;
                                                ///////////////////

                                                newdxx_g_add_real = _out;
                                            }
                                            {
                                                bool _out;

                                                ///////////////////
                                                // Tasklet code (set_newdxx_g_add_imaginary)
                                                _out = -1;
                                                ///////////////////

                                                newdxx_g_add_imaginary = _out;
                                            }

                                        }
                                        if_cond_835 = (((newdxx_g_add_complex || newdxx_g_add_real) || newdxx_g_add_imaginary) != true);


                                        if (if_cond_835) {
                                            {
                                                int __assoc_scalar_10;

                                                {
                                                    int _out;

                                                    ///////////////////
                                                    // Tasklet code (set___assoc_scalar_10)
                                                    _out = 1;
                                                    ///////////////////

                                                    __assoc_scalar_10 = _out;
                                                }

                                            }
                                        }


                                        if_cond_839 = ((gamma_only[0] != true) && (newdxx_g_add_real || newdxx_g_add_imaginary));


                                        if (if_cond_839) {
                                            {
                                                int __assoc_scalar_11;

                                                {
                                                    int _out;

                                                    ///////////////////
                                                    // Tasklet code (set___assoc_scalar_11)
                                                    _out = 2;
                                                    ///////////////////

                                                    __assoc_scalar_11 = _out;
                                                }

                                            }
                                        }


                                        if_cond_843 = (gamma_only[0] && newdxx_g_add_complex);


                                        if (if_cond_843) {
                                            {
                                                int __assoc_scalar_12;

                                                {
                                                    int _out;

                                                    ///////////////////
                                                    // Tasklet code (set___assoc_scalar_12)
                                                    _out = 3;
                                                    ///////////////////

                                                    __assoc_scalar_12 = _out;
                                                }

                                            }
                                        }


                                        if_cond_847 = (((newdxx_g_add_complex && (1 != true)) || (newdxx_g_add_real && (0 != true))) || (newdxx_g_add_imaginary && (0 != true)));


                                        if (if_cond_847) {
                                            {
                                                int __assoc_scalar_13;

                                                {
                                                    int _out;

                                                    ///////////////////
                                                    // Tasklet code (set___assoc_scalar_13)
                                                    _out = 2;
                                                    ///////////////////

                                                    __assoc_scalar_13 = _out;
                                                }

                                            }
                                        }


                                        auxvc_d0 = newdxx_g_ngms;

                                        auxvc_allocated = 1;

                                        newdxx_g_eigqts_d0 = nat;

                                        newdxx_g_eigqts_allocated = 1;


                                        for (_loop_it_904 = 1; (_loop_it_904 < (nat + 1)); _loop_it_904 = (_loop_it_904 + 1)) {


                                            for (_loop_it_905 = 1; (_loop_it_905 < (3 + 1)); _loop_it_905 = (_loop_it_905 + 1)) {
                                                {

                                                    {
                                                        double _in_tau_0 = tau[((_loop_it_905 + (tau_d0 * (_loop_it_904 - 1))) - 1)];
                                                        double _in_xkp_0 = xkp[(_loop_it_905 - 1)];
                                                        double _in_xkq_0 = xkq[(_loop_it_905 - 1)];
                                                        double _out__mask_13;

                                                        ///////////////////
                                                        // Tasklet code (t_0)
                                                        _out__mask_13 = ((_in_xkp_0 - _in_xkq_0) * _in_tau_0);
                                                        ///////////////////

                                                        _mask_13[(_loop_it_905 - 1)] = _out__mask_13;
                                                    }

                                                }

                                            }

                                            ei0 = 4;

                                            {
                                                double _QQred_lift_5;
                                                double newdxx_g_arg;

                                                reduce_46_6_2(__state, &_mask_13[0], &_QQred_lift_5);
                                                {
                                                    double _in__QQred_lift_5 = _QQred_lift_5;
                                                    double _out;

                                                    ///////////////////
                                                    // Tasklet code (set_newdxx_g_arg)
                                                    _out = (_in__QQred_lift_5 * 6.283185307179586);
                                                    ///////////////////

                                                    newdxx_g_arg = _out;
                                                }
                                                {
                                                    double _in_newdxx_g_arg = newdxx_g_arg;
                                                    dace::complex128 _out_newdxx_g_eigqts;

                                                    ///////////////////
                                                    // Tasklet code (t_859)
                                                    _out_newdxx_g_eigqts = (cos(_in_newdxx_g_arg) + (dace::complex128(0.0, 1.0) * (- sin(_in_newdxx_g_arg))));
                                                    ///////////////////

                                                    newdxx_g_eigqts[(_loop_it_904 - 1)] = _out_newdxx_g_eigqts;
                                                }

                                            }

                                        }

                                        newdxx_g_na = (nat + 1);


                                        newdxx_g_na = newdxx_g_na;

                                        auxvc = new dace::complex128 DACE_ALIGN(64)[newdxx_g_ngms];

                                        if (newdxx_g_add_complex) {

                                            for (_loop_it_906 = 1; (_loop_it_906 < (newdxx_g_ngms + 1)); _loop_it_906 = (_loop_it_906 + 1)) {

                                                dfftt__nl_at39 = dfftt__nl[(_loop_it_906 - 1)];

                                                {

                                                    {
                                                        dace::complex128 _in_newdxx_g_vc_0 = vc[((dfftt__nl_at39 + (nrxxs * (_loop_it_903 - jstart))) - 1)];
                                                        dace::complex128 _out_auxvc;

                                                        ///////////////////
                                                        // Tasklet code (t_0)
                                                        _out_auxvc = _in_newdxx_g_vc_0;
                                                        ///////////////////

                                                        auxvc[(_loop_it_906 - 1)] = _out_auxvc;
                                                    }

                                                }

                                            }

                                            ei0 = (newdxx_g_ngms + 1);

                                            {

                                                {
                                                    double _in_omega = omega[0];
                                                    double _out;

                                                    ///////////////////
                                                    // Tasklet code (set_fact)
                                                    _out = _in_omega;
                                                    ///////////////////

                                                    fact = _out;
                                                }

                                            }
                                        } else {


                                            if (newdxx_g_add_real) {

                                                for (_loop_it_907 = 1; (_loop_it_907 < (newdxx_g_ngms + 1)); _loop_it_907 = (_loop_it_907 + 1)) {

                                                    dfftt__nl_at40 = dfftt__nl[(_loop_it_907 - 1)];

                                                    dfftt_nlm_at41 = dfftt_nlm[(_loop_it_907 - 1)];
                                                    {

                                                        {
                                                            dace::complex128 _in_newdxx_g_vc_0 = vc[((dfftt__nl_at40 + (nrxxs * (_loop_it_903 - jstart))) - 1)];
                                                            dace::complex128 _in_newdxx_g_vc_1 = vc[((dfftt_nlm_at41 + (nrxxs * (_loop_it_903 - jstart))) - 1)];
                                                            dace::complex128 _in_newdxx_g_vc_2 = vc[((dfftt__nl_at40 + (nrxxs * (_loop_it_903 - jstart))) - 1)];
                                                            dace::complex128 _in_newdxx_g_vc_3 = vc[((dfftt_nlm_at41 + (nrxxs * (_loop_it_903 - jstart))) - 1)];
                                                            dace::complex128 _out_fp;

                                                            ///////////////////
                                                            // Tasklet code (t_873)
                                                            _out_fp = ((_in_newdxx_g_vc_0 + _in_newdxx_g_vc_1) / (2.0 + (dace::complex128(0.0, 1.0) * 0.0)));
                                                            ///////////////////

                                                            fp = _out_fp;
                                                        }

                                                    }
                                                    dfftt__nl_at42 = dfftt__nl[(_loop_it_907 - 1)];

                                                    dfftt_nlm_at43 = dfftt_nlm[(_loop_it_907 - 1)];
                                                    {

                                                        {
                                                            dace::complex128 _in_newdxx_g_vc_0 = vc[((dfftt__nl_at42 + (nrxxs * (_loop_it_903 - jstart))) - 1)];
                                                            dace::complex128 _in_newdxx_g_vc_1 = vc[((dfftt_nlm_at43 + (nrxxs * (_loop_it_903 - jstart))) - 1)];
                                                            dace::complex128 _in_newdxx_g_vc_2 = vc[((dfftt__nl_at42 + (nrxxs * (_loop_it_903 - jstart))) - 1)];
                                                            dace::complex128 _in_newdxx_g_vc_3 = vc[((dfftt_nlm_at43 + (nrxxs * (_loop_it_903 - jstart))) - 1)];
                                                            dace::complex128 _out_fm;

                                                            ///////////////////
                                                            // Tasklet code (t_876)
                                                            _out_fm = ((_in_newdxx_g_vc_0 - _in_newdxx_g_vc_1) / (2.0 + (dace::complex128(0.0, 1.0) * 0.0)));
                                                            ///////////////////

                                                            fm = _out_fm;
                                                        }
                                                        {
                                                            dace::complex128 _in_fm = fm;
                                                            dace::complex128 _in_fp = fp;
                                                            dace::complex128 _out_auxvc;

                                                            ///////////////////
                                                            // Tasklet code (t_877)
                                                            _out_auxvc = (_in_fp.real() + (dace::complex128(0.0, 1.0) * _in_fm.imag()));
                                                            ///////////////////

                                                            auxvc[(_loop_it_907 - 1)] = _out_auxvc;
                                                        }

                                                    }

                                                }

                                                newdxx_g_ig = (newdxx_g_ngms + 1);


                                                newdxx_g_ig = newdxx_g_ig;
                                                {

                                                    {
                                                        double _in_omega = omega[0];
                                                        double _out;

                                                        ///////////////////
                                                        // Tasklet code (set_fact)
                                                        _out = (_in_omega * 2.0);
                                                        ///////////////////

                                                        fact = _out;
                                                    }

                                                }
                                            } else {


                                                if (newdxx_g_add_imaginary) {

                                                    for (_loop_it_908 = 1; (_loop_it_908 < (newdxx_g_ngms + 1)); _loop_it_908 = (_loop_it_908 + 1)) {

                                                        dfftt__nl_at44 = dfftt__nl[(_loop_it_908 - 1)];

                                                        dfftt_nlm_at45 = dfftt_nlm[(_loop_it_908 - 1)];
                                                        {

                                                            {
                                                                dace::complex128 _in_newdxx_g_vc_0 = vc[((dfftt__nl_at44 + (nrxxs * (_loop_it_903 - jstart))) - 1)];
                                                                dace::complex128 _in_newdxx_g_vc_1 = vc[((dfftt_nlm_at45 + (nrxxs * (_loop_it_903 - jstart))) - 1)];
                                                                dace::complex128 _in_newdxx_g_vc_2 = vc[((dfftt__nl_at44 + (nrxxs * (_loop_it_903 - jstart))) - 1)];
                                                                dace::complex128 _in_newdxx_g_vc_3 = vc[((dfftt_nlm_at45 + (nrxxs * (_loop_it_903 - jstart))) - 1)];
                                                                dace::complex128 _out_fp;

                                                                ///////////////////
                                                                // Tasklet code (t_886)
                                                                _out_fp = ((_in_newdxx_g_vc_0 + _in_newdxx_g_vc_1) / (2.0 + (dace::complex128(0.0, 1.0) * 0.0)));
                                                                ///////////////////

                                                                fp = _out_fp;
                                                            }

                                                        }
                                                        dfftt__nl_at46 = dfftt__nl[(_loop_it_908 - 1)];

                                                        dfftt_nlm_at47 = dfftt_nlm[(_loop_it_908 - 1)];
                                                        {

                                                            {
                                                                dace::complex128 _in_newdxx_g_vc_0 = vc[((dfftt__nl_at46 + (nrxxs * (_loop_it_903 - jstart))) - 1)];
                                                                dace::complex128 _in_newdxx_g_vc_1 = vc[((dfftt_nlm_at47 + (nrxxs * (_loop_it_903 - jstart))) - 1)];
                                                                dace::complex128 _in_newdxx_g_vc_2 = vc[((dfftt__nl_at46 + (nrxxs * (_loop_it_903 - jstart))) - 1)];
                                                                dace::complex128 _in_newdxx_g_vc_3 = vc[((dfftt_nlm_at47 + (nrxxs * (_loop_it_903 - jstart))) - 1)];
                                                                dace::complex128 _out_fm;

                                                                ///////////////////
                                                                // Tasklet code (t_889)
                                                                _out_fm = ((_in_newdxx_g_vc_0 - _in_newdxx_g_vc_1) / (2.0 + (dace::complex128(0.0, 1.0) * 0.0)));
                                                                ///////////////////

                                                                fm = _out_fm;
                                                            }
                                                            {
                                                                dace::complex128 _in_fm = fm;
                                                                dace::complex128 _in_fp = fp;
                                                                dace::complex128 _out_auxvc;

                                                                ///////////////////
                                                                // Tasklet code (t_890)
                                                                _out_auxvc = (_in_fp.imag() + (dace::complex128(0.0, 1.0) * (- _in_fm.real())));
                                                                ///////////////////

                                                                auxvc[(_loop_it_908 - 1)] = _out_auxvc;
                                                            }

                                                        }

                                                    }

                                                    newdxx_g_ig = (newdxx_g_ngms + 1);


                                                    newdxx_g_ig = newdxx_g_ig;
                                                    {

                                                        {
                                                            double _in_omega = omega[0];
                                                            double _out;

                                                            ///////////////////
                                                            // Tasklet code (set_fact)
                                                            _out = (_in_omega * 2.0);
                                                            ///////////////////

                                                            fact = _out;
                                                        }

                                                    }
                                                }

                                            }

                                        }


                                        newdxx_g_numblock = dace::math::ifloor((newdxx_g_ngms + 255) / 256);

                                        newdxx_g_aux1_d0 = 256;

                                        newdxx_g_aux1_allocated = 1;

                                        newdxx_g_aux2_d0 = 256;

                                        newdxx_g_aux2_allocated = 1;


                                        for (_loop_it_909 = 1; (_loop_it_909 < (newdxx_g_numblock + 1)); _loop_it_909 = (_loop_it_909 + 1)) {

                                            newdxx_g_offset = ((_loop_it_909 - 1) * 256);

                                            newdxx_g_realblocksize = min((newdxx_g_ngms - newdxx_g_offset), 256);


                                            for (_loop_it_910 = 1; (_loop_it_910 < (nat + 1)); _loop_it_910 = (_loop_it_910 + 1)) {

                                                newdxx_g_nt = ityp[(_loop_it_910 - 1)];


                                                if (upf_tvanp) {

                                                    newdxx_g_nij = nij_type[(newdxx_g_nt - 1)];
                                                    {

                                                        {
                                                            int _in_ofsbeta_0 = ofsbeta[(_loop_it_910 - 1)];
                                                            int _out_newdxx_g_ijkb0;

                                                            ///////////////////
                                                            // Tasklet code (t_909)
                                                            _out_newdxx_g_ijkb0 = _in_ofsbeta_0;
                                                            ///////////////////

                                                            newdxx_g_ijkb0 = _out_newdxx_g_ijkb0;
                                                        }

                                                    }
                                                    loopend_910 = (((newdxx_g_offset + newdxx_g_realblocksize) - (newdxx_g_offset + 1)) + 1);


                                                    for (_loop_it_911 = 1; (_loop_it_911 < (loopend_910 + 1)); _loop_it_911 = (_loop_it_911 + 1)) {

                                                        mill_at48 = mill[(mill_d0 * ((_loop_it_911 + newdxx_g_offset) - 1))];

                                                        mill_at49 = mill[((mill_d0 * ((_loop_it_911 + newdxx_g_offset) - 1)) + 1)];

                                                        mill_at50 = mill[((mill_d0 * ((_loop_it_911 + newdxx_g_offset) - 1)) + 2)];

                                                        {

                                                            {
                                                                dace::complex128 _in_auxvc_0 = auxvc[((_loop_it_911 + newdxx_g_offset) - 1)];
                                                                dace::complex128 _in_auxvc_1 = auxvc[((_loop_it_911 + newdxx_g_offset) - 1)];
                                                                dace::complex128 _in_eigts1_0 = eigts1[(((eigts1_d0 * (_loop_it_910 - 1)) + mill_at48) - 1)];
                                                                dace::complex128 _in_eigts2_0 = eigts2[(((eigts2_d0 * (_loop_it_910 - 1)) + mill_at49) - 1)];
                                                                dace::complex128 _in_eigts3_0 = eigts3[(((eigts3_d0 * (_loop_it_910 - 1)) + mill_at50) - 1)];
                                                                dace::complex128 _in_newdxx_g_eigqts_0 = newdxx_g_eigqts[(_loop_it_910 - 1)];
                                                                dace::complex128 _out_newdxx_g_aux2;

                                                                ///////////////////
                                                                // Tasklet code (t_0)
                                                                _out_newdxx_g_aux2 = ((((conj(_in_auxvc_0) * _in_newdxx_g_eigqts_0) * _in_eigts1_0) * _in_eigts2_0) * _in_eigts3_0);
                                                                ///////////////////

                                                                newdxx_g_aux2[(_loop_it_911 - 1)] = _out_newdxx_g_aux2;
                                                            }

                                                        }

                                                    }

                                                    ei0 = (loopend_910 + 1);


                                                    loopend_916 = nh[(newdxx_g_nt - 1)];


                                                    for (_loop_it_912 = 1; (_loop_it_912 < (loopend_916 + 1)); _loop_it_912 = (_loop_it_912 + 1)) {

                                                        newdxx_g_ikb = (newdxx_g_ijkb0 + _loop_it_912);

                                                        loopend_921 = ((1 + 256) - 1);


                                                        for (_loop_it_913 = 1; (_loop_it_913 < (loopend_921 + 1)); _loop_it_913 = (_loop_it_913 + 1)) {
                                                            {

                                                                {
                                                                    dace::complex128 _out_newdxx_g_aux1;

                                                                    ///////////////////
                                                                    // Tasklet code (t_0)
                                                                    _out_newdxx_g_aux1 = (0.0 + (dace::complex128(0.0, 1.0) * 0.0));
                                                                    ///////////////////

                                                                    newdxx_g_aux1[(_loop_it_913 - 1)] = _out_newdxx_g_aux1;
                                                                }

                                                            }

                                                        }

                                                        as_0 = (loopend_921 + 1);


                                                        loopend_923 = nh[(newdxx_g_nt - 1)];


                                                        for (_loop_it_914 = 1; (_loop_it_914 < (loopend_923 + 1)); _loop_it_914 = (_loop_it_914 + 1)) {

                                                            newdxx_g_jkb = (newdxx_g_ijkb0 + _loop_it_914);

                                                            if_cond_928 = gamma_only[0];


                                                            if (if_cond_928) {

                                                                for (_loop_it_915 = 1; (_loop_it_915 < (newdxx_g_realblocksize + 1)); _loop_it_915 = (_loop_it_915 + 1)) {

                                                                    ijtoh_at51 = ijtoh[(((_loop_it_912 + ((ijtoh_d0 * ijtoh_d1) * (newdxx_g_nt - 1))) + (ijtoh_d0 * (_loop_it_914 - 1))) - 1)];

                                                                    {

                                                                        {
                                                                            double _in_becphi_r_0 = becphi_r[(newdxx_g_jkb - 1)];
                                                                            dace::complex128 _in_newdxx_g_aux1_0 = newdxx_g_aux1[(_loop_it_915 - 1)];
                                                                            dace::complex128 _in_qgm_0 = qgm[(((_loop_it_915 + (dfftt_ngm * ((ijtoh_at51 + newdxx_g_nij) - 1))) + newdxx_g_offset) - 1)];
                                                                            dace::complex128 _in_qgm_1 = qgm[(((_loop_it_915 + (dfftt_ngm * ((ijtoh_at51 + newdxx_g_nij) - 1))) + newdxx_g_offset) - 1)];
                                                                            dace::complex128 _out_newdxx_g_aux1;

                                                                            ///////////////////
                                                                            // Tasklet code (t_0)
                                                                            _out_newdxx_g_aux1 = (_in_newdxx_g_aux1_0 + ((_in_becphi_r_0 + (dace::complex128(0.0, 1.0) * 0.0)) * conj(_in_qgm_0)));
                                                                            ///////////////////

                                                                            newdxx_g_aux1[(_loop_it_915 - 1)] = _out_newdxx_g_aux1;
                                                                        }

                                                                    }

                                                                }

                                                                ei0 = (newdxx_g_realblocksize + 1);

                                                            } else {

                                                                for (_loop_it_916 = 1; (_loop_it_916 < (newdxx_g_realblocksize + 1)); _loop_it_916 = (_loop_it_916 + 1)) {

                                                                    ijtoh_at52 = ijtoh[(((_loop_it_912 + ((ijtoh_d0 * ijtoh_d1) * (newdxx_g_nt - 1))) + (ijtoh_d0 * (_loop_it_914 - 1))) - 1)];

                                                                    {

                                                                        {
                                                                            dace::complex128 _in_becphi_c_0 = becxx_k[(((((becxx_k_d0 * becxx_k_d1) * (_loop_it_903 - 1)) + (becxx_k_d0 * (newdxx_g_jkb - 1))) + ikq) - 1)];
                                                                            dace::complex128 _in_newdxx_g_aux1_0 = newdxx_g_aux1[(_loop_it_916 - 1)];
                                                                            dace::complex128 _in_qgm_0 = qgm[(((_loop_it_916 + (dfftt_ngm * ((ijtoh_at52 + newdxx_g_nij) - 1))) + newdxx_g_offset) - 1)];
                                                                            dace::complex128 _in_qgm_1 = qgm[(((_loop_it_916 + (dfftt_ngm * ((ijtoh_at52 + newdxx_g_nij) - 1))) + newdxx_g_offset) - 1)];
                                                                            dace::complex128 _out_newdxx_g_aux1;

                                                                            ///////////////////
                                                                            // Tasklet code (t_0)
                                                                            _out_newdxx_g_aux1 = (_in_newdxx_g_aux1_0 + (_in_becphi_c_0 * conj(_in_qgm_0)));
                                                                            ///////////////////

                                                                            newdxx_g_aux1[(_loop_it_916 - 1)] = _out_newdxx_g_aux1;
                                                                        }

                                                                    }

                                                                }

                                                                ei0 = (newdxx_g_realblocksize + 1);

                                                            }


                                                        }

                                                        newdxx_g_jh = (loopend_923 + 1);


                                                        newdxx_g_jh = newdxx_g_jh;
                                                        {
                                                            dace::complex128 _QQred_lift_6;

                                                            dot_product__QQred_lift_6_938_sdfg_321_8_7(__state, &newdxx_g_aux2[0], &newdxx_g_aux1[0], _QQred_lift_6, newdxx_g_realblocksize);
                                                            {
                                                                dace::complex128 _in_newdxx_g_deexx_0 = deexx[((newdxx_g_ikb + (nkb * (_loop_it_871 - 1))) - 1)];
                                                                dace::complex128 _in__QQred_lift_6 = _QQred_lift_6;
                                                                double _in_fact = fact;
                                                                dace::complex128 _out_newdxx_g_deexx;

                                                                ///////////////////
                                                                // Tasklet code (t_939)
                                                                _out_newdxx_g_deexx = (_in_newdxx_g_deexx_0 + ((_in_fact + (dace::complex128(0.0, 1.0) * 0.0)) * _in__QQred_lift_6));
                                                                ///////////////////

                                                                deexx[((newdxx_g_ikb + (nkb * (_loop_it_871 - 1))) - 1)] = _out_newdxx_g_deexx;
                                                            }

                                                        }
                                                        if_cond_940 = ((gamma_only[0] && (gstart == 2)) && (_loop_it_909 == 1));


                                                        if (if_cond_940) {
                                                            {

                                                                {
                                                                    dace::complex128 _in_newdxx_g_aux1_0 = newdxx_g_aux1[0];
                                                                    dace::complex128 _in_newdxx_g_aux2_0 = newdxx_g_aux2[0];
                                                                    dace::complex128 _in_newdxx_g_aux2_1 = newdxx_g_aux2[0];
                                                                    dace::complex128 _in_newdxx_g_deexx_0 = deexx[((newdxx_g_ikb + (nkb * (_loop_it_871 - 1))) - 1)];
                                                                    double _in_omega = omega[0];
                                                                    dace::complex128 _out_newdxx_g_deexx;

                                                                    ///////////////////
                                                                    // Tasklet code (t_943)
                                                                    _out_newdxx_g_deexx = (_in_newdxx_g_deexx_0 - (((_in_omega + (dace::complex128(0.0, 1.0) * 0.0)) * conj(_in_newdxx_g_aux2_0)) * _in_newdxx_g_aux1_0));
                                                                    ///////////////////

                                                                    deexx[((newdxx_g_ikb + (nkb * (_loop_it_871 - 1))) - 1)] = _out_newdxx_g_deexx;
                                                                }

                                                            }
                                                        }


                                                    }

                                                    newdxx_g_ih = (loopend_916 + 1);


                                                    newdxx_g_ih = newdxx_g_ih;

                                                }


                                            }

                                            newdxx_g_na = (nat + 1);


                                            newdxx_g_na = newdxx_g_na;


                                        }

                                        newdxx_g_iblock = (newdxx_g_numblock + 1);


                                        newdxx_g_iblock = newdxx_g_iblock;

                                        newdxx_g_aux2_allocated = 0;
                                        {

                                            {
                                                #pragma omp parallel for
                                                for (auto __i0 = 0; __i0 < 256; __i0 += 1) {
                                                    {
                                                        dace::complex128 _out;

                                                        ///////////////////
                                                        // Tasklet code (set_newdxx_g_aux2)
                                                        _out = 0;
                                                        ///////////////////

                                                        newdxx_g_aux2[__i0] = _out;
                                                    }
                                                }
                                            }

                                        }
                                        newdxx_g_aux1_allocated = 0;
                                        {

                                            {
                                                #pragma omp parallel for
                                                for (auto __i0 = 0; __i0 < 256; __i0 += 1) {
                                                    {
                                                        dace::complex128 _out;

                                                        ///////////////////
                                                        // Tasklet code (set_newdxx_g_aux1)
                                                        _out = 0;
                                                        ///////////////////

                                                        newdxx_g_aux1[__i0] = _out;
                                                    }
                                                }
                                            }

                                        }
                                        newdxx_g_eigqts_allocated = 0;
                                        {

                                            {
                                                #pragma omp parallel for
                                                for (auto __i0 = 0; __i0 < nat; __i0 += 1) {
                                                    {
                                                        dace::complex128 _out;

                                                        ///////////////////
                                                        // Tasklet code (set_newdxx_g_eigqts)
                                                        _out = 0;
                                                        ///////////////////

                                                        newdxx_g_eigqts[__i0] = _out;
                                                    }
                                                }
                                            }

                                        }
                                        auxvc_allocated = 0;
                                        {

                                            {
                                                #pragma omp parallel for
                                                for (auto __i0 = 0; __i0 < newdxx_g_ngms; __i0 += 1) {
                                                    {
                                                        dace::complex128 _out;

                                                        ///////////////////
                                                        // Tasklet code (set_auxvc)
                                                        _out = 0;
                                                        ///////////////////

                                                        auxvc[__i0] = _out;
                                                    }
                                                }
                                            }
                                            delete[] auxvc;

                                        }
                                    }


                                }

                                jbnd = (jend + 1);


                                jbnd = jbnd;
                                {

                                    copy_vc_d_956_sdfg_289_3_2(__state, &vc[0], &vc_d[0], jblock, nrxxs);

                                }
                            }


                            for (_loop_it_917 = jstart; (_loop_it_917 < (jend + 1)); _loop_it_917 = (_loop_it_917 + many_fft[0])) {

                                jcurr = min(many_fft[0], ((jend - _loop_it_917) + 1));

                                {
                                    // MANUAL FIX: removed a spurious ``vc_d = new`` realloc here (same SDFG
                                    // lowering bug as rhoc_d above) -- it discarded the computed vc before
                                    // this invfft. pvc_d must slice the EXISTING vc_d.
                                    dace::complex128* pvc_d;
                                    pvc_d = &vc_d[(nrxxs * (_loop_it_917 - jstart))];

                                    dace_libraries_fft_algorithms_dft_idft_explicit_333_2_4(__state, &pvc_d[0], &pvc_d[0], jcurr, nrxxs);

                                }

                            }

                            jbnd = ((jend + many_fft[0]) - ((jend - jstart) % many_fft[0]));


                            jbnd = jbnd;

                            if_cond_964 = (okvan[0] && tqr[0]);


                            if (if_cond_964) {
                                {

                                    copy_vc_967_sdfg_336_0_2(__state, &vc_d[0], &vc[0], jblock, nrxxs);

                                }

                                for (_loop_it_918 = jstart; (_loop_it_918 < (jend + 1)); _loop_it_918 = (_loop_it_918 + 1)) {
                                    {

                                        {
                                            int _in_dfftt_nr2 = dfftt_nr2;
                                            int _in_dfftt_nr1 = dfftt_nr1;
                                            int _in_dfftt_nr3 = dfftt_nr3;
                                            double _in_omega = omega[0];
                                            double _out;

                                            ///////////////////
                                            // Tasklet code (set_domega)
                                            _out = (_in_omega / dace::float64(((_in_dfftt_nr1 * _in_dfftt_nr2) * _in_dfftt_nr3)));
                                            ///////////////////

                                            domega = _out;
                                        }

                                    }

                                    for (_loop_it_919 = 1; (_loop_it_919 < (nat + 1)); _loop_it_919 = (_loop_it_919 + 1)) {

                                        newdxx_r_mbia = tabxx_maxbox[(_loop_it_919 - 1)];

                                        if_cond_973 = (newdxx_r_mbia != 0);


                                        if (if_cond_973) {

                                            newdxx_r_nt = ityp[(_loop_it_919 - 1)];


                                            if (upf_tvanp) {

                                                loopend_978 = nh[(newdxx_r_nt - 1)];


                                                for (_loop_it_920 = 1; (_loop_it_920 < (loopend_978 + 1)); _loop_it_920 = (_loop_it_920 + 1)) {

                                                    loopend_981 = nh[(newdxx_r_nt - 1)];


                                                    for (_loop_it_921 = 1; (_loop_it_921 < (loopend_981 + 1)); _loop_it_921 = (_loop_it_921 + 1)) {
                                                        {

                                                            {
                                                                int _in_ofsbeta_0 = ofsbeta[(_loop_it_919 - 1)];
                                                                int _out_newdxx_r_ijkb0;

                                                                ///////////////////
                                                                // Tasklet code (t_985)
                                                                _out_newdxx_r_ijkb0 = _in_ofsbeta_0;
                                                                ///////////////////

                                                                newdxx_r_ijkb0 = _out_newdxx_r_ijkb0;
                                                            }

                                                        }
                                                        newdxx_r_ikb = (newdxx_r_ijkb0 + _loop_it_920);

                                                        newdxx_r_jkb = (newdxx_r_ijkb0 + _loop_it_921);
                                                        {

                                                            {
                                                                dace::complex128 _out;

                                                                ///////////////////
                                                                // Tasklet code (set_aux)
                                                                _out = (0.0 + (dace::complex128(0.0, 1.0) * 0.0));
                                                                ///////////////////

                                                                aux = _out;
                                                            }

                                                        }

                                                        for (_loop_it_922 = 1; (_loop_it_922 < (newdxx_r_mbia + 1)); _loop_it_922 = (_loop_it_922 + 1)) {

                                                            ijtoh_at53 = ijtoh[(((_loop_it_920 + ((ijtoh_d0 * ijtoh_d1) * (newdxx_r_nt - 1))) + (ijtoh_d0 * (_loop_it_921 - 1))) - 1)];

                                                            tabxx_box_at54 = tabxx_box[((_loop_it_919 + (tabxx_box_d0 * (_loop_it_922 - 1))) - 1)];

                                                            {

                                                                {
                                                                    double _in_tabxx_qr_0 = tabxx_qr[(((_loop_it_919 + ((tabxx_qr_d0 * tabxx_qr_d1) * (ijtoh_at53 - 1))) + (tabxx_qr_d0 * (_loop_it_922 - 1))) - 1)];
                                                                    dace::complex128 _in_vc_0 = vc[(((nrxxs * (_loop_it_918 - jstart)) + tabxx_box_at54) - 1)];
                                                                    dace::complex128 _in_aux = aux;
                                                                    dace::complex128 _out_aux;

                                                                    ///////////////////
                                                                    // Tasklet code (t_0)
                                                                    _out_aux = (_in_aux + ((_in_tabxx_qr_0 + (dace::complex128(0.0, 1.0) * 0.0)) * _in_vc_0));
                                                                    ///////////////////

                                                                    aux = _out_aux;
                                                                }

                                                            }

                                                        }

                                                        newdxx_r_ir = (newdxx_r_mbia + 1);


                                                        newdxx_r_ir = newdxx_r_ir;
                                                        {

                                                            {
                                                                dace::complex128 _in_newdxx_r_becphi_0 = becxx_k[(((((becxx_k_d0 * becxx_k_d1) * (_loop_it_918 - 1)) + (becxx_k_d0 * (newdxx_r_jkb - 1))) + ikq) - 1)];
                                                                dace::complex128 _in_newdxx_r_deexx_0 = deexx[((newdxx_r_ikb + (nkb * (_loop_it_871 - 1))) - 1)];
                                                                dace::complex128 _in_aux = aux;
                                                                double _in_domega = domega;
                                                                dace::complex128 _out_newdxx_r_deexx;

                                                                ///////////////////
                                                                // Tasklet code (t_994)
                                                                _out_newdxx_r_deexx = (_in_newdxx_r_deexx_0 + ((_in_newdxx_r_becphi_0 * (_in_domega + (dace::complex128(0.0, 1.0) * 0.0))) * _in_aux));
                                                                ///////////////////

                                                                deexx[((newdxx_r_ikb + (nkb * (_loop_it_871 - 1))) - 1)] = _out_newdxx_r_deexx;
                                                            }

                                                        }

                                                    }

                                                    newdxx_r_jh = (loopend_981 + 1);


                                                    newdxx_r_jh = newdxx_r_jh;


                                                }

                                                newdxx_r_ih = (loopend_978 + 1);


                                                newdxx_r_ih = newdxx_r_ih;

                                            }

                                        }


                                    }

                                    newdxx_r_ia = (nat + 1);


                                    newdxx_r_ia = newdxx_r_ia;


                                }

                                jbnd = (jend + 1);


                                jbnd = jbnd;
                                {

                                    copy_vc_d_1003_sdfg_336_3_2(__state, &vc[0], &vc_d[0], jblock, nrxxs);

                                }
                            }


                            if_cond_1005 = okpaw[0];


                            if (if_cond_1005) {
                                {

                                    copy_vc_1008_sdfg_349_0_2(__state, &vc_d[0], &vc[0], jblock, nrxxs);

                                }

                                for (_loop_it_923 = jstart; (_loop_it_923 < (jend + 1)); _loop_it_923 = (_loop_it_923 + 1)) {
                                    {

                                        {
                                            double _in_x_occupation_0 = x_occupation[((_loop_it_923 + (x_occupation_d0 * (ik - 1))) - 1)];
                                            double _out___assoc_scalar_14;

                                            ///////////////////
                                            // Tasklet code (t_1011)
                                            _out___assoc_scalar_14 = (_in_x_occupation_0 / dace::float64(nqs));
                                            ///////////////////

                                            __assoc_scalar_14 = _out___assoc_scalar_14;
                                        }

                                    }
                                    if_cond_1012 = (paw_has_init_paw_fockrnl[0] != true);


                                    if (if_cond_1012) {
                                        {
                                            int __assoc_scalar_15;

                                            {
                                                int _out;

                                                ///////////////////
                                                // Tasklet code (set___assoc_scalar_15)
                                                _out = 1;
                                                ///////////////////

                                                __assoc_scalar_15 = _out;
                                            }

                                        }
                                    }


                                    if_cond_1016 = ionode[0];


                                    if (if_cond_1016) {

                                        for (_loop_it_924 = 1; (_loop_it_924 < (nsp + 1)); _loop_it_924 = (_loop_it_924 + 1)) {


                                            if (upf_tpawp) {

                                                for (_loop_it_925 = 1; (_loop_it_925 < (nat + 1)); _loop_it_925 = (_loop_it_925 + 1)) {

                                                    {

                                                        {
                                                            int _in_ityp_0 = ityp[(_loop_it_925 - 1)];
                                                            int64_t _out_if_cond_1023;

                                                            ///////////////////
                                                            // Tasklet code (t_1024)
                                                            _out_if_cond_1023 = (_in_ityp_0 == _loop_it_924);
                                                            ///////////////////

                                                            if_cond_1023 = _out_if_cond_1023;
                                                        }

                                                    }

                                                    if (if_cond_1023) {
                                                        {

                                                            {
                                                                int _in_ofsbeta_0 = ofsbeta[(_loop_it_925 - 1)];
                                                                int _out_paw_newdxx_ijkb0;

                                                                ///////////////////
                                                                // Tasklet code (t_1027)
                                                                _out_paw_newdxx_ijkb0 = _in_ofsbeta_0;
                                                                ///////////////////

                                                                paw_newdxx_ijkb0 = _out_paw_newdxx_ijkb0;
                                                            }

                                                        }
                                                        loopend_1028 = nh[(_loop_it_924 - 1)];


                                                        for (_loop_it_926 = 1; (_loop_it_926 < (loopend_1028 + 1)); _loop_it_926 = (_loop_it_926 + 1)) {

                                                            ukb = (paw_newdxx_ijkb0 + _loop_it_926);

                                                            loopend_1032 = nh[(_loop_it_924 - 1)];


                                                            for (_loop_it_927 = 1; (_loop_it_927 < (loopend_1032 + 1)); _loop_it_927 = (_loop_it_927 + 1)) {

                                                                okb = (paw_newdxx_ijkb0 + _loop_it_927);

                                                                loopend_1036 = nh[(_loop_it_924 - 1)];


                                                                for (_loop_it_928 = 1; (_loop_it_928 < (loopend_1036 + 1)); _loop_it_928 = (_loop_it_928 + 1)) {

                                                                    paw_newdxx_jkb = (paw_newdxx_ijkb0 + _loop_it_928);

                                                                    loopend_1040 = nh[(_loop_it_924 - 1)];


                                                                    for (_loop_it_929 = 1; (_loop_it_929 < (loopend_1040 + 1)); _loop_it_929 = (_loop_it_929 + 1)) {

                                                                        paw_newdxx_ikb = (paw_newdxx_ijkb0 + _loop_it_929);
                                                                        {

                                                                            {
                                                                                double _in_ke_k_0 = ke_k[(((((_loop_it_924 + ((((ke_k_d0 * ke_k_d1) * ke_k_d2) * ke_k_d3) * (_loop_it_926 - 1))) + (((ke_k_d0 * ke_k_d1) * ke_k_d2) * (_loop_it_927 - 1))) + ((ke_k_d0 * ke_k_d1) * (_loop_it_928 - 1))) + (ke_k_d0 * (_loop_it_929 - 1))) - 1)];
                                                                                dace::complex128 _in_paw_newdxx_becphi_0 = becxx_k[(((((becxx_k_d0 * becxx_k_d1) * (_loop_it_923 - 1)) + (becxx_k_d0 * (paw_newdxx_jkb - 1))) + ikq) - 1)];
                                                                                dace::complex128 _in_paw_newdxx_becphi_1 = becxx_k[(((((becxx_k_d0 * becxx_k_d1) * (_loop_it_923 - 1)) + (becxx_k_d0 * (ukb - 1))) + ikq) - 1)];
                                                                                dace::complex128 _in_paw_newdxx_becphi_2 = becxx_k[(((((becxx_k_d0 * becxx_k_d1) * (_loop_it_923 - 1)) + (becxx_k_d0 * (ukb - 1))) + ikq) - 1)];
                                                                                dace::complex128 _in_paw_newdxx_becpsi_0 = becpsi_k[(((becpsi_k_d0 * (ibnd - offset_becpsi_k_d1)) - offset_becpsi_k_d0) + okb)];
                                                                                dace::complex128 _in_paw_newdxx_deexx_0 = deexx[(((nkb * (_loop_it_871 - 1)) + paw_newdxx_ikb) - 1)];
                                                                                double _in___assoc_scalar_14 = __assoc_scalar_14;
                                                                                dace::complex128 _out_paw_newdxx_deexx;

                                                                                ///////////////////
                                                                                // Tasklet code (t_0)
                                                                                _out_paw_newdxx_deexx = (_in_paw_newdxx_deexx_0 + ((((((_in___assoc_scalar_14 * 0.5) * _in_ke_k_0) + (dace::complex128(0.0, 1.0) * 0.0)) * _in_paw_newdxx_becphi_0) * conj(_in_paw_newdxx_becphi_1)) * _in_paw_newdxx_becpsi_0));
                                                                                ///////////////////

                                                                                deexx[(((nkb * (_loop_it_871 - 1)) + paw_newdxx_ikb) - 1)] = _out_paw_newdxx_deexx;
                                                                            }

                                                                        }

                                                                    }

                                                                    paw_newdxx_ih = (loopend_1040 + 1);


                                                                    paw_newdxx_ih = paw_newdxx_ih;


                                                                }

                                                                paw_newdxx_jh = (loopend_1036 + 1);


                                                                paw_newdxx_jh = paw_newdxx_jh;


                                                            }

                                                            oh = (loopend_1032 + 1);


                                                            oh = oh;


                                                        }

                                                        uh = (loopend_1028 + 1);


                                                        uh = uh;

                                                    }


                                                }

                                                paw_newdxx_na = (nat + 1);


                                                paw_newdxx_na = paw_newdxx_na;

                                            }


                                        }

                                        paw_newdxx_np = (nsp + 1);


                                        paw_newdxx_np = paw_newdxx_np;

                                    }


                                }

                                jbnd = (jend + 1);


                                jbnd = jbnd;
                                {

                                    copy_vc_d_1057_sdfg_349_3_2(__state, &vc[0], &vc_d[0], jblock, nrxxs);

                                }
                            }


                            all_start_tmp = all_start[(wegrp - 1)];


                            for (_loop_it_930 = jstart; (_loop_it_930 < (jend + 1)); _loop_it_930 = (_loop_it_930 + 1)) {

                                for (_loop_it_931 = 1; (_loop_it_931 < (nrxxs + 1)); _loop_it_931 = (_loop_it_931 + 1)) {

                                    if_cond_1063 = noncolin[0];


                                    if (if_cond_1063) {
                                        {

                                            {
                                                dace::complex128 _in_exxbuff_d_0 = exxbuff_d[(((_loop_it_931 + ((exxbuff_d_d0 * exxbuff_d_d1) * (ikq - 1))) + (exxbuff_d_d0 * (((_loop_it_930 - all_start_tmp) + iexx_start) - 1))) - 1)];
                                                dace::complex128 _in_result_nc_d_0 = result_nc_d[((_loop_it_931 + ((npol * nrxxs) * (_loop_it_871 - 1))) - 1)];
                                                dace::complex128 _in_vc_d_0 = vc_d[((_loop_it_931 + (nrxxs * (_loop_it_930 - jstart))) - 1)];
                                                dace::complex128 _out_result_nc_d;

                                                ///////////////////
                                                // Tasklet code (t_1066)
                                                _out_result_nc_d = (_in_result_nc_d_0 + (_in_vc_d_0 * _in_exxbuff_d_0));
                                                ///////////////////

                                                result_nc_d[((_loop_it_931 + ((npol * nrxxs) * (_loop_it_871 - 1))) - 1)] = _out_result_nc_d;
                                            }

                                        }
                                        {

                                            {
                                                dace::complex128 _in_exxbuff_d_0 = exxbuff_d[((((_loop_it_931 + ((exxbuff_d_d0 * exxbuff_d_d1) * (ikq - 1))) + (exxbuff_d_d0 * (((_loop_it_930 - all_start_tmp) + iexx_start) - 1))) + nrxxs) - 1)];
                                                dace::complex128 _in_result_nc_d_0 = result_nc_d[(((_loop_it_931 + ((npol * nrxxs) * (_loop_it_871 - 1))) + nrxxs) - 1)];
                                                dace::complex128 _in_vc_d_0 = vc_d[((_loop_it_931 + (nrxxs * (_loop_it_930 - jstart))) - 1)];
                                                dace::complex128 _out_result_nc_d;

                                                ///////////////////
                                                // Tasklet code (t_1068)
                                                _out_result_nc_d = (_in_result_nc_d_0 + (_in_vc_d_0 * _in_exxbuff_d_0));
                                                ///////////////////

                                                result_nc_d[(((_loop_it_931 + ((npol * nrxxs) * (_loop_it_871 - 1))) + nrxxs) - 1)] = _out_result_nc_d;
                                            }

                                        }
                                    } else {
                                        {

                                            {
                                                dace::complex128 _in_exxbuff_d_0 = exxbuff_d[(((_loop_it_931 + ((exxbuff_d_d0 * exxbuff_d_d1) * (ikq - 1))) + (exxbuff_d_d0 * (((_loop_it_930 - all_start_tmp) + iexx_start) - 1))) - 1)];
                                                dace::complex128 _in_result_d_0 = result_d[((_loop_it_931 + (nrxxs * (_loop_it_871 - 1))) - 1)];
                                                dace::complex128 _in_vc_d_0 = vc_d[((_loop_it_931 + (nrxxs * (_loop_it_930 - jstart))) - 1)];
                                                dace::complex128 _out_result_d;

                                                ///////////////////
                                                // Tasklet code (t_1070)
                                                _out_result_d = (_in_result_d_0 + (_in_vc_d_0 * _in_exxbuff_d_0));
                                                ///////////////////

                                                result_d[((_loop_it_931 + (nrxxs * (_loop_it_871 - 1))) - 1)] = _out_result_d;
                                            }

                                        }
                                    }


                                }

                                ir = (nrxxs + 1);


                                ir = ir;


                            }

                            jbnd = (jend + 1);


                            jbnd = jbnd;

                        }

                    }


                }

                ii = (loopend_585 + 1);


                ii = ii;


            }

            ijt = (njt + 1);


            ijt = ijt;

            if_cond_1079 = (negrp > 1);


            if (if_cond_1079) {
                {

                    copy_exxbuff_d_1082_sdfg_373_0_2(__state, &exxbuff[0], &exxbuff_d[0], exxbuff_d0, exxbuff_d1, exxbuff_d2, exxbuff_d_d0, exxbuff_d_d1);

                }
            }


        }

        iegrp = (negrp + 1);


        iegrp = iegrp;

        if_cond_1085 = (okvan[0] && (tqr[0] != true));


        if (if_cond_1085) {

            qgm_allocated = 0;
            {

                {
                    #pragma omp parallel for
                    for (auto __i0 = 0; __i0 < dfftt_ngm; __i0 += 1) {
                        for (auto __i1 = 0; __i1 < qvan_init_nij; __i1 += 1) {
                            {
                                dace::complex128 _out;

                                ///////////////////
                                // Tasklet code (set_qgm)
                                _out = 0;
                                ///////////////////

                                qgm[(__i0 + (__i1 * dfftt_ngm))] = _out;
                            }
                        }
                    }
                }

            }
            nij_type_allocated = 0;
            {

                {
                    #pragma omp parallel for
                    for (auto __i0 = 0; __i0 < nsp; __i0 += 1) {
                        {
                            int _out;

                            ///////////////////
                            // Tasklet code (set_nij_type)
                            _out = 0;
                            ///////////////////

                            nij_type[__i0] = _out;
                        }
                    }
                }

            }
        }


    }

    delete[] temppsic_d;
    delete[] temppsic_nc_d;
    iq = (nqs + 1);


    iq = iq;

    loopend_1092 = nibands[my_egrp_id];


    for (_loop_it_932 = 1; (_loop_it_932 < (loopend_1092 + 1)); _loop_it_932 = (_loop_it_932 + 1)) {

        ibnd = ibands[((_loop_it_932 + (ibands_d0 * my_egrp_id)) - 1)];

        if_cond_1096 = (((ibnd == 0) || (ibnd > m)) != true);


        if (if_cond_1096) {

            if_cond_1099 = okvan[0];


            if (if_cond_1099) {

            }


            if_cond_1102 = noncolin[0];


            if (if_cond_1102) {

                {

                    dft_nd_383_1_2(__state, &result_nc_d[0], &result_nc_d[0], ialloc, npol, nrxxs);

                }
                {

                    dft_nd_383_1_2(__state, &result_nc_d[0], &result_nc_d[0], ialloc, npol, nrxxs);

                }

                for (_loop_it_933 = 1; (_loop_it_933 < (n + 1)); _loop_it_933 = (_loop_it_933 + 1)) {

                    igk_exx_d_at55 = igk_exx_d[((_loop_it_933 + (igk_exx_d_d0 * (current_k - 1))) - 1)];

                    dfftt__nl_at56 = dfftt__nl[(igk_exx_d_at55 - 1)];

                    {

                        {
                            dace::complex128 _in_big_result_d_0 = big_result_d[((_loop_it_933 + ((n * npol) * (ibnd - 1))) - 1)];
                            dace::complex128 _in_result_nc_d_0 = result_nc_d[((dfftt__nl_at56 + ((npol * nrxxs) * (_loop_it_932 - 1))) - 1)];
                            double _in_exxalfa = exxalfa[0];
                            dace::complex128 _out_big_result_d;

                            ///////////////////
                            // Tasklet code (t_0)
                            _out_big_result_d = (_in_big_result_d_0 - ((_in_exxalfa + (dace::complex128(0.0, 1.0) * 0.0)) * _in_result_nc_d_0));
                            ///////////////////

                            big_result_d[((_loop_it_933 + ((n * npol) * (ibnd - 1))) - 1)] = _out_big_result_d;
                        }

                    }
                    {

                        {
                            dace::complex128 _in_big_result_d_0 = big_result_d[(((_loop_it_933 + ((n * npol) * (ibnd - 1))) + n) - 1)];
                            dace::complex128 _in_result_nc_d_0 = result_nc_d[(((dfftt__nl_at56 + ((npol * nrxxs) * (_loop_it_932 - 1))) + nrxxs) - 1)];
                            double _in_exxalfa = exxalfa[0];
                            dace::complex128 _out_big_result_d;

                            ///////////////////
                            // Tasklet code (t_1)
                            _out_big_result_d = (_in_big_result_d_0 - ((_in_exxalfa + (dace::complex128(0.0, 1.0) * 0.0)) * _in_result_nc_d_0));
                            ///////////////////

                            big_result_d[(((_loop_it_933 + ((n * npol) * (ibnd - 1))) + n) - 1)] = _out_big_result_d;
                        }

                    }

                }

                ig = (n + 1);


                ig = ig;

            } else {

                {

                    dft_nd_387_1_2(__state, &result_d[0], &result_d[0], ialloc, nrxxs);

                }

                for (_loop_it_934 = 1; (_loop_it_934 < (n + 1)); _loop_it_934 = (_loop_it_934 + 1)) {

                    igk_exx_d_at57 = igk_exx_d[((_loop_it_934 + (igk_exx_d_d0 * (current_k - 1))) - 1)];

                    dfftt__nl_at58 = dfftt__nl[(igk_exx_d_at57 - 1)];

                    {

                        {
                            dace::complex128 _in_big_result_d_0 = big_result_d[((_loop_it_934 + ((n * npol) * (ibnd - 1))) - 1)];
                            dace::complex128 _in_result_d_0 = result_d[((dfftt__nl_at58 + (nrxxs * (_loop_it_932 - 1))) - 1)];
                            double _in_exxalfa = exxalfa[0];
                            dace::complex128 _out_big_result_d;

                            ///////////////////
                            // Tasklet code (t_0)
                            _out_big_result_d = (_in_big_result_d_0 - ((_in_exxalfa + (dace::complex128(0.0, 1.0) * 0.0)) * _in_result_d_0));
                            ///////////////////

                            big_result_d[((_loop_it_934 + ((n * npol) * (ibnd - 1))) - 1)] = _out_big_result_d;
                        }

                    }

                }

                ig = (n + 1);


                ig = ig;

            }


            loopend_1125 = ((1 + (n * npol)) - 1);


            for (_loop_it_935 = 1; (_loop_it_935 < (loopend_1125 + 1)); _loop_it_935 = (_loop_it_935 + 1)) {
                {

                    {
                        dace::complex128 _in_big_result_d_0 = big_result_d[((_loop_it_935 + ((n * npol) * (ibnd - 1))) - 1)];
                        dace::complex128 _out_big_result;

                        ///////////////////
                        // Tasklet code (t_0)
                        _out_big_result = _in_big_result_d_0;
                        ///////////////////

                        big_result[((_loop_it_935 + ((n * npol) * (ibnd - 1))) - 1)] = _out_big_result;
                    }

                }

            }

            ss_0 = (loopend_1125 + 1);


            if_cond_1129 = okvan[0];


            if (if_cond_1129) {
                {

                    {
                        double _out;

                        ///////////////////
                        // Tasklet code (set___assoc_scalar_16)
                        _out = 1e-08;
                        ///////////////////

                        __assoc_scalar_16 = _out;
                    }

                }
                __al_30 = 0;

                if_cond_1133 = okvan[0];


                if (if_cond_1133) {

                    vkbp_d0 = npwx;

                    vkbp_d1 = nkb;

                    vkbp_allocated = 1;
                    {

                        {
                            bool _out;

                            ///////////////////
                            // Tasklet code (set_run_on_gpu)
                            _out = 0;
                            ///////////////////

                            run_on_gpu = _out;
                        }

                    }

                    if (0) {
                        {

                            {
                                bool _in_run_on_gpu_ = run_on_gpu_;
                                bool _out;

                                ///////////////////
                                // Tasklet code (set_run_on_gpu)
                                _out = _in_run_on_gpu_;
                                ///////////////////

                                run_on_gpu = _out;
                            }

                        }
                    }


                    __al_31 = 0;

                    __al_32 = 0;

                    __al_33 = 0;

                    __al_34 = 0;

                    __al_35 = 0;

                    __al_36 = 0;

                    if_cond_1148 = (lmaxkb < 0);


                    if (if_cond_1148) {

                    } else {

                        vkb1_d0 = n;

                        vkb1_d1 = nhm;

                        vkb1_allocated = 1;

                        sk_d0 = n;

                        sk_allocated = 1;

                        init_us_2_acc_qg_d0 = n;

                        init_us_2_acc_qg_allocated = 1;

                        vq_d0 = n;

                        vq_d1 = nbetam;

                        vq_allocated = 1;

                        ylm_d0 = n;

                        ylm_allocated = 1;

                        gk_d0 = 3;

                        gk_d1 = n;

                        gk_allocated = 1;
                        {

                            {
                                double _in_xkp_0 = xkp[0];
                                double _out_q1;

                                ///////////////////
                                // Tasklet code (t_1166)
                                _out_q1 = _in_xkp_0;
                                ///////////////////

                                q1 = _out_q1;
                            }
                            {
                                double _in_xkp_0 = xkp[1];
                                double _out_q2;

                                ///////////////////
                                // Tasklet code (t_1167)
                                _out_q2 = _in_xkp_0;
                                ///////////////////

                                q2 = _out_q2;
                            }
                            {
                                double _in_xkp_0 = xkp[2];
                                double _out_q3;

                                ///////////////////
                                // Tasklet code (t_1168)
                                _out_q3 = _in_xkp_0;
                                ///////////////////

                                q3 = _out_q3;
                            }

                        }

                        for (_loop_it_936 = 1; (_loop_it_936 < (npwx + 1)); _loop_it_936 = (_loop_it_936 + 1)) {

                            for (_loop_it_937 = 1; (_loop_it_937 < (nkb + 1)); _loop_it_937 = (_loop_it_937 + 1)) {
                                {

                                    {
                                        dace::complex128 _out_vkbp;

                                        ///////////////////
                                        // Tasklet code (t_0)
                                        _out_vkbp = (0.0 + (dace::complex128(0.0, 1.0) * 0.0));
                                        ///////////////////

                                        vkbp[((_loop_it_936 + (npwx * (_loop_it_937 - 1))) - 1)] = _out_vkbp;
                                    }

                                }

                            }

                            as_1 = (nkb + 1);


                        }

                        as_0 = (npwx + 1);


                        for (_loop_it_938 = 1; (_loop_it_938 < (n + 1)); _loop_it_938 = (_loop_it_938 + 1)) {

                            iv_d = igk_exx[((_loop_it_938 + (igk_exx_d0 * (current_k - 1))) - 1)];
                            {

                                {
                                    double _in_g_0 = g[(g_d0 * (iv_d - 1))];
                                    double _in_q1 = q1;
                                    double _out_gk;

                                    ///////////////////
                                    // Tasklet code (t_0)
                                    _out_gk = (_in_q1 + _in_g_0);
                                    ///////////////////

                                    gk[((3 * _loop_it_938) - 3)] = _out_gk;
                                }

                            }
                            {

                                {
                                    double _in_g_0 = g[((g_d0 * (iv_d - 1)) + 1)];
                                    double _in_q2 = q2;
                                    double _out_gk;

                                    ///////////////////
                                    // Tasklet code (t_1)
                                    _out_gk = (_in_q2 + _in_g_0);
                                    ///////////////////

                                    gk[((3 * _loop_it_938) - 2)] = _out_gk;
                                }

                            }
                            {

                                {
                                    double _in_g_0 = g[((g_d0 * (iv_d - 1)) + 2)];
                                    double _in_q3 = q3;
                                    double _out_gk;

                                    ///////////////////
                                    // Tasklet code (t_2)
                                    _out_gk = (_in_q3 + _in_g_0);
                                    ///////////////////

                                    gk[((3 * _loop_it_938) - 1)] = _out_gk;
                                }

                            }
                            {

                                {
                                    double _in_gk_0 = gk[((3 * _loop_it_938) - 3)];
                                    double _in_gk_1 = gk[((3 * _loop_it_938) - 3)];
                                    double _in_gk_2 = gk[((3 * _loop_it_938) - 2)];
                                    double _in_gk_3 = gk[((3 * _loop_it_938) - 2)];
                                    double _in_gk_4 = gk[((3 * _loop_it_938) - 1)];
                                    double _in_gk_5 = gk[((3 * _loop_it_938) - 1)];
                                    double _out_init_us_2_acc_qg;

                                    ///////////////////
                                    // Tasklet code (t_3)
                                    _out_init_us_2_acc_qg = (((_in_gk_0 * _in_gk_1) + (_in_gk_2 * _in_gk_3)) + (_in_gk_4 * _in_gk_5));
                                    ///////////////////

                                    init_us_2_acc_qg[(_loop_it_938 - 1)] = _out_init_us_2_acc_qg;
                                }

                            }

                        }

                        init_us_2_acc_ig = (n + 1);


                        init_us_2_acc_ig = init_us_2_acc_ig;

                        __assoc_scalar_17 = (dace::math::ipow((lmaxkb + 1), 2));
                        {

                            {
                                bool _out;

                                ///////////////////
                                // Tasklet code (set_goto_10)
                                _out = 0;
                                ///////////////////

                                goto_10 = _out;
                            }

                        }
                        if_cond_1179 = ((n < 1) || (__assoc_scalar_17 < 1));


                        if (if_cond_1179) {

                        } else {

                            __al_37 = 21;

                            lmax = 0;


                            for (; true; ) {
                                {

                                    {
                                        bool _in_goto_10 = goto_10;
                                        int _out;

                                        ///////////////////
                                        // Tasklet code (set___brkc_4)
                                        _out = ((__al_37 > 0) ? (! _in_goto_10) : 0);
                                        ///////////////////

                                        __brkc_4 = _out;
                                    }

                                }
                                if_cond_1186 = (__al_37 > 0);


                                if (if_cond_1186) {

                                    if_cond_1189 = ((dace::math::ipow((lmax + 1), 2)) == __assoc_scalar_17);


                                    if (if_cond_1189) {
                                        {

                                            {
                                                bool _out;

                                                ///////////////////
                                                // Tasklet code (set_goto_10)
                                                _out = -1;
                                                ///////////////////

                                                goto_10 = _out;
                                            }

                                        }
                                    }



                                    if (goto_10) {

                                    } else {

                                        __al_37 = (__al_37 - 1);

                                        lmax = (lmax + 1);

                                    }

                                    {

                                        {
                                            bool _in_goto_10 = goto_10;
                                            int _out;

                                            ///////////////////
                                            // Tasklet code (set___sc_1)
                                            _out = (_in_goto_10 != true);
                                            ///////////////////

                                            __sc_1 = _out;
                                        }

                                    }
                                } else {
                                    {

                                        {
                                            int _out;

                                            ///////////////////
                                            // Tasklet code (set___sc_1)
                                            _out = false;
                                            ///////////////////

                                            __sc_1 = _out;
                                        }

                                    }
                                }


                                if_cond_1200 = (! __brkc_4);


                                if (if_cond_1200) {
                                    break;
                                }


                            }


                            if_cond_1204 = (goto_10 != true);


                            if (if_cond_1204) {

                            }

                            {

                                {
                                    bool _out;

                                    ///////////////////
                                    // Tasklet code (set_goto_10)
                                    _out = 0;
                                    ///////////////////

                                    goto_10 = _out;
                                }

                            }
                            if_cond_1207 = (lmax == 0);


                            if (if_cond_1207) {

                                for (_loop_it_939 = 1; (_loop_it_939 < (n + 1)); _loop_it_939 = (_loop_it_939 + 1)) {
                                    {

                                        {
                                            double _out_ylm;

                                            ///////////////////
                                            // Tasklet code (t_0)
                                            _out_ylm = 0.28209479177387814;
                                            ///////////////////

                                            ylm[(_loop_it_939 - 1)] = _out_ylm;
                                        }

                                    }

                                }

                                as_0 = (n + 1);

                            } else {

                                for (_loop_it_940 = 1; (_loop_it_940 < (n + 1)); _loop_it_940 = (_loop_it_940 + 1)) {
                                    {

                                        {
                                            double _in_init_us_2_acc_qg_0 = init_us_2_acc_qg[(_loop_it_940 - 1)];
                                            double _out_gmod;

                                            ///////////////////
                                            // Tasklet code (t_1212)
                                            _out_gmod = sqrt(_in_init_us_2_acc_qg_0);
                                            ///////////////////

                                            gmod = _out_gmod;
                                        }

                                    }
                                    if_cond_1213 = (gmod < 1e-09);


                                    if (if_cond_1213) {
                                        {

                                            {
                                                double _out;

                                                ///////////////////
                                                // Tasklet code (set_cost)
                                                _out = 0.0;
                                                ///////////////////

                                                cost = _out;
                                            }

                                        }
                                    } else {
                                        {

                                            {
                                                double _in_gk_0 = gk[((3 * _loop_it_940) - 1)];
                                                double _in_gmod = gmod;
                                                double _out_cost;

                                                ///////////////////
                                                // Tasklet code (t_1217)
                                                _out_cost = (_in_gk_0 / _in_gmod);
                                                ///////////////////

                                                cost = _out_cost;
                                            }

                                        }
                                    }

                                    {

                                        {
                                            double _in_cost = cost;
                                            double _out;

                                            ///////////////////
                                            // Tasklet code (set_sent)
                                            _out = sqrt(max(0.0, (1.0 - (_in_cost * _in_cost))));
                                            ///////////////////

                                            sent = _out;
                                        }
                                        {
                                            double _out_ylm;

                                            ///////////////////
                                            // Tasklet code (t_1219)
                                            _out_ylm = 1.0;
                                            ///////////////////

                                            ylm[(_loop_it_940 - 1)] = _out_ylm;
                                        }

                                    }
                                    {

                                        {
                                            double _in_cost = cost;
                                            double _out_ylm;

                                            ///////////////////
                                            // Tasklet code (t_1221)
                                            _out_ylm = _in_cost;
                                            ///////////////////

                                            ylm[((_loop_it_940 + n) - 1)] = _out_ylm;
                                        }

                                    }
                                    {

                                        {
                                            double _in_sent = sent;
                                            double _out_ylm;

                                            ///////////////////
                                            // Tasklet code (t_1223)
                                            _out_ylm = (- (_in_sent / 1.4142135623730951));
                                            ///////////////////

                                            ylm[((_loop_it_940 + (3 * n)) - 1)] = _out_ylm;
                                        }

                                    }

                                    for (_loop_it_941 = 2; (_loop_it_941 < (lmax + 1)); _loop_it_941 = (_loop_it_941 + 1)) {

                                        loopend_1225 = (_loop_it_941 - 2);


                                        for (_loop_it_942 = 0; (_loop_it_942 < (loopend_1225 + 1)); _loop_it_942 = (_loop_it_942 + 1)) {

                                            ylmr2_lm = (((dace::math::ipow(_loop_it_941, 2)) + 1) + (_loop_it_942 * 2));
                                            lm1 = (((dace::math::ipow((_loop_it_941 - 1), 2)) + 1) + (_loop_it_942 * 2));
                                            lm2 = (((dace::math::ipow((_loop_it_941 - 2), 2)) + 1) + (_loop_it_942 * 2));
                                            {

                                                {
                                                    double _in_ylm_0 = ylm[((_loop_it_940 + (n * (lm1 - 1))) - 1)];
                                                    double _in_ylm_1 = ylm[((_loop_it_940 + (n * (lm2 - 1))) - 1)];
                                                    double _in_cost = cost;
                                                    double _out_ylm;

                                                    ///////////////////
                                                    // Tasklet code (t_0)
                                                    _out_ylm = ((((_in_cost * dace::float64(((_loop_it_941 * 2) - 1))) / sqrt(dace::float64(((_loop_it_941 * _loop_it_941) - (_loop_it_942 * _loop_it_942))))) * _in_ylm_0) - ((sqrt(dace::float64((((_loop_it_941 - 1) * (_loop_it_941 - 1)) - (_loop_it_942 * _loop_it_942)))) / sqrt(dace::float64(((_loop_it_941 * _loop_it_941) - (_loop_it_942 * _loop_it_942))))) * _in_ylm_1));
                                                    ///////////////////

                                                    ylm[((_loop_it_940 + (n * (ylmr2_lm - 1))) - 1)] = _out_ylm;
                                                }

                                            }

                                        }

                                        ylmr2_m = (loopend_1225 + 1);


                                        ylmr2_m = ylmr2_m;

                                        ylmr2_lm = (((dace::math::ipow(_loop_it_941, 2)) + 1) + (_loop_it_941 * 2));

                                        lm1 = (((dace::math::ipow(_loop_it_941, 2)) + 1) + ((_loop_it_941 - 1) * 2));

                                        lm2 = (((dace::math::ipow((_loop_it_941 - 1), 2)) + 1) + ((_loop_it_941 - 1) * 2));
                                        {

                                            {
                                                double _in_ylm_0 = ylm[((_loop_it_940 + (n * (lm2 - 1))) - 1)];
                                                double _in_cost = cost;
                                                double _out_ylm;

                                                ///////////////////
                                                // Tasklet code (t_1234)
                                                _out_ylm = ((_in_cost * sqrt(dace::float64(((_loop_it_941 * 2) - 1)))) * _in_ylm_0);
                                                ///////////////////

                                                ylm[((_loop_it_940 + (n * (lm1 - 1))) - 1)] = _out_ylm;
                                            }

                                        }
                                        {

                                            {
                                                double _in_ylm_0 = ylm[((_loop_it_940 + (n * (lm2 - 1))) - 1)];
                                                double _in_sent = sent;
                                                double _out_ylm;

                                                ///////////////////
                                                // Tasklet code (t_1236)
                                                _out_ylm = (- (((sqrt(dace::float64(((_loop_it_941 * 2) - 1))) / sqrt(dace::float64((_loop_it_941 * 2)))) * _in_sent) * _in_ylm_0));
                                                ///////////////////

                                                ylm[((_loop_it_940 + (n * (ylmr2_lm - 1))) - 1)] = _out_ylm;
                                            }

                                        }

                                    }

                                    ylmr2_l = (lmax + 1);


                                    ylmr2_l = ylmr2_l;

                                    {

                                        {
                                            double _in_gk_0 = gk[((3 * _loop_it_940) - 3)];
                                            int64_t _out_if_cond_1239;

                                            ///////////////////
                                            // Tasklet code (t_1240)
                                            _out_if_cond_1239 = (_in_gk_0 > 1e-09);
                                            ///////////////////

                                            if_cond_1239 = _out_if_cond_1239;
                                        }

                                    }

                                    if (if_cond_1239) {
                                        {

                                            {
                                                double _in_gk_0 = gk[((3 * _loop_it_940) - 2)];
                                                double _in_gk_1 = gk[((3 * _loop_it_940) - 3)];
                                                double _out_phi;

                                                ///////////////////
                                                // Tasklet code (t_1243)
                                                _out_phi = atan((_in_gk_0 / _in_gk_1));
                                                ///////////////////

                                                phi = _out_phi;
                                            }

                                        }
                                    } else {

                                        {

                                            {
                                                double _in_gk_0 = gk[((3 * _loop_it_940) - 3)];
                                                int64_t _out_if_cond_1245;

                                                ///////////////////
                                                // Tasklet code (t_1246)
                                                _out_if_cond_1245 = (_in_gk_0 < -1e-09);
                                                ///////////////////

                                                if_cond_1245 = _out_if_cond_1245;
                                            }

                                        }

                                        if (if_cond_1245) {
                                            {

                                                {
                                                    double _in_gk_0 = gk[((3 * _loop_it_940) - 2)];
                                                    double _in_gk_1 = gk[((3 * _loop_it_940) - 3)];
                                                    double _out_phi;

                                                    ///////////////////
                                                    // Tasklet code (t_1249)
                                                    _out_phi = (atan((_in_gk_0 / _in_gk_1)) + 3.141592653589793);
                                                    ///////////////////

                                                    phi = _out_phi;
                                                }

                                            }
                                        } else {
                                            {

                                                {
                                                    double _in_gk_0 = gk[((3 * _loop_it_940) - 2)];
                                                    double _out_phi;

                                                    ///////////////////
                                                    // Tasklet code (t_1251)
                                                    _out_phi = copysign(1.5707963267948966, _in_gk_0);
                                                    ///////////////////

                                                    phi = _out_phi;
                                                }

                                            }
                                        }

                                    }


                                    ylmr2_lm = 1;
                                    {

                                        {
                                            double _in_ylm_0 = ylm[(_loop_it_940 - 1)];
                                            double _out_ylm;

                                            ///////////////////
                                            // Tasklet code (t_1254)
                                            _out_ylm = (_in_ylm_0 / 3.5449077018110318);
                                            ///////////////////

                                            ylm[(_loop_it_940 - 1)] = _out_ylm;
                                        }

                                    }

                                    for (_loop_it_943 = 1; (_loop_it_943 < (lmax + 1)); _loop_it_943 = (_loop_it_943 + 1)) {
                                        {

                                            {
                                                double _out;

                                                ///////////////////
                                                // Tasklet code (set_c)
                                                _out = sqrt((dace::float64(((_loop_it_943 * 2) + 1)) / 12.566370614359172));
                                                ///////////////////

                                                c = _out;
                                            }

                                        }
                                        ylmr2_lm = (ylmr2_lm + 1);
                                        {

                                            {
                                                double _in_ylm_0 = ylm[((_loop_it_940 + (n * (ylmr2_lm - 1))) - 1)];
                                                double _in_c = c;
                                                double _out_ylm;

                                                ///////////////////
                                                // Tasklet code (t_1258)
                                                _out_ylm = (_in_c * _in_ylm_0);
                                                ///////////////////

                                                ylm[((_loop_it_940 + (n * (ylmr2_lm - 1))) - 1)] = _out_ylm;
                                            }

                                        }

                                        for (_loop_it_944 = 1; (_loop_it_944 < (_loop_it_943 + 1)); _loop_it_944 = (_loop_it_944 + 1)) {

                                            ylmr2_lm = (ylmr2_lm + 2);
                                            {

                                                {
                                                    double _in_ylm_0 = ylm[((_loop_it_940 + (n * (ylmr2_lm - 1))) - 1)];
                                                    double _in_c = c;
                                                    double _in_phi = phi;
                                                    double _out_ylm;

                                                    ///////////////////
                                                    // Tasklet code (t_0)
                                                    _out_ylm = (((_in_c * 1.4142135623730951) * _in_ylm_0) * cos((dace::float64(_loop_it_944) * _in_phi)));
                                                    ///////////////////

                                                    ylm[((_loop_it_940 + (n * (ylmr2_lm - 2))) - 1)] = _out_ylm;
                                                }

                                            }
                                            {

                                                {
                                                    double _in_ylm_0 = ylm[((_loop_it_940 + (n * (ylmr2_lm - 1))) - 1)];
                                                    double _in_c = c;
                                                    double _in_phi = phi;
                                                    double _out_ylm;

                                                    ///////////////////
                                                    // Tasklet code (t_1)
                                                    _out_ylm = (((_in_c * 1.4142135623730951) * _in_ylm_0) * sin((dace::float64(_loop_it_944) * _in_phi)));
                                                    ///////////////////

                                                    ylm[((_loop_it_940 + (n * (ylmr2_lm - 1))) - 1)] = _out_ylm;
                                                }

                                            }

                                        }

                                        ylmr2_m = (_loop_it_943 + 1);


                                        ylmr2_m = ylmr2_m;


                                    }

                                    ylmr2_l = (lmax + 1);


                                    ylmr2_l = ylmr2_l;


                                }

                                ylmr2_ig = (n + 1);


                                ylmr2_ig = ylmr2_ig;

                            }

                        }


                        for (_loop_it_945 = 1; (_loop_it_945 < (n + 1)); _loop_it_945 = (_loop_it_945 + 1)) {
                            {

                                {
                                    double _in_init_us_2_acc_qg_0 = init_us_2_acc_qg[(_loop_it_945 - 1)];
                                    double _in_tpiba = tpiba[0];
                                    double _out_init_us_2_acc_qg;

                                    ///////////////////
                                    // Tasklet code (t_0)
                                    _out_init_us_2_acc_qg = (sqrt(_in_init_us_2_acc_qg_0) * _in_tpiba);
                                    ///////////////////

                                    init_us_2_acc_qg[(_loop_it_945 - 1)] = _out_init_us_2_acc_qg;
                                }

                            }

                        }

                        init_us_2_acc_ig = (n + 1);


                        init_us_2_acc_ig = init_us_2_acc_ig;

                        init_us_2_acc_jkb = 0;


                        for (_loop_it_946 = 1; (_loop_it_946 < (nsp + 1)); _loop_it_946 = (_loop_it_946 + 1)) {

                            nbnt = upf_nbeta[(_loop_it_946 - 1)];


                            for (_loop_it_947 = 1; (_loop_it_947 < (nbnt + 1)); _loop_it_947 = (_loop_it_947 + 1)) {

                                for (_loop_it_948 = 1; (_loop_it_948 < (n + 1)); _loop_it_948 = (_loop_it_948 + 1)) {
                                    {

                                        {
                                            double _in_init_us_2_acc_qg_0 = init_us_2_acc_qg[(_loop_it_948 - 1)];
                                            double _out_qgr;

                                            ///////////////////
                                            // Tasklet code (t_1278)
                                            _out_qgr = _in_init_us_2_acc_qg_0;
                                            ///////////////////

                                            qgr = _out_qgr;
                                        }
                                        {
                                            double _in_qgr = qgr;
                                            double _out;

                                            ///////////////////
                                            // Tasklet code (set_interp_beta_px)
                                            _out = ((_in_qgr / 0.01) - dace::float64(dace::int32((_in_qgr / 0.01))));
                                            ///////////////////

                                            interp_beta_px = _out;
                                        }
                                        {
                                            double _in_interp_beta_px = interp_beta_px;
                                            double _out;

                                            ///////////////////
                                            // Tasklet code (set_interp_beta_ux)
                                            _out = (1.0 - _in_interp_beta_px);
                                            ///////////////////

                                            interp_beta_ux = _out;
                                        }
                                        {
                                            double _in_interp_beta_px = interp_beta_px;
                                            double _out;

                                            ///////////////////
                                            // Tasklet code (set_interp_beta_vx)
                                            _out = (2.0 - _in_interp_beta_px);
                                            ///////////////////

                                            interp_beta_vx = _out;
                                        }
                                        {
                                            double _in_interp_beta_px = interp_beta_px;
                                            double _out;

                                            ///////////////////
                                            // Tasklet code (set_interp_beta_wx)
                                            _out = (3.0 - _in_interp_beta_px);
                                            ///////////////////

                                            interp_beta_wx = _out;
                                        }

                                    }
                                    interp_beta_i0 = (dace::int32((qgr / 0.01)) + 1);

                                    interp_beta_i1 = (interp_beta_i0 + 1);

                                    interp_beta_i2 = (interp_beta_i0 + 2);

                                    interp_beta_i3 = (interp_beta_i0 + 3);

                                    if_cond_1283 = (interp_beta_i3 <= nqx);


                                    if (if_cond_1283) {
                                        {

                                            {
                                                double _in_tab_beta_0 = tab_beta[(((interp_beta_i0 + ((tab_beta_d0 * tab_beta_d1) * (_loop_it_946 - 1))) + (tab_beta_d0 * (_loop_it_947 - 1))) - 1)];
                                                double _in_tab_beta_1 = tab_beta[(((interp_beta_i1 + ((tab_beta_d0 * tab_beta_d1) * (_loop_it_946 - 1))) + (tab_beta_d0 * (_loop_it_947 - 1))) - 1)];
                                                double _in_tab_beta_2 = tab_beta[(((interp_beta_i2 + ((tab_beta_d0 * tab_beta_d1) * (_loop_it_946 - 1))) + (tab_beta_d0 * (_loop_it_947 - 1))) - 1)];
                                                double _in_tab_beta_3 = tab_beta[(((interp_beta_i3 + ((tab_beta_d0 * tab_beta_d1) * (_loop_it_946 - 1))) + (tab_beta_d0 * (_loop_it_947 - 1))) - 1)];
                                                double _in_interp_beta_px = interp_beta_px;
                                                double _in_interp_beta_ux = interp_beta_ux;
                                                double _in_interp_beta_vx = interp_beta_vx;
                                                double _in_interp_beta_wx = interp_beta_wx;
                                                double _out_vq;

                                                ///////////////////
                                                // Tasklet code (t_1286)
                                                _out_vq = (((((((_in_tab_beta_0 * _in_interp_beta_ux) * _in_interp_beta_vx) * _in_interp_beta_wx) / 6.0) + ((((_in_tab_beta_1 * _in_interp_beta_px) * _in_interp_beta_vx) * _in_interp_beta_wx) / 2.0)) - ((((_in_tab_beta_2 * _in_interp_beta_px) * _in_interp_beta_ux) * _in_interp_beta_wx) / 2.0)) + ((((_in_tab_beta_3 * _in_interp_beta_px) * _in_interp_beta_ux) * _in_interp_beta_vx) / 6.0));
                                                ///////////////////

                                                vq[((_loop_it_948 + (n * (_loop_it_947 - 1))) - 1)] = _out_vq;
                                            }

                                        }
                                    } else {
                                        {

                                            {
                                                double _out_vq;

                                                ///////////////////
                                                // Tasklet code (t_1288)
                                                _out_vq = 0.0;
                                                ///////////////////

                                                vq[((_loop_it_948 + (n * (_loop_it_947 - 1))) - 1)] = _out_vq;
                                            }

                                        }
                                    }


                                }

                                interp_beta_ig = (n + 1);


                                interp_beta_ig = interp_beta_ig;


                            }

                            interp_beta_nb = (nbnt + 1);


                            interp_beta_nb = interp_beta_nb;

                            nhnt = nh[(_loop_it_946 - 1)];


                            for (_loop_it_949 = 1; (_loop_it_949 < (nhnt + 1)); _loop_it_949 = (_loop_it_949 + 1)) {

                                for (_loop_it_950 = 1; (_loop_it_950 < (n + 1)); _loop_it_950 = (_loop_it_950 + 1)) {

                                    init_us_2_acc_nb = indv[((_loop_it_949 + (indv_d0 * (_loop_it_946 - 1))) - 1)];
                                    init_us_2_acc_lm = nhtolm[((_loop_it_949 + (nhtolm_d0 * (_loop_it_946 - 1))) - 1)];
                                    {

                                        {
                                            double _in_vq_0 = vq[((_loop_it_950 + (n * (init_us_2_acc_nb - 1))) - 1)];
                                            double _in_ylm_0 = ylm[((_loop_it_950 + (n * (init_us_2_acc_lm - 1))) - 1)];
                                            double _out_vkb1;

                                            ///////////////////
                                            // Tasklet code (t_0)
                                            _out_vkb1 = (_in_ylm_0 * _in_vq_0);
                                            ///////////////////

                                            vkb1[((_loop_it_950 + (n * (_loop_it_949 - 1))) - 1)] = _out_vkb1;
                                        }

                                    }

                                }

                                init_us_2_acc_ig = (n + 1);


                                init_us_2_acc_ig = init_us_2_acc_ig;


                            }

                            init_us_2_acc_ih = (nhnt + 1);


                            init_us_2_acc_ih = init_us_2_acc_ih;


                            for (_loop_it_951 = 1; (_loop_it_951 < (nat + 1)); _loop_it_951 = (_loop_it_951 + 1)) {

                                {

                                    {
                                        int _in_ityp_0 = ityp[(_loop_it_951 - 1)];
                                        int64_t _out_if_cond_1303;

                                        ///////////////////
                                        // Tasklet code (t_1304)
                                        _out_if_cond_1303 = (_in_ityp_0 == _loop_it_946);
                                        ///////////////////

                                        if_cond_1303 = _out_if_cond_1303;
                                    }

                                }

                                if (if_cond_1303) {
                                    {

                                        {
                                            double _in_tau_0 = tau[(tau_d0 * (_loop_it_951 - 1))];
                                            double _in_tau_1 = tau[((tau_d0 * (_loop_it_951 - 1)) + 1)];
                                            double _in_tau_2 = tau[((tau_d0 * (_loop_it_951 - 1)) + 2)];
                                            double _in_q1 = q1;
                                            double _in_q2 = q2;
                                            double _in_q3 = q3;
                                            double _out_init_us_2_acc_arg;

                                            ///////////////////
                                            // Tasklet code (t_1307)
                                            _out_init_us_2_acc_arg = ((((_in_q1 * _in_tau_0) + (_in_q2 * _in_tau_1)) + (_in_q3 * _in_tau_2)) * 6.283185307179586);
                                            ///////////////////

                                            init_us_2_acc_arg = _out_init_us_2_acc_arg;
                                        }

                                    }

                                    for (_loop_it_952 = 1; (_loop_it_952 < (n + 1)); _loop_it_952 = (_loop_it_952 + 1)) {

                                        mill_at59 = mill[(mill_d0 * (iv_d - 1))];

                                        mill_at60 = mill[((mill_d0 * (iv_d - 1)) + 1)];

                                        mill_at61 = mill[((mill_d0 * (iv_d - 1)) + 2)];

                                        iv_d = igk_exx[((_loop_it_952 + (igk_exx_d0 * (current_k - 1))) - 1)];
                                        {

                                            {
                                                dace::complex128 _in_eigts1_0 = eigts1[(((eigts1_d0 * (_loop_it_951 - 1)) + mill_at59) - 1)];
                                                dace::complex128 _in_eigts2_0 = eigts2[(((eigts2_d0 * (_loop_it_951 - 1)) + mill_at60) - 1)];
                                                dace::complex128 _in_eigts3_0 = eigts3[(((eigts3_d0 * (_loop_it_951 - 1)) + mill_at61) - 1)];
                                                double _in_init_us_2_acc_arg = init_us_2_acc_arg;
                                                dace::complex128 _out_sk;

                                                ///////////////////
                                                // Tasklet code (t_0)
                                                _out_sk = (((_in_eigts1_0 * _in_eigts2_0) * _in_eigts3_0) * (cos(_in_init_us_2_acc_arg) + (dace::complex128(0.0, 1.0) * (- sin(_in_init_us_2_acc_arg)))));
                                                ///////////////////

                                                sk[(_loop_it_952 - 1)] = _out_sk;
                                            }

                                        }

                                    }

                                    init_us_2_acc_ig = (n + 1);


                                    init_us_2_acc_ig = init_us_2_acc_ig;


                                    for (_loop_it_953 = 1; (_loop_it_953 < (nhnt + 1)); _loop_it_953 = (_loop_it_953 + 1)) {

                                        for (_loop_it_954 = 1; (_loop_it_954 < (n + 1)); _loop_it_954 = (_loop_it_954 + 1)) {
                                            {
                                                dace::complex128 pref;

                                                {
                                                    int _in_nhtol_0 = nhtol[((_loop_it_953 + (nhtol_d0 * (_loop_it_946 - 1))) - 1)];
                                                    dace::complex128 _out_pref;

                                                    ///////////////////
                                                    // Tasklet code (t_1318)
                                                    _out_pref = dace::math::pow((0.0 + (dace::complex128(0.0, 1.0) * -1.0)), _in_nhtol_0);
                                                    ///////////////////

                                                    pref = _out_pref;
                                                }
                                                {
                                                    dace::complex128 _in_sk_0 = sk[(_loop_it_954 - 1)];
                                                    double _in_vkb1_0 = vkb1[((_loop_it_954 + (n * (_loop_it_953 - 1))) - 1)];
                                                    dace::complex128 _in_pref = pref;
                                                    dace::complex128 _out_vkbp;

                                                    ///////////////////
                                                    // Tasklet code (t_1319)
                                                    _out_vkbp = (((_in_vkb1_0 + (dace::complex128(0.0, 1.0) * 0.0)) * _in_sk_0) * _in_pref);
                                                    ///////////////////

                                                    vkbp[((_loop_it_954 + (npwx * ((_loop_it_953 + init_us_2_acc_jkb) - 1))) - 1)] = _out_vkbp;
                                                }

                                            }

                                        }

                                        init_us_2_acc_ig = (n + 1);


                                        init_us_2_acc_ig = init_us_2_acc_ig;


                                    }

                                    init_us_2_acc_ih = (nhnt + 1);


                                    init_us_2_acc_ih = init_us_2_acc_ih;

                                    init_us_2_acc_jkb = (init_us_2_acc_jkb + nhnt);

                                }


                            }

                            init_us_2_acc_na = (nat + 1);


                            init_us_2_acc_na = init_us_2_acc_na;


                        }

                        init_us_2_acc_nt = (nsp + 1);


                        init_us_2_acc_nt = init_us_2_acc_nt;

                        gk_allocated = 0;
                        {

                            {
                                #pragma omp parallel for
                                for (auto __i0 = 0; __i0 < 3; __i0 += 1) {
                                    for (auto __i1 = 0; __i1 < n; __i1 += 1) {
                                        {
                                            double _out;

                                            ///////////////////
                                            // Tasklet code (set_gk)
                                            _out = 0;
                                            ///////////////////

                                            gk[(__i0 + (3 * __i1))] = _out;
                                        }
                                    }
                                }
                            }

                        }
                        ylm_allocated = 0;
                        {

                            {
                                #pragma omp parallel for
                                for (auto __i0 = 0; __i0 < n; __i0 += 1) {
                                    for (auto __i1 = 0; __i1 < ylm_d1; __i1 += 1) {
                                        {
                                            double _out;

                                            ///////////////////
                                            // Tasklet code (set_ylm)
                                            _out = 0;
                                            ///////////////////

                                            ylm[(__i0 + (__i1 * n))] = _out;
                                        }
                                    }
                                }
                            }

                        }
                        vq_allocated = 0;
                        {

                            {
                                #pragma omp parallel for
                                for (auto __i0 = 0; __i0 < n; __i0 += 1) {
                                    for (auto __i1 = 0; __i1 < nbetam; __i1 += 1) {
                                        {
                                            double _out;

                                            ///////////////////
                                            // Tasklet code (set_vq)
                                            _out = 0;
                                            ///////////////////

                                            vq[(__i0 + (__i1 * n))] = _out;
                                        }
                                    }
                                }
                            }

                        }
                        init_us_2_acc_qg_allocated = 0;
                        {

                            {
                                #pragma omp parallel for
                                for (auto __i0 = 0; __i0 < n; __i0 += 1) {
                                    {
                                        double _out;

                                        ///////////////////
                                        // Tasklet code (set_init_us_2_acc_qg)
                                        _out = 0;
                                        ///////////////////

                                        init_us_2_acc_qg[__i0] = _out;
                                    }
                                }
                            }

                        }
                        sk_allocated = 0;
                        {

                            {
                                #pragma omp parallel for
                                for (auto __i0 = 0; __i0 < n; __i0 += 1) {
                                    {
                                        dace::complex128 _out;

                                        ///////////////////
                                        // Tasklet code (set_sk)
                                        _out = 0;
                                        ///////////////////

                                        sk[__i0] = _out;
                                    }
                                }
                            }

                        }
                        vkb1_allocated = 0;
                        {

                            {
                                #pragma omp parallel for
                                for (auto __i0 = 0; __i0 < n; __i0 += 1) {
                                    for (auto __i1 = 0; __i1 < nhm; __i1 += 1) {
                                        {
                                            double _out;

                                            ///////////////////
                                            // Tasklet code (set_vkb1)
                                            _out = 0;
                                            ///////////////////

                                            vkb1[(__i0 + (__i1 * n))] = _out;
                                        }
                                    }
                                }
                            }

                        }
                    }


                    for (_loop_it_955 = 1; (_loop_it_955 < (nsp + 1)); _loop_it_955 = (_loop_it_955 + 1)) {


                        if (upf_tvanp) {

                            for (_loop_it_956 = 1; (_loop_it_956 < (nat + 1)); _loop_it_956 = (_loop_it_956 + 1)) {

                                {

                                    {
                                        int _in_ityp_0 = ityp[(_loop_it_956 - 1)];
                                        int64_t _out_if_cond_1340;

                                        ///////////////////
                                        // Tasklet code (t_1341)
                                        _out_if_cond_1340 = (_in_ityp_0 == _loop_it_955);
                                        ///////////////////

                                        if_cond_1340 = _out_if_cond_1340;
                                    }

                                }

                                if (if_cond_1340) {

                                    loopend_1343 = nh[(_loop_it_955 - 1)];


                                    for (_loop_it_957 = 1; (_loop_it_957 < (loopend_1343 + 1)); _loop_it_957 = (_loop_it_957 + 1)) {

                                        add_nlxx_pot_ikb = (ofsbeta[(_loop_it_956 - 1)] + _loop_it_957);

                                        if_cond_1348 = ((abs(deexx[((add_nlxx_pot_ikb + (nkb * (_loop_it_932 - 1))) - 1)]) < __assoc_scalar_16) != true);


                                        if (if_cond_1348) {

                                            if_cond_1351 = gamma_only[0];


                                            if (if_cond_1351) {

                                                for (_loop_it_958 = 1; (_loop_it_958 < (n + 1)); _loop_it_958 = (_loop_it_958 + 1)) {
                                                    {

                                                        {
                                                            dace::complex128 _in_add_nlxx_pot_deexx_0 = deexx[((add_nlxx_pot_ikb + (nkb * (_loop_it_932 - 1))) - 1)];
                                                            dace::complex128 _in_add_nlxx_pot_hpsi_0 = big_result[((_loop_it_958 + ((n * npol) * (ibnd - 1))) - 1)];
                                                            dace::complex128 _in_vkbp_0 = vkbp[((_loop_it_958 + (npwx * (add_nlxx_pot_ikb - 1))) - 1)];
                                                            double _in_exxalfa = exxalfa[0];
                                                            dace::complex128 _out_add_nlxx_pot_hpsi;

                                                            ///////////////////
                                                            // Tasklet code (t_0)
                                                            _out_add_nlxx_pot_hpsi = (_in_add_nlxx_pot_hpsi_0 - (((_in_exxalfa * _in_add_nlxx_pot_deexx_0.real()) + (dace::complex128(0.0, 1.0) * 0.0)) * _in_vkbp_0));
                                                            ///////////////////

                                                            big_result[((_loop_it_958 + ((n * npol) * (ibnd - 1))) - 1)] = _out_add_nlxx_pot_hpsi;
                                                        }

                                                    }

                                                }

                                                add_nlxx_pot_ig = (n + 1);


                                                add_nlxx_pot_ig = add_nlxx_pot_ig;

                                            } else {

                                                for (_loop_it_959 = 1; (_loop_it_959 < (n + 1)); _loop_it_959 = (_loop_it_959 + 1)) {
                                                    {

                                                        {
                                                            dace::complex128 _in_add_nlxx_pot_deexx_0 = deexx[((add_nlxx_pot_ikb + (nkb * (_loop_it_932 - 1))) - 1)];
                                                            dace::complex128 _in_add_nlxx_pot_hpsi_0 = big_result[((_loop_it_959 + ((n * npol) * (ibnd - 1))) - 1)];
                                                            dace::complex128 _in_vkbp_0 = vkbp[((_loop_it_959 + (npwx * (add_nlxx_pot_ikb - 1))) - 1)];
                                                            double _in_exxalfa = exxalfa[0];
                                                            dace::complex128 _out_add_nlxx_pot_hpsi;

                                                            ///////////////////
                                                            // Tasklet code (t_0)
                                                            _out_add_nlxx_pot_hpsi = (_in_add_nlxx_pot_hpsi_0 - (((_in_exxalfa + (dace::complex128(0.0, 1.0) * 0.0)) * _in_add_nlxx_pot_deexx_0) * _in_vkbp_0));
                                                            ///////////////////

                                                            big_result[((_loop_it_959 + ((n * npol) * (ibnd - 1))) - 1)] = _out_add_nlxx_pot_hpsi;
                                                        }

                                                    }

                                                }

                                                add_nlxx_pot_ig = (n + 1);


                                                add_nlxx_pot_ig = add_nlxx_pot_ig;

                                            }

                                        }


                                    }

                                    add_nlxx_pot_ih = (loopend_1343 + 1);


                                    add_nlxx_pot_ih = add_nlxx_pot_ih;

                                }


                            }

                            add_nlxx_pot_na = (nat + 1);


                            add_nlxx_pot_na = add_nlxx_pot_na;

                        }


                    }

                    add_nlxx_pot_np = (nsp + 1);


                    add_nlxx_pot_np = add_nlxx_pot_np;

                    vkbp_allocated = 0;
                    {

                        {
                            #pragma omp parallel for
                            for (auto __i0 = 0; __i0 < npwx; __i0 += 1) {
                                for (auto __i1 = 0; __i1 < nkb; __i1 += 1) {
                                    {
                                        dace::complex128 _out;

                                        ///////////////////
                                        // Tasklet code (set_vkbp)
                                        _out = 0;
                                        ///////////////////

                                        vkbp[(__i0 + (__i1 * npwx))] = _out;
                                    }
                                }
                            }
                        }

                    }
                }

            }

        }


    }

    delete[] deexx;
    delete[] result_d;
    delete[] result_nc_d;
    ii = (loopend_1092 + 1);


    ii = ii;

    rhoc_allocated = 0;
    {

        {
            #pragma omp parallel for
            for (auto __i0 = 0; __i0 < nrxxs; __i0 += 1) {
                for (auto __i1 = 0; __i1 < jblock; __i1 += 1) {
                    {
                        dace::complex128 _out;

                        ///////////////////
                        // Tasklet code (set_rhoc)
                        _out = 0;
                        ///////////////////

                        rhoc[(__i0 + (__i1 * nrxxs))] = _out;
                    }
                }
            }
        }
        delete[] rhoc;

    }
    vc_allocated = 0;
    {

        {
            #pragma omp parallel for
            for (auto __i0 = 0; __i0 < nrxxs; __i0 += 1) {
                for (auto __i1 = 0; __i1 < jblock; __i1 += 1) {
                    {
                        dace::complex128 _out;

                        ///////////////////
                        // Tasklet code (set_vc)
                        _out = 0;
                        ///////////////////

                        vc[(__i0 + (__i1 * nrxxs))] = _out;
                    }
                }
            }
        }
        delete[] vc;

    }
    rhoc_d_allocated = 0;
    {

        {
            #pragma omp parallel for
            for (auto __i0 = 0; __i0 < nrxxs; __i0 += 1) {
                for (auto __i1 = 0; __i1 < jblock; __i1 += 1) {
                    {
                        dace::complex128 _out;

                        ///////////////////
                        // Tasklet code (set_rhoc_d)
                        _out = 0;
                        ///////////////////

                        rhoc_d[(__i0 + (__i1 * nrxxs))] = _out;
                    }
                }
            }
        }
        delete[] rhoc_d;

    }
    vc_d_allocated = 0;
    {
        int __assoc_scalar_18;

        {
            #pragma omp parallel for
            for (auto __i0 = 0; __i0 < nrxxs; __i0 += 1) {
                for (auto __i1 = 0; __i1 < jblock; __i1 += 1) {
                    {
                        dace::complex128 _out;

                        ///////////////////
                        // Tasklet code (set_vc_d)
                        _out = 0;
                        ///////////////////

                        vc_d[(__i0 + (__i1 * nrxxs))] = _out;
                    }
                }
            }
        }
        {
            int _out;

            ///////////////////
            // Tasklet code (set___assoc_scalar_18)
            _out = (n * npol);
            ///////////////////

            __assoc_scalar_18 = _out;
        }
        copy_big_result_d_1372_sdfg_0_119_8(__state, &big_result[0], &big_result_d[0], m, n, npol);
        delete[] vc_d;

    }
    {

        {
            int _in_iexx_istart_0 = iexx_istart[my_egrp_id];
            int64_t _out_if_cond_1373;

            ///////////////////
            // Tasklet code (t_1374)
            _out_if_cond_1373 = (_in_iexx_istart_0 > 0);
            ///////////////////

            if_cond_1373 = _out_if_cond_1373;
        }

    }

    if (if_cond_1373) {

        if_cond_1377 = (negrp == 1);


        if (if_cond_1377) {

            ending_im = m;

        } else {

            ending_im = ((iexx_iend[my_egrp_id] - iexx_istart[my_egrp_id]) + 1);

        }


        if_cond_1384 = noncolin[0];


        if (if_cond_1384) {

            for (_loop_it_960 = 1; (_loop_it_960 < (ending_im + 1)); _loop_it_960 = (_loop_it_960 + 1)) {

                for (_loop_it_961 = 1; (_loop_it_961 < (n + 1)); _loop_it_961 = (_loop_it_961 + 1)) {

                    iexx_istart_d_at62 = iexx_istart_d[my_egrp_id];

                    {

                        {
                            dace::complex128 _in_big_result_d_0 = big_result_d[((_loop_it_961 + ((n * npol) * ((_loop_it_960 + iexx_istart_d_at62) - 2))) - 1)];
                            dace::complex128 _in_hpsi_d_0 = hpsi_d[((_loop_it_961 + (hpsi_d_d0 * (_loop_it_960 - 1))) - 1)];
                            dace::complex128 _out_hpsi_d;

                            ///////////////////
                            // Tasklet code (t_0)
                            _out_hpsi_d = (_in_hpsi_d_0 + _in_big_result_d_0);
                            ///////////////////

                            hpsi_d[((_loop_it_961 + (hpsi_d_d0 * (_loop_it_960 - 1))) - 1)] = _out_hpsi_d;
                        }

                    }
                    {

                        {
                            dace::complex128 _in_big_result_d_0 = big_result_d[(((_loop_it_961 + ((n * npol) * ((_loop_it_960 + iexx_istart_d_at62) - 2))) + n) - 1)];
                            dace::complex128 _in_hpsi_d_0 = hpsi_d[(((_loop_it_961 + (hpsi_d_d0 * (_loop_it_960 - 1))) + lda) - 1)];
                            dace::complex128 _out_hpsi_d;

                            ///////////////////
                            // Tasklet code (t_1)
                            _out_hpsi_d = (_in_hpsi_d_0 + _in_big_result_d_0);
                            ///////////////////

                            hpsi_d[(((_loop_it_961 + (hpsi_d_d0 * (_loop_it_960 - 1))) + lda) - 1)] = _out_hpsi_d;
                        }

                    }

                }

                ig = (n + 1);


                ig = ig;


            }

            program_im = (ending_im + 1);


            im = program_im;

        } else {

            for (_loop_it_962 = 1; (_loop_it_962 < (ending_im + 1)); _loop_it_962 = (_loop_it_962 + 1)) {

                for (_loop_it_963 = 1; (_loop_it_963 < (n + 1)); _loop_it_963 = (_loop_it_963 + 1)) {

                    iexx_istart_d_at63 = iexx_istart_d[my_egrp_id];

                    {

                        {
                            dace::complex128 _in_big_result_d_0 = big_result_d[((_loop_it_963 + ((n * npol) * ((_loop_it_962 + iexx_istart_d_at63) - 2))) - 1)];
                            dace::complex128 _in_hpsi_d_0 = hpsi_d[((_loop_it_963 + (hpsi_d_d0 * (_loop_it_962 - 1))) - 1)];
                            dace::complex128 _out_hpsi_d;

                            ///////////////////
                            // Tasklet code (t_0)
                            _out_hpsi_d = (_in_hpsi_d_0 + _in_big_result_d_0);
                            ///////////////////

                            hpsi_d[((_loop_it_963 + (hpsi_d_d0 * (_loop_it_962 - 1))) - 1)] = _out_hpsi_d;
                        }

                    }

                }

                ig = (n + 1);


                ig = ig;


            }

            program_im = (ending_im + 1);


            im = program_im;

        }

    }

    {

        copy_hpsi_1404_sdfg_0_122_2(__state, &hpsi_d[0], &hpsi[0], hpsi_d_d0, hpsi_d_d1, lda, npol);

    }
    big_result_allocated = 0;
    {

        {
            #pragma omp parallel for
            for (auto __i0 = 0; __i0 < (n * npol); __i0 += 1) {
                for (auto __i1 = 0; __i1 < m; __i1 += 1) {
                    {
                        dace::complex128 _out;

                        ///////////////////
                        // Tasklet code (set_big_result)
                        _out = 0;
                        ///////////////////

                        big_result[(__i0 + ((__i1 * n) * npol))] = _out;
                    }
                }
            }
        }

    }
    fac_allocated = 0;
    {
        double *fac;
        fac = new double DACE_ALIGN(64)[dfftt_ngm];

        {
            #pragma omp parallel for
            for (auto __i0 = 0; __i0 < dfftt_ngm; __i0 += 1) {
                {
                    double _out;

                    ///////////////////
                    // Tasklet code (set_fac)
                    _out = 0;
                    ///////////////////

                    fac[__i0] = _out;
                }
            }
        }
        delete[] fac;

    }
    facb_allocated = 0;
    {

        {
            #pragma omp parallel for
            for (auto __i0 = 0; __i0 < nrxxs; __i0 += 1) {
                {
                    double _out;

                    ///////////////////
                    // Tasklet code (set_facb)
                    _out = 0;
                    ///////////////////

                    facb[__i0] = _out;
                }
            }
        }
        delete[] facb;

    }
    big_result_d_allocated = 0;
    {

        {
            #pragma omp parallel for
            for (auto __i0 = 0; __i0 < (n * npol); __i0 += 1) {
                for (auto __i1 = 0; __i1 < m; __i1 += 1) {
                    {
                        dace::complex128 _out;

                        ///////////////////
                        // Tasklet code (set_big_result_d)
                        _out = 0;
                        ///////////////////

                        big_result_d[(__i0 + ((__i1 * n) * npol))] = _out;
                    }
                }
            }
        }

    }
    facb_d_allocated = 0;
    {

        {
            #pragma omp parallel for
            for (auto __i0 = 0; __i0 < nrxxs; __i0 += 1) {
                {
                    double _out;

                    ///////////////////
                    // Tasklet code (set_facb_d)
                    _out = 0;
                    ///////////////////

                    facb_d[__i0] = _out;
                }
            }
        }
        delete[] facb_d;

    }
    // MANUAL FIX (not auto-generated): closing ``hpsi = hpsi_d`` SOURCE= copy-
    // back -- the counterpart of the input capture near the top. The SDFG
    // lowering dropped it (same "uninitialized transient" prune), so the result
    // accumulated in hpsi_d never reached the host hpsi. Reverse of the input
    // copy: host stride lda*npol, transient stride hpsi_d_d0.
    for (int64_t __i1 = 0; __i1 < hpsi_d_d1; __i1 += 1)
        for (int64_t __i0 = 0; __i0 < hpsi_d_d0; __i0 += 1)
            hpsi[(__i0 + ((__i1 * lda) * npol))] = hpsi_d[(__i0 + (__i1 * hpsi_d_d0))];
    hpsi_d_allocated = 0;
    {

        {
            #pragma omp parallel for
            for (auto __i0 = 0; __i0 < hpsi_d_d0; __i0 += 1) {
                for (auto __i1 = 0; __i1 < hpsi_d_d1; __i1 += 1) {
                    {
                        dace::complex128 _out;

                        ///////////////////
                        // Tasklet code (set_hpsi_d)
                        _out = 0;
                        ///////////////////

                        hpsi_d[(__i0 + (__i1 * hpsi_d_d0))] = _out;
                    }
                }
            }
        }

    }
    delete[] big_result;
    delete[] big_result_d;
    delete[] dfftt__nl;
    delete[] hpsi_d;
    delete[] psi_d;
    delete[] xkp;
    delete[] xkq;
    delete[] nqhalf_dble;
    delete[] odg;
    delete[] g2_convolution_q;
    delete[] grid_factor_track;
    delete[] qq_track;
    delete[] i;
    delete[] i_real;
    delete[] a;
    delete[] qvan_init_q;
    delete[] qmod;
    delete[] qvan_init_qq;
    delete[] ylmk0;
    delete[] addusxx_g_aux1;
    delete[] addusxx_g_aux2;
    delete[] addusxx_g_eigqts;
    delete[] newdxx_g_aux1;
    delete[] newdxx_g_aux2;
    delete[] newdxx_g_eigqts;
    delete[] vkbp;
    delete[] gk;
    delete[] init_us_2_acc_qg;
    delete[] sk;
    delete[] vkb1;
    delete[] vq;
    delete[] ylm;
    delete[] _libtmp_0;
    delete[] _libsrc_0;
    delete[] _libsrc_2;
    delete[] _mask_4;
    delete[] _libsrc_6;
    delete[] _libtmp_1;
    delete[] _mask_5;
    delete[] _mask_7;
    delete[] _mask_8;
    delete[] _mask_10;
    delete[] _mask_11;
    delete[] _mask_12;
    delete[] _mask_13;
}

DACE_EXPORTED void __program_vexx_bp_k_gpu(vexx_bp_k_gpu_state_t *__state, int * __restrict__ all_end, int * __restrict__ all_start, double * __restrict__ ap, double * __restrict__ at, double * __restrict__ becphi_r, dace::complex128 * __restrict__ becpsi_k, int * __restrict__ becpsi_nbnd, dace::complex128 * __restrict__ becpsi_nc, double * __restrict__ becpsi_r, dace::complex128 * __restrict__ becxx_k, bool * __restrict__ coulomb_done, double * __restrict__ coulomb_fac, int * __restrict__ dfftt_nl, int * __restrict__ dfftt_nlm, int * __restrict__ egrp_pairs, dace::complex128 * __restrict__ eigts1, dace::complex128 * __restrict__ eigts2, dace::complex128 * __restrict__ eigts3, double * __restrict__ eps, double * __restrict__ eps_qdiv, double * __restrict__ erf_scrlen, double * __restrict__ erfc_scrlen, double * __restrict__ exxalfa, dace::complex128 * __restrict__ exxbuff, dace::complex128 * __restrict__ exxbuff_d, double * __restrict__ exxdiv, double * __restrict__ g, bool * __restrict__ gamma_only, double * __restrict__ gau_scrlen, double * __restrict__ grid_factor, double * __restrict__ gt, dace::complex128 * __restrict__ hpsi, int * __restrict__ ibands, int * __restrict__ iexx_iend, int * __restrict__ iexx_istart, int * __restrict__ iexx_istart_d, int * __restrict__ igk_exx, int * __restrict__ igk_exx_d, int * __restrict__ ijtoh, int * __restrict__ index_xk, int * __restrict__ index_xkq, int * __restrict__ indv, int * __restrict__ inter_egrp_comm, int * __restrict__ intra_egrp_comm, bool * __restrict__ ionode, int * __restrict__ ityp, double * __restrict__ ke_k, int * __restrict__ kunit, int * __restrict__ lpl, int * __restrict__ lpx, int * __restrict__ many_fft, int * __restrict__ me_egrp, int * __restrict__ mill, int * __restrict__ nh, int * __restrict__ nhtol, int * __restrict__ nhtolm, int * __restrict__ nibands, int * __restrict__ nij_type, int * __restrict__ nkstot, bool * __restrict__ noncolin, int * __restrict__ npool, int * __restrict__ nq1, int * __restrict__ nq2, int * __restrict__ nq3, int * __restrict__ ofsbeta, bool * __restrict__ okpaw, bool * __restrict__ okvan, double * __restrict__ omega, bool * __restrict__ paw_has_init_paw_fockrnl, dace::complex128 * __restrict__ psi, dace::complex128 * __restrict__ qgm, double * __restrict__ tab_beta, double * __restrict__ tab_qrad, int * __restrict__ tabxx_box, int * __restrict__ tabxx_maxbox, double * __restrict__ tabxx_qr, double * __restrict__ tau, double * __restrict__ tpiba, double * __restrict__ tpiba2, bool * __restrict__ tqr, int * __restrict__ upf_nbeta, bool * __restrict__ upf_tpawp, bool * __restrict__ upf_tvanp, bool * __restrict__ use_coulomb_vcut_spheric, bool * __restrict__ use_coulomb_vcut_ws, double * __restrict__ vcut_a, double * __restrict__ vcut_corrected, bool * __restrict__ x_gamma_extrapolation, double * __restrict__ x_occupation, double * __restrict__ x_occupation_d, double * __restrict__ xk, double * __restrict__ xkq_collect, double * __restrict__ yukawa, int64_t becpsi_k_d0, int64_t becpsi_nc_d0, int64_t becpsi_nc_d1, int64_t becxx_k_d0, int64_t becxx_k_d1, int current_k, int64_t dfftt__nl_d0, int dfftt_ngm, int dfftt_nnr, int64_t egrp_pairs_d0, int64_t egrp_pairs_d1, int64_t eigts1_d0, int64_t eigts2_d0, int64_t eigts3_d0, int64_t exxbuff_d0, int64_t exxbuff_d1, int64_t exxbuff_d2, int64_t exxbuff_d_d0, int64_t exxbuff_d_d1, int64_t g_d0, int gstart, int64_t gt_d0, int64_t hpsi_d_d0, int64_t hpsi_d_d1, int64_t ibands_d0, int iexx_start, int64_t igk_exx_d0, int64_t igk_exx_d_d0, int64_t ijtoh_d0, int64_t ijtoh_d1, int64_t index_xkq_d0, int64_t indv_d0, int jblock, int64_t ke_k_d0, int64_t ke_k_d1, int64_t ke_k_d2, int64_t ke_k_d3, int lda, int lmaxkb, int lmaxq, int m, int max_pairs, int64_t mill_d0, int my_egrp_id, int my_pool_id, int n, int nat, int nbetam, int negrp, int nhm, int64_t nhtol_d0, int64_t nhtolm_d0, int nkb, int npol, int npwx, int nqs, int nqx, int nsp, int64_t offset_becpsi_k_d0, int64_t offset_becpsi_k_d1, int64_t psi_d_d0, int64_t psi_d_d1, bool run_on_gpu_, int64_t tab_beta_d0, int64_t tab_beta_d1, int64_t tab_qrad_d0, int64_t tab_qrad_d1, int64_t tab_qrad_d2, int64_t tabxx_box_d0, int64_t tabxx_qr_d0, int64_t tabxx_qr_d1, int64_t tau_d0, int64_t vcut_corrected_d0, int64_t vcut_corrected_d1, int64_t vcut_corrected_d2, int64_t x_occupation_d0, int64_t x_occupation_d_d0, int64_t xkq_collect_d0, int64_t ylm_d1)
{
    __program_vexx_bp_k_gpu_internal(__state, all_end, all_start, ap, at, becphi_r, becpsi_k, becpsi_nbnd, becpsi_nc, becpsi_r, becxx_k, coulomb_done, coulomb_fac, dfftt_nl, dfftt_nlm, egrp_pairs, eigts1, eigts2, eigts3, eps, eps_qdiv, erf_scrlen, erfc_scrlen, exxalfa, exxbuff, exxbuff_d, exxdiv, g, gamma_only, gau_scrlen, grid_factor, gt, hpsi, ibands, iexx_iend, iexx_istart, iexx_istart_d, igk_exx, igk_exx_d, ijtoh, index_xk, index_xkq, indv, inter_egrp_comm, intra_egrp_comm, ionode, ityp, ke_k, kunit, lpl, lpx, many_fft, me_egrp, mill, nh, nhtol, nhtolm, nibands, nij_type, nkstot, noncolin, npool, nq1, nq2, nq3, ofsbeta, okpaw, okvan, omega, paw_has_init_paw_fockrnl, psi, qgm, tab_beta, tab_qrad, tabxx_box, tabxx_maxbox, tabxx_qr, tau, tpiba, tpiba2, tqr, upf_nbeta, upf_tpawp, upf_tvanp, use_coulomb_vcut_spheric, use_coulomb_vcut_ws, vcut_a, vcut_corrected, x_gamma_extrapolation, x_occupation, x_occupation_d, xk, xkq_collect, yukawa, becpsi_k_d0, becpsi_nc_d0, becpsi_nc_d1, becxx_k_d0, becxx_k_d1, current_k, dfftt__nl_d0, dfftt_ngm, dfftt_nnr, egrp_pairs_d0, egrp_pairs_d1, eigts1_d0, eigts2_d0, eigts3_d0, exxbuff_d0, exxbuff_d1, exxbuff_d2, exxbuff_d_d0, exxbuff_d_d1, g_d0, gstart, gt_d0, hpsi_d_d0, hpsi_d_d1, ibands_d0, iexx_start, igk_exx_d0, igk_exx_d_d0, ijtoh_d0, ijtoh_d1, index_xkq_d0, indv_d0, jblock, ke_k_d0, ke_k_d1, ke_k_d2, ke_k_d3, lda, lmaxkb, lmaxq, m, max_pairs, mill_d0, my_egrp_id, my_pool_id, n, nat, nbetam, negrp, nhm, nhtol_d0, nhtolm_d0, nkb, npol, npwx, nqs, nqx, nsp, offset_becpsi_k_d0, offset_becpsi_k_d1, psi_d_d0, psi_d_d1, run_on_gpu_, tab_beta_d0, tab_beta_d1, tab_qrad_d0, tab_qrad_d1, tab_qrad_d2, tabxx_box_d0, tabxx_qr_d0, tabxx_qr_d1, tau_d0, vcut_corrected_d0, vcut_corrected_d1, vcut_corrected_d2, x_occupation_d0, x_occupation_d_d0, xkq_collect_d0, ylm_d1);
}

DACE_EXPORTED vexx_bp_k_gpu_state_t *__dace_init_vexx_bp_k_gpu(int64_t becpsi_k_d0, int64_t becpsi_nc_d0, int64_t becpsi_nc_d1, int64_t becxx_k_d0, int64_t becxx_k_d1, int current_k, int64_t dfftt__nl_d0, int dfftt_ngm, int dfftt_nnr, int64_t egrp_pairs_d0, int64_t egrp_pairs_d1, int64_t eigts1_d0, int64_t eigts2_d0, int64_t eigts3_d0, int64_t exxbuff_d0, int64_t exxbuff_d1, int64_t exxbuff_d2, int64_t exxbuff_d_d0, int64_t exxbuff_d_d1, int64_t g_d0, int gstart, int64_t gt_d0, int64_t hpsi_d_d0, int64_t hpsi_d_d1, int64_t ibands_d0, int iexx_start, int64_t igk_exx_d0, int64_t igk_exx_d_d0, int64_t ijtoh_d0, int64_t ijtoh_d1, int64_t index_xkq_d0, int64_t indv_d0, int jblock, int64_t ke_k_d0, int64_t ke_k_d1, int64_t ke_k_d2, int64_t ke_k_d3, int lda, int lmaxkb, int lmaxq, int m, int max_pairs, int64_t mill_d0, int my_egrp_id, int my_pool_id, int n, int nat, int nbetam, int negrp, int nhm, int64_t nhtol_d0, int64_t nhtolm_d0, int nkb, int npol, int npwx, int nqs, int nqx, int nsp, int64_t offset_becpsi_k_d0, int64_t offset_becpsi_k_d1, int64_t psi_d_d0, int64_t psi_d_d1, int64_t tab_beta_d0, int64_t tab_beta_d1, int64_t tab_qrad_d0, int64_t tab_qrad_d1, int64_t tab_qrad_d2, int64_t tabxx_box_d0, int64_t tabxx_qr_d0, int64_t tabxx_qr_d1, int64_t tau_d0, int64_t vcut_corrected_d0, int64_t vcut_corrected_d1, int64_t vcut_corrected_d2, int64_t x_occupation_d0, int64_t x_occupation_d_d0, int64_t xkq_collect_d0, int64_t ylm_d1)
{

    int __result = 0;
    vexx_bp_k_gpu_state_t *__state = new vexx_bp_k_gpu_state_t;

    if (__result) {
        delete __state;
        return nullptr;
    }

    return __state;
}

DACE_EXPORTED int __dace_exit_vexx_bp_k_gpu(vexx_bp_k_gpu_state_t *__state)
{

    int __err = 0;
    delete __state;
    return __err;
}
