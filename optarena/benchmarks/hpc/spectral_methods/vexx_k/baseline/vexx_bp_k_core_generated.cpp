/* DaCe AUTO-GENERATED FILE. DO NOT MODIFY */
#include <dace/dace.h>
#include "../../include/hash.h"

struct vexx_bp_k_core_state_t {

};

void __program_vexx_bp_k_core_internal(vexx_bp_k_core_state_t*__state, dace::complex128 * __restrict__ exxbuff, double * __restrict__ facb, dace::complex128 * __restrict__ result, dace::complex128 * __restrict__ temppsic, int jcount, double nqs_inv, int nrxxs, double occ, double omega_inv)
{
    dace::complex128 *rhoc;
    rhoc = new dace::complex128 DACE_ALIGN(64)[((nrxxs * (jcount - 1)) + nrxxs)];
    dace::complex128 *vc;
    vc = new dace::complex128 DACE_ALIGN(64)[((nrxxs * (jcount - 1)) + nrxxs)];
    int j;
    int64_t _loop_it_0;
    int ir;
    int64_t _loop_it_1;
    int64_t _loop_it_2;
    int64_t _loop_it_3;
    int64_t _loop_it_4;
    int64_t _loop_it_5;


    for (_loop_it_0 = 1; (_loop_it_0 < (jcount + 1)); _loop_it_0 = (_loop_it_0 + 1)) {

        for (_loop_it_1 = 1; (_loop_it_1 < (nrxxs + 1)); _loop_it_1 = (_loop_it_1 + 1)) {
            {

                {
                    dace::complex128 _in_exxbuff_0 = exxbuff[((_loop_it_1 + (nrxxs * (_loop_it_0 - 1))) - 1)];
                    dace::complex128 _in_exxbuff_1 = exxbuff[((_loop_it_1 + (nrxxs * (_loop_it_0 - 1))) - 1)];
                    dace::complex128 _in_temppsic_0 = temppsic[(_loop_it_1 - 1)];
                    double _in_omega_inv = omega_inv;
                    dace::complex128 _out_rhoc;

                    ///////////////////
                    // Tasklet code (t_0)
                    _out_rhoc = ((conj(_in_exxbuff_0) * _in_temppsic_0) * (_in_omega_inv + (dace::complex128(0.0, 1.0) * 0.0)));
                    ///////////////////

                    rhoc[((_loop_it_1 + (nrxxs * (_loop_it_0 - 1))) - 1)] = _out_rhoc;
                }

            }

        }

        ir = (nrxxs + 1);


        ir = ir;


    }

    j = (jcount + 1);


    j = j;


    for (_loop_it_2 = 1; (_loop_it_2 < (jcount + 1)); _loop_it_2 = (_loop_it_2 + 1)) {

        for (_loop_it_3 = 1; (_loop_it_3 < (nrxxs + 1)); _loop_it_3 = (_loop_it_3 + 1)) {
            {

                {
                    double _in_facb_0 = facb[(_loop_it_3 - 1)];
                    dace::complex128 _in_rhoc_0 = rhoc[((_loop_it_3 + (nrxxs * (_loop_it_2 - 1))) - 1)];
                    double _in_nqs_inv = nqs_inv;
                    double _in_occ = occ;
                    dace::complex128 _out_vc;

                    ///////////////////
                    // Tasklet code (t_0)
                    _out_vc = ((((_in_facb_0 + (dace::complex128(0.0, 1.0) * 0.0)) * _in_rhoc_0) * (_in_occ + (dace::complex128(0.0, 1.0) * 0.0))) * (_in_nqs_inv + (dace::complex128(0.0, 1.0) * 0.0)));
                    ///////////////////

                    vc[((_loop_it_3 + (nrxxs * (_loop_it_2 - 1))) - 1)] = _out_vc;
                }

            }

        }

        ir = (nrxxs + 1);


        ir = ir;


    }

    j = (jcount + 1);


    j = j;


    for (_loop_it_4 = 1; (_loop_it_4 < (jcount + 1)); _loop_it_4 = (_loop_it_4 + 1)) {

        for (_loop_it_5 = 1; (_loop_it_5 < (nrxxs + 1)); _loop_it_5 = (_loop_it_5 + 1)) {
            {

                {
                    dace::complex128 _in_exxbuff_0 = exxbuff[((_loop_it_5 + (nrxxs * (_loop_it_4 - 1))) - 1)];
                    dace::complex128 _in_result_0 = result[(_loop_it_5 - 1)];
                    dace::complex128 _in_vc_0 = vc[((_loop_it_5 + (nrxxs * (_loop_it_4 - 1))) - 1)];
                    dace::complex128 _out_result;

                    ///////////////////
                    // Tasklet code (t_0)
                    _out_result = (_in_result_0 + (_in_vc_0 * _in_exxbuff_0));
                    ///////////////////

                    result[(_loop_it_5 - 1)] = _out_result;
                }

            }

        }

        ir = (nrxxs + 1);


        ir = ir;


    }

    j = (jcount + 1);


    j = j;

    delete[] rhoc;
    delete[] vc;
}

DACE_EXPORTED void __program_vexx_bp_k_core(vexx_bp_k_core_state_t *__state, dace::complex128 * __restrict__ exxbuff, double * __restrict__ facb, dace::complex128 * __restrict__ result, dace::complex128 * __restrict__ temppsic, int jcount, double nqs_inv, int nrxxs, double occ, double omega_inv)
{
    __program_vexx_bp_k_core_internal(__state, exxbuff, facb, result, temppsic, jcount, nqs_inv, nrxxs, occ, omega_inv);
}

DACE_EXPORTED vexx_bp_k_core_state_t *__dace_init_vexx_bp_k_core(int jcount, int nrxxs)
{

    int __result = 0;
    vexx_bp_k_core_state_t *__state = new vexx_bp_k_core_state_t;

    if (__result) {
        delete __state;
        return nullptr;
    }

    return __state;
}

DACE_EXPORTED int __dace_exit_vexx_bp_k_core(vexx_bp_k_core_state_t *__state)
{

    int __err = 0;
    delete __state;
    return __err;
}
