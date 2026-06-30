/* DaCe AUTO-GENERATED FILE. DO NOT MODIFY */
#include <dace/dace.h>
#include "../../include/hash.h"

struct cegterg_rr_state_t {

};

void __program_cegterg_rr_internal(cegterg_rr_state_t*__state, dace::complex128 * __restrict__ hc, dace::complex128 * __restrict__ sc, int nbase, int nvecx)
{
    int n;
    int64_t _loop_it_0;
    int64_t loopbegin_4;
    int m;
    int64_t _loop_it_1;


    for (_loop_it_0 = 1; (_loop_it_0 < (nbase + 1)); _loop_it_0 = (_loop_it_0 + 1)) {
        {

            {
                dace::complex128 _in_hc_0 = hc[((_loop_it_0 + (nvecx * (_loop_it_0 - 1))) - 1)];
                dace::complex128 _out_hc;

                ///////////////////
                // Tasklet code (t_2)
                _out_hc = (_in_hc_0.real() + (dace::complex128(0.0, 1.0) * 0.0));
                ///////////////////

                hc[((_loop_it_0 + (nvecx * (_loop_it_0 - 1))) - 1)] = _out_hc;
            }
            {
                dace::complex128 _in_sc_0 = sc[((_loop_it_0 + (nvecx * (_loop_it_0 - 1))) - 1)];
                dace::complex128 _out_sc;

                ///////////////////
                // Tasklet code (t_3)
                _out_sc = (_in_sc_0.real() + (dace::complex128(0.0, 1.0) * 0.0));
                ///////////////////

                sc[((_loop_it_0 + (nvecx * (_loop_it_0 - 1))) - 1)] = _out_sc;
            }

        }
        loopbegin_4 = (_loop_it_0 + 1);


        for (_loop_it_1 = loopbegin_4; (_loop_it_1 < (nbase + 1)); _loop_it_1 = (_loop_it_1 + 1)) {
            {

                {
                    dace::complex128 _in_hc_0 = hc[((_loop_it_1 + (nvecx * (_loop_it_0 - 1))) - 1)];
                    dace::complex128 _in_hc_1 = hc[((_loop_it_1 + (nvecx * (_loop_it_0 - 1))) - 1)];
                    dace::complex128 _out_hc;

                    ///////////////////
                    // Tasklet code (t_0)
                    _out_hc = conj(_in_hc_0);
                    ///////////////////

                    hc[((_loop_it_0 + (nvecx * (_loop_it_1 - 1))) - 1)] = _out_hc;
                }
                {
                    dace::complex128 _in_sc_0 = sc[((_loop_it_1 + (nvecx * (_loop_it_0 - 1))) - 1)];
                    dace::complex128 _in_sc_1 = sc[((_loop_it_1 + (nvecx * (_loop_it_0 - 1))) - 1)];
                    dace::complex128 _out_sc;

                    ///////////////////
                    // Tasklet code (t_1)
                    _out_sc = conj(_in_sc_0);
                    ///////////////////

                    sc[((_loop_it_0 + (nvecx * (_loop_it_1 - 1))) - 1)] = _out_sc;
                }

            }

        }

        m = (nbase + 1);


        m = m;


    }

    n = (nbase + 1);


    n = n;

}

DACE_EXPORTED void __program_cegterg_rr(cegterg_rr_state_t *__state, dace::complex128 * __restrict__ hc, dace::complex128 * __restrict__ sc, int nbase, int nvecx)
{
    __program_cegterg_rr_internal(__state, hc, sc, nbase, nvecx);
}

DACE_EXPORTED cegterg_rr_state_t *__dace_init_cegterg_rr(int nbase, int nvecx)
{

    int __result = 0;
    cegterg_rr_state_t *__state = new cegterg_rr_state_t;

    if (__result) {
        delete __state;
        return nullptr;
    }

    return __state;
}

DACE_EXPORTED int __dace_exit_cegterg_rr(cegterg_rr_state_t *__state)
{

    int __err = 0;
    delete __state;
    return __err;
}
