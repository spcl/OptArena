! Self-contained, FFT-free numeric core of exx_bp::vexx_bp_k (collinear /
! norm-conserving path) -- the three pointwise stages that bracket the band-pair
! convolution, lifted VERBATIM from the inlined kernel (vexx_bp_k_inlined.f90):
!
!   stage A  rhoc(ir,j) = CONJG(exxbuff(ir,j)) * temppsic(ir) * omega_inv   [l.668]
!   stage B  vc(ir,j)   = facb(ir) * rhoc(ir,j) * occ * nqs_inv             [l.691]
!   stage C  result(ir) = result(ir) + vc(ir,j) * exxbuff(ir,j)            [l.724]
!
! The two FFTs that sit between A->B and B->C are the irreducible external
! (``fwfft``/``invfft``, left unresolved by the inliner exactly as cegterg left
! FFT/devxlib external); this slice is the largest contiguous piece of vexx_bp_k
! the dace-fortran bridge lowers cleanly to C++, the analogue of cegterg's
! ``cegterg_rr`` Hermitianization core. The numpy cross-check (baseline/
! soa_cpp_check.py) runs the IDENTICAL FFT-free composite, so C++ == numpy
! bit-for-bit validates the flat-SoA complex-arithmetic lowering.
SUBROUTINE vexx_bp_k_core(nrxxs, jcount, omega_inv, nqs_inv, occ, &
                          facb, exxbuff, temppsic, result)
  IMPLICIT NONE
  INTEGER, INTENT(IN) :: nrxxs, jcount
  REAL(KIND=8), INTENT(IN) :: omega_inv, nqs_inv, occ
  REAL(KIND=8), INTENT(IN) :: facb(nrxxs)
  COMPLEX(KIND=8), INTENT(IN) :: exxbuff(nrxxs, jcount), temppsic(nrxxs)
  COMPLEX(KIND=8), INTENT(INOUT) :: result(nrxxs)
  COMPLEX(KIND=8) :: rhoc(nrxxs, jcount), vc(nrxxs, jcount)
  INTEGER :: ir, j
  DO j = 1, jcount
     DO ir = 1, nrxxs
        rhoc(ir, j) = CONJG(exxbuff(ir, j)) * temppsic(ir) * omega_inv
     END DO
  END DO
  DO j = 1, jcount
     DO ir = 1, nrxxs
        vc(ir, j) = facb(ir) * rhoc(ir, j) * occ * nqs_inv
     END DO
  END DO
  DO j = 1, jcount
     DO ir = 1, nrxxs
        result(ir) = result(ir) + vc(ir, j) * exxbuff(ir, j)
     END DO
  END DO
END SUBROUTINE vexx_bp_k_core
