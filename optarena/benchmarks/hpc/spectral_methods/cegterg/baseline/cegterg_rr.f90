SUBROUTINE cegterg_rr(nbase, nvecx, hc, sc)
  IMPLICIT NONE
  INTEGER, INTENT(IN) :: nbase, nvecx
  COMPLEX(KIND=8), INTENT(INOUT) :: hc(nvecx, nvecx), sc(nvecx, nvecx)
  INTEGER :: n, m
  DO n = 1, nbase
     hc(n, n) = CMPLX(REAL(hc(n, n)), 0.0D0, kind=8)
     sc(n, n) = CMPLX(REAL(sc(n, n)), 0.0D0, kind=8)
     DO m = n + 1, nbase
        hc(n, m) = CONJG(hc(m, n))
        sc(n, m) = CONJG(sc(m, n))
     END DO
  END DO
END SUBROUTINE cegterg_rr
