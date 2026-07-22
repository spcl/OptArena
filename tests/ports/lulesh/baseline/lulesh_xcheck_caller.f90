! ============================================================================
! GPL-3.0-or-later (derives from the vendored GPL LULESH; see NOTICE.md).
!
! bind(c) cross-check harness for the HPCAgent-Bench numpy LULESH port. It calls the
! GENUINE vendored LULESH kernels (USE lulesh_comp_kernels) so the numpy SoA
! reference can be pinned numerically against the authoritative Fortran.
!
! Two layers:
!   * LEAF kernels (plain (0:7)-per-element array dummies) -- forwarded directly:
!     CalcElemVolume, CalcElemShapeFunctionDerivatives, CalcElemNodeNormals,
!     CalcElemVolumeDerivative, CalcElemCharacteristicLength,
!     CalcElemVelocityGrandient, CalcElemFBHourglassForce.
!   * INTEGRATED nodal-force assembly (c_volume_force) -- built ENTIRELY from
!     those genuine leaf kernels, with the element loop + node scatter-add done
!     here. This validates the full stress-integration + Flanagan-Belytschko
!     hourglass force assembly authoritatively while AVOIDING the vendored
!     domain-iteration routines (IntegrateStressForElems /
!     CalcFBHourglassForceForElems / CalcMonotonicQGradientsForElems), whose
!     serial paths carry never-executed upstream bugs (a 1-based loop over the
!     0-based sig array in InitStressTermsForElems; an unallocated fx_local in
!     IntegrateStressForElems; 0-vs-1-based row-slice pointer bounds in the
!     ``elemToNode => m_nodelist(i,:)`` consumers). See the test docstring.
!   * EOS (c_eos) -- drives the genuine ApplyMaterialPropertiesForElems in the
!     single-region configuration (regElemlist == identity), which does NOT use
!     the buggy row-slice pointers, so it runs as-is.
! ============================================================================
MODULE lulesh_xcheck
  USE lulesh_comp_kernels
  USE ISO_C_BINDING
  IMPLICIT NONE
  TYPE(domain_type) :: dom
CONTAINS

  REAL(C_DOUBLE) FUNCTION c_elem_volume(x, y, z) BIND(C, name="c_elem_volume")
    REAL(C_DOUBLE), DIMENSION(0:7), INTENT(IN) :: x, y, z
    c_elem_volume = CalcElemVolume(x, y, z)
  END FUNCTION

  SUBROUTINE c_shape_fn(x, y, z, b, vol) BIND(C, name="c_shape_fn")
    REAL(C_DOUBLE), DIMENSION(0:7), INTENT(IN) :: x, y, z
    REAL(C_DOUBLE), DIMENSION(0:23), INTENT(OUT) :: b   ! (0:7,0:2) col-major
    REAL(C_DOUBLE), INTENT(OUT) :: vol
    REAL(KIND=8), DIMENSION(0:7, 0:2) :: bb
    REAL(KIND=8) :: x0(0:7), y0(0:7), z0(0:7)
    x0 = x; y0 = y; z0 = z
    CALL CalcElemShapeFunctionDerivatives(x0, y0, z0, bb, vol)
    b = RESHAPE(bb, (/24/))
  END SUBROUTINE

  SUBROUTINE c_node_normals(x, y, z, pf) BIND(C, name="c_node_normals")
    REAL(C_DOUBLE), DIMENSION(0:7), INTENT(IN) :: x, y, z
    REAL(C_DOUBLE), DIMENSION(0:23), INTENT(OUT) :: pf
    REAL(KIND=8) :: pfx(0:7), pfy(0:7), pfz(0:7), x0(0:7), y0(0:7), z0(0:7)
    x0 = x; y0 = y; z0 = z
    CALL CalcElemNodeNormals(pfx, pfy, pfz, x0, y0, z0)
    pf(0:7) = pfx; pf(8:15) = pfy; pf(16:23) = pfz
  END SUBROUTINE

  SUBROUTINE c_vol_deriv(x, y, z, dvdx, dvdy, dvdz) BIND(C, name="c_vol_deriv")
    REAL(C_DOUBLE), DIMENSION(0:7), INTENT(IN) :: x, y, z
    REAL(C_DOUBLE), DIMENSION(0:7), INTENT(OUT) :: dvdx, dvdy, dvdz
    REAL(KIND=8) :: x0(0:7), y0(0:7), z0(0:7)
    x0 = x; y0 = y; z0 = z
    CALL CalcElemVolumeDerivative(dvdx, dvdy, dvdz, x0, y0, z0)
  END SUBROUTINE

  REAL(C_DOUBLE) FUNCTION c_char_len(x, y, z, vol) BIND(C, name="c_char_len")
    REAL(C_DOUBLE), DIMENSION(0:7), INTENT(IN) :: x, y, z
    REAL(C_DOUBLE), VALUE :: vol
    REAL(KIND=8) :: x0(0:7), y0(0:7), z0(0:7)
    x0 = x; y0 = y; z0 = z
    c_char_len = CalcElemCharacteristicLength(x0, y0, z0, vol)
  END FUNCTION

  SUBROUTINE c_vel_grad(xv, yv, zv, b, detJ, d) BIND(C, name="c_vel_grad")
    REAL(C_DOUBLE), DIMENSION(0:7), INTENT(IN) :: xv, yv, zv
    REAL(C_DOUBLE), DIMENSION(0:23), INTENT(IN) :: b
    REAL(C_DOUBLE), VALUE :: detJ
    REAL(C_DOUBLE), DIMENSION(0:5), INTENT(OUT) :: d
    REAL(KIND=8), DIMENSION(0:7, 0:2) :: bb
    bb = RESHAPE(b, (/8, 3/))
    CALL CalcElemVelocityGrandient(xv, yv, zv, bb, detJ, d)
  END SUBROUTINE

  SUBROUTINE c_fb_hg_force(xd, yd, zd, hourgam, coeff, hgfx, hgfy, hgfz) BIND(C, name="c_fb_hg_force")
    REAL(C_DOUBLE), DIMENSION(0:7), INTENT(IN) :: xd, yd, zd
    REAL(C_DOUBLE), DIMENSION(0:31), INTENT(IN) :: hourgam  ! (0:3,0:7) col-major
    REAL(C_DOUBLE), VALUE :: coeff
    REAL(C_DOUBLE), DIMENSION(0:7), INTENT(OUT) :: hgfx, hgfy, hgfz
    REAL(KIND=8), DIMENSION(0:3, 0:7) :: hg
    hg = RESHAPE(hourgam, (/4, 8/))
    CALL CalcElemFBHourglassForce(xd, yd, zd, hg, coeff, hgfx, hgfy, hgfz)
  END SUBROUTINE

  ! Full nodal-force assembly from genuine leaf kernels (see header).
  SUBROUTINE c_volume_force(numElem, numNode, nodelist, x, y, z, &
                            sig, ssv, elemMass, volo, vrel, xd, yd, zd, &
                            hgcoef, fx, fy, fz) BIND(C, name="c_volume_force")
    INTEGER(C_INT), VALUE :: numElem, numNode
    INTEGER(C_INT), DIMENSION(0:numElem*8 - 1), INTENT(IN) :: nodelist
    REAL(C_DOUBLE), DIMENSION(0:numNode - 1), INTENT(IN) :: x, y, z, xd, yd, zd
    REAL(C_DOUBLE), DIMENSION(0:numElem - 1), INTENT(IN) :: sig, ssv, elemMass, volo, vrel
    REAL(C_DOUBLE), VALUE :: hgcoef
    REAL(C_DOUBLE), DIMENSION(0:numNode - 1), INTENT(OUT) :: fx, fy, fz
    REAL(KIND=8) :: xl(0:7), yl(0:7), zl(0:7), B(0:7, 0:2), determ_e
    REAL(KIND=8) :: floc_x(0:7), floc_y(0:7), floc_z(0:7)
    REAL(KIND=8) :: dvdx(0:7), dvdy(0:7), dvdz(0:7)
    REAL(KIND=8) :: gamma(0:7, 0:3), hourgam(0:3, 0:7)
    REAL(KIND=8) :: hourmodx, hourmody, hourmodz, volinv, coeff, ss1, mass1, volume13
    REAL(KIND=8) :: xd1(0:7), yd1(0:7), zd1(0:7), hgfx(0:7), hgfy(0:7), hgfz(0:7)
    REAL(KIND=8) :: determ(0:numElem - 1)
    INTEGER :: e, j, g, i1, k

    fx = 0.0_8; fy = 0.0_8; fz = 0.0_8

    DO e = 0, numElem - 1
      DO j = 0, 7
        g = nodelist(e*8 + j); xl(j) = x(g); yl(j) = y(g); zl(j) = z(g)
      END DO
      CALL CalcElemShapeFunctionDerivatives(xl, yl, zl, B, determ_e)
      determ(e) = determ_e
      CALL CalcElemNodeNormals(B(:, 0), B(:, 1), B(:, 2), xl, yl, zl)
      CALL SumElemStressesToNodeForces(B, sig(e), sig(e), sig(e), floc_x, floc_y, floc_z)
      DO j = 0, 7
        g = nodelist(e*8 + j)
        fx(g) = fx(g) + floc_x(j); fy(g) = fy(g) + floc_y(j); fz(g) = fz(g) + floc_z(j)
      END DO
    END DO

    gamma(0, 0) = 1; gamma(1, 0) = 1; gamma(2, 0) = -1; gamma(3, 0) = -1
    gamma(4, 0) = -1; gamma(5, 0) = -1; gamma(6, 0) = 1; gamma(7, 0) = 1
    gamma(0, 1) = 1; gamma(1, 1) = -1; gamma(2, 1) = -1; gamma(3, 1) = 1
    gamma(4, 1) = -1; gamma(5, 1) = 1; gamma(6, 1) = 1; gamma(7, 1) = -1
    gamma(0, 2) = 1; gamma(1, 2) = -1; gamma(2, 2) = 1; gamma(3, 2) = -1
    gamma(4, 2) = 1; gamma(5, 2) = -1; gamma(6, 2) = 1; gamma(7, 2) = -1
    gamma(0, 3) = -1; gamma(1, 3) = 1; gamma(2, 3) = -1; gamma(3, 3) = 1
    gamma(4, 3) = 1; gamma(5, 3) = -1; gamma(6, 3) = 1; gamma(7, 3) = -1

    DO e = 0, numElem - 1
      DO j = 0, 7
        g = nodelist(e*8 + j); xl(j) = x(g); yl(j) = y(g); zl(j) = z(g)
      END DO
      CALL CalcElemVolumeDerivative(dvdx, dvdy, dvdz, xl, yl, zl)
      determ(e) = volo(e)*vrel(e)
      volinv = 1.0_8/determ(e)
      DO i1 = 0, 3
        hourmodx = 0; hourmody = 0; hourmodz = 0
        DO k = 0, 7
          hourmodx = hourmodx + xl(k)*gamma(k, i1)
          hourmody = hourmody + yl(k)*gamma(k, i1)
          hourmodz = hourmodz + zl(k)*gamma(k, i1)
        END DO
        DO k = 0, 7
          hourgam(i1, k) = gamma(k, i1) - volinv*(dvdx(k)*hourmodx + dvdy(k)*hourmody + dvdz(k)*hourmodz)
        END DO
      END DO
      ss1 = ssv(e); mass1 = elemMass(e); volume13 = CBRT(determ(e))
      DO j = 0, 7
        g = nodelist(e*8 + j); xd1(j) = xd(g); yd1(j) = yd(g); zd1(j) = zd(g)
      END DO
      coeff = -hgcoef*0.01_8*ss1*mass1/volume13
      CALL CalcElemFBHourglassForce(xd1, yd1, zd1, hourgam, coeff, hgfx, hgfy, hgfz)
      DO j = 0, 7
        g = nodelist(e*8 + j)
        fx(g) = fx(g) + hgfx(j); fy(g) = fy(g) + hgfy(j); fz(g) = fz(g) + hgfz(j)
      END DO
    END DO
  END SUBROUTINE

  ! Full EOS via the genuine ApplyMaterialPropertiesForElems (single region).
  SUBROUTINE c_eos(numElem, e, p, q, qq, ql, v, vnew, volo, delv, elemMass, &
                   eo, po, qo, sso) BIND(C, name="c_eos")
    INTEGER(C_INT), VALUE :: numElem
    REAL(C_DOUBLE), DIMENSION(0:numElem - 1), INTENT(IN) :: e, p, q, qq, ql, v, vnew, volo, delv, elemMass
    REAL(C_DOUBLE), DIMENSION(0:numElem - 1), INTENT(OUT) :: eo, po, qo, sso
    INTEGER :: i
    IF (.NOT. ALLOCATED(dom%m_e)) THEN
      dom%m_numElem = numElem; dom%m_numNode = numElem; dom%m_numReg = 1; dom%m_cost = 1
      CALL AllocateElemPersistent(dom, numElem)
      CALL AllocateNodalPersistent(dom, numElem)
      ALLOCATE (dom%m_regElemSize(0:0), dom%m_regElemKeys(0:1), dom%m_regElemlist(0:numElem - 1))
      dom%m_regElemSize(0) = numElem; dom%m_regElemKeys(0) = 0; dom%m_regElemKeys(1) = numElem
      DO i = 0, numElem - 1
        dom%m_regElemlist(i) = i
      END DO
      dom%m_e_cut = 1.0e-7_8; dom%m_p_cut = 1.0e-7_8; dom%m_q_cut = 1.0e-7_8
      dom%m_ss4o3 = 4.0_8/3.0_8; dom%m_eosvmax = 1.0e+9_8; dom%m_eosvmin = 1.0e-9_8
      dom%m_pmin = 0.0_8; dom%m_emin = -1.0e+15_8; dom%m_refdens = 1.0_8
    END IF
    DO i = 0, numElem - 1
      dom%m_e(i) = e(i); dom%m_p(i) = p(i); dom%m_q(i) = q(i); dom%m_qq(i) = qq(i)
      dom%m_ql(i) = ql(i); dom%m_v(i) = v(i); dom%m_vnew(i) = vnew(i)
      dom%m_volo(i) = volo(i); dom%m_delv(i) = delv(i); dom%m_elemMass(i) = elemMass(i); dom%m_ss(i) = 0.0_8
    END DO
    CALL ApplyMaterialPropertiesForElems(dom)
    DO i = 0, numElem - 1
      eo(i) = dom%m_e(i); po(i) = dom%m_p(i); qo(i) = dom%m_q(i); sso(i) = dom%m_ss(i)
    END DO
  END SUBROUTINE

  ! Full bit-exact end-to-end reference: build the Sedov cubic mesh exactly like
  ! the driver lulesh.f90, run `nsteps` genuine LagrangeLeapFrog cycles, and
  ! return the final state. Single-region (numReg==1). Uses a FRESH domain each
  ! call (deallocated first) so repeated invocations are independent.
  SUBROUTINE c_run_full(edgeElems, nsteps, eo, po, qo, vo, &
                        xo, yo, zo, xdo, ydo, zdo) BIND(C, name="c_run_full")
    INTEGER(C_INT), VALUE :: edgeElems, nsteps
    REAL(C_DOUBLE), DIMENSION(0:edgeElems**3 - 1), INTENT(OUT) :: eo, po, qo, vo
    REAL(C_DOUBLE), DIMENSION(0:(edgeElems + 1)**3 - 1), INTENT(OUT) :: xo, yo, zo, xdo, ydo, zdo
    INTEGER, PARAMETER :: XI_M_SYMM = INT(z'001'), XI_P_FREE = INT(z'008')
    INTEGER, PARAMETER :: ETA_M_SYMM = INT(z'010'), ETA_P_FREE = INT(z'080')
    INTEGER, PARAMETER :: ZETA_M_SYMM = INT(z'100'), ZETA_P_FREE = INT(z'800')
    REAL(KIND=8), PARAMETER :: ebase = 3.948746e+7_8
    TYPE(domain_type) :: d
    INTEGER :: edgeNodes, domElems, nidx, zidx, col, row, plane, i, j, k, lnode, gnode, idx
    INTEGER :: planeInc, rowInc
    REAL(KIND=8) :: tx, ty, tz, volume, scale, einit
    REAL(KIND=8) :: x_local(0:7), y_local(0:7), z_local(0:7)

    edgeNodes = edgeElems + 1
    d%m_symm_is_set = .FALSE.
    d%m_sizeX = edgeElems; d%m_sizeY = edgeElems; d%m_sizeZ = edgeElems
    d%m_numElem = edgeElems**3; d%m_numNode = edgeNodes**3
    domElems = d%m_numElem

    ALLOCATE (d%m_regNumList(0:d%m_numElem - 1))
    CALL AllocateElemPersistent(d, d%m_numElem)
    CALL AllocateNodalPersistent(d, d%m_numNode)
    CALL AllocateNodesets(d, edgeNodes*edgeNodes)

    DO i = 0, d%m_numElem - 1
      d%m_e(i) = 0.0_8; d%m_p(i) = 0.0_8; d%m_q(i) = 0.0_8; d%m_ss(i) = 0.0_8; d%m_v(i) = 1.0_8
    END DO
    DO i = 0, d%m_numNode - 1
      d%m_xd(i) = 0.0_8; d%m_yd(i) = 0.0_8; d%m_zd(i) = 0.0_8
      d%m_xdd(i) = 0.0_8; d%m_ydd(i) = 0.0_8; d%m_zdd(i) = 0.0_8; d%m_nodalMass(i) = 0.0_8
    END DO

    ! BuildMesh nodal coordinates (colLoc/rowLoc/planeLoc=0, m_tp=1).
    nidx = 0; tz = 0.0_8
    DO plane = 0, edgeNodes - 1
      ty = 0.0_8
      DO row = 0, edgeNodes - 1
        tx = 0.0_8
        DO col = 0, edgeNodes - 1
          d%m_x(nidx) = tx; d%m_y(nidx) = ty; d%m_z(nidx) = tz
          nidx = nidx + 1
          tx = 1.125_8*(col + 1)/edgeElems
        END DO
        ty = 1.125_8*(row + 1)/edgeElems
      END DO
      tz = 1.125_8*(plane + 1)/edgeElems
    END DO

    ! nodelist
    nidx = 0; zidx = 0
    DO plane = 0, edgeElems - 1
      DO row = 0, edgeElems - 1
        DO col = 0, edgeElems - 1
          d%m_nodelist(zidx, 0) = nidx
          d%m_nodelist(zidx, 1) = nidx + 1
          d%m_nodelist(zidx, 2) = nidx + edgeNodes + 1
          d%m_nodelist(zidx, 3) = nidx + edgeNodes
          d%m_nodelist(zidx, 4) = nidx + edgeNodes*edgeNodes
          d%m_nodelist(zidx, 5) = nidx + edgeNodes*edgeNodes + 1
          d%m_nodelist(zidx, 6) = nidx + edgeNodes*edgeNodes + edgeNodes + 1
          d%m_nodelist(zidx, 7) = nidx + edgeNodes*edgeNodes + edgeNodes
          zidx = zidx + 1; nidx = nidx + 1
        END DO
        nidx = nidx + 1
      END DO
      nidx = nidx + edgeNodes
    END DO

    ! single region
    d%m_numReg = 1; d%m_cost = 1
    ALLOCATE (d%m_regElemSize(0:0), d%m_regElemKeys(0:1), d%m_regElemlist(0:d%m_numElem - 1))
    d%m_regElemSize(0) = d%m_numElem; d%m_regElemKeys(0) = 0; d%m_regElemKeys(1) = d%m_numElem
    DO i = 0, d%m_numElem - 1
      d%m_regNumList(i) = 1; d%m_regElemlist(i) = i
    END DO

    ! volo / elemMass / nodalMass
    DO i = 0, domElems - 1
      DO lnode = 0, 7
        gnode = d%m_nodelist(i, lnode)
        x_local(lnode) = d%m_x(gnode); y_local(lnode) = d%m_y(gnode); z_local(lnode) = d%m_z(gnode)
      END DO
      volume = CalcElemVolume(x_local, y_local, z_local)
      d%m_volo(i) = volume; d%m_elemMass(i) = volume
      DO j = 0, 7
        idx = d%m_nodelist(i, j)
        d%m_nodalMass(idx) = d%m_nodalMass(idx) + volume/8.0_8
      END DO
    END DO

    scale = edgeElems/45.0_8
    einit = ebase*scale*scale*scale
    d%m_e(0) = einit

    ! symmetry nodesets
    nidx = 0
    DO i = 0, edgeNodes - 1
      planeInc = i*edgeNodes*edgeNodes; rowInc = i*edgeNodes
      DO j = 0, edgeNodes - 1
        d%m_symmX(nidx) = planeInc + j*edgeNodes; d%m_symmY(nidx) = planeInc + j; d%m_symmZ(nidx) = rowInc + j
        nidx = nidx + 1
      END DO
    END DO
    d%m_symm_is_set = .TRUE.

    ! face connectivity (lxip FREE face clamped to in-range, matching the numpy port)
    d%m_lxim(0) = 0
    DO i = 1, domElems - 1
      d%m_lxim(i) = i - 1; d%m_lxip(i - 1) = i
    END DO
    d%m_lxip(domElems - 1) = domElems - 1
    DO i = 0, edgeElems - 1
      d%m_letam(i) = i; d%m_letap(domElems - edgeElems + i) = domElems - edgeElems + i
    END DO
    DO i = edgeElems, domElems - 1
      d%m_letam(i) = i - edgeElems; d%m_letap(i - edgeElems) = i
    END DO
    DO i = 0, edgeElems*edgeElems - 1
      d%m_lzetam(i) = i; d%m_lzetap(domElems - edgeElems*edgeElems + i) = domElems - edgeElems*edgeElems + i
    END DO
    DO i = edgeElems*edgeElems, domElems - 1
      d%m_lzetam(i) = i - edgeElems*edgeElems; d%m_lzetap(i - edgeElems*edgeElems) = i
    END DO

    ! BCs
    d%m_elemBC = 0
    DO i = 0, edgeElems - 1
      planeInc = i*edgeElems*edgeElems; rowInc = i*edgeElems
      DO j = 0, edgeElems - 1
        d%m_elemBC(planeInc + j*edgeElems) = IOR(d%m_elemBC(planeInc + j*edgeElems), XI_M_SYMM)
 d%m_elemBC(planeInc + j*edgeElems + edgeElems - 1) = IOR(d%m_elemBC(planeInc + j*edgeElems + edgeElems - 1), XI_P_FREE)
        d%m_elemBC(planeInc + j) = IOR(d%m_elemBC(planeInc + j), ETA_M_SYMM)
          d%m_elemBC(planeInc+j+edgeElems*edgeElems-edgeElems)=IOR(d%m_elemBC(planeInc+j+edgeElems*edgeElems-edgeElems), ETA_P_FREE)
        d%m_elemBC(rowInc + j) = IOR(d%m_elemBC(rowInc + j), ZETA_M_SYMM)
   d%m_elemBC(rowInc+j+domElems-edgeElems*edgeElems)=IOR(d%m_elemBC(rowInc+j+domElems-edgeElems*edgeElems), ZETA_P_FREE)
      END DO
    END DO

    ! material params
    d%m_dtfixed = -1.0e-7_8; d%m_deltatime = 1.0e-7_8; d%m_deltatimemultlb = 1.1_8
    d%m_deltatimemultub = 1.2_8; d%m_stoptime = 1.0e-2_8; d%m_dtcourant = 1.0e+20_8
    d%m_dthydro = 1.0e+20_8; d%m_dtmax = 1.0e-2_8; d%m_time = 0.0_8; d%m_cycle = 0
    d%m_e_cut = 1.0e-7_8; d%m_p_cut = 1.0e-7_8; d%m_q_cut = 1.0e-7_8; d%m_u_cut = 1.0e-7_8; d%m_v_cut = 1.0e-10_8
    d%m_hgcoef = 3.0_8; d%m_ss4o3 = 4.0_8/3.0_8; d%m_qstop = 1.0e+12_8
    d%m_monoq_max_slope = 1.0_8; d%m_monoq_limiter_mult = 2.0_8
    d%m_qlc_monoq = 0.5_8; d%m_qqc_monoq = 2.0_8/3.0_8; d%m_qqc = 2.0_8
    d%m_pmin = 0.0_8; d%m_emin = -1.0e+15_8; d%m_dvovmax = 0.1_8
    d%m_eosvmax = 1.0e+9_8; d%m_eosvmin = 1.0e-9_8; d%m_refdens = 1.0_8

    DO k = 1, nsteps
      CALL TimeIncrement(d)
      CALL LagrangeLeapFrog(d)
    END DO

    DO i = 0, d%m_numElem - 1
      eo(i) = d%m_e(i); po(i) = d%m_p(i); qo(i) = d%m_q(i); vo(i) = d%m_v(i)
    END DO
    DO i = 0, d%m_numNode - 1
      xo(i) = d%m_x(i); yo(i) = d%m_y(i); zo(i) = d%m_z(i)
      xdo(i) = d%m_xd(i); ydo(i) = d%m_yd(i); zdo(i) = d%m_zd(i)
    END DO
  END SUBROUTINE

END MODULE
