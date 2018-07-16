

subroutine shoot1orderlocal(x0, alpha, y0, nu0, a, b, sig, ord, neighbors, num_neighbors, withjac, withnu, t0, &
        numx, dimx, numy, tot_neighbors, x, y, jac, nu)
    implicit none
    integer :: t0, numx, dimx, numy, ord, withjac, withnu, tot_neighbors
    real(8) :: sig
    real(8) :: x0(numx, dimx)
    real(8) :: x(t0+1, numx, dimx)
    real(8) :: alpha(t0, numx, dimx)
    real(8) :: y0(numy, dimx)
    real(8) :: y(t0+1, numy, dimx)
    real(8) :: a(t0,dimx, dimx)
    real(8) :: b(t0,dimx)
    integer(4) :: neighbors(tot_neighbors)
    integer(4) :: num_neighbors(dimx)
    real(8) :: jac(t0+1, numx)
    real(8) :: nu0(numx, dimx)
    real(8) :: nu(t0+1, numx, dimx)



    !f2py integer, intent(in) :: t0, numx, dimx, numy, ord, withJ, withnu, tot_neighbors
    !f2py real(8), intent(in) :: sig
    !f2py real(8), intent(in), dimension(numx, dimx) :: x0
    !f2py real(8), intent(in), dimension(numy, dimx) :: y0
    !f2py real(8), intent(in), dimension(numx, dimx) :: nu0
    !f2py real(8), intent(in), dimension(t0, numx, dimx) :: alpha
    !f2py real(8), intent(in), dimension(t0, dimx, dimx) :: a
    !f2py real(8), intent(in), dimension(t0, dimx) :: b
    !f2py integer(4), intent(in), dimension(tot_neighbors) :: neighbors
    !f2py integer(4), intent(in), dimension(dimx) :: num_neighbors
    !f2py real(8), intent(out), dimension(t0+1, numx, dimx) :: x
    !f2py real(8), intent(out), dimension(t0+1, numy, dimx) :: y
    !f2py real(8), intent(out), dimension(t0+1, numx) :: jac
    !f2py real(8), intent(out), dimension(t0+1, numx, dimx) :: nu

    !  interface
    !    function matmult(dima, dimx1, dimx2, a,x) result(b)
    !     integer dima, dimx1, dimx2
    !     real(8), dimension(dima,dimx1), intent(in) :: a
    !     real(8), dimension(dimx1,dimx2), intent(in) :: x
    !     real(8), dimension(dima,dimx2) :: b
    !    end function
    !  end interface
    real(8) :: dotproduct
    real(8) :: ut, Kv, Kv_diff, lpt, ppp
    integer :: t, k, l, ip, jp, j, ii, i0, jj, kk, nSets
    integer :: startSet(dimx), endSet(dimx)
    real(8) :: dx(dimx),dy(dimx),dnu(dimx),djac, sqdist(dimx), sqdiff(dimx)
    real(8) :: c_(5, 5), c1_(4, 4)
    real(8) :: dt
    real(8) :: xt(numx, dimx)
    real(8) :: alphat(numx, dimx)
    real(8) :: yt(numy, dimx)
    real(8) :: at(dimx, dimx)
    real(8) :: bt(dimx)
    real(8) :: nut(numx, dimx)

    nSets = 1
    startSet(1) = 1
    do k=1,tot_neighbors,1
        if (neighbors(k) == 0) then
            endSet(nSets) = k-1
            if (k < tot_neighbors) then
                nSets = nSets+1
                startSet(nSets) = k+1
            end if
        end if
    end do
    !Print *, "shoot nSets = ", nSets
    !do kk=1, nSets, 1
     !   Print *, dimx, kk, startSet(kk), endSet(kk)
    !end do

    dt = 1./t0
    x(1,:,:) = x0
    if (numy > 0) then
        y(1,:,:) = y0
    end if
    if (withjac > 0) then
        jac(1,:) = 0
    end if
    if (withnu > 0) then
        nu(1,:,:) = nu0
    end if
    c_= reshape((/1.,1.,1.,1.,1.,    0.,1.,1.,1.,1.,    0.,0., 1./3, 0.4, 3./7,  &
            0.,0.,0., 1./15, 2./21,   0.,0.,0.,0., 1./105 /), (/5,5/))
    c1_ = reshape((/1., 1./3, 1./5, 1./7,    0., 1./3, 1./5, 1./7,    0.,0., 1./15, 2./35,&
            0.,0.,0., 1./105 /), (/4,4/))

    do t = 1, t0, 1
        xt = x(t,:,:)
        at = a(t,:,:)
        alphat = alpha(t,:,:)
        bt = b(t,:)
        nut = nu(t,:,:)
        yt = y(t,:,:)
        !$omp parallel do private(k,l,ut,lpt,Kv, Kv_diff, ppp, ip, jp, ii, j,jj, &
        !$omp& dx, dnu, djac, sqdist, sqdiff) shared (alphat,at,bt,xt,nut,x,nu,jac,sig,ord,c_, &
        !$omp c1_,numy,numx,withnu,withjac,num_neighbors,neighbors, startSet, endSet, nSets)
        do k = 1, numx, 1
            dx=0
            dnu=0
            djac = 0
            do l=1, numx, 1
                do kk =1, nSets, 1
                    ut = 0
                    do jj=startSet(kk),endSet(kk),1
                        ut = ut + (xt(k,neighbors(jj))-xt(l,neighbors(jj)))**2
                    end do
                    ut = sqrt(ut)/sig
                    if (ord > 4) then
                        ut = ut * ut
                        if (ut < 1e-8) then
                            Kv = 1.0
                        else
                            Kv = exp(-0.5*ut)
                        end if
                    else
                        if (ut < 1e-8) then
                            Kv = 1.0
                        else
                            lpt = c_(ord+1, 1) + c_(ord+1,2)*ut + c_(ord+1,3)*ut**2 + c_(ord+1,4)*ut**3 + c_(ord+1,5)*ut**4
                            Kv = lpt * exp(-1.0*ut)
                        end if
                    end if

                    if (withnu > 0 .or. withjac > 0) then
                        if (ord > 4) then
                            if (ut < 1e-8) then
                                Kv_diff = - 1.0/(2*sig**2)
                            else
                                Kv_diff = (-1.0)* exp(-0.5*ut)/(2*sig**2)
                            end if
                        else
                            if (ut < 1e-8) then
                                Kv_diff = - c1_(ord,1)/(2*sig*sig)
                            else
                                lpt = c1_(ord,1) + c1_(ord,2)*ut + c1_(ord,3)*ut**2 + c1_(ord,4)*ut**3
                                Kv_diff = -lpt * exp(-1.0*ut)/(2*sig**2)
                            end if
                        end if
                    end if
                    sqdist(kk) = Kv
                    sqdiff(kk) = Kv_diff
                end do
                do j = 1, dimx, 1
                    kk = num_neighbors(j)
                    dx(j) = dx(j) + sqdist(kk) * alphat(l,j)
                    if (withjac > 0) then
                        do jj =startSet(kk), endSet(kk), 1
                            ii = neighbors(jj)
                            djac = djac + 2* sqdiff(kk) * (xt(k,ii) - xt(l,ii)*alphat(l,j))
                        end do
                    end if
                    if (withnu > 0) then
                        do jj =startSet(kk), endSet(kk), 1
                            ii = neighbors(jj)
                            dnu(ii) = dnu(ii) - 2 * sqdiff(kk) * nut(j,l)*alphat(l,j) * (xt(k,ii) - xt(l,ii))
                        end do !jj
                    end if
                end do !j
            end do !l
            do ip = 1,dimx,1
                x(t+1, k, ip) = dt*(dx(ip)+bt(ip))
                do jp=1,dimx,1
                    x(t+1, k, ip) = x(t+1, k, ip) + at(ip,jp) * xt(k, jp)
                end do
            end do
            !        x(t+1, k,:) = matmult(dimx, dimx, 1, at,xt(k,:)) + dt*(dx+bt)
            if (withnu > 0) then
                do ip = 1,dimx,1
                    nu(t+1, k, ip) = 0
                    do jp=1,dimx,1
                        nu(t+1, k, ip) = nu(t+1, k, ip) + nut(k,jp) * at(jp,ip)
                    end do
                end do
                !          nu(t+1,k,:) = matmult(1, dimx, dimx, nut(k,:),at) + dt * dnu
            end if
            if (withjac > 0) then
                jac(t+1,k) = jac(t,k) + dt * djac
            end if
        end do !k
        !$omp end parallel do

        !$omp parallel do private(k,l,ut,lpt,Kv,dy,j,jj,ii,sqdist) &
        !$omp& shared (alphat,at,bt,xt,yt,y,sig,ord,c_,c1_,numy,numx,num_neighbors,neighbors, startSet, endSet, nSets)
        do k = 1, numy, 1
            dy=0
            do l=1, numx, 1
                do kk =1, nSets, 1
                    ut = 0
                    do jj=startSet(kk),endSet(kk),1
                        ut = ut + (yt(k,neighbors(jj))-xt(l,neighbors(jj)))**2
                    end do
                    ut = sqrt(ut)/sig
                    if (ord > 4) then
                        ut = ut * ut
                        if (ut < 1e-8) then
                            Kv = 1.0
                        else
                            Kv = exp(-0.5*ut)
                        end if
                    else
                        if (ut < 1e-8) then
                            Kv = 1.0
                        else
                            lpt = c_(ord+1, 1) + c_(ord+1,2)*ut + c_(ord+1,3)*ut**2 + c_(ord+1,4)*ut**3 + c_(ord+1,5)*ut**4
                            Kv = lpt * exp(-1.0*ut)
                        end if
                    end if
                    sqdist(kk) = Kv
                end do
                do j = 1, dimx, 1
                    dy(j) = dy(j) + Kv*alphat(l, j)
                end do !j
            end do !l
            do ip = 1,dimx,1
                ppp = 0
                do jp=1,dimx,1
                    ppp = ppp + at(ip,jp)*yt(k,jp)
                end do
                y(t+1,k,ip) = ppp + dt * (dy(ip) + bt(ip))
            end do
            !        y(t+1, k,:) = matmul(at, yt(k,:)) + dt*(dy + bt)
        end do !k
        !$omp end parallel do
    end do !t
end subroutine shoot1orderlocal



subroutine adjoint1orderlocal(xt, alpha, px1, a, sig, ord, neighbors, num_neighbors, &
        regweight, t0, numx, dimx, tot_neighbors, px)
    implicit none
    integer :: t0, numx, dimx, ord, tot_neighbors
    real(8) :: sig, regweight
    real(8) :: px1(numx, dimx)
    real(8) :: xt(t0+1, numx, dimx)
    real(8) :: alpha(t0, numx, dimx)
    real(8) :: a(t0, dimx, dimx)
    integer :: neighbors(tot_neighbors)
    integer :: num_neighbors(dimx)
    real(8) :: px(t0+1, numx, dimx)

    !f2py integer, intent(in) :: t0, numx, dimx, ord
    !f2py real(8), intent(in) :: sig, regweight
    !f2py real(8), intent(in), dimension(t0+1,numx, dimx) :: xt
    !f2py real(8), intent(in), dimension(numx, dimx) :: px1
    !f2py real(8), intent(in), dimension(t0, numx, dimx) :: alpha
    !f2py real(8), intent(in), dimension(t0, dimx, dimx) :: a
    !f2py integer, intent(in), dimension(tot_neighbors) :: neighbors
    !f2py integer, intent(in), dimension(dimx) :: num_neighbors
    !f2py real(8), intent(out), dimension(t0+1, numx, dimx) :: px

    real(8) :: dotproduct
    real(8) :: ut, Kv_diff, lpt, ppp
    integer :: t, k, l, ip, jp, ii, jj, kk, j, nSets
    integer :: startSet(dimx), endSet(dimx)
    real(8) :: dpx(dimx), sqdist(dimx)
    real(8) :: c_(5, 5), c1_(4, 4)
    real(8) :: dt
    real(8) :: xtt(numx, dimx)
    real(8) :: alphat(numx, dimx)
    real(8) :: at(dimx, dimx)
    real(8) :: pxt(numx, dimx)

    nSets = 1
    startSet(1) = 1
    do k=1,tot_neighbors,1
        if (neighbors(k) == 0) then
            endSet(nSets) = k-1
            if (k < tot_neighbors) then
                nSets = nSets+1
                startSet(nSets) = k+1
            end if
        end if
    end do
    !Print *, "adj nSets = ", nSets

    dt = 1./t0
    px(t0+1,:,:) = px1
    c_= reshape((/1.,1.,1.,1.,1.,    0.,1.,1.,1.,1.,    0.,0., 1./3, 0.4, 3./7,  &
            0.,0.,0., 1./15, 2./21,   0.,0.,0.,0., 1./105 /), (/5,5/))
    c1_ = reshape((/1., 1./3, 1./5, 1./7,    0., 1./3, 1./5, 1./7,    0.,0., 1./15, 2./35,&
            0.,0.,0., 1./105 /), (/4,4/))

    do t = t0, 1, -1
        alphat = alpha(t,:,:)
        xtt = xt(t,:,:)
        pxt = px(t+1,:,:)
        at = a(t,:,:)
        !$omp parallel do private(k,l,ut,lpt,Kv_diff,ppp,ip,jp,ii,jj,j, &
        !$omp& dpx,sqdist) shared (alphat,xtt,px,pxt,regweight,at,sig,ord,c_,c1_,neighbors,num_neighbors, startSet, endSet, nSets)
        do k = 1, numx, 1
            dpx=0
            do l=1, numx, 1
                do kk =1, nSets, 1
                    ut = 0
                    do jj=startSet(kk),endSet(kk),1
                        ut = ut + (xtt(k,neighbors(jj))-xtt(l,neighbors(jj)))**2
                    end do
                    ut = sqrt(ut)/sig
                    if (ord > 4) then
                        ut = ut * ut
                        if (ut < 1e-8) then
                            Kv_diff = - 1.0/(2*sig**2)
                        else
                            Kv_diff = (-1.0)* exp(-0.5*ut)/(2*sig**2)
                        end if
                    else
                        if (ut < 1e-8) then
                            Kv_diff = - c1_(ord,1)/(2*sig*sig)
                        else
                            lpt = c1_(ord,1) + c1_(ord,2)*ut + c1_(ord,3)*ut**2 + c1_(ord,4)*ut**3
                            Kv_diff = -lpt * exp(-1.0*ut)/(2*sig**2)
                        end if
                    end if
                    sqdist(kk) = 2*Kv_diff
                end do
                do j = 1, dimx, 1
                    ut = sqdist(num_neighbors(j)) * (pxt(k,j)*alphat(l, j) + pxt(l, j)*alphat(k, j)&
                                - 2*regweight*alphat(k,j)*alphat(l, j))
                    do ii = startSet(j),endSet(j),1
                        jj = neighbors(ii)
                        dpx(jj) = dpx(jj) + ut * (xtt(k,jj) - xtt(l,jj))
                    end do !jj
                end do !j
            end do !l
            do ip = 1,dimx,1
                ppp = 0
                do jp=1,dimx,1
                    ppp = ppp + at(jp,ip)*pxt(k,jp)
                end do
                px(t,k,ip) = ppp + dt * dpx(ip)
            end do
            !        px(t, k,:) = matmul(transpose(at), pxt(k, :)) + dt* dpx
        end do !k
        !$omp end parallel do
    end do !t
end subroutine adjoint1orderlocal



subroutine shoot1order(x0, alpha, y0, nu0, a, b, sig, ord, withjac, withnu, t0, numx, dimx,  numy, x, y, jac, nu)
    implicit none
    integer :: t0, numx, dimx, numy, ord, withjac, withnu
    real(8) :: sig
    real(8) :: x0(numx, dimx)
    real(8) :: x(t0+1, numx, dimx)
    real(8) :: alpha(t0, numx, dimx)
    real(8) :: y0(numy, dimx)
    real(8) :: y(t0+1, numy, dimx)
    real(8) :: a(t0,dimx, dimx)
    real(8) :: b(t0,dimx)
    real(8) :: jac(t0+1, numx)
    real(8) :: nu0(numx, dimx)
    real(8) :: nu(t0+1, numx, dimx)



    !f2py integer, intent(in) :: t0, numx, dimx, numy, ord, withJ, withnu
    !f2py real(8), intent(in) :: sig
    !f2py real(8), intent(in), dimension(numx, dimx) :: x0
    !f2py real(8), intent(in), dimension(numy, dimx) :: y0
    !f2py real(8), intent(in), dimension(numx, dimx) :: nu0
    !f2py real(8), intent(in), dimension(t0, numx, dimx) :: alpha
    !f2py real(8), intent(in), dimension(t0, dimx, dimx) :: a
    !f2py real(8), intent(in), dimension(t0, dimx) :: b
    !f2py real(8), intent(out), dimension(t0+1, numx, dimx) :: x
    !f2py real(8), intent(out), dimension(t0+1, numy, dimx) :: y
    !f2py real(8), intent(out), dimension(t0+1, numx) :: jac
    !f2py real(8), intent(out), dimension(t0+1, numx, dimx) :: nu

    !  interface
    !    function matmult(dima, dimx1, dimx2, a,x) result(b)
    !     integer dima, dimx1, dimx2
    !     real(8), dimension(dima,dimx1), intent(in) :: a
    !     real(8), dimension(dimx1,dimx2), intent(in) :: x
    !     real(8), dimension(dima,dimx2) :: b
    !    end function
    !  end interface
    real(8) :: dotproduct
    real(8) :: ut, Kv, Kv_diff, lpt, ppp
    integer :: t, k, l, ip, jp, ii, jj, i, j, i0
    real(8) :: dx(dimx),dy(dimx),dnu(dimx),djac
    real(8) :: c_(5, 5), c1_(4, 4)
    real(8) :: dt
    real(8) :: xt(numx, dimx)
    real(8) :: alphat(numx, dimx)
    real(8) :: yt(numy, dimx)
    real(8) :: at(dimx, dimx)
    real(8) :: bt(dimx)
    real(8) :: nut(numx, dimx)


    dt = 1./t0
    x(1,:,:) = x0
    if (numy > 0) then
        y(1,:,:) = y0
    end if
    if (withjac > 0) then
        jac(1,:) = 0
    end if
    if (withnu > 0) then
        nu(1,:,:) = nu0
    end if
    c_= reshape((/1.,1.,1.,1.,1.,    0.,1.,1.,1.,1.,    0.,0., 1./3, 0.4, 3./7,  &
            0.,0.,0., 1./15, 2./21,   0.,0.,0.,0., 1./105 /), (/5,5/))
    c1_ = reshape((/1., 1./3, 1./5, 1./7,    0., 1./3, 1./5, 1./7,    0.,0., 1./15, 2./35,&
            0.,0.,0., 1./105 /), (/4,4/))

    do t = 1, t0, 1
        xt = x(t,:,:)
        at = a(t,:,:)
        alphat = alpha(t,:,:)
        bt = b(t,:)
        nut = nu(t,:,:)
        yt = y(t,:,:)
        !$omp parallel do private(k,l,ut,lpt,Kv, Kv_diff, ppp, ip, jp, &
        !$omp& dx, dy, dnu, djac) shared (alphat,at,bt,xt,nut,x,nu,jac,sig,ord,c_, &
        !$omp c1_,numy,numx,withnu,withjac)
        do k = 1, numx, 1
            dx=0
            dnu=0
            djac = 0
            do l=1, numx, 1
                ut = sqrt(sum((xt(k,:) - xt(l,:))**2)) / sig
                if (ord > 4) then
                    ut = ut * ut
                    if (ut < 1e-8) then
                        Kv = 1.0
                    else
                        Kv = exp(-0.5*ut)
                    end if
                else
                    if (ut < 1e-8) then
                        Kv = 1.0
                    else
                        lpt = c_(ord+1, 1) + c_(ord+1,2)*ut + c_(ord+1,3)*ut**2 + c_(ord+1,4)*ut**3 + c_(ord+1,5)*ut**4
                        Kv = lpt * exp(-1.0*ut)
                    end if
                end if

                if (withnu > 0 .or. withjac > 0) then
                    if (ord > 4) then
                        ut = ut * ut
                        if (ut < 1e-8) then
                            Kv_diff = - 1.0/(2*sig**2)
                        else
                            Kv_diff = (-1.0)* exp(-0.5*ut)/(2*sig**2)
                        end if
                    else
                        if (ut < 1e-8) then
                            Kv_diff = - c1_(ord,1)/(2*sig*sig)
                        else
                            lpt = c1_(ord,1) + c1_(ord,2)*ut + c1_(ord,3)*ut**2 + c1_(ord,4)*ut**3
                            Kv_diff = -lpt * exp(-1.0*ut)/(2*sig**2)
                        end if
                    end if
                end if
                dx = dx + Kv*alphat(l, :)
                if (withjac > 0) then
                    djac = djac + 2* Kv_diff * sum((xt(k,:) - xt(l,:))*alphat(l,:))
                end if
                if (withnu > 0) then
                    dnu = dnu - 2 * Kv_diff * sum(nut(:,l)*alphat(l,:)) * (xt(k,:) - xt(l,:))
                end if
            end do !l
            do ip = 1,dimx,1
                x(t+1, k, ip) = dt*(dx(ip)+bt(ip))
                do jp=1,dimx,1
                    x(t+1, k, ip) = x(t+1, k, ip) + at(ip,jp) * xt(k, jp)
                end do
            end do
            !        x(t+1, k,:) = matmult(dimx, dimx, 1, at,xt(k,:)) + dt*(dx+bt)
            if (withnu > 0) then
                do ip = 1,dimx,1
                    nu(t+1, k, ip) = 0
                    do jp=1,dimx,1
                        nu(t+1, k, ip) = nu(t+1, k, ip) + nut(k,jp) * at(jp,ip)
                    end do
                end do
                !          nu(t+1,k,:) = matmult(1, dimx, dimx, nut(k,:),at) + dt * dnu
            end if
            if (withjac > 0) then
                jac(t+1,k) = jac(t,k) + dt * djac
            end if
        end do !k
        !$omp end parallel do

        !$omp parallel do private(k,l,ut,lpt,Kv,dy) &
        !$omp& shared (alphat,at,bt,xt,yt,y,sig,ord,c_,c1_)
        do k = 1, numy, 1
            dy=0
            do l=1, numx, 1
                ut = sqrt(sum((yt(k,:)-xt(l,:))**2)) / sig
                if (ord > 4) then
                    ut = ut * ut
                    if (ut < 1e-8) then
                        Kv = 1.0
                    else
                        Kv = exp(-0.5*ut)
                    end if
                else
                    if (ut < 1e-8) then
                        Kv = 1.0
                    else
                        lpt = c_(ord+1, 1) + c_(ord+1,2)*ut + c_(ord+1,3)*ut**2 + c_(ord+1,4)*ut**3 + c_(ord+1,5)*ut**4
                        Kv = lpt * exp(-1.0*ut)
                    end if
                end if
                dy = dy + Kv*alphat(l, :)
            end do !l
            do ip = 1,dimx,1
                ppp = 0
                do jp=1,dimx,1
                    ppp = ppp + at(ip,jp)*yt(k,jp)
                end do
                y(t+1,k,ip) = ppp + dt * (dy(ip) + bt(ip))
            end do
            !        y(t+1, k,:) = matmul(at, yt(k,:)) + dt*(dy + bt)
        end do !k
        !$omp end parallel do
    end do !t
end subroutine shoot1order

subroutine adjoint1order(xt, alpha, px1, a, sig, ord, regweight, t0, numx, dimx, px)
    implicit none
    integer :: t0, numx, dimx, ord
    real(8) :: sig, regweight
    real(8) :: px1(numx, dimx)
    real(8) :: xt(t0+1, numx, dimx)
    real(8) :: alpha(t0, numx, dimx)
    real(8) :: a(t0, dimx, dimx)
    real(8) :: px(t0+1, numx, dimx)

    !f2py integer, intent(in) :: t0, numx, dimx, ord
    !f2py real(8), intent(in) :: sig, regweight
    !f2py real(8), intent(in), dimension(t0+1,numx, dimx) :: xt
    !f2py real(8), intent(in), dimension(numx, dimx) :: px1
    !f2py real(8), intent(in), dimension(t0, numx, dimx) :: alpha
    !f2py real(8), intent(in), dimension(t0, dimx, dimx) :: a
    !f2py real(8), intent(out), dimension(t0+1, numx, dimx) :: px

    real(8) :: dotproduct
    real(8) :: ut, Kv_diff, lpt, ppp
    integer :: t, k, l, ip, jp
    real(8) :: dpx(dimx)
    real(8) :: c_(5, 5), c1_(4, 4)
    real(8) :: dt
    real(8) :: xtt(numx, dimx)
    real(8) :: alphat(numx, dimx)
    real(8) :: at(dimx, dimx)
    real(8) :: pxt(numx, dimx)


    dt = 1./t0
    px(t0+1,:,:) = px1
    c_= reshape((/1.,1.,1.,1.,1.,    0.,1.,1.,1.,1.,    0.,0., 1./3, 0.4, 3./7,  &
            0.,0.,0., 1./15, 2./21,   0.,0.,0.,0., 1./105 /), (/5,5/))
    c1_ = reshape((/1., 1./3, 1./5, 1./7,    0., 1./3, 1./5, 1./7,    0.,0., 1./15, 2./35,&
            0.,0.,0., 1./105 /), (/4,4/))

    do t = t0, 1, -1
        alphat = alpha(t,:,:)
        xtt = xt(t,:,:)
        pxt = px(t+1,:,:)
        at = a(t,:,:)
        !$omp parallel do private(k,l,ut,lpt,Kv_diff,ppp,ip,jp, &
        !$omp& dpx) shared (alphat,xtt,px,pxt,regweight,at,sig,ord,c_,c1_)
        do k = 1, numx, 1
            dpx=0
            do l=1, numx, 1
                ut = sqrt(sum((xt(t,k,:) - xt(t,l,:))**2)) / sig
                if (ord > 4) then
                    ut = ut * ut
                    if (ut < 1e-8) then
                        Kv_diff = - 1.0/(2*sig**2)
                    else
                        Kv_diff = (-1.0)* exp(-0.5*ut)/(2*sig**2)
                    end if
                else
                    if (ut < 1e-8) then
                        Kv_diff = - c1_(ord,1)/(2*sig*sig)
                    else
                        lpt = c1_(ord,1) + c1_(ord,2)*ut + c1_(ord,3)*ut**2 + c1_(ord,4)*ut**3
                        Kv_diff = -lpt * exp(-1.0*ut)/(2*sig**2)
                    end if
                end if
                dpx = dpx + 2*Kv_diff * (sum(pxt(k,:)*alphat(l, :)) &
                        + sum(pxt(l, :)*alphat(k, :))&
                        - 2*regweight*sum(alphat(k,:)*alphat(l, :)))&
                        * (xtt(k,:) - xtt(l,:))
            end do !l
            do ip = 1,dimx,1
                ppp = 0
                do jp=1,dimx,1
                    ppp = ppp + at(jp,ip)*pxt(k,jp)
                end do
                px(t,k,ip) = ppp + dt * dpx(ip)
            end do
            !        px(t, k,:) = matmul(transpose(at), pxt(k, :)) + dt* dpx
        end do !k
        !$omp end parallel do
    end do !t
end subroutine adjoint1order


!function matmult(dima, dimx1, dimx2, a,x) result(b)
! implicit none
! integer dima, dimx1, dimx2
! real(8), dimension(dima,dimx1), intent(in) :: a
! real(8), dimension(dimx1,dimx2), intent(in) :: x
! real(8), dimension(dima,dimx2) :: b
!
! integer ip, jp
! do ip = 1,dima,1
!    b(ip,:) = 0
!    do jp=1,dimx1,1
!        b(ip,:) = b(ip,:) + a(ip,jp)*x(jp,:)
!    end do
! end do
!end function


function dotproduct(d,x,y) result (res)
    implicit none
    integer d
    real(8), dimension(d),intent(in) :: x
    real(8), dimension(d),intent(in) :: y
    real(8) res

    integer i
    res = sum(x*y)
    ! do i=1,d,1
    !  res = res + x(i)*y(i)
    ! end do
end function
