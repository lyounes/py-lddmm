subroutine kernelmatrix(x, y, sig, ord, num_nodes, num_nodes_y, num_sig, dim, f)
    implicit none
    integer :: num_nodes, num_nodes_y, dim, num_sig
    real(8) :: x(num_nodes, dim)
    real(8) :: y(num_nodes_y, dim)
    real(8) :: sig(num_sig)
    integer :: ord
    real(8) :: f(num_nodes, num_nodes_y)

    !f2py integer, intent(in) :: num_nodes, num_nodes_y, dim, num_sig
    !f2py real(8), intent(in), dimension(num_nodes, dim) :: x
    !f2py real(8), intent(in), dimension(num_nodes_y, dim) :: y
    !f2py real(8), intent(in), dimension(num_sig) :: sig
    !f2py integer, intent(in) :: ord
    !f2py real(8), intent(out), dimension(num_nodes, num_nodes_y) :: f

    real(8) :: ut,Kh,ut0
    real(8) :: lpt
    integer :: s,k,l
    real(8) :: df(num_nodes_y)
    real(8) :: c_(5, 5)
    c_= reshape((/1.,1.,1.,1.,1.,    0.,1.,1.,1.,1.,    0.,0., 1./3, 0.4, 3./7,  &
            0.,0.,0., 1./15, 2./21,   0.,0.,0.,0., 1./105 /), (/5,5/))

    !$omp parallel do private(k,l,s,ut,ut0,lpt,Kh,df) shared &
    !$omp& (num_nodes, num_nodes_y, num_sig, f, sig, ord, c_)
    do k = 1, num_nodes, 1
        do l = 1, num_nodes_y, 1
            ut0 = sqrt(sum((x(k,:) - y(l,:))**2))
            Kh = 0
            do s = 1, num_sig, 1
                ut = ut0 / sig(s)
                if (ord > 4) then
                    ut = ut * ut
                    if (ut < 1e-8) then
                        Kh = Kh + 1.0
                    else
                        Kh = Kh + exp(-0.5*ut)
                    end if
                else
                    if (ut < 1e-8) then
                        Kh = Kh + 1.0
                    else
                        lpt = c_(ord+1, 1) + c_(ord+1,2)*ut + c_(ord+1,3)*ut**2 + c_(ord+1,4)*ut**3 + c_(ord+1,5)*ut**4
                        Kh = Kh + lpt * exp(-1.0*ut)
                    end if
                end if
            end do !s
            Kh = Kh / num_sig
            df(l) = Kh
        end do
        f(k,:) = df
    end do
    !$omp end parallel do
end subroutine kernelmatrix

!subroutine applyk(x, y, beta, sig, ord, num_nodes, num_nodes_y, dim, dimb, f)
!    implicit none
!    integer :: num_nodes, num_nodes_y, dim, dimb
!    real(8) :: x(num_nodes, dim)
!    real(8) :: y(num_nodes_y, dim)
!    real(8) :: sig
!    integer :: ord
!    real(8) :: f(num_nodes, dimb)
!    real(8) :: beta(num_nodes_y, dimb)
!    real(8) :: Kh
!
!    !f2py integer, intent(in) :: num_nodes, num_nodes_y, dim, dimb
!    !f2py real(8), intent(in), dimension(num_nodes, dim) :: x
!    !f2py real(8), intent(in), dimension(num_nodes_y, dim) :: y
!    !f2py real(8), intent(in), dimension(num_nodes_y, dimb) :: beta
!    !f2py real(8), intent(in) :: sig
!    !f2py integer, intent(in) :: ord
!    !f2py real(8), intent(out), dimension(num_nodes, dimb) :: f
!
!    real(8) :: ut
!    real(8) :: lpt
!    integer :: k,l
!    real(8) :: df(dimb)
!    real(8) :: c_(5, 5)
!    c_= reshape((/1.,1.,1.,1.,1.,    0.,1.,1.,1.,1.,    0.,0., 1./3, 0.4, 3./7,  &
!            0.,0.,0., 1./15, 2./21,   0.,0.,0.,0., 1./105 /), (/5,5/))
!
!    !$omp parallel do private(k,l,ut,lpt,Kh,df) shared &
!    !$omp& (num_nodes, num_nodes_y, f, sig, ord, beta, c_)
!    do k = 1, num_nodes, 1
!        df = 0
!        do l = 1, num_nodes_y, 1
!            ut = sqrt(sum((x(k,:) - y(l,:))**2)) / sig
!            if (ord > 4) then
!                ut = ut * ut
!                if (ut < 1e-8) then
!                    Kh = 1.0
!                else
!                    Kh = exp(-0.5*ut)
!                end if
!            else
!                if (ut < 1e-8) then
!                    Kh = 1.0
!                else
!                    lpt = c_(ord+1, 1) + c_(ord+1,2)*ut + c_(ord+1,3)*ut**2 + c_(ord+1,4)*ut**3 + c_(ord+1,5)*ut**4
!                    Kh = lpt * exp(-1.0*ut)
!                end if
!            end if
!            df = df + Kh * beta(l,:)
!        end do
!        f(k,:) = df
!    end do
!    !$omp end parallel do
!end subroutine applyK

subroutine applyk(x, y, beta, sig, ord, num_nodes, num_nodes_y, num_sig, dim, dimb, f)
    implicit none
    integer :: num_nodes, num_nodes_y, dim, dimb, num_sig
    real(8) :: x(num_nodes, dim)
    real(8) :: y(num_nodes_y, dim)
    real(8) :: sig(num_sig)
    integer :: ord
    real(8) :: f(num_nodes, dimb)
    real(8) :: beta(num_nodes_y, dimb)
    real(8) :: Kh

    !f2py integer, intent(in) :: num_nodes, num_nodes_y, dim, dimb, num_sig
    !f2py real(8), intent(in), dimension(num_nodes, dim) :: x
    !f2py real(8), intent(in), dimension(num_nodes_y, dim) :: y
    !f2py real(8), intent(in), dimension(num_nodes_y, dimb) :: beta
    !f2py real(8), intent(in), dimension(num_sig) :: sig
    !f2py integer, intent(in) :: ord
    !f2py real(8), intent(out), dimension(num_nodes, dimb) :: f

    real(8) :: lpt, ut, ut0
    integer :: k,l,s
    real(8) :: df(dimb)
    real(8) :: c_(5, 5)
    c_= reshape((/1.,1.,1.,1.,1.,    0.,1.,1.,1.,1.,    0.,0., 1./3, 0.4, 3./7,  &
            0.,0.,0., 1./15, 2./21,   0.,0.,0.,0., 1./105 /), (/5,5/))

    !$omp parallel do private(k,l,ut,ut0,lpt,Kh,df) shared &
    !$omp& (num_nodes, num_nodes_y, num_sig, f, sig, ord, beta, c_)
    do k = 1, num_nodes, 1
        df = 0
        do l = 1, num_nodes_y, 1
            ut0 = sqrt(sum((x(k,:) - y(l,:))**2))
            Kh = 0
            do s = 1, num_sig, 1
                ut = ut0 / sig(s)
                if (ord > 4) then
                    ut = ut * ut
                    if (ut < 1e-8) then
                        Kh = Kh + 1.0
                    else
                        Kh = Kh + exp(-0.5*ut)
                    end if
                else
                    if (ut < 1e-8) then
                        Kh = Kh + 1.0
                    else
                        lpt = c_(ord+1, 1) + c_(ord+1,2)*ut + c_(ord+1,3)*ut**2 + c_(ord+1,4)*ut**3 + c_(ord+1,5)*ut**4
                        Kh = Kh + lpt * exp(-1.0*ut)
                    end if
                end if
            end do
            Kh = Kh / num_sig
            df = df + Kh * beta(l,:)
        end do
        f(k,:) = df
    end do
    !$omp end parallel do
end subroutine applyk

!subroutine applylocalk(x, y, beta, sig, ord, neighbors, num_neighbors, num_nodes, num_nodes_y, dim, tot_neighbors, f)
!    implicit none
!    integer :: num_nodes, num_nodes_y, dim, tot_neighbors
!    real(8) :: x(num_nodes, dim)
!    real(8) :: y(num_nodes_y, dim)
!    real(8) :: sig
!    integer :: ord
!    real(8) :: f(num_nodes, dim)
!    real(8) :: beta(num_nodes_y, dim)
!    integer :: neighbors(tot_neighbors)
!    integer :: num_neighbors(dim)
!
!    !f2py integer, intent(in) :: num_nodes, num_nodes_y, dim, tot_neighbors
!    !f2py real(8), intent(in), dimension(num_nodes, dim) :: x
!    !f2py real(8), intent(in), dimension(num_nodes_y, dim) :: y
!    !f2py real(8), intent(in), dimension(num_nodes_y, dim) :: beta
!    !f2py real(8), intent(in) :: sig
!    !f2py integer, intent(in) :: ord
!    !f2py integer, intent(in), dimension(tot_neighbors) :: neighbors
!    !f2py integer, intent(in), dimension(dim) :: num_neighbors
!    !f2py real(8), intent(out), dimension(num_nodes, dim) :: f
!
!    real(8) :: Kv, ut, lpt
!    integer :: k,l, j, jj, kk, nSets
!    integer :: startSet(dim), endSet(dim)
!    real(8) :: df(dim), sqdist(dim)
!    real(8) :: c_(5, 5)
!    c_= reshape((/1.,1.,1.,1.,1.,    0.,1.,1.,1.,1.,    0.,0., 1./3, 0.4, 3./7,  &
!            0.,0.,0., 1./15, 2./21,   0.,0.,0.,0., 1./105 /), (/5,5/))
!
!    nSets = 1
!    startSet(1) = 1
!    do k=1,tot_neighbors,1
!        if (neighbors(k) == 0) then
!            endSet(nSets) = k-1
!            if (k < tot_neighbors) then
!                nSets = nSets+1
!                startSet(nSets) = k+1
!            end if
!        end if
!    end do
!    !Print *, "apply nSets = ", nSets
!
!
!    !$omp parallel do private(k,l,jj,kk,ut,lpt,Kv,df,sqdist) shared &
!    !$omp& (num_nodes, num_nodes_y, f, sig, ord, beta, c_,num_neighbors,neighbors, startSet, endSet, nSets)
!    do k = 1, num_nodes, 1
!        df = 0
!        do l = 1, num_nodes_y, 1
!            do kk =1, nSets, 1
!                ut = 0
!                do jj=startSet(kk),endSet(kk),1
!                    ut = ut + (x(k,neighbors(jj))-y(l,neighbors(jj)))**2
!                end do
!                ut = sqrt(ut)/sig
!                if (ord > 4) then
!                    ut = ut * ut
!                    if (ut < 1e-8) then
!                        Kv = 1.0
!                    else
!                        Kv = exp(-0.5*ut)
!                    end if
!                else
!                    if (ut < 1e-8) then
!                        Kv = 1.0
!                    else
!                        lpt = c_(ord+1, 1) + c_(ord+1,2)*ut + c_(ord+1,3)*ut**2 + c_(ord+1,4)*ut**3 + c_(ord+1,5)*ut**4
!                        Kv = lpt * exp(-1.0*ut)
!                    end if
!                end if
!                sqdist(kk) = Kv
!            end do
!
!            do j = 1, dim, 1
!                df(j) = df(j) + sqdist(num_neighbors(j)) * beta(l,j)
!            end do
!        end do
!        f(k,:) = df
!    end do
!    !$omp end parallel do
!end subroutine applylocalk

subroutine applylocalk(x, y, beta, sig, ord, neighbors, num_neighbors, num_nodes, num_nodes_y, &
                            num_sig, dim, tot_neighbors, f)
    implicit none
    integer :: num_nodes, num_nodes_y, dim, tot_neighbors, num_sig
    real(8) :: x(num_nodes, dim)
    real(8) :: y(num_nodes_y, dim)
    real(8) :: sig(num_sig)
    integer :: ord
    real(8) :: f(num_nodes, dim)
    real(8) :: beta(num_nodes_y, dim)
    integer :: neighbors(tot_neighbors)
    integer :: num_neighbors(dim)

    !f2py integer, intent(in) :: num_nodes, num_nodes_y, dim, tot_neighbors, num_sig
    !f2py real(8), intent(in), dimension(num_nodes, dim) :: x
    !f2py real(8), intent(in), dimension(num_nodes_y, dim) :: y
    !f2py real(8), intent(in), dimension(num_nodes_y, dim) :: beta
    !f2py real(8), intent(in), dimension(num_sig) :: sig
    !f2py integer, intent(in) :: ord
    !f2py integer, intent(in), dimension(tot_neighbors) :: neighbors
    !f2py integer, intent(in), dimension(dim) :: num_neighbors
    !f2py real(8), intent(out), dimension(num_nodes, dim) :: f

    real(8) :: Kv, ut, lpt, ut0
    integer :: k,l, s, j, jj, kk, nSets
    integer :: startSet(dim), endSet(dim)
    real(8) :: df(dim), sqdist(dim)
    real(8) :: c_(5, 5)
    c_= reshape((/1.,1.,1.,1.,1.,    0.,1.,1.,1.,1.,    0.,0., 1./3, 0.4, 3./7,  &
            0.,0.,0., 1./15, 2./21,   0.,0.,0.,0., 1./105 /), (/5,5/))

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
    !Print *, "apply nSets = ", nSets


    !$omp parallel do private(k,l,s,jj,kk,ut,ut0,lpt,Kv,df,sqdist) shared &
    !$omp& (num_nodes, num_nodes_y, num_sig, f, sig, ord, beta, c_,num_neighbors,neighbors,startSet, endSet, nSets)
    do k = 1, num_nodes, 1
        df = 0
        do l = 1, num_nodes_y, 1
            do kk =1, nSets, 1
                ut0 = 0
                do jj=startSet(kk),endSet(kk),1
                    ut0 = ut0 + (x(k,neighbors(jj))-y(l,neighbors(jj)))**2
                end do
                ut0 = sqrt(ut0)
                Kv = 0
                do s = 1, num_sig, 1
                    ut = ut0/sig(s)
                    if (ord > 4) then
                        ut = ut * ut
                        if (ut < 1e-8) then
                            Kv = Kv + 1.0
                        else
                            Kv = Kv + exp(-0.5*ut)
                        end if
                    else
                        if (ut < 1e-8) then
                            Kv = Kv + 1.0
                        else
                            lpt = c_(ord+1, 1) + c_(ord+1,2)*ut + c_(ord+1,3)*ut**2 &
                                  + c_(ord+1,4)*ut**3 + c_(ord+1,5)*ut**4
                            Kv = Kv + lpt * exp(-1.0*ut)
                        end if
                    end if
                end do
                sqdist(kk) = Kv/num_sig
            end do

            do j = 1, dim, 1
                df(j) = df(j) + sqdist(num_neighbors(j)) * beta(l,j)
            end do
        end do
        f(k,:) = df
    end do
    !$omp end parallel do
end subroutine applylocalk

!subroutine applylocalkdifft(x, y, a1, a2, sig, ord, neighbors, num_neighbors, num_nodes, num_nodes_y, dim, &
!                            tot_neighbors, na, f)
!    implicit none
!    integer :: num_nodes, num_nodes_y, dim, tot_neighbors, na
!    real(8) :: x(num_nodes, dim)
!    real(8) :: y(num_nodes_y, dim)
!    real(8) :: sig
!    integer :: ord
!    real(8) :: f(num_nodes, dim)
!    real(8) :: a1(na, num_nodes, dim)
!    real(8) :: a2(na, num_nodes_y, dim)
!    integer :: neighbors(tot_neighbors), num_neighbors(dim)
!
!    real(8) :: Kv_diff
!
!    !f2py integer, intent(in) :: num_nodes, num_nodes_y, dim, na
!    !f2py real(8), intent(in), dimension(num_nodes, dim) :: x
!    !f2py real(8), intent(in), dimension(num_nodes_y, dim) :: y
!    !f2py real(8), intent(in), dimension(na,num_nodes, dimb) :: a1
!    !f2py real(8), intent(in), dimension(na,num_nodes_y, dimb) :: a2
!    !f2py real(8), intent(in) :: sig
!    !f2py integer, intent(in) :: ord
!    !f2py integer, intent(in), dimension(tot_neighbors) :: neighbors
!    !f2py integer, intent(in), dimension(dim) :: num_neighbors
!    !f2py real(8), intent(out), dimension(num_nodes, dim) :: f
!
!    real(8) :: ut
!    real(8) :: lpt
!    integer :: k,l, j, jj, ii, kk, nSets
!    integer :: startSet(dim), endSet(dim)
!    real(8) :: df(dim), sqdist(dim)
!    real(8) :: c_(5, 5), c1_(4, 4)
!    c_= reshape((/1.,1.,1.,1.,1.,    0.,1.,1.,1.,1.,    0.,0., 1./3, 0.4, 3./7,  &
!            0.,0.,0., 1./15, 2./21,   0.,0.,0.,0., 1./105 /), (/5,5/))
!    c1_ = reshape((/1., 1./3, 1./5, 1./7,    0., 1./3, 1./5, 1./7,    0.,0., 1./15, 2./35,&
!            0.,0.,0., 1./105 /), (/4,4/))
!
!    nSets = 1
!    startSet(1) = 1
!    do k=1,tot_neighbors,1
!        if (neighbors(k) == 0) then
!            endSet(nSets) = k-1
!            if (k < tot_neighbors) then
!                nSets = nSets+1
!                startSet(nSets) = k+1
!            end if
!        end if
!    end do
!    !Print *, "apply diff nSets = ", nSets
!
!
!
!    !$omp parallel do private(k,l,ut,lpt,Kv_diff,df,ii,jj,j,sqdist,kk) shared &
!    !$omp& (num_nodes, num_nodes_y, f, sig, ord, a1, a2, c_, c1_,num_neighbors,neighbors, startSet, endSet, nSets)
!    do k = 1, num_nodes, 1
!        df = 0
!        do l = 1, num_nodes_y, 1
!            do kk =1, nSets, 1
!                ut = 0
!                do jj=startSet(kk),endSet(kk),1
!                    ut = ut + (x(k,neighbors(jj))-y(l,neighbors(jj)))**2
!                end do
!                ut = sqrt(ut)/sig
!                if (ord > 4) then
!                    if (ut < 1e-8) then
!                        Kv_diff = - 1.0/(2*sig**2)
!                    else
!                        Kv_diff = (-1.0)* exp(-0.5*ut)/(2*sig**2)
!                    end if
!                else
!                    if (ut < 1e-8) then
!                        Kv_diff = - c1_(ord,1)/(2*sig*sig)
!                    else
!                        lpt = c1_(ord,1) + c1_(ord,2)*ut + c1_(ord,3)*ut**2 + c1_(ord,4)*ut**3
!                        Kv_diff = -lpt * exp(-1.0*ut)/(2*sig**2)
!                    end if
!                end if
!                sqdist(kk) = Kv_diff
!            end do
!            do j = 1, dim, 1
!                kk = num_neighbors(j)
!                do jj =startSet(kk), endSet(kk), 1
!                    ii = neighbors(jj)
!                    df(ii) = df(ii) + sqdist(kk) * 2*(x(k,ii)-y(l,ii))* sum(a1(:,k,j)*a2(:,l,j))
!                end do
!            end do
!        end do
!        f(k, :) = df
!    end do
!    !$omp end parallel do
!end subroutine applylocalkdifft

subroutine applylocalkdifft(x, y, a1, a2, sig, ord, neighbors, num_neighbors, num_nodes, num_nodes_y, num_sig, &
                                  dim, tot_neighbors, na, f)
    implicit none
    integer :: num_nodes, num_nodes_y, dim, tot_neighbors, na, num_sig
    real(8) :: x(num_nodes, dim)
    real(8) :: y(num_nodes_y, dim)
    real(8) :: sig(num_sig)
    integer :: ord
    real(8) :: f(num_nodes, dim)
    real(8) :: a1(na, num_nodes, dim)
    real(8) :: a2(na, num_nodes_y, dim)
    integer :: neighbors(tot_neighbors), num_neighbors(dim)

    real(8) :: Kv_diff

    !f2py integer, intent(in) :: num_nodes, num_nodes_y, dim, na, num_sig
    !f2py real(8), intent(in), dimension(num_nodes, dim) :: x
    !f2py real(8), intent(in), dimension(num_nodes_y, dim) :: y
    !f2py real(8), intent(in), dimension(na,num_nodes, dimb) :: a1
    !f2py real(8), intent(in), dimension(na,num_nodes_y, dimb) :: a2
    !f2py real(8), intent(in), dimension(num_sig) :: sig
    !f2py integer, intent(in) :: ord
    !f2py integer, intent(in), dimension(tot_neighbors) :: neighbors
    !f2py integer, intent(in), dimension(dim) :: num_neighbors
    !f2py real(8), intent(out), dimension(num_nodes, dim) :: f

    real(8) :: ut, ut0
    real(8) :: lpt
    integer :: k,l, s, j, jj, ii, kk, nSets
    integer :: startSet(dim), endSet(dim)
    real(8) :: df(dim), sqdist(dim)
    real(8) :: c_(5, 5), c1_(4, 4)
    c_= reshape((/1.,1.,1.,1.,1.,    0.,1.,1.,1.,1.,    0.,0., 1./3, 0.4, 3./7,  &
            0.,0.,0., 1./15, 2./21,   0.,0.,0.,0., 1./105 /), (/5,5/))
    c1_ = reshape((/1., 1./3, 1./5, 1./7,    0., 1./3, 1./5, 1./7,    0.,0., 1./15, 2./35,&
            0.,0.,0., 1./105 /), (/4,4/))

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
    !Print *, "apply diff nSets = ", nSets



    !$omp parallel do private(k,l,s,ut,ut0,lpt,Kv_diff,df,ii,jj,j,sqdist,kk) shared &
    !$omp& (num_nodes, num_nodes_y, num_sig, f, sig, ord, a1, a2, c_, c1_,num_neighbors,neighbors, startSet, endSet, nSets)
    do k = 1, num_nodes, 1
        df = 0
        do l = 1, num_nodes_y, 1
            do kk =1, nSets, 1
                ut0 = 0
                do jj=startSet(kk),endSet(kk),1
                    ut0 = ut0 + (x(k,neighbors(jj))-y(l,neighbors(jj)))**2
                end do
                ut0 = sqrt(ut0)
                Kv_diff = 0
                do s = 1, num_sig, 1
                    ut = ut0/sig(s)
                    if (ord > 4) then
                        if (ut < 1e-8) then
                            Kv_diff = Kv_diff - 1.0/(2*sig(s)**2)
                        else
                            Kv_diff = Kv_diff - exp(-0.5*ut)/(2*sig(s)**2)
                        end if
                    else
                        if (ut < 1e-8) then
                            Kv_diff = Kv_diff - c1_(ord,1)/(2*sig(s)**2)
                        else
                            lpt = c1_(ord,1) + c1_(ord,2)*ut + c1_(ord,3)*ut**2 + c1_(ord,4)*ut**3
                            Kv_diff = Kv_diff - lpt * exp(-1.0*ut)/(2*sig(s)**2)
                        end if
                    end if
                end do
                sqdist(kk) = Kv_diff/num_sig
            end do
            do j = 1, dim, 1
                kk = num_neighbors(j)
                do jj =startSet(kk), endSet(kk), 1
                    ii = neighbors(jj)
                    df(ii) = df(ii) + sqdist(kk) * 2*(x(k,ii)-y(l,ii))* sum(a1(:,k,j)*a2(:,l,j))
                end do
            end do
        end do
        f(k, :) = df
    end do
    !$omp end parallel do
end subroutine applylocalkdifft

!subroutine applykdifft(x, y, a1, a2, sig, ord, num_nodes, num_nodes_y, dim, dimb, na, f)
!    implicit none
!    integer :: num_nodes, num_nodes_y, dim, dimb, na
!    real(8) :: x(num_nodes, dim)
!    real(8) :: y(num_nodes_y, dim)
!    real(8) :: sig
!    integer :: ord
!    real(8) :: f(num_nodes, dim)
!    real(8) :: a1(na, num_nodes, dimb)
!    real(8) :: a2(na, num_nodes_y, dimb)
!
!    real(8) :: Kh_diff
!
!    !f2py integer, intent(in) :: num_nodes, num_nodes_y, dim, dimb, na
!    !f2py real(8), intent(in), dimension(num_nodes, dim) :: x
!    !f2py real(8), intent(in), dimension(num_nodes_y, dim) :: y
!    !f2py real(8), intent(in), dimension(na,num_nodes, dimb) :: a1
!    !f2py real(8), intent(in), dimension(na,num_nodes_y, dimb) :: a2
!    !f2py real(8), intent(in) :: sig
!    !f2py integer, intent(in) :: ord
!    !f2py real(8), intent(out), dimension(num_nodes, dim) :: f
!
!    real(8) :: ut
!    real(8) :: lpt
!    integer :: k,l
!    real(8) :: df(dim)
!    real(8) :: c_(5, 5), c1_(4, 4)
!    c_= reshape((/1.,1.,1.,1.,1.,    0.,1.,1.,1.,1.,    0.,0., 1./3, 0.4, 3./7,  &
!            0.,0.,0., 1./15, 2./21,   0.,0.,0.,0., 1./105 /), (/5,5/))
!    c1_ = reshape((/1., 1./3, 1./5, 1./7,    0., 1./3, 1./5, 1./7,    0.,0., 1./15, 2./35,&
!            0.,0.,0., 1./105 /), (/4,4/))
!
!    !$omp parallel do private(k,l,ut,lpt,Kh_diff,df) shared &
!    !$omp& (num_nodes, num_nodes_y, f, sig, ord, a1, a2, c_, c1_)
!    do k = 1, num_nodes, 1
!        df = 0
!        do l = 1, num_nodes_y, 1
!            ut = sqrt(sum((x(k,:) - y(l,:))**2)) / sig
!            if (ord > 4) then
!                ut = ut * ut
!                if (ut < 1e-8) then
!                    Kh_diff = - 1.0/(2*sig**2)
!                else
!                    Kh_diff = (-1.0)* exp(-0.5*ut)/(2*sig**2)
!                end if
!            else
!                if (ut < 1e-8) then
!                    Kh_diff = - c1_(ord,1)/(2*sig*sig)
!                else
!                    lpt = c1_(ord,1) + c1_(ord,2)*ut + c1_(ord,3)*ut**2 + c1_(ord,4)*ut**3
!                    Kh_diff = -lpt * exp(-1.0*ut)/(2*sig**2)
!                end if
!            end if
!            df = df + Kh_diff * 2*(x(k,:)-y(l,:))* sum(a1(:,k,:)*a2(:,l,:)) ;
!        end do
!        f(k, :) = df
!    end do
!    !$omp end parallel do
!end subroutine applykdifft

subroutine applykdifft(x, y, a1, a2, sig, ord, num_nodes, num_nodes_y, num_sig, dim, dimb, na, f)
    implicit none
    integer :: num_nodes, num_nodes_y, dim, dimb, na, num_sig
    real(8) :: x(num_nodes, dim)
    real(8) :: y(num_nodes_y, dim)
    real(8) :: sig(num_sig)
    integer :: ord
    real(8) :: f(num_nodes, dim)
    real(8) :: a1(na, num_nodes, dimb)
    real(8) :: a2(na, num_nodes_y, dimb)

    real(8) :: Kh_diff

    !f2py integer, intent(in) :: num_nodes, num_nodes_y, dim, dimb, num_sig, na
    !f2py real(8), intent(in), dimension(num_nodes, dim) :: x
    !f2py real(8), intent(in), dimension(num_nodes_y, dim) :: y
    !f2py real(8), intent(in), dimension(na,num_nodes, dimb) :: a1
    !f2py real(8), intent(in), dimension(na,num_nodes_y, dimb) :: a2
    !f2py real(8), intent(in), dimension(num_sig) :: sig
    !f2py integer, intent(in) :: ord
    !f2py real(8), intent(out), dimension(num_nodes, dim) :: f

    real(8) :: ut, ut0
    real(8) :: lpt
    integer :: k,l,s
    real(8) :: df(dim)
    real(8) :: c_(5, 5), c1_(4, 4)
    c_= reshape((/1.,1.,1.,1.,1.,    0.,1.,1.,1.,1.,    0.,0., 1./3, 0.4, 3./7,  &
            0.,0.,0., 1./15, 2./21,   0.,0.,0.,0., 1./105 /), (/5,5/))
    c1_ = reshape((/1., 1./3, 1./5, 1./7,    0., 1./3, 1./5, 1./7,    0.,0., 1./15, 2./35,&
            0.,0.,0., 1./105 /), (/4,4/))

    !$omp parallel do private(k,l,s,ut,ut0,lpt,Kh_diff,df) shared &
    !$omp& (num_nodes, num_nodes_y, num_sig, f, sig, ord, a1, a2, c_, c1_)
    do k = 1, num_nodes, 1
        df = 0
        do l = 1, num_nodes_y, 1
            ut0 = sqrt(sum((x(k,:) - y(l,:))**2))
            Kh_diff = 0
            do s = 1, num_sig, 1
                ut = ut0 / sig(s)
                if (ord > 4) then
                    ut = ut * ut
                    if (ut < 1e-8) then
                        Kh_diff = Kh_diff - 1.0/(2*sig(s)**2)
                    else
                        Kh_diff = Kh_diff - exp(-0.5*ut)/(2*sig(s)**2)
                    end if
                else
                    if (ut < 1e-8) then
                        Kh_diff = Kh_diff - c1_(ord,1)/(2*sig(s)**2)
                    else
                        lpt = c1_(ord,1) + c1_(ord,2)*ut + c1_(ord,3)*ut**2 + c1_(ord,4)*ut**3
                        Kh_diff = Kh_diff - lpt * exp(-1.0*ut)/(2*sig(s)**2)
                    end if
                end if
            end do
            Kh_diff = Kh_diff / num_sig
            df = df + Kh_diff * 2*(x(k,:)-y(l,:))* sum(a1(:,k,:)*a2(:,l,:))
        end do
        f(k, :) = df
    end do
    !$omp end parallel do
end subroutine applykdifft

subroutine applykdiff1(x, a1, a2, sig, ord, num_nodes, num_sig, dim, f)
    implicit none
    integer :: num_nodes, dim, num_sig
    real(8) :: x(num_nodes, dim)
    real(8) :: sig(num_sig)
    integer :: ord
    real(8) :: f(num_nodes, dim)
    real(8) :: a1(num_nodes, dim)
    real(8) :: a2(num_nodes, dim)

    real(8) :: Kh_diff

    !f2py integer, intent(in) :: num_nodes, num_nodes, dim, num_sig
    !f2py real(8), intent(in), dimension(num_nodes, dim) :: x
    !f2py real(8), intent(in), dimension(num_nodes, dim) :: a1
    !f2py real(8), intent(in), dimension(num_nodes, dim) :: a2
    !f2py real(8), intent(in), dimension(num_sig) :: sig
    !f2py integer, intent(in) :: ord
    !f2py real(8), intent(out), dimension(num_nodes, dim) :: f

    real(8) :: ut, ut0
    real(8) :: lpt
    integer :: k,l, s
    real(8) :: df(dim)
    real(8) :: c_(5, 5), c1_(4, 4)
    c_= reshape((/1.,1.,1.,1.,1.,    0.,1.,1.,1.,1.,    0.,0., 1./3, 0.4, 3./7,  &
            0.,0.,0., 1./15, 2./21,   0.,0.,0.,0., 1./105 /), (/5,5/))
    c1_ = reshape((/1., 1./3, 1./5, 1./7,    0., 1./3, 1./5, 1./7,    0.,0., 1./15, 2./35,&
            0.,0.,0., 1./105 /), (/4,4/))

    !$omp parallel do private(k,l,s,ut,ut0,lpt,Kh_diff,df) shared &
    !$omp& (num_nodes, num_sig, f, sig, ord, a1, a2, c_, c1_)
    do k = 1, num_nodes, 1
        df = 0
        do l = 1, num_nodes, 1
            ut0 = sqrt(sum((x(k,:) - x(l,:))**2))
            Kh_diff = 0
            do s = 1, num_sig, 1
                ut = ut0 / sig(s)
                if (ord > 4) then
                    ut = ut * ut
                    if (ut < 1e-8) then
                        Kh_diff = Kh_diff - 1.0/(2*sig(s)**2)
                    else
                        Kh_diff = Kh_diff - exp(-0.5*ut)/(2*sig(s)**2)
                    end if
                else
                    if (ut < 1e-8) then
                        Kh_diff = Kh_diff - c1_(ord,1)/(2*sig(s)**2)
                    else
                        lpt = c1_(ord,1) + c1_(ord,2)*ut + c1_(ord,3)*ut**2 + c1_(ord,4)*ut**3
                        Kh_diff = Kh_diff -lpt * exp(-1.0*ut)/(2*sig(s)**2)
                    end if
                end if
            end do !s
            Kh_diff = Kh_diff/num_sig
            df = df + Kh_diff * 2*sum((x(k,:)-x(l,:))*a1(k,:)) * a2(l,:) ;
        end do
        f(k, :) = f(k, :) + df
    end do
    !$omp end parallel do
end subroutine applykdiff1

subroutine applykdiff2(x, a1, a2, sig, ord, num_nodes, num_sig, dim, f)
    implicit none
    integer :: num_nodes, dim, num_sig
    real(8) :: x(num_nodes, dim)
    real(8) :: sig(num_sig)
    integer :: ord
    real(8) :: f(num_nodes, dim)
    real(8) :: a1(num_nodes, dim)
    real(8) :: a2(num_nodes, dim)

    real(8) :: Kh_diff

    !f2py integer, intent(in) :: num_nodes, dim, num_sig
    !f2py real(8), intent(in), dimension(num_nodes, dim) :: x
    !f2py real(8), intent(in), dimension(num_nodes, dim) :: a1
    !f2py real(8), intent(in), dimension(num_nodes, dim) :: a2
    !f2py real(8), intent(in), dimension(num_sig) :: sig
    !f2py integer, intent(in) :: ord
    !f2py real(8), intent(out), dimension(num_nodes, dim) :: f

    real(8) :: ut, ut0
    real(8) :: lpt
    integer :: k,l, s
    real(8) :: df(dim)
    real(8) :: c_(5, 5), c1_(4, 4)
    c_= reshape((/1.,1.,1.,1.,1.,    0.,1.,1.,1.,1.,    0.,0., 1./3, 0.4, 3./7,  &
            0.,0.,0., 1./15, 2./21,   0.,0.,0.,0., 1./105 /), (/5,5/))
    c1_ = reshape((/1., 1./3, 1./5, 1./7,    0., 1./3, 1./5, 1./7,    0.,0., 1./15, 2./35,&
            0.,0.,0., 1./105 /), (/4,4/))

    !$omp parallel do private(k,l,s,ut,ut0,lpt,Kh_diff,df) shared &
    !$omp& (num_nodes, num_sig, f, sig, ord, a1, a2, c_, c1_)
    do k = 1, num_nodes, 1
        df = 0
        do l = 1, num_nodes, 1
            ut0 = sqrt(sum((x(k,:) - x(l,:))**2))
            Kh_diff = 0
            do s = 1, num_sig, 1
                ut = ut0 / sig(s)
                if (ord > 4) then
                    ut = ut * ut
                    if (ut < 1e-8) then
                        Kh_diff = Kh_diff - 1.0/(2*sig(s)**2)
                    else
                        Kh_diff = Kh_diff - exp(-0.5*ut)/(2*sig(s)**2)
                    end if
                else
                    if (ut < 1e-8) then
                        Kh_diff = Kh_diff - c1_(ord,1)/(2*sig(s)**2)
                    else
                        lpt = c1_(ord,1) + c1_(ord,2)*ut + c1_(ord,3)*ut**2 + c1_(ord,4)*ut**3
                        Kh_diff = Kh_diff - lpt * exp(-1.0*ut)/(2*sig(s)**2)
                    end if
                end if
            end do !s
            Kh_diff = Kh_diff/num_sig
            df = df - Kh_diff * 2*sum((x(k,:)-x(l,:))*a1(l,:)) * a2(l,:) ;
        end do
        f(k, :) = f(k, :) + df
    end do
    !$omp end parallel do
end subroutine applykdiff2


subroutine applykdiff1and2(x, a1, a2, sig, ord, num_nodes, num_sig, dim, f)
    implicit none
    integer :: num_nodes, dim, num_sig
    real(8) :: x(num_nodes, dim)
    real(8) :: sig(num_sig)
    integer :: ord
    real(8) :: f(num_nodes, dim)
    real(8) :: a1(num_nodes, dim)
    real(8) :: a2(num_nodes, dim)

    real(8) :: Kh_diff

    !f2py integer, intent(in) :: num_nodes, dim, num_sig
    !f2py real(8), intent(in), dimension(num_nodes, dim) :: x
    !f2py real(8), intent(in), dimension(num_nodes, dim) :: a1
    !f2py real(8), intent(in), dimension(num_nodes, dim) :: a2
    !f2py real(8), intent(in), dimension(num_sig) :: sig
    !f2py integer, intent(in) :: ord
    !f2py real(8), intent(out), dimension(num_nodes, dim) :: f

    real(8) :: ut,ut0
    real(8) :: lpt
    integer :: k,l,s
    real(8) :: df(dim), dx(dim)
    real(8) :: c_(5, 5), c1_(4, 4)
    c_= reshape((/1.,1.,1.,1.,1.,    0.,1.,1.,1.,1.,    0.,0., 1./3, 0.4, 3./7,  &
            0.,0.,0., 1./15, 2./21,   0.,0.,0.,0., 1./105 /), (/5,5/))
    c1_ = reshape((/1., 1./3, 1./5, 1./7,    0., 1./3, 1./5, 1./7,    0.,0., 1./15, 2./35,&
            0.,0.,0., 1./105 /), (/4,4/))

    !$omp parallel do private(k,l,s,ut,ut0,lpt,Kh_diff,dx,df) shared &
    !$omp& (num_nodes, num_sig, x, f, sig, ord, a1, a2, c_, c1_)
    do k = 1, num_nodes, 1
        df = 0
        do l = 1, num_nodes, 1
            dx=x(k,:) - x(l,:)
            ut0 = sqrt(sum(dx**2))
            Kh_diff = 0
            do s=1,num_sig,1
                ut = ut0 / sig(s)
                if (ord > 4) then
                    ut = ut * ut
                    if (ut < 1e-8) then
                        Kh_diff = Kh_diff - 1.0/(2*sig(s)**2)
                    else
                        Kh_diff = Kh_diff - exp(-0.5*ut)/(2*sig(s)**2)
                    end if
                else
                    if (ut < 1e-8) then
                        Kh_diff = Kh_diff - c1_(ord,1)/(2*sig(s)**2)
                    else
                        lpt = c1_(ord,1) + c1_(ord,2)*ut + c1_(ord,3)*ut**2 + c1_(ord,4)*ut**3
                        Kh_diff = Kh_diff - lpt * exp(-1.0*ut)/(2*sig(s)**2)
                    end if
                end if
            end do !s
            Kh_diff = Kh_diff/ num_sig
            df = df + Kh_diff * 2*sum(dx*(a1(k,:)-a1(l,:))) * a2(l,:) ;
        end do
        f(k, :) = df
    end do
    !$omp end parallel do
end subroutine applykdiff1and2


subroutine applykdiff11(x, a1, a2, p, sig, ord, num_nodes, num_sig, dim, f)
    implicit none
    integer :: num_nodes, num_nodes_y, dim, num_sig
    real(8) :: x(num_nodes, dim)
    real(8) :: y(num_nodes, dim)
    real(8) :: sig(num_sig)
    integer :: ord
    real(8) :: f(num_nodes, dim)
    real(8) :: a1(num_nodes, dim)
    real(8) :: a2(num_nodes, dim)
    real(8) :: p(num_nodes, dim)

    real(8) :: Kh_diff, Kh_diff2

    !f2py integer, intent(in) :: num_nodes, num_nodes, dim, num_sig
    !f2py real(8), intent(in), dimension(num_nodes, dim) :: x
    !f2py real(8), intent(in), dimension(num_nodes, dim) :: a1
    !f2py real(8), intent(in), dimension(num_nodes, dim) :: a2
    !f2py real(8), intent(in), dimension(num_nodes, dim) :: p
    !f2py real(8), intent(in), dimension(num_sig) :: sig
    !f2py integer, intent(in) :: ord
    !f2py real(8), intent(out), dimension(num_nodes, dim) :: f

    real(8) :: ut,ut0
    real(8) :: lpt
    integer :: k,l,s
    real(8) :: df(dim)
    real(8) :: c_(5, 5), c1_(4, 4), c2_(3,3)
    c_= reshape((/1.,1.,1.,1.,1.,    0.,1.,1.,1.,1.,    0.,0., 1./3, 0.4, 3./7,  &
            0.,0.,0., 1./15, 2./21,   0.,0.,0.,0., 1./105 /), (/5,5/))
    c1_ = reshape((/1., 1./3, 1./5, 1./7,    0., 1./3, 1./5, 1./7,    0.,0., 1./15, 2./35,&
            0.,0.,0., 1./105 /), (/4,4/))
    c2_ = reshape((/ 1.0/3, 1./15, 1./35,    0., 1./15, 1./35,   0.,0., 1./105 /), (/3,3/))

    !$omp parallel do private(k,l,s,ut,ut0,lpt,Kh_diff,df) shared &
    !$omp& (num_nodes, num_nodes_y, num_sig, f, sig, ord, a1, a2, c_, c1_)
    do k = 1, num_nodes, 1
        df = 0
        do l = 1, num_nodes_y, 1
            ut0 = sqrt(sum((x(k,:) - y(l,:))**2))
            Kh_diff = 0
            Kh_diff2 = 0
            do s = 1, num_sig, 1
                ut = ut0/sig(s)
                if (ord > 4) then
                    ut = ut * ut
                    if (ut < 1e-8) then
                        Kh_diff = Kh_diff - 1.0/(2*sig(s)**2)
                        Kh_diff2 = Kh_diff2 + 1.0/(4*sig(s)**4)
                    else
                        Kh_diff = Kh_diff - exp(-0.5*ut)/(2*sig(s)**2)
                        Kh_diff2 = Kh_diff2 + exp(-0.5*ut)/(4*sig(s)**4)
                    end if
                else
                    if (ut < 1e-8) then
                        Kh_diff = Kh_diff - c1_(ord,1)/(2*sig(s)**2)
                        Kh_diff2 = Kh_diff2 + c2_(ord-1,1)/(4*sig(s)**4)
                    else
                        lpt = c1_(ord,1) + c1_(ord,2)*ut + c1_(ord,3)*ut**2 + c1_(ord,4)*ut**3
                        Kh_diff = Kh_diff - lpt * exp(-1.0*ut)/(2*sig(s)**2)
                        lpt = c2_(ord-1,1) + c2_(ord-1,2)*ut + c2_(ord-1,3)*ut**2
                        Kh_diff2 = Kh_diff2 + lpt * exp(-1.0*ut)/(4*sig(s)**4)
                    end if
                end if
            end do !s
            Kh_diff = Kh_diff / num_sig
            Kh_diff2 = Kh_diff2 / num_sig
            df = df + Kh_diff2 * 4*sum(a1(k,:) * a2(l,:)) * sum((x(k,:)-y(l,:))*p(k,:)) *(x(k,:)-y(l,:)) &
                    & + Kh_diff * 2 * sum(a1(k,:) * a2(l,:)) * p(k,:)
        end do
        f(k, :) = f(k, :) + df
    end do
    !$omp end parallel do
end subroutine applykdiff11

subroutine applykdiff12(x, a1, a2, p, sig, ord, num_nodes, num_sig, dim, f)
    implicit none
    integer :: num_nodes, num_nodes_y, dim, num_sig
    real(8) :: x(num_nodes, dim)
    real(8) :: y(num_nodes, dim)
    real(8) :: sig(num_sig)
    integer :: ord
    real(8) :: f(num_nodes, dim)
    real(8) :: a1(num_nodes, dim)
    real(8) :: a2(num_nodes, dim)
    real(8) :: p(num_nodes, dim)

    real(8) :: Kh_diff, Kh_diff2

    !f2py integer, intent(in) :: num_nodes, num_nodes, dim, num_sig
    !f2py real(8), intent(in), dimension(num_nodes, dim) :: x
    !f2py real(8), intent(in), dimension(num_nodes, dim) :: a1
    !f2py real(8), intent(in), dimension(num_nodes, dim) :: a2
    !f2py real(8), intent(in), dimension(num_nodes, dim) :: p
    !f2py real(8), intent(in), dimension(num_sig) :: sig
    !f2py integer, intent(in) :: ord
    !f2py real(8), intent(out), dimension(num_nodes, dim) :: f

    real(8) :: ut,ut0
    real(8) :: lpt
    integer :: k,l,s
    real(8) :: df(dim)
    real(8) :: c_(5, 5), c1_(4, 4), c2_(3,3)
    c_= reshape((/1.,1.,1.,1.,1.,    0.,1.,1.,1.,1.,    0.,0., 1./3, 0.4, 3./7,  &
            0.,0.,0., 1./15, 2./21,   0.,0.,0.,0., 1./105 /), (/5,5/))
    c1_ = reshape((/1., 1./3, 1./5, 1./7,    0., 1./3, 1./5, 1./7,    0.,0., 1./15, 2./35,&
            0.,0.,0., 1./105 /), (/4,4/))
    c2_ = reshape((/ 1.0/3, 1./15, 1./35,    0., 1./15, 1./35,   0.,0., 1./105 /), (/3,3/))

    !$omp parallel do private(k,l,s,ut,ut0,lpt,Kh_diff,df) shared &
    !$omp& (num_nodes, num_nodes_y, num_sig, f, sig, ord, a1, a2, c_, c1_)
    do k = 1, num_nodes, 1
        df = 0
        do l = 1, num_nodes_y, 1
            ut = sqrt(sum((x(k,:) - y(l,:))**2))
            Kh_diff = 0
            Kh_diff2 = 0
            do s=1, num_sig, 1
                ut = ut0/sig(s)
                if (ord > 4) then
                    ut = ut * ut
                    if (ut < 1e-8) then
                        Kh_diff = Kh_diff - 1.0/(2*sig(s)**2)
                        Kh_diff2 = Kh_diff2 + 1.0/(4*sig(s)**4)
                    else
                        Kh_diff = Kh_diff - exp(-0.5*ut)/(2*sig(s)**2)
                        Kh_diff2 = Kh_diff2 + exp(-0.5*ut)/(4*sig(s)**4)
                    end if
                else
                    if (ut < 1e-8) then
                        Kh_diff = Kh_diff - c1_(ord,1)/(2*sig(s)**2)
                        Kh_diff2 = Kh_diff2 + c2_(ord-1,1)/(4*sig(s)**4)
                    else
                        lpt = c1_(ord,1) + c1_(ord,2)*ut + c1_(ord,3)*ut**2 + c1_(ord,4)*ut**3
                        Kh_diff = Kh_diff - lpt * exp(-1.0*ut)/(2*sig(s)**2)
                        lpt = c2_(ord-1,1) + c2_(ord-1,2)*ut + c2_(ord-1,3)*ut**2
                        Kh_diff2 = Kh_diff2 + lpt * exp(-1.0*ut)/(4*sig(s)**4)
                    end if
                end if
            end do !s
            Kh_diff = Kh_diff / num_sig
            Kh_diff2 = Kh_diff2 / num_sig
            df = df - Kh_diff2 * 4*sum(a1(k,:) * a2(l,:)) &
                    &* sum((x(k,:)-y(l,:))*p(l,:)) *(x(k,:)-y(l,:)) &
                    & - Kh_diff * 2 * sum(a1(k,:) * a2(l,:)) * p(l,:)
        end do
        f(k, :) = f(k, :) + df
    end do
    !$omp end parallel do
end subroutine applykdiff12

subroutine applykdiff11and12(x, a1, a2, p, sig, ord, num_nodes, num_sig, dim, f)
    implicit none
    integer :: num_nodes, dim, num_sig
    real(8) :: x(num_nodes, dim)
    real(8) :: sig(num_sig)
    integer :: ord
    real(8) :: f(num_nodes, dim)
    real(8) :: a1(num_nodes, dim)
    real(8) :: a2(num_nodes, dim)
    real(8) :: p(num_nodes, dim)

    real(8) :: Kh_diff, Kh_diff2

    !f2py integer, intent(in) :: num_nodes, dim, num_sig
    !f2py real(8), intent(in), dimension(num_nodes, dim) :: x
    !f2py real(8), intent(in), dimension(num_nodes, dim) :: a1
    !f2py real(8), intent(in), dimension(num_nodes, dim) :: a2
    !f2py real(8), intent(in), dimension(num_nodes, dim) :: p
    !f2py real(8), intent(in), dimension(num_sig) :: sig
    !f2py integer, intent(in) :: ord
    !f2py real(8), intent(out), dimension(num_nodes, dim) :: f

    real(8) :: ut,ut0
    real(8) :: lpt
    integer :: k,l,s
    real(8) :: df(dim), dx(dim), dp(dim)
    real(8) :: c_(5, 5), c1_(4, 4), c2_(3,3)
    c_= reshape((/1.,1.,1.,1.,1.,    0.,1.,1.,1.,1.,    0.,0., 1./3, 0.4, 3./7,  &
            0.,0.,0., 1./15, 2./21,   0.,0.,0.,0., 1./105 /), (/5,5/))
    c1_ = reshape((/1., 1./3, 1./5, 1./7,    0., 1./3, 1./5, 1./7,    0.,0., 1./15, 2./35,&
            0.,0.,0., 1./105 /), (/4,4/))
    c2_ = reshape((/ 1.0/3, 1./15, 1./35,    0., 1./15, 1./35,   0.,0., 1./105 /), (/3,3/))

    !$omp parallel do private(k,l,s,ut,ut0,lpt,Kh_diff,Kh_diff2,df,dx,dp) shared &
    !$omp& (num_nodes, num_sig, x, p, f, sig, ord, a1, a2, c2_, c1_)
    do k = 1, num_nodes, 1
        df = 0
        do l = 1, num_nodes, 1
            dx = x(k,:) - x(l,:)
            dp = p(k,:) - p(l,:)
            ut0 = sqrt(sum(dx**2))
            Kh_diff = 0
            Kh_diff2 = 0
            do s = 1, num_sig, 1
                ut = ut0/sig(s)
                if (ord > 4) then
                    ut = ut * ut
                    if (ut < 1e-8) then
                        Kh_diff = Kh_diff - 1.0/(2*sig(s)**2)
                        Kh_diff2 = Kh_diff2 + 1.0/(4*sig(s)**4)
                    else
                        Kh_diff = Kh_diff - exp(-0.5*ut)/(2*sig(s)**2)
                        Kh_diff2 = Kh_diff2 + exp(-0.5*ut)/(4*sig(s)**4)
                    end if
                else
                    if (ut < 1e-8) then
                        Kh_diff = Kh_diff - c1_(ord,1)/(2*sig(s)**2)
                        Kh_diff2 = Kh_diff2 + c2_(ord-1,1)/(4*sig(s)**4)
                    else
                        lpt = c1_(ord,1) + c1_(ord,2)*ut + c1_(ord,3)*ut**2 + c1_(ord,4)*ut**3
                        Kh_diff = Kh_diff - lpt * exp(-1.0*ut)/(2*sig(s)**2)
                        lpt = c2_(ord-1,1) + c2_(ord-1,2)*ut + c2_(ord-1,3)*ut**2
                        Kh_diff2 = Kh_diff2 + lpt * exp(-1.0*ut)/(4*sig(s)**4)
                    end if
                end if
            end do !s
            Kh_diff = Kh_diff/num_sig
            Kh_diff2 = Kh_diff2/num_sig
            df = df + 2 * sum(a1(k,:)*a2(l,:)) *  (2 * Kh_diff2 *sum(dx*dp) *dx + Kh_diff * dp)
        end do
        f(k, :) = df
    end do
    !$omp end parallel do
end subroutine applykdiff11and12


subroutine applykmat(x, y, beta, sig, ord, num_nodes, num_nodes_y, num_sig, dim, dimb, f)
    implicit none
    integer :: num_nodes, num_nodes_y, dim, dimb, num_sig
    real(8) :: x(num_nodes, dim)
    real(8) :: y(num_nodes_y, dim)
    real(8) :: sig(num_sig)
    integer :: ord
    real(8) :: f(num_nodes,dimb)
    real(8) :: beta(num_nodes, num_nodes_y,dimb)
    real(8) :: Kh

    !f2py integer, intent(in) :: num_nodes, num_nodes_y, dim, dimb,num_sig
    !f2py real(8), intent(in), dimension(num_nodes, dim) :: x
    !f2py real(8), intent(in), dimension(num_nodes_y, dim) :: y
    !f2py real(8), intent(in), dimension(num_nodes, num_nodes_y,dimb) :: beta
    !f2py real(8), intent(in), dimension(num_sig) :: sig
    !f2py integer, intent(in) :: ord
    !f2py real(8), intent(out), dimension(num_nodes,dimb) :: f

    real(8) :: ut, ut0
    real(8) :: lpt
    integer :: k,l, s
    real(8) :: df(dimb)
    real(8) :: c_(5, 5)
    c_= reshape((/1.,1.,1.,1.,1.,    0.,1.,1.,1.,1.,    0.,0., 1./3, 0.4, 3./7,  &
            0.,0.,0., 1./15, 2./21,   0.,0.,0.,0., 1./105 /), (/5,5/))

    !$omp parallel do private(k,l,s,ut,ut0,lpt,Kh,df) shared &
    !$omp& (num_nodes, num_nodes_y, num_sig, f, sig, ord, beta, c_)
    do k = 1, num_nodes, 1
        df = 0
        do l = 1, num_nodes_y, 1
            ut0 = sqrt(sum((x(k,:) - y(l,:))**2))
            Kh = 0
            do s = 1, num_sig, 1
                ut = ut0/sig(s)
                if (ord > 4) then
                    ut = ut * ut
                    if (ut < 1e-8) then
                        Kh = Kh + 1.0
                    else
                        Kh = Kh + exp(-0.5*ut)
                    end if
                else
                    if (ut < 1e-8) then
                        Kh = Kh + 1.0
                    else
                        lpt = c_(ord+1, 1) + c_(ord+1,2)*ut + c_(ord+1,3)*ut**2 + c_(ord+1,4)*ut**3 + c_(ord+1,5)*ut**4
                        Kh = Kh + lpt * exp(-1.0*ut)
                    end if
                end if
            end do !s
            Kh = Kh / num_sig
            df = df + Kh * beta(k,l,:)
        end do
        f(k,:) = df
    end do
    !$omp end parallel do
end subroutine applyKmat

subroutine applykdiffmat(x, y, beta, sig, ord, num_nodes, num_nodes_y, num_sig, dim, f)
    implicit none
    integer :: num_nodes, num_nodes_y, dim, num_sig
    real(8) :: x(num_nodes, dim)
    real(8) :: y(num_nodes_y, dim)
    real(8) :: sig(num_sig)
    integer :: ord
    real(8) :: f(num_nodes, dim)
    real(8) :: beta(num_nodes, num_nodes_y)

    real(8) :: Kh_diff

    !f2py integer, intent(in) :: num_nodes, num_nodes_y, dim, num_sig
    !f2py real(8), intent(in), dimension(num_nodes, dim) :: x
    !f2py real(8), intent(in), dimension(num_nodes_y, dim) :: y
    !f2py real(8), intent(in), dimension(num_nodes, num_nodes_y) :: beta
    !f2py real(8), intent(in), dimension(num_sig) :: sig
    !f2py integer, intent(in) :: ord
    !f2py real(8), intent(out), dimension(num_nodes, dim) :: f

    real(8) :: ut,ut0
    real(8) :: lpt
    integer :: k,l,s
    real(8) :: df(dim)
    real(8) :: c_(5, 5), c1_(4, 4)
    c_= reshape((/1.,1.,1.,1.,1.,    0.,1.,1.,1.,1.,    0.,0., 1./3, 0.4, 3./7,  &
            0.,0.,0., 1./15, 2./21,   0.,0.,0.,0., 1./105 /), (/5,5/))
    c1_ = reshape((/1., 1./3, 1./5, 1./7,    0., 1./3, 1./5, 1./7,    0.,0., 1./15, 2./35,&
            0.,0.,0., 1./105 /), (/4,4/))

    !$omp parallel do private(k,l,s,ut,ut0,lpt,Kh_diff,df) shared &
    !$omp& (num_nodes, num_nodes_y, num_sig, f, sig, ord, beta, c_, c1_)
    do k = 1, num_nodes, 1
        df = 0
        do l = 1, num_nodes_y, 1
            ut0 = sqrt(sum((x(k,:) - y(l,:))**2))
            Kh_diff = 0
            do s=1,num_sig,1
                ut = ut0/sig(s)
                if (ord > 4) then
                    ut = ut * ut
                    if (ut < 1e-8) then
                        Kh_diff = Kh_diff - 1.0/(2*sig(s)**2)
                    else
                        Kh_diff = Kh_diff - exp(-0.5*ut)/(2*sig(s)**2)
                    end if
                else
                    if (ut < 1e-8) then
                        Kh_diff = Kh_diff - c1_(ord,1)/(2*sig(s)**2)
                    else
                        lpt = c1_(ord,1) + c1_(ord,2)*ut + c1_(ord,3)*ut**2 + c1_(ord,4)*ut**3
                        Kh_diff = Kh_diff - lpt * exp(-1.0*ut)/(2*sig(s)**2)
                    end if
                end if
            end do!
            Kh_diff = Kh_diff / num_sig
            df = df + Kh_diff * 2*(x(k,:)-y(l,:))* beta(k,l)
        end do
        f(k, :) = df
    end do
    !$omp end parallel do
end subroutine applykdiffmat


