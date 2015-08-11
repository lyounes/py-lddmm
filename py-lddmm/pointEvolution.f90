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

  real(8) :: ut, Kv, Kv_diff, lpt
  real(8) :: ada
  integer :: t, k, l
  real(8) :: dx(dimx),dy(dimx),dnu(dimx),djac
  real(8) :: c_(5, 5), c1_(4, 4)
  real(8) :: dt
  real(8) :: xt(numx, dimx)
  real(8) :: alphat(numx, dimx)
  real(8) :: yt(numy, dimx)
  real(8) :: at(dimx, dimx)
  real(8) :: bt(dimx)
  real(8) :: jact(numx)
  real(8) :: nut(numx, dimx)

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
     !$omp parallel do private(k,l,ut,lpt,Kv, Kv_diff, &
     !$omp& dx, dy, dnu, djac) shared (alphat,at,bt,xt,nut,x,nu,jac,sig,ord,c_,c1_,numy,numx)
     do k = 1, numx, 1
        dx=0
        dnu=0
        djac = 0
        do l=1, numx, 1
          ut = sqrt(sum((x(t,k,:) - x(t,l,:))**2)) / sig
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
            djac = djac + 2* Kv_diff * dot_product(xt(k,:) - xt(l,:),alphat(l,:))
          end if
          if (withnu > 0) then
            dnu = dnu - 2 * Kv_diff * dot_product(nut(:,l),alphat(l,:)) * (xt(k,:) - xt(l,:))
          end if
        end do !l
        x(t+1, k,:) = matmul(at,xt(k,:)) + dt*(dx+bt)
        if (withnu > 0) then
          nu(t+1,k,:) = matmul(nut(k,:),at) + dt * dnu
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
          ut = sqrt(sum((y(t,k,:)-x(t,l,:))**2)) / sig 
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
        y(t+1, k,:) = matmul(at, yt(t,:)) + dt*(dy + bt)
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

  real(8) :: ut, Kv, Kv_diff, lpt
  integer :: t, k, l
  real(8) :: dx(dimx), dpx(dimx)
  real(8) :: c_(5, 5), c1_(4, 4)
  real(8) :: dt
  real(8) :: xtt(numx, dimx)
  real(8) :: alphat(numx, dimx)
  real(8) :: at(dimx, dimx)
  real(8) :: pxt(numx, dimx)

  !f2py integer, intent(in) :: t0, numx, dimx, ord
  !f2py real(8), intent(in) :: sig, regweight
  !f2py real(8), intent(in), dimension(t0+1,numx, dimx) :: xt
  !f2py real(8), intent(in), dimension(numx, dimx) :: px1
  !f2py real(8), intent(in), dimension(t0, numx, dimx) :: alpha
  !f2py real(8), intent(in), dimension(t0, dimx, dimx) :: a
  !f2py real(8), intent(out), dimension(t0+1, numx, dimx) :: px

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
     !$omp parallel do private(k,l,ut,lpt,Kv_diff, &
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
          dpx = dpx + 2*Kv_diff * (dot_product(pxt(k,:),alphat(l, :)) &
                                + dot_product(pxt(l, :),alphat(k, :))&
                                - 2*regweight*dot_product(alphat(k,:),alphat(l, :)))&
                              * (xtt(k,:) - xtt(l,:))
        end do !l
        px(t, k,:) = matmul(transpose(at), pxt(k, :)) + dt* dpx
     end do !k
  end do !t
end subroutine adjoint1order

