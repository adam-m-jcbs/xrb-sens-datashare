subroutine rk4(x0,y0,y1,dx,n,w)
  implicit none
  
  common /rk4out/ z0, z1
  
  real*8, intent(in)  :: x0,y0,y1
  real*8, intent(in)  :: dx,n,w
  real*8 :: z0, z1
  real*8 :: xh, dh
  real*8 :: k10,k11,k20,k21,k30,k31,k40,k41            
 
!f2py real(8), intent(in) :: x0, dx, n, w
!f2py real(8), intent(in) :: y0, y1

  xh = x0 + 0.5d0 * dx
  dh = 0.5d0 * dx
  
  k10 = y1
  if (x0.EQ.0) then
     k11 = (w - 1.d0)/3.d0
  else
     k11 = -2.d0 / x0 * y1 - (max(y0, 0.d0))**n + w
  endif

  k20 = y1 + dh*k11
  k21 = -2.d0 / xh * k20 - (max(y0 + dh*k10,0.d0))**n
  
  k30 = y1 + dh*k21
  k31 = -2.d0 / xh * k30 - (max(y0 + dh*k20,0.d0))**n

  k40 = y1 + dx*k31
  k41 = -2.d0 / (x0+dx) * k40 - (max(y0 + dx*k30,0.d0))**n
  
  z0 = y0 + dx*(k10 + 2d0 * (k20 + k30) + k40) / 6.d0
  z1 = y1 + dx*(k11 + 2.d0 * (k21 + k31) + k41) / 6.d0
  
end subroutine rk4
