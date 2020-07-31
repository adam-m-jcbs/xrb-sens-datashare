! 20111212 Alexander Heger

! 1/z**2 d/dz (z**2 d/dz theta(z)) + theta(z)**n = 0
! for small z one can approximate
! theta(z) = 1. + (-1/6.)*z**2 + (n/120.)*z**4 + O(z**6)
! Therefore lim(z)-->0 d**2 theta(z)/d z**2 = -1/3

! 20120423 Alexander Heger

! if we include a constant rotation rate Omega the equation becomes
! 1/z**2 d/dz (z**2 d/dz theta(z)) + theta(z)**n - w = 0
! where 
! w = W/rho_c
! W = 2 Omega**2 / 4 pi G
! for small z one can approximate
! theta(z) = 1. + (w - 1)/6. *z**2 + (1.-w)*(n/120.)*z**4 + O(z**6)
! Therefore lim(z)-->0 d**2 theta(z)/d z**2 = (w-1.)/3

program test
  implicit none

  integer, parameter :: maxdata = 2**20-1
  real*8, dimension(0:maxdata,0:1) :: theta
  integer :: ndata
  common /laneout/ theta, ndata
  
  real*8 :: dz,n,w
  real*8 :: z1,t0,t1,d0,d1,f,y,t,d

  dz = 2.d0**(-14)
  n  = 3.d0
  w  = 0.d0
  call lane(dz,n,w)
  
  z1 = dz*ndata
  t0 = theta(ndata-1,0)
  t1 = theta(ndata  ,0)
  d0 = theta(ndata-1,1)
  d1 = theta(ndata  ,1)
  f  = t1 / (t0-t1)
  y = z1 + f * dz	
  t = t1 + f * (t1-t0)
  d = d1 + f * (d1-d0)

  print*,y,t,d
  
end program test


subroutine lane(dx,n,w)
  implicit none
  
!f2py real(8), intent(in) :: dx, n, w
  
  real*8, intent(in) :: dx, n, w
  
  common /rk4out/ z0, z1
  real*8 :: z0, z1
  
  real*8 :: x,y0,y1
  integer :: i
  
  integer, parameter :: maxdata = 2**20-1
  real*8, dimension(0:maxdata,0:1) :: theta
  integer :: ndata
  common /laneout/ theta, ndata
  
  ndata = 0
  x = 0.d0
  y0 = 1.d0
  y1 = 0.d0 
  
  theta(i,0:1) = (/y0,y1/)
  do i = 1, maxdata
     call rk4(x,y0,y1,dx,n,w)
     theta(i,0:1) = (/z0,z1/)
     x = x + dx
     y0 = z0
     y1 = z1
     
     if (y0 < 0.d0) exit
  enddo
  ndata = i
  
end subroutine lane

