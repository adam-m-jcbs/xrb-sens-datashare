!=======================================================================

      module data

      implicit real*8 (a-h,o-z)
      save

      include 'kepcom'

c$$$c.... requires fix in crackfortran.py
c$$$1163,1166c1163,1164
c$$$<             if c[0] in commonkey:
c$$$<                 outmess('analyzeline: previously defined common block encountered. Skipping.\n')
c$$$<                 continue
c$$$<             commonkey[c[0]]=[]
c$$$---
c$$$>             if c[0] not in commonkey:
c$$$>                 commonkey[c[0]]=[]

      logical :: py_done
      common python, py_done

      end module data
c=======================================================================

c$$$      subroutine loadbuf_(namedat0,datbuf,jmin,jmax,datlabel,ierr)
c$$$
c$$$c     implicit none
c$$$c      integer(kind=4), parameter:: jmz = 1983
c$$$      implicit real*8 (a-h,o-z)
c$$$      include 'kepcom'
c$$$
c$$$      character(len=8), intent(in):: namedat0
c$$$      integer(kind=4), intent(in):: jmin, jmax
c$$$      character(len = 48), intent(out):: datlabel
c$$$      integer(kind=4), intent(out):: ierr
c$$$      real(kind=8), intent(out), dimension(0:jmz) :: datbuf
c$$$      character(len=8):: namedat
c$$$
c$$$      namedat = trim(namedat0)
c$$$
c$$$      call loadbuf(namedat,datbuf,jmin,jmax,datlabel,ierr)
c$$$
c$$$      end subroutine loadbuf_
