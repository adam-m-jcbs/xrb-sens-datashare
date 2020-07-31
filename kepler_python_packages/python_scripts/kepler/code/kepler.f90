program test
  implicit none

  integer :: nargs
  character(80), dimension(0:20) :: inputarg

!.... KEPLER signature information

  call progvers
  call proggit
  call proguuid

!.... set program arguments

  nargs = 1
  inputarg(0) = 'python'
  inputarg(1) = '-v'

  call kepinfo(inputarg, nargs)

end program test

!=======================================================================

subroutine start_(nargs, input)

  integer :: i,j

  integer(kind=4), intent(IN) :: nargs
  character*1, dimension(80*21), intent(IN) :: input

  character*80, dimension(0:20):: inputarg


!.... KEPLER signature information

  call progvers
  call proggit
  call proguuid

!.... get program arguments

  DO i=0,20
     DO j=1,80
        inputarg(i)(j:j) = input(j+i*80)
     enddo
     inputarg(i) = trim(inputarg(i))
  enddo

  call kepinfo(inputarg, nargs)

!.... some initialization that is not contained in a block data

  call kepinit

!.... some platform specific settings

  call syspec

!.... setup takes care of restarting or generating

  write(6, "(A)") ' [PYTHON]  setting up kepler ...'

  call setup(inputarg, nargs)

end subroutine start_

!=======================================================================

subroutine execute_(xcmdline, logitx)

  implicit none
  character*(*), intent(IN) :: xcmdline
  integer(kind=4), intent(IN) :: logitx

!f2py intent(in) xcmdline
!f2py intent(in) logitx

  call execute(xcmdline, len(xcmdline), logitx)

end subroutine execute_

!=======================================================================

subroutine cycle_(interactive)

  implicit none
  logical, intent(IN):: interactive

  call cycle(interactive)

end subroutine cycle_

!=======================================================================

subroutine terminate_(s)

  character*(*), intent(in) :: s

  call terminate(s)

end subroutine terminate_

!=======================================================================

subroutine loadbuf_(namedat,datbuf,jmin,jmax,datlabel,ierr)

  implicit none

  include 'typecom'
  include 'gridcom'

  character*8, intent(in) :: &
       namedat
  integer(kind=int32), intent(in) :: &
       jmin, jmax
  character*48, intent(out) :: &
       datlabel
  integer(kind=int32), intent(out):: &
       ierr
  real(kind=real64), intent(out), dimension(0:jmz) :: &
       datbuf
  character*8 :: &
       name

  name = trim(namedat)

  call loadbuf(name,datbuf,jmin,jmax,datlabel,ierr)

end subroutine loadbuf_

!=======================================================================

subroutine pyexit(code)

  implicit none
  integer(kind=4), intent(IN):: code

  !f2py    intent(callback, hide) endkepler(code)
  external endkepler

  call endkepler(code)

end subroutine pyexit


!=======================================================================

subroutine pyplot

  implicit none

  !f2py    intent(callback, hide) plotkepler()
  external plotkepler

  call plotkepler()

end subroutine pyplot

!=======================================================================

subroutine pygets(ttymsg)

  implicit none
  character*(*), intent(inout) :: ttymsg

  integer, parameter :: n = 132
  integer :: i
  integer(kind=1), dimension(n) :: data

  !f2py    intent(callback, hide) ttykepler(data)
  external ttykepler

  call ttykepler(data)

  do i = 1, min(n, len(ttymsg))
     ttymsg(i:i) = char(data(i))
  end do

  !ttymsg = 'aaa'
  !read (5,'(1a)') ttymsg

end subroutine pygets

!=======================================================================

subroutine getentropies_(datbuf, jmin, jmax)

! this interface avoids having to include kepcom

  implicit none

  include 'typecom'
  include 'gridcom'

  integer(kind=int32), intent(in) :: &
       jmin, jmax
  real(kind=real64), intent(out), dimension(0:jmz, 0:5) :: &
       datbuf

  call getentropies(datbuf, jmin, jmax)

end subroutine getentropies_
