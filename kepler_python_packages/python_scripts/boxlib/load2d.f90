! Load plotfile data into memory. To be wrapped using f2py for usage
! from python. Based on fIDLdump.f90 by M. Zingale.
!
! The plotfile data is  mapped onto a single, uniform grid, with 
! resolution set to the resolution of the highest AMR level we specify.
!
! Laurens Keek 2011

module load2d

  use plotfile_module
  use filler_module
  use bl_IO_module

  implicit none
  !real(kind=dp_t), allocatable :: c_fab(:,:,:)
  real(8), allocatable :: c_fab(:,:,:) ! F2py needs this declaration, but no longer platform independent.  
  integer :: plo(2), phi(2) ! Indices of lower and upper x,y bounds
  real(8) :: time
  character (len=20), allocatable :: names(:) ! Names of available variables
  integer :: max_available_level

contains
  
  subroutine load(filename, max_level, compname)
    ! filename: plotfile directory name
    ! max_level: AMR level at which we load the data

    implicit none

    integer :: f, i, farg
    integer :: n, nc, ncc, ncs
    integer :: index_store, index
    
    integer, allocatable :: comps(:)
    character (len=64), allocatable :: compnames(:)    
    character (len=64) :: compname
    logical :: comp_error
    
    character(len=256) :: filename
    integer unit
    
    integer :: indslsh, ntime
    integer :: max_level
    
    type(plotfile) :: pf
    
    nc = 1 ! 1 component, as given in compname
    ncc = 0
    ncs = 0
    comp_error = .false.
    
    allocate(comps(nc))
    allocate(compnames(nc))
    
    ncs = ncs + 1
    compnames(ncs) = trim(compname) 
    
    unit = unit_new()      
    call build(pf, filename, unit) ! pf represents the plotfile directory
    
    ! assume that all the files have the same components stored.  Convert
    ! components that were specified via string names into their 
    ! corresponding integer indices in the comps() array.  Don't worry
    ! about double counting for now.
    
    ! the names of the pf%nvars components are stored in pf%names()
    
    index_store = ncc + 1
    do n = 1, ncs
       index = plotfile_var_index(pf, compnames(n))
       if (index>-1) then
          comps(index_store) = index
          index_store = index_store + 1
       else
          if (len_trim(compnames(n))>0) print *, 'ERROR: component = ', compnames(n), ' not found'
          comp_error = .true.
       endif
    enddo
    ! now all nc components are in the comps array
    
    ! Limit max_level to the maximum level available
    max_available_level = plotfile_nlevels(pf)
    if (max_level>max_available_level) then
       print*, 'WARNING: requested max_level not available. Limiting to', max_available_level
    end if
    max_level = min(max_level, max_available_level)
    
    call l2d_clear() ! Free memory
    if (.not.comp_error) then
       ! allocate storage for the data on the finest mesh we are using
       plo = lwb(plotfile_get_pd_box(pf, max_level))
       phi = upb(plotfile_get_pd_box(pf, max_level))
       allocate(c_fab(plo(1):phi(1), plo(2):phi(2), nc))
       
       ! build a single FAB, an then loop over the 
       ! individual components we wish to store, and store them one by one
       call blow_out_to_fab(c_fab, plo, pf, comps, max_level)
    end if
    
    time = plotfile_time(pf)
    allocate(names(plotfile_nvars(pf)))
    names = pf%names
    
    call plotfile_destroy(pf)
    deallocate(comps)
    deallocate(compnames)
    
  end subroutine load
  
  subroutine l2d_clear()
    ! Free memory
    if (allocated(c_fab)) deallocate(c_fab)
    if (allocated(names)) deallocate(names)
  end subroutine l2d_clear

end module load2d
