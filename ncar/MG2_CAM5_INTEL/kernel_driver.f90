
! KGEN-generated Fortran source file
!
! Filename    : kernel_driver.f90
! Generated at: 2015-03-30 21:09:53
! KGEN version: 0.4.5


PROGRAM kernel_driver
    USE micro_mg_cam, ONLY : micro_mg_cam_tend
    USE shr_kind_mod, ONLY: r8 => shr_kind_r8
    USE micro_mg_cam, ONLY : kgen_read_externs_micro_mg_cam, watch
    USE micro_mg_utils, ONLY : kgen_read_externs_micro_mg_utils
    USE micro_mg2_0, ONLY : kgen_read_externs_micro_mg2_0
    USE wv_sat_methods, ONLY : kgen_read_externs_wv_sat_methods

    IMPLICIT NONE

    INTEGER :: kgen_mpi_rank
    CHARACTER(LEN=16) ::kgen_mpi_rank_conv
    INTEGER, DIMENSION(3), PARAMETER :: kgen_mpi_rank_at = (/ 0, 100, 300 /)
    INTEGER :: kgen_ierr, kgen_unit
    INTEGER :: kgen_repeat_counter
    INTEGER :: kgen_counter
    CHARACTER(LEN=16) :: kgen_counter_conv
    INTEGER, DIMENSION(3), PARAMETER :: kgen_counter_at = (/ 10, 100, 50 /)
    CHARACTER(LEN=1024) :: kgen_filepath
    REAL(KIND=r8) :: dtime
#ifdef QFPD
    type(watch) :: watchv
    character(128) :: temp
    character(128) :: input_set
    !we're passing VAR INDX1 INDX2 DATASET
    CALL GET_COMMAND_ARGUMENT(1, watchv%var)
    CALL GET_COMMAND_ARGUMENT(2, temp)
    READ( temp, '(i10)') watchv%indx1
    CALL GET_COMMAND_ARGUMENT(3, temp)
    READ( temp, '(i10)') watchv%indx2
    CALL GET_COMMAND_ARGUMENT(4, input_set)
#endif

    
#ifndef QFPD
#ifdef QFPC
    WRITE (*,*) "["
#endif
    DO kgen_repeat_counter = 0, 8
        kgen_counter = kgen_counter_at(mod(kgen_repeat_counter, 3)+1)
        WRITE( kgen_counter_conv, * ) kgen_counter
        kgen_mpi_rank = kgen_mpi_rank_at(mod(kgen_repeat_counter, 3)+1)
        WRITE( kgen_mpi_rank_conv, * ) kgen_mpi_rank
        CALL GETCWD(kgen_filepath)
        kgen_filepath = trim(kgen_filepath) // "/micro_mg_tend2_0." // trim(adjustl(kgen_counter_conv)) // "." // trim(adjustl(kgen_mpi_rank_conv))
#else
        kgen_filepath = input_set
#endif
        kgen_unit = kgen_get_newunit()
        OPEN (UNIT=kgen_unit, FILE=kgen_filepath, STATUS="OLD", ACCESS="STREAM", FORM="UNFORMATTED", ACTION="READ", IOSTAT=kgen_ierr, CONVERT="BIG_ENDIAN")
#ifndef QFPC
        WRITE (*,*)
#endif
        IF ( kgen_ierr /= 0 ) THEN
            CALL kgen_error_stop( "FILE OPEN ERROR: " // trim(adjustl(kgen_filepath)) )
         END IF
#ifndef QFPC         
        WRITE (*,*)
        WRITE (*,*) "************ Verification against '" // trim(adjustl(kgen_filepath)) // "' ************"
#endif
! #else
!         WRITE (*,*)
!         write (*,*) trim(adjustl(kgen_filepath))
! #endif
            CALL kgen_read_externs_micro_mg_cam(kgen_unit)
            CALL kgen_read_externs_micro_mg_utils(kgen_unit)
            CALL kgen_read_externs_micro_mg2_0(kgen_unit)
            CALL kgen_read_externs_wv_sat_methods(kgen_unit)

            ! driver variables
            READ(UNIT=kgen_unit) dtime
#ifdef QFPD
            call micro_mg_cam_tend(dtime, kgen_unit, kgen_filepath, watchv)
#else
            call micro_mg_cam_tend(dtime, kgen_unit, kgen_filepath)
#endif
            CLOSE (UNIT=kgen_unit)
#ifndef QFPD
         END DO
#ifdef QFPC
         WRITE(*,*) "]"
#endif
#endif
    CONTAINS

        ! write subroutines
        ! No subroutines
        FUNCTION kgen_get_newunit() RESULT(new_unit)
           INTEGER, PARAMETER :: UNIT_MIN=100, UNIT_MAX=1000000
           LOGICAL :: is_opened
           INTEGER :: nunit, new_unit, counter
        
           new_unit = -1
           DO counter=UNIT_MIN, UNIT_MAX
               inquire(UNIT=counter, OPENED=is_opened)
               IF (.NOT. is_opened) THEN
                   new_unit = counter
                   EXIT
               END IF
           END DO
        END FUNCTION
        
        SUBROUTINE kgen_error_stop( msg )
            IMPLICIT NONE
            CHARACTER(LEN=*), INTENT(IN) :: msg
        
            WRITE (*,*) msg
            STOP 1
        END SUBROUTINE 


    END PROGRAM kernel_driver
