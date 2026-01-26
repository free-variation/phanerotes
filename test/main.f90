program tester
    use, intrinsic :: iso_fortran_env, only: error_unit
    use testdrive, only: run_testsuite, new_testsuite, testsuite_type
    use test_command, only: collect_command_tests
    use test_image, only: collect_image_tests
    use test_interpreter, only: collect_interpreter_tests
    use test_utilities, only: collect_utilities_tests
    implicit none

    integer :: stat, is
    type(testsuite_type), allocatable :: testsuites(:)
    character(len=*), parameter :: fmt = '("#", *(1x, a))'

    stat = 0

    testsuites = [ &
        new_testsuite("command", collect_command_tests), &
        new_testsuite("image", collect_image_tests), &
        new_testsuite("interpreter", collect_interpreter_tests), &
        new_testsuite("utilities", collect_utilities_tests) &
    ]

    do is = 1, size(testsuites)
        write(error_unit, fmt) "Testing:", testsuites(is)%name
        call run_testsuite(testsuites(is)%collect, error_unit, stat)
    end do

    if (stat > 0) then
        write(error_unit, '(i0, 1x, a)') stat, "test(s) failed!"
        error stop
    end if

end program tester
