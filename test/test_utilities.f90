module test_utilities
    use testdrive, only: new_unittest, unittest_type, error_type, check
    use utilities
    implicit none
    private

    public :: collect_utilities_tests

contains

    subroutine collect_utilities_tests(testsuite)
        type(unittest_type), allocatable, intent(out) :: testsuite(:)

        testsuite = [ &
            new_unittest("directory_files_returns_files", test_directory_files_returns_files), &
            new_unittest("directory_files_count", test_directory_files_count), &
            new_unittest("split_basic", test_split_basic), &
            new_unittest("split_newline", test_split_newline), &
            new_unittest("split_empty", test_split_empty) &
        ]
    end subroutine


    subroutine test_directory_files_returns_files(error)
        type(error_type), allocatable, intent(out) :: error
        character(256), allocatable :: files(:)

        ! Use the test directory itself - we know it has files
        files = directory_files("test")

        call check(error, size(files) > 0, "should find files in test directory")
    end subroutine


    subroutine test_directory_files_count(error)
        type(error_type), allocatable, intent(out) :: error
        character(256), allocatable :: files(:)
        logical :: found_main
        integer :: i

        files = directory_files("test")

        ! We know main.f90 exists in test/
        found_main = .false.
        do i = 1, size(files)
            if (trim(files(i)) == "main.f90") found_main = .true.
        end do

        call check(error, found_main, "should find main.f90 in test directory")
    end subroutine


    subroutine test_split_basic(error)
        type(error_type), allocatable, intent(out) :: error
        character(256), allocatable :: parts(:)

        parts = split_string("one,two,three", ",")

        call check(error, size(parts) == 3, "should have 3 parts")
        if (allocated(error)) return
        call check(error, trim(parts(1)) == "one", "first part should be 'one'")
        if (allocated(error)) return
        call check(error, trim(parts(2)) == "two", "second part should be 'two'")
        if (allocated(error)) return
        call check(error, trim(parts(3)) == "three", "third part should be 'three'")
    end subroutine


    subroutine test_split_newline(error)
        type(error_type), allocatable, intent(out) :: error
        character(256), allocatable :: parts(:)
        character(1) :: nl

        nl = char(10)
        parts = split_string("file1.txt" // nl // "file2.txt" // nl // "file3.txt", nl)

        call check(error, size(parts) == 3, "should have 3 lines")
        if (allocated(error)) return
        call check(error, trim(parts(1)) == "file1.txt", "first line")
        if (allocated(error)) return
        call check(error, trim(parts(2)) == "file2.txt", "second line")
    end subroutine


    subroutine test_split_empty(error)
        type(error_type), allocatable, intent(out) :: error
        character(256), allocatable :: parts(:)

        parts = split_string("", ",")

        call check(error, size(parts) == 0, "empty string should return empty array")
    end subroutine

end module test_utilities
