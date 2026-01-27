program phanerotes
    use :: interpreter
    implicit none

    character(4096) :: line
    logical :: quit
    integer :: iostat, unit_num

    quit = .false.

    ! Load base dictionary
    open(newunit=unit_num, file="lib/words.phan", status="old", iostat=iostat)
    if (iostat == 0) then
        do while (.not. quit)
            read(unit_num, '(A)', iostat=iostat) line
            if (iostat /= 0) exit
            call execute_line(line, quit)
        end do
        close(unit_num)
    end if

    ! REPL
    do while (.not. quit)
        read(*, '(A)', iostat=iostat) line
        if (iostat /= 0) exit
        call execute_line(line, quit)
    end do
end program
