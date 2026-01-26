program phanerotes
    use :: interpreter
    implicit none

    character(4096) :: line
    logical :: running
    integer :: iostat, unit_num

    running = .true.

    ! Load prelude
    open(newunit=unit_num, file="lib/words.phan", status="old", iostat=iostat)
    if (iostat == 0) then
        do while (running)
            read(unit_num, '(A)', iostat=iostat) line
            if (iostat /= 0) exit
            call execute_line(line, running)
        end do
        close(unit_num)
    end if

    ! REPL
    do while (running)
        read(*, '(A)', iostat=iostat) line
        if (iostat /= 0) exit
        call execute_line(line, running)
    end do
end program
