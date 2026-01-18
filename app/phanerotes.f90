program phanerotes
    use :: interpreter
    implicit none

    character(4096) :: line
    logical :: running
    integer :: iostat

    running = .true.
    do while (running)
        read(*, '(A)', iostat=iostat) line
        if (iostat /= 0) exit
        call execute_line(line, running)
    end do
end program
