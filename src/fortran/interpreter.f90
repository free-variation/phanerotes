module interpreter
    use :: command
    implicit none

contains

    subroutine execute_line(line, running)
        character(*), intent(in) :: line
        logical, intent(inout) :: running
        character(MAX_STRING_LENGTH) :: token
        integer :: pos

        pos = 1
        do while (pos <= len_trim(line))
            call next_token(line, pos, token)
            if (len_trim(token) == 0) exit

            call dispatch(token, running)
            if (.not. running) exit
        end do
    end subroutine

    subroutine next_token(line, pos, token)
        character(*), intent(in) :: line
        integer, intent(inout) :: pos
        character(*), intent(out) :: token
        integer :: start, line_len

        token = ""
        line_len = len_trim(line)

        ! skip whitespace
        do while (pos <= line_len)
            if (line(pos:pos) /= ' ') exit
            pos = pos + 1
        end do

        if (pos > line_len) return

        ! comment - rest of line ignored
        if (line(pos:pos) == "#") return

        ! string literal
        if (line(pos:pos) == '"') then
            pos = pos + 1
            start = pos
            do while (pos <= line_len)
                if (line(pos:pos) == '"') exit
                pos = pos + 1
            end do

            token = '"' // line(start:pos-1)
            if (pos <= line_len) pos = pos + 1
            return
        end if

        ! regular token
        start = pos
        do while (pos <= line_len)
            if (line(pos:pos) == ' ') exit
            pos = pos + 1
        end do
        token = line(start:pos-1)
    end subroutine

    subroutine dispatch(token, running)
        character(*), intent(in) :: token
        logical, intent(inout) :: running
        real :: num
        integer :: iostat

        if (len_trim(token) == 0) return

        ! string literal
        if (token(1:1) == '"') then
            call push_string(token(2:))
            return
        end if

        ! try as a number
        read(token, *, iostat = iostat) num
        if (iostat == 0) then
            call push_number(num)
            return
        end if

        ! words
        select case (trim(token))
            case ("quit")
                running = .false.
            case ("dup")
                call dup_image()
            case ("drop")
                call drop_image()
            case ("swap")
                call swap_image()
            case ("over")
                call over_image()
            case ("load")
                call load()
            case ("save")
                call save()
            case ("transpose")
                call transpose()
            case ("fliph")
                call fliph()
            case ("flipv")
                call flipv()
            case ("transform")
                call transform()
            case ("split")
                call split()
            case ("merge")
                call merge()
            case default
                print *, "unknown word: ", trim(token)
        end select
    end subroutine

end module interpreter
