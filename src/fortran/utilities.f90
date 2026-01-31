module utilities
    implicit none

    contains
        pure function split_string(s, delim)
            character(*), intent(in) :: s, delim
            character(256), allocatable :: split_string(:)

            integer :: num_strings, i, start, pos, delim_len, s_len

            s_len = len_trim(s)
            delim_len = len(delim)

            if (s_len == 0) then
                allocate(split_string(0))
                return
            end if

            ! Count delimiters to determine array size
            num_strings = 1
            i = 1
            do while (i <= s_len - delim_len + 1)
                if (s(i:i+delim_len-1) == delim) then
                    num_strings = num_strings + 1
                    i = i + delim_len
                else
                    i = i + 1
                end if
            end do

            allocate(split_string(num_strings))

            ! Extract substrings
            start = 1
            pos = 1
            i = 1
            do while (i <= s_len - delim_len + 1)
                if (s(i:i+delim_len-1) == delim) then
                    if (i > start) then
                        split_string(pos) = s(start:i-1)
                    else
                        split_string(pos) = ""
                    end if
                    pos = pos + 1
                    i = i + delim_len
                    start = i
                else
                    i = i + 1
                end if
            end do

            ! Last segment
            if (start <= s_len) then
                split_string(pos) = s(start:s_len)
            else
                split_string(pos) = ""
            end if
        end function

        function run_command(command)
            use stdlib_system, only: run, process_type
            character(*), intent(in) :: command
            character(256), allocatable :: run_command(:)

            type(process_type) :: p
            character(1) :: nl

            nl = char(10)
            p = run(command, want_stdout=.true.)
            run_command = split_string(trim(p%stdout), nl)
        end function

        function directory_files(dir, glob)
            character(*), intent(in) :: dir
            character(*), intent(in), optional :: glob
            character(256), allocatable :: directory_files(:)

            character(256), allocatable :: raw(:)
            integer :: i, slash_pos

            if (present(glob)) then
                raw = run_command("sh -c ""find " // trim(dir) // " -maxdepth 1 -iname '" // trim(glob) // "' -type f | sort""")

                ! strip directory prefix to get just filenames
                allocate(directory_files(size(raw)))
                do i = 1, size(raw)
                    slash_pos = index(raw(i), '/', back=.true.)
                    if (slash_pos > 0) then
                        directory_files(i) = raw(i)(slash_pos+1:)
                    else
                        directory_files(i) = raw(i)
                    end if
                end do
            else
                directory_files = run_command("ls " // trim(dir))
            end if
        end function
end module
