module command
    use :: image
    use :: utilities

    implicit none
    
    integer, parameter :: MAX_STACK = 1000
    integer, parameter :: MAX_STRING_LENGTH = 256

    type image_entry 
        real, allocatable :: pixels(:,:,:)
    end type
    
    type(image_entry):: image_stack(MAX_STACK)
    real :: number_stack(MAX_STACK)
    character(MAX_STRING_LENGTH) :: string_stack(MAX_STACK)

    integer :: image_stack_top = 0
    integer :: number_stack_top = 0
    integer :: string_stack_top = 0

    contains

        ! ---------- Stack management ----------
        subroutine push_string(s)
            character(*), intent(in) :: s

            if (string_stack_top >= MAX_STACK) error stop "string stack overflow"

            string_stack_top = string_stack_top + 1
            string_stack(string_stack_top) = s
        end subroutine push_string
        
        subroutine push_number(n)
            real, intent(in) :: n

            if (number_stack_top >= MAX_STACK) error stop "number stack overflow"

            number_stack_top = number_stack_top + 1
            number_stack(number_stack_top) = n
        end subroutine push_number

        subroutine push_image(pixels)
            real, intent(in) :: pixels(:,:,:)

            if (image_stack_top >= MAX_STACK) error stop "image stack overflow"            

            image_stack_top = image_stack_top + 1
            image_stack(image_stack_top)%pixels = pixels
        end subroutine push_image

        function pop_string()
            character(MAX_STRING_LENGTH) :: pop_string

            if (string_stack_top == 0) error stop "string stack underflow"
            
            pop_string = string_stack(string_stack_top)
            string_stack_top = string_stack_top - 1
        end function
        
        function pop_number()
            real :: pop_number

            if (number_stack_top == 0) error stop "number stack underflow"
            
            pop_number = number_stack(number_stack_top)
            number_stack_top = number_stack_top - 1
        end function

        function peek_number()
            real :: peek_number

            if (number_stack_top == 0) error stop "number stack underflow"
            
            peek_number = number_stack(number_stack_top)
        end function

 
        function pop_image() result(pixels)
            real, allocatable :: pixels(:,:,:)
            
            if (image_stack_top == 0) error stop "image stack underflow"
            
            pixels = image_stack(image_stack_top)%pixels
            deallocate(image_stack(image_stack_top)%pixels)
            image_stack_top = image_stack_top - 1
        end function

        ! ---------- Stack words ----------
        subroutine dup_image()
            real, allocatable :: pixels(:, :, :) 
            
            pixels = pop_image()
            call push_image(pixels)
            call push_image(pixels)
        end subroutine dup_image

        subroutine drop_image()
            real, allocatable :: discard(:,:,:)

            discard = pop_image()
        end subroutine drop_image

        subroutine swap_image() 
            real, allocatable :: pixels1(:,:,:), pixels2(:,:,:)

            pixels1 = pop_image()
            pixels2 = pop_image()
            call push_image(pixels1)
            call push_image(pixels2)
        end subroutine swap_image

        subroutine over_image()
            real, allocatable :: pixels1(:,:,:), pixels2(:,:,:)

            pixels1 = pop_image()
            pixels2 = pop_image()
            call push_image(pixels2)
            call push_image(pixels1)
            call push_image(pixels2)
        end subroutine over_image

        subroutine dot()
            real :: number

            number = pop_number()
            write(*, '(G0, A)', advance='no') number, ' '
        end subroutine

        subroutine sdot()
            character(MAX_STRING_LENGTH) :: s

            s = pop_string()
            write(*, '(A, A)', advance='no') trim(s), ' '
        end subroutine
        
        ! ---------- Utility words ----------
        subroutine list_files()
            character(MAX_STRING_LENGTH) :: dir
            character(MAX_STRING_LENGTH), allocatable :: filenames(:)
            integer :: i

            dir = pop_string()
            filenames = directory_files(dir)

            do i = 1, size(filenames)
                call push_string(filenames(i))
            end do

            call push_number(real(size(filenames)))
        end subroutine


        ! ---------- Image manipulation words ----------
        subroutine load() 
            character(MAX_STRING_LENGTH) :: filename
            real, allocatable :: pixels(:,:,:)

            filename = pop_string()
            pixels = load_image(filename)
            call push_image(pixels)
        end subroutine

        subroutine save() 
            character(MAX_STRING_LENGTH) :: filename
            real, allocatable :: pixels(:,:,:)
            logical :: success

            filename = pop_string()
            pixels = pop_image()
            
            call save_image(filename, pixels, success)
            if (.not. success) error stop "failed to save image"
        end subroutine

        subroutine transpose()
            real, allocatable :: pixels(:,:,:), transposed_image(:,:,:)

            pixels = pop_image()
            transposed_image = transpose_image(pixels)
            call push_image(transposed_image)
        end subroutine

        subroutine fliph()
            real, allocatable :: pixels(:,:,:), flipped_image(:,:,:)

            pixels = pop_image()
            flipped_image = flip_image_horizontal(pixels)
            call push_image(flipped_image)
        end subroutine

        subroutine flipv()
            real, allocatable :: pixels(:,:,:), flipped_image(:,:,:)

            pixels = pop_image()
            flipped_image = flip_image_vertical(pixels)
            call push_image(flipped_image)
        end subroutine

        subroutine transform()
            real, allocatable :: pixels(:,:,:), transformed_pixels(:,:,:)
            real :: cx, cy
            real :: angle 
            real :: sx, sy
            real :: tx, ty
            real :: rot(3,3), trans(3,3), scale(3,3), M(3,3)

            ty = pop_number()
            tx = pop_number()
            sy = pop_number()
            sx = pop_number()
            angle = pop_number()
            cy = pop_number()
            cx = pop_number()

            pixels = pop_image()
            
            rot = rotation_matrix(cx, cy, angle)
            trans = translation_matrix(tx, ty)
            scale = scale_matrix(cx, cy, sx, sy)
            M = matmul(trans, matmul(rot, scale))

            transformed_pixels = affine_transform(pixels, M)
            call push_image(transformed_pixels)
        end subroutine

        subroutine split()
            real, allocatable :: pixels(:,:,:)
            integer :: num_channels

            pixels = pop_image()
            num_channels = size(pixels, 1)
            call push_number(real(num_channels))

            ! a greyscale image can be pushed right back as is
            if (num_channels == 1) then
                call push_image(pixels)
                return
            end if

            call push_image(pixels(1:1,:,:))
            call push_image(pixels(2:2,:,:))
            call push_image(pixels(3:3,:,:))

            ! check for alpha channel
            if (num_channels == 4) call push_image(pixels(4:4,:,:))

        end subroutine

        subroutine merge()
            integer :: num_channels
            real, allocatable :: pixels(:,:,:)
            real, allocatable :: red(:,:,:), green(:,:,:), blue(:,:,:), alpha(:,:,:)

            num_channels = int(pop_number())
            if (.not. any(num_channels == [1, 3, 4])) &
                error stop "failed to merge: number of channels must be 1, 3, or 4"

            if (num_channels == 1) return

            if (num_channels == 4) then
                alpha = pop_image()
                if (size(alpha, 1) /= 1) error stop "failed to merge: alpha not greyscale"
            end if
            blue = pop_image()
            green = pop_image()
            red = pop_image()

            if (size(red, 1) /= 1 .or. size(green, 1) /= 1 .or. size(blue, 1) /= 1) &
                error stop "failed to merge: component images not all greyscale"

            if (any(shape(red) /= shape(green)) .or. any(shape(red) /= shape(blue))) &
                error stop "failed to merge: dimensions mismatch"

            if (num_channels == 4 .and. any(shape(red) /= shape(alpha))) &
                error stop "failed to merge: alpha dimensions mismatch"

            allocate(pixels(num_channels, size(red, 2), size(red, 3)))
            pixels(1,:,:) = red(1,:,:)
            pixels(2,:,:) = green(1,:,:)
            pixels(3,:,:) = blue(1,:,:)
            if (num_channels == 4) pixels(4,:,:) = alpha(1,:,:)

            call push_image(pixels)
        end subroutine

        subroutine fill()
            real, allocatable :: pixels(:,:,:)
            integer :: num_channels

            pixels = pop_image()
            num_channels = size(pixels, 1)

            if (num_channels == 4) then
                pixels(4, :, :) = pop_number()
            end if 

            if (num_channels >= 3) then
                pixels(3, :, :) = pop_number()
                pixels(2, :, :) = pop_number()
            end if

            pixels(1, :, :) = pop_number()

            call push_image(pixels)
        end subroutine

        subroutine interpolate()
            real, allocatable :: pixels1(:,:,:), pixels2(:,:,:)
            real, allocatable :: interpolated_pixels(:,:,:)
            real :: alpha

            alpha = pop_number()
            pixels1 = pop_image()
            pixels2 = pop_image()

            interpolated_pixels = alpha*pixels1 + (1 - alpha) * pixels2

            call push_image(interpolated_pixels)
        end subroutine

        subroutine interpolate_frames()
            real, allocatable :: pixels1(:,:,:), pixels2(:,:,:)
            real, allocatable :: interpolated_pixels(:,:,:)
            integer :: i, num_frames
            real :: alpha

            num_frames = pop_number() - 1
            pixels1 = pop_image()
            if (num_frames <= 0) then
                call push_image(pixels1)
                return
            end if

            pixels2 = pop_image()

            do i = 0, num_frames
                alpha = real(i) / real(num_frames)
                interpolated_pixels = alpha*pixels1 + (1 - alpha) * pixels2
                call push_image(interpolated_pixels)
            end do
        end subroutine


        ! ---------- Create movies ----------

end module
