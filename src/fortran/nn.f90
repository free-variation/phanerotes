module nn

    contains
        pure function relu_forward(x)
            real, intent(in) :: x(:,:,:)
            real, allocatable :: relu_forward(:,:,:)

            relu_forward = max(0.0, x)
        end function

        pure function relu_backward(x, grad_out)
            real, intent(in) :: x(:,:,:), grad_out(:,:,:)
            real, allocatable :: relu_backward(:,:,:)

            allocate(relu_backward, mold = x)
            where (x > 0.0)
                relu_backward = grad_out
            elsewhere
                relu_backward = 0.0
            end where
        end function

        pure function sigmoid_forward(x)
            real, intent(in) :: x(:,:,:)
            real, allocatable :: sigmoid_forward(:,:,:)

            sigmoid_forward = 1.0 / (1.0 + exp(-x))
        end function

        pure function sigmoid_backward(y, grad_out)
            real, intent(in) :: y(:,:,:), grad_out(:,:,:)
            real, allocatable :: sigmoid_backward(:,:,:)

            sigmoid_backward = grad_out * y * (1.0 - y)
        end function


end module
