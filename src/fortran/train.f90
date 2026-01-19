module train
    use cnn_autoencoder
    use cnn_core

    implicit none

    contains
        pure function mse_loss(output, target)
            real, intent(in) :: output(:,:,:)
            real, intent(in) :: target(:,:,:)
            real :: mse_loss

            mse_loss = sum((output - target)**2) / size(output)
        end function

        pure function mse_loss_grad(output, target)
            real, intent(in) :: output(:,:,:)
            real, intent(in) :: target(:,:,:)
            real, allocatable :: mse_loss_grad(:,:,:)

            mse_loss_grad = 2.0 * (output - target) / size(output)
        end function

        pure subroutine sgd_update(layer, learning_rate)
            type(conv_layer), intent(inout) :: layer
            real, intent(in) :: learning_rate

            layer%weights = layer%weights - learning_rate * layer%weights_grad
            layer%bias = layer%bias - learning_rate * layer%bias_grad
        end subroutine

        pure subroutine sgd_update_all(net, learning_rate)
            type(autoencoder), intent(inout) :: net
            real, intent(in) :: learning_rate
            
            integer :: i

            do concurrent (i = 1:net%config%num_layers)
                call sgd_update(net%encoder(i), learning_rate)
                call sgd_update(net%decoder(i), learning_rate)
            end do
        end subroutine

        subroutine train_network(net, images, num_epochs, learning_rate, run_name)
            type(autoencoder), intent(inout) :: net
            real, intent(in) :: images(:,:,:,:)
            integer, intent(in) :: num_epochs
            real, intent(in) :: learning_rate
            character(*), intent(in), optional :: run_name

            integer :: epoch, i
            real, allocatable :: latent(:,:,:), output(:,:,:)
            real, allocatable :: grad_loss(:,:,:)
            real :: total_loss
            character(len=256) :: checkpoint_file

            do epoch = 1, num_epochs
                total_loss = 0.0

                do i = 1, size(images, 1)
                    call autoencoder_forward(net, images(i,:,:,:), latent, output)

                    total_loss = total_loss + mse_loss(output, images(i,:,:,:))
                    grad_loss = mse_loss_grad(output, images(i,:,:,:))
                    call autoencoder_backward(net, output, grad_loss)
                    call sgd_update_all(net, learning_rate)
                end do

                print *, "epoch", epoch, "loss:", total_loss/size(images, 1)

                if (present(run_name)) then
                    write(checkpoint_file, '(A,A,I0,A)') trim(run_name), "_epoch_", epoch, ".bin"
                    call save_weights(net, trim(checkpoint_file))
                end if
            end do
        end subroutine 

end module
