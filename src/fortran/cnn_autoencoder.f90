module cnn_autoencoder
    use :: cnn_core
    use :: nn

    implicit none

    type :: autoencoder_config
        integer :: input_channels   ! e.g. 3 for RGB, 1 for greyscale
        integer :: num_layers       
        integer :: base_channels    ! channels after first layer
        integer :: max_channels     ! channel cap (e.g. 512)
        integer :: kernel_width     ! typically 3
        integer :: kernel_height    ! typically 3
        integer :: stride           ! typically 2 for encoder
        integer :: padding          ! typically 1 for 3x3 kernel
        logical :: concatenate      ! whether skip connections are concatenated or summed
    end type

    type :: tensor_cache
        real, allocatable :: tensor(:, :,:,:)
    end type

    type :: dropout_cache
        real, allocatable :: dropout(:)
    end type

    type :: autoencoder
        type(autoencoder_config) :: config
        type(conv_layer), allocatable :: encoder(:)
        type(conv_layer), allocatable :: decoder(:)
        type(conv_layer), allocatable :: skip_projection(:)

        type(tensor_cache), allocatable :: encoder_preact(:)
        type(tensor_cache), allocatable :: decoder_preact(:)
        type(dropout_cache), allocatable :: skip_dropout_cache(:)
    end type

    contains
        function init_layer(config, in_channels, out_channels, stride) result(layer)
            type(autoencoder_config), intent(in) :: config
            integer, intent(in) :: in_channels, out_channels, stride
            type(conv_layer) :: layer
            real :: scale

            layer%in_channels = in_channels
            layer%out_channels = out_channels
            layer%kernel_width = config%kernel_width
            layer%kernel_height = config%kernel_height
            layer%stride = stride
            layer%padding = config%padding

            allocate(layer%weights(out_channels, in_channels, layer%kernel_width, layer%kernel_height))
            allocate(layer%bias(out_channels))
            allocate(layer%weights_grad, mold = layer%weights)
            allocate(layer%bias_grad, mold = layer%bias)

            layer%bias = 0.0
            layer%weights_grad = 0.0
            layer%bias_grad = 0.0

            scale = sqrt(2.0 / (layer%in_channels * layer%kernel_width * layer%kernel_height))
            call random_number(layer%weights)
            layer%weights = (layer%weights - 0.5) * 2.0 * scale

            layer%training = .true.
        end function


        function autoencoder_init(config) result(net)
            type(autoencoder_config), intent(in) :: config
            type(autoencoder) :: net

            integer :: i, in_channels, out_channels
            type(autoencoder_config) :: config_1x1

            net%config = config

            allocate(net%encoder(config%num_layers))
            allocate(net%decoder(config%num_layers))
            allocate(net%encoder_preact(config%num_layers))
            allocate(net%decoder_preact(config%num_layers))
            
            allocate(net%skip_projection(config%num_layers - 1))
            allocate(net%skip_dropout_cache(config%num_layers - 1))

            do i = 1, config%num_layers
                if (i == 1) then
                    in_channels = config%input_channels
                else
                    in_channels = min(config%base_channels * 2**(i - 1), config%max_channels)
                end if
                out_channels = min(config%base_channels * 2**i, config%max_channels)
                
                net%encoder(i) = init_layer(config, in_channels, out_channels, config%stride)
            end do

            do i = 1, config%num_layers
                in_channels = net%encoder(config%num_layers - i + 1)%out_channels
                if (config%concatenate .and. i < config%num_layers) then
                    in_channels = in_channels + net%encoder(config%num_layers - i)%out_channels
                end if
                out_channels = net%encoder(config%num_layers - i + 1)%in_channels

                net%decoder(i) = init_layer(config, in_channels, out_channels, 1)
            end do

            config_1x1 = config
            config_1x1%kernel_width = 1
            config_1x1%kernel_height = 1
            config_1x1%padding = 0

            do i = 1, config%num_layers - 1
                in_channels = net%encoder(config%num_layers - i)%out_channels
                out_channels = net%decoder(i)%in_channels

                net%skip_projection(i) = init_layer(config_1x1, in_channels, out_channels, 1)
                allocate(net%skip_dropout_cache(i)%dropout(in_channels))
            end do
        end function

        subroutine set_training(net, training)
           type(autoencoder), intent(inout) ::net
           logical, intent(in) :: training 

           integer :: i

           do i = 1, net%config%num_layers
               net%encoder(i)%training = training
               net%decoder(i)%training = training
           end do

           do i = 1, net%config%num_layers - 1
               net%skip_projection(i)%training = training
           end do
       end subroutine

       pure function concatenate_channels(a, b)
           ! Layout: (channels, height, width, batch)
           real, intent(in) :: a(:,:,:,:), b(:,:,:,:)
           real, allocatable :: concatenate_channels(:,:,:,:)
           integer :: num_channels1, num_channels2

           num_channels1 = size(a, 1)
           num_channels2 = size(b, 1)

           allocate(concatenate_channels(num_channels1 + num_channels2, size(a, 2), size(a, 3), size(a, 4)))
           concatenate_channels(1:num_channels1, :, :, :) = a
           concatenate_channels(num_channels1 + 1:num_channels1 + num_channels2, :, :, :) = b
       end function

       subroutine dropout_channels(a, dropout_rate, dropped_a, dropped_cache)
           ! Layout: (channels, height, width, batch)
           real, intent(in) :: a(:,:,:,:)
           real, intent(in) :: dropout_rate
           real, intent(out), allocatable :: dropped_a(:,:,:,:)
           type(dropout_cache) :: dropped_cache

           real :: rand
           integer :: c, num_channels
           real :: scale

           num_channels = size(a, 1)

           dropped_a = a
           if (dropout_rate == 0.0) then
               dropped_cache%dropout = 1.0
               return
           end if

           scale = 1.0 / (1.0 - dropout_rate)

           do c = 1, num_channels
               call random_number(rand)
               if (rand < dropout_rate) then
                   dropped_a(c, :, :, :) = 0.0
                   dropped_cache%dropout(c) = 0.0
               else
                   dropped_a(c, :, :, :) = dropped_a(c, :, :, :) * scale
                   dropped_cache%dropout(c) = scale
               end if
           end do
       end subroutine

       subroutine autoencoder_forward(net, input, dropout_rate, latent, output, encoder_acts_out)
           type(autoencoder), intent(inout) :: net
           real, intent(in) :: dropout_rate
           real, intent(in) :: input(:, :,:,:)
           real, allocatable, intent(out) :: latent(:, :,:,:)
           real, allocatable, intent(out) :: output(:, :,:,:)
           type(tensor_cache), allocatable, intent(out), optional :: encoder_acts_out(:)

           real, allocatable :: layer_input(:, :,:,:), layer_output(:, :,:,:)
           type(tensor_cache), allocatable :: encoder_activations(:)
           real, allocatable :: skip_input(:, :,:,:), skip_output(:, :,:,:), aggregated_input(:, :,:,:)
           integer :: i

           allocate(encoder_activations(net%config%num_layers - 1))

           layer_input = input
           do i = 1, net%config%num_layers
               call conv_forward(net%encoder(i), layer_input, layer_output)
               if (net%encoder(i)%training) net%encoder_preact(i)%tensor = layer_output

               layer_output = relu_forward(layer_output)
               if (i < net%config%num_layers) encoder_activations(i)%tensor = layer_output

               layer_input = layer_output
           end do

           latent = layer_output
           if (present(encoder_acts_out)) encoder_acts_out = encoder_activations
           layer_input = layer_output

           do i = 1, net%config%num_layers - 1
               layer_input = upsample(layer_input, net%config%stride)

               if (dropout_rate > 0.0) then
                   call dropout_channels(encoder_activations(net%config%num_layers - i)%tensor, dropout_rate, &
                       skip_input, net%skip_dropout_cache(i))
               else
                   skip_input = encoder_activations(net%config%num_layers - i)%tensor
               end if

               if (net%config%concatenate) then
                   aggregated_input = concatenate_channels(layer_input, skip_input)
               else
                   call conv_forward(net%skip_projection(i), skip_input, skip_output)
                   aggregated_input = layer_input + skip_output
               end if

               call conv_forward(net%decoder(i), aggregated_input, layer_output)
               if (net%decoder(i)%training) net%decoder_preact(i)%tensor = layer_output
               layer_output = relu_forward(layer_output)

               layer_input = layer_output
           end do

           layer_input = upsample(layer_input, net%config%stride)
           call conv_forward(net%decoder(net%config%num_layers), layer_input, output)
           output = sigmoid_forward(output)
       end subroutine

       subroutine encoder_forward(net, input, latent)
           type(autoencoder), intent(inout) :: net
           real, intent(in) :: input(:,:,:,:)
           real, allocatable, intent(out) :: latent(:,:,:,:)

           real, allocatable :: layer_input(:,:,:,:), layer_output(:,:,:,:)
           integer :: i

           layer_input = input
           do i = 1, net%config%num_layers
               call conv_forward(net%encoder(i), layer_input, layer_output)
               layer_output = relu_forward(layer_output)
               layer_input = layer_output
           end do

           latent = layer_output
       end subroutine

       subroutine autoencoder_backward(net, output, grad_loss)
           ! Layout: (channels, height, width, batch)
           type(autoencoder), intent(inout) :: net
           real, intent(in) :: output(:,:,:,:)
           real, intent(in) :: grad_loss(:,:,:,:)

           real, allocatable :: grad_input(:,:,:,:), grad_output(:,:,:,:), skip_grad(:,:,:,:)
           type(tensor_cache), allocatable :: skip_gradients(:)
           integer :: i, c, num_skip_channels

           allocate(skip_gradients(net%config%num_layers - 1))

           grad_output = sigmoid_backward(output, grad_loss)
           call conv_backward(net%decoder(net%config%num_layers), grad_output, grad_input)
           do i = net%config%num_layers - 1, 1, -1
               grad_output = upsample_backward(grad_input, net%config%stride)
               grad_output = relu_backward(net%decoder_preact(i)%tensor, grad_output)

               call conv_backward(net%decoder(i), grad_output, grad_input)

               if (net%config%concatenate) then
                   num_skip_channels = net%encoder(net%config%num_layers - i)%out_channels

                   ! Channels is dimension 1
                   skip_gradients(i)%tensor = grad_input(size(grad_input, 1) - num_skip_channels + 1:, :, :, :)
                   do c = 1, size(skip_gradients(i)%tensor, 1)
                       skip_gradients(i)%tensor(c, :, :, :) = skip_gradients(i)%tensor(c, :, :, :) * &
                           net%skip_dropout_cache(i)%dropout(c)
                   end do

                   grad_input = grad_input(1:size(grad_input, 1) - num_skip_channels, :, :, :)
               else
                   call conv_backward(net%skip_projection(i), grad_input, skip_grad)

                   skip_gradients(i)%tensor = skip_grad
                   do c = 1, size(skip_gradients(i)%tensor, 1)
                       skip_gradients(i)%tensor(c, :, :, :) = skip_gradients(i)%tensor(c, :, :, :) * &
                           net%skip_dropout_cache(i)%dropout(c)
                   end do
               end if
           end do

           grad_input = upsample_backward(grad_input, net%config%stride)

           do i = net%config%num_layers, 1, -1
               if (i < net%config%num_layers) then
                   grad_input = grad_input + skip_gradients(net%config%num_layers - i)%tensor
               end if
               grad_output = relu_backward(net%encoder_preact(i)%tensor, grad_input)
               call conv_backward(net%encoder(i), grad_output, grad_input)
           end do
       end subroutine

       subroutine save_autoencoder(net, filename)
           type(autoencoder), intent(in) :: net
           character(*), intent(in) :: filename

           integer :: i, unit, concat_int

           open(newunit=unit, file = filename, form = "unformatted", access = "stream")

           ! Write config
           write(unit) net%config%input_channels
           write(unit) net%config%num_layers
           write(unit) net%config%base_channels
           write(unit) net%config%max_channels
           write(unit) net%config%kernel_width
           write(unit) net%config%kernel_height
           write(unit) net%config%stride
           write(unit) net%config%padding
           concat_int = merge(1, 0, net%config%concatenate)
           write(unit) concat_int

           ! Write weights
           do i = 1, net%config%num_layers
               write(unit) net%encoder(i)%weights
               write(unit) net%encoder(i)%bias
           end do

           do i = 1, net%config%num_layers
               write(unit) net%decoder(i)%weights
               write(unit) net%decoder(i)%bias
           end do

           do i = 1, net%config%num_layers - 1
               write(unit) net%skip_projection(i)%weights
               write(unit) net%skip_projection(i)%bias
           end do

           close(unit)
       end subroutine

       function load_autoencoder(filename) result(net)
           character(*), intent(in) :: filename
           type(autoencoder) :: net

           type(autoencoder_config) :: config
           integer :: i, unit, concat_int

           open(newunit=unit, file=filename, form="unformatted", access="stream", status="old")

           ! Read config
           read(unit) config%input_channels
           read(unit) config%num_layers
           read(unit) config%base_channels
           read(unit) config%max_channels
           read(unit) config%kernel_width
           read(unit) config%kernel_height
           read(unit) config%stride
           read(unit) config%padding
           read(unit) concat_int
           config%concatenate = (concat_int == 1)

           ! Initialize network with config
           net = autoencoder_init(config)

           ! Read weights
           do i = 1, net%config%num_layers
               read(unit) net%encoder(i)%weights
               read(unit) net%encoder(i)%bias
           end do

           do i = 1, net%config%num_layers
               read(unit) net%decoder(i)%weights
               read(unit) net%decoder(i)%bias
           end do

           do i = 1, net%config%num_layers - 1
               read(unit) net%skip_projection(i)%weights
               read(unit) net%skip_projection(i)%bias
           end do

           close(unit)
       end function

       subroutine decode_latent(net, latent, output)
           ! Layout: (channels, height, width, batch)
           type(autoencoder), intent(inout) :: net
           real, intent(in) :: latent(:,:,:,:)
           real, allocatable, intent(out) :: output(:,:,:,:)

           real, allocatable :: layer_input(:,:,:,:), layer_output(:,:,:,:), padded_input(:,:,:,:)
           integer :: i, batch_size, decoder_channels, skip_channels, width, height

           layer_input = latent
           do i = 1, net%config%num_layers - 1
               layer_input = upsample(layer_input, net%config%stride)

               if (net%config%concatenate) then
                   decoder_channels = size(layer_input, 1)
                   height = size(layer_input, 2)
                   width = size(layer_input, 3)
                   batch_size = size(layer_input, 4)
                   skip_channels = net%encoder(net%config%num_layers - i)%out_channels

                   allocate(padded_input(decoder_channels + skip_channels, height, width, batch_size))
                   padded_input(1:decoder_channels, :, :, :) = layer_input
                   padded_input(decoder_channels + 1:, :, :, :) = 0.0

                   call conv_forward(net%decoder(i), padded_input, layer_output)
                   deallocate(padded_input)
               else
                   call conv_forward(net%decoder(i), layer_input, layer_output)
               end if

               layer_output = relu_forward(layer_output)
               layer_input = layer_output
           end do

           layer_input = upsample(layer_input, net%config%stride)
           call conv_forward(net%decoder(net%config%num_layers), layer_input, output)
           output = sigmoid_forward(output)
       end subroutine

       subroutine decode_latent_interpolated(net, latent_a, latent_b, &
               encoder_acts_a, encoder_acts_b, alpha, output)
           type(autoencoder), intent(inout) :: net
           real, intent(in) :: latent_a(:, :,:,:), latent_b(:, :,:,:)
           type(tensor_cache), intent(in) :: encoder_acts_a(:), encoder_acts_b(:)
           real, intent(in) :: alpha
           real, allocatable, intent(out) :: output(:, :,:,:)

           real, allocatable :: layer_input(:, :,:,:), layer_output(:, :,:,:)
           real, allocatable :: skip_interp(:, :,:,:), skip_projected(:, :,:,:)
           real, allocatable :: aggregated_input(:, :,:,:)
           integer :: i, skip_idx

           layer_input = alpha * latent_a + (1.0 - alpha) * latent_b

           do i = 1, net%config%num_layers - 1
               layer_input = upsample(layer_input, net%config%stride)

               skip_idx = net%config%num_layers - i
               skip_interp = alpha * encoder_acts_a(skip_idx)%tensor + &
                   (1.0 - alpha) * encoder_acts_b(skip_idx)%tensor

               if (net%config%concatenate) then
                   aggregated_input = concatenate_channels(layer_input, skip_interp)
               else
                   call conv_forward(net%skip_projection(i), skip_interp, skip_projected)
                   aggregated_input = layer_input + skip_projected
               end if

               call conv_forward(net%decoder(i), aggregated_input, layer_output)
               layer_output = relu_forward(layer_output)
               layer_input = layer_output
           end do

           layer_input = upsample(layer_input, net%config%stride)
           call conv_forward(net%decoder(net%config%num_layers), layer_input, output)
           output = sigmoid_forward(output)
       end subroutine

   end module
