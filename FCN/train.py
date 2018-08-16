import Models.VGG16


if __name__ == '__main__':

    input_shape = (224, 224, 3)
    fcn_vgg16 = FCN_VGG16(input_shape, train=True)

    batch_size = 16
    epochs = 250
    lr_base = 0.01 * (float(batch_size) / 16)
