from tensorflow.keras import layers, Model

def conv_block(x, filters):
    x = layers.Conv2D(
        filters, 3, padding="same",
        kernel_initializer="he_normal"
    )(x)
    x = layers.Activation("relu")(x)

    x = layers.Conv2D(
        filters, 3, padding="same",
        kernel_initializer="he_normal"
    )(x)
    x = layers.Activation("relu")(x)
    return x


def build_unet():
    inputs = layers.Input(shape=(None, None, 3))

    # Encoder
    c1 = conv_block(inputs, 32)
    p1 = layers.MaxPooling2D(2)(c1)

    c2 = conv_block(p1, 64)
    p2 = layers.MaxPooling2D(2)(c2)

    c3 = conv_block(p2, 128)
    p3 = layers.MaxPooling2D(2)(c3)

    # Bottleneck
    b = conv_block(p3, 256)

    # Decoder
    u3 = layers.UpSampling2D(2)(b)
    u3 = layers.Concatenate()([u3, c3])
    d3 = conv_block(u3, 128)

    u2 = layers.UpSampling2D(2)(d3)
    u2 = layers.Concatenate()([u2, c2])
    d2 = conv_block(u2, 64)

    u1 = layers.UpSampling2D(2)(d2)
    u1 = layers.Concatenate()([u1, c1])
    d1 = conv_block(u1, 32)

    # เรียนรู้ส่วนแก้ไขเพิ่มเติมจาก Input
    residual = layers.Conv2D(
        3, 3, padding="same"
    )(d1)

    restored = layers.Add()([inputs, residual])

    # จำกัด Output ให้อยู่ในช่วง 0-1
    outputs = layers.ReLU(max_value=1.0)(restored)

    return Model(inputs, outputs, name="Residual_UNet")
