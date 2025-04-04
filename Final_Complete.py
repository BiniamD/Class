def build_cnn_bilstm_model(seq_length, num_features, num_classes=1):
    """
    Build a CNN-BiLSTM model with attention for time series forecasting.
    
    Architecture:
    - Input Layer: (seq_length, num_features)
    - CNN Block: Conv1D(64) + MaxPool + Dropout
    - BiLSTM Block 1: BiLSTM(128) + Dropout
    - BiLSTM Block 2: BiLSTM(32) + Dropout
    - Attention Block: Dense(1) + Softmax + Weighted Sum
    - Output Block: Dense(32) + Dense(1)
    """
    # Input Layer
    inputs = Input(shape=(seq_length, num_features), name='Input_Layer')
    
    # CNN Block
    x = Conv1D(filters=64, kernel_size=3, activation='relu', name='Conv1D')(inputs)
    x = MaxPooling1D(pool_size=2, name='MaxPooling1D')(x)
    x = Dropout(0.2, name='CNN_Dropout')(x)
    
    # BiLSTM Block 1
    x = Bidirectional(LSTM(128, return_sequences=True), name='BiLSTM_1')(x)
    x = Dropout(0.2, name='BiLSTM1_Dropout')(x)
    
    # BiLSTM Block 2
    sequence = Bidirectional(LSTM(32, return_sequences=True), name='BiLSTM_2')(x)
    x = Dropout(0.2, name='BiLSTM2_Dropout')(sequence)

    # Attention Mechanism
    attention_scores = Dense(1, name='Attention_Dense')(sequence)
    attention_weights = Activation('softmax', name='Attention_Softmax')(attention_scores)
    attention_output = Lambda(
        lambda x: tf.reduce_sum(x[0] * tf.expand_dims(tf.squeeze(x[1], -1), -1), axis=1),
        name='Attention_Weighted_Sum'
    )([sequence, attention_weights])

    # Output Block
    x = Dense(32, activation='relu', name='Dense_ReLU')(attention_output)
    outputs = Dense(num_classes, activation='sigmoid', name='Output_Sigmoid')(x)
    
    # Create and compile model
    model = Model(inputs=inputs, outputs=outputs, name='CNN_BiLSTM_Attention')
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model 