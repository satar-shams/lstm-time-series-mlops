from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, Callback

def create_callbacks()-> list[Callback]:
    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=10,
        restore_best_weights=True,
        verbose=0,
    )
    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        verbose=1,
    )
    return [early_stopping, reduce_lr]