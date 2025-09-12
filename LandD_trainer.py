import os
import argparse
import tensorflow as tf
from dataset import LandD_Dataset
from models.Mymodels import CNN_model

def main(args):
    # Dataset loader
    dataset_loader = LandD_Dataset(
        data_dir=args.data_dir,
        inp_shape=(args.img_height, args.img_width, args.img_channels),
        batch_size=args.batch_size,
        test_split=args.test_split,
        seed=args.seed
    )
    train_ds, val_ds = dataset_loader.get_datasets()
    num_classes = len(dataset_loader.class_names)
    print("Number classes:", num_classes, dataset_loader.class_names)

    # Build model từ file model/
    model = CNN_model((args.img_height, args.img_width, args.img_channels), num_classes)
    model.summary()

    # Compile
    model.compile(loss='sparse_categorical_crossentropy', metrics=['accuracy'], learning_rate=args.learning_rate)

    # Create checkpoint + log
    chkpt_dir = os.path.join(args.base_dir, "chkpt", args.name)
    log_dir = os.path.join(args.base_dir, "logs", args.name)
    os.makedirs(chkpt_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # Callback checkpoint
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(chkpt_dir, "epoch-{epoch:03d}-val_loss-{val_loss:.3f}.weights.h5"),
        monitor="val_loss",
        save_best_only=True,
        mode="min",
        verbose=1,
        save_weights_only=True
    )

    # TensorBoard log
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)

    # Train
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=[checkpoint_callback, tensorboard_callback]
    )

    # Save last model
    final_model_path = os.path.join(args.base_dir+'/models', "LandD_last_model.h5")
    model.save(final_model_path)
    print(f"Model saved in {final_model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Data args
    parser.add_argument("--data_dir", type=str, default="./data/characters",
                        help="Path to input dir (36 folder)")
    parser.add_argument("--img_height", type=int, default=64,
                        help="Height of the input image")
    parser.add_argument("--img_width", type=int, default=64,
                        help="Width of the input image")
    parser.add_argument("--img_channels", type=int, default=1,
                        help="(1 = grayscale, 3 = RGB)")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size")
    parser.add_argument("--test_split", type=float, default=0.125,
                        help="Val test split ratio")
    parser.add_argument("--seed", type=int, default=123,
                        help="Seed to split data")

    # Train args
    parser.add_argument("-lr", "--learning_rate", type=float, default=1.5e-4,
                        help="Learning rate")
    parser.add_argument("--epochs", type=int, default=25,
                        help="Number of epochs")
    parser.add_argument("--base_dir", type=str, default=".",
                        help="Base path checkpoint + log")
    parser.add_argument("-n", "--name", type=str, required=True,
                        help="Run name (checkpoint + log folder)")

    args = parser.parse_args()
    main(args)
