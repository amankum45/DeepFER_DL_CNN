from src.preprocess import DataPreprocessor

TRAIN_DIR = "dataset/test/train"
TEST_DIR = "dataset/test/train/test/test"

data = DataPreprocessor(
    train_dir=TRAIN_DIR,
    test_dir=TEST_DIR
)

train_gen, val_gen, test_gen = data.get_data_generators()

print("\nClass Labels:")
print(train_gen.class_indices)

print("\nTraining Samples :", train_gen.samples)
print("Validation Samples :", val_gen.samples)
print("Testing Samples :", test_gen.samples)