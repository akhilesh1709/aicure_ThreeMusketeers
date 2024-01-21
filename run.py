import sys
import joblib
import pandas as pd

def load_model(model_filename):
    try:
        model = joblib.load(model_filename)
        return model
    except Exception as e:
        print(f"Error loading the model from {model_filename}: {e}")
        sys.exit(1)

def load_test_data(csv_filename):
    try:
        test_data = pd.read_csv(csv_filename)
        # Extract 'uuid' column for later use
        uuid_column = test_data['uuid']
        # Drop unnecessary columns, including 'uuid'
        columns_to_drop = ['datasetId', 'condition', 'uuid']
        test_data = test_data.drop(columns=columns_to_drop)
        return test_data, uuid_column
    except Exception as e:
        print(f"Error loading the test data from {csv_filename}: {e}")
        sys.exit(1)

def make_predictions(model, test_data):
    predictions = model.predict(test_data)
    return predictions

def save_predictions_csv(uuid_column, predictions, output_csv_filename):
    # Save predictions and UUIDs to a new CSV file
    result_df = pd.DataFrame({
        'uuid': uuid_column,  # Include 'uuid' column in the output
        'HR': predictions
    })
    result_df.to_csv(output_csv_filename, index=False)
    print(f"Predictions saved to {output_csv_filename}")

def main():
    # Check if the correct number of command-line arguments is provided
    if len(sys.argv) != 3:
        print("Usage: python run.py <model_filename> <test_csv_filename>")
        sys.exit(1)

    # Extract command-line arguments
    model_filename = sys.argv[1]
    csv_filename = sys.argv[2]

    output_csv_filename = 'results.csv'

    model = load_model(model_filename)

    # Load the test data and extract 'uuid' column
    test_data, uuid_column = load_test_data(csv_filename)

    # Make predictions using the loaded model
    predictions = make_predictions(model, test_data)

    # Save predictions to a new CSV file
    save_predictions_csv(uuid_column, predictions, output_csv_filename)

if __name__ == "__main__":
    main()
