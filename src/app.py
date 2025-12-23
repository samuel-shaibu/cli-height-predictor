import click
import joblib
import numpy as np
import os

# Load model at the top level
# We use os.path to ensure we find the file regardless of where the command is run
MODEL_PATH = os.path.join(os.path.dirname(__file__), '../model.joblib')

@click.command()
@click.option('--weight', prompt='Weight(lbs)', help='The person\'s weight in lbs.')
def predict(weight):
    """Simple CLI to predict height based on weight."""

    # Load the model
    try:
        model = joblib.load(MODEL_PATH)
    except FileNotFoundError:
        click.echo("Error: Model file not found. Did you run train_model.py?")
        return

    # Process input (reshape because sklearn expects 2D array for features)
    input_data = np.array([[float(weight)]])

    # Predict
    prediction = model.predict(input_data)[0]

    # Output the result
    click.echo(f"Predicted Height: {prediction:.2f} inches")


if __name__ == '__main__':
    predict()