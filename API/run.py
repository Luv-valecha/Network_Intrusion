import os
import subprocess

def run_preprocessing():
    """Run the preprocessing script."""
    print("Running preprocessing...")
    preprocess_script = os.path.join("API", "scripts", "preprocess.py")
    if os.path.exists(preprocess_script):
        subprocess.run(["python", preprocess_script], check=True)
    else:
        print(f"Preprocessing script not found at {preprocess_script}")
        exit(1)

def run_app():
    """Run the Flask app with Waitress."""
    print("Starting the Flask app with Waitress...")
    subprocess.run(["python", "-m", "waitress", "--host=0.0.0.0", "--port=5000", "API.wsgi:app"], check=True)

if __name__ == "__main__":
    run_preprocessing()
    run_app()