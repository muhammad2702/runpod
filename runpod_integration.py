import runpod
import os
import json
import time

# Set RunPod API Key
runpod.api_key = os.getenv("RUNPOD_API_KEY")
if not runpod.api_key:
    raise ValueError("RunPod API key not found. Please set 'RUNPOD_API_KEY' as an environment variable.")

def create_runpod_job(input_data_path, output_data_path, model_path="best_short_term_transformer_model.pth"):
    """
    Creates a RunPod job for predictions.

    Args:
        input_data_path (str): Path to the input data (e.g., latest fetched data).
        output_data_path (str): Path to save prediction results.
        model_path (str): Path to the trained model weights.

    Returns:
        str: Job ID if the job is successfully created.
    """
    payload = {
        "input_data_path": input_data_path,
        "output_data_path": output_data_path,
        "model_path": model_path,
    }

    try:
        # Create the RunPod job
        job = runpod.create_job(
            container="your_docker_container_id",  # Replace with your RunPod container ID
            payload=payload,
            timeout_seconds=3600,  # Job timeout
        )
        print(f"RunPod Job Created: {job['id']}")
        return job['id']
    except Exception as e:
        print(f"Error creating RunPod job: {e}")
        return None


def monitor_runpod_job(job_id):
    """
    Monitors the status of a RunPod job until it completes.

    Args:
        job_id (str): The job ID to monitor.

    Returns:
        bool: True if the job completes successfully, False otherwise.
    """
    while True:
        try:
            status = runpod.get_job_status(job_id)
            print(f"Job {job_id} Status: {status['status']}")

            if status['status'] == "COMPLETED":
                return True
            elif status['status'] == "FAILED":
                return False

            # Wait before checking again
            time.sleep(30)

        except Exception as e:
            print(f"Error checking job status: {e}")
            return False


def download_runpod_results(job_id, output_file_path):
    """
    Downloads the results of a completed RunPod job.

    Args:
        job_id (str): The job ID.
        output_file_path (str): Path to save the downloaded results.

    Returns:
        str: Path to the downloaded file.
    """
    try:
        result = runpod.get_job_result(job_id)
        with open(output_file_path, 'w') as f:
            json.dump(result, f)
        print(f"Results downloaded to {output_file_path}")
        return output_file_path
    except Exception as e:
        print(f"Error downloading results: {e}")
        return None


def run_prediction_pipeline(input_data_path, output_data_path, model_path):
    """
    Complete pipeline: create, monitor, and download RunPod job results.

    Args:
        input_data_path (str): Path to input data for prediction.
        output_data_path (str): Path to save prediction results.
        model_path (str): Path to the model file.
    """
    print("Starting RunPod Prediction Pipeline...")

    # Step 1: Create Job
    job_id = create_runpod_job(input_data_path, output_data_path, model_path)
    if not job_id:
        print("Failed to create RunPod job.")
        return

    # Step 2: Monitor Job
    if not monitor_runpod_job(job_id):
        print("RunPod job failed.")
        return

    # Step 3: Download Results
    download_runpod_results(job_id, output_data_path)
    print("RunPod Prediction Pipeline Completed.")


if __name__ == "__main__":
    # Example usage
    INPUT_DATA = "preprocessed_data/latest_data.csv"
    OUTPUT_DATA = "predictions/latest_predictions.json"
    MODEL_FILE = "best_short_term_transformer_model.pth"

    run_prediction_pipeline(INPUT_DATA, OUTPUT_DATA, MODEL_FILE)

