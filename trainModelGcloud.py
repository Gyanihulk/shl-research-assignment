from google.cloud import aiplatform

# Initialize Vertex AI environment with the staging bucket
aiplatform.init(
    project='bhagwasanatantimes',
    location='us-central1',
    staging_bucket='gs://review-stage-bucket'  # Replace with your staging bucket
)

# Define the custom training job
job = aiplatform.CustomTrainingJob(
    display_name='gpt2-review-creation',
    script_path='train.py',  # Path to your training script
    container_uri='us-docker.pkg.dev/vertex-ai/training/tf-gpu.2-6:latest',  # Pre-built container with dependencies
    model_serving_container_image_uri='us-docker.pkg.dev/vertex-ai/prediction/tf2-cpu.2-6:latest'
)

# Run the training job
job.run(
    replica_count=1,
    model_display_name='gpt2-review-creation',
    machine_type='n1-standard-4',  # Standard machine for training
    accelerator_type='NVIDIA_TESLA_T4',  # Using T4 GPU for faster training
    accelerator_count=1,
    args=['--bucket-name', 'gs://review-creation/dataset']  # Pass necessary arguments (your dataset bucket)
)
