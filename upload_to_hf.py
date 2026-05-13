import os
from huggingface_hub import HfApi, login

def upload_model():
    print("=== Hugging Face Model Uploader ===")
    
    # Prompt for HF token
    token = input("1. Enter your Hugging Face Write Token (get it from https://huggingface.co/settings/tokens): ").strip()
    if not token:
        print("Error: Token cannot be empty.")
        return
        
    try:
        login(token=token)
    except Exception as e:
        print(f"Failed to login: {e}")
        return

    # Prompt for Model Name
    repo_id = input("2. Enter the repository name you want to create (e.g., your-username/mental-health-distilbert): ").strip()
    if not repo_id:
        print("Error: Repository name cannot be empty.")
        return

    api = HfApi()
    
    print(f"\nCreating repository '{repo_id}' on Hugging Face...")
    try:
        api.create_repo(repo_id=repo_id, exist_ok=True)
    except Exception as e:
        print(f"Failed to create repository: {e}")
        return

    print("Uploading model files. This might take a few minutes depending on your internet connection...")
    try:
        api.upload_folder(
            folder_path="models/distilbert",
            repo_id=repo_id,
            repo_type="model"
        )
        print("\n✅ Upload complete!")
        print(f"Your model is now hosted at: https://huggingface.co/{repo_id}")
        print("\nPlease share the repository name with Antigravity so app.py can be updated!")
    except Exception as e:
        print(f"Upload failed: {e}")

if __name__ == "__main__":
    # Ensure huggingface_hub is installed
    try:
        import huggingface_hub
    except ImportError:
        import subprocess
        import sys
        print("Installing huggingface_hub...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "huggingface_hub"])
        from huggingface_hub import HfApi, login
        
    upload_model()
