# infra/main.tf

terraform {
  required_providers {
    # aws    = { source = "hashicorp/aws", version = "~> 5.0" }
    google = { source = "hashicorp/google", version = "~> 5.0" }
  }
}

# provider "aws" { region = "us-east-1" }

provider "google" {
  project = var.gcp_project
  region  = var.gcp_region
}

# ── S3 bucket for DVC remote storage ──────────────────
# resource "aws_s3_bucket" "dvc_remote" {
#   bucket = "mlops-dvc-artifacts-${var.project_name}"
#   tags   = { Project = var.project_name, ManagedBy = "terraform" }
# }

# ── ECR repository for Docker images ──────────────────
# resource "aws_ecr_repository" "api" {
#   name                 = "${var.project_name}/ride-api"
#   image_tag_mutability = "MUTABLE"

#   image_scanning_configuration { scan_on_push = true }
# }

# ── GCS bucket for MLflow + DVC remote storage ────────
resource "google_storage_bucket" "mlops_artifacts" {
  name     = "mlops-artifacts-${var.project_name}"
  location = var.gcp_region

  uniform_bucket_level_access = true
  force_destroy               = false

  versioning { enabled = true }

  labels = {
    project    = var.project_name
    managed_by = "terraform"
  }
}

# ── Outputs ───────────────────────────────────────────
# output "s3_bucket_name" { value = aws_s3_bucket.dvc_remote.bucket }
# output "ecr_repo_url" { value = aws_ecr_repository.api.repository_url }
output "gcs_bucket_name" { value = google_storage_bucket.mlops_artifacts.name }
output "gcs_bucket_url" { value = google_storage_bucket.mlops_artifacts.url }
