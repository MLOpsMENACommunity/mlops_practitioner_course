# infra/versions.tf — provider requirements & configuration.

terraform {
  required_version = ">= 1.5"

  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
  }

  # Remote state. Uncomment once the bucket from session 2 exists — without it,
  # terraform.tfstate lives on your laptop and nobody else can safely `apply`.
  # backend "gcs" {
  #   bucket = "mlops-artifacts-ride-duration"
  #   prefix = "session_3/infra"
  # }
}

provider "google" {
  project = var.gcp_project
  region  = var.gcp_region
  zone    = var.gcp_zone
}
