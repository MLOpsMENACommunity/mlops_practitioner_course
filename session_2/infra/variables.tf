# infra/variables.tf
variable "project_name" {
  description = "Unique project identifier used in resource names"
  type        = string
  default     = "ride-duration"
}

variable "gcp_project" {
  description = "GCP project ID to create resources in"
  type        = string
}

variable "gcp_region" {
  description = "GCP region/location for the GCS bucket"
  type        = string
  default     = "us-central1"
}
