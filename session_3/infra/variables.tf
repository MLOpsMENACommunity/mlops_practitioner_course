# infra/variables.tf — every knob this stack exposes.

variable "project_name" {
  description = "Unique project identifier used in resource names."
  type        = string
  default     = "ride-duration"
}

variable "gcp_project" {
  description = "GCP project ID to create resources in. No default — set it in terraform.tfvars."
  type        = string
}

variable "gcp_region" {
  description = "GCP region for regional resources (static IP)."
  type        = string
  default     = "us-central1"
}

variable "gcp_zone" {
  description = "GCP zone for the VM. Must be inside gcp_region."
  type        = string
  default     = "us-central1-a"
}

# ── The VM ─────────────────────────────────────────────
variable "machine_type" {
  description = "VM size. e2-small (2 vCPU shared, 2 GB) is enough for the demo API."
  type        = string
  default     = "e2-small"
}

variable "boot_disk_size_gb" {
  description = "Boot disk size in GB. Docker images eat space; 20 GB is a safe floor."
  type        = number
  default     = 20
}

# ── The container to run ───────────────────────────────
variable "docker_image" {
  description = "Fully-qualified image the VM pulls and runs, e.g. yourusername/ride-api:latest."
  type        = string
}

variable "container_name" {
  description = "Name given to the running Docker container (used by stop/rm on redeploy)."
  type        = string
  default     = "ride-api"
}

variable "app_port" {
  description = "Port the container listens on AND the port opened in the firewall. Keep them in sync."
  type        = number
  default     = 8000
}

variable "model_path" {
  description = "Value of MODEL_PATH passed into the container."
  type        = string
  default     = "/models/v1/model.pkl"
}

# ── Access control ─────────────────────────────────────
variable "allowed_source_ranges" {
  description = <<-EOT
    CIDRs allowed to reach the API port. Defaults to the whole internet, which is
    fine for a classroom demo and wrong for anything else — narrow it to your own
    IP (e.g. ["203.0.113.4/32"]) as soon as this holds a real model.
  EOT
  type        = list(string)
  default     = ["0.0.0.0/0"]
}

variable "ssh_source_ranges" {
  description = "CIDRs allowed to SSH. 35.235.240.0/20 is GCP's IAP range — SSH via `gcloud compute ssh` without exposing port 22 to the internet."
  type        = list(string)
  default     = ["35.235.240.0/20"]
}
