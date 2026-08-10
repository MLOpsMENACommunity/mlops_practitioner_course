# infra/outputs.tf — what you need after `terraform apply`.
#
# These replace the manual `gcloud compute instances describe ... --format='get(...)'`
# incantation from the README: the IP is a first-class value, not something you
# grep out of a describe.

output "vm_name" {
  description = "Name of the VM instance."
  value       = google_compute_instance.ride_api.name
}

output "external_ip" {
  description = "Reserved static IP of the API. Stable across VM stop/start."
  value       = google_compute_address.ride_api.address
}

output "api_url" {
  description = "Base URL of the deployed API."
  value       = "http://${google_compute_address.ride_api.address}:${var.app_port}"
}

output "health_check" {
  description = "Copy-paste command to verify the deployment."
  value       = "curl http://${google_compute_address.ride_api.address}:${var.app_port}/health"
}

output "ssh_command" {
  description = "SSH into the VM (through IAP — port 22 is not public)."
  value       = "gcloud compute ssh ${google_compute_instance.ride_api.name} --zone ${var.gcp_zone} --tunnel-through-iap"
}

output "service_account_email" {
  description = "Least-privilege SA the VM runs as."
  value       = google_service_account.ride_api.email
}
