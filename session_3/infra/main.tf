# infra/main.tf
#
# The Terraform equivalent of the manual `gcloud` block in the README's section 6.
# Same result — a VM running the ride-api container, reachable on var.app_port —
# but declared once and reproducible, instead of typed into a shell and forgotten.
#
# Manual command                             → resource declared here
# ─────────────────────────────────────────────────────────────────────────────
# gcloud compute instances create ride-api-vm → google_compute_instance.ride_api
# gcloud compute firewall-rules create        → google_compute_firewall.allow_app / allow_ssh_iap
# curl get.docker.com | sh  (on the VM)       → startup-script metadata (startup.sh)
# docker pull && docker run  (on the VM)      → startup-script metadata (startup.sh)
# (nothing — the IP was ephemeral)            → google_compute_address.ride_api
# (nothing — the VM used the default SA)      → google_service_account.ride_api

# ── Static external IP ────────────────────────────────
# The manual flow gave the VM an *ephemeral* IP that changes on every stop/start,
# so the URL you handed out yesterday is dead today. Reserving one fixes that.
resource "google_compute_address" "ride_api" {
  name   = "${var.project_name}-api-ip"
  region = var.gcp_region
}

# ── Service account for the VM ────────────────────────
# The manual VM ran as the *default* compute SA, which is broadly privileged.
# A dedicated SA with only object-read on the bucket is the least-privilege path:
# the container can pull models from GCS and do nothing else.
resource "google_service_account" "ride_api" {
  account_id   = "${var.project_name}-api-vm"
  display_name = "Ride API VM (session 3)"
}

resource "google_project_iam_member" "ride_api_gcs_read" {
  project = var.gcp_project
  role    = "roles/storage.objectViewer"
  member  = "serviceAccount:${google_service_account.ride_api.email}"
}

# ── Firewall: the API port ────────────────────────────
# Tag-scoped, exactly like `--target-tags http-server` in the manual flow: the
# rule applies ONLY to VMs carrying the tag, not to every machine in the network.
resource "google_compute_firewall" "allow_app" {
  name          = "${var.project_name}-allow-${var.app_port}"
  network       = "default"
  description   = "Allow inbound traffic to the ride-api container."
  source_ranges = var.allowed_source_ranges
  target_tags   = ["ride-api"]

  allow {
    protocol = "tcp"
    ports    = [tostring(var.app_port)]
  }
}

# ── Firewall: SSH via IAP only ────────────────────────
# Port 22 is NOT open to the internet. GCP's Identity-Aware Proxy tunnels SSH
# from 35.235.240.0/20, so `gcloud compute ssh --tunnel-through-iap` still works.
resource "google_compute_firewall" "allow_ssh_iap" {
  name          = "${var.project_name}-allow-ssh-iap"
  network       = "default"
  description   = "Allow SSH from GCP's IAP range only — no public port 22."
  source_ranges = var.ssh_source_ranges
  target_tags   = ["ride-api"]

  allow {
    protocol = "tcp"
    ports    = ["22"]
  }
}

# ── The VM ────────────────────────────────────────────
resource "google_compute_instance" "ride_api" {
  name         = "${var.project_name}-api-vm"
  machine_type = var.machine_type
  zone         = var.gcp_zone

  # The tag both firewall rules target. No tag → no traffic reaches this VM,
  # no matter how the rules are written.
  tags = ["ride-api", "http-server"]

  boot_disk {
    initialize_params {
      image = "ubuntu-os-cloud/ubuntu-2204-lts"
      size  = var.boot_disk_size_gb
      type  = "pd-balanced"
    }
  }

  network_interface {
    network = "default"

    # Attaching the reserved IP. Drop this block entirely and the VM becomes
    # private (no public IP) — the right move once a load balancer fronts it.
    access_config {
      nat_ip = google_compute_address.ride_api.address
    }
  }

  service_account {
    email  = google_service_account.ride_api.email
    scopes = ["cloud-platform"] # actual permissions come from the IAM role above
  }

  # Everything the manual flow did *inside* the SSH session — install Docker,
  # pull, run — now runs automatically on first boot.
  metadata = {
    startup-script = templatefile("${path.module}/startup.sh", {
      docker_image   = var.docker_image
      container_name = var.container_name
      app_port       = var.app_port
      model_path     = var.model_path
    })
  }

  # The startup script is the deploy. Changing the image re-runs it on a fresh VM
  # rather than silently leaving the old container running.
  allow_stopping_for_update = true

  labels = {
    project    = var.project_name
    managed_by = "terraform"
    session    = "3"
  }
}
