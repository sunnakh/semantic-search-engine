terraform {
  required_version = ">= 1.7"
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = ">= 7, < 8"
    }
  }
}

variable "project_id" {
  type = string
}

variable "region" {
  type    = string
  default = "us-central1"
}

variable "name" {
  type    = string
  default = "scifact-demo"
}

variable "runtime_service_account" {
  type        = string
  description = "Existing service account email used by the container."
}

variable "invoker" {
  type        = string
  description = "Single IAM member allowed to invoke the demo, such as user:you@example.com."
  validation {
    condition     = can(regex("^(user|serviceAccount):[^ @]+@[^ @]+$", var.invoker))
    error_message = "Use a specific user or service account as the invoker."
  }
}

variable "release" {
  description = "Omit on the first apply to create only the image repository."
  type = object({
    digest   = string
    revision = string
    traffic = optional(list(object({
      revision = string
      percent  = number
      tag      = optional(string)
    })), [])
  })
  default = null
  validation {
    condition = var.release == null ? true : (
      can(regex("^sha256:[0-9a-f]{64}$", var.release.digest)) &&
      can(regex("^[a-z0-9]([a-z0-9-]*[a-z0-9])?$", var.release.revision)) &&
      (length(var.release.traffic) == 0 ? true : (
        sum(var.release.traffic[*].percent) == 100 &&
        alltrue([for entry in var.release.traffic :
          entry.percent >= 0 && entry.percent <= 100 &&
          floor(entry.percent) == entry.percent &&
          (entry.percent > 0 || entry.tag != null)
        ])
      ))
    )
    error_message = "Use a SHA-256 digest, a valid revision suffix, and whole traffic percentages totaling 100. Tag any zero-percent revision."
  }
}

locals {
  repository_url = "${var.region}-docker.pkg.dev/${var.project_id}/${google_artifact_registry_repository.demo.repository_id}"
}

resource "google_artifact_registry_repository" "demo" {
  project       = var.project_id
  location      = var.region
  repository_id = var.name
  format        = "DOCKER"
}

resource "google_cloud_run_v2_service" "search" {
  count               = var.release == null ? 0 : 1
  project             = var.project_id
  location            = var.region
  name                = var.name
  deletion_protection = false

  template {
    revision                         = "${var.name}-${var.release.revision}"
    service_account                  = var.runtime_service_account
    max_instance_request_concurrency = 5
    timeout                          = "15s"

    scaling {
      min_instance_count = 0
      max_instance_count = 1
    }

    containers {
      image = "${local.repository_url}/search@${var.release.digest}"

      ports {
        container_port = 8000
      }

      resources {
        cpu_idle = true
        limits   = { cpu = "2", memory = "2Gi" }
      }

      startup_probe {
        http_get {
          path = "/health/ready"
        }
        period_seconds    = 5
        timeout_seconds   = 4
        failure_threshold = 24
      }
    }
  }

  dynamic "traffic" {
    for_each = var.release.traffic
    content {
      type     = "TRAFFIC_TARGET_ALLOCATION_TYPE_REVISION"
      revision = "${var.name}-${traffic.value.revision}"
      percent  = traffic.value.percent
      tag      = traffic.value.tag
    }
  }
}

resource "google_cloud_run_v2_service_iam_member" "invoker" {
  count    = length(google_cloud_run_v2_service.search)
  project  = var.project_id
  location = var.region
  name     = google_cloud_run_v2_service.search[count.index].name
  role     = "roles/run.invoker"
  member   = var.invoker
}

output "image_repository" {
  value = "${local.repository_url}/search"
}

output "service_url" {
  value = one(google_cloud_run_v2_service.search[*].uri)
}
