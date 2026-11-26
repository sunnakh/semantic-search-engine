mock_provider "google" {}

variables {
  project_id              = "scifact-test"
  runtime_service_account = "runtime@scifact-test.iam.gserviceaccount.com"
  invoker                 = "user:operator@example.com"
}

run "bootstrap_creates_only_repository" {
  command = plan

  assert {
    condition     = length(google_cloud_run_v2_service.search) == 0 && length(google_cloud_run_v2_service_iam_member.invoker) == 0
    error_message = "Bootstrap must wait for an image before creating the service."
  }
}

run "stage_candidate_with_zero_traffic" {
  command = plan

  variables {
    release = {
      digest   = "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
      revision = "candidate"
      traffic = [
        { revision = "champion", percent = 100 },
        { revision = "candidate", percent = 0, tag = "shadow" },
      ]
    }
  }

  assert {
    condition     = google_cloud_run_v2_service.search[0].template[0].containers[0].image == "us-central1-docker.pkg.dev/scifact-test/scifact-demo/search@${var.release.digest}"
    error_message = "Cloud Run must use the supplied image digest."
  }

  assert {
    condition = (
      one([for target in google_cloud_run_v2_service.search[0].traffic : target.percent if target.revision == "scifact-demo-champion"]) == 100 &&
      one([for target in google_cloud_run_v2_service.search[0].traffic : target.percent if target.tag == "shadow" && target.revision == "scifact-demo-candidate"]) == 0
    )
    error_message = "The champion must receive all default traffic while the tagged candidate receives zero."
  }
}

run "reject_public_invoker" {
  command = plan

  variables {
    invoker = "allUsers"
  }

  expect_failures = [var.invoker]
}
