# SciFact Cloud Run demo

Use a dedicated, billing-enabled Google Cloud project. Enable the Cloud Run and Artifact Registry APIs and select an existing runtime service account. The deployer needs permission to manage the service and repository and to act as that runtime identity. Sign in to both the gcloud CLI and Terraform's Application Default Credentials. Check the approved project, region, and run budget before applying.

Terraform creates a dedicated Docker repository and an authenticated CPU service. The service scales to zero and allows one instance per revision. Each revision pins a digest that contains the complete portable release. Store Terraform variables and state locally for the duration of the demo.

## Prepare and deploy

From the repository root, create `infra/terraform/demo.tfvars` with the approved values:

```hcl
project_id              = "your-demo-project"
region                  = "us-central1"
runtime_service_account = "your-runtime-account@your-demo-project.iam.gserviceaccount.com"
invoker                 = "user:you@example.com"
```

Omit `release` on the first apply so Terraform creates only the repository.

```sh
terraform -chdir=infra/terraform init
terraform -chdir=infra/terraform plan -var-file=demo.tfvars -out=bootstrap.tfplan
terraform -chdir=infra/terraform apply bootstrap.tfplan
IMAGE=$(terraform -chdir=infra/terraform output -raw image_repository)
gcloud auth configure-docker "${IMAGE%%/*}"
```

Export the selected, gate-approved model and build the Cloud Run image from the same checkout. Repeat with a second release and separate tag before staging a candidate.

```sh
make export-release EXPORT_ARGS='--model-version 3 --output artifacts/models/scifact-release-3'
docker buildx build --platform linux/amd64 --target cloud \
  --build-context "release=$(pwd -P)/artifacts/models/scifact-release-3" \
  --tag "$IMAGE:champion" --push .
```

Copy the pushed manifest digest from the build output into `demo.tfvars`. Use a new revision suffix whenever the image changes.

```hcl
release = {
  digest   = "sha256:<champion digest>"
  revision = "champion"
}
```

```sh
terraform -chdir=infra/terraform plan -var-file=demo.tfvars -out=service.tfplan
terraform -chdir=infra/terraform apply service.tfplan
gcloud run services proxy scifact-demo --project "$PROJECT_ID" --region "$REGION" --port 8080
```

Set `PROJECT_ID` and `REGION` to the same values as the Terraform inputs before running gcloud. The proxy uses the active gcloud identity. Check `/health/ready` and `/api/v1/models` through localhost and record the expected release, training commit, and data version. Then run the existing [load test](serving.md#local-verification) with `--url http://127.0.0.1:8080/api/v1/search`. Record server p95, client p95, and fallback count.

## Shadow replay, promotion, and rollback

Apply a new release with the candidate digest, `revision = "candidate"`, and this traffic allocation:

```hcl
traffic = [
  { revision = "champion", percent = 100 },
  { revision = "candidate", percent = 0, tag = "shadow" },
]
```

Start a second authenticated proxy with the same command and `--tag shadow --port 8081`. Run the same validation queries through both proxies using `src.monitoring.sentinel.replay` and the prepared manifest:

```sh
uv run --locked --extra dense --extra registry --extra serving python - "$SCIFACT_MANIFEST" <<'PY'
import json
import sys
from dataclasses import asdict
from pathlib import Path
from src.data.prepared import load_validation
from src.monitoring.sentinel import replay

data = load_validation(Path(sys.argv[1]))
for port in (8080, 8081):
    print(json.dumps(asdict(replay(f"http://127.0.0.1:{port}", data))))
PY
```

Record both releases' identities, quality scores, fallback share, and load-test results. The tagged endpoint receives only the labeled replay and test traffic sent to it. After approval, set the candidate to 100% and the champion to 0%, with a tag for the champion if keeping that entry. Keep the candidate digest and revision in the Terraform template when restoring traffic to the champion. Apply the rollback allocation, then verify the default endpoint reports the original champion lineage and returns dense results.

See [Cloud Run traffic and rollback](https://docs.cloud.google.com/run/docs/rollouts-rollbacks-traffic-migration) and [authenticated service proxy](https://docs.cloud.google.com/sdk/gcloud/reference/run/services/proxy).

## Teardown

Stop both local proxies, review the destroy plan, and remove the demo resources.

```sh
terraform -chdir=infra/terraform plan -destroy -var-file=demo.tfvars -out=destroy.tfplan
terraform -chdir=infra/terraform apply destroy.tfplan
terraform -chdir=infra/terraform state list
gcloud run services list --project "$PROJECT_ID" --region "$REGION"
gcloud artifacts repositories list --project "$PROJECT_ID" --location "$REGION"
```

Confirm that the demo service and its repository are absent. The repository owns the demo images. Retain run evidence and keep state and plan files outside Git. Live deployment, replay, rollback, and teardown are acceptance checks for the selected cloud project.
