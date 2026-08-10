# Standard Terraform workflow

# 1. Initialise — download providers, setup backend
terraform init

# 2. Format — auto-format all .tf files
terraform fmt

# 3. Validate — check syntax without calling AWS
terraform validate

# 4. Plan — preview what will change (read-only)
terraform plan -out=tfplan

# 5. Apply — create/update resources
terraform apply tfplan

# 6. Show outputs (e.g. to get ECR URL for CI)
terraform output ecr_repo_url

# 7. Destroy — tear everything down (be careful!)
terraform destroy
