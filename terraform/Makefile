# Updates the documentation of the README with the resources of the terraform files
docs:
	terraform-docs markdown table --output-file README.md --output-mode inject .

# Checks that the section of the documentation of the README with the resources is up-to-date
docs-check:
	terraform-docs markdown table --output-file README.md --output-check .

# Format the files
fmt:
	terraform fmt

# Check that the formatting is correct
fmt-check:
	terraform fmt -check

# Validate the syntax is valid
validate:
	terraform validate

# Validate files follow the correct practices and conventions
lint:
	tflint
