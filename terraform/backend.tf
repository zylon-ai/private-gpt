terraform {
  backend "s3" {
    bucket  = "private-gpt-terraform-backend"
    region  = "us-east-1"
    key     = "state.tfstate"
    encrypt = true
  }
}
