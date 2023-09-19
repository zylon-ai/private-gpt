variable "additional_tags" {
  description = "Additional resource tags"
  type        = map(string)
  default = {
    terraform = "true"
  }
}

variable "aws_region" {
  description = "Region in which the resources are provisioned"
  type        = string
  default     = "us-east-1"
}

variable "api_name" {
  description = "Name appended to resources specific to the infrastructure of the api"
  type        = string
  default     = "api"
}

variable "ec2_instance_type" {
  description = "Type of the aws EC2 instance to use in the api"
  type        = string
  default     = "t2.medium"
}

variable "ec2_memory_reserved" {
  description = "The amount of memory (in MiB) to reserve for the api container inside the EC2. Warning: you must select a value below the RAM capacity of the chosen instance"
  type        = number
  default     = 3500
}

variable "enable_nat_gateway" {
  description = "Should be true if you want to provision NAT Gateways for your VPC to allow your EC2 instance to access the public internet"
  type        = bool
  default     = "true"
}
