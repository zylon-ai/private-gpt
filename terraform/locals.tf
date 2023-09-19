locals {
  ecs_cluster_name = "${terraform.workspace}_${var.api_name}_cluster"
  vpc_name         = "${terraform.workspace}_${var.api_name}_application_vpc"
}
