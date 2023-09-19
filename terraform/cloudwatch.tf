resource "aws_cloudwatch_log_group" "this" {
  name = "ecs/${terraform.workspace}_${var.api_name}_logs"
}
