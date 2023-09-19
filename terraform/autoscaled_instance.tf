data "aws_ami" "ecs_optimized_ami" {
  owners = ["amazon"]

  filter {
    name   = "name"
    values = ["amzn2-ami-ecs-hvm-2.0.20230301-x86_64-ebs"] # AMI optimized for ECS, created on 01/03/2023
  }
  tags = var.additional_tags
}

resource "aws_launch_configuration" "this" {
  name                 = "${terraform.workspace}_${var.api_name}_launch_config"
  image_id             = data.aws_ami.ecs_optimized_ami.id
  iam_instance_profile = aws_iam_instance_profile.this.arn
  security_groups = [
    aws_security_group.http_communication_inside_vpc.id,

    # https communication on all cidr blocks is enabled to allow ECS to find and register the instance, as
    # ECS is not inside VPC. PS: use a NAT gateway to improve security.
    aws_security_group.outbounds_https_communication_on_all_cidr_blocks.id
  ]

  instance_type               = var.ec2_instance_type
  associate_public_ip_address = false

  # The following line of code allows to specify a config inside the EC2 to make sure the instance is launched in
  # the correct ECS cluster
  user_data = "#!/bin/bash\necho ECS_CLUSTER=${local.ecs_cluster_name} >> /etc/ecs/ecs.config"
}

resource "aws_autoscaling_group" "this" {
  name                 = "${terraform.workspace}_${var.api_name}_autoscaling_group"
  vpc_zone_identifier  = module.vpc.private_subnets_ids
  launch_configuration = aws_launch_configuration.this.name

  min_size                  = 1
  max_size                  = 2
  desired_capacity          = 1
  health_check_grace_period = 180
  health_check_type         = "EC2"
  default_cooldown          = 200 # Slightly higher than the health check grace period
}

resource "aws_appautoscaling_target" "this" {
  max_capacity       = 2
  min_capacity       = 1
  resource_id        = "service/${aws_ecs_cluster.this.name}/${aws_ecs_service.this.name}"
  scalable_dimension = "ecs:service:DesiredCount"
  service_namespace  = "ecs"
}

resource "aws_appautoscaling_policy" "model_api_cpu_tracking" {
  name               = "${terraform.workspace}_${var.api_name}_application_scaling_policy_cpu"
  policy_type        = "TargetTrackingScaling"
  resource_id        = aws_appautoscaling_target.this.resource_id
  scalable_dimension = aws_appautoscaling_target.this.scalable_dimension
  service_namespace  = aws_appautoscaling_target.this.service_namespace

  target_tracking_scaling_policy_configuration {
    predefined_metric_specification {
      predefined_metric_type = "ECSServiceAverageCPUUtilization"
    }
    target_value       = 80
    scale_in_cooldown  = 60
    scale_out_cooldown = 60
  }
  depends_on = [aws_appautoscaling_target.this]
}

resource "aws_appautoscaling_policy" "model_api_memory_tracking" {
  name               = "${terraform.workspace}_${var.api_name}_application_scaling_policy_memory"
  policy_type        = "TargetTrackingScaling"
  resource_id        = aws_appautoscaling_target.this.resource_id
  scalable_dimension = aws_appautoscaling_target.this.scalable_dimension
  service_namespace  = aws_appautoscaling_target.this.service_namespace

  target_tracking_scaling_policy_configuration {
    predefined_metric_specification {
      predefined_metric_type = "ECSServiceAverageMemoryUtilization"
    }
    target_value       = 80
    scale_in_cooldown  = 60
    scale_out_cooldown = 60
  }
  depends_on = [aws_appautoscaling_target.this]
}
