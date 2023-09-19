resource "aws_lb" "this" {
  name               = "${terraform.workspace}ApiAlb" # load balancer name must only contain alphanumeric (i.e. no "_")
  internal           = true
  load_balancer_type = "application"
  security_groups    = [aws_security_group.http_communication_inside_vpc.id]
  subnets            = module.vpc.private_subnets_ids
  tags               = var.additional_tags
}

resource "aws_lb_target_group" "this" {
  name        = "${terraform.workspace}ApiGroup" # target group name must only contain alphanumeric (i.e. no "_")
  target_type = "instance"
  port        = 80
  protocol    = "HTTP"
  vpc_id      = module.vpc.vpc_id
  health_check {
    enabled = true
    path    = "/health"
  }
  tags = var.additional_tags
}

resource "aws_lb_listener" "this" {
  load_balancer_arn = aws_lb.this.arn
  port              = 80
  protocol          = "HTTP"
  tags              = var.additional_tags

  default_action {
    target_group_arn = aws_lb_target_group.this.arn
    type             = "forward"
  }
}
