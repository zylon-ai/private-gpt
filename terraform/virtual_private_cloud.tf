module "vpc" {
  source = "git::https://github.com/padok-team/terraform-aws-network.git?ref=v0.1.1"

  vpc_name              = local.vpc_name
  vpc_availability_zone = ["${var.aws_region}a", "${var.aws_region}b"]
  vpc_cidr              = "172.16.0.0/24"
  public_subnet_cidr    = ["172.16.0.0/25"]
  enable_nat_gateway    = var.enable_nat_gateway
  private_subnet_cidr   = ["172.16.0.128/26", "172.16.0.192/26"]
  tags                  = var.additional_tags
}

resource "aws_security_group" "http_communication_inside_vpc" {
  name        = "${terraform.workspace}_${var.api_name}_http_communication_inside_vpc"
  description = "Allow all http communication inside vpc"
  vpc_id      = module.vpc.vpc_id
  tags        = var.additional_tags
}

resource "aws_security_group" "outbounds_https_communication_on_all_cidr_blocks" {
  name        = "${terraform.workspace}_${var.api_name}_outbounds_https_communication_on_all_cidr_blocks"
  description = "Allow outbounds https communication on all cidr blocks"
  vpc_id      = module.vpc.vpc_id
  tags        = var.additional_tags
}

resource "aws_security_group_rule" "https_egress_on_all_cidr_blocks" {
  description       = "allow https outbounds communications on all cidr blocks"
  type              = "egress"
  from_port         = 443
  to_port           = 443
  protocol          = "tcp"
  cidr_blocks       = ["0.0.0.0/0"]
  security_group_id = aws_security_group.outbounds_https_communication_on_all_cidr_blocks.id
}

resource "aws_security_group_rule" "http_ingress" {
  description       = "allow http inbounds communications inside vpc"
  type              = "ingress"
  from_port         = 80
  to_port           = 80
  protocol          = "tcp"
  cidr_blocks       = ["172.16.0.0/24"]
  security_group_id = aws_security_group.http_communication_inside_vpc.id
}

resource "aws_security_group_rule" "http_egress" {
  description       = "allow http outbounds communications inside vpc"
  type              = "egress"
  from_port         = 80
  to_port           = 80
  protocol          = "tcp"
  cidr_blocks       = ["172.16.0.0/24"]
  security_group_id = aws_security_group.http_communication_inside_vpc.id
}
