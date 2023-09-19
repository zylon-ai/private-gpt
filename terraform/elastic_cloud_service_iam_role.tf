data "aws_iam_policy_document" "this" {
  statement {
    actions = ["sts:AssumeRole"]

    principals {
      identifiers = ["ec2.amazonaws.com"]
      type        = "Service"
    }
  }
}

resource "aws_iam_role" "this" {
  assume_role_policy = data.aws_iam_policy_document.this.json
  name               = "${terraform.workspace}EC2ForECSRole"
  tags               = var.additional_tags
}

resource "aws_iam_instance_profile" "this" {
  name = "${terraform.workspace}EC2ForECSProfile"
  role = aws_iam_role.this.name
  tags = var.additional_tags

}

# Add policy to access ECS and ECR
resource "aws_iam_role_policy_attachment" "this" {
  role       = aws_iam_role.this.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonEC2ContainerServiceforEC2Role"
}
