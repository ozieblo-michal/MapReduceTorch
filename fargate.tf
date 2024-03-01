resource "aws_ecs_cluster" "fargate_cluster" {
  name = "data-formatting-cluster"
}

resource "aws_iam_role" "ecs_task_execution_role" {
  name = "ecs_task_execution_role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17",
    Statement = [
      {
        Action = "sts:AssumeRole",
        Principal = {
          Service = "ecs-tasks.amazonaws.com"
        },
        Effect = "Allow",
        Sid    = ""
      },
    ]
  })

  tags = {
    Environment = "Development"
    Project     = "Data Formatting"
  }

}

resource "aws_iam_role_policy_attachment" "ecs_task_execution_role_policy_attachment" {
  role       = aws_iam_role.ecs_task_execution_role.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
}

resource "aws_ecs_task_definition" "data_formatting_task" {
  family                   = "data-formatting-task"
  cpu                      = "256"
  memory                   = "512"
  network_mode             = "awsvpc"
  requires_compatibilities = ["FARGATE"]
  execution_role_arn       = aws_iam_role.ecs_task_execution_role.arn

  container_definitions = jsonencode([
  {
    name      = "data-formatting-container",
    image     = "your_ecr_repository_uri/your_image:tag",
    cpu       = 256,
    memory    = 512,
    essential = true,
    command   = ["poetry", "run", "python", "src/main.py"],
    environment = [
        {
          name  = "S3_BUCKET_PATH",
          value = "s3://shared-bucket-for-emr-and-fargate/processed-data/"
        }
      ],
    logConfiguration = {
      logDriver = "awslogs",
      options = {
        "awslogs-group"         = "/ecs/data-formatting-task",
        "awslogs-region"        = var.aws_region,
        "awslogs-stream-prefix" = "ecs",
        "s3_bucket_name"        = "${aws_s3_bucket.shared_bucket.bucket}"
      }
    }
  }
])

}

resource "aws_ecs_service" "data_formatting_service" {
  name            = "data-formatting-service"
  cluster         = aws_ecs_cluster.fargate_cluster.id
  task_definition = aws_ecs_task_definition.data_formatting_task.arn
  desired_count   = 1
  launch_type     = "FARGATE"

  network_configuration {
    subnets         = [aws_subnet.emr_subnet.id]
    security_groups = [aws_security_group.emr_master_sg.id, aws_security_group.emr_slave_sg.id]
    assign_public_ip = false
  }
}
