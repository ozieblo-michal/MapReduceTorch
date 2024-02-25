resource "aws_ecs_cluster" "fargate_cluster" {
  name = "data-formatting-cluster"
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
      name      = "data-formatting-container"
      image     = "your_docker_image_for_data_formatting"
      cpu       = 256
      memory    = 512
      essential = true
      command   = ["python", "run_formatting.py"]
    }
  ])
}

resource "aws_iam_role" "ecs_task_execution_role" {
  name = "ecs_task_execution_role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Principal = {
          Service = "ecs-tasks.amazonaws.com"
        }
        Effect = "Allow"
        Sid    = ""
      },
    ]
  })
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
      name      = "data-formatting-container"
      image     = "your_ecr_repository_uri/your_image:tag"
      cpu       = 256
      memory    = 512
      essential = true
      command   = ["python", "run_formatting.py"]
    }
  ])
}
