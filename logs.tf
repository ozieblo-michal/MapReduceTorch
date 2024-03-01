resource "aws_cloudwatch_log_group" "ecs_logs" {
  name = "/ecs/data-formatting-task"
  retention_in_days = 3

  tags = {
    Environment = "Development"
    Project     = "Data Formatting"
  }
}
