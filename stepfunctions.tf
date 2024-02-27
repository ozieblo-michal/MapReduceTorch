resource "aws_iam_role" "step_functions_execution_role" {
  name = "step_functions_execution_role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17",
    Statement = [
      {
        Action = "sts:AssumeRole",
        Principal = {
          Service = "states.amazonaws.com"
        },
        Effect = "Allow",
        Sid    = ""
      },
    ]
  })
}

resource "aws_iam_policy" "step_functions_policy" {
  name        = "step_functions_policy"
  description = "A policy that allows execution of ECS tasks and EMR clusters"

  policy = jsonencode({
    Version = "2012-10-17",
    Statement = [
      {
        Action = [
          "ecs:RunTask",
          "ecs:StopTask",
          "ecs:DescribeTasks",
          "elasticmapreduce:RunJobFlow",
          "elasticmapreduce:DescribeCluster",
          "elasticmapreduce:TerminateClusters"
        ],
        Resource = "*",
        Effect   = "Allow"
      },
    ]
  })
}

resource "aws_iam_role_policy_attachment" "step_functions_policy_attachment" {
  role       = aws_iam_role.step_functions_execution_role.name
  policy_arn = aws_iam_policy.step_functions_policy.arn
}


resource "aws_sfn_state_machine" "example_state_machine" {
  name     = "example_state_machine"
  role_arn = aws_iam_role.step_functions_execution_role.arn

  definition = <<EOF
{
  "Comment": "An example of AWS Step Function to run a Fargate task and then an EMR cluster",
  "StartAt": "RunFargateTask",
  "States": {
    "RunFargateTask": {
      "Type": "Task",
      "Resource": "arn:aws:states:::ecs:runTask.sync",
      "Parameters": {
        "LaunchType": "FARGATE",
        "Cluster": "aws_ecs_cluster.fargate_cluster.arn",
        "TaskDefinition": "aws_ecs_task_definition.data_formatting_task.arn",
        ...
      },
      "Next": "RunEMRCluster"
    },
    "RunEMRCluster": {
      "Type": "Task",
      "Resource": "arn:aws:states:::elasticmapreduce:createCluster.sync",
      "Parameters": {
        "Name": "EMR Cluster from Step Function",
        "ReleaseLabel": "emr-5.30.0",
        ...
      },
      "End": true
    }
  }
}
EOF
}
