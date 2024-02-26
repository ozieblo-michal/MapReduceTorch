resource "aws_emr_cluster" "dask_cluster" {
  name          = "dask-emr-cluster"
  release_label = "emr-6.2.0"
  applications  = ["Hadoop", "Spark"]

  ec2_attributes {
    subnet_id                         = aws_subnet.emr_subnet.id
    emr_managed_master_security_group = aws_security_group.emr_master_sg.id
    emr_managed_slave_security_group  = aws_security_group.emr_slave_sg.id
    instance_profile                  = aws_iam_instance_profile.emr_ec2_instance_profile.name
  }

  service_role = "EMR_DefaultRole"
  autoscaling_role = "EMR_AutoScaling_DefaultRole"

  master_instance_group {
    instance_type  = "m5.xlarge"
    instance_count = 1
  }

  core_instance_group {
    instance_type  = "m5.xlarge"
    instance_count = 2
  }

  tags = {
    "Purpose" = "Dask Processing"
  }
}

resource "aws_iam_role" "emr_default_role" {
  name = "EMR_DefaultRole"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Principal = {
          Service = "elasticmapreduce.amazonaws.com"
        }
        Effect = "Allow"
        Sid    = ""
      },
    ]
  })
}

resource "aws_iam_role" "emr_autoscaling_default_role" {
  name = "EMR_AutoScaling_DefaultRole"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Principal = {
          Service = "application-autoscaling.amazonaws.com"
        }
        Effect = "Allow"
        Sid    = ""
      },
    ]
  })
}

resource "aws_emr_step" "example" {
  cluster_id = aws_emr_cluster.dask_cluster.id
  name       = "RunMyDaskCode"

  application {
    name = "Spark"
  }

  action_on_failure = "CONTINUE"
  hadoop_jar_step {
    jar  = "command-runner.jar"
    args = ["spark-submit", "--deploy-mode", "cluster", "s3://book2flash/main.py"]
  }
}
