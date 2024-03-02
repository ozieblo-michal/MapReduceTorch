resource "aws_s3_bucket_object" "dask_bootstrap_script" {
  bucket  = aws_s3_bucket.shared_bucket.bucket
  key     = "bootstrap-scripts/dask_bootstrap.sh"
  content = <<EOF
            #!/bin/bash
            sudo pip install dask[complete] distributed --upgrade
            EOF
  acl     = "private"
}



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

  service_role = aws_iam_role.emr_default_role.arn
  autoscaling_role = aws_iam_role.emr_autoscaling_default_role.arn

  master_instance_group {
    instance_type = "m5.xlarge"
  }

  core_instance_group {
    instance_type = "m5.xlarge"
    instance_count = 2
    ebs_config {
      size = 64
      type = "gp2"
      volumes_per_instance = 1
    }
  }

  tags = {
    "Purpose"     = "Dask Processing"
    "Environment" = "Development"
    "Project"     = "LLM Training"
  }

  bootstrap_action {
    path = "s3://${aws_s3_bucket.shared_bucket.bucket}/bootstrap-scripts/dask_bootstrap.sh"
    name = "Dask Bootstrap Action"
  }
}

resource "aws_emr_instance_group" "dask_cluster_spot" {
  cluster_id   = aws_emr_cluster.dask_cluster.id
  instance_type = "m5.2xlarge"
  instance_count = 3
  bid_price = "0.30" 

  ebs_config {
    size = 128
    type = "gp2"
    volumes_per_instance = 1
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
