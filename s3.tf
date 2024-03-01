resource "aws_s3_bucket" "shared_bucket" {
  bucket = "shared-bucket-for-emr-and-fargate"
  acl    = "private"

  lifecycle_rule {
    id      = "log"
    enabled = true

    expiration {
      days = 90
    }
  }


  server_side_encryption_configuration {
    rule {
      apply_server_side_encryption_by_default {
        sse_algorithm = "AES256"
      }
    }
  }

  tags = {
    Purpose = "Shared bucket for EMR and Fargate scripts and data"
  }
}
