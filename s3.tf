resource "aws_s3_bucket" "shared_bucket" {
  bucket = "shared-bucket-for-emr-and-fargate"
  acl    = "private"

  tags = {
    Purpose = "Shared bucket for EMR and Fargate scripts and data"
  }
}
