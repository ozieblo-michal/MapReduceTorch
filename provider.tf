terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.38"
    }
  }
}


provider "aws" {
  region = "eu-west-1"
}

variable "aws_region" {
  default = "eu-west-1"
}