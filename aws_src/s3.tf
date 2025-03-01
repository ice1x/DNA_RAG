resource "aws_s3_bucket" "dna_bucket" {
  bucket = "dna-csv-bucket" # Replace with a unique bucket name
  acl    = "private"

  tags = {
    Name = "DNA CSV Bucket"
  }
}