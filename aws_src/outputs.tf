output "s3_bucket_name" {
  value = aws_s3_bucket.dna_bucket.bucket
}

output "generate_snp_dict_lambda_name" {
  value = aws_lambda_function.generate_snp_dict_lambda.function_name
}

output "api_gateway_url" {
  value = aws_api_gateway_deployment.dna_api_deployment.invoke_url
}