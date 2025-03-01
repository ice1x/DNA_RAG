# API Gateway
resource "aws_api_gateway_rest_api" "dna_api" {
  name        = "dna_api"
  description = "API for DNA processing service"
}

# Resource for Generate SNP Dictionary
resource "aws_api_gateway_resource" "generate_snp_dict_resource" {
  rest_api_id = aws_api_gateway_rest_api.dna_api.id
  parent_id   = aws_api_gateway_rest_api.dna_api.root_resource_id
  path_part   = "generate-snp-dict"
}

# POST Method for Generate SNP Dictionary
resource "aws_api_gateway_method" "generate_snp_dict_method" {
  rest_api_id   = aws_api_gateway_rest_api.dna_api.id
  resource_id   = aws_api_gateway_resource.generate_snp_dict_resource.id
  http_method   = "POST"
  authorization = "NONE"
}

# Integration with Generate SNP Dictionary Lambda
resource "aws_api_gateway_integration" "generate_snp_dict_integration" {
  rest_api_id             = aws_api_gateway_rest_api.dna_api.id
  resource_id             = aws_api_gateway_resource.generate_snp_dict_resource.id
  http_method             = aws_api_gateway_method.generate_snp_dict_method.http_method
  integration_http_method = "POST"
  type                    = "AWS_PROXY"
  uri                     = aws_lambda_function.generate_snp_dict_lambda.invoke_arn
}

# Deploy API Gateway
resource "aws_api_gateway_deployment" "dna_api_deployment" {
  rest_api_id = aws_api_gateway_rest_api.dna_api.id
  stage_name  = "prod"
}

# Permission for API Gateway to Invoke Lambda
resource "aws_lambda_permission" "api_gateway_generate_snp_dict" {
  statement_id  = "AllowAPIGatewayInvoke"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.generate_snp_dict_lambda.function_name
  principal     = "apigateway.amazonaws.com"
  source_arn    = "${aws_api_gateway_rest_api.dna_api.execution_arn}/*/*"
}

resource "aws_api_gateway_resource" "file_upload_resource" {
  rest_api_id = aws_api_gateway_rest_api.dna_api.id
  parent_id   = aws_api_gateway_rest_api.dna_api.root_resource_id
  path_part   = "upload"
}

resource "aws_api_gateway_method" "file_upload_method" {
  rest_api_id   = aws_api_gateway_rest_api.dna_api.id
  resource_id   = aws_api_gateway_resource.file_upload_resource.id
  http_method   = "POST"
  authorization = "NONE"
}

resource "aws_api_gateway_integration" "file_upload_integration" {
  rest_api_id             = aws_api_gateway_rest_api.dna_api.id
  resource_id             = aws_api_gateway_resource.file_upload_resource.id
  http_method             = aws_api_gateway_method.file_upload_method.http_method
  integration_http_method = "POST"
  type                    = "AWS_PROXY"
  uri                     = aws_lambda_function.file_upload_lambda.invoke_arn
}