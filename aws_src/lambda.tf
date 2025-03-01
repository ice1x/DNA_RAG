# Generate SNP Dictionary Lambda
resource "aws_lambda_function" "generate_snp_dict_lambda" {
  function_name = "GenerateSNPDictLambda"
  handler       = "lambda_function.lambda_handler"
  runtime       = "python3.9"
  role          = aws_iam_role.lambda_role.arn
  timeout       = 30

  # Replace with the path to your Lambda deployment package
  filename         = "generate_snp_dict_lambda.zip"
  source_code_hash = filebase64sha256("generate_snp_dict_lambda.zip")

  environment {
    variables = {
      OPENAI_API_KEY = "your-openai-api-key" # Replace with your OpenAI API key
    }
  }
}

# Process DNA Data Lambda
resource "aws_lambda_function" "process_dna_data_lambda" {
  function_name = "ProcessDNADataLambda"
  handler       = "lambda_function.lambda_handler"
  runtime       = "python3.9"
  role          = aws_iam_role.lambda_role.arn
  timeout       = 30

  # Replace with the path to your Lambda deployment package
  filename         = "process_dna_data_lambda.zip"
  source_code_hash = filebase64sha256("process_dna_data_lambda.zip")
}

# Process with Second LLM Lambda
resource "aws_lambda_function" "process_with_second_llm_lambda" {
  function_name = "ProcessWithSecondLLMLambda"
  handler       = "lambda_function.lambda_handler"
  runtime       = "python3.9"
  role          = aws_iam_role.lambda_role.arn
  timeout       = 30

  # Replace with the path to your Lambda deployment package
  filename         = "process_with_second_llm_lambda.zip"
  source_code_hash = filebase64sha256("process_with_second_llm_lambda.zip")

  environment {
    variables = {
      OPENAI_API_KEY = "your-openai-api-key" # Replace with your OpenAI API key
    }
  }
}

resource "aws_lambda_function" "file_upload_lambda" {
  function_name = "FileUploadLambda"
  handler       = "lambda_function.lambda_handler"
  runtime       = "python3.9"
  role          = aws_iam_role.lambda_role.arn
  timeout       = 30

  filename         = "file_upload_lambda.zip"
  source_code_hash = filebase64sha256("file_upload_lambda.zip")
}

# Add permission for API Gateway to invoke this Lambda
resource "aws_lambda_permission" "api_gateway_file_upload" {
  statement_id  = "AllowAPIGatewayInvokeFileUpload"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.file_upload_lambda.function_name
  principal     = "apigateway.amazonaws.com"
  source_arn    = "${aws_api_gateway_rest_api.dna_api.execution_arn}/*/*"
}