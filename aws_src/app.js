import React, { useState } from "react";
import axios from "axios";
import { API } from "aws-amplify";

function App() {
  const [query, setQuery] = useState("");
  const [file, setFile] = useState(null);
  const [result, setResult] = useState("");
  const [loading, setLoading] = useState(false);

  const handleFileChange = (e) => {
    setFile(e.target.files[0]);
  };

  const handleQueryChange = (e) => {
    setQuery(e.target.value);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);

    try {
      // Step 1: Upload the file to S3
      const uploadResponse = await axios.post(
        "https://your-api-gateway-url/upload", // Replace with your API Gateway upload endpoint
        { file },
        { headers: { "Content-Type": "multipart/form-data" } }
      );
      const fileKey = uploadResponse.data.fileKey;

      // Step 2: Generate SNP Dictionary
      const snpDictResponse = await API.post("dna-api", "/generate-snp-dict", {
        body: { query },
      });
      const snpDict = snpDictResponse.data;

      // Step 3: Process DNA Data
      const processResponse = await API.post("dna-api", "/process-dna-data", {
        body: { fileKey, snpDict },
      });
      const filteredData = processResponse.data;

      // Step 4: Process with Second LLM
      const llmResponse = await API.post("dna-api", "/process-with-second-llm", {
        body: { filteredData, query },
      });
      setResult(llmResponse.data);
    } catch (error) {
      console.error("Error:", error);
      setResult("An error occurred. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="App">
      <h1>DNA Processing Service</h1>
      <form onSubmit={handleSubmit}>
        <div>
          <label>
            Enter your query (e.g., "lactose tolerance"):
            <input
              type="text"
              value={query}
              onChange={handleQueryChange}
              required
            />
          </label>
        </div>
        <div>
          <label>
            Upload your DNA CSV file:
            <input type="file" onChange={handleFileChange} required />
          </label>
        </div>
        <button type="submit" disabled={loading}>
          {loading ? "Processing..." : "Submit"}
        </button>
      </form>
      {result && (
        <div>
          <h2>Result:</h2>
          <pre>{result}</pre>
        </div>
      )}
    </div>
  );
}

export default App;