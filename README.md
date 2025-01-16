# AI Answer Comparison Tool

A Go application that queries both Google's Gemini and Groq's LLM APIs for answers to questions and compares their responses for consistency. This tool is particularly optimized for handling multiple choice questions.

## Features

- Parallel querying of Gemini and Groq AI models
- Automatic response comparison using Gemini
- Enhanced prompt formatting for multiple choice questions
- Graceful error handling with configurable retries
- Interactive command-line interface
- Ctrl+C handling for graceful shutdown

## Prerequisites

- Go 1.x or higher
- Google Gemini API key
- Groq API key

## Installation

1. Clone the repository
2. Install dependencies:
```bash
go mod download
```

## Configuration

1. Create a `.env` file in the project root with your API keys:
```
GEMINI_API_KEY=your_gemini_api_key
GROQ_API_KEY=your_groq_api_key
```

## Usage

Run the application:
```bash
go run main.go [--max-retries N]
```

Options:
- `--max-retries`: Maximum number of retry attempts (default: 3)

### Interactive Usage

1. Enter your question when prompted
2. Press Ctrl+D (Unix) or Ctrl+Z (Windows) on a new line to submit
3. Type 'exit' to quit the application

The tool will:
1. Send the question to both Gemini and Groq
2. Compare their responses
3. If responses are similar, return the Gemini response
4. If responses differ after max retries, show both responses

## Error Handling

- Automatically retries on API failures
- Graceful handling of network issues
- Clear error messages for troubleshooting

## License

This project is open source and available under the MIT License. 
