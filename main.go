package main

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"net/http"
	"os"
	"os/signal"
	"strings"
	"syscall"

	"github.com/google/generative-ai-go/genai"
	"github.com/joho/godotenv"
	"google.golang.org/api/option"
)

type Config struct {
	GeminiKey  string
	GroqKey    string
	MaxRetries int
}

type Response struct {
	Source  string
	Content string
}

// Groq API types
type GroqRequest struct {
	Model    string    `json:"model"`
	Messages []Message `json:"messages"`
}

type Message struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type GroqResponse struct {
	Choices []struct {
		Message struct {
			Content string `json:"content"`
		} `json:"message"`
	} `json:"choices"`
}

func main() {
	// Set up signal handling
	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, os.Interrupt, syscall.SIGTERM)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	// Handle Ctrl+C in a separate goroutine
	go func() {
		<-sigChan
		fmt.Println("\nReceived interrupt signal. Shutting down gracefully...")
		cancel()
		fmt.Println("Goodbye!")
		os.Exit(0)
	}()

	// Load .env file
	if err := godotenv.Load(); err != nil {
		log.Fatal("Error loading .env file")
	}

	// Get API keys from environment variables
	geminiKey := os.Getenv("GEMINI_API_KEY")
	groqKey := os.Getenv("GROQ_API_KEY")

	// Parse command line flags
	maxRetries := flag.Int("max-retries", 3, "Maximum number of retry attempts")
	flag.Parse()

	if geminiKey == "" || groqKey == "" {
		log.Fatal("Please set GEMINI_API_KEY and GROQ_API_KEY in .env file")
	}

	config := Config{
		GeminiKey:  geminiKey,
		GroqKey:    groqKey,
		MaxRetries: *maxRetries,
	}

	reader := bufio.NewReader(os.Stdin)

	fmt.Println("Interactive AI Question Answering System")
	fmt.Println("Enter your questions (type 'exit' to quit)")
	fmt.Println("Type your question and press Ctrl+D (Unix) or Ctrl+Z (Windows) on a new line to finish")
	fmt.Println("----------------------------------------")

	for {
		fmt.Print("\nEnter your question:\n")
		var questionBuilder strings.Builder

		for {
			line, err := reader.ReadString('\n')
			if err != nil {
				if err.Error() == "EOF" {
					break
				}
				log.Printf("Error reading input: %v", err)
				continue
			}

			// Check for exit command
			if strings.TrimSpace(strings.ToLower(line)) == "exit" {
				fmt.Println("Goodbye!")
				return
			}

			questionBuilder.WriteString(line)
		}

		question := strings.TrimSpace(questionBuilder.String())
		if question == "" {
			continue
		}

		response, err := processQuestion(ctx, question, config)
		if err != nil {
			log.Printf("Error processing question: %v", err)
			continue
		}

		fmt.Printf("\nResponse:\n%s\n", response)

		// Clear the input buffer
		reader = bufio.NewReader(os.Stdin)
	}
}

func processQuestion(ctx context.Context, question string, config Config) (string, error) {
	var geminiResp, groqResp *Response
	var err error
	var lastGeminiErr, lastGroqErr error

	for attempt := 0; attempt < config.MaxRetries; attempt++ {
		// Get responses from both APIs
		geminiResp, err = getGeminiResponse(ctx, question, config.GeminiKey)
		if err != nil {
			log.Printf("Attempt %d: Gemini API error: %v", attempt+1, err)
			lastGeminiErr = err
			continue
		}

		groqResp, err = getGroqResponse(question, config.GroqKey)
		if err != nil {
			log.Printf("Attempt %d: Groq API error: %v", attempt+1, err)
			lastGroqErr = err
			continue
		}

		// Compare responses using Gemini
		similar, err := compareResponses(ctx, geminiResp.Content, groqResp.Content, config.GeminiKey)
		if err != nil {
			log.Printf("Attempt %d: Comparison error: %v", attempt+1, err)
			continue
		}

		if similar {
			return geminiResp.Content, nil
		}

		log.Printf("Attempt %d: Responses differ, retrying...", attempt+1)
	}

	// If we've exhausted retries, return error message
	if geminiResp == nil && groqResp == nil {
		return "", fmt.Errorf("failed to get responses after %d attempts. Gemini error: %v, Groq error: %v",
			config.MaxRetries, lastGeminiErr, lastGroqErr)
	}

	// Return both responses if available
	geminiContent := "Error getting response"
	groqContent := "Error getting response"

	if geminiResp != nil {
		geminiContent = formatResponse(geminiResp.Content)
	}
	if groqResp != nil {
		groqContent = formatResponse(groqResp.Content)
	}

	return fmt.Sprintf("Responses after %d attempts:\n\nGemini Response:\n%s\n\nGroq Response:\n%s",
		config.MaxRetries, geminiContent, groqContent), nil
}

func getGeminiResponse(ctx context.Context, question, apiKey string) (*Response, error) {
	client, err := genai.NewClient(ctx, option.WithAPIKey(apiKey))
	if err != nil {
		return nil, fmt.Errorf("failed to create Gemini client: %v", err)
	}
	defer client.Close()

	// Enhance the prompt for multiple choice questions
	enhancedPrompt := fmt.Sprintf(`If this is a multiple choice question, please:
1. Analyze each option carefully
2. Provide a clear "Yes" or "No" for each option
3. Explain the reasoning for each option
4. At the end, summarize which options are correct

Here's the question:
%s`, question)

	model := client.GenerativeModel("gemini-pro")
	resp, err := model.GenerateContent(ctx, genai.Text(enhancedPrompt))
	if err != nil {
		return nil, fmt.Errorf("failed to generate Gemini response: %v", err)
	}

	return &Response{
		Source:  "Gemini",
		Content: string(resp.Candidates[0].Content.Parts[0].(genai.Text)),
	}, nil
}

func getGroqResponse(question, apiKey string) (*Response, error) {
	url := "https://api.groq.com/openai/v1/chat/completions"

	// Enhance the prompt for multiple choice questions
	enhancedPrompt := fmt.Sprintf(`If this is a multiple choice question, please:
1. Analyze each option carefully
2. Provide a clear "Yes" or "No" for each option
3. Explain the reasoning for each option
4. At the end, summarize which options are correct

Here's the question:
%s`, question)

	reqBody := GroqRequest{
		Model: "llama-3.3-70b-versatile",
		Messages: []Message{
			{
				Role:    "user",
				Content: enhancedPrompt,
			},
		},
	}

	jsonData, err := json.Marshal(reqBody)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %v", err)
	}

	req, err := http.NewRequest("POST", url, bytes.NewBuffer(jsonData))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %v", err)
	}

	req.Header.Set("Authorization", "Bearer "+apiKey)
	req.Header.Set("Content-Type", "application/json")

	client := &http.Client{}
	resp, err := client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("failed to send request: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("API request failed with status: %d", resp.StatusCode)
	}

	var groqResp GroqResponse
	if err := json.NewDecoder(resp.Body).Decode(&groqResp); err != nil {
		return nil, fmt.Errorf("failed to decode response: %v", err)
	}

	if len(groqResp.Choices) == 0 {
		return nil, fmt.Errorf("no response choices returned")
	}

	return &Response{
		Source:  "Groq",
		Content: groqResp.Choices[0].Message.Content,
	}, nil
}

func compareResponses(ctx context.Context, resp1, resp2, geminiKey string) (bool, error) {
	client, err := genai.NewClient(ctx, option.WithAPIKey(geminiKey))
	if err != nil {
		return false, fmt.Errorf("failed to create Gemini client for comparison: %v", err)
	}
	defer client.Close()

	comparisonPrompt := fmt.Sprintf(`Compare these two responses and determine if they convey the same meaning. 
	Only respond with "true" if they are semantically equivalent, or "false" if they differ significantly in meaning.
	
	Response 1:
	%s

	Response 2:
	%s`, resp1, resp2)

	model := client.GenerativeModel("gemini-pro")
	resp, err := model.GenerateContent(ctx, genai.Text(comparisonPrompt))
	if err != nil {
		return false, fmt.Errorf("failed to compare responses using Gemini: %v", err)
	}

	result := strings.ToLower(strings.TrimSpace(string(resp.Candidates[0].Content.Parts[0].(genai.Text))))
	return result == "true", nil
}

func formatResponse(response string) string {
	// Add some visual separation between sections
	response = strings.ReplaceAll(response, "Option", "\nOption")
	response = strings.ReplaceAll(response, "Summary", "\n\nSummary")
	return response
}
