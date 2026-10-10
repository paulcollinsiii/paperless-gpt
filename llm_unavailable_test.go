package main

import (
	"context"
	"net/http"
	"testing"

	"paperless-gpt/ocr"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// TestCreateLLMOllama_ReportsUnavailable is the wiring test for the suggestions
// client: an Ollama that answers 503 (a mesh sidecar in front of a scaled-to-zero
// pod) must come back as an unavailable backend, not a document failure.
func TestCreateLLMOllama_ReportsUnavailable(t *testing.T) {
	t.Setenv("OLLAMA_HEADERS", "")
	t.Setenv("LLM_MAX_RETRIES", "1")

	server := testOllamaServer(t, func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusServiceUnavailable)
		_, _ = w.Write([]byte("no healthy upstream"))
	})
	t.Setenv("OLLAMA_HOST", server.URL)

	previousProvider, previousModel := llmProvider, llmModel
	llmProvider, llmModel = "ollama", "test-model"
	t.Cleanup(func() { llmProvider, llmModel = previousProvider, previousModel })

	model, err := createLLM()
	require.NoError(t, err)
	_, err = model.Call(context.Background(), "Return a title")
	require.Error(t, err)
	assert.True(t, ocr.IsBackendUnavailable(err), "%v", err)
}
