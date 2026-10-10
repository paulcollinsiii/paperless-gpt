package ocr

import (
	"context"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"github.com/tmc/langchaingo/llms"
)

// The vision Ollama client must report a stopped or starting Ollama behind a
// mesh (HTTP 503) as unavailable rather than as a failure of the document.
func TestCreateOllamaClient_ReportsUnavailable(t *testing.T) {
	t.Setenv("OLLAMA_HEADERS", "")

	t.Run("HTTP 503", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.WriteHeader(http.StatusServiceUnavailable)
			_, _ = w.Write([]byte("no healthy upstream"))
		}))
		defer server.Close()
		t.Setenv("OLLAMA_HOST", server.URL)

		model, err := createOllamaClient(Config{VisionLLMModel: "test-model"})
		require.NoError(t, err)
		_, err = llms.GenerateFromSinglePrompt(context.Background(), model, "hi")
		require.Error(t, err)
		assert.True(t, IsBackendUnavailable(err), "%v", err)
	})

	t.Run("HTTP 500 is a document failure", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.WriteHeader(http.StatusInternalServerError)
			_, _ = w.Write([]byte(`{"error":"model failed to load"}`))
		}))
		defer server.Close()
		t.Setenv("OLLAMA_HOST", server.URL)

		model, err := createOllamaClient(Config{VisionLLMModel: "test-model"})
		require.NoError(t, err)
		_, err = llms.GenerateFromSinglePrompt(context.Background(), model, "hi")
		require.Error(t, err)
		assert.False(t, IsBackendUnavailable(err), "%v", err)
	})

	t.Run("connection refused", func(t *testing.T) {
		t.Setenv("OLLAMA_HOST", closedPortURL(t))

		model, err := createOllamaClient(Config{VisionLLMModel: "test-model"})
		require.NoError(t, err)
		_, err = llms.GenerateFromSinglePrompt(context.Background(), model, "hi")
		require.Error(t, err)
		assert.True(t, IsBackendUnavailable(err), "%v", err)
	})
}
