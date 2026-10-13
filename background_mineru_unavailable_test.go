package main

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"
	"time"

	"paperless-gpt/ocr"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// providerProcessor is a DocumentProcessor that runs a real OCR provider and
// wraps its error the way App.ProcessDocumentOCR does.
type providerProcessor struct {
	provider ocr.Provider
}

func (p providerProcessor) ProcessDocumentOCR(ctx context.Context, documentID int, options OCROptions, jobID string) (*ProcessedDocument, error) {
	if _, err := p.provider.ProcessImage(ctx, []byte("%PDF-1.4 synthetic"), 0); err != nil {
		return nil, fmt.Errorf("error performing OCR for document %d: %w", documentID, err)
	}
	return &ProcessedDocument{ID: documentID, Text: "text"}, nil
}

// A MinerU behind a proxy that answers 503 while the service is scaled to zero
// must read as "unavailable" at the cycle level: the auto-tag poll still runs,
// nothing is counted, and the cycle gets the short backoff cap.
func TestRunBackgroundCycle_RealMinerUBehind503(t *testing.T) {
	prevFailTag, prevAutoOcrTag, prevAutoTag, prevOcrMax := failTag, autoOcrTag, autoTag, ocrMaxRetries
	t.Cleanup(func() {
		failTag, autoOcrTag, autoTag, ocrMaxRetries = prevFailTag, prevAutoOcrTag, prevAutoTag, prevOcrMax
	})
	failTag = "paperless-gpt-failed"
	autoOcrTag = "paperless-gpt-ocr-auto"
	autoTag = "paperless-gpt-auto"
	ocrMaxRetries = 3
	isolateWorkflowSettings(t)

	var hits atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		hits.Add(1)
		w.WriteHeader(http.StatusServiceUnavailable)
		_, _ = w.Write([]byte("upstream connect error or disconnect/reset before headers"))
	}))
	defer server.Close()

	provider, err := ocr.NewProvider(ocr.Config{Provider: "mineru", MinerUURL: server.URL})
	require.NoError(t, err)

	newApp := func(client *recordingClient) *App {
		return &App{
			Client:       client,
			docProcessor: providerProcessor{provider: provider},
			Database:     newOCRRunTestDB(t),
			ocrProvider:  provider,
		}
	}
	docs := map[string][]Document{
		autoOcrTag: {
			{ID: 1, Title: "One", Tags: []string{autoOcrTag}},
			{ID: 2, Title: "Two", Tags: []string{autoOcrTag}},
		},
	}

	t.Run("only unavailable errors", func(t *testing.T) {
		client := &recordingClient{taggedDocuments: docs}
		app := newApp(client)

		for round := 1; round <= ocrMaxRetries+2; round++ {
			_, err := runBackgroundCycle(context.Background(), app)
			require.Error(t, err, "round %d", round)
			assert.True(t, allBackendUnavailable(err), "round %d: %v", round, err)
		}
		assert.Contains(t, client.tagFetches, autoTag, "the auto-tag poll runs although OCR is down")
		assert.Empty(t, client.calls, "no fail tag, no tag removal")
		assert.Empty(t, app.ocrFailures.failures)

		_, err := runBackgroundCycle(context.Background(), app)
		sleep, _ := nextBackoff(time.Hour, allBackendUnavailable(err), 2*time.Minute)
		assert.Equal(t, 2*time.Minute, sleep, "the short cap applies")
	})

	t.Run("one document unavailable and another failing for an ordinary reason", func(t *testing.T) {
		client := &recordingClient{taggedDocuments: docs}
		app := newApp(client)
		// Document 1 reaches the (down) service; document 2 fails on its own.
		app.docProcessor = splitProcessor{
			unavailable: providerProcessor{provider: provider},
			ordinary:    &countingProcessor{errs: map[int]error{2: errors.New("pdf is broken")}},
		}
		// Process both documents in one batch: stop-on-unavailable only applies
		// after the first, so order the ordinary failure first.
		client.taggedDocuments = map[string][]Document{autoOcrTag: {docs[autoOcrTag][1], docs[autoOcrTag][0]}}

		_, err := runBackgroundCycle(context.Background(), app)
		require.Error(t, err)
		assert.False(t, allBackendUnavailable(err), "a mixed batch is an ordinary error: %v", err)
	})
}

// splitProcessor sends document 1 to one processor and the rest to another.
type splitProcessor struct {
	unavailable, ordinary DocumentProcessor
}

func (s splitProcessor) ProcessDocumentOCR(ctx context.Context, documentID int, options OCROptions, jobID string) (*ProcessedDocument, error) {
	if documentID == 1 {
		return s.unavailable.ProcessDocumentOCR(ctx, documentID, options, jobID)
	}
	return s.ordinary.ProcessDocumentOCR(ctx, documentID, options, jobID)
}

// A single error with several %w causes is one failure, whatever order the
// causes are in; it must not be mistaken for a batch of independent failures.
func TestAllBackendUnavailable_MultipleWrappedCauses(t *testing.T) {
	multi := fmt.Errorf("mineru upload: HTTP 503: %w and %w", errors.New("descriptive cause"), ocr.ErrBackendUnavailable)
	assert.True(t, allBackendUnavailable(multi))
	assert.True(t, allBackendUnavailable(fmt.Errorf("one or more errors occurred: %w",
		joinBatch(fmt.Errorf("document 1 OCR error: %w", multi)))))
	assert.False(t, allBackendUnavailable(fmt.Errorf("one or more errors occurred: %w",
		joinBatch(fmt.Errorf("document 1 OCR error: %w", multi), errors.New("pdf is broken")))))
}

func TestRunBackgroundCycle_ReportsProcessedCountOnError(t *testing.T) {
	stub := &cycleStub{ocrErr: errBackendDown}
	count, err := runBackgroundCycle(context.Background(), stub)
	require.Error(t, err)
	assert.True(t, allBackendUnavailable(err))
	assert.Equal(t, 2, count, "documents processed by the auto-tag poll are still reported")

	stub = &cycleStub{tagErr: errors.New("boom")}
	count, err = runBackgroundCycle(context.Background(), stub)
	require.Error(t, err)
	assert.Equal(t, 2, count)
}
