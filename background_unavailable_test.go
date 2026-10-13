package main

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"testing"
	"time"

	"paperless-gpt/ocr"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

var errBackendDown = fmt.Errorf("upstream connect error: %w", ocr.ErrBackendUnavailable)

// countingProcessor is a DocumentProcessor that records which documents it was
// asked to OCR.
type countingProcessor struct {
	errs  map[int]error
	asked []int
}

func (c *countingProcessor) ProcessDocumentOCR(ctx context.Context, documentID int, options OCROptions, jobID string) (*ProcessedDocument, error) {
	c.asked = append(c.asked, documentID)
	if err := c.errs[documentID]; err != nil {
		return nil, err
	}
	return &ProcessedDocument{ID: documentID, Text: "text"}, nil
}

// tagsCountingClient counts how many documents reach suggestion generation,
// which asks for the tag list first.
type tagsCountingClient struct {
	*recordingClient
	suggestionAttempts int
}

func (c *tagsCountingClient) GetAllTags(ctx context.Context) (map[string]int, error) {
	c.suggestionAttempts++
	return c.recordingClient.GetAllTags(ctx)
}

func TestAllBackendUnavailable(t *testing.T) {
	ordinary := errors.New("pdf is broken")
	wrapped := fmt.Errorf("document 1 OCR error: %w", errBackendDown)

	assert.False(t, allBackendUnavailable(nil))
	assert.True(t, allBackendUnavailable(errBackendDown))
	assert.True(t, allBackendUnavailable(wrapped))
	assert.False(t, allBackendUnavailable(ordinary))
	assert.True(t, allBackendUnavailable(joinBatch(wrapped, errBackendDown)))
	assert.False(t, allBackendUnavailable(joinBatch(wrapped, ordinary)), "one ordinary failure keeps the long backoff")
	assert.False(t, allBackendUnavailable(fmt.Errorf("cycle: %w", joinBatch(wrapped, ordinary))))
	assert.True(t, allBackendUnavailable(fmt.Errorf("cycle: %w", joinBatch(wrapped, errBackendDown))))
	assert.False(t, allBackendUnavailable(context.Canceled))
}

func TestNextBackoff(t *testing.T) {
	const unavailableMax = 2 * time.Minute

	t.Run("unavailable doubles from the minimum and stops at the cap", func(t *testing.T) {
		var slept []time.Duration
		backoff := minBackoffDuration
		for i := 0; i < 7; i++ {
			var s time.Duration
			s, backoff = nextBackoff(backoff, true, unavailableMax)
			slept = append(slept, s)
		}
		assert.Equal(t, []time.Duration{
			10 * time.Second, 20 * time.Second, 40 * time.Second, 80 * time.Second,
			120 * time.Second, 120 * time.Second, 120 * time.Second,
		}, slept)
	})

	t.Run("ordinary errors keep the one hour cap", func(t *testing.T) {
		backoff := minBackoffDuration
		var last time.Duration
		for i := 0; i < 20; i++ {
			last, backoff = nextBackoff(backoff, false, unavailableMax)
		}
		assert.Equal(t, time.Hour, last)
		assert.Equal(t, time.Hour, backoff)

		s, _ := nextBackoff(10*time.Second, false, unavailableMax)
		assert.Equal(t, 10*time.Second, s)
		s, next := nextBackoff(80*time.Second, false, unavailableMax)
		assert.Equal(t, 80*time.Second, s, "an ordinary error is not held to the unavailable cap")
		assert.Equal(t, 160*time.Second, next)
	})

	t.Run("a long ordinary backoff is cut down once the cause is an unavailable backend", func(t *testing.T) {
		s, next := nextBackoff(time.Hour, true, unavailableMax)
		assert.Equal(t, unavailableMax, s)
		assert.Equal(t, unavailableMax, next)
	})
}

func TestUnavailableMaxBackoffFromEnv(t *testing.T) {
	cases := []struct {
		name, value string
		want        time.Duration
	}{
		{"unset", "", 2 * time.Minute},
		{"valid", "45s", 45 * time.Second},
		{"valid minutes", "5m", 5 * time.Minute},
		{"invalid", "soon", 2 * time.Minute},
		{"missing unit", "30", 2 * time.Minute},
		{"zero", "0s", 2 * time.Minute},
		{"negative", "-1m", 2 * time.Minute},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Setenv("UNAVAILABLE_MAX_BACKOFF", tc.value)
			assert.Equal(t, tc.want, unavailableMaxBackoffFromEnv())
		})
	}
}

// cycleStub is a BackgroundProcessor with canned results.
type cycleStub struct {
	ocrErr, tagErr error
	ocrCalls       int
	tagCalls       int
}

func (c *cycleStub) isOcrEnabled() bool { return true }
func (c *cycleStub) processAutoOcrTagDocuments(ctx context.Context) (int, error) {
	c.ocrCalls++
	return 0, c.ocrErr
}
func (c *cycleStub) processAutoTagDocuments(ctx context.Context) (int, error) {
	c.tagCalls++
	return 2, c.tagErr
}

func TestRunBackgroundCycle(t *testing.T) {
	t.Run("OCR backend unavailable still lets auto-tagging run", func(t *testing.T) {
		stub := &cycleStub{ocrErr: fmt.Errorf("document 4 OCR error: %w", errBackendDown)}
		_, err := runBackgroundCycle(context.Background(), stub)
		require.Error(t, err)
		assert.Equal(t, 1, stub.tagCalls)
		assert.True(t, allBackendUnavailable(err), "the cycle is reported as unavailable only")
	})

	t.Run("OCR unavailable and a different auto-tag failure is an ordinary error", func(t *testing.T) {
		stub := &cycleStub{ocrErr: errBackendDown, tagErr: errors.New("exceeds the context size")}
		_, err := runBackgroundCycle(context.Background(), stub)
		require.Error(t, err)
		assert.Equal(t, 1, stub.tagCalls)
		assert.False(t, allBackendUnavailable(err))
		assert.ErrorIs(t, err, ocr.ErrBackendUnavailable)
	})

	t.Run("an ordinary OCR error still ends the cycle early", func(t *testing.T) {
		stub := &cycleStub{ocrErr: errors.New("pdf is broken")}
		_, err := runBackgroundCycle(context.Background(), stub)
		require.Error(t, err)
		assert.Equal(t, 0, stub.tagCalls)
		assert.False(t, allBackendUnavailable(err))
	})

	t.Run("OCR errors that mix unavailable and ordinary end the cycle early", func(t *testing.T) {
		stub := &cycleStub{ocrErr: joinBatch(errBackendDown, errors.New("pdf is broken"))}
		_, err := runBackgroundCycle(context.Background(), stub)
		require.Error(t, err)
		assert.Equal(t, 0, stub.tagCalls)
	})

	t.Run("both unavailable", func(t *testing.T) {
		stub := &cycleStub{ocrErr: errBackendDown, tagErr: errBackendDown}
		_, err := runBackgroundCycle(context.Background(), stub)
		require.Error(t, err)
		assert.True(t, allBackendUnavailable(err))
	})

	t.Run("success counts both polls", func(t *testing.T) {
		stub := &cycleStub{}
		count, err := runBackgroundCycle(context.Background(), stub)
		require.NoError(t, err)
		assert.Equal(t, 2, count)
	})
}

func TestProcessAutoOcrTagDocuments_BackendUnavailableIsNotCounted(t *testing.T) {
	prevFailTag, prevAutoOcrTag, prevMaxRetries := failTag, autoOcrTag, ocrMaxRetries
	t.Cleanup(func() { failTag, autoOcrTag, ocrMaxRetries = prevFailTag, prevAutoOcrTag, prevMaxRetries })
	failTag = "paperless-gpt-failed"
	autoOcrTag = "paperless-gpt-ocr-auto"
	ocrMaxRetries = 3

	docs := []Document{
		{ID: 1, Title: "One", Tags: []string{autoOcrTag}},
		{ID: 2, Title: "Two", Tags: []string{autoOcrTag}},
	}
	newApp := func(client *recordingClient, processor *countingProcessor) *App {
		return &App{Client: client, docProcessor: processor, Database: newOCRRunTestDB(t)}
	}

	t.Run("never fail-tagged however many cycles fail, and the batch stops", func(t *testing.T) {
		client := &recordingClient{taggedDocuments: map[string][]Document{autoOcrTag: docs}}
		processor := &countingProcessor{errs: map[int]error{1: errBackendDown, 2: errBackendDown}}
		app := newApp(client, processor)

		for round := 1; round <= ocrMaxRetries+3; round++ {
			_, err := app.processAutoOcrTagDocuments(context.Background())
			require.Error(t, err, "round %d", round)
			assert.True(t, allBackendUnavailable(err), "round %d: %v", round, err)
		}
		assert.Empty(t, client.calls, "no fail tag and no tag removal")
		assert.Empty(t, app.ocrFailures.failures, "nothing is counted")
		assert.Equal(t, []int{1, 1, 1, 1, 1, 1}, processor.asked, "the second document is not attempted after the first finds the backend down")
	})

	t.Run("a backend outage does not erase an existing streak, nor add to it", func(t *testing.T) {
		client := &recordingClient{taggedDocuments: map[string][]Document{autoOcrTag: docs[:1]}}
		processor := &countingProcessor{errs: map[int]error{1: errors.New("pdf is broken")}}
		app := newApp(client, processor)

		for round := 1; round <= 2; round++ {
			_, err := app.processAutoOcrTagDocuments(context.Background())
			require.Error(t, err)
		}
		processor.errs[1] = errBackendDown
		for round := 1; round <= 5; round++ {
			_, err := app.processAutoOcrTagDocuments(context.Background())
			require.Error(t, err)
		}
		assert.Empty(t, client.calls, "outage cycles do not push the document over the limit")

		processor.errs[1] = errors.New("pdf is broken")
		_, err := app.processAutoOcrTagDocuments(context.Background())
		assert.NoError(t, err)
		require.Len(t, client.calls, 1, "the third ordinary failure still fail-tags the document")
		assert.Equal(t, []string{failTag}, client.calls[0].SuggestedTags)
	})

	t.Run("ordinary errors still count and the batch continues", func(t *testing.T) {
		client := &recordingClient{taggedDocuments: map[string][]Document{autoOcrTag: docs}}
		processor := &countingProcessor{errs: map[int]error{1: errors.New("pdf is broken"), 2: errors.New("pdf is broken")}}
		app := newApp(client, processor)

		_, err := app.processAutoOcrTagDocuments(context.Background())
		require.Error(t, err)
		assert.False(t, allBackendUnavailable(err))
		assert.Equal(t, []int{1, 2}, processor.asked)
		assert.Equal(t, 1, app.ocrFailures.failures[1])
		assert.Equal(t, 1, app.ocrFailures.failures[2])
	})
}

func TestProcessAutoTagDocuments_BackendUnavailableIsNotCounted(t *testing.T) {
	prevFailTag, prevAutoTag, prevMaxRetries := failTag, autoTag, autoTagMaxRetries
	t.Cleanup(func() { failTag, autoTag, autoTagMaxRetries = prevFailTag, prevAutoTag, prevMaxRetries })
	failTag = "paperless-gpt-failed"
	autoTag = "paperless-gpt-auto"
	autoTagMaxRetries = 3

	newClient := func(tagsErr error) *tagsCountingClient {
		return &tagsCountingClient{recordingClient: &recordingClient{
			taggedDocuments: map[string][]Document{autoTag: {
				{ID: 11, Title: "One", Tags: []string{autoTag}},
				{ID: 12, Title: "Two", Tags: []string{autoTag}},
			}},
			getAllTagsErr: tagsErr,
		}}
	}

	t.Run("never fail-tagged however many cycles fail, and the batch stops", func(t *testing.T) {
		client := newClient(errBackendDown)
		app := &App{Client: client}

		for round := 1; round <= autoTagMaxRetries+3; round++ {
			_, err := app.processAutoTagDocuments(context.Background())
			require.Error(t, err, "round %d", round)
			assert.True(t, allBackendUnavailable(err), "round %d: %v", round, err)
		}
		assert.Empty(t, client.calls, "no fail tag and no tag removal")
		assert.Empty(t, app.suggestionFailures.failures, "nothing is counted")
		assert.Equal(t, autoTagMaxRetries+3, client.suggestionAttempts, "only the first document is attempted each cycle")
	})

	t.Run("a dial error from the LLM is treated the same way", func(t *testing.T) {
		dialErr := (&http.Client{Timeout: 5 * time.Second})
		_, err := dialErr.Get("http://127.0.0.1:1/")
		require.Error(t, err)
		client := newClient(fmt.Errorf("Ollama chat request: %w", err))
		app := &App{Client: client}

		for round := 1; round <= autoTagMaxRetries+1; round++ {
			_, err := app.processAutoTagDocuments(context.Background())
			require.Error(t, err)
		}
		assert.Empty(t, client.calls)
		assert.Empty(t, app.suggestionFailures.failures)
	})

	t.Run("ordinary errors still count and the batch continues", func(t *testing.T) {
		client := newClient(errors.New("exceeds the available context size"))
		app := &App{Client: client}

		_, err := app.processAutoTagDocuments(context.Background())
		require.Error(t, err)
		assert.False(t, allBackendUnavailable(err))
		assert.Equal(t, 2, client.suggestionAttempts)
		assert.Equal(t, 1, app.suggestionFailures.failures[11])
		assert.Equal(t, 1, app.suggestionFailures.failures[12])
	})
}

func TestProcessTagDocuments_WorkflowOCRBackendUnavailableIsNotCounted(t *testing.T) {
	prevFailTag, prevAutoTag, prevOcrMax := failTag, autoTag, ocrMaxRetries
	t.Cleanup(func() { failTag, autoTag, ocrMaxRetries = prevFailTag, prevAutoTag, prevOcrMax })
	failTag = "paperless-gpt-failed"
	autoTag = "paperless-gpt-auto"
	ocrMaxRetries = 3
	isolateWorkflowSettings(t)

	const trigger = "paperless-gpt-scans"
	on := true
	useWorkflows(t, WorkflowConfig{ID: "scans", TriggerTag: trigger, EnableOCR: &on})

	docs := []Document{
		{ID: 31, Title: "One", Tags: []string{trigger}},
		{ID: 32, Title: "Two", Tags: []string{trigger}},
	}
	newApp := func(client *recordingClient, processor *countingProcessor) *App {
		return &App{
			Client:       client,
			docProcessor: processor,
			Database:     newOCRRunTestDB(t),
			ocrProvider:  &mockOCRProvider{text: "x"},
		}
	}

	t.Run("not counted, never fail-tagged, and the batch stops", func(t *testing.T) {
		client := &recordingClient{taggedDocuments: map[string][]Document{trigger: docs}}
		processor := &countingProcessor{errs: map[int]error{31: errBackendDown, 32: errBackendDown}}
		app := newApp(client, processor)

		for round := 1; round <= ocrMaxRetries+3; round++ {
			_, err := app.processAutoTagDocuments(context.Background())
			require.Error(t, err, "round %d", round)
			assert.True(t, allBackendUnavailable(err), "round %d: %v", round, err)
		}
		assert.Empty(t, client.calls)
		assert.Empty(t, app.ocrFailures.failures)
		assert.Equal(t, []int{31, 31, 31, 31, 31, 31}, processor.asked)
	})

	t.Run("ordinary OCR errors still count to the limit", func(t *testing.T) {
		client := &recordingClient{taggedDocuments: map[string][]Document{trigger: docs[:1]}}
		processor := &countingProcessor{errs: map[int]error{31: errors.New("pdf is broken")}}
		app := newApp(client, processor)

		for round := 1; round <= 2; round++ {
			_, err := app.processAutoTagDocuments(context.Background())
			require.Error(t, err)
		}
		assert.Empty(t, client.calls)
		_, err := app.processAutoTagDocuments(context.Background())
		assert.NoError(t, err)
		require.Len(t, client.calls, 1)
		assert.Equal(t, []string{failTag}, client.calls[0].SuggestedTags)
	})
}
