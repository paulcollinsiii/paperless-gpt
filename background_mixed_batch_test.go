package main

import (
	"context"
	"errors"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// Both tests use workflow OCR so that two documents in one poll can fail for
// different reasons. The poll stops at the first unavailable error, so the
// ordinary failure comes first.

func setupMixedBatch(t *testing.T) {
	t.Helper()
	prevFailTag, prevAutoTag, prevOcrMax := failTag, autoTag, ocrMaxRetries
	t.Cleanup(func() { failTag, autoTag, ocrMaxRetries = prevFailTag, prevAutoTag, prevOcrMax })
	failTag = "paperless-gpt-failed"
	autoTag = "paperless-gpt-auto"
	ocrMaxRetries = 3
	isolateWorkflowSettings(t)
}

// One tag whose batch holds an ordinary failure followed by an unavailable one.
func TestProcessAutoTagDocuments_MixedBatchInOneTagIsOrdinary(t *testing.T) {
	setupMixedBatch(t)
	const trigger = "paperless-gpt-scans"
	on := true
	useWorkflows(t, WorkflowConfig{ID: "scans", TriggerTag: trigger, EnableOCR: &on})

	client := &recordingClient{taggedDocuments: map[string][]Document{trigger: {
		{ID: 41, Title: "Broken", Tags: []string{trigger}},
		{ID: 42, Title: "Backend down", Tags: []string{trigger}},
	}}}
	processor := &countingProcessor{errs: map[int]error{
		41: errors.New("pdf is broken"),
		42: errBackendDown,
	}}
	app := &App{Client: client, docProcessor: processor, Database: newOCRRunTestDB(t), ocrProvider: &mockOCRProvider{text: "x"}}

	_, err := app.processAutoTagDocuments(context.Background())
	require.Error(t, err)
	assert.Equal(t, []int{41, 42}, processor.asked, "both documents were attempted")
	assert.False(t, allBackendUnavailable(err), "a batch with an ordinary failure must keep the long backoff: %v", err)
}

// Two tags, one failing for an ordinary reason and one with the backend down.
func TestProcessAutoTagDocuments_MixedAcrossTagsIsOrdinary(t *testing.T) {
	setupMixedBatch(t)
	const first, second = "paperless-gpt-first", "paperless-gpt-second"
	on := true
	useWorkflows(t,
		WorkflowConfig{ID: "first", TriggerTag: first, EnableOCR: &on},
		WorkflowConfig{ID: "second", TriggerTag: second, EnableOCR: &on},
	)

	client := &recordingClient{taggedDocuments: map[string][]Document{
		first:  {{ID: 51, Title: "Broken", Tags: []string{first}}},
		second: {{ID: 52, Title: "Backend down", Tags: []string{second}}},
	}}
	processor := &countingProcessor{errs: map[int]error{
		51: errors.New("pdf is broken"),
		52: errBackendDown,
	}}
	app := &App{Client: client, docProcessor: processor, Database: newOCRRunTestDB(t), ocrProvider: &mockOCRProvider{text: "x"}}

	_, err := app.processAutoTagDocuments(context.Background())
	require.Error(t, err)
	assert.Equal(t, []int{51, 52}, processor.asked, "both tags were polled")
	assert.False(t, allBackendUnavailable(err), "an ordinary failure in any tag keeps the long backoff: %v", err)
}
