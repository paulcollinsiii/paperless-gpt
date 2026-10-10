package ocr

import (
	"context"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// fault decides whether a request to the front server is answered with a fixed
// status instead of being passed to the fake MinerU service.
type fault func(r *http.Request) (status int, ok bool)

// mineruWithFault serves the fake MinerU behind a handler that injects faults
// and counts the requests that reach it, by "METHOD path".
func mineruWithFault(t *testing.T, f *fakeMinerU, inject fault) (*MinerUProvider, func(string) int) {
	t.Helper()
	counts := map[string]*atomic.Int32{}
	for _, key := range []string{"POST /v1/uploads", "POST /v1/parse/jobs", "GET /v1/files/file-out/content", "GET /v1/parse/jobs/job_1"} {
		counts[key] = &atomic.Int32{}
	}
	front := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if c, ok := counts[r.Method+" "+r.URL.Path]; ok {
			c.Add(1)
		}
		if status, ok := inject(r); ok {
			w.WriteHeader(status)
			_, _ = w.Write([]byte(`{"error":"no healthy upstream"}`))
			return
		}
		f.handle(w, r)
	}))
	t.Cleanup(front.Close)

	p, err := newMinerUProvider(Config{MinerUURL: front.URL})
	require.NoError(t, err)
	p.pollInterval = 5 * time.Millisecond
	p.retryWait = time.Millisecond
	return p, func(key string) int { return int(counts[key].Load()) }
}

func onPath(path string, status int) fault {
	return func(r *http.Request) (int, bool) {
		return status, strings.HasSuffix(r.URL.Path, path)
	}
}

func TestMinerU_ServiceDownIsUnavailable(t *testing.T) {
	for _, status := range []int{http.StatusBadGateway, http.StatusServiceUnavailable} {
		f := newFakeMinerU(t)
		p, count := mineruWithFault(t, f, func(*http.Request) (int, bool) { return status, true })

		_, err := p.ProcessImage(context.Background(), testPDF, 0)
		require.Error(t, err)
		assert.True(t, IsBackendUnavailable(err), "HTTP %d: %v", status, err)
		assert.Contains(t, err.Error(), "HTTP")
		assert.Equal(t, 1, count("POST /v1/uploads"), "upload creation is not retried")
		assert.Equal(t, 0, count("POST /v1/parse/jobs"))
	}
}

func TestMinerU_JobCreationSentOnceWhenUnavailable(t *testing.T) {
	f := newFakeMinerU(t)
	p, count := mineruWithFault(t, f, onPath("/v1/parse/jobs", http.StatusServiceUnavailable))

	_, err := p.ProcessImage(context.Background(), testPDF, 0)
	require.Error(t, err)
	assert.True(t, IsBackendUnavailable(err), "%v", err)
	assert.Equal(t, 1, count("POST /v1/parse/jobs"), "job creation must not be retried, to avoid duplicate jobs")
	assert.Equal(t, 0, f.jobsCreated)
}

func TestMinerU_RetryableStageRecoversFromUnavailable(t *testing.T) {
	f := newFakeMinerU(t)
	var failures atomic.Int32
	p, count := mineruWithFault(t, f, func(r *http.Request) (int, bool) {
		if strings.HasSuffix(r.URL.Path, "/v1/files/file-out/content") && failures.Add(1) <= 2 {
			return http.StatusServiceUnavailable, true
		}
		return 0, false
	})

	res, err := p.ProcessImage(context.Background(), testPDF, 0)
	require.NoError(t, err)
	assert.Contains(t, res.Text, "Hello world.")
	assert.Equal(t, 3, count("GET /v1/files/file-out/content"), "download retries 503 within the provider")
	assert.Equal(t, 1, f.jobsCreated, "the job is created exactly once")
}

func TestMinerU_RetryableStageStillDownIsUnavailable(t *testing.T) {
	f := newFakeMinerU(t)
	p, count := mineruWithFault(t, f, onPath("/v1/files/file-out/content", http.StatusServiceUnavailable))

	_, err := p.ProcessImage(context.Background(), testPDF, 0)
	require.Error(t, err)
	assert.True(t, IsBackendUnavailable(err), "%v", err)
	assert.Equal(t, 3, count("GET /v1/files/file-out/content"), "all retry attempts are still made")
	assert.Equal(t, 1, f.jobsCreated)
}

func TestMinerU_OtherServerErrorsAreNotUnavailable(t *testing.T) {
	for _, status := range []int{http.StatusInternalServerError, http.StatusGatewayTimeout, http.StatusBadRequest, http.StatusUnauthorized} {
		f := newFakeMinerU(t)
		p, _ := mineruWithFault(t, f, onPath("/v1/uploads", status))

		_, err := p.ProcessImage(context.Background(), testPDF, 0)
		require.Error(t, err)
		assert.False(t, IsBackendUnavailable(err), "HTTP %d: %v", status, err)
	}
}

func TestMinerU_ConnectionRefusedIsUnavailable(t *testing.T) {
	p, err := newMinerUProvider(Config{MinerUURL: closedPortURL(t)})
	require.NoError(t, err)
	p.retryWait = time.Millisecond

	_, err = p.ProcessImage(context.Background(), testPDF, 0)
	require.Error(t, err)
	assert.True(t, IsBackendUnavailable(err), "%v", err)
}
