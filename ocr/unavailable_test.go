package ocr

import (
	"context"
	"errors"
	"fmt"
	"net"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// closedPortURL returns the URL of a local port nothing listens on.
func closedPortURL(t *testing.T) string {
	t.Helper()
	l, err := net.Listen("tcp", "127.0.0.1:0")
	require.NoError(t, err)
	addr := l.Addr().String()
	require.NoError(t, l.Close())
	return "http://" + addr
}

func getVia(client *http.Client, url string) error {
	resp, err := client.Get(url)
	if err != nil {
		return err
	}
	return resp.Body.Close()
}

func TestIsBackendUnavailable_Dial(t *testing.T) {
	t.Run("connection refused on a closed port", func(t *testing.T) {
		err := getVia(&http.Client{Timeout: 5 * time.Second}, closedPortURL(t))
		require.Error(t, err)
		assert.True(t, IsBackendUnavailable(err))
	})

	t.Run("DNS resolution failure", func(t *testing.T) {
		err := getVia(&http.Client{Timeout: 10 * time.Second}, "http://backend.invalid/")
		require.Error(t, err)
		var dnsErr *net.DNSError
		require.ErrorAs(t, err, &dnsErr)
		assert.True(t, IsBackendUnavailable(err))
	})

	t.Run("dial timeout", func(t *testing.T) {
		err := &net.OpError{Op: "dial", Net: "tcp", Err: errors.New("i/o timeout")}
		assert.True(t, IsBackendUnavailable(fmt.Errorf("request failed: %w", err)))
	})
}

func TestIsBackendUnavailable_Chains(t *testing.T) {
	refused := getVia(&http.Client{Timeout: 5 * time.Second}, closedPortURL(t))
	require.Error(t, refused)

	assert.True(t, IsBackendUnavailable(fmt.Errorf("outer: %w", fmt.Errorf("inner: %w", refused))))
	assert.True(t, IsBackendUnavailable(errors.Join(errors.New("other"), refused)))
	assert.True(t, IsBackendUnavailable(fmt.Errorf("wrapped: %w", ErrBackendUnavailable)))
	assert.True(t, IsBackendUnavailable(errors.Join(errors.New("other"), fmt.Errorf("x: %w", ErrBackendUnavailable))))
}

func TestIsBackendUnavailable_NotUnavailable(t *testing.T) {
	assert.False(t, IsBackendUnavailable(nil))
	assert.False(t, IsBackendUnavailable(errors.New("backend unavailable")), "a plain string without a typed cause")
	assert.False(t, IsBackendUnavailable(context.Canceled))
	assert.False(t, IsBackendUnavailable(fmt.Errorf("wrapped: %w", context.DeadlineExceeded)))
	assert.False(t, IsBackendUnavailable(errors.Join(errors.New("a"), errors.New("b"))))

	t.Run("a context error from a dial does not count", func(t *testing.T) {
		err := &net.OpError{Op: "dial", Net: "tcp", Err: context.DeadlineExceeded}
		assert.False(t, IsBackendUnavailable(err))
	})

	t.Run("timeout waiting for headers on an accepted connection", func(t *testing.T) {
		release := make(chan struct{})
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			select {
			case <-release:
			case <-r.Context().Done():
			}
		}))
		defer server.Close()
		defer close(release)

		client := &http.Client{Transport: &http.Transport{ResponseHeaderTimeout: 100 * time.Millisecond}}
		err := getVia(client, server.URL)
		require.Error(t, err)
		assert.False(t, IsBackendUnavailable(err))
	})

	t.Run("client timeout after the connection was established", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			<-r.Context().Done()
		}))
		defer server.Close()

		err := getVia(&http.Client{Timeout: 100 * time.Millisecond}, server.URL)
		require.Error(t, err)
		assert.False(t, IsBackendUnavailable(err))
	})
}

func TestUnavailableTransport(t *testing.T) {
	statusServer := func(status int, body string) *httptest.Server {
		return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.WriteHeader(status)
			_, _ = w.Write([]byte(body))
		}))
	}

	for _, status := range []int{http.StatusBadGateway, http.StatusServiceUnavailable} {
		t.Run(fmt.Sprintf("HTTP %d is unavailable", status), func(t *testing.T) {
			server := statusServer(status, "no healthy upstream")
			defer server.Close()

			err := getVia(WithUnavailableTransport(&http.Client{}), server.URL)
			require.Error(t, err)
			assert.True(t, IsBackendUnavailable(err))
			assert.ErrorIs(t, err, ErrBackendUnavailable)
			assert.Contains(t, err.Error(), fmt.Sprintf("HTTP %d", status))
			assert.Contains(t, err.Error(), "no healthy upstream")
		})
	}

	for _, status := range []int{http.StatusOK, http.StatusBadRequest, http.StatusNotFound, http.StatusInternalServerError, http.StatusGatewayTimeout} {
		t.Run(fmt.Sprintf("HTTP %d passes through", status), func(t *testing.T) {
			server := statusServer(status, "payload")
			defer server.Close()

			resp, err := WithUnavailableTransport(&http.Client{}).Get(server.URL)
			require.NoError(t, err)
			defer resp.Body.Close()
			assert.Equal(t, status, resp.StatusCode)
		})
	}

	t.Run("body snippet is bounded", func(t *testing.T) {
		server := statusServer(http.StatusServiceUnavailable, strings.Repeat("x", 5000))
		defer server.Close()

		err := getVia(WithUnavailableTransport(&http.Client{}), server.URL)
		require.Error(t, err)
		assert.Less(t, len(err.Error()), 500)
	})

	t.Run("transport errors are returned unchanged and classified", func(t *testing.T) {
		err := getVia(WithUnavailableTransport(&http.Client{Timeout: 5 * time.Second}), closedPortURL(t))
		require.Error(t, err)
		assert.True(t, IsBackendUnavailable(err))
		assert.NotErrorIs(t, err, ErrBackendUnavailable)
	})

	t.Run("wraps the inner transport", func(t *testing.T) {
		var used bool
		base := roundTripFunc(func(req *http.Request) (*http.Response, error) {
			used = true
			return nil, errors.New("boom")
		})
		client := &http.Client{Transport: NewUnavailableTransport(base)}
		assert.Error(t, getVia(client, "http://example.invalid/"))
		assert.True(t, used)
	})

	t.Run("nil client stays nil", func(t *testing.T) {
		assert.Nil(t, WithUnavailableTransport(nil))
	})
}
