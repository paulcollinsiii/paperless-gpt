package ocr

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"strings"
	"syscall"
)

// ErrBackendUnavailable marks an error that means a backend (OCR service, LLM
// server) could not be reached or reported itself unavailable, as opposed to
// failing on the document it was given. Callers use IsBackendUnavailable
// rather than matching it directly.
var ErrBackendUnavailable = errors.New("backend unavailable")

// IsBackendUnavailable reports whether err means the backend is down or still
// starting, so the work should be retried later without counting against the
// document. It is true when the chain contains ErrBackendUnavailable or a
// failure to establish a connection: a refused/unreachable dial, a dial
// timeout, or a DNS resolution failure.
//
// It is deliberately false for everything else: context cancellation or
// deadline expiry, a timeout on a connection that was already established
// (e.g. waiting for response headers), HTTP 500/504/4xx, and errors with no
// typed cause. A slow or poisonous document must still reach the retry limit.
func IsBackendUnavailable(err error) bool {
	if err == nil {
		return false
	}
	if errors.Is(err, ErrBackendUnavailable) {
		return true
	}
	if errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
		return false
	}
	var dnsErr *net.DNSError
	if errors.As(err, &dnsErr) {
		return true
	}
	for _, errno := range []syscall.Errno{syscall.ECONNREFUSED, syscall.EHOSTUNREACH, syscall.ENETUNREACH, syscall.EHOSTDOWN} {
		if errors.Is(err, errno) {
			return true
		}
	}
	var opErr *net.OpError
	if errors.As(err, &opErr) && opErr.Op == "dial" {
		return true
	}
	return false
}

// unavailableTransport turns HTTP 502 and 503 responses into errors wrapping
// ErrBackendUnavailable. A service mesh sidecar or ingress answers with these
// while the upstream has no ready pods, which looks like a document failure to
// the caller unless it is classified here.
type unavailableTransport struct {
	base http.RoundTripper
}

// NewUnavailableTransport wraps base (http.DefaultTransport when nil) so that
// 502 and 503 responses are returned as errors wrapping ErrBackendUnavailable.
// The error reaches the caller wrapped in *url.Error, so errors.Is still works.
func NewUnavailableTransport(base http.RoundTripper) http.RoundTripper {
	if base == nil {
		base = http.DefaultTransport
	}
	return &unavailableTransport{base: base}
}

func (t *unavailableTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	resp, err := t.base.RoundTrip(req)
	if err != nil {
		return resp, err
	}
	if resp.StatusCode != http.StatusBadGateway && resp.StatusCode != http.StatusServiceUnavailable {
		return resp, nil
	}
	snippet, _ := io.ReadAll(io.LimitReader(resp.Body, 200))
	resp.Body.Close()
	return nil, fmt.Errorf("%w: HTTP %d: %s", ErrBackendUnavailable, resp.StatusCode, strings.TrimSpace(string(snippet)))
}

// WithUnavailableTransport makes client report 502/503 responses as
// ErrBackendUnavailable. A nil client is returned unchanged.
func WithUnavailableTransport(client *http.Client) *http.Client {
	if client == nil {
		return nil
	}
	client.Transport = NewUnavailableTransport(client.Transport)
	return client
}
