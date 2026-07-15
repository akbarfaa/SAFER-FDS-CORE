"""
SAFER FDS — Rate Limiter Middleware

Simple in-memory rate limiter for sandbox/evaluation mode.
Limits API calls per IP to prevent abuse while allowing demo usage.
"""

import time
from collections import defaultdict
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware


# ─── Configuration ──────────────────────────────────────────────────────────

MAX_REQUESTS_PER_HOUR = 200   # Per IP address
CLEANUP_INTERVAL = 3600       # Cleanup old entries every hour
MAX_DB_TRANSACTIONS = 2000    # Max transactions in database before pruning

# Whitelist paths that bypass rate limiting
BYPASS_PATHS = {"/health", "/docs", "/openapi.json", "/redoc"}


class RateLimiterMiddleware(BaseHTTPMiddleware):
    """In-memory sliding window rate limiter."""

    def __init__(self, app):
        super().__init__(app)
        # {ip: [timestamp1, timestamp2, ...]}
        self.request_log: dict[str, list[float]] = defaultdict(list)
        self.last_cleanup = time.time()

    async def dispatch(self, request: Request, call_next):
        # Bypass rate limiting for health checks and docs
        path = request.url.path
        if path in BYPASS_PATHS:
            return await call_next(request)

        # Get client IP
        client_ip = request.client.host if request.client else "unknown"
        forwarded = request.headers.get("x-forwarded-for", "")
        if forwarded:
            client_ip = forwarded.split(",")[0].strip()

        now = time.time()

        # Periodic cleanup of old entries
        if now - self.last_cleanup > CLEANUP_INTERVAL:
            self._cleanup(now)

        # Remove entries older than 1 hour
        window_start = now - 3600
        self.request_log[client_ip] = [
            t for t in self.request_log[client_ip] if t > window_start
        ]

        # Check rate limit
        current_count = len(self.request_log[client_ip])
        remaining = max(0, MAX_REQUESTS_PER_HOUR - current_count)

        if current_count >= MAX_REQUESTS_PER_HOUR:
            return Response(
                content=(
                    '{"detail": "Rate limit exceeded. This is a SAFER Sandbox — '
                    'limited to ' + str(MAX_REQUESTS_PER_HOUR) + ' requests/hour. '
                    'Contact us for full access.", '
                    '"sandbox_mode": true}'
                ),
                status_code=429,
                media_type="application/json",
                headers={
                    "X-RateLimit-Limit": str(MAX_REQUESTS_PER_HOUR),
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reset": str(int(window_start + 3600)),
                    "Retry-After": "60",
                },
            )

        # Log this request
        self.request_log[client_ip].append(now)

        # Process request
        response = await call_next(request)

        # Add rate limit headers to every response
        response.headers["X-RateLimit-Limit"] = str(MAX_REQUESTS_PER_HOUR)
        response.headers["X-RateLimit-Remaining"] = str(remaining - 1)
        response.headers["X-Sandbox-Mode"] = "true"

        return response

    def _cleanup(self, now: float):
        """Remove IPs with no recent activity."""
        cutoff = now - 3600
        stale_ips = [
            ip for ip, timestamps in self.request_log.items()
            if not timestamps or max(timestamps) < cutoff
        ]
        for ip in stale_ips:
            del self.request_log[ip]
        self.last_cleanup = now
