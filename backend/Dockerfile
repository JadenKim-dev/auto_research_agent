FROM python:3.12-alpine as builder

WORKDIR /app

RUN apk add --no-cache \
    gcc \
    g++ \
    musl-dev \
    python3-dev \
    rust \
    cargo \
    libffi-dev \
    openssl-dev \
    libc-dev \
    linux-headers

COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir --user -r requirements.txt

FROM python:3.12-alpine

WORKDIR /app

RUN apk add --no-cache \
    libstdc++ \
    libgcc

RUN addgroup -g 1001 -S appuser && \
    adduser -S -D -u 1001 -s /sbin/nologin -G appuser appuser

COPY --from=builder /root/.local /home/appuser/.local

COPY . .

RUN chown -R appuser:appuser /app /home/appuser/.local

USER appuser

ENV PATH=/home/appuser/.local/bin:$PATH

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
