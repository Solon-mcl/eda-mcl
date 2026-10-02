FROM eda-coverage-base:1.0 AS compiled

COPY app/inference /app/inference
RUN python3 -m compileall -q -b /app/inference \
    && find /app/inference -type f -name '*.py' -delete

# Keep the submitted layers free of our Python source files.
FROM eda-coverage-base:1.0
COPY --from=compiled /app/inference /app/inference
COPY entrypoint.sh /entrypoint.sh
RUN chmod 755 /entrypoint.sh

ENV PYTHONPATH=/app
ENTRYPOINT ["/entrypoint.sh"]
CMD ["python3", "-c", "from inference import InferenceInterface; print('inference image ready')"]
