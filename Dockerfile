FROM eda-coverage-base:1.0

COPY app/inference /app/inference
COPY entrypoint.sh /entrypoint.sh

# Install a sourceless package in the submitted image, as requested by the
# competition rules.  NumPy is already present in the official base image.
RUN python3 -m compileall -q -b /app/inference \
    && rm -f /app/inference/__init__.py \
    && chmod 755 /entrypoint.sh

ENV PYTHONPATH=/app
ENTRYPOINT ["/entrypoint.sh"]
CMD ["python3", "-c", "from inference import InferenceInterface; print('inference image ready')"]

