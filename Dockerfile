# Builder stage
FROM ollama/ollama:0.6.6 AS builder

# Run the serve command in the background and then run the model
RUN ollama serve & \
    sleep 5 && \
    ollama run qwen2:0.5b && \
    ollama run smollm2:135m-instruct-q4_K_M

# Final stage
FROM ollama/ollama:0.6.6

# Copy the necessary files from the builder stage
COPY --from=builder /root/.ollama /root/.ollama

EXPOSE 11434

# Command to run the serve command
CMD ["serve"]
