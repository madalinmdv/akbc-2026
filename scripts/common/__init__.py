"""Shared building blocks for the prediction generators.

The generator scripts differ only in how they prompt the model; everything
else -- locating the data, talking to Ollama, parsing the reply, writing the
prediction file -- lives here so the strategies stay comparable.
"""
